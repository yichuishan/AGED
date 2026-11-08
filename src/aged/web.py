"""Loopback-only AGED training dashboard. No additional runtime dependency."""

import argparse
from collections import deque
from dataclasses import fields
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import os
import pickle
from pathlib import Path
import secrets
import signal
import string
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlsplit
import uuid
import webbrowser

import torch

from .config import Config, RESUME_MUTABLE, VARIANTS
from .data import ACTIONS, TRAIN_SUBJECTS
from .runtime import format_command, load_checkpoint, training_arguments
from .web_worker import EVENT_PREFIX

ACTIVE = {"running", "stopping"}

# Labels and groups are UI metadata; Config remains the validation authority.
GROUPS = {
    "Training & model": {
        "variant": "Model variant", "epochs": "Total epochs", "steps_per_epoch": "Updates per epoch",
        "batch_size": "Batch size", "observed_length": "Observed frames", "prediction_length": "Predicted frames",
        "hidden_dim": "Predictor hidden size", "critic_hidden_dim": "Critic hidden size",
        "condition_decoder": "Condition decoder on action", "share_recurrent_weights": "Share recurrent weights",
    },
    "Optimizers & loss": {
        "optimizer": "G optimizer", "learning_rate": "G learning rate", "critic_optimizer": "D optimizer",
        "critic_learning_rate": "D learning rate", "lambda_adv": "Adversarial weight (lambda)",
        "gradient_clip": "Gradient clip norm", "learning_rate_decay": "Learning-rate decay factor",
        "learning_rate_decay_every": "Decay interval (updates)", "adversarial_objective": "Generator objective",
        "geodesic_reduction": "Geodesic reduction", "critic_architecture": "Critic architecture",
        "critic_output_layout": "Critic output layout", "separate_critic_optimizers": "Separate critic optimizers",
    },
    "Data & validation": {
        "actions": "Action classes", "val_subject": "Validation subject", "validation_windows": "Validation windows",
        "sampling_strategy": "Sampling strategy", "minimum_start": "Earliest window start", "seed": "Random seed",
        "select_on": "Checkpoint selection metric", "patience": "Early-stop patience (0: off)",
        "save_every": "Snapshot interval (epochs, 0: off)",
    },
    "Device & logging": {
        "device": "Device", "threads": "CPU threads", "num_workers": "DataLoader workers",
        "track": "Additional tracking backend", "project": "wandb project",
    },
}
CHOICES = {
    "variant": list(VARIANTS), "optimizer": ["sgd", "adam"], "critic_optimizer": [None, "sgd", "adam"],
    "adversarial_objective": ["minimax", "non_saturating"], "geodesic_reduction": ["sum", "mean"],
    "critic_architecture": ["projected", "direct"], "critic_output_layout": ["flat", "column"],
    "val_subject": [None, *TRAIN_SUBJECTS], "sampling_strategy": ["windows", "recordings"],
    "select_on": ["angle_error", "val_loss"], "track": ["none", "tensorboard", "wandb"],
}
HELP = {
    "critic_optimizer": "Leave unset to use the G optimizer.", "critic_learning_rate": "Leave blank to use the G learning rate.",
    "val_subject": "Use S11 for development. Unset uses all six training subjects without best.pt. Reserve S5 for final testing.",
    "save_every": "last.pt is saved each epoch; validation improvements update best.pt. This option retains extra snapshots.",
    "learning_rate_decay_every": "Decay starts after this many updates.",
    "learning_rate": "Earlier Adam + 0.005 runs were unstable. The P0 preset checks SGD + 0.005 first.",
    "adversarial_objective": "minimax follows the paper; report non_saturating as an engineering variant.",
    "geodesic_reduction": "sum follows the time/joint sum in Eq. 4; mean is an ablation.",
    "num_workers": "Start with 0. Training runs in a separate Python process on Windows too.",
    "device": "Use cuda, cuda:0, cpu, auto or mps in the Python environment running this server.",
}


def schema():
    defaults = Config().to_dict()
    result = []
    for group, labels in GROUPS.items():
        for name, label in labels.items():
            value = defaults[name]
            kind = ("actions" if name == "actions" else "select" if name in CHOICES else
                    "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else
                    "number" if isinstance(value, float) or name == "critic_learning_rate" else "text")
            result.append({"name": name, "label": label, "group": group, "kind": kind,
                           "choices": CHOICES.get(name), "help": HELP.get(name, ""),
                           "nullable": name == "critic_learning_rate", "resume_mutable": name in RESUME_MUTABLE})
    assert {x["name"] for x in result} == {field.name for field in fields(Config)}
    return result


def profiles():
    short = Config().to_dict()
    team = dict(short, critic_architecture="direct", critic_output_layout="column",
                separate_critic_optimizers=True, optimizer="sgd", learning_rate_decay=0.95,
                sampling_strategy="recordings", minimum_start=16,
                adversarial_objective="non_saturating", val_subject="S11", patience=8)
    return {
        "p0": {"label": "P0 · Quick SGD check (recommended)", "config": dict(short, optimizer="sgd", epochs=2,
                steps_per_epoch=1000, val_subject="S11", save_every=1)},
        "short": {"label": "Compatibility defaults · Short-term / Adam", "config": short},
        "long": {"label": "Compatibility defaults · Long-term / Adam", "config": dict(short, prediction_length=25)},
        "team": {"label": "Team profile · Multiple changes", "config": team},
    }


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, base, state_dir):
        self.base = Path(base).resolve()
        self.state_dir = self.path(state_dir)
        self.jobs = {}
        self.lock = threading.RLock()

    def path(self, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Path must not be empty")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.base / path).resolve()

    def prepare(self, payload):
        """Read-only preflight; failed requests do not create run directories."""
        mode = payload.get("mode", "new")
        if mode not in ("new", "resume", "warm"):
            raise ValueError("Unknown training mode")
        config = Config(**payload["config"])
        root, out = self.path(payload["root"]), self.path(payload["out"])
        if not root.is_dir():
            raise ValueError(f"Dataset directory does not exist: {root}")
        if out == root or root.is_relative_to(out) or out.is_relative_to(root):
            raise ValueError("Keep output and dataset directories separate to avoid writing run files into the dataset")
        if out.exists() and not out.is_dir():
            raise ValueError(f"Output path is not a directory: {out}")
        if mode != "resume" and out.exists() and any(out.iterdir()):
            raise ValueError("Output directory is not empty. Choose a new directory or select resume mode")
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this Python environment. Check the GPU setup or select CPU")
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable in this environment")
        torch.device(config.device if config.device != "auto" else "cpu")
        saved = None
        checkpoint = None
        if mode != "new":
            checkpoint = self.path(payload.get("checkpoint", ""))
            saved = load_checkpoint(checkpoint)
            if config.epochs <= saved["epoch"]:
                raise ValueError(f"Total epochs must exceed checkpoint epoch {saved['epoch']}")
            if mode == "resume":
                for key, value in config.to_dict().items():
                    if key not in RESUME_MUTABLE and value != saved["config"][key]:
                        raise ValueError(f"Resume cannot change {key}. Load the checkpoint configuration first")
            elif config.optimizer != saved["config"]["optimizer"]:
                raise ValueError("Warm start must retain the source G optimizer type")
        required = [root / subject / f"{action}_{take}.txt"
                    for subject in TRAIN_SUBJECTS for action in config.actions for take in (1, 2)]
        missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"Missing {len(missing)}/{len(required)} training/validation files, for example: " + ", ".join(missing[:6]))
        return {"config": config.to_dict(), "root": str(root), "out": str(out),
                "resume": str(checkpoint) if mode == "resume" else None,
                "init_from": str(checkpoint) if mode == "warm" else None,
                "required_files": len(required), "mode": mode}

    def start(self, payload):
        prepared = self.prepare(payload)
        with self.lock:
            if any(job["status"] in ACTIVE for job in self.jobs.values()):
                raise RuntimeError("A training job is already active. Wait for it to finish or stop it first")
            ident = uuid.uuid4().hex[:12]
            folder = self.state_dir / ident
            folder.mkdir(parents=True)
            config_path = folder / "config.json"
            config_path.write_text(json.dumps(prepared["config"], indent=2), encoding="utf-8")
            arguments = training_arguments(Config(**prepared["config"]), prepared["root"], prepared["out"],
                                           config_path, prepared["resume"], prepared["init_from"])
            prepared.update(stop_file=str(folder / "stop"), command_args=arguments)
            request_path = folder / "request.json"
            request_path.write_text(json.dumps(prepared, indent=2), encoding="utf-8")
            command = [sys.executable, "-m", "aged.train", *arguments]
            job = {"id": ident, "status": "running", "created": timestamp(), "finished": None,
                   "out": prepared["out"], "root": prepared["root"], "config": prepared["config"],
                   "mode": prepared["mode"], "checkpoint": prepared["resume"] or prepared["init_from"] or "",
                   "command": format_command(command),
                   "log_path": str(folder / "console.log"), "folder": folder,
                   "logs": deque(maxlen=3000), "cursor": 0, "history": [], "series": deque(maxlen=600),
                   "progress": None, "returncode": None, "error": None, "forced": False}
            self.jobs[ident] = job
            environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            try:
                job["process"] = subprocess.Popen(
                    [sys.executable, "-u", "-m", "aged.web_worker", "--request", str(request_path)],
                    cwd=self.base, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
            except OSError as error:
                job.update(status="failed", error=str(error), finished=timestamp())
                self._persist(job)
                raise
            self._persist(job)
            job["reader"] = threading.Thread(target=self._read, args=(job,), daemon=True)
            job["reader"].start()
            return self.snapshot(ident)

    def _persist(self, job):
        values = {key: job[key] for key in ("id", "status", "created", "finished", "out", "root",
                  "config", "mode", "checkpoint", "command", "log_path", "returncode", "error")}
        temporary = job["folder"] / "session.json.tmp"
        temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(job["folder"] / "session.json")

    def _read(self, job):
        try:
            with Path(job["log_path"]).open("w", encoding="utf-8") as log:
                for line in job["process"].stdout:
                    log.write(line)
                    log.flush()
                    with self.lock:
                        event = None
                        if line.startswith(EVENT_PREFIX):
                            try:
                                event = json.loads(line[len(EVENT_PREFIX):])
                            except ValueError:
                                pass
                        if event and event.get("type") == "batch":
                            job["progress"] = event
                            job["series"].append({"step": event["step"], **event["metrics"]})
                        elif event and event.get("type") == "epoch":
                            job["history"].append(event["metrics"])
                        elif event and event.get("type") == "history":
                            job["history"] = event["history"]
                        else:
                            job["cursor"] += 1
                            job["logs"].append({"id": job["cursor"], "text": line.rstrip("\r\n")})
            code = job["process"].wait()
            with self.lock:
                job["returncode"] = code
                job["status"] = ("terminated" if job["forced"] else "failed" if code else
                                 "stopped" if (job["folder"] / "stop").exists() else "completed")
                if code and not job["forced"]:
                    job["error"] = "\n".join(row["text"] for row in list(job["logs"])[-20:])
        finally:
            with self.lock:
                if job["status"] in ACTIVE:
                    job.update(status="failed", error="Could not read training output. Check console.log")
                job["finished"] = timestamp()
                self._persist(job)
            job["process"].stdout.close()

    def snapshot(self, ident, after=0):
        with self.lock:
            job = self.jobs[ident]
            result = {key: job[key] for key in ("id", "status", "created", "finished", "out", "root",
                      "config", "mode", "checkpoint", "command", "log_path", "cursor", "history", "progress", "returncode", "error")}
            result["logs"] = [row for row in job["logs"] if row["id"] > after]
            result["series"] = list(job["series"])
            # Copy mutable collections before the HTTP thread serializes them.
            result["history"] = list(result["history"])
            result["checkpoints"] = [name for name in ("last.pt", "best.pt")
                                     if (Path(job["out"]) / name).is_file()]
            return result

    def stop(self, ident, force=False):
        with self.lock:
            job = self.jobs[ident]
            if job["status"] not in ACTIVE:
                raise ValueError("This job has already ended")
            if force:
                job["forced"] = True
                self._terminate(job["process"])
            else:
                (job["folder"] / "stop").touch()
            job["status"] = "stopping"
        return self.snapshot(ident)

    @staticmethod
    def _terminate(process, kill=False):
        if process.poll() is not None:
            return
        if os.name == "nt":
            # Include DataLoader children when workers > 0 on Windows.
            try:
                result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                if result.returncode and process.poll() is None:
                    process.kill()
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
            except ProcessLookupError:
                pass

    def close(self):
        with self.lock:
            processes = [job["process"] for job in self.jobs.values()
                         if job["status"] in ACTIVE and "process" in job]
            for job in self.jobs.values():
                if job["status"] in ACTIVE:
                    job["forced"] = True
        for process in processes:
            self._terminate(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._terminate(process, kill=True)
                process.wait(timeout=5)
        # Finish draining stdout and persist the terminal state before exiting.
        for job in list(self.jobs.values()):
            if "reader" in job:
                job["reader"].join(timeout=5)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager):
        self.manager = manager
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "AGEDLocal/1.0"

    def log_message(self, *args):
        pass

    def _send(self, status, data, content_type="application/json; charset=utf-8", attachment=False):
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        elif isinstance(data, str):
            data = data.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="console.log"')
        self.end_headers()
        self.wfile.write(data)

    def _allowed(self, write=False):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed:
            self._send(403, {"error": "Only local localhost requests are accepted"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in allowed}:
            self._send(403, {"error": "Cross-origin requests are not allowed"})
            return False
        if write and not secrets.compare_digest(self.headers.get("X-AGED-Token", ""), self.server.token):
            self._send(403, {"error": "Refresh the local page and try again"})
            return False
        return True

    def do_GET(self):
        if not self._allowed():
            return
        route = urlsplit(self.path)
        query = parse_qs(route.query)
        manager = self.server.manager
        try:
            if route.path == "/api/bootstrap":
                device = {"cuda": torch.cuda.is_available(), "torch": torch.__version__,
                          "python": sys.version.split()[0], "cwd": str(manager.base),
                          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
                self._send(200, {"token": self.server.token, "schema": schema(), "profiles": profiles(),
                                 "actions": ACTIONS, "environment": device})
            elif route.path == "/api/jobs":
                with manager.lock:
                    rows = [{k: job[k] for k in ("id", "status", "created", "out")}
                            for job in manager.jobs.values()]
                self._send(200, rows)
            elif route.path.startswith("/api/jobs/"):
                parts = route.path.strip("/").split("/")
                ident = parts[2]
                if len(parts) == 4 and parts[3] == "log":
                    path = Path(manager.snapshot(ident)["log_path"])
                    self._send(200, path.read_bytes() if path.exists() else b"", "text/plain; charset=utf-8", True)
                elif len(parts) == 3:
                    self._send(200, manager.snapshot(ident, int(query.get("after", ["0"])[0])))
                else:
                    self._send(404, {"error": "Unknown endpoint"})
            elif route.path == "/api/browse":
                directory = manager.path(query.get("path", [str(manager.base)])[0])
                if not directory.is_dir():
                    directory = directory.parent
                entries = []
                for path in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
                    if path.name.startswith("."):
                        continue
                    if path.is_dir() or path.suffix in (".pt", ".pth"):
                        entries.append({"name": path.name, "path": str(path), "directory": path.is_dir()})
                    if len(entries) >= 500:
                        break
                drives = [f"{letter}:/" for letter in string.ascii_uppercase if Path(f"{letter}:/").is_dir()] if os.name == "nt" else []
                self._send(200, {"path": str(directory), "parent": str(directory.parent), "entries": entries, "drives": drives})
            else:
                asset = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}.get(route.path)
                if asset:
                    content_type = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[asset]
                    self._send(200, files("aged").joinpath("web_assets").joinpath(asset).read_bytes(), content_type + "; charset=utf-8")
                else:
                    self._send(404, {"error": "Page not found"})
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            self._send(400, {"error": str(error)})

    def do_POST(self):
        if not self._allowed(write=True):
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024 * 1024:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object")
            manager = self.server.manager
            if self.path == "/api/validate":
                prepared = manager.prepare(payload)
                self._send(200, {"ok": True, "files": prepared["required_files"],
                                 "root": prepared["root"], "out": prepared["out"]})
            elif self.path == "/api/start":
                self._send(201, manager.start(payload))
            elif self.path == "/api/checkpoint":
                saved = load_checkpoint(manager.path(payload["path"]))
                self._send(200, {"config": saved["config"], "epoch": saved["epoch"], "step": saved["step"]})
            elif self.path.startswith("/api/jobs/") and self.path.endswith("/stop"):
                self._send(200, manager.stop(self.path.split("/")[3], payload.get("force") is True))
            else:
                self._send(404, {"error": "Unknown endpoint"})
        except RuntimeError as error:
            self._send(409, {"error": str(error)})
        except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError, IndexError) as error:
            self._send(400, {"error": str(error)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", default="experiments/web-ui")
    parser.add_argument("--open", action="store_true", help="open the local browser")
    args = parser.parse_args(argv)
    manager = JobManager(Path.cwd(), args.state_dir)
    server = DashboardServer(("127.0.0.1", args.port), manager)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"AGED Training Studio: {url}\nWorking directory: {manager.base}\nCtrl+C stops the server and its active training process.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
        server.server_close()


if __name__ == "__main__":
    main()
