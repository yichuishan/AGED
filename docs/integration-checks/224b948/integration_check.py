"""Small GPU integration check of the committed AGED pipeline on the full Human3.6M export.

Runs the repository's own entry points (no wrappers, no code changes) with short
step counts. Writes a small, shareable record (manifest, logs, per-run JSON) to
<record dir> and the heavy outputs (checkpoints, TensorBoard events) to <work dir>.

usage (from the repo root, with the project venv):
    python docs/integration-checks/224b948/integration_check.py <record dir> <work dir> [steps]
"""
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

RECORD = Path(sys.argv[1]).resolve()
WORK = Path(sys.argv[2]).resolve()
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
DIAG_STEPS = 2000
PY = sys.executable
ROOT = "data/h36m"
VARIANTS = ["full", "geodesic", "euclidean", "plain", "fidelity", "continuity"]
HERE = Path(__file__).resolve().parent

for sub in ("logs", "results"):
    (RECORD / sub).mkdir(parents=True, exist_ok=False)
WORK.mkdir(parents=True, exist_ok=False)
runs = WORK / "runs"
steps_log = []


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def run(name, argv, kind="check"):
    cmd = [PY, *argv]
    started = time.time()
    with (RECORD / "logs" / f"{name}.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
    entry = {"name": name, "kind": kind, "command": subprocess.list2cmdline(["python", *argv]),
             "returncode": proc.returncode, "seconds": round(time.time() - started, 1)}
    steps_log.append(entry)
    status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"[{status:>11}] {name:<30} {entry['seconds']:7.1f}s", flush=True)
    return proc.returncode == 0


def rel(path):
    return Path(path).resolve().relative_to(Path.cwd()).as_posix()


# ---------------------------------------------------------------- provenance
import torch  # noqa: E402

sha = git("rev-parse", "HEAD")
integration = "origin/integration/full-aged"
manifest = {
    "purpose": "small GPU integration check on the full dataset; NOT a benchmark result",
    "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "git": {
        "commit": sha,
        "short": sha[:7],
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "subject": git("log", "-1", "--format=%s"),
        "commit_date": git("log", "-1", "--format=%cI"),
        "origin_main": git("rev-parse", "origin/main"),
        "integration_branch": {"ref": integration, "commit": git("rev-parse", integration)},
        "src_and_configs_identical_to_integration_branch":
            git("rev-parse", "HEAD:src") == git("rev-parse", f"{integration}:src")
            and git("rev-parse", "HEAD:configs") == git("rev-parse", f"{integration}:configs"),
        "files_differing_from_integration_branch": git("diff", "--name-only", integration, "HEAD").splitlines(),
        "tracked_changes": git("status", "--porcelain", "--untracked-files=no") or "none (clean)",
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if torch.cuda.is_available() else None,
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip(),
        "numpy": __import__("numpy").__version__,
    },
    "outputs": {"record": rel(RECORD), "checkpoints_and_tensorboard": rel(WORK)},
}

files = sorted(Path(ROOT).glob("S*/*.txt"))
digest = hashlib.sha256()
for path in files:
    digest.update(path.as_posix().encode())
    digest.update(path.read_bytes())
per_subject = {}
for path in files:
    per_subject[path.parent.name] = per_subject.get(path.parent.name, 0) + 1
manifest["dataset"] = {
    "root": ROOT,
    "files": len(files),
    "files_per_subject": per_subject,
    "bytes": sum(p.stat().st_size for p in files),
    "sha256_of_paths_and_contents": digest.hexdigest(),
}
manifest["configs"] = {name: Path(f"configs/{name}").read_text() for name in ("short-term.yaml", "long-term.yaml")}
manifest["check_overrides"] = {
    "variant runs": f"--epochs 1 --steps-per-epoch {STEPS}",
    "resume": "--resume full-short/last.pt --epochs 2",
    "long-term": f"--epochs 1 --steps-per-epoch {STEPS}",
    "dev (S11 validation)": f"--val-subject S11 --track tensorboard --epochs 2 --steps-per-epoch {STEPS // 2}",
    "device": "cuda",
}
manifest["implementation_choices_at_this_commit"] = {
    "optimizer": "torch.optim.Adam (default betas/eps, no weight decay) for predictor and for critics, "
                 "lr from config (src/aged/train.py:136,138); hard-coded, not set by the YAML configs",
    "lr_schedule": "none: constant learning rate (no scheduler in src/aged/train.py)",
    "update_schedule": "per batch: one joint critic step (both critics, one optimizer) then one predictor step "
                       "(src/aged/train.py:43-85)",
    "adversarial_objective": "minimax: predictor minimises -softplus(logit) = log(1-D(fake)); critics trained with "
                             "BCE on logits; lambda_adv 0.6",
    "gradient_clipping": "clip_grad_norm 5.0 separately for critics and predictor; error_if_nonfinite=True",
    "discriminator": "src/aged/critics.py: Linear(K->H) -> 1-layer GRU(H) -> Linear(H->1) on the final hidden "
                     "state, H = critic_hidden_dim 1024; fidelity critic sees the predicted 10/25 frames, "
                     "continuity critic sees observed+predicted",
    "sampling": "src/aged/data.py:96-130: random windows; recording chosen with probability proportional to its "
                "number of valid windows, start uniform within it; RNG keyed by (seed, epoch, index); "
                "steps_per_epoch*batch_size windows per epoch; no shuffling beyond that",
    "geodesic_loss": "src/aged/losses.py:61-69: sum over future frames and joints of d_G (not squared), "
                     "mean over batch",
    "features": "K = varying body-rotation channels from index 6 onward; root rotation and translation excluded",
    "evaluation": "src/aged/evaluate.py: S5, 8 fixed windows per action (RandomState 1234567890), "
                  "Euler-angle L2 on raw targets",
}
(RECORD / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"commit {sha}  ({manifest['git']['branch']}, {manifest['git']['tracked_changes']})")
print(f"dataset {len(files)} files, sha256 {digest.hexdigest()[:16]}...", flush=True)

# ---------------------------------------------------------------- checks
common = ["--root", ROOT, "--device", "cuda"]
run("pytest", ["-m", "pytest", "-q"])
run("baseline-short", ["-m", "aged.evaluate", "--root", ROOT, "--baseline",
                       "--output", str(RECORD / "results" / "zero-velocity.json")])
run("baseline-long", ["-m", "aged.evaluate", "--root", ROOT, "--baseline", "--horizon", "25",
                      "--output", str(RECORD / "results" / "zero-velocity-long.json")])
for variant in VARIANTS:
    out = runs / f"{variant}-short"
    if run(f"train-{variant}-short", ["-m", "aged.train", *common, "--config", "configs/short-term.yaml",
                                      "--variant", variant, "--epochs", "1", "--steps-per-epoch", str(STEPS),
                                      "--out", str(out)]):
        run(f"eval-{variant}-short", ["-m", "aged.evaluate", "--root", ROOT, "--device", "cuda",
                                      "--checkpoint", str(out / "last.pt"),
                                      "--output", str(out / "test-results.json")])
if (runs / "full-short" / "last.pt").exists():
    shutil.copy(runs / "full-short" / "results.json", runs / "full-short" / "results-epoch1.json")
    if run("resume-full-short-to-epoch2", ["-m", "aged.train", "--root", ROOT, "--device", "cuda",
                                           "--out", str(runs / "full-short"),
                                           "--resume", str(runs / "full-short" / "last.pt"), "--epochs", "2"]):
        run("eval-full-short-resumed", ["-m", "aged.evaluate", "--root", ROOT, "--device", "cuda",
                                        "--checkpoint", str(runs / "full-short" / "last.pt"),
                                        "--output", str(runs / "full-short" / "test-results-epoch2.json")])
long_out = runs / "full-long"
if run("train-full-long", ["-m", "aged.train", *common, "--config", "configs/long-term.yaml",
                           "--epochs", "1", "--steps-per-epoch", str(STEPS), "--out", str(long_out)]):
    run("eval-full-long-h25", ["-m", "aged.evaluate", "--root", ROOT, "--device", "cuda", "--horizon", "25",
                               "--checkpoint", str(long_out / "last.pt"),
                               "--output", str(long_out / "test-results.json")])
    run("eval-full-long-h100", ["-m", "aged.evaluate", "--root", ROOT, "--device", "cuda", "--horizon", "100",
                                "--checkpoint", str(long_out / "last.pt"),
                                "--output", str(long_out / "rollout-100.json")])
dev_out = runs / "dev-full"
if run("train-dev-full-S11", ["-m", "aged.train", *common, "--config", "configs/short-term.yaml",
                              "--val-subject", "S11", "--track", "tensorboard",
                              "--epochs", "2", "--steps-per-epoch", str(STEPS // 2), "--out", str(dev_out)]):
    run("eval-dev-full-best", ["-m", "aged.evaluate", "--root", ROOT, "--device", "cuda",
                               "--checkpoint", str(dev_out / "best.pt"),
                               "--output", str(dev_out / "test-results.json")])

# ---------------------------------------------------------------- optimizer diagnostic (not a pipeline check)
for variant in ("plain", "full"):
    for optimizer in ("adam", "sgd"):
        run(f"diag-{variant}-{optimizer}", [str(HERE / "optim_diag.py"), ROOT, variant, optimizer, "0.005",
                                            str(DIAG_STEPS), "250"], kind="diagnostic")


# ---------------------------------------------------------------- summary
def finite(obj):
    if isinstance(obj, dict):
        return all(finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(finite(v) for v in obj)
    return not isinstance(obj, float) or math.isfinite(obj)


summary = {}
for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
    if not (run_dir / "results.json").exists():
        summary[run_dir.name] = {"error": "no results.json"}
        continue
    res = json.loads((run_dir / "results.json").read_text())
    last = res["history"][-1]
    row = {
        "epochs_completed": len(res["history"]),
        "recorded_commit": res["environment"].get("commit"),
        "selected_device": res["environment"].get("selected_device"),
        "ms_per_step_last_epoch": round(1000 * last["seconds"] / res["config"]["steps_per_epoch"], 1),
        "last_epoch": {k: v for k, v in last.items() if k != "seconds"},
        "checkpoints": {p.name: p.stat().st_size for p in sorted(run_dir.glob("*.pt"))},
        "tensorboard_event_files": len(list(run_dir.glob("tb/events.*"))),
        "history_finite": finite(res["history"]),
    }
    for name in ("test-results.json", "test-results-epoch2.json", "rollout-100.json"):
        path = run_dir / name
        if path.exists():
            ev = json.loads(path.read_text())
            row[name] = {"epoch": ev["epoch"], "prediction_length": ev["prediction_length"],
                         "average": ev["average"], "finite": finite(ev["per_action"])}
    summary[run_dir.name] = row
    # small, shareable copies next to the record (checkpoints stay in WORK)
    dest = RECORD / "results" / run_dir.name
    dest.mkdir()
    for path in run_dir.glob("*.json"):
        shutil.copy(path, dest / path.name)
    for path in run_dir.glob("*.txt"):
        shutil.copy(path, dest / path.name)

manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
manifest["steps"] = steps_log
manifest["all_checks_succeeded"] = all(s["returncode"] == 0 for s in steps_log if s["kind"] == "check")
manifest["all_diagnostics_succeeded"] = all(s["returncode"] == 0 for s in steps_log if s["kind"] == "diagnostic")
manifest["runs"] = summary
manifest["checkpoint_bytes_total"] = sum(p.stat().st_size for p in WORK.rglob("*.pt"))
(RECORD / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nall checks succeeded: {manifest['all_checks_succeeded']}; "
      f"diagnostics: {manifest['all_diagnostics_succeeded']}")
print(f"manifest: {RECORD / 'manifest.json'}")
