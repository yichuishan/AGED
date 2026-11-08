"""Real HTTP -> isolated worker -> production training/checkpoint tests."""

from contextlib import contextmanager
from dataclasses import fields
import http.client
import json
from pathlib import Path
import threading
import time

import pytest
import torch

from aged.config import Config
from aged.runtime import load_checkpoint
from aged.web import DashboardServer, JobManager, schema
from aged.train import run_training


@contextmanager
def dashboard(tmp_path):
    manager = JobManager(tmp_path, "web-state")
    server = DashboardServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, **extra):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20)
        headers = {"X-AGED-Token": server.token, "Content-Type": "application/json", **extra}
        connection.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = connection.getresponse()
        content = response.read()
        result = json.loads(content) if response.getheader("Content-Type", "").startswith("application/json") else content
        status = response.status
        connection.close()
        return status, result

    try:
        yield request, manager
    finally:
        manager.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def wait_for_job(request, ident):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status, job = request("GET", f"/api/jobs/{ident}")
        assert status == 200
        if job["status"] not in ("running", "stopping"):
            return job
        time.sleep(0.05)
    pytest.fail("Web training did not finish within 30 seconds")


def settings(motion_export, tmp_path, **overrides):
    config = Config(actions=["walking", "eating"], hidden_dim=8, critic_hidden_dim=8,
                    epochs=3, steps_per_epoch=2, batch_size=2, validation_windows=4,
                    val_subject="S11", optimizer="sgd", save_every=1, device="cpu")
    return dict(root=str(motion_export), out=str(tmp_path / "run with spaces"),
                config=config.to_dict(), **overrides)


def test_web_training_stop_save_resume_and_download(motion_export, tmp_path):
    with dashboard(tmp_path) as (request, manager):
        assert request("GET", "/")[0] == 200
        assert request("GET", "/app.js")[0] == 200
        assert {row["name"] for row in schema()} == {field.name for field in fields(Config)}
        body = settings(motion_export, tmp_path)
        status, checked = request("POST", "/api/validate", body)
        assert status == 200 and checked["files"] == 24
        status, job = request("POST", "/api/start", body)
        assert status == 201
        assert request("POST", "/api/start", dict(body, out=str(tmp_path / "another")))[0] == 409
        assert request("POST", f"/api/jobs/{job['id']}/stop", {})[0] == 200
        finished = wait_for_job(request, job["id"])
        assert finished["status"] == "stopped", finished["error"]
        assert finished["returncode"] == 0 and len(finished["history"]) == 1
        assert finished["progress"]["step"] == 2
        assert finished["series"] and all(row["reconstruction"] > 0 for row in finished["series"])
        out = Path(body["out"])
        assert (out / "epoch_001.pt").exists() and (out / "best.pt").exists()
        checkpoint = out / "last.pt"
        status, loaded = request("POST", "/api/checkpoint", {"path": str(checkpoint)})
        assert status == 200 and loaded["epoch"] == 1
        resumed = dict(body, mode="resume", checkpoint=str(checkpoint),
                       config=dict(loaded["config"], epochs=2))
        assert request("POST", "/api/validate", dict(resumed, config=dict(resumed["config"], learning_rate=0.1)))[0] == 400
        status, second = request("POST", "/api/start", resumed)
        assert status == 201
        done = wait_for_job(request, second["id"])
        assert done["status"] == "completed", done["error"]
        assert len(done["history"]) == 2 and done["history"][-1]["step"] == 4
        assert load_checkpoint(checkpoint)["epoch"] == 2
        status, log = request("GET", f"/api/jobs/{second['id']}/log")
        assert status == 200 and b"Checkpoint:" in log
        assert request("GET", f"/api/jobs/{second['id']}?after={done['cursor']}")[1]["logs"] == []
        assert (manager.state_dir / second["id"] / "session.json").is_file()
        # UI progress/stop hooks and resume must not change the training trajectory.
        expected = load_checkpoint(run_training(Config(**resumed["config"]), motion_export, tmp_path / "cli-control"))
        actual = load_checkpoint(checkpoint)
        for key in expected["model"]:
            torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)


def test_web_errors_preflight_and_local_boundary(motion_export, tmp_path):
    with dashboard(tmp_path) as (request, _):
        body = settings(motion_export, tmp_path)
        assert request("POST", "/api/start", body, **{"Origin": "https://example.org"})[0] == 403
        assert request("POST", "/api/start", body, **{"X-AGED-Token": "invalid"})[0] == 403
        assert request("GET", "/api/bootstrap", **{"Host": "attacker.example"})[0] == 403
        assert request("POST", "/api/start", dict(body, root=str(tmp_path / "missing")))[0] == 400
        assert not Path(body["out"]).exists()
        browse_status, browse = request("GET", "/api/browse")
        assert browse_status == 200 and browse["path"] == str(tmp_path)
        bad_checkpoint = tmp_path / "bad.pt"
        bad_checkpoint.write_text("not a checkpoint")
        assert request("POST", "/api/checkpoint", {"path": str(bad_checkpoint)})[0] == 400
        torch.save(["not an AGED checkpoint"], bad_checkpoint)
        assert request("POST", "/api/checkpoint", {"path": str(bad_checkpoint)})[0] == 400
        # Paths exist, but malformed content must surface as a real worker traceback.
        bad_root = tmp_path / "bad-data"
        for subject in ("S1", "S6", "S7", "S8", "S9", "S11"):
            (bad_root / subject).mkdir(parents=True)
            for take in (1, 2):
                (bad_root / subject / f"walking_{take}.txt").write_text("not,numeric")
        broken = dict(body, root=str(bad_root), config=dict(body["config"], actions=["walking"]))
        status, job = request("POST", "/api/start", broken)
        assert status == 201
        failed = wait_for_job(request, job["id"])
        assert failed["status"] == "failed" and failed["returncode"] != 0
        assert "ValueError" in failed["error"]
        assert any("Traceback" in row["text"] for row in failed["logs"])


def test_web_force_stop_cleans_up_worker(motion_export, tmp_path):
    with dashboard(tmp_path) as (request, manager):
        body = settings(motion_export, tmp_path)
        status, job = request("POST", "/api/start", body)
        assert status == 201
        assert request("POST", f"/api/jobs/{job['id']}/stop", {"force": True})[0] == 200
        done = wait_for_job(request, job["id"])
        assert done["status"] == "terminated"
        assert manager.jobs[job["id"]]["process"].poll() is not None


def test_web_shutdown_drains_logs_and_persists_terminal_state(motion_export, tmp_path):
    with dashboard(tmp_path) as (request, manager):
        status, job = request("POST", "/api/start", settings(motion_export, tmp_path))
        assert status == 201
        process = manager.jobs[job["id"]]["process"]
        reader = manager.jobs[job["id"]]["reader"]
        metadata = manager.state_dir / job["id"] / "session.json"
    assert process.poll() is not None and not reader.is_alive()
    assert json.loads(metadata.read_text())["status"] == "terminated"
