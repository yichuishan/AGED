import json
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from aged.config import Config
from aged.data import TRAIN_SUBJECTS, fit_normalisation
from aged.evaluate import mean_angle_error, pose_to_euler, published_baseline_report, raw_angle_error, sample_test_set, score, srnn_test_indices, zero_velocity
from aged.losses import expmap_to_rotation_matrix, geodesic_distance
from aged.runtime import build_model


def test_fixed_windows_rotation_units_and_baseline(motion_export):
    norm = fit_normalisation(motion_export, TRAIN_SUBJECTS, ["walking", "eating"])
    samples = sample_test_set(motion_export, norm, ["walking"], horizon=25)
    observed, target = samples["walking"]
    assert observed.shape == (8, 50, 8) and target.shape == (8, 25, 8)
    assert srnn_test_indices((210, 210)) == srnn_test_indices((210, 210), horizon=25)
    assert [take for take, index in srnn_test_indices((210, 210))] == [1, 2] * 4
    assert all(66 <= index < 110 for take, index in srnn_test_indices((210, 210)))
    assert all(value == 0 for value in mean_angle_error(target, target, norm).values())
    baseline = zero_velocity(observed, 25)
    assert baseline.shape == target.shape
    assert mean_angle_error(baseline, target, norm)[400] > 0
    metrics = score(motion_export, actions=["walking", "eating"], horizon=25)
    assert set(metrics["average"]) == {"zero_velocity"}
    _, _, raw_observed, raw_target = sample_test_set(motion_export, norm, ["walking"], 25, include_raw=True)["walking"]
    assert raw_observed.shape[-1] == raw_target.shape[-1] == 99
    raw_prediction = zero_velocity(raw_observed, 25)
    raw_prediction[..., :6] += 1e4  # Global motion cannot contribute to the body metric.
    np.testing.assert_allclose(list(raw_angle_error(raw_prediction, raw_target).values()),
                               list(metrics["per_action"]["walking"]["zero_velocity"].values()))
    # A channel absent from training can vary in S5; its real target must survive.
    novel_target = raw_target.copy()
    novel_target[..., 30] = np.linspace(0, 0.5, 25)
    assert raw_angle_error(raw_prediction, novel_target)[400] > raw_angle_error(raw_prediction, raw_target)[400]
    # Exercise actual Euler conversion and SO(3) units used by the complete pipeline.
    raw = np.zeros((1, 2, 99))
    raw[..., 6] = math.pi / 2
    euler = pose_to_euler(raw)
    np.testing.assert_allclose(np.abs(euler[..., 3]), math.pi / 2, atol=1e-7)
    np.testing.assert_array_equal(euler[..., :3], 0)
    rotations = torch.tensor([[0., 0., 0.], [math.pi / 2, 0., 0.], [math.pi, 0., 0.]], dtype=torch.float64, requires_grad=True)
    distance = geodesic_distance(rotations, torch.zeros_like(rotations))
    torch.testing.assert_close(distance, torch.tensor([0., math.pi / 2, math.pi], dtype=torch.float64))
    distance.sum().backward()
    assert torch.isfinite(rotations.grad).all()
    # Crucial zero-angle derivative: small axis-angle perturbations must rotate.
    jacobian = torch.autograd.functional.jacobian(expmap_to_rotation_matrix, torch.zeros(3, dtype=torch.float64))
    assert jacobian.abs().sum() == 6
    for invalid in ({"horizon": 0}, {"horizon": -1}, {"horizon": 101}, {"horizon": 2.5},
                    {"actions": []}, {"actions": ["walking", "walking"]}):
        with pytest.raises(ValueError, match="valid actions"):
            score(motion_export, **invalid)
    # A matching 80-ms number must not certify missing 160/320/400-ms entries.
    partial = {"per_action": {"walking": {"zero_velocity": {80: 0.39}}}}
    report = published_baseline_report(partial)
    assert not report["passed"] and report["checked_entries"] == 4
    assert sum(row["actual"] is None for row in report["entries"]) == 3


def test_paper_sized_model_runs_the_complete_objective(motion_export):
    from aged.losses import GeodesicLoss
    norm = fit_normalisation(motion_export, TRAIN_SUBJECTS, ["walking"])
    model = build_model(Config(), norm).eval()
    observed = torch.zeros(2, 50, norm.n_dims)
    with torch.no_grad():
        predicted = model(observed, torch.tensor([12, 12]))
        geometry = GeodesicLoss(norm)(predicted, torch.zeros_like(predicted))
        fidelity = model.fidelity_discriminator(predicted)
        continuity = model.continuity_discriminator(torch.cat((observed, predicted), 1))
    assert predicted.shape == (2, 10, norm.n_dims)
    assert torch.isfinite(geometry) and torch.isfinite(fidelity).all() and torch.isfinite(continuity).all()
    assert ((fidelity >= 0) & (fidelity <= 1)).all()


def test_actual_train_resume_evaluate_commands(motion_export, tmp_path):
    root = Path(__file__).resolve().parents[2]
    run = tmp_path / "cli-run"
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))

    def execute(arguments):
        completed = subprocess.run([sys.executable, *arguments], cwd=root, env=environment,
                                   capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed.stdout

    tracking_args = ["--track", "tensorboard"] if importlib.util.find_spec("tensorboard") else []
    execute(["-m", "aged.train", "--root", str(motion_export), "--out", str(run),
             "--actions", "walking", "eating", "--epochs", "1", "--steps-per-epoch", "2",
             "--batch-size", "2", "--hidden-dim", "16", "--critic-hidden-dim", "12", "--device", "cpu", *tracking_args])
    execute(["-m", "aged.train", "--root", str(motion_export), "--out", str(run),
             "--resume", str(run / "last.pt"), "--epochs", "2"])
    output = tmp_path / "scores.json"
    execute(["-m", "aged.evaluate", "--root", str(motion_export), "--checkpoint", str(run / "last.pt"), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["epoch"] == 2
    assert set(result["average"]["model"]) == {"80", "160", "320", "400"}
    if tracking_args:
        assert list((run / "tb").glob("events.out.tfevents.*"))
