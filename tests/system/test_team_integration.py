"""Team contributions exercised through the installed data/model/train/eval pipeline."""

from dataclasses import replace
import json
import shlex
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from aged import critics as C
from aged import data as D
from aged.config import Config
from aged.evaluate import published_baseline_report, score
from aged.losses import GeodesicLoss
from aged.runtime import build_model, load_checkpoint, load_model, seed_everything
from aged import train as T


def config(**changes):
    settings = dict(actions=["walking", "eating"], hidden_dim=16, critic_hidden_dim=12,
                    epochs=1, steps_per_epoch=2, batch_size=2, validation_windows=4,
                    device="cpu", seed=4)
    settings.update(changes)
    return Config(**settings)


@pytest.mark.parametrize("layout", ["flat", "column"])
def test_team_critic_objectives_and_gradient_isolation_in_pipeline(motion_export, layout):
    settings = config(critic_architecture="direct", critic_output_layout=layout)
    norm = D.fit_normalisation(motion_export, D.TRAIN_SUBJECTS, settings.actions)
    observed, target, action = D.WindowDataset(motion_export, norm, ["S1"], settings.actions, length=2)[0]
    model = build_model(settings, norm)
    predicted = model(observed[None], action[None])
    fidelity, continuity = model.fidelity_discriminator, model.continuity_discriminator
    assert not ({p.data_ptr() for p in fidelity.parameters()} & {p.data_ptr() for p in continuity.parameters()})
    assert fidelity.gru.input_size == norm.n_dims
    assert fidelity(predicted).shape == ((1,) if layout == "flat" else (1, 1))
    joined = C.continuity_input(observed[None], predicted)
    torch.testing.assert_close(joined[:, :50], observed[None])
    torch.testing.assert_close(joined[:, 50:], predicted)
    jumped = joined.detach().clone()
    jumped[:, 50] += 1
    assert not torch.equal(continuity(joined.detach()), continuity(jumped))
    discriminator, _ = C.critic_loss(fidelity, target[None], predicted.detach())
    discriminator.backward()
    assert all(parameter.grad is None for parameter in model.predictor.parameters())
    for module in (fidelity, continuity):
        module.zero_grad(set_to_none=True)
        module.requires_grad_(False)
    geometry = GeodesicLoss(norm)
    loss = geometry(predicted, target[None]) + 0.6 * (C.generator_loss(fidelity, predicted) + C.generator_loss(continuity, joined))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.predictor.parameters())
    assert all(p.grad is None for module in (fidelity, continuity) for p in module.parameters())

    # Preserve the fidelity branch's objective/derivative regression protection.
    class FixedLogits(nn.Module):
        def __init__(self):
            super().__init__()
            self.values = nn.Parameter(torch.tensor([-1000., -2., 0., 2., 1000.], dtype=torch.float64))

        def logits(self, sequence):
            return self.values

    probe = FixedLogits()
    for non_saturating in (False, True):
        probe.zero_grad(set_to_none=True)
        objective = C.generator_loss_non_saturating if non_saturating else C.generator_loss
        value = objective(probe, None)
        value.backward()
        expected = (probe.values.detach().sigmoid() - 1 if non_saturating else -probe.values.detach().sigmoid()) / 5
        torch.testing.assert_close(probe.values.grad, expected)
        assert torch.isfinite(value)
    exported = {key.replace("output_projection.", "classifier."): value for key, value in fidelity.state_dict().items()}
    imported = C.Critic(n_dims=norm.n_dims, hidden=12, architecture="direct", output_layout=layout)
    imported.load_team_state_dict(exported)
    torch.testing.assert_close(imported(predicted.detach()), fidelity(predicted.detach()), rtol=0, atol=0)
    with pytest.raises(ValueError, match="architecture='direct'"):
        C.Critic(norm.n_dims, 12).load_team_state_dict(exported)


@pytest.mark.parametrize("separate", [False, True])
def test_team_optimizer_schedule_and_resume(motion_export, tmp_path, separate):
    settings = config(epochs=2, optimizer="sgd", critic_optimizer="adam",
                      critic_learning_rate=0.001, learning_rate_decay=0.5,
                      learning_rate_decay_every=2, separate_critic_optimizers=separate,
                      critic_architecture="direct", critic_output_layout="column",
                      sampling_strategy="recordings", minimum_start=16,
                      adversarial_objective="non_saturating")
    whole = load_checkpoint(T.run_training(settings, motion_export, tmp_path/"whole"))
    first = T.run_training(replace(settings, epochs=1), motion_export, tmp_path/"part")
    resumed = load_checkpoint(T.run_training(settings, motion_export, tmp_path/"part", resume=first))
    assert resumed["step"] == 4
    assert resumed["history"][-1]["learning_rate"] == pytest.approx(0.0025)
    assert not resumed["generator_optimizer"]["state"]  # Plain SGD, no hidden momentum.
    if separate:
        assert set(resumed["discriminator_optimizers"]) == {"fidelity", "continuity"}
        assert all(item["state"] for item in resumed["discriminator_optimizers"].values())
    else:
        assert resumed["discriminator_optimizer"]["state"]
    for name, value in whole["model"].items():
        torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
    result = score(motion_export, first)
    assert set(result["average"]) == {"model", "zero_velocity"}


def test_warm_start_keeps_predictor_and_creates_fresh_critics(motion_export, tmp_path, monkeypatch):
    settings = config(variant="plain", learning_rate_decay=0.5, learning_rate_decay_every=2)
    source = T.run_training(settings, motion_export, tmp_path/"plain")
    saved = load_checkpoint(source)
    warm_settings = replace(settings, variant="full", epochs=2, critic_architecture="direct", separate_critic_optimizers=True)
    norm = D.Normalisation.from_dict(saved["normalisation"])
    seed_everything(warm_settings.seed)
    fresh = build_model(warm_settings, norm)
    expected_critics = {key: value.clone() for key, value in fresh.state_dict().items() if not key.startswith("predictor.")}
    original = T.train_step
    seen = []

    def check_first_step(model, geometry, batch, g_optimizer, d_optimizer, cfg):
        if not seen:
            for key, value in model.state_dict().items():
                expected = saved["model"][key] if key.startswith("predictor.") else expected_critics[key]
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            assert g_optimizer.state
            assert all(not item.state for item in d_optimizer.values())
        seen.append(True)
        return original(model, geometry, batch, g_optimizer, d_optimizer, cfg)

    with monkeypatch.context() as patch:
        patch.setattr(T, "train_step", check_first_step)
        path = T.run_training(warm_settings, motion_export, tmp_path/"warm", init_from=source)
    checkpoint = load_checkpoint(path)
    assert checkpoint["epoch"] == 2 and checkpoint["step"] == 4
    assert len(checkpoint["history"]) == 1
    assert checkpoint["history"][0]["learning_rate"] == pytest.approx(0.0025)
    assert checkpoint["initialization_checkpoint"] == str(source)
    frozen_path = T.run_training(replace(warm_settings, learning_rate=0, critic_learning_rate=0),
                                 motion_export, tmp_path/"frozen-warm", init_from=source)
    frozen = load_checkpoint(frozen_path)
    for key, value in frozen["model"].items():
        expected = saved["model"][key] if key.startswith("predictor.") else expected_critics[key]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    source.unlink()
    continued = T.run_training(replace(warm_settings, epochs=3), motion_export, tmp_path/"warm", resume=path)
    assert load_checkpoint(continued)["history"][-1]["learning_rate"] == pytest.approx(0.00125)


def test_early_stop_and_periodic_checkpoints(motion_export, tmp_path, monkeypatch):
    original = T.validate

    def plateau(*args):
        loss, errors = original(*args)  # Still execute the actual validation pipeline.
        return 1.0, errors  # Controlled selection plateau, independent of toy-data noise.

    monkeypatch.setattr(T, "validate", plateau)
    settings = config(epochs=5, val_subject="S11", select_on="val_loss", patience=1, save_every=1)
    path = T.run_training(settings, motion_export, tmp_path/"plateau")
    saved = load_checkpoint(path)
    assert saved["epoch"] == 2 and saved["since_best"] == 1
    assert load_checkpoint(path.parent/"best.pt")["epoch"] == 1
    assert (path.parent/"epoch_001.pt").exists() and (path.parent/"epoch_002.pt").exists()
    assert not (path.parent/"epoch_003.pt").exists()


def test_uniform_sampling_and_minimum_start(motion_export, tmp_path):
    root = tmp_path/"export"
    (root/"S1").mkdir(parents=True)
    for action in ("walking", "eating"):
        for take in (1, 2):
            raw = np.loadtxt(motion_export/"S1"/f"{action}_{take}.txt", delimiter=",")
            if action == "walking" and take == 1:
                raw = np.tile(raw, (3, 1))
            raw[:, 6] = np.arange(len(raw))  # Retained body-coordinate marker.
            np.savetxt(root/"S1"/f"{action}_{take}.txt", raw, delimiter=",")
    norm = D.fit_normalisation(root, ["S1"], ["walking", "eating"])
    ratios = {}
    for strategy in ("windows", "recordings"):
        dataset = D.WindowDataset(root, norm, ["S1"], ["walking", "eating"],
                                  length=1000, sampling_strategy=strategy, minimum_start=16)
        count = 0
        for index in range(1000):
            observed, target, label = dataset[index]
            raw = norm.restore(observed.numpy())
            assert raw[0, 6] / 2 >= 16 - 1e-4
            count += int(label) == D.ACTIONS.index("walking")
        ratios[strategy] = count / 1000
    assert 0.45 < ratios["recordings"] < 0.55
    assert ratios["windows"] > 0.6


def test_v1_checkpoint_missing_new_fields_is_backward_compatible(motion_export, tmp_path):
    settings = config(epochs=2)
    whole = load_checkpoint(T.run_training(settings, motion_export, tmp_path/"whole"))
    path = T.run_training(replace(settings, epochs=1), motion_export, tmp_path/"part")
    old = torch.load(path, weights_only=True)
    new_fields = {"optimizer", "critic_optimizer", "critic_learning_rate", "learning_rate_decay", "learning_rate_decay_every",
                  "sampling_strategy", "minimum_start", "critic_architecture", "critic_output_layout", "separate_critic_optimizers",
                  "patience", "save_every", "select_on", "project"}
    old["config"] = {key: value for key, value in old["config"].items() if key not in new_fields}
    for key in ("step", "since_best", "discriminator_optimizers", "initialization_checkpoint"):
        old.pop(key, None)
    torch.save(old, path)
    actual = load_checkpoint(T.run_training(settings, motion_export, tmp_path/"part", resume=path))
    for key, value in whole["model"].items():
        torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)


def test_team_cli_aliases_and_baseline_gate(motion_export, tmp_path):
    def execute(arguments):
        return subprocess.run([sys.executable, "-m", *arguments], capture_output=True, text=True, timeout=90)

    run = tmp_path/"cli"
    result = execute(["aged.train", "--root", str(motion_export), "--out", str(run), "--variant", "adv-euc",
                      "--actions", "walking", "eating", "--epochs", "1", "--steps-per-epoch", "2", "--batch-size", "2",
                      "--hidden", "16", "--critic-hidden-dim", "12", "--optimizer", "sgd", "--critic-optimizer", "adam",
                      "--lr", "0.005", "--lr-critic", "0.001", "--lr-decay", "0.5", "--lr-decay-every", "2",
                      "--adv-form", "non-saturating", "--critic-architecture", "direct", "--critic-output-layout", "column",
                      "--separate-critic-optimizers", "--sampling-strategy", "recordings", "--minimum-start", "16",
                      "--final", "--device", "cpu"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert load_checkpoint(run/"last.pt")["config"]["variant"] == "euclidean"
    recorded = shlex.split((run/"commands.txt").read_text().strip())
    assert recorded[1:3] == ["-m", "aged.train"]
    recorded[recorded.index("--out") + 1] = str(tmp_path/"cli-replay")
    replay = subprocess.run(recorded, capture_output=True, text=True, timeout=90)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    warm = tmp_path/"cli-warm"
    result = execute(["aged.train", "--root", str(motion_export), "--out", str(warm),
                      "--init-from", str(run/"last.pt"), "--variant", "full", "--epochs", "2"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert load_checkpoint(warm/"last.pt")["step"] == 4
    output = tmp_path/"baseline.json"
    result = execute(["aged.evaluate", "--root", str(motion_export), "--baseline", "--actions", "walking", "eating",
                      "--check-baseline", "--output", str(output)])
    assert result.returncode != 0  # Synthetic data must not be certified as the published benchmark.
    actual = json.loads(output.read_text())
    assert actual["baseline_check"]["checked_entries"] == 8
    assert not actual["baseline_check"]["passed"]
    assert published_baseline_report(actual) == actual["baseline_check"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware/runtime not available")
def test_team_direct_critics_and_optimizer_modes_on_cuda(motion_export, tmp_path):
    settings = config(device="cuda", critic_architecture="direct", critic_output_layout="column",
                      separate_critic_optimizers=True, optimizer="sgd", critic_optimizer="adam",
                      sampling_strategy="recordings", minimum_start=16)
    path = T.run_training(settings, motion_export, tmp_path/"team-cuda")
    assert np.isfinite(score(motion_export, path, device="cuda")["average"]["model"][400])
