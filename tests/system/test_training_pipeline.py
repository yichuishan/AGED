from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from aged.config import Config, VARIANTS
from aged.data import ACTIONS, Normalisation, TRAIN_SUBJECTS, WindowDataset, fit_normalisation
from aged.evaluate import score
from aged.losses import GeodesicLoss
from aged.runtime import build_model, load_checkpoint, load_model, seed_everything
from aged.train import run_training


def small_config(**overrides):
    values = dict(actions=["walking", "eating"], hidden_dim=16, critic_hidden_dim=12,
                  epochs=1, steps_per_epoch=2, batch_size=2, validation_windows=4,
                  val_subject="S11", device="cpu", seed=4)
    values.update(overrides)
    return Config(**values)


@pytest.mark.parametrize("variant", VARIANTS)
def test_complete_training_checkpoint_and_s5_evaluation(motion_export, tmp_path, variant):
    config = small_config(variant=variant)
    norm = fit_normalisation(motion_export, [s for s in TRAIN_SUBJECTS if s != "S11"], config.actions)
    assert norm.n_dims == 8
    assert np.all(norm.used_dims >= 6)
    seed_everything(config.seed)
    initial = build_model(config, norm).state_dict()
    checkpoint = run_training(config, motion_export, tmp_path / variant)
    saved = load_checkpoint(checkpoint)
    assert saved["epoch"] == 1
    assert len(saved["history"]) == 1
    assert saved["action_names"] == ACTIONS
    assert (checkpoint.parent / "best.pt").is_file()
    for name in ("predictor", "fidelity_discriminator", "continuity_discriminator"):
        keys = [key for key in initial if key.startswith(name + ".")]
        if keys:
            assert any(not torch.equal(initial[key], saved["model"][key]) for key in keys), name
    row = saved["history"][0]
    assert all(np.isfinite(value) for value in row.values() if isinstance(value, (int, float)))
    assert row["generator_grad_norm"] > 0
    if saved["discriminator_optimizer"]:
        assert row["discriminator_grad_norm"] > 0
    metadata = json.loads((checkpoint.parent / "results.json").read_text())
    assert "S5" not in metadata["train_subjects"]
    assert "S11" not in metadata["train_subjects"]
    metrics = score(motion_export, checkpoint)
    assert set(metrics["per_action"]) == {"walking", "eating"}
    assert metrics["test_subject"] == "S5"
    assert set(metrics["average"]) == {"model", "zero_velocity"}
    assert all(np.isfinite(list(values.values())).all() for values in metrics["average"].values())
    model, restored_norm, _, _ = load_model(checkpoint)
    data = WindowDataset(motion_export, norm, ["S1"], config.actions, length=2)
    observed, target, label = data[0]
    with torch.no_grad():
        short = model.predict(observed[None], label[None], 10)
        long = model.predict(observed[None], label[None], 25)
        repeat = model.predict(observed[None], label[None], 10)
    torch.testing.assert_close(short, long[:, :10], rtol=0, atol=0)
    torch.testing.assert_close(short, repeat, rtol=0, atol=0)
    assert restored_norm.to_dict() == norm.to_dict()


def test_resumed_training_exactly_matches_uninterrupted_training(motion_export, tmp_path):
    config = small_config(epochs=2, val_subject=None)
    uninterrupted = load_checkpoint(run_training(config, motion_export, tmp_path / "uninterrupted"))
    first = run_training(replace(config, epochs=1), motion_export, tmp_path / "resumed")
    resumed = load_checkpoint(run_training(config, motion_export, tmp_path / "resumed", resume=first))
    assert resumed["epoch"] == 2
    assert len(resumed["history"]) == 2
    for key in uninterrupted["model"]:
        torch.testing.assert_close(uninterrupted["model"][key], resumed["model"][key], rtol=0, atol=0)
    for name in ("generator_optimizer", "discriminator_optimizer"):
        for key, state in uninterrupted[name]["state"].items():
            for item, value in state.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, resumed[name]["state"][key][item], rtol=0, atol=0)
    with pytest.raises(ValueError, match="resume cannot change"):
        run_training(replace(config, epochs=3, lambda_adv=0.9), motion_export, tmp_path / "resumed", first)


def test_geometry_restoration_and_gradients_in_complete_model(motion_export):
    config = small_config()
    norm = fit_normalisation(motion_export, TRAIN_SUBJECTS, config.actions)
    dataset = WindowDataset(motion_export, norm, ["S1"], config.actions, length=2)
    observed, target, action = dataset[0]
    model = build_model(config, norm)
    geometry = GeodesicLoss(norm)
    prediction = model.predict(observed[None], action[None])
    loss = geometry(prediction, target[None])
    loss.backward()
    assert loss.item() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.predictor.parameters())
    assert geometry(prediction.detach(), prediction.detach()).item() == pytest.approx(0, abs=1e-6)
    raw = geometry.restore(target[None]).detach().numpy()
    np.testing.assert_allclose(raw, norm.restore(target[None].numpy()), atol=1e-7)
    with pytest.raises(ValueError, match="training subjects only"):
        fit_normalisation(motion_export, ["S5"], config.actions)
    dataset.set_epoch(1)
    assert not torch.equal(dataset[0][0], observed)


@pytest.mark.parametrize("objective", ["minimax", "non_saturating"])
def test_long_term_complete_training(motion_export, tmp_path, objective):
    config = small_config(prediction_length=25, adversarial_objective=objective,
                          share_recurrent_weights=(objective == "non_saturating"))
    checkpoint = run_training(config, motion_export, tmp_path / objective)
    metric = score(motion_export, checkpoint, horizon=100)
    assert set(metric["average"]["model"]) == {80, 160, 320, 400, 560, 1000}
    assert metric["prediction_length"] == 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware/runtime not available")
def test_complete_cuda_training(motion_export, tmp_path):
    checkpoint = run_training(small_config(device="cuda"), motion_export, tmp_path / "cuda")
    result = score(motion_export, checkpoint, device="cuda")
    assert np.isfinite(result["average"]["model"][400])
