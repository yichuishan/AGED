"""Fixed eight-window S5 evaluation in Euler-angle space, always with zero-velocity."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .runtime import load_model, pick_device

TEST_SEED = 1234567890


def srnn_test_indices(lengths: tuple[int, int], horizon=10):
    """Eight alternating take windows from the protocol documented by the team.

    Independently implemented from the documented seed, bounds and prefix/suffix;
    no external implementation was retrieved or copied. Returned index is the
    first future frame; the preceding 50 frames are observed.
    """
    rng = np.random.RandomState(TEST_SEED)
    suffix = max(100, horizon)
    indices = []
    for sample in range(8):
        take = sample % 2
        upper = lengths[take] - 50 - suffix
        if upper <= 16:
            raise ValueError("test recordings are too short for fixed eight-seed evaluation")
        start = int(rng.randint(16, upper))
        indices.append((take + 1, start + 50))
    return indices


def sample_test_set(root, norm, actions=D.ACTIONS, horizon=10, include_raw=False):
    result = {}
    for action in actions:
        takes = {take: D.load_raw(root, D.TEST_SUBJECT, action, take) for take in (1, 2)}
        observed, target, raw_observed, raw_target = [], [], [], []
        for take, boundary in srnn_test_indices((len(takes[1]), len(takes[2])), horizon):
            raw_window = takes[take][boundary - 50:boundary + horizon]
            window = norm.apply(raw_window)
            observed.append(window[:50])
            target.append(window[50:])
            raw_observed.append(raw_window[:50])
            raw_target.append(raw_window[50:])
        result[action] = (np.stack(observed), np.stack(target))
        if include_raw:
            result[action] += (np.stack(raw_observed), np.stack(raw_target))
    return result


def rotation_to_euler(rotation):
    """Euler convention R = Rx(-a) Ry(-b) Rz(-c), including gimbal-lock handling."""
    sine = np.clip(-rotation[..., 0, 2], -1.0, 1.0)
    beta = np.arcsin(sine)
    alpha = np.arctan2(rotation[..., 1, 2], rotation[..., 2, 2])
    gamma = np.arctan2(rotation[..., 0, 1], rotation[..., 0, 0])
    singular = np.abs(np.cos(beta)) < 1e-7
    alpha = np.where(singular, np.arctan2(-rotation[..., 2, 1], rotation[..., 1, 1]), alpha)
    gamma = np.where(singular, 0.0, gamma)
    return np.stack((alpha, beta, gamma), -1)


def pose_to_euler(raw):
    """Convert full export poses; zero the six global-motion coordinates."""
    from .losses import expmap_to_rotation_matrix
    vectors = torch.from_numpy(np.ascontiguousarray(raw[..., 3:])).double().reshape(*raw.shape[:-1], 32, 3)
    rotations = expmap_to_rotation_matrix(vectors).numpy()
    euler = rotation_to_euler(rotations).reshape(*raw.shape[:-1], 96)
    euler[..., :3] = 0  # Root rotation; translation was already excluded.
    return euler


def mean_angle_error(predicted, target, norm):
    """Mean per-sequence Euler L2 error, excluding constant target channels.

    This is a norm in radians, not MSE or a geodesic training loss. Mask is
    determined from each ground-truth future only, never the prediction.
    """
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("expected matching [B,H,K] prediction and target arrays")
    return raw_angle_error(norm.restore(predicted), norm.restore(target))


def raw_angle_error(predicted_raw, target_raw):
    """Full-pose metric; S5 target channels are never reconstructed from train means."""
    pred = pose_to_euler(predicted_raw)
    truth = pose_to_euler(target_raw)
    keep = truth.std(axis=1) > 1e-4
    per_sequence = np.sqrt(np.sum((pred - truth) ** 2 * keep[:, None, :], axis=-1))
    per_frame = per_sequence.mean(axis=0)
    return {ms: float(per_frame[index]) for ms, index in D.REPORT_MS.items() if index < predicted_raw.shape[1]}


def zero_velocity(observed, horizon):
    return np.repeat(observed[:, -1:, :], horizon, axis=1)


def score(root, checkpoint=None, actions=None, horizon=None, device="cpu"):
    selected_device = pick_device(device)
    if checkpoint:
        model, norm, config, saved = load_model(checkpoint, str(selected_device))
        if config.observed_length != 50:
            raise ValueError("fixed evaluation protocol requires a 50-frame observed window")
        actions = actions or config.actions
        horizon = horizon or config.prediction_length
    else:
        model, config, saved = None, None, {}
        actions, horizon = actions or D.ACTIONS, horizon or 10
        norm = D.fit_normalisation(root, D.TRAIN_SUBJECTS, actions)
    if not 1 <= horizon <= 100 or set(actions) - set(D.ACTIONS):
        raise ValueError("choose valid actions and a prediction horizon in [1,100]")
    test_set = sample_test_set(root, norm, actions, horizon, include_raw=True)
    rows = {}
    for action, (observed, target, raw_observed, raw_target) in test_set.items():
        row = {"zero_velocity": raw_angle_error(zero_velocity(raw_observed, horizon), raw_target)}
        if model is not None:
            batch = torch.from_numpy(observed).to(selected_device)
            labels = torch.full((len(batch),), D.ACTIONS.index(action), dtype=torch.long, device=selected_device)
            with torch.no_grad():
                prediction = model.predict(batch, labels, horizon).cpu().numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError("nonfinite evaluation predictions")
            row["model"] = raw_angle_error(norm.restore(prediction), raw_target)
        rows[action] = row
    keys = next(iter(rows.values())).keys()
    average = {key: {ms: float(np.mean([row[key][ms] for row in rows.values()]))
                     for ms in next(iter(rows.values()))[key]} for key in keys}
    return {
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
        "epoch": saved.get("epoch"), "test_subject": D.TEST_SUBJECT,
        "protocol": "fixed-eight-seed-euler", "test_seed": TEST_SEED,
        "prediction_length": horizon, "sequences_per_action": 8,
        "training_config": config.to_dict() if config else None,
        "benchmark_claim": "Protocol implemented; published results require validation on licensed real data.",
        "per_action": rows, "average": average,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint")
    mode.add_argument("--baseline", action="store_true")
    parser.add_argument("--actions", nargs="+", choices=D.ACTIONS)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="experiments/evaluation.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = score(args.root, args.checkpoint, args.actions, args.horizon, args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for action, row in result["per_action"].items():
        print(action, json.dumps(row))
    print(f"Evaluation written to {path}")


if __name__ == "__main__":
    main()
