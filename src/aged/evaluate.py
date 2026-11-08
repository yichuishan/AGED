"""Fixed eight-window S5 evaluation in Euler-angle space, always with zero-velocity."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .runtime import load_model, pick_device

TEST_SEED = 1234567890

# Team training/evaluation contribution: published zero-velocity reference table.
# Sitting at 160 ms uses the team's documented 0.63 correction, inferred from
# the table's average; the printed 1.63 is not silently treated as a match.
PUBLISHED_ZERO_VELOCITY = {
    "directions": (0.39, 0.59, 0.79, 0.89), "discussion": (0.31, 0.67, 0.94, 1.04),
    "eating": (0.27, 0.48, 0.73, 0.86), "greeting": (0.54, 0.89, 1.30, 1.49),
    "phoning": (0.64, 1.21, 1.65, 1.83), "posing": (0.28, 0.57, 1.13, 1.37),
    "purchases": (0.62, 0.88, 1.19, 1.27), "sitting": (0.40, 0.63, 1.02, 1.18),
    "sittingdown": (0.39, 0.74, 1.07, 1.19), "smoking": (0.26, 0.48, 0.97, 0.95),
    "takingphoto": (0.25, 0.51, 0.79, 0.92), "waiting": (0.34, 0.67, 1.22, 1.47),
    "walking": (0.39, 0.68, 0.99, 1.15), "walkingdog": (0.60, 0.98, 1.36, 1.50),
    "walkingtogether": (0.33, 0.66, 0.94, 0.99),
}


def published_baseline_report(result, tolerance=0.0055):
    """Compare actual baseline numbers; never label a protocol flag as accuracy proof."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("baseline tolerance must be finite and nonnegative")
    entries = []
    for action, row in result["per_action"].items():
        if action not in PUBLISHED_ZERO_VELOCITY:
            continue
        for ms, reference in zip((80, 160, 320, 400), PUBLISHED_ZERO_VELOCITY[action]):
            values = row["zero_velocity"]
            actual = values.get(ms, values.get(str(ms)))
            difference = abs(float(actual) - reference) if actual is not None else None
            entries.append({"action": action, "milliseconds": ms,
                            "actual": float(actual) if actual is not None else None,
                            "reference": reference, "absolute_error": difference,
                            "passed": bool(actual is not None and np.isfinite(actual) and difference <= tolerance)})
    return {"passed": bool(entries) and all(entry["passed"] for entry in entries),
            "checked_entries": len(entries), "tolerance": tolerance, "entries": entries,
            "note": "Short-term table only; Sitting 160 ms uses the documented inferred 0.63 correction."}


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
        actions = config.actions if actions is None else actions
        horizon = config.prediction_length if horizon is None else horizon
    else:
        model, config, saved = None, None, {}
        actions = D.ACTIONS if actions is None else actions
        horizon = 10 if horizon is None else horizon
    if (isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 100
            or not actions or len(set(actions)) != len(actions) or set(actions) - set(D.ACTIONS)):
        raise ValueError("choose valid actions and a prediction horizon in [1,100]")
    if model is None:
        norm = D.fit_normalisation(root, D.TRAIN_SUBJECTS, actions)
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


def main(argv=None):
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
    parser.add_argument("--check-baseline", action="store_true", help="compare real zero-velocity scores to the published table; nonzero exit if mismatched")
    parser.add_argument("--baseline-tolerance", type=float, default=0.0055)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    result = score(args.root, args.checkpoint, args.actions, args.horizon, args.device)
    if args.check_baseline:
        result["baseline_check"] = published_baseline_report(result, args.baseline_tolerance)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for action, row in result["per_action"].items():
        print(action, json.dumps(row))
    print(f"Evaluation written to {path}")
    if args.check_baseline and not result["baseline_check"]["passed"]:
        raise SystemExit("Zero-velocity baseline does not match the published table; inspect baseline_check in the saved JSON.")


if __name__ == "__main__":
    main()
