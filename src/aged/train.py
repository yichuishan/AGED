"""Complete AGED trainer: supervised geometry and alternating recurrent adversaries."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import random
import shlex
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import Config, VARIANTS
from . import data as D
from .evaluate import mean_angle_error
from .losses import GeodesicLoss
from .runtime import CHECKPOINT_VERSION, build_model, load_checkpoint, pick_device, save_checkpoint, seed_everything
from .tracking import Tracker, environment


def critic_parameters(model):
    return [parameter for critic in (model.fidelity_discriminator, model.continuity_discriminator)
            if critic is not None for parameter in critic.parameters()]


def critic_inputs(model, observed, predicted, target):
    if model.fidelity_discriminator is not None:
        yield "fidelity", model.fidelity_discriminator, predicted, target
    if model.continuity_discriminator is not None:
        yield "continuity", model.continuity_discriminator, torch.cat((observed, predicted), 1), torch.cat((observed, target), 1)


def reconstruction(config, geometry, prediction, target):
    if config.variant in ("plain", "euclidean"):
        return F.mse_loss(prediction, target)
    return geometry(prediction, target)


def train_step(model, geometry, batch, generator_optimizer, discriminator_optimizer, config):
    observed, target, action = batch
    generator_optimizer.zero_grad(set_to_none=True)
    predicted = model.predict(observed, action, config.prediction_length)
    d_parameters = critic_parameters(model)
    diagnostics = {}
    if discriminator_optimizer is not None:
        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss = predicted.new_zeros(())
        for name, critic, fake, real in critic_inputs(model, observed, predicted.detach(), target):
            fake_logits, real_logits = critic.logits(fake), critic.logits(real)
            discriminator_loss = discriminator_loss + F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
            diagnostics[name + "_accuracy"] = float(((fake_logits < 0).float().mean() + (real_logits >= 0).float().mean()).detach() / 2)
        discriminator_loss.backward()
        d_norm = torch.nn.utils.clip_grad_norm_(d_parameters, config.gradient_clip, error_if_nonfinite=True)
        discriminator_optimizer.step()
        diagnostics.update(discriminator_loss=float(discriminator_loss.detach()), discriminator_grad_norm=float(d_norm))
        discriminator_optimizer.zero_grad(set_to_none=True)

    # Freeze discriminator weights while preserving d(score)/d(predicted).
    for parameter in d_parameters:
        parameter.requires_grad_(False)
    try:
        regression = reconstruction(config, geometry, predicted, target)
        adversarial = predicted.new_zeros(())
        for _, critic, fake, _ in critic_inputs(model, observed, predicted, target):
            logits = critic.logits(fake)
            if config.adversarial_objective == "minimax":
                adversarial = adversarial - F.softplus(logits).mean()  # log(1 - sigmoid(logit)), Eq. 6/7.
            else:
                adversarial = adversarial + F.softplus(-logits).mean()  # Explicit optional deviation.
        loss = regression + config.lambda_adv * adversarial
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite generator objective")
        loss.backward()
        g_norm = torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), config.gradient_clip, error_if_nonfinite=True)
        generator_optimizer.step()
    finally:
        for parameter in d_parameters:
            parameter.requires_grad_(True)
    diagnostics.update(generator_loss=float(loss.detach()), reconstruction=float(regression.detach()),
                       generator_adversarial=float(adversarial.detach()), generator_grad_norm=float(g_norm))
    return diagnostics


def validate(model, geometry, loader, norm, device, config):
    model.eval()
    total, count, predictions, targets = 0.0, 0, [], []
    with torch.no_grad():
        for observed, target, action in loader:
            observed, target, action = observed.to(device), target.to(device), action.to(device)
            predicted = model.predict(observed, action)
            total += float(reconstruction(config, geometry, predicted, target)) * len(observed)
            count += len(observed)
            predictions.append(predicted.cpu().numpy())
            targets.append(target.cpu().numpy())
    model.train()
    return total / count, mean_angle_error(np.concatenate(predictions), np.concatenate(targets), norm)


def run_training(config: Config, root, out, resume=None):
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    device = pick_device(config.device)
    run = Path(out)
    saved = load_checkpoint(resume) if resume else None
    if saved:
        old = saved["config"]
        for key, value in config.to_dict().items():
            if key not in ("epochs", "device", "threads", "num_workers", "track") and value != old[key]:
                raise ValueError(f"resume cannot change {key}; start a separate run instead")
        norm = D.Normalisation.from_dict(saved["normalisation"])
        start_epoch = saved["epoch"] + 1
        if start_epoch > config.epochs:
            raise ValueError("epochs must exceed the resumed checkpoint epoch")
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError(f"{run} is not empty; choose a new --out or use --resume")
        subjects = [s for s in D.TRAIN_SUBJECTS if s != config.val_subject]
        norm = D.fit_normalisation(root, subjects, config.actions)
        start_epoch = 1
    run.mkdir(parents=True, exist_ok=True)
    subjects = [s for s in D.TRAIN_SUBJECTS if s != config.val_subject]
    train_set = D.WindowDataset(root, norm, subjects, config.actions, config.observed_length,
                               config.prediction_length, config.steps_per_epoch * config.batch_size, config.seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    validation_loader = None
    if config.val_subject:
        validation_set = D.WindowDataset(root, norm, [config.val_subject], config.actions,
                                        config.observed_length, config.prediction_length, config.validation_windows, 999)
        validation_loader = DataLoader(validation_set, batch_size=config.batch_size, num_workers=config.num_workers)
    model = build_model(config, norm).to(device)
    geometry = GeodesicLoss(norm, config.geodesic_reduction).to(device)
    generator_optimizer = torch.optim.Adam(model.predictor.parameters(), lr=config.learning_rate)
    d_parameters = critic_parameters(model)
    discriminator_optimizer = torch.optim.Adam(d_parameters, lr=config.learning_rate) if d_parameters else None
    history, best = [], float("inf")
    if saved:
        model.load_state_dict(saved["model"])
        generator_optimizer.load_state_dict(saved["generator_optimizer"])
        if discriminator_optimizer:
            discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
        history, best = saved["history"], saved["best_score"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    (run / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (run / "normalisation.json").write_text(json.dumps(norm.to_dict(), indent=2) + "\n")
    with (run / "commands.txt").open("a", encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable, *sys.argv]) + "\n")
    tracker = Tracker(config.track, str(run), config.to_dict())
    runtime = environment()
    runtime["selected_device"] = str(device)
    print(f"variant={config.variant} device={device} motion_dim={norm.n_dims} horizon={config.prediction_length}")
    print(f"train_subjects={subjects}; validation={config.val_subject}; S5 excluded from training")
    model.train()
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            started = time.monotonic()
            train_set.set_epoch(epoch - 1)
            totals = {}
            for batch in train_loader:
                diagnostics = train_step(model, geometry, tuple(value.to(device) for value in batch),
                                         generator_optimizer, discriminator_optimizer, config)
                for key, value in diagnostics.items():
                    totals[key] = totals.get(key, 0.0) + value
            row = {"epoch": epoch, **{key: value / len(train_loader) for key, value in totals.items()},
                   "seconds": time.monotonic() - started}
            improved = False
            if validation_loader is not None:
                val_loss, errors = validate(model, geometry, validation_loader, norm, device, config)
                row.update(validation_loss=val_loss, validation_angle_error=errors)
                score = float(np.mean(list(errors.values())))
                improved = score < best
                best = min(best, score)
            history.append(row)
            payload = {
                "format_version": CHECKPOINT_VERSION, "config": config.to_dict(), "epoch": epoch,
                "model": model.state_dict(), "normalisation": norm.to_dict(), "action_names": D.ACTIONS,
                "generator_optimizer": generator_optimizer.state_dict(),
                "discriminator_optimizer": discriminator_optimizer.state_dict() if discriminator_optimizer else None,
                "history": history, "best_score": best,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }
            save_checkpoint(run / "last.pt", payload)
            if improved:
                save_checkpoint(run / "best.pt", payload)
            results = {"config": config.to_dict(), "environment": runtime, "history": history,
                       "data_root": str(Path(root).resolve()), "train_subjects": subjects,
                       "test_subject_used_for_selection": False}
            (run / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            tracker.log(epoch, {key: value for key, value in row.items() if isinstance(value, (float, int))})
            print(json.dumps(row, allow_nan=False))
    finally:
        tracker.close()
    return run / "last.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True, help="new run directory, or existing run with --resume")
    parser.add_argument("--config", help="YAML config; defaults to paper short-term settings")
    parser.add_argument("--resume")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--actions", nargs="+", choices=D.ACTIONS)
    for name in ("observed_length", "prediction_length", "hidden_dim", "critic_hidden_dim", "epochs",
                 "batch_size", "steps_per_epoch", "validation_windows", "seed", "threads", "num_workers"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("learning_rate", "lambda_adv", "gradient_clip"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--device")
    parser.add_argument("--val-subject", choices=D.TRAIN_SUBJECTS)
    parser.add_argument("--track", choices=("none", "tensorboard"))
    parser.add_argument("--adversarial-objective", choices=("minimax", "non_saturating"))
    parser.add_argument("--geodesic-reduction", choices=("sum", "mean"))
    args = parser.parse_args()
    if args.resume and args.config:
        parser.error("resume loads its saved configuration; do not supply --config")
    values = load_checkpoint(args.resume)["config"] if args.resume else Config.from_yaml(args.config).to_dict() if args.config else Config().to_dict()
    for field in fields(Config):
        value = getattr(args, field.name, None)
        if value is not None:
            values[field.name] = value
    checkpoint = run_training(Config(**values), args.root, args.out, args.resume)
    print(f"Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
