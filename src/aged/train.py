"""Complete AGED trainer: supervised geometry and alternating recurrent adversaries."""

import argparse
from dataclasses import fields
import hashlib
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
from . import critics as C
from .evaluate import mean_angle_error
from .losses import GeodesicLoss
from .runtime import CHECKPOINT_VERSION, build_model, load_checkpoint, pick_device, save_checkpoint, seed_everything
from .tracking import Tracker, environment


def critic_parameters(model):
    return [parameter for critic in (model.fidelity_discriminator, model.continuity_discriminator)
            if critic is not None for parameter in critic.parameters()]


def critic_inputs(model, observed, predicted, target):
    if model.fidelity_discriminator is not None:
        yield "fidelity", model.fidelity_discriminator, C.fidelity_input(predicted), C.fidelity_input(target)
    if model.continuity_discriminator is not None:
        yield "continuity", model.continuity_discriminator, C.continuity_input(observed, predicted), C.continuity_input(observed, target)


def build_optimiser(name, parameters, lr):
    """Team optimizer capability, without changing the deployed Adam default."""
    return {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[name](parameters, lr=lr)


def set_learning_rate(optimizer, base, step, decay, every):
    for group in optimizer.param_groups:
        group["lr"] = base * decay ** (step // every)


def discriminator_optimizers(model, config):
    rate = config.critic_learning_rate if config.critic_learning_rate is not None else config.learning_rate
    name = config.critic_optimizer or config.optimizer
    if not critic_parameters(model):
        return None
    if config.separate_critic_optimizers:
        return {key: build_optimiser(name, critic.parameters(), rate)
                for key, critic in (("fidelity", model.fidelity_discriminator),
                                    ("continuity", model.continuity_discriminator)) if critic is not None}
    return build_optimiser(name, critic_parameters(model), rate)


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
        separate = isinstance(discriminator_optimizer, dict)
        if not separate:
            discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss = predicted.new_zeros(())
        norms = []
        for name, critic, fake, real in critic_inputs(model, observed, predicted.detach(), target):
            if separate:
                loss, accuracy = C.critic_loss(critic, real, fake)
            else:
                fake_loss, real_loss, accuracy = C._critic_terms(critic, real, fake)
                loss = fake_loss + real_loss
            diagnostics[name + "_accuracy"] = accuracy
            diagnostics[name + "_loss"] = float(loss.detach())
            if separate:
                optimizer = discriminator_optimizer[name]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norms.append(float(torch.nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip, error_if_nonfinite=True)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                discriminator_loss = discriminator_loss + loss.detach()
            else:
                discriminator_loss = discriminator_loss + fake_loss + real_loss
        if separate:
            d_norm = sum(value ** 2 for value in norms) ** 0.5
        else:
            discriminator_loss.backward()
            d_norm = torch.nn.utils.clip_grad_norm_(d_parameters, config.gradient_clip, error_if_nonfinite=True)
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad(set_to_none=True)
        diagnostics.update(discriminator_loss=float(discriminator_loss.detach()), discriminator_grad_norm=float(d_norm))

    # Freeze discriminator weights while preserving d(score)/d(predicted).
    for parameter in d_parameters:
        parameter.requires_grad_(False)
    try:
        regression = reconstruction(config, geometry, predicted, target)
        adversarial = predicted.new_zeros(())
        for _, critic, fake, _ in critic_inputs(model, observed, predicted, target):
            if config.adversarial_objective == "minimax":
                adversarial = adversarial + C.generator_loss(critic, fake)
            else:
                adversarial = adversarial + C.generator_loss_non_saturating(critic, fake)
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


def run_training(config: Config, root, out, resume=None, init_from=None, command_args=None,
                 progress_callback=None, stop_requested=None):
    if resume and init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    torch.set_num_threads(config.threads)
    seed_everything(config.seed)
    device = pick_device(config.device)
    run = Path(out)
    saved = load_checkpoint(resume) if resume else None
    warm = load_checkpoint(init_from) if init_from else None
    if saved:
        old = saved["config"]
        for key, value in config.to_dict().items():
            if key not in ("epochs", "device", "threads", "num_workers", "track", "project", "patience", "save_every") and value != old[key]:
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
        start_epoch = warm["epoch"] + 1 if warm else 1
        if warm:
            previous = D.Normalisation.from_dict(warm["normalisation"])
            if not (np.array_equal(norm.used_dims, previous.used_dims)
                    and np.allclose(norm.mean, previous.mean, rtol=1e-7, atol=1e-9)
                    and np.allclose(norm.std, previous.std, rtol=1e-7, atol=1e-9)):
                raise ValueError("warm start requires the same normalisation statistics and feature mask")
            if config.optimizer != warm["config"]["optimizer"]:
                raise ValueError("warm start must retain the predictor optimizer type")
            if start_epoch > config.epochs:
                raise ValueError("epochs must exceed the warm-start checkpoint epoch")
    run.mkdir(parents=True, exist_ok=True)
    subjects = [s for s in D.TRAIN_SUBJECTS if s != config.val_subject]
    train_set = D.WindowDataset(root, norm, subjects, config.actions, config.observed_length,
                               config.prediction_length, config.steps_per_epoch * config.batch_size, config.seed,
                               config.sampling_strategy, config.minimum_start)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    validation_loader = None
    if config.val_subject:
        validation_set = D.WindowDataset(root, norm, [config.val_subject], config.actions,
                                        config.observed_length, config.prediction_length, config.validation_windows, 999,
                                        config.sampling_strategy, config.minimum_start)
        validation_loader = DataLoader(validation_set, batch_size=config.batch_size, num_workers=config.num_workers)
    model = build_model(config, norm).to(device)
    geometry = GeodesicLoss(norm, config.geodesic_reduction).to(device)
    generator_optimizer = build_optimiser(config.optimizer, model.predictor.parameters(), config.learning_rate)
    discriminator_optimizer = discriminator_optimizers(model, config)
    history, best = [], float("inf")
    step, since_best = 0, 0
    if warm:
        state = {key.removeprefix("predictor."): value for key, value in warm["model"].items() if key.startswith("predictor.")}
        model.predictor.load_state_dict(state, strict=True)
        generator_optimizer.load_state_dict(warm["generator_optimizer"])
        step = warm["step"]
    if saved:
        model.load_state_dict(saved["model"])
        generator_optimizer.load_state_dict(saved["generator_optimizer"])
        if isinstance(discriminator_optimizer, dict):
            for name, optimizer in discriminator_optimizer.items():
                optimizer.load_state_dict(saved["discriminator_optimizers"][name])
        elif discriminator_optimizer is not None:
            discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
        history, best = saved["history"], saved["best_score"]
        step, since_best = saved["step"], saved["since_best"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    config_text = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    (run / "config.json").write_text(config_text)
    snapshots = run / "configs"
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / (hashlib.sha256(config_text.encode()).hexdigest()[:16] + ".json")
    snapshot.write_text(config_text)
    (run / "normalisation.json").write_text(json.dumps(norm.to_dict(), indent=2) + "\n")
    if command_args is None:
        command_args = ["--root", str(Path(root).resolve()), "--out", str(run.resolve())]
        if resume:
            command_args += ["--resume", str(Path(resume).resolve())]
            for name in ("epochs", "device", "threads", "num_workers", "track", "project", "patience", "save_every"):
                command_args += ["--" + name.replace("_", "-"), str(getattr(config, name))]
        else:
            command_args += ["--config", str(snapshot.resolve())]
            if init_from:
                command_args += ["--init-from", str(Path(init_from).resolve())]
    with (run / "commands.txt").open("a", encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable, "-m", "aged.train", *command_args]) + "\n")
    tracker = Tracker(config.track, str(run), config.to_dict(), project=config.project)
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
            for batch_index, batch in enumerate(train_loader, 1):
                set_learning_rate(generator_optimizer, config.learning_rate, step,
                                  config.learning_rate_decay, config.learning_rate_decay_every)
                optimizers = (discriminator_optimizer.values() if isinstance(discriminator_optimizer, dict)
                              else [discriminator_optimizer] if discriminator_optimizer is not None else [])
                rate = config.critic_learning_rate if config.critic_learning_rate is not None else config.learning_rate
                for optimizer in optimizers:
                    set_learning_rate(optimizer, rate, step, config.learning_rate_decay, config.learning_rate_decay_every)
                diagnostics = train_step(model, geometry, tuple(value.to(device) for value in batch),
                                         generator_optimizer, discriminator_optimizer, config)
                step += 1
                for key, value in diagnostics.items():
                    totals[key] = totals.get(key, 0.0) + value
                if progress_callback is not None:
                    progress_callback({"type": "batch", "epoch": epoch, "epochs": config.epochs,
                                       "batch": batch_index, "steps_per_epoch": config.steps_per_epoch,
                                       "step": step, "metrics": diagnostics})
            row = {"epoch": epoch, **{key: value / len(train_loader) for key, value in totals.items()},
                   "seconds": time.monotonic() - started, "step": step,
                   "learning_rate": generator_optimizer.param_groups[0]["lr"]}
            improved = False
            if validation_loader is not None:
                val_loss, errors = validate(model, geometry, validation_loader, norm, device, config)
                row.update(validation_loss=val_loss, validation_angle_error=errors)
                score = val_loss if config.select_on == "val_loss" else float(np.mean(list(errors.values())))
                improved = score < best
                best = min(best, score)
                since_best = 0 if improved else since_best + 1
                row["score"] = score
            history.append(row)
            payload = {
                "format_version": CHECKPOINT_VERSION, "config": config.to_dict(), "epoch": epoch,
                "step": step, "since_best": since_best,
                "model": model.state_dict(), "normalisation": norm.to_dict(), "action_names": D.ACTIONS,
                "generator_optimizer": generator_optimizer.state_dict(),
                "discriminator_optimizer": discriminator_optimizer.state_dict() if discriminator_optimizer is not None and not isinstance(discriminator_optimizer, dict) else None,
                "discriminator_optimizers": {name: opt.state_dict() for name, opt in discriminator_optimizer.items()} if isinstance(discriminator_optimizer, dict) else None,
                "history": history, "best_score": best,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "initialization_checkpoint": (str(init_from) if init_from else saved.get("initialization_checkpoint") if saved else None),
            }
            save_checkpoint(run / "last.pt", payload)
            if improved:
                save_checkpoint(run / "best.pt", payload)
            if config.save_every and epoch % config.save_every == 0:
                save_checkpoint(run / f"epoch_{epoch:03d}.pt", payload)
            results = {"config": config.to_dict(), "environment": runtime, "history": history,
                       "tracking_backend": tracker.backend,
                       "data_root": str(Path(root).resolve()), "train_subjects": subjects,
                       "test_subject_used_for_selection": False,
                       "initialization_checkpoint": payload["initialization_checkpoint"]}
            (run / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            tracker.log(epoch, {key: value for key, value in row.items() if isinstance(value, (float, int))})
            print(json.dumps(row, allow_nan=False))
            if progress_callback is not None:
                progress_callback({"type": "epoch", "metrics": row})
            if stop_requested is not None and stop_requested():
                print("Stop requested: saved the completed epoch; resume from last.pt")
                break
            if validation_loader is not None and config.patience and since_best >= config.patience:
                print(f"Early stopping: {since_best} epochs without validation improvement")
                break
    finally:
        tracker.close()
    return run / "last.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True, help="new run directory, or existing run with --resume")
    parser.add_argument("--config", help="YAML config; defaults to paper short-term settings")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume")
    start.add_argument("--init-from")
    parser.add_argument("--variant", choices=(*VARIANTS, "adv-euc"))
    parser.add_argument("--actions", nargs="+", choices=D.ACTIONS)
    for name in ("observed_length", "prediction_length", "hidden_dim", "critic_hidden_dim", "epochs",
                 "batch_size", "steps_per_epoch", "validation_windows", "seed", "threads", "num_workers",
                 "learning_rate_decay_every", "minimum_start", "patience", "save_every"):
        flags = ["--" + name.replace("_", "-")]
        if name == "hidden_dim": flags.append("--hidden")
        if name == "learning_rate_decay_every": flags.append("--lr-decay-every")
        parser.add_argument(*flags, type=int)
    for name in ("learning_rate", "lambda_adv", "gradient_clip", "critic_learning_rate", "learning_rate_decay"):
        flags = ["--" + name.replace("_", "-")]
        if name == "learning_rate": flags.append("--lr")
        if name == "critic_learning_rate": flags.append("--lr-critic")
        if name == "learning_rate_decay": flags.append("--lr-decay")
        parser.add_argument(*flags, type=float)
    parser.add_argument("--device")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--val-subject", choices=D.TRAIN_SUBJECTS)
    validation.add_argument("--final", action="store_true")
    parser.add_argument("--track", choices=("none", "tensorboard", "wandb"))
    parser.add_argument("--project")
    parser.add_argument("--optimizer", choices=("adam", "sgd"))
    parser.add_argument("--critic-optimizer", choices=("adam", "sgd"))
    parser.add_argument("--sampling-strategy", choices=("windows", "recordings"))
    parser.add_argument("--critic-architecture", choices=("projected", "direct"))
    parser.add_argument("--critic-output-layout", choices=("flat", "column"))
    parser.add_argument("--separate-critic-optimizers", action="store_true", default=None)
    parser.add_argument("--select-on", choices=("angle_error", "val_loss"))
    parser.add_argument("--adversarial-objective", "--adv-form", choices=("minimax", "non_saturating", "non-saturating"))
    parser.add_argument("--geodesic-reduction", choices=("sum", "mean"))
    args = parser.parse_args(argv)
    if args.resume and args.config:
        parser.error("resume loads its saved configuration; do not supply --config")
    if args.resume:
        values = load_checkpoint(args.resume)["config"]
    elif args.config:
        values = Config.from_yaml(args.config).to_dict()
    elif args.init_from:
        values = load_checkpoint(args.init_from)["config"]
    else:
        values = Config().to_dict()
    for field in fields(Config):
        value = getattr(args, field.name, None)
        if value is not None:
            values[field.name] = value
    if args.final:
        values["val_subject"] = None
    checkpoint = run_training(Config(**values), args.root, args.out, args.resume, args.init_from,
                              command_args=list(sys.argv[1:] if argv is None else argv))
    print(f"Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
