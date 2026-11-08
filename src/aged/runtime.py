"""Device selection, model assembly and portable checkpoints shared by both entry points."""

import os
from pathlib import Path
import random
import shlex
import subprocess

import numpy as np
import torch

from .config import Config, RESUME_MUTABLE
from .critics import Critic
from .data import ACTIONS, Normalisation
from .models import AGEDModel, MotionPredictor

CHECKPOINT_VERSION = 1


def training_arguments(config, root, out, config_path=None, resume=None, init_from=None):
    """Replayable CLI arguments shared by programmatic and web training."""
    args = ["--root", str(Path(root).resolve()), "--out", str(Path(out).resolve())]
    if resume:
        args += ["--resume", str(Path(resume).resolve())]
        for name in RESUME_MUTABLE:
            args += ["--" + name.replace("_", "-"), str(getattr(config, name))]
    else:
        args += ["--config", str(Path(config_path).resolve())]
        if init_from:
            args += ["--init-from", str(Path(init_from).resolve())]
    return args


def format_command(arguments):
    return subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch runtime has no usable CUDA device")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested, but it is not available")
    return device


def build_model(config: Config, norm: Normalisation):
    predictor = MotionPredictor(
        motion_dim=norm.n_dims, hidden_dim=config.hidden_dim, num_actions=len(ACTIONS),
        prediction_length=config.prediction_length, condition_decoder=config.condition_decoder,
        share_recurrent_weights=config.share_recurrent_weights,
    )
    options = {"architecture": config.critic_architecture, "output_layout": config.critic_output_layout}
    fidelity = Critic(norm.n_dims, config.critic_hidden_dim, **options) if config.variant in ("full", "euclidean", "fidelity") else None
    continuity = Critic(norm.n_dims, config.critic_hidden_dim, **options) if config.variant in ("full", "euclidean", "continuity") else None
    return AGEDModel(predictor, fidelity, continuity)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint; legacy standalone predictor weights are not full AGED checkpoints")
    if checkpoint.get("action_names") != ACTIONS:
        raise ValueError("checkpoint action mapping differs from this package")
    # Optional fields extend v1 without changing existing architecture or defaults.
    checkpoint["config"] = Config(**checkpoint["config"]).to_dict()
    checkpoint.setdefault("step", checkpoint["epoch"] * checkpoint["config"]["steps_per_epoch"])
    checkpoint.setdefault("since_best", 0)
    return checkpoint


def load_model(path, device="cpu"):
    checkpoint = load_checkpoint(path)
    config = Config(**checkpoint["config"])
    norm = Normalisation.from_dict(checkpoint["normalisation"])
    model = build_model(config, norm).to(pick_device(device))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, norm, config, checkpoint
