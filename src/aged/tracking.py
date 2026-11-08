"""Optional local TensorBoard logging and reproducible runtime metadata."""

import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
import yaml


def environment():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        if dirty:
            commit += "-dirty"
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "PyYAML": yaml.__version__,
        "cuda_available": torch.cuda.is_available(),
        "commit": commit,
    }


class Tracker:
    """JSON artifacts are always written by the trainer; TensorBoard is optional."""

    def __init__(self, backend, run_dir, config, project="aged"):
        self.writer = None
        self.run = None
        self.backend = "none"
        if backend == "none":
            return
        if backend == "wandb":
            try:
                import wandb
                self.run = wandb.init(project=project, dir=str(run_dir), config=config,
                                      name=Path(run_dir).name)
                self.backend = "wandb" if self.run is not None else "none"
            except Exception as error:
                print(f"wandb unavailable ({type(error).__name__}); continuing with JSON results.")
            return
        if backend != "tensorboard":
            raise ValueError("supported tracking backends are none, tensorboard and wandb")
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("TensorBoard is not installed; continuing with JSON results.")
            return
        self.writer = SummaryWriter(log_dir=str(Path(run_dir) / "tb"))
        self.backend = "tensorboard"
        self.writer.add_text("config", json.dumps(config, indent=2))

    def log(self, step, metrics, prefix=""):
        metrics = {prefix + key: value for key, value in metrics.items() if isinstance(value, (float, int))}
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        if self.run is not None:
            self.run.log(metrics, step=step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.run is not None:
            self.run.finish()
