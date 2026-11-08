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

    def __init__(self, backend, run_dir, config):
        self.writer = None
        if backend == "none":
            return
        if backend != "tensorboard":
            raise ValueError("supported tracking backends are none and tensorboard")
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("TensorBoard is not installed; continuing with JSON results.")
            return
        self.writer = SummaryWriter(log_dir=str(Path(run_dir) / "tb"))
        self.writer.add_text("config", json.dumps(config, indent=2))

    def log(self, step, metrics):
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
