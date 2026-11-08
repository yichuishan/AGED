"""Human3.6M exponential-map text loading, train-only statistics and action labels.

Based on the supplied team's data pipeline, with root exclusion, complete pose
restoration, streaming statistics, configurable windows and epoch-varying samples.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ACTIONS = ["directions", "discussion", "eating", "greeting", "phoning", "posing",
           "purchases", "sitting", "sittingdown", "smoking", "takingphoto", "waiting",
           "walking", "walkingdog", "walkingtogether"]
TRAIN_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
TEST_SUBJECT = "S5"
RAW_DIM = 99
REPORT_MS = {80: 1, 160: 3, 320: 7, 400: 9, 560: 13, 1000: 24}


def load_raw(root: str | Path, subject: str, action: str, take: int) -> np.ndarray:
    """Read 50 Hz [T,99] and downsample exactly once to 25 Hz."""
    path = Path(root) / subject / f"{action}_{take}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; see data/README.md for the required export layout")
    frames = np.loadtxt(path, delimiter=",", ndmin=2)
    if frames.shape[1] != RAW_DIM or not np.isfinite(frames).all():
        raise ValueError(f"{path}: expected finite [T,{RAW_DIM}] exponential-map data")
    return frames[::2]


@dataclass
class Normalisation:
    mean: np.ndarray
    std: np.ndarray
    used_dims: np.ndarray

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.std = np.asarray(self.std, dtype=np.float64)
        self.used_dims = np.asarray(self.used_dims, dtype=np.int64)
        if self.mean.shape != (RAW_DIM,) or self.std.shape != (RAW_DIM,):
            raise ValueError("normalisation must describe the complete 99-channel export")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("normalisation statistics must be finite with positive scales")
        if self.used_dims.ndim != 1 or not len(self.used_dims):
            raise ValueError("no varying body-rotation channels found")
        if len(np.unique(self.used_dims)) != len(self.used_dims) or np.any(self.used_dims < 6) or np.any(self.used_dims >= RAW_DIM):
            raise ValueError("used_dims must uniquely select body rotations (channels 6:99)")

    @property
    def n_dims(self):
        return len(self.used_dims)

    def apply(self, frames: np.ndarray) -> np.ndarray:
        return ((frames[..., self.used_dims] - self.mean[self.used_dims]) / self.std[self.used_dims]).astype(np.float32)

    def restore(self, motion: np.ndarray) -> np.ndarray:
        """Restore full raw poses; dropped channels take their training means."""
        if motion.shape[-1] != self.n_dims:
            raise ValueError("motion feature count differs from checkpoint normalisation")
        result = np.broadcast_to(self.mean, (*motion.shape[:-1], RAW_DIM)).copy()
        result[..., self.used_dims] = motion * self.std[self.used_dims] + self.mean[self.used_dims]
        return result

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "used_dims": self.used_dims.tolist()}

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


def fit_normalisation(root, subjects=TRAIN_SUBJECTS, actions=ACTIONS, threshold=1e-4):
    """Streaming population moments; S5 and unknown subjects are rejected."""
    if not subjects or set(subjects) - set(TRAIN_SUBJECTS):
        raise ValueError("normalisation may use training subjects only; S5 is held out")
    count, mean, m2 = 0, np.zeros(RAW_DIM), np.zeros(RAW_DIM)
    for subject in subjects:
        for action in actions:
            for take in (1, 2):
                frames = load_raw(root, subject, action, take)
                n = len(frames)
                delta = frames.mean(axis=0) - mean
                m2 += frames.var(axis=0) * n + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
    std = np.sqrt(np.maximum(m2 / count, 0))
    used = np.flatnonzero((std >= threshold) & (np.arange(RAW_DIM) >= 6))
    std[std < threshold] = 1.0
    return Normalisation(mean, std, used)


class WindowDataset(Dataset):
    """Action-labeled random windows, deterministic by (seed, epoch, index)."""

    def __init__(self, root, norm, subjects, actions=ACTIONS, observed_length=50,
                 prediction_length=10, length=32000, seed=0):
        self.norm, self.observed_length = norm, observed_length
        self.prediction_length, self.length, self.seed = prediction_length, length, seed
        self.epoch = 0
        self.window = observed_length + prediction_length
        self.sequences, self.labels = [], []
        for subject in subjects:
            for action in actions:
                for take in (1, 2):
                    raw = load_raw(root, subject, action, take)
                    if len(raw) >= self.window:
                        self.sequences.append(norm.apply(raw))
                        self.labels.append(ACTIONS.index(action))
        if not self.sequences:
            raise ValueError("no recordings are long enough for the configured window")
        self.weights = np.array([len(seq) - self.window + 1 for seq in self.sequences], dtype=float)
        self.weights /= self.weights.sum()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        selected = int(rng.choice(len(self.sequences), p=self.weights))
        sequence = self.sequences[selected]
        start = int(rng.integers(len(sequence) - self.window + 1))
        window = torch.from_numpy(sequence[start:start + self.window].copy())
        return window[:self.observed_length], window[self.observed_length:], torch.tensor(self.labels[selected])
