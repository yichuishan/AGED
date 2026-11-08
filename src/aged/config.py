"""Serializable experiment settings; paper defaults and explicit interpretation switches."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

import yaml

from .data import ACTIONS, TRAIN_SUBJECTS

VARIANTS = ("full", "geodesic", "euclidean", "plain", "fidelity", "continuity")


@dataclass
class Config:
    variant: str = "full"
    observed_length: int = 50
    prediction_length: int = 10
    hidden_dim: int = 1024
    critic_hidden_dim: int = 1024
    condition_decoder: bool = True
    share_recurrent_weights: bool = False
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.005
    lambda_adv: float = 0.6
    gradient_clip: float = 5.0
    steps_per_epoch: int = 2000
    validation_windows: int = 256
    val_subject: str | None = None
    actions: list[str] = field(default_factory=lambda: list(ACTIONS))
    seed: int = 0
    device: str = "auto"
    threads: int = 1
    num_workers: int = 0
    track: str = "none"
    adversarial_objective: str = "minimax"
    geodesic_reduction: str = "sum"

    def __post_init__(self):
        for name in ("observed_length", "prediction_length", "hidden_dim", "critic_hidden_dim",
                     "epochs", "batch_size", "steps_per_epoch", "validation_windows", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0 or isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("seed and num_workers must be nonnegative integers")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.val_subject is not None and self.val_subject not in TRAIN_SUBJECTS:
            raise ValueError("validation must use a training subject, never S5")
        if not self.actions or len(set(self.actions)) != len(self.actions) or set(self.actions) - set(ACTIONS):
            raise ValueError("actions must be a nonempty unique subset of Human3.6M actions")
        if any(not math.isfinite(value) for value in (self.learning_rate, self.lambda_adv, self.gradient_clip)) or self.learning_rate <= 0 or self.lambda_adv < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid learning rate, adversarial weight or gradient clipping")
        for name in ("condition_decoder", "share_recurrent_weights"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.adversarial_objective not in ("minimax", "non_saturating"):
            raise ValueError("unsupported adversarial objective")
        if self.geodesic_reduction not in ("sum", "mean"):
            raise ValueError("geodesic_reduction must be sum or mean")
        if self.track not in ("none", "tensorboard"):
            raise ValueError("track must be none or tensorboard")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides):
        with Path(path).open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            raise ValueError("config must contain a YAML mapping")
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)
