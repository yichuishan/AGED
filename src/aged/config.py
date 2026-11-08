"""Serializable experiment settings; paper defaults and explicit interpretation switches."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

import yaml

from .data import ACTIONS, TRAIN_SUBJECTS

VARIANTS = ("full", "geodesic", "euclidean", "plain", "fidelity", "continuity")
RESUME_MUTABLE = ("epochs", "device", "threads", "num_workers", "track", "project", "patience", "save_every")


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
    # Team capabilities are opt-in; old configurations retain their exact defaults.
    optimizer: str = "adam"
    critic_optimizer: str | None = None
    critic_learning_rate: float | None = None
    learning_rate_decay: float = 1.0
    learning_rate_decay_every: int = 10000
    sampling_strategy: str = "windows"
    minimum_start: int = 0
    critic_architecture: str = "projected"
    critic_output_layout: str = "flat"
    separate_critic_optimizers: bool = False
    patience: int = 0
    save_every: int = 0
    select_on: str = "angle_error"
    project: str = "aged"

    def __post_init__(self):
        for name in ("observed_length", "prediction_length", "hidden_dim", "critic_hidden_dim",
                     "epochs", "batch_size", "steps_per_epoch", "validation_windows", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0 or isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("seed and num_workers must be nonnegative integers")
        if self.variant == "adv-euc":
            self.variant = "euclidean"
        self.adversarial_objective = self.adversarial_objective.replace("-", "_")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.val_subject is not None and self.val_subject not in TRAIN_SUBJECTS:
            raise ValueError("validation must use a training subject, never S5")
        if not self.actions or len(set(self.actions)) != len(self.actions) or set(self.actions) - set(ACTIONS):
            raise ValueError("actions must be a nonempty unique subset of Human3.6M actions")
        if any(not math.isfinite(value) for value in (self.learning_rate, self.lambda_adv, self.gradient_clip)) or self.learning_rate < 0 or self.lambda_adv < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid learning rate, adversarial weight or gradient clipping")
        for name in ("condition_decoder", "share_recurrent_weights", "separate_critic_optimizers"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.adversarial_objective not in ("minimax", "non_saturating"):
            raise ValueError("unsupported adversarial objective")
        if self.geodesic_reduction not in ("sum", "mean"):
            raise ValueError("geodesic_reduction must be sum or mean")
        if self.track not in ("none", "tensorboard", "wandb"):
            raise ValueError("track must be none, tensorboard or wandb")
        if self.optimizer not in ("adam", "sgd") or self.critic_optimizer not in (None, "adam", "sgd"):
            raise ValueError("optimizers must be adam or sgd")
        if self.critic_learning_rate is not None and (not math.isfinite(self.critic_learning_rate) or self.critic_learning_rate < 0):
            raise ValueError("critic_learning_rate must be finite and nonnegative")
        if not math.isfinite(self.learning_rate_decay) or not 0 < self.learning_rate_decay <= 1:
            raise ValueError("learning_rate_decay must be in (0,1]")
        for name in ("minimum_start", "patience", "save_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(self.learning_rate_decay_every, bool) or not isinstance(self.learning_rate_decay_every, int) or self.learning_rate_decay_every <= 0:
            raise ValueError("learning_rate_decay_every must be a positive integer")
        if self.sampling_strategy not in ("windows", "recordings"):
            raise ValueError("sampling_strategy must be windows or recordings")
        if self.critic_architecture not in ("projected", "direct") or self.critic_output_layout not in ("flat", "column"):
            raise ValueError("invalid critic architecture or output layout")
        if self.select_on not in ("angle_error", "val_loss"):
            raise ValueError("select_on must be angle_error or val_loss")

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
