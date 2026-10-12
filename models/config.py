"""Single model configuration, with K supplied by preprocessing or the caller."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from ._inputs import check_recurrent_config, positive_int
from .predictor import MotionPredictor


@dataclass(frozen=True)
class ModelConfig:
    motion_dim: int
    hidden_dim: int = 1024
    num_actions: int | None = 15
    observed_length: int = 50
    prediction_length: int = 10
    num_layers: int = 1
    dropout: float = 0.0
    residual_prediction: bool = True
    condition_decoder: bool = True
    share_recurrent_weights: bool = False

    def __post_init__(self) -> None:
        for name in ("motion_dim", "observed_length", "prediction_length"):
            positive_int(name, getattr(self, name))
        check_recurrent_config(self.hidden_dim, self.num_layers, self.dropout)
        if self.num_actions is not None:
            positive_int("num_actions", self.num_actions)
        for name in ("residual_prediction", "condition_decoder", "share_recurrent_weights"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "ModelConfig":
        with Path(path).open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            raise ValueError("model YAML must contain a mapping")
        values.update(overrides)
        return cls(**values)

    def build(self) -> MotionPredictor:
        # observed_length is a data-window default, not a constraint on GRU input.
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values.pop("observed_length")
        return MotionPredictor(**values)
