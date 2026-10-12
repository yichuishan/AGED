"""Small tensor checks shared by the encoder and decoder."""

from numbers import Integral

import torch
from torch import Tensor
from torch.nn import functional as F


def positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def check_motion(motion: Tensor, motion_dim: int, ndim: int) -> None:
    if motion.ndim != ndim or motion.shape[-1] != motion_dim:
        layout = "[B, T, K]" if ndim == 3 else "[B, K]"
        raise ValueError(f"motion must have shape {layout} with K={motion_dim}")
    if any(size == 0 for size in motion.shape):
        raise ValueError("motion dimensions must be nonempty")
    if not motion.is_floating_point():
        raise TypeError("motion must be a floating-point tensor")


def condition_motion(motion: Tensor, action: Tensor | None, num_actions: int | None) -> Tensor:
    """Append one-hot labels to [B,K] or [B,T,K], preserving motion dtype/device."""
    if num_actions is None:
        if action is not None:
            raise ValueError("action was supplied to an unconditioned model")
        return motion
    if action is None or action.ndim != 1 or action.shape[0] != motion.shape[0]:
        raise ValueError("conditioned models require action IDs of shape [B]")
    if action.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("action IDs must have integer dtype")
    if torch.any(action < 0) or torch.any(action >= num_actions):
        raise ValueError(f"action IDs must be in [0, {num_actions})")
    labels = F.one_hot(action.to(device=motion.device, dtype=torch.long), num_actions)
    labels = labels.to(dtype=motion.dtype)  # [B, A]
    if motion.ndim == 3:
        labels = labels[:, None, :].expand(-1, motion.shape[1], -1)  # [B, T, A]
    return torch.cat((motion, labels), dim=-1)


def check_recurrent_config(hidden_dim: int, num_layers: int, dropout: float) -> None:
    positive_int("hidden_dim", hidden_dim)
    positive_int("num_layers", num_layers)
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if num_layers == 1 and dropout != 0:
        raise ValueError("GRU dropout acts between layers; use 0 with a single layer")
