"""SO(3) geometry from the team's loss, with stable zero-angle derivatives and pose restoration."""

import math

import torch
from torch import Tensor, nn

from .data import Normalisation


def expmap_to_rotation_matrix(vector: Tensor) -> Tensor:
    """Rodrigues formula, valid at zero without dividing by a clamped angle."""
    if vector.shape[-1] != 3:
        raise ValueError("rotation vectors must end in three coordinates")
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(vector, dim=-1)
    a = torch.sinc(angle / math.pi)
    b = 0.5 * torch.sinc(angle / (2 * math.pi)).square()
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return eye + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def geodesic_distance(predicted: Tensor, target: Tensor) -> Tensor:
    """Rotation-vector tensors [...,3] -> shortest angular distance [...]."""
    if predicted.shape != target.shape:
        raise ValueError("rotation shapes must match")
    relative = expmap_to_rotation_matrix(target) @ expmap_to_rotation_matrix(predicted).transpose(-1, -2)
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    sine_vector = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                               relative[..., 0, 2] - relative[..., 2, 0],
                               relative[..., 1, 0] - relative[..., 0, 1]), dim=-1) / 2
    sine = torch.linalg.vector_norm(sine_vector, dim=-1)
    return torch.atan2(sine, cosine)


class GeodesicLoss(nn.Module):
    """Undo normalization and channel selection before computing the paper's loss.

    sum: sum time/joints, mean batch (Eq. 4); mean: mean all elements, an ablation.
    Root translation/rotation are excluded; reduced channels need not form triplets.
    """

    def __init__(self, norm: Normalisation, reduction: str = "sum"):
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError("unknown reduction")
        self.reduction = reduction
        self.register_buffer("mean", torch.tensor(norm.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(norm.std, dtype=torch.float32))
        self.register_buffer("dims", torch.tensor(norm.used_dims))

    def restore(self, motion: Tensor) -> Tensor:
        mean, std = self.mean.to(motion.dtype), self.std.to(motion.dtype)
        full = mean.expand(*motion.shape[:-1], len(mean))
        index = self.dims.expand(*motion.shape[:-1], len(self.dims))
        values = motion * std[self.dims] + mean[self.dims]
        return full.scatter(-1, index, values)

    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != len(self.dims):
            raise ValueError("expected matching [B,T,K] motion tensors")
        pred_raw, target_raw = self.restore(predicted)[..., 6:], self.restore(target)[..., 6:]
        distance = geodesic_distance(pred_raw.reshape(*pred_raw.shape[:-1], -1, 3),
                                     target_raw.reshape(*target_raw.shape[:-1], -1, 3))
        if self.reduction == "mean":
            return distance.mean()
        return distance.sum(dim=(1, 2)).mean()
