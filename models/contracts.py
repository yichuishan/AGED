"""Typing-only contracts; these do not implement losses or discriminators."""

from typing import Protocol

from torch import Tensor


class GeodesicLossContract(Protocol):
    def __call__(self, predicted: Tensor, ground_truth: Tensor) -> Tensor:
        """[B,H,K], [B,H,K] -> scalar; own differentiable pose restoration."""
        ...


class FidelityDiscriminatorContract(Protocol):
    def __call__(self, future_motion: Tensor) -> Tensor:
        """Future-only [B,H,K] -> real probabilities [B,1]."""
        ...


class ContinuityDiscriminatorContract(Protocol):
    def __call__(self, observed: Tensor, future: Tensor) -> Tensor:
        """Internally concatenate [B,T,K], [B,H,K] along time -> [B,1]."""
        ...
