"""Reconciled continuity/fidelity critic with the deployed interface preserved.

The team's direct GRU is selectable; projected/flat stays the legacy default.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models._inputs import check_motion, positive_int


class Critic(nn.Module):
    """One architecture, instantiated independently for fidelity and continuity."""

    def __init__(self, motion_dim: int | None = None, hidden_dim: int | None = None,
                 *, n_dims: int | None = None, hidden: int | None = None,
                 architecture: str = "projected", output_layout: str = "flat"):
        super().__init__()
        if motion_dim is not None and n_dims is not None and motion_dim != n_dims:
            raise ValueError("motion_dim and n_dims disagree")
        if hidden_dim is not None and hidden is not None and hidden_dim != hidden:
            raise ValueError("hidden_dim and hidden disagree")
        motion_dim = motion_dim if motion_dim is not None else n_dims
        hidden_dim = hidden_dim if hidden_dim is not None else hidden if hidden is not None else 1024
        positive_int("motion_dim", motion_dim)
        positive_int("hidden_dim", hidden_dim)
        if architecture not in ("projected", "direct") or output_layout not in ("flat", "column"):
            raise ValueError("invalid critic architecture or output layout")
        self.architecture, self.output_layout = architecture, output_layout
        self.motion_dim = motion_dim
        self.n_dims = motion_dim
        self.input_projection = nn.Linear(motion_dim, hidden_dim) if architecture == "projected" else nn.Identity()
        self.gru = nn.GRU(hidden_dim if architecture == "projected" else motion_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, 1)

    def logits(self, sequence: Tensor) -> Tensor:
        """[B,T,K] -> [B] (default) or [B,1]; stable log-probability scores."""
        check_motion(sequence, self.motion_dim, 3)
        _, hidden = self.gru(self.input_projection(sequence))
        result = self.output_projection(hidden[-1])
        return result.squeeze(-1) if self.output_layout == "flat" else result

    def forward(self, sequence: Tensor) -> Tensor:
        """Return P(real), as specified by the paper."""
        return self.logits(sequence).sigmoid()

    @property
    def classifier(self):
        """Team head alias, without registering duplicate state-dict keys."""
        return self.output_projection

    @property
    def fc(self):
        return self.output_projection

    def load_team_state_dict(self, state):
        """Explicitly import one direct-GRU component's fc/classifier checkpoint."""
        if self.architecture != "direct":
            raise ValueError("team direct-GRU weights require architecture='direct'")
        converted = {}
        for name, value in state.items():
            if name.startswith("fc.") or name.startswith("classifier."):
                name = "output_projection." + name.split(".", 1)[1]
            if name in converted:
                raise ValueError("ambiguous classifier keys in checkpoint")
            converted[name] = value
        return self.load_state_dict(converted, strict=True)


def fidelity_input(predicted: Tensor) -> Tensor:
    return predicted


def continuity_input(observed: Tensor, predicted: Tensor) -> Tensor:
    """Preserve the team's two-input contract; concatenate time with one seed."""
    if observed.ndim != 3 or predicted.ndim != 3 or observed.shape[0] != predicted.shape[0] or observed.shape[-1] != predicted.shape[-1]:
        raise ValueError("observed and predicted must share batch and feature dimensions")
    return torch.cat((observed, predicted), dim=1)


def _critic_terms(critic, real: Tensor, fake: Tensor):
    # Preserve deployed forward/backward ordering for old checkpoints.
    fake_logits, real_logits = critic.logits(fake), critic.logits(real)
    accuracy = ((real_logits > 0).float().mean() + (fake_logits < 0).float().mean()) / 2
    return F.softplus(fake_logits).mean(), F.softplus(-real_logits).mean(), float(accuracy.detach())


def critic_loss(critic, real: Tensor, fake: Tensor):
    """Team negative Eq. 6/7 helper; caller detaches fake before D updates."""
    fake_loss, real_loss, accuracy = _critic_terms(critic, real, fake)
    return fake_loss + real_loss, accuracy


def generator_loss(critic, fake: Tensor) -> Tensor:
    """Paper minimax; identical semantics to the fidelity reconciliation branch."""
    return -F.softplus(critic.logits(fake)).mean()


def generator_loss_non_saturating(critic, fake: Tensor) -> Tensor:
    return F.softplus(-critic.logits(fake)).mean()
