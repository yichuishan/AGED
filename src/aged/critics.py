"""Paper Fig. 2c: linear input mapping, single GRU, linear head and sigmoid."""

from torch import Tensor, nn

from .models._inputs import check_motion, positive_int


class Critic(nn.Module):
    """One architecture, instantiated independently for fidelity and continuity."""

    def __init__(self, motion_dim: int, hidden_dim: int = 1024):
        super().__init__()
        positive_int("motion_dim", motion_dim)
        positive_int("hidden_dim", hidden_dim)
        self.motion_dim = motion_dim
        self.input_projection = nn.Linear(motion_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, 1)

    def logits(self, sequence: Tensor) -> Tensor:
        """[B,T,K] -> [B]; logits used for stable log-probability objectives."""
        check_motion(sequence, self.motion_dim, 3)
        _, hidden = self.gru(self.input_projection(sequence))
        return self.output_projection(hidden[-1]).squeeze(-1)

    def forward(self, sequence: Tensor) -> Tensor:
        """Return P(real), as specified by the paper."""
        return self.logits(sequence).sigmoid()
