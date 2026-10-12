"""Recurrent encoder for already-preprocessed motion tensors."""

from torch import Tensor, nn

from ._inputs import check_motion, check_recurrent_config, condition_motion, positive_int


class MotionEncoder(nn.Module):
    """Linear input mapping followed by a unidirectional GRU (paper Fig. 2b)."""

    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int = 1024,
        num_actions: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        positive_int("motion_dim", motion_dim)
        check_recurrent_config(hidden_dim, num_layers, dropout)
        if num_actions is not None:
            positive_int("num_actions", num_actions)
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.input_projection = nn.Linear(motion_dim + (num_actions or 0), hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, observed_motion: Tensor, action: Tensor | None = None) -> Tensor:
        """Encode all observed frames; return hidden state [L, B, H]."""
        check_motion(observed_motion, self.motion_dim, ndim=3)
        conditioned = condition_motion(observed_motion, action, self.num_actions)
        embedded = self.input_projection(conditioned)  # [B, T_obs, H]
        initial_hidden = embedded.new_zeros(self.num_layers, embedded.shape[0], self.hidden_dim)
        _, hidden = self.gru(embedded, initial_hidden)
        return hidden
