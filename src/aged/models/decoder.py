"""Readable free-running, residual motion decoder."""

import torch
from torch import Tensor, nn

from ._inputs import check_motion, check_recurrent_config, condition_motion, positive_int


class MotionDecoder(nn.Module):
    """Predict future poses, feeding each prediction into the next GRU step."""

    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int = 1024,
        num_actions: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual_prediction: bool = True,
        prediction_length: int = 10,
    ) -> None:
        super().__init__()
        positive_int("motion_dim", motion_dim)
        positive_int("prediction_length", prediction_length)
        check_recurrent_config(hidden_dim, num_layers, dropout)
        if num_actions is not None:
            positive_int("num_actions", num_actions)
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.residual_prediction = residual_prediction
        self.prediction_length = prediction_length
        self.input_projection = nn.Linear(motion_dim + (num_actions or 0), hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.output_projection = nn.Linear(hidden_dim, motion_dim)

    def forward(
        self,
        seed_motion: Tensor,
        hidden: Tensor,
        action: Tensor | None = None,
        pred_len: int | None = None,
    ) -> Tensor:
        """Seed [B,K], encoder hidden [L,B,H] -> future motion [B,T_pred,K]."""
        check_motion(seed_motion, self.motion_dim, ndim=2)
        horizon = self.prediction_length if pred_len is None else pred_len
        positive_int("pred_len", horizon)
        expected_hidden = (self.num_layers, seed_motion.shape[0], self.hidden_dim)
        if tuple(hidden.shape) != expected_hidden:
            raise ValueError(f"hidden must have shape {expected_hidden}")
        if hidden.device != seed_motion.device or hidden.dtype != seed_motion.dtype:
            raise ValueError("hidden and seed must have the same device and dtype")

        previous_motion = seed_motion  # x_T; it is NOT part of the returned future.
        predictions = []
        for _ in range(horizon):
            # Default decoder conditioning is a documented assumption (Sec. 3.1).
            conditioned = condition_motion(previous_motion, action, self.num_actions)
            embedded = self.input_projection(conditioned).unsqueeze(1)  # [B, 1, H]
            recurrent_output, hidden = self.gru(embedded, hidden)
            delta_or_pose = self.output_projection(recurrent_output[:, 0, :])  # [B, K]
            if self.residual_prediction:
                next_motion = previous_motion + delta_or_pose
            else:
                next_motion = delta_or_pose  # Optional ablation, not default AGED.
            predictions.append(next_motion)
            previous_motion = next_motion  # No detach, teacher forcing, or future GT.
        return torch.stack(predictions, dim=1)
