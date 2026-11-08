"""Main predictor interface for teammates and training code."""

from torch import Tensor, nn

from .decoder import MotionDecoder
from .encoder import MotionEncoder


class MotionPredictor(nn.Module):
    """GRU encoder + autoregressive decoder, independent of losses and data loading."""

    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int = 1024,
        num_actions: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual_prediction: bool = True,
        prediction_length: int = 10,
        condition_decoder: bool = True,
        share_recurrent_weights: bool = False,
    ) -> None:
        super().__init__()
        if share_recurrent_weights and num_actions is not None and not condition_decoder:
            raise ValueError("sharing requires matching encoder and decoder input dimensions")
        self.encoder = MotionEncoder(motion_dim, hidden_dim, num_actions, num_layers, dropout)
        self.decoder = MotionDecoder(
            motion_dim, hidden_dim, num_actions if condition_decoder else None,
            num_layers, dropout, residual_prediction, prediction_length,
        )
        self.condition_decoder = condition_decoder
        self.share_recurrent_weights = share_recurrent_weights
        if share_recurrent_weights:
            # Tie the entire recurrent input path; output projection belongs to decoder.
            self.decoder.input_projection = self.encoder.input_projection
            self.decoder.gru = self.encoder.gru

    def forward(
        self,
        observed_motion: Tensor,
        action: Tensor | None = None,
        pred_len: int | None = None,
    ) -> Tensor:
        """[B,T_obs,K] + optional action IDs [B] -> [B,T_pred,K]."""
        hidden = self.encoder(observed_motion, action)
        seed = observed_motion[:, -1, :]  # Last observed frame x_T.
        decoder_action = action if self.condition_decoder else None
        return self.decoder(seed, hidden, decoder_action, pred_len)
