"""Integration scaffold; adversarial components remain optional."""

from torch import Tensor, nn

from .predictor import MotionPredictor


class AGEDModel(nn.Module):
    """Register independent components without implementing their algorithms."""

    def __init__(
        self,
        predictor: MotionPredictor,
        fidelity_discriminator: nn.Module | None = None,
        continuity_discriminator: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.fidelity_discriminator = fidelity_discriminator
        self.continuity_discriminator = continuity_discriminator

    def predict(
        self, observed_motion: Tensor, action: Tensor | None = None, pred_len: int | None = None,
    ) -> Tensor:
        """Delegate prediction without evaluating either discriminator."""
        return self.predictor(observed_motion, action, pred_len)

    def forward(
        self, observed_motion: Tensor, action: Tensor | None = None, pred_len: int | None = None,
    ) -> Tensor:
        return self.predict(observed_motion, action, pred_len)
