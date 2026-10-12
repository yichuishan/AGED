"""Public model classes (no loss/discriminator implementations)."""

from .aged import AGEDModel
from .decoder import MotionDecoder
from .encoder import MotionEncoder
from .predictor import MotionPredictor

__all__ = ["MotionEncoder", "MotionDecoder", "MotionPredictor", "AGEDModel"]
