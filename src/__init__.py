"""ViralPredict - ML model for predicting social media viral potential."""

from src.model import ViralPredictor
from src.predict import predict_virality

__all__ = ["ViralPredictor", "predict_virality"]
__version__ = "0.1.0"
