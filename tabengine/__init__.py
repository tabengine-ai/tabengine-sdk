"""
tabengine: Zero-shot predictive machine learning platform powered by Tabular Foundation Models (TabFM).
"""

from .classifier import ZeroShotClassifier
from .regressor import ZeroShotRegressor
from .indexer import TabularIndex

__version__ = "0.1.0"
__all__ = ["ZeroShotClassifier", "ZeroShotRegressor", "TabularIndex"]
