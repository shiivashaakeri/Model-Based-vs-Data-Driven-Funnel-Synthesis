"""
Data collection and management module for DDFS.

This module provides tools for:
- Offline trajectory collection from the plant
- Trajectory segmentation into time windows
- Hankel matrix construction
- Informativity (persistence of excitation) checking
"""

from .collector import OfflineDataCollector
from .hankel import HankelMatrixBuilder
from .informativity import InformativityChecker
from .segmenter import TrajectorySegmenter

__all__ = [
    "HankelMatrixBuilder",
    "InformativityChecker",
    "OfflineDataCollector",
    "TrajectorySegmenter",
]
