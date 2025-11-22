# ddfs/ddfs/data_collection/__init__.py

"""
Data Collection Module for Phase 2: Offline Data Collection

This module handles:
- Collecting trajectories from the plant with excitation signals
- Segmenting trajectories into time windows
- Building Hankel matrices from segmented data
"""

from ddfs.data_collection.collector import (
    DataCollector,
    ExcitationSignalGenerator,
    Trajectory,
)
from ddfs.data_collection.hankel import (
    HankelMatrixBuilder,
    SegmentHankelMatrices,
)
from ddfs.data_collection.segmenter import (
    SegmentedData,
    TrajectorySegmenter,
)

__all__ = [
    "DataCollector",
    "ExcitationSignalGenerator",
    "HankelMatrixBuilder",
    "SegmentHankelMatrices",
    "SegmentedData",
    "Trajectory",
    "TrajectorySegmenter",
]
