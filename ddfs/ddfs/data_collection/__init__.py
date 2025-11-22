# ddfs/ddfs/data_collection/__init__.py

"""
Data collection package for Phase 2.

This package handles offline data collection from the plant for
uncertainty quantification and data-driven control synthesis.

Phase 2 Process
---------------
1. Collect M trajectories from plant with excitation signals
2. Segment trajectories into overlapping/non-overlapping windows
3. Build Hankel matrices from segmented data
4. Verify data informativity (persistence of excitation)

Key Components
--------------
DataCollector : Manages trajectory collection from plant
    Applies open-loop control with excitation: u(k) = u_nom(k) + ε(k)

Trajectory : Single trajectory container
    Stores states, inputs, and deviations from nominal

ExcitationSignalGenerator : Generates excitation signals
    Types: gaussian, chirp, multisine, prbs

TrajectorySegmenter : Segments trajectories into time windows
    Creates overlapping or non-overlapping segments

SegmentedData : Container for segmented trajectories
    Organizes trajectories by segment

HankelMatrixBuilder : Builds Hankel matrices from segments
    Creates H, H+, Ξ matrices for data-driven control

SegmentHankelMatrices : Hankel matrices for one segment
    Checks informativity and condition number

Usage Examples
--------------
Basic data collection:
    >>> from ddfs.core import DDFSConfig
    >>> from ddfs.models import UnicycleTwin, UnicyclePlant
    >>> from ddfs.data_collection import DataCollector
    >>>
    >>> # Setup
    >>> config = DDFSConfig('config/ddfs_config.yaml')
    >>> twin = UnicycleTwin(dt=0.131)
    >>> plant = UnicyclePlant(twin, velocity_scale=0.95)
    >>>
    >>> # Collect (assuming we have nominal trajectory)
    >>> collector = DataCollector(plant, nominal, config={'M': 50})
    >>> trajectories = collector.collect_trials()

Segmentation:
    >>> from ddfs.data_collection import TrajectorySegmenter
    >>>
    >>> segmenter = TrajectorySegmenter(T=100, L=60)
    >>> segmented_data = segmenter.segment(trajectories)

Building Hankel matrices:
    >>> from ddfs.data_collection import HankelMatrixBuilder
    >>>
    >>> builder = HankelMatrixBuilder(verbose=True)
    >>> all_matrices = builder.build_all_segments(segmented_data)
    >>>
    >>> # Check informativity
    >>> for matrices in all_matrices:
    ...     is_informative, rank, required = matrices.check_informativity()
    ...     print(f"Segment {matrices.segment_idx}: informative={is_informative}")

Notes
-----
- Obstacles should be REMOVED during data collection for safety
- Open-loop control is used (no feedback)
- Excitation signals improve data richness
- Informativity condition: rank([H; Xi]) = n + m
"""

# Collector
from ddfs.data_collection.collector import (
    DataCollector,
    ExcitationSignalGenerator,
    Trajectory,
)

# Hankel matrices
from ddfs.data_collection.hankel import (
    HankelMatrixBuilder,
    SegmentHankelMatrices,
)

# Segmentation
from ddfs.data_collection.segmenter import (
    SegmentedData,
    TrajectorySegmenter,
)

__all__ = [
    # Collector
    "DataCollector",
    "ExcitationSignalGenerator",
    # Hankel matrices
    "HankelMatrixBuilder",
    "SegmentHankelMatrices",
    "SegmentedData",
    "Trajectory",
    # Segmentation
    "TrajectorySegmenter",
]

__version__ = "0.1.0"
