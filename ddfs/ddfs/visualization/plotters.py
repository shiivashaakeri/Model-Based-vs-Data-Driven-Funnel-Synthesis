"""Base plotting classes for trajectory visualization.

This module provides the TrajectoryPlotter base class for visualizing
trajectories, states, and inputs. System-specific plotters inherit
from this base class.
"""


class TrajectoryPlotter:
    """TrajectoryPlotter base class for trajectory visualization.
    
    This base class defines the interface for plotting trajectories,
    including methods for visualizing states, inputs, and trajectories
    in 2D or 3D space. System-specific plotters inherit from this class.
    """
    
    def __init__(self):
        """Initialize the trajectory plotter."""
        pass

