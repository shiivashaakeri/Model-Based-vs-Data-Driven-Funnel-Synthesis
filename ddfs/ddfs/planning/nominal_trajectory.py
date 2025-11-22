"""Nominal trajectory data structure.

This module provides the NominalTrajectory dataclass for representing
planned trajectories, including states, inputs, and time information.
"""

from dataclasses import dataclass


@dataclass
class NominalTrajectory:
    """NominalTrajectory dataclass for representing planned trajectories.
    
    This dataclass stores the nominal trajectory information including
    state sequences, input sequences, and corresponding time steps.
    Used as the output of trajectory planning algorithms.
    """
    pass

