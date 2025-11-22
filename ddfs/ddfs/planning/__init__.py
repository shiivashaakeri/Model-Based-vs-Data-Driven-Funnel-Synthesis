# ddfs/ddfs/planning/__init__.py

"""
Trajectory planning package for Phase 1.

This package provides trajectory planning functionality using
SCvx (Sequential Convex Programming) to compute nominal trajectories
that avoid obstacles while satisfying system constraints.

Key Components
--------------
NominalTrajectory : Dataclass for storing planned trajectories
    Contains state trajectory, input trajectory, and time information

SCvxPlanner : Sequential convex trajectory planner (coming soon)
    Computes collision-free trajectories using convex optimization

Usage
-----
Basic usage:
    >>> from ddfs.planning import NominalTrajectory
    >>> import numpy as np
    >>>
    >>> # Create trajectory
    >>> x_nom = np.random.randn(11, 3)  # 11 states, 3D
    >>> u_nom = np.random.randn(10, 2)  # 10 inputs, 2D
    >>> traj = NominalTrajectory(x_nom, u_nom, N=10, dt=0.1)
    >>>
    >>> # Access properties
    >>> print(traj.state_dim, traj.input_dim)
    3 2
    >>> print(traj.tf)
    1.0
    >>>
    >>> # Save/load
    >>> traj.save('trajectory.pkl')
    >>> loaded = NominalTrajectory.load('trajectory.pkl')

With config:
    >>> from ddfs.core import DDFSConfig
    >>> from ddfs.planning import NominalTrajectory
    >>>
    >>> config = DDFSConfig('config/ddfs_config.yaml')
    >>> params = config.get_planning_params()
    >>> print(params['N'], params['dt'])
"""

from ddfs.planning.nominal_trajectory import NominalTrajectory
from ddfs.planning.scvx import SCvxPlanner

__all__ = [
    "NominalTrajectory",
    "SCvxPlanner",
]

__version__ = "0.1.0"
