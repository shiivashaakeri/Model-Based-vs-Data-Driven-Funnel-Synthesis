"""Feasibility envelope computation module for DDFS Phase 4.

This module provides tools for computing Maximum Volume Inscribed Ellipsoids
(MVIE) that characterize feasibility envelopes. These ellipsoids ensure that
trajectories remain within safe regions while accounting for uncertainty.

Key Components:
    - EllipsoidSolver: Main solver for MVIE optimization
    - EllipsoidParams: Container for ellipsoid parameters (P, c)
    - FeasibilityEnvelope: Container for complete envelope with bootstrap consistency

Example:
    >>> from ddfs.feasibility import EllipsoidSolver, EllipsoidParams
    >>> from ddfs.core.config import DDFSConfig
    >>> from ddfs.core.obstacles import CircularObstacle
    >>>
    >>> config = DDFSConfig(...)
    >>> obstacles = [CircularObstacle(x=5.0, y=5.0, radius=1.0)]
    >>> solver = EllipsoidSolver(config, obstacles)
    >>>
    >>> # Define nominal and initial ellipsoids
    >>> P_0 = EllipsoidParams(
    ...     P=np.eye(3),
    ...     c=np.array([2.0, 2.0, 0.0]),
    ...     segment_index=0
    ... )
    >>> P_min_0_init = EllipsoidParams(
    ...     P=2*np.eye(3),
    ...     c=np.array([2.0, 2.0, 0.0]),
    ...     segment_index=0
    ... )
    >>>
    >>> # Solve MVIE
    >>> P_min_0 = solver.solve_mvie(
    ...     P_0=P_0,
    ...     P_min_0_init=P_min_0_init,
    ...     beta=0.01,  # From Phase 3 uncertainty quantification
    ... )
    >>>
    >>> # Compute envelope for multiple segments
    >>> envelope = solver.compute_envelope(
    ...     P_0_list=[P_0_seg0, P_0_seg1, ...],
    ...     P_min_0_init_list=[init_seg0, init_seg1, ...],
    ...     beta_list=constants.beta_i,  # From Phase 3
    ... )
"""

from ddfs.feasibility.ellipsoid_solver import (
    EllipsoidParams,
    EllipsoidSolver,
    FeasibilityEnvelope,
)

__all__ = [
    "EllipsoidParams",
    "EllipsoidSolver",
    "FeasibilityEnvelope",
]
