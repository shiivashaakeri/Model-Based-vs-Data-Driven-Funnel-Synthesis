"""Ellipsoid-based feasibility solver for Phase 4.

This module provides the EllipsoidSolver class for computing Maximum Volume
Inscribed Ellipsoids (MVIE) that characterize feasibility envelopes. These
ellipsoids ensure that trajectories remain within safe regions while
accounting for uncertainty from Phase 3.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import cvxpy as cp
import numpy as np
from core.config import DDFSConfig
from core.obstacles import Obstacle
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class EllipsoidParams:
    """Parameters defining an ellipsoid.

    The ellipsoid is defined as: {x | (x - c)^T P^{-1} (x - c) <= 1}
    where P is positive definite.

    Attributes:
        P: Shape matrix (positive definite), shape (nx, nx)
        c: Center point, shape (nx,)
        segment_index: Segment index this ellipsoid corresponds to
    """

    P: NDArray[np.float64]  # Shape: (nx, nx)
    c: NDArray[np.float64]  # Shape: (nx,)
    segment_index: int

    def __post_init__(self):
        """Validate ellipsoid parameters."""
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError(f"P must be square, got shape {self.P.shape}")

        if len(self.c) != self.P.shape[0]:
            raise ValueError(f"Center dimension {len(self.c)} != P dimension {self.P.shape[0]}")

        # Check positive definiteness
        try:
            np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            raise ValueError("P must be positive definite")

    def contains(self, x: NDArray[np.float64]) -> bool:
        """Check if point x is inside the ellipsoid.

        Args:
            x: Point to check, shape (nx,)

        Returns:
            True if x is inside the ellipsoid
        """
        diff = x - self.c
        P_inv = np.linalg.inv(self.P)
        return float(diff.T @ P_inv @ diff) <= 1.0

    def volume(self) -> float:
        """Compute volume of the ellipsoid.

        For an ellipsoid in R^n, volume = (π^{n/2} / Γ(n/2 + 1)) * sqrt(det(P))

        Returns:
            Volume of the ellipsoid
        """
        n = len(self.c)
        det_P = np.linalg.det(self.P)

        # Compute π^{n/2} / Γ(n/2 + 1)
        from scipy.special import gamma  # noqa: PLC0415

        volume_constant = (np.pi ** (n / 2)) / gamma(n / 2 + 1)

        return volume_constant * np.sqrt(det_P)


@dataclass
class FeasibilityEnvelope:
    """Container for feasibility envelope consisting of ellipsoids.

    Attributes:
        P_0: Initial ellipsoid (from nominal planner)
        P_min_0: Minimum initial ellipsoid (from MVIE optimization)
        P_min_0_init: Initial guess for P_min_0
        segment_indices: Segment indices
        bootstrap_consistent: Whether bootstrap consistency holds (P_0 ⊆ P_min_0 ⊆ P_min_0_init)
    """

    P_0: EllipsoidParams
    P_min_0: EllipsoidParams
    P_min_0_init: EllipsoidParams
    segment_indices: List[int]
    bootstrap_consistent: bool = True

    def __repr__(self) -> str:
        status = "✓" if self.bootstrap_consistent else "✗"
        return f"FeasibilityEnvelope(segments={len(self.segment_indices)}, bootstrap_consistent={status})"


class EllipsoidSolver:
    """EllipsoidSolver class for MVIE-based feasibility checking.

    This class solves Maximum Volume Inscribed Ellipsoid (MVIE) optimization
    problems to compute feasibility envelopes that:
    1. Contain the nominal trajectory uncertainty
    2. Avoid obstacles with safety margins
    3. Remain within the workspace
    4. Satisfy bootstrap consistency

    The solver uses CVXPY with SDP (semidefinite programming) for convex
    optimization.

    Example:
        >>> config = DDFSConfig(...)
        >>> obstacles = [CircularObstacle(...), ...]
        >>> solver = EllipsoidSolver(config, obstacles)
        >>>
        >>> # Compute P_min_0 for a segment
        >>> P_0 = EllipsoidParams(P=np.eye(3), c=np.zeros(3), segment_index=0)
        >>> P_min_0_init = EllipsoidParams(P=2*np.eye(3), c=np.zeros(3), segment_index=0)
        >>>
        >>> P_min_0 = solver.solve_mvie(
        ...     P_0=P_0,
        ...     P_min_0_init=P_min_0_init,
        ...     beta=0.01,  # Uncertainty bound from Phase 3
        ... )
    """

    def __init__(
        self,
        config: DDFSConfig,
        obstacles: Optional[List[Obstacle]] = None,
    ):
        """Initialize the ellipsoid solver.

        Args:
            config: DDFS configuration
            obstacles: List of obstacles to avoid (optional)
        """
        self.config = config
        self.obstacles = obstacles or []

        logger.info("Initialized EllipsoidSolver")
        logger.info(f"  State dimension: {self.config.nx}")
        logger.info(f"  Number of obstacles: {len(self.obstacles)}")

    def solve_mvie(
        self,
        P_0: EllipsoidParams,
        P_min_0_init: EllipsoidParams,
        beta: float,
        verbose: bool = False,
    ) -> EllipsoidParams:
        """Solve MVIE optimization for a single segment.

        Computes P_min_0 that maximizes volume while satisfying:
        1. P_0 ⊆ P_min_0 (contains nominal uncertainty)
        2. P_min_0 ⊆ P_min_0_init (bootstrap consistency)
        3. P_min_0 avoids obstacles with margin beta
        4. P_min_0 is within workspace

        Args:
            P_0: Nominal ellipsoid from planner
            P_min_0_init: Initial guess for P_min_0
            beta: Uncertainty bound from Phase 3 (used for obstacle margins)
            verbose: Enable CVXPY solver output

        Returns:
            P_min_0: Optimized ellipsoid

        Raises:
            ValueError: If optimization fails to find feasible solution
        """
        logger.info(f"Solving MVIE for segment {P_0.segment_index}...")
        logger.info(f"  P_0 center: {P_0.c}")
        logger.info(f"  P_0 volume: {P_0.volume():.6e}")
        logger.info(f"  P_min_0_init volume: {P_min_0_init.volume():.6e}")
        logger.info(f"  beta = {beta:.6e}")

        nx = self.config.nx

        # Decision variables: P (shape matrix) and c (center)
        P = cp.Variable((nx, nx), symmetric=True)
        c = cp.Variable(nx)

        # Objective: maximize log(det(P)) (equivalent to maximizing volume)
        objective = cp.Maximize(cp.log_det(P))

        # Constraints
        constraints = []

        # [1] P must be positive definite
        constraints.append(P >> 0)  # P is PSD with eigenvalues > 0

        # [2] P_0 ⊆ P_min_0 (containment constraint)
        # This is equivalent to: [P, P_0.c - c; (P_0.c - c)^T, P_0.P] ⪰ 0
        diff_0 = P_0.c - c
        M_0 = cp.bmat([[P, cp.reshape(diff_0, (nx, 1))], [cp.reshape(diff_0, (1, nx)), P_0.P]])
        constraints.append(M_0 >> 0)

        # [3] P_min_0 ⊆ P_min_0_init (bootstrap consistency)
        # Similar LMI constraint
        diff_init = c - P_min_0_init.c
        M_init = cp.bmat([[P_min_0_init.P, cp.reshape(diff_init, (nx, 1))], [cp.reshape(diff_init, (1, nx)), P]])
        constraints.append(M_init >> 0)

        # [4] Obstacle avoidance constraints
        for obs in self.obstacles:
            obs_constraint = self._obstacle_constraint(P, c, obs, beta)
            if obs_constraint is not None:
                constraints.append(obs_constraint)

        # [5] Workspace constraints
        workspace_constraints = self._workspace_constraints(P, c)
        constraints.extend(workspace_constraints)

        # Solve
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(
                solver=cp.SCS,
                verbose=verbose,
                eps=1e-6,
                max_iters=5000,
            )
        except Exception as e:
            logger.error(f"CVXPY solver failed: {e}")
            raise ValueError(f"MVIE optimization failed: {e}")

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.error(f"Optimization status: {problem.status}")
            raise ValueError(f"MVIE optimization infeasible: {problem.status}")

        # Extract solution
        P_opt = P.value
        c_opt = c.value

        # Ensure P is symmetric (numerical issues)
        P_opt = (P_opt + P_opt.T) / 2

        # Create result ellipsoid
        P_min_0 = EllipsoidParams(
            P=P_opt,
            c=c_opt,
            segment_index=P_0.segment_index,
        )

        logger.info(f"  P_min_0 center: {P_min_0.c}")
        logger.info(f"  P_min_0 volume: {P_min_0.volume():.6e}")
        logger.info(f"  Optimization status: {problem.status}")

        return P_min_0

    def compute_envelope(
        self,
        P_0_list: List[EllipsoidParams],
        P_min_0_init_list: List[EllipsoidParams],
        beta_list: List[float],
        verbose: bool = False,
    ) -> FeasibilityEnvelope:
        """Compute feasibility envelope for all segments.

        Args:
            P_0_list: List of nominal ellipsoids (one per segment)
            P_min_0_init_list: List of initial guesses (one per segment)
            beta_list: List of uncertainty bounds (one per segment)
            verbose: Enable solver output

        Returns:
            FeasibilityEnvelope containing all ellipsoids

        Example:
            >>> envelope = solver.compute_envelope(
            ...     P_0_list=[P_0_seg0, P_0_seg1, ...],
            ...     P_min_0_init_list=[init_seg0, init_seg1, ...],
            ...     beta_list=[beta_0, beta_1, ...],
            ... )
        """
        logger.info("=" * 70)
        logger.info("PHASE 4: COMPUTING FEASIBILITY ENVELOPE")
        logger.info("=" * 70)

        num_segments = len(P_0_list)
        logger.info(f"Number of segments: {num_segments}")

        # Validate inputs
        if len(P_min_0_init_list) != num_segments:
            raise ValueError(f"P_min_0_init_list length {len(P_min_0_init_list)} != num_segments {num_segments}")

        if len(beta_list) != num_segments:
            raise ValueError(f"beta_list length {len(beta_list)} != num_segments {num_segments}")

        # Solve MVIE for each segment
        P_min_0_list = []

        for i in range(num_segments):
            logger.info(f"\n[Segment {i + 1}/{num_segments}]")

            P_min_0 = self.solve_mvie(
                P_0=P_0_list[i],
                P_min_0_init=P_min_0_init_list[i],
                beta=beta_list[i],
                verbose=verbose,
            )

            P_min_0_list.append(P_min_0)

        # Verify bootstrap consistency for all segments
        logger.info("\nVerifying bootstrap consistency...")
        bootstrap_consistent = True

        for i in range(num_segments):
            P_0 = P_0_list[i]
            P_min_0 = P_min_0_list[i]
            P_min_0_init = P_min_0_init_list[i]

            # Check P_0 ⊆ P_min_0 ⊆ P_min_0_init
            consistent = self._verify_bootstrap_consistency(P_0, P_min_0, P_min_0_init)

            if not consistent:
                logger.warning(f"  Segment {i}: Bootstrap consistency violated!")
                bootstrap_consistent = False
            else:
                logger.info(f"  Segment {i}: ✓ Bootstrap consistent")

        segment_indices = [P.segment_index for P in P_0_list]

        envelope = FeasibilityEnvelope(
            P_0=P_0_list[0],  # Store first segment as representative
            P_min_0=P_min_0_list[0],
            P_min_0_init=P_min_0_init_list[0],
            segment_indices=segment_indices,
            bootstrap_consistent=bootstrap_consistent,
        )

        logger.info("\n" + "=" * 70)
        logger.info("FEASIBILITY ENVELOPE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"{envelope}")

        return envelope

    def _obstacle_constraint(
        self,
        P: cp.Variable,
        c: cp.Variable,
        obstacle: Obstacle,
        beta: float,
    ) -> Optional[cp.Constraint]:
        """Generate obstacle avoidance constraint.

        For a circular obstacle at position (x_obs, y_obs) with radius r,
        the constraint is:
        ||c_xy - [x_obs; y_obs]|| - r >= beta + ||P_xy^{1/2}||_2

        where c_xy and P_xy are the x-y components of c and P.

        Args:
            P: Shape matrix variable
            c: Center variable
            obstacle: Obstacle to avoid
            beta: Safety margin

        Returns:
            Constraint or None if obstacle type not supported
        """
        from core.obstacles import CircularObstacle, EllipsoidalObstacle  # noqa: PLC0415

        if isinstance(obstacle, CircularObstacle):
            # Extract x-y components (assume first 2 dimensions)
            c_xy = c[:2]
            P_xy = P[:2, :2]

            # Obstacle center and radius
            obs_center = np.array([obstacle.x, obstacle.y])
            obs_radius = obstacle.radius

            # Distance from ellipsoid center to obstacle
            dist = cp.norm(c_xy - obs_center, 2)

            # Ellipsoid "reach" in x-y plane
            ellipsoid_reach = cp.norm(cp.sqrt(P_xy), 2)

            # Constraint: distance >= obstacle_radius + beta + ellipsoid_reach
            return dist >= obs_radius + beta + ellipsoid_reach

        elif isinstance(obstacle, EllipsoidalObstacle):
            # More complex constraint - for now, use conservative approximation
            # Approximate as circular obstacle with radius = max semiaxis
            max_semiaxis = np.max(np.sqrt(np.diag(obstacle.Q)))

            c_xy = c[:2]
            P_xy = P[:2, :2]
            obs_center = np.array([obstacle.x, obstacle.y])

            dist = cp.norm(c_xy - obs_center, 2)
            ellipsoid_reach = cp.norm(cp.sqrt(P_xy), 2)

            return dist >= max_semiaxis + beta + ellipsoid_reach

        else:
            logger.warning(f"Obstacle type {type(obstacle).__name__} not supported, skipping")
            return None

    def _workspace_constraints(
        self,
        P: cp.Variable,
        c: cp.Variable,
    ) -> List[cp.Constraint]:
        """Generate workspace containment constraints.

        Ensures that the ellipsoid stays within workspace bounds.

        Args:
            P: Shape matrix variable
            c: Center variable

        Returns:
            List of constraints
        """
        constraints = []

        # Extract x-y components
        c_x = c[0]
        c_y = c[1]
        P_x = P[0, 0]  # Variance in x direction
        P_y = P[1, 1]  # Variance in y direction

        # Workspace bounds
        x_min = self.config.workspace.x_min
        x_max = self.config.workspace.x_max
        y_min = self.config.workspace.y_min
        y_max = self.config.workspace.y_max

        # Conservative constraints: center ± reach must be in workspace
        # reach_x = sqrt(P_x), reach_y = sqrt(P_y)

        constraints.append(c_x - cp.sqrt(P_x) >= x_min)
        constraints.append(c_x + cp.sqrt(P_x) <= x_max)
        constraints.append(c_y - cp.sqrt(P_y) >= y_min)
        constraints.append(c_y + cp.sqrt(P_y) <= y_max)

        return constraints

    def _verify_bootstrap_consistency(
        self,
        P_0: EllipsoidParams,
        P_min_0: EllipsoidParams,
        P_min_0_init: EllipsoidParams,
        n_samples: int = 100,
    ) -> bool:
        """Verify bootstrap consistency: P_0 ⊆ P_min_0 ⊆ P_min_0_init.

        Uses sampling to check containment.

        Args:
            P_0: Nominal ellipsoid
            P_min_0: Optimized ellipsoid
            P_min_0_init: Initial guess
            n_samples: Number of samples to test

        Returns:
            True if bootstrap consistency holds
        """
        # Sample points on boundary of P_0
        points_P_0 = self._sample_ellipsoid_boundary(P_0, n_samples)

        # Check if all points are in P_min_0
        for point in points_P_0:
            if not P_min_0.contains(point):
                logger.debug(f"Point {point} in P_0 but not in P_min_0")
                return False

        # Sample points on boundary of P_min_0
        points_P_min_0 = self._sample_ellipsoid_boundary(P_min_0, n_samples)

        # Check if all points are in P_min_0_init
        for point in points_P_min_0:
            if not P_min_0_init.contains(point):
                logger.debug(f"Point {point} in P_min_0 but not in P_min_0_init")
                return False

        return True

    def _sample_ellipsoid_boundary(
        self,
        ellipsoid: EllipsoidParams,
        n_samples: int,
    ) -> NDArray[np.float64]:
        """Sample points on the boundary of an ellipsoid.

        Args:
            ellipsoid: Ellipsoid to sample from
            n_samples: Number of samples

        Returns:
            Array of points, shape (n_samples, nx)
        """
        nx = len(ellipsoid.c)

        # Sample points on unit sphere
        samples_unit = np.random.randn(n_samples, nx)
        samples_unit = samples_unit / np.linalg.norm(samples_unit, axis=1, keepdims=True)

        # Transform to ellipsoid: x = c + P^{1/2} * u where ||u|| = 1
        P_sqrt = np.linalg.cholesky(ellipsoid.P)
        samples = ellipsoid.c + (P_sqrt @ samples_unit.T).T

        return samples
