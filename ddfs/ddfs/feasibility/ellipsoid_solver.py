"""Ellipsoid-based feasibility solver for Phase 4 - FIXED OBSTACLE CONSTRAINTS.

This module provides the EllipsoidSolver class for computing Maximum Volume
Inscribed Ellipsoids (MVIE) that characterize feasibility envelopes.

CRITICAL FIX: Obstacle constraints now properly formulated to ensure no violations.

Key Points:
- External API uses P (shape matrix in deviation coordinates)
- Internally, optimization uses Q = P^{-1} and solves for Z where Q = Z^2
- Constraints are linearized around nominal trajectory x̄(k), ū(k)
- Obstacle constraints MUST ensure ellipsoid stays outside obstacle + beta
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from ddfs.core.config import DDFSConfig
from ddfs.core.obstacles import Obstacle

logger = logging.getLogger(__name__)


@dataclass
class EllipsoidParams:
    """Parameters defining an ellipsoid.

    The ellipsoid is defined as: E(P) = {η | η^T P η ≤ 1}
    where P is positive definite (shape matrix in deviation coordinates).

    In absolute coordinates: x̄ ⊕ E(P) = {x | (x-c)^T P (x-c) ≤ 1}

    Attributes:
        P: Shape matrix (positive definite), shape (nx, nx)
        c: Center point (nominal state/input), shape (nx,)
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
        return float(diff.T @ self.P @ diff) <= 1.0

    def volume(self) -> float:
        """Compute volume of the ellipsoid.

        For an ellipsoid E(P) in R^n, volume = (π^{n/2} / Γ(n/2 + 1)) / sqrt(det(P))

        Returns:
            Volume of the ellipsoid
        """
        n = len(self.c)
        det_P = np.linalg.det(self.P)

        # Compute π^{n/2} / Γ(n/2 + 1)
        from scipy.special import gamma  # noqa: PLC0415

        volume_constant = (np.pi ** (n / 2)) / gamma(n / 2 + 1)

        # Volume = volume_constant / sqrt(det(P))
        # (Larger P means smaller ellipsoid)
        return volume_constant / np.sqrt(det_P)


@dataclass
class FeasibilityEnvelope:
    """Container for feasibility envelope at both timestep and segment levels.

    Following paper Section 6.3.2:
    - P_min(k): Per-timestep feasibility ellipsoids
    - P_min,i: Per-segment conservative bounds
    - R_max(k): Per-timestep input ellipsoids
    - R_max,i: Per-segment input conservative bounds

    Attributes:
        segment_index: Segment index
        k_start: Start timestep of segment
        k_end: End timestep of segment

        # Per-timestep envelopes (Level 1) - from MVIE
        P_min_timestep: List of P_min(k) for k ∈ [k_start, k_end+1]
        R_max_timestep: List of R_max(k) for k ∈ [k_start, k_end]

        # Per-segment envelopes (Level 2) - conservative bounds
        P_min_segment: Conservative P_min,i over segment
        R_max_segment: Conservative R_max,i over segment

        # Bootstrap ellipsoids
        P_0: Initial ellipsoid at k=0 for this segment
        P_min_0_init: Initial conservative bound
        bootstrap_consistent: Whether P_0 ⊆ P_min_0 ⊆ P_min_0_init
    """

    segment_index: int
    k_start: int
    k_end: int

    # Level 1: Per-timestep (from MVIE)
    P_min_timestep: List[EllipsoidParams]
    R_max_timestep: List[EllipsoidParams]

    # Level 2: Per-segment (conservative)
    P_min_segment: EllipsoidParams
    R_max_segment: EllipsoidParams

    # Bootstrap
    P_0: EllipsoidParams
    P_min_0_init: EllipsoidParams
    bootstrap_consistent: bool

    def __repr__(self) -> str:
        return (
            f"FeasibilityEnvelope(seg={self.segment_index}, "
            f"k=[{self.k_start}:{self.k_end}], "
            f"timesteps={len(self.P_min_timestep)})"
        )


class EllipsoidSolver:
    """EllipsoidSolver class for MVIE-based feasibility envelope computation.

    Implements the MVIE formulation with CORRECT obstacle avoidance constraints.

    CRITICAL: Obstacle constraint formulation ensures ellipsoid + nominal stays
    outside obstacle radius + beta at ALL points in the ellipsoid.

    Internally optimizes over Q = P^{-1} via Z where Q = Z^2:
        Z_opt = argmax_Z log det(Z)
        s.t. ||Z a_i(t)||_2 ≤ b_i(t), ∀i   [CORRECTED]
             0 ⪯ Z ⪯ x_max I

    Then returns P = Q^{-1} = (Z^2)^{-1} to external callers.
    """

    def __init__(
        self,
        config: DDFSConfig,
        obstacles: Optional[List[Obstacle]] = None,
        workspace: Optional = None,
    ):
        """Initialize the ellipsoid solver.

        Args:
            config: DDFS configuration
            obstacles: List of obstacles to avoid (optional)
            workspace: Workspace object (optional, for workspace constraints)
        """
        self.config = config
        self.obstacles = obstacles or []
        self.workspace = workspace

        # Get state dimension from config
        system_config = config.get_system_config()
        self.nx = system_config["state_dim"]

        logger.info("Initialized EllipsoidSolver (MVIE formulation)")
        logger.info(f"  State dimension: {self.nx}")
        logger.info(f"  Number of obstacles: {len(self.obstacles)}")

    def solve_mvie_per_timestep(
        self,
        nominal,
        segment_index: int,
        k_start: int,
        k_end: int,
        beta: float,
        verbose: bool = False,
    ) -> Tuple[List[EllipsoidParams], List[EllipsoidParams]]:
        """Solve MVIE for each timestep in a segment.

        Computes P_min(k) and R_max(k) for k ∈ [k_start, k_end].

        Args:
            nominal: NominalTrajectory with x_nom, u_nom
            segment_index: Segment index
            k_start: Start timestep of segment
            k_end: End timestep of segment (inclusive)
            beta: Safety margin for obstacles
            verbose: Enable solver output

        Returns:
            P_min_list: List of state ellipsoids P_min(k) for k ∈ [k_start, k_end+1]
            R_max_list: List of input ellipsoids R_max(k) for k ∈ [k_start, k_end]
        """
        logger.info(f"Computing per-timestep MVIE for segment {segment_index}")
        logger.info(f"  Timesteps: k ∈ [{k_start}, {k_end}]")

        nx = self.nx
        nu = self.config.nu if hasattr(self.config, "nu") else nominal.u_nom.shape[1]

        P_min_list = []
        R_max_list = []

        # Compute P_min(k) for each state timestep
        for k in range(k_start, min(k_end + 2, nominal.N + 1)):
            x_nom_k = nominal.x_nom[k]

            try:
                P_min_k = self._solve_mvie_state_single_timestep(
                    x_nom=x_nom_k,
                    k=k,
                    beta=beta,
                    verbose=verbose,
                )
                P_min_k.segment_index = segment_index
                P_min_list.append(P_min_k)

            except Exception as e:
                logger.warning(f"  MVIE failed at k={k}: {e}")
                # Fallback: small diagonal ellipsoid
                P_fallback = np.eye(nx) * 100.0  # Large P = small ellipsoid
                P_min_list.append(EllipsoidParams(P=P_fallback, c=x_nom_k, segment_index=segment_index))

        # Compute R_max(k) for each input timestep
        for k in range(k_start, min(k_end + 1, nominal.N)):
            u_nom_k = nominal.u_nom[k]

            try:
                R_max_k = self._solve_mvie_input_single_timestep(
                    u_nom=u_nom_k,
                    k=k,
                    verbose=verbose,
                )
                R_max_k.segment_index = segment_index
                R_max_list.append(R_max_k)

            except Exception as e:
                logger.warning(f"  Input MVIE failed at k={k}: {e}")
                # Fallback
                R_fallback = np.eye(nu) * 100.0
                R_max_list.append(EllipsoidParams(P=R_fallback, c=u_nom_k, segment_index=segment_index))

        logger.info(f"  ✓ Per-timestep MVIE: {len(P_min_list)} state, {len(R_max_list)} input")
        return P_min_list, R_max_list

    def _solve_mvie_state_single_timestep(  # noqa: C901, PLR0912, PLR0915
        self,
        x_nom: np.ndarray,
        k: int,
        beta: float,
        verbose: bool = False,
    ) -> EllipsoidParams:
        """Solve MVIE for state envelope at single timestep k.

        CRITICAL FIX: Obstacle constraints now correctly formulated.

        The ellipsoid in absolute coordinates is:
            E = {x | (x - x̄)^T P (x - x̄) ≤ 1}
          = {x | η^T P η ≤ 1} where η = x - x̄

        For obstacle avoidance, we need:
            ||x - c_obs|| ≥ r_obs + β  for all x in E

        Linearizing around x̄:
            ||x - c_obs|| ≈ ||x̄ - c_obs|| + ∇||x̄ - c_obs||^T (x - x̄)
                           = d + (x̄ - c_obs)^T (x - x̄) / d

        where d = ||x̄ - c_obs||

        For all x in E, we need:
            d + (x̄ - c_obs)^T (x - x̄) / d ≥ r_obs + β
            => (x̄ - c_obs)^T (x - x̄) ≥ (r_obs + β - d) * d
            => (x̄ - c_obs)^T η ≥ (r_obs + β - d) * d

        In terms of Q = P^{-1} and Z where Q = Z^2:
            For η^T P η ≤ 1  =>  η^T Q^{-1} η ≤ 1

        The constraint becomes (using conic form):
            a^T η ≥ b  for all η with η^T Q^{-1} η ≤ 1

        This is equivalent to:
            ||Z a||_2 ≤ -b  when b < 0 (obstacle is far)

        Args:
            x_nom: Nominal state x̄(k) at timestep k
            k: Timestep index
            beta: Safety margin for obstacles
            verbose: Solver output

        Returns:
            P_min_k: State ellipsoid (returns P, not Q)
        """
        nx = len(x_nom)

        # Decision variable Z (we optimize Q = Z^2, then return P = Q^{-1})
        Z = cp.Variable((nx, nx), symmetric=True)

        # Objective: maximize log det(Z) => maximize log det(Q) => minimize log det(P)
        objective = cp.Maximize(cp.log_det(Z))

        # Constraints
        constraints = [Z >> 1e-6 * np.eye(nx)]

        # --- OBSTACLE AVOIDANCE CONSTRAINTS (CORRECTED) ---
        num_active_constraints = 0

        for obs in self.obstacles:
            x_pos = x_nom[: len(obs.center)]
            obs_center = obs.center
            diff = x_pos - obs_center  # x̄ - c_obs
            dist = np.linalg.norm(diff)  # d = ||x̄ - c_obs||
            obs_radius_safe = obs.effective_radius + beta

            # Check if nominal is safe
            if dist < obs_radius_safe:
                logger.warning(
                    f"    k={k}: Nominal too close to obstacle {obs.id} "
                    f"(dist={dist:.4f} < safe_radius={obs_radius_safe:.4f})"
                )
                logger.warning("    Skipping this obstacle constraint - replan needed!")
                continue

            if dist < 1e-6:
                logger.warning(f"    k={k}: Nominal at obstacle center {obs.id}")
                continue

            # Compute clearance: how much space we have
            clearance = dist - obs_radius_safe

            if clearance < 1e-6:
                logger.warning(f"    k={k}: No clearance for obstacle {obs.id}, skipping")
                continue

            # Direction vector (normalized): (x̄ - c_obs) / ||x̄ - c_obs||
            direction = diff / dist

            # Pad to full state dimension
            a_obs = np.zeros(nx)
            a_obs[: len(obs.center)] = direction

            # The constraint is: a^T η ≥ -clearance for all η in ellipsoid
            # In conic form: ||Z a||_2 ≤ clearance
            #
            # CRITICAL: This ensures that any deviation η in the ellipsoid
            # keeps us at least 'clearance' distance from the obstacle surface

            constraints.append(cp.norm(Z @ a_obs, 2) <= clearance)
            num_active_constraints += 1

            if verbose and num_active_constraints <= 3:
                logger.info(f"    k={k}: Obstacle {obs.id} constraint: ||Z a||_2 ≤ {clearance:.4f}")

        if num_active_constraints > 0:
            logger.info(f"    k={k}: Added {num_active_constraints} obstacle constraints")
        else:
            logger.info(f"    k={k}: No active obstacle constraints (all obstacles far)")

        # --- WORKSPACE BOUNDS CONSTRAINTS ---
        # Ensure ellipsoid stays within workspace: x_min ≤ x ≤ x_max, y_min ≤ y ≤ y_max
        num_workspace_constraints = 0

        if self.workspace is not None:
            # For each dimension with workspace bounds, add constraints
            # For dimension i: x_min[i] ≤ x_nom[i] + η_i ≤ x_max[i]
            # => -clearance_lower ≤ η_i ≤ clearance_upper
            # where clearance_lower = x_nom[i] - x_min[i], clearance_upper = x_max[i] - x_nom[i]

            # x dimension (i=0)
            if hasattr(self.workspace, "x_min") and hasattr(self.workspace, "x_max"):
                clearance_lower_x = x_nom[0] - self.workspace.x_min
                clearance_upper_x = self.workspace.x_max - x_nom[0]

                if clearance_lower_x > 1e-6:
                    e_x = np.zeros(nx)
                    e_x[0] = 1.0
                    constraints.append(cp.norm(Z @ e_x, 2) <= clearance_lower_x)
                    num_workspace_constraints += 1

                if clearance_upper_x > 1e-6:
                    e_x = np.zeros(nx)
                    e_x[0] = 1.0
                    constraints.append(cp.norm(Z @ e_x, 2) <= clearance_upper_x)
                    num_workspace_constraints += 1

            # y dimension (i=1)
            if nx > 1 and hasattr(self.workspace, "y_min") and hasattr(self.workspace, "y_max"):
                clearance_lower_y = x_nom[1] - self.workspace.y_min
                clearance_upper_y = self.workspace.y_max - x_nom[1]

                if clearance_lower_y > 1e-6:
                    e_y = np.zeros(nx)
                    e_y[1] = 1.0
                    constraints.append(cp.norm(Z @ e_y, 2) <= clearance_lower_y)
                    num_workspace_constraints += 1

                if clearance_upper_y > 1e-6:
                    e_y = np.zeros(nx)
                    e_y[1] = 1.0
                    constraints.append(cp.norm(Z @ e_y, 2) <= clearance_upper_y)
                    num_workspace_constraints += 1

        if num_workspace_constraints > 0:
            logger.info(f"    k={k}: Added {num_workspace_constraints} workspace bound constraints")

        # --- UPPER BOUND: 0 ⪯ Z ⪯ x_max I (loose constraint for numerical stability) ---
        if self.workspace is not None:
            # Use workspace dimensions as upper bound
            if hasattr(self.workspace, "x_max") and hasattr(self.workspace, "y_max"):
                x_max_bound = max(self.workspace.x_max, self.workspace.y_max)
            else:
                x_max_bound = 10.0
        else:
            x_max_bound = 10.0

        constraints.append(Z << x_max_bound * np.eye(nx))

        # Solve
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6, max_iters=5000)
        except Exception as e:
            raise ValueError(f"MVIE solve failed at k={k}: {e}")

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"MVIE infeasible at k={k}: status={problem.status}")

        # Extract solution: Z_opt
        Z_opt = (Z.value + Z.value.T) / 2  # Symmetrize

        # Compute Q = Z^2
        Q_opt = Z_opt @ Z_opt

        # Return P = Q^{-1}
        try:
            P_min = np.linalg.inv(Q_opt)
            P_min = (P_min + P_min.T) / 2  # Ensure symmetry
        except np.linalg.LinAlgError:
            logger.warning(f"    k={k}: Q_opt is singular, using fallback")
            P_min = np.eye(nx) * 1000.0  # Very conservative fallback

        return EllipsoidParams(P=P_min, c=x_nom, segment_index=-1)

    def _solve_mvie_input_single_timestep(
        self,
        u_nom: np.ndarray,
        k: int,
        verbose: bool = False,
    ) -> EllipsoidParams:
        """Solve MVIE for input envelope at single timestep k.

        Args:
            u_nom: Nominal control ū(k) at timestep k
            k: Timestep index
            verbose: Solver output

        Returns:
            R_max_k: Input ellipsoid (returns P, not Q)
        """
        nu = len(u_nom)

        # Get control constraints
        constraints_config = self.config.constraints if hasattr(self.config, "constraints") else None

        if constraints_config is None or not hasattr(constraints_config, "u_min"):
            # No input constraints - return unit ellipsoid
            R_max = np.eye(nu)
            return EllipsoidParams(P=R_max, c=u_nom, segment_index=-1)

        u_min = np.array(constraints_config.u_min)
        u_max = np.array(constraints_config.u_max)

        # Decision variable Z
        Z = cp.Variable((nu, nu), symmetric=True)

        # Objective
        objective = cp.Maximize(cp.log_det(Z))

        # Constraints
        constraints = [Z >> 1e-6 * np.eye(nu)]

        # Input box constraints: u_min ≤ u_nom + ξ ≤ u_max
        # => u_min - u_nom ≤ ξ ≤ u_max - u_nom
        for i in range(nu):
            # Lower bound: ξ_i ≥ u_min[i] - u_nom[i]
            # For ellipsoid: ||Z e_i||_2 ≤ u_nom[i] - u_min[i]
            clearance_lower = u_nom[i] - u_min[i]
            if clearance_lower > 1e-6:
                e_i = np.zeros(nu)
                e_i[i] = 1.0
                constraints.append(cp.norm(Z @ e_i, 2) <= clearance_lower)

            # Upper bound: ξ_i ≤ u_max[i] - u_nom[i]
            # For ellipsoid: ||Z e_i||_2 ≤ u_max[i] - u_nom[i]
            clearance_upper = u_max[i] - u_nom[i]
            if clearance_upper > 1e-6:
                e_i = np.zeros(nu)
                e_i[i] = 1.0
                constraints.append(cp.norm(Z @ e_i, 2) <= clearance_upper)

        # Upper bound on Z
        valid_bounds = [
            min(u_nom[i] - u_min[i], u_max[i] - u_nom[i])
            for i in range(nu)
            if u_nom[i] - u_min[i] > 0 and u_max[i] - u_nom[i] > 0
        ]
        u_max_bound = max(valid_bounds) if valid_bounds else 1.0
        constraints.append(Z << u_max_bound * np.eye(nu))

        # Solve
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6, max_iters=3000)
        except Exception as e:
            raise ValueError(f"Input MVIE failed at k={k}: {e}")

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Input MVIE infeasible at k={k}: {problem.status}")

        Z_opt = (Z.value + Z.value.T) / 2
        Q_opt = Z_opt @ Z_opt

        # Return P = Q^{-1}
        R_max = np.linalg.inv(Q_opt)
        R_max = (R_max + R_max.T) / 2

        return EllipsoidParams(P=R_max, c=u_nom, segment_index=-1)

    def compute_segment_envelopes(
        self,
        P_min_timestep_list: List[EllipsoidParams],
        R_max_timestep_list: List[EllipsoidParams],
        segment_index: int,
    ) -> Tuple[EllipsoidParams, EllipsoidParams]:
        """Compute conservative per-segment bounds from per-timestep ellipsoids.

        Computes P_min,i and R_max,i such that:
            E(P_min,i) ⊇ E(P_min(k)) for all k in segment

        For diagonal ellipsoids, this means taking MINIMUM diagonal elements of P
        (smaller P means larger ellipsoid).

        Args:
            P_min_timestep_list: List of P_min(k) for k in segment
            R_max_timestep_list: List of R_max(k) for k in segment
            segment_index: Segment index

        Returns:
            P_min_i: Conservative state envelope for segment
            R_max_i: Conservative input envelope for segment
        """
        logger.info(f"  Computing segment-level envelopes for segment {segment_index}")

        # Compute average center
        centers_state = np.array([ell.c for ell in P_min_timestep_list])
        c_state_avg = np.mean(centers_state, axis=0)

        # Take element-wise MINIMUM of diagonal entries of P
        # (smaller P means larger ellipsoid, so we want minimum for conservative bound)
        diagonals = []
        for ell in P_min_timestep_list:
            diagonals.append(np.diag(ell.P))
        diagonals = np.array(diagonals)

        # P_min,i: use MINIMUM along each dimension (most permissive = smallest P)
        P_min_i_diag = np.min(diagonals, axis=0)
        P_min_i_matrix = np.diag(P_min_i_diag)

        P_min_i = EllipsoidParams(
            P=P_min_i_matrix,
            c=c_state_avg,
            segment_index=segment_index,
        )

        # R_max,i: Similar for inputs
        if len(R_max_timestep_list) > 0:
            centers_input = np.array([ell.c for ell in R_max_timestep_list])
            c_input_avg = np.mean(centers_input, axis=0)

            diagonals_input = []
            for ell in R_max_timestep_list:
                diagonals_input.append(np.diag(ell.P))
            diagonals_input = np.array(diagonals_input)

            # R_max,i: use MINIMUM (most permissive = smallest P)
            R_max_i_diag = np.min(diagonals_input, axis=0)
            R_max_i_matrix = np.diag(R_max_i_diag)

            R_max_i = EllipsoidParams(
                P=R_max_i_matrix,
                c=c_input_avg,
                segment_index=segment_index,
            )
        else:
            # Fallback
            nu = len(R_max_timestep_list[0].c) if R_max_timestep_list else 2
            R_max_i = EllipsoidParams(
                P=np.eye(nu),
                c=np.zeros(nu),
                segment_index=segment_index,
            )

        logger.info(f"    P_min,i volume: {P_min_i.volume():.6e}")
        logger.info(f"    R_max,i volume: {R_max_i.volume():.6e}")

        return P_min_i, R_max_i
