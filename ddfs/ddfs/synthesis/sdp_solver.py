"""SDP (Semidefinite Programming) solver for Phase 5 funnel synthesis.

This module formulates and solves the SDP problem for computing funnel
shape matrices P_i that maximize volume while satisfying stability,
containment, and control constraints.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from ddfs.data_collection.hankel import SegmentHankelMatrices
from ddfs.synthesis.lmi_builder import LMIBuilder
from ddfs.uncertainty.constants import UncertaintyConstants

logger = logging.getLogger(__name__)


@dataclass
class SDPSolution:
    """Container for SDP solution.

    Attributes:
        P_i: Optimized shape matrix (nxn)
        L_i: Optimized control gain matrix (mxn)
        K_i: Feedback gain K_i = L_i P_i^{-1} (mxn)
        nu: Lyapunov decrease rate
        lambda1: S-procedure multiplier for Ñ₁
        lambda2: S-procedure multiplier for Ñ₂
        objective_value: log(det(P_i)) - volume objective
        volume: Actual ellipsoid volume
        status: Solver status
        segment_idx: Segment index
    """

    P_i: NDArray[np.float64]
    L_i: NDArray[np.float64]
    K_i: NDArray[np.float64]
    nu: float
    lambda1: float
    lambda2: float
    objective_value: float
    volume: float
    status: str
    segment_idx: int

    def __repr__(self) -> str:
        return f"SDPSolution(segment={self.segment_idx}, volume={self.volume:.6e}, status={self.status})"


class SDPSolver:
    """SDP solver for funnel synthesis.

    Solves the optimization problem:

        maximize    log(det(P_i))
        subject to  P_i ≻ 0
                    lambda1 ≥ 0, lambda2 ≥ 0, nu ≥ 0
                    S(P_i, L_i, nu) - lambda1 * N1_tilde - lambda2 * N2_tilde ≻ 0  [Stability LMI]
                    P_i ⪰ P_min_i                        [Lower bound]
                    P_i ⪯ μ P_{i-1}                      [Predecessor containment]
                    [R_max_i, L_i; L_i^T, P_i] ⪰ 0       [Control constraint]

    The objective maximizes ellipsoid volume while ensuring:
    1. Robust stability (via S-procedure with Ñ₁, Ñ₂)
    2. Containment of minimum feasible set P_min_i
    3. Smooth funnel progression (bounded by μ times predecessor)
    4. Control input constraints (via R_max_i)

    Example:
        >>> from ddfs.synthesis import SDPSolver, LMIBuilder
        >>>
        >>> # Setup
        >>> n, m = 3, 2
        >>> lmi_builder = LMIBuilder(n, m, alpha=0.95)
        >>> solver = SDPSolver(lmi_builder)
        >>>
        >>> # Solve for segment i
        >>> solution = solver.solve_segment(
        ...     segment_idx=0,
        ...     P_min_i=P_min_ellipsoid,
        ...     R_max_i=control_limit_matrix,
        ...     constants=uncertainty_constants,
        ...     hankel=hankel_matrices,
        ...     T=segment_length,
        ...     P_prev=None,  # First segment
        ...     mu=1.1,
        ... )
    """

    def __init__(
        self,
        lmi_builder: LMIBuilder,
        solver_name: str = "SCS",
        verbose: bool = False,
    ):
        """Initialize SDP solver.

        Args:
            lmi_builder: LMI builder instance
            solver_name: CVXPY solver to use (SCS, MOSEK, CVXOPT)
            verbose: Enable solver output
        """
        self.lmi_builder = lmi_builder
        self.solver_name = solver_name
        self.verbose = verbose

        self.n = lmi_builder.n
        self.m = lmi_builder.m

        logger.info(f"Initialized SDPSolver (solver={solver_name})")
        logger.info(f"  State dimension: {self.n}")
        logger.info(f"  Control dimension: {self.m}")

    def solve_segment(  # noqa: PLR0915
        self,
        segment_idx: int,
        P_min_i: NDArray[np.float64],
        R_max_i: NDArray[np.float64],
        constants: UncertaintyConstants,
        hankel: SegmentHankelMatrices,
        T: int,
        P_prev: Optional[NDArray[np.float64]] = None,
        mu: float = 1.1,
        eps_psd: float = 1e-6,
    ) -> SDPSolution:
        """Solve SDP for a single funnel segment.

        Args:
            segment_idx: Segment index i
            P_min_i: Minimum feasible shape matrix from Phase 4 (nxn)
            R_max_i: Maximum control constraint matrix (mxm)
            constants: Uncertainty constants from Phase 3
            hankel: Hankel matrices from Phase 2
            T: Segment length
            P_prev: Previous segment shape matrix P_{i-1} (nxn), None for first segment
            mu: Funnel growth bound (P_i ⪯ μ P_{i-1})
            eps_psd: Small positive constant for numerical stability

        Returns:
            SDPSolution with optimized matrices and gains

        Raises:
            ValueError: If optimization fails
        """
        logger.info("=" * 70)
        logger.info(f"SOLVING SDP FOR SEGMENT {segment_idx}")
        logger.info("=" * 70)

        # Create decision variables
        P_i = cp.Variable((self.n, self.n), symmetric=True)
        L_i = cp.Variable((self.m, self.n))
        nu = cp.Variable(nonneg=True)
        lambda1 = cp.Variable(nonneg=True)
        lambda2 = cp.Variable(nonneg=True)

        # Validate variable dimensions
        self.lmi_builder.validate_dimensions(P_i, L_i)

        # Objective: maximize log(det(P_i)) ≡ maximize volume
        objective = cp.Maximize(cp.log_det(P_i))

        # Constraints list
        constraints = []

        # [1] Auxiliary constraints: P_i ≻ 0, λ₁ ≥ 0, λ₂ ≥ 0, nu ≥ 0
        logger.info("\n[1/5] Adding auxiliary constraints...")
        aux_constraints = self.lmi_builder.build_auxiliary_constraints(P_i, lambda1, lambda2, nu)
        constraints.extend(aux_constraints)
        logger.info(f"  Added {len(aux_constraints)} auxiliary constraints")

        # [2] Stability LMI: S(P_i, L_i, nu) - λ₁ Ñ₁ - λ₂ Ñ₂ ≻ 0
        logger.info("\n[2/5] Building stability LMI...")
        stability_lmi = self.lmi_builder.build_stability_lmi(
            P_i=P_i,
            L_i=L_i,
            nu=nu,
            lambda1=lambda1,
            lambda2=lambda2,
            segment_idx=segment_idx,
            constants=constants,
            hankel=hankel,
            T=T,
        )
        constraints.append(stability_lmi)
        logger.info("  Added stability LMI constraint")

        # [3] Lower bound constraint: P_i ⪰ P_min_i
        logger.info("\n[3/5] Adding lower bound constraint...")
        constraints.append(P_i - P_min_i >> eps_psd * np.eye(self.n))
        logger.info(f"  P_i ⪰ P_min_i (with eps={eps_psd})")

        # [4] Control constraint: [R_max_i, L_i; L_i^T, P_i] ⪰ 0
        logger.info("\n[4/5] Adding control constraint...")
        control_lmi = cp.bmat(
            [
                [R_max_i, L_i],
                [L_i.T, P_i],
            ]
        )
        constraints.append(control_lmi >> eps_psd * np.eye(self.n + self.m))
        logger.info("  Added control constraint LMI")

        # [5] Predecessor containment: P_i ⪯ μ P_{i-1} (if not first segment)
        if P_prev is not None:
            logger.info("\n[5/5] Adding predecessor containment constraint...")
            constraints.append(mu * P_prev - P_i >> eps_psd * np.eye(self.n))
            logger.info(f"  P_i ⪯ {mu} P_{{i-1}}")
        else:
            logger.info("\n[5/5] Skipping predecessor constraint (first segment)")

        # Formulate problem
        logger.info("\nFormulating SDP problem...")
        logger.info(f"  Variables: P_i ({self.n}x{self.n}), L_i ({self.m}x{self.n}), nu, λ₁, λ₂")
        logger.info(f"  Total constraints: {len(constraints)}")

        problem = cp.Problem(objective, constraints)

        # Solve
        logger.info(f"\nSolving with {self.solver_name}...")

        try:
            if self.solver_name == "SCS":
                problem.solve(
                    solver=cp.SCS,
                    verbose=self.verbose,
                    eps=1e-5,
                    max_iters=5000,
                )
            elif self.solver_name == "MOSEK":
                problem.solve(
                    solver=cp.MOSEK,
                    verbose=self.verbose,
                )
            elif self.solver_name == "CVXOPT":
                problem.solve(
                    solver=cp.CVXOPT,
                    verbose=self.verbose,
                )
            else:
                problem.solve(verbose=self.verbose)

        except Exception as e:
            logger.error(f"Solver failed: {e}")
            raise ValueError(f"SDP optimization failed: {e}")

        # Check solution status
        logger.info(f"\nSolver status: {problem.status}")

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.error(f"Optimization failed with status: {problem.status}")
            raise ValueError(f"SDP infeasible: {problem.status}")

        # Extract solution
        P_i_opt = P_i.value
        L_i_opt = L_i.value
        nu_opt = nu.value
        lambda1_opt = lambda1.value
        lambda2_opt = lambda2.value

        # Ensure P_i is symmetric (numerical issues)
        P_i_opt = (P_i_opt + P_i_opt.T) / 2

        # Compute feedback gain: K_i = L_i P_i^{-1}
        try:
            P_i_inv = np.linalg.inv(P_i_opt)
            K_i_opt = L_i_opt @ P_i_inv
        except np.linalg.LinAlgError:
            logger.warning("P_i is singular, using pseudo-inverse")
            P_i_inv = np.linalg.pinv(P_i_opt)
            K_i_opt = L_i_opt @ P_i_inv

        # Compute volume
        det_P = np.linalg.det(P_i_opt)
        volume = np.sqrt(det_P) * self._volume_constant(self.n)

        # Create solution object
        solution = SDPSolution(
            P_i=P_i_opt,
            L_i=L_i_opt,
            K_i=K_i_opt,
            nu=float(nu_opt),
            lambda1=float(lambda1_opt),
            lambda2=float(lambda2_opt),
            objective_value=problem.value,
            volume=volume,
            status=problem.status,
            segment_idx=segment_idx,
        )

        # Log results
        logger.info("\n" + "=" * 70)
        logger.info("SDP SOLUTION")
        logger.info("=" * 70)
        logger.info(f"Status: {solution.status}")
        logger.info(f"Objective (log det P_i): {solution.objective_value:.6f}")
        logger.info(f"Volume: {solution.volume:.6e}")
        logger.info(f"nu: {solution.nu:.6e}")
        logger.info(f"λ₁: {solution.lambda1:.6e}")
        logger.info(f"λ₂: {solution.lambda2:.6e}")
        logger.info(f"P_i eigenvalues: {np.linalg.eigvalsh(P_i_opt)}")
        logger.info(f"K_i norm: {np.linalg.norm(K_i_opt):.6f}")
        logger.info("=" * 70)

        return solution

    def _volume_constant(self, n: int) -> float:
        """Compute volume constant for n-dimensional ellipsoid.

        Volume of ellipsoid in R^n: V = (π^{n/2} / Γ(n/2 + 1)) * sqrt(det(P))

        Args:
            n: Dimension

        Returns:
            Constant term π^{n/2} / Γ(n/2 + 1)
        """
        from scipy.special import gamma  # noqa: PLC0415

        return (np.pi ** (n / 2)) / gamma(n / 2 + 1)

    def solve_sequence(
        self,
        segment_indices: list[int],
        P_min_list: list[NDArray[np.float64]],
        R_max_list: list[NDArray[np.float64]],
        constants: UncertaintyConstants,
        hankel_list: list[SegmentHankelMatrices],
        T_list: list[int],
        mu: float = 1.1,
    ) -> list[SDPSolution]:
        """Solve SDP for a sequence of funnel segments.

        Solves segments sequentially, using each solution as the predecessor
        for the next segment (P_{i-1} → P_i).

        Args:
            segment_indices: List of segment indices
            P_min_list: List of minimum feasible matrices
            R_max_list: List of control constraint matrices
            constants: Uncertainty constants
            hankel_list: List of Hankel matrices (one per segment)
            T_list: List of segment lengths
            mu: Funnel growth bound

        Returns:
            List of SDPSolution objects
        """
        logger.info("=" * 70)
        logger.info(f"SOLVING SDP SEQUENCE FOR {len(segment_indices)} SEGMENTS")
        logger.info("=" * 70)

        num_segments = len(segment_indices)

        # Validate inputs
        if len(P_min_list) != num_segments:
            raise ValueError(f"P_min_list length {len(P_min_list)} != {num_segments}")
        if len(R_max_list) != num_segments:
            raise ValueError(f"R_max_list length {len(R_max_list)} != {num_segments}")
        if len(hankel_list) != num_segments:
            raise ValueError(f"hankel_list length {len(hankel_list)} != {num_segments}")
        if len(T_list) != num_segments:
            raise ValueError(f"T_list length {len(T_list)} != {num_segments}")

        solutions = []
        P_prev = None

        for i, seg_idx in enumerate(segment_indices):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"SEGMENT {i + 1}/{num_segments} (index={seg_idx})")
            logger.info(f"{'=' * 70}")

            solution = self.solve_segment(
                segment_idx=seg_idx,
                P_min_i=P_min_list[i],
                R_max_i=R_max_list[i],
                constants=constants,
                hankel=hankel_list[i],
                T=T_list[i],
                P_prev=P_prev,
                mu=mu,
            )

            solutions.append(solution)
            P_prev = solution.P_i  # Use this solution for next segment

            logger.info(f"\nSegment {seg_idx} complete")
            logger.info(f"  Volume: {solution.volume:.6e}")
            logger.info(f"  Status: {solution.status}")

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("SEQUENCE SOLUTION SUMMARY")
        logger.info("=" * 70)

        volumes = [sol.volume for sol in solutions]
        logger.info(f"Segments solved: {len(solutions)}")
        logger.info(f"All successful: {all(sol.status in ['optimal', 'optimal_inaccurate'] for sol in solutions)}")
        logger.info("\nVolume statistics:")
        logger.info(f"  Min:  {np.min(volumes):.6e}")
        logger.info(f"  Max:  {np.max(volumes):.6e}")
        logger.info(f"  Mean: {np.mean(volumes):.6e}")

        # Check volume progression
        volume_ratios = [volumes[i + 1] / volumes[i] for i in range(len(volumes) - 1)]
        if volume_ratios:
            logger.info(f"\nVolume ratios (V_{i + 1} / V_i):")
            logger.info(f"  Min:  {np.min(volume_ratios):.3f}")
            logger.info(f"  Max:  {np.max(volume_ratios):.3f}")
            logger.info(f"  Mean: {np.mean(volume_ratios):.3f}")

        logger.info("=" * 70)

        return solutions

    def validate_solution(
        self,
        solution: SDPSolution,
        P_min_i: NDArray[np.float64],
        R_max_i: NDArray[np.float64],
        P_prev: Optional[NDArray[np.float64]] = None,
        mu: float = 1.1,
        tol: float = 1e-6,
    ) -> bool:
        """Validate that solution satisfies all constraints.

        Args:
            solution: SDP solution to validate
            P_min_i: Minimum feasible matrix
            R_max_i: Control constraint matrix
            P_prev: Previous segment matrix (optional)
            mu: Funnel growth bound
            tol: Tolerance for constraint violation

        Returns:
            True if all constraints satisfied
        """
        logger.info(f"Validating solution for segment {solution.segment_idx}...")

        P_i = solution.P_i
        L_i = solution.L_i

        all_valid = True

        # [1] P_i ≻ 0
        eigvals = np.linalg.eigvalsh(P_i)
        if np.min(eigvals) <= tol:
            logger.warning(f"  ✗ P_i not positive definite (min eigval={np.min(eigvals):.6e})")
            all_valid = False
        else:
            logger.info(f"  ✓ P_i ≻ 0 (min eigval={np.min(eigvals):.6e})")

        # [2] P_i ⪰ P_min_i
        diff = P_i - P_min_i
        eigvals_diff = np.linalg.eigvalsh(diff)
        if np.min(eigvals_diff) < -tol:
            logger.warning(f"  ✗ P_i ⪰ P_min_i violated (min eigval of diff={np.min(eigvals_diff):.6e})")
            all_valid = False
        else:
            logger.info("  ✓ P_i ⪰ P_min_i")

        # [3] Control constraint: [R_max_i, L_i; L_i^T, P_i] ⪰ 0
        control_matrix = np.block(
            [
                [R_max_i, L_i],
                [L_i.T, P_i],
            ]
        )
        eigvals_control = np.linalg.eigvalsh(control_matrix)
        if np.min(eigvals_control) < -tol:
            logger.warning(f"  ✗ Control constraint violated (min eigval={np.min(eigvals_control):.6e})")
            all_valid = False
        else:
            logger.info("  ✓ Control constraint satisfied")

        # [4] P_i ⪯ μ P_{i-1} (if applicable)
        if P_prev is not None:
            diff_prev = mu * P_prev - P_i
            eigvals_prev = np.linalg.eigvalsh(diff_prev)
            if np.min(eigvals_prev) < -tol:
                logger.warning(f"  ✗ Predecessor constraint violated (min eigval={np.min(eigvals_prev):.6e})")
                all_valid = False
            else:
                logger.info("  ✓ P_i ⪯ μ P_{{i-1}}")

        if all_valid:
            logger.info(f"  ✓ All constraints satisfied for segment {solution.segment_idx}")
        else:
            logger.warning(f"  ✗ Some constraints violated for segment {solution.segment_idx}")

        return all_valid
