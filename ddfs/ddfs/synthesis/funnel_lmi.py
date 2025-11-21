"""
Funnel LMI Synthesis for Data-Driven Control

This module implements the main SDP optimization for funnel synthesis using
the LMI framework. It solves the optimization problem:

maximize log det(P_i)
subject to:
    P_i ≻ 0, λ₁ ≥ 0, λ₂ ≥ 0, nu > 0
    S(P_i, L_i, nu) - λ₁Ñ₁ - λ₂Ñ₂ ⪰ 0
    P_i ⪰ P_min,i
    [R_max,i  L_i^T] ⪰ 0
    [L_i      P_i  ]
    P_{i+1} ⪯ μP_i  (optional coupling)

Variables:
    P_i ∈ ℝⁿˣⁿ: State Lyapunov matrix
    L_i ∈ ℝᵐˣⁿ: Controller parameterization (K_i = L_i P_i⁻¹)
    λ₁ ≥ 0: Data informativity multiplier
    λ₂ ≥ 0: Uncertainty bound multiplier
    nu > 0: Slack variable
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

from ddfs.synthesis.lmi_matrices import LMIMatrixConstructor

logger = logging.getLogger(__name__)


@dataclass
class FunnelSynthesisConfig:
    """
    Configuration for funnel synthesis optimization.

    Attributes:
        alpha: Lyapunov decay rate (0 < alpha < 1), typically 0.95-0.99
        nu_lower_bound: Lower bound for nu slack variable (> 0), typically 1e-6
        mu: Coupling parameter for P_{i+1} << mu * P_i, typically > 1
        solver: CVXPY solver to use ('SCS', 'MOSEK', 'CVXOPT')
        verbose: Whether to print solver output
        enable_coupling: Whether to enforce P_{i+1} ⪯ μP_i coupling
        max_iters: Maximum solver iterations
        eps: Solver convergence tolerance
    """

    alpha: float = 0.97
    nu_lower_bound: float = 1e-6
    mu: float = 1.2
    solver: str = "SCS"
    verbose: bool = True
    enable_coupling: bool = False
    max_iters: int = 10000
    eps: float = 1e-6


@dataclass
class SegmentData:
    """
    Data for a single segment needed for funnel synthesis.

    Attributes:
        segment_idx: Segment index
        H_prev: Hankel matrix H_{i-1} (nxL)
        H_plus_prev: Hankel matrix H⁺_{i-1} (nxL)
        Xi_prev: Control Hankel matrix Ξ_{i-1} (mxL)
        beta_prev: Informativity constant β_{i-1}
        C: Increment bound constant
        T_segment: Segment duration
        P_min: Minimum Lyapunov matrix P_min,i (nxn)
        R_max: Maximum control effort matrix R_max,i (mxm)
    """

    segment_idx: int
    H_prev: np.ndarray
    H_plus_prev: np.ndarray
    Xi_prev: np.ndarray
    beta_prev: float
    C: float
    T_segment: float
    P_min: np.ndarray
    R_max: np.ndarray


@dataclass
class FunnelSynthesisResult:
    """
    Result from funnel synthesis for a single segment.

    Attributes:
        segment_idx: Segment index
        P: Optimal Lyapunov matrix P_i (nxn)
        K: Optimal controller gain K_i = L_i P_i⁻¹ (mxn)
        L: Optimal controller parameterization L_i (mxn)
        lambda1: Optimal data multiplier λ₁
        lambda2: Optimal uncertainty multiplier λ₂
        nu: Optimal slack variable nu
        objective_value: log det(P_i)
        success: Whether optimization succeeded
        solver_status: CVXPY solver status
        solve_time: Time to solve (seconds)
    """

    segment_idx: int
    P: np.ndarray
    K: np.ndarray
    L: np.ndarray
    lambda1: float
    lambda2: float
    nu: float
    objective_value: float
    success: bool
    solver_status: str
    solve_time: float


class FunnelLMISynthesizer:
    """
    Main class for LMI-based funnel synthesis.

    This class formulates and solves the SDP optimization problem for each
    segment to synthesize robust funnels with data-driven guarantees.
    """

    def __init__(
        self, n_states: int, n_controls: int, data_length: int, config: Optional[FunnelSynthesisConfig] = None
    ):
        """
        Initialize funnel LMI synthesizer.

        Args:
            n_states: State dimension (n)
            n_controls: Control dimension (m)
            data_length: Data trajectory length (L)
            config: Synthesis configuration (uses defaults if None)
        """
        self.n = n_states
        self.m = n_controls
        self.L = data_length

        self.config = config if config is not None else FunnelSynthesisConfig()

        # Initialize matrix constructor
        self.matrix_constructor = LMIMatrixConstructor(n_states, n_controls, data_length)

        # Storage for results
        self.results: List[FunnelSynthesisResult] = []

        logger.info(f"Initialized FunnelLMISynthesizer: n={self.n}, m={self.m}, L={self.L}")
        logger.info(f"Config: alpha={self.config.alpha}, nu_lb={self.config.nu_lower_bound},"
         "solver={self.config.solver}")

    def synthesize_segment(  # noqa: PLR0915
        self, segment_data: SegmentData, P_prev: Optional[np.ndarray] = None
    ) -> FunnelSynthesisResult:
        """
        Synthesize funnel for a single segment by solving the SDP.

        Args:
            segment_data: Data for this segment
            P_prev: Previous segment's P matrix (for coupling constraint)

        Returns:
            FunnelSynthesisResult with optimal P_i, K_i, and solver info
        """
        logger.info(f"Synthesizing segment {segment_data.segment_idx}...")

        # Define optimization variables
        P_i = cp.Variable((self.n, self.n), symmetric=True)
        L_i = cp.Variable((self.m, self.n))
        lambda1 = cp.Variable(nonneg=True)
        lambda2 = cp.Variable(nonneg=True)
        nu = cp.Variable(pos=True)  # nu is a positive variable

        # Construct LMI matrices using matrix constructor
        S, N1_tilde, N2_tilde = self.matrix_constructor.construct_all_lmi_matrices(
            P_i=P_i,
            L_i=L_i,
            nu=nu,  # Pass nu as variable
            alpha=self.config.alpha,
            H_prev=segment_data.H_prev,
            H_plus_prev=segment_data.H_plus_prev,
            Xi_prev=segment_data.Xi_prev,
            beta_prev=segment_data.beta_prev,
            C=segment_data.C,
            T_segment=segment_data.T_segment,
        )

        # Define objective: maximize log det(P_i)
        objective = cp.Maximize(cp.log_det(P_i))

        # Define constraints
        constraints = []

        # 1. P_i ≻ 0 (positive definite) - enforced by log_det
        constraints.append(P_i >> 0)

        # 2. nu > 0 (positive) with lower bound
        constraints.append(nu >= self.config.nu_lower_bound)

        # 3. Robust stability LMI: S - λ₁Ñ₁ - λ₂Ñ₂ ⪰ 0
        constraints.append(S - lambda1 * N1_tilde - lambda2 * N2_tilde >> 0)

        # 4. State feasibility: P_i ⪰ P_min,i
        constraints.append(P_i >> segment_data.P_min)

        # 5. Input feasibility (Schur complement): [R_max  L_i^T] ⪰ 0
        #                                            [L_i    P_i  ]
        input_constraint_matrix = cp.bmat([[segment_data.R_max, L_i.T], [L_i, P_i]])
        constraints.append(input_constraint_matrix >> 0)

        # 6. Optional coupling: P_i ⪯ μP_{i-1}
        if self.config.enable_coupling and P_prev is not None:
            constraints.append(P_i << self.config.mu * P_prev)
            logger.debug(
                f"Added coupling constraint: P_{segment_data.segment_idx} << "
                f"{self.config.mu} * P_{segment_data.segment_idx - 1}"
            )

        # Formulate problem
        problem = cp.Problem(objective, constraints)

        # Solve
        logger.info(f"Solving SDP for segment {segment_data.segment_idx}...")

        try:
            # Select solver and options
            solver_opts = {"max_iters": self.config.max_iters, "eps": self.config.eps, "verbose": self.config.verbose}

            if self.config.solver == "SCS":
                problem.solve(solver=cp.SCS, **solver_opts)
            elif self.config.solver == "MOSEK":
                problem.solve(solver=cp.MOSEK, verbose=self.config.verbose)
            elif self.config.solver == "CVXOPT":
                problem.solve(solver=cp.CVXOPT, verbose=self.config.verbose)
            else:
                problem.solve(solver=cp.SCS, **solver_opts)

            # Check if solved successfully
            success = problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]

            if success:
                # Extract solution
                P_opt = P_i.value
                L_opt = L_i.value
                lambda1_opt = lambda1.value
                lambda2_opt = lambda2.value
                nu_opt = nu.value

                # Compute controller gain: K_i = L_i P_i^{-1}
                try:
                    K_opt = L_opt @ np.linalg.inv(P_opt)
                except np.linalg.LinAlgError:
                    logger.warning(f"Segment {segment_data.segment_idx}: P_i is singular, using pseudo-inverse")
                    K_opt = L_opt @ np.linalg.pinv(P_opt)

                logger.info(
                    f"Segment {segment_data.segment_idx} SUCCESS: "
                    f"obj={problem.value:.4f}, λ₁={lambda1_opt:.4e}, λ₂={lambda2_opt:.4e}, nu={nu_opt:.4e}"
                )

                result = FunnelSynthesisResult(
                    segment_idx=segment_data.segment_idx,
                    P=P_opt,
                    K=K_opt,
                    L=L_opt,
                    lambda1=lambda1_opt,
                    lambda2=lambda2_opt,
                    nu=nu_opt,
                    objective_value=problem.value,
                    success=True,
                    solver_status=problem.status,
                    solve_time=problem.solver_stats.solve_time if problem.solver_stats else 0.0,
                )
            else:
                logger.warning(f"Segment {segment_data.segment_idx} FAILED: status={problem.status}")

                result = FunnelSynthesisResult(
                    segment_idx=segment_data.segment_idx,
                    P=np.zeros((self.n, self.n)),
                    K=np.zeros((self.m, self.n)),
                    L=np.zeros((self.m, self.n)),
                    lambda1=0.0,
                    lambda2=0.0,
                    nu=0.0,
                    objective_value=-np.inf,
                    success=False,
                    solver_status=problem.status,
                    solve_time=problem.solver_stats.solve_time if problem.solver_stats else 0.0,
                )

        except Exception as e:
            logger.error(f"Segment {segment_data.segment_idx} ERROR: {e}")

            result = FunnelSynthesisResult(
                segment_idx=segment_data.segment_idx,
                P=np.zeros((self.n, self.n)),
                K=np.zeros((self.m, self.n)),
                L=np.zeros((self.m, self.n)),
                lambda1=0.0,
                lambda2=0.0,
                nu=0.0,
                objective_value=-np.inf,
                success=False,
                solver_status="ERROR",
                solve_time=0.0,
            )

        return result

    def synthesize_all_segments(
        self, segments_data: List[SegmentData], use_coupling: bool = False
    ) -> List[FunnelSynthesisResult]:
        """
        Synthesize funnels for all segments.

        Args:
            segments_data: List of SegmentData for each segment
            use_coupling: Whether to enforce P_{i+1} ⪯ μP_i coupling

        Returns:
            List of FunnelSynthesisResult for each segment
        """
        logger.info(f"Synthesizing {len(segments_data)} segments...")

        # Enable/disable coupling
        original_coupling = self.config.enable_coupling
        self.config.enable_coupling = use_coupling

        results = []
        P_prev = None

        for segment_data in segments_data:
            result = self.synthesize_segment(segment_data, P_prev=P_prev)
            results.append(result)

            # Update P_prev for next segment if coupling enabled
            if use_coupling and result.success:
                P_prev = result.P

        # Restore original coupling setting
        self.config.enable_coupling = original_coupling

        # Store results
        self.results = results

        # Summary
        n_success = sum(1 for r in results if r.success)
        logger.info(f"Synthesis complete: {n_success}/{len(segments_data)} segments successful")

        return results

    def verify_constraints(
        self, result: FunnelSynthesisResult, segment_data: SegmentData, tolerance: float = 1e-6
    ) -> Dict[str, bool]:
        """
        Verify that a solution satisfies all constraints.

        Args:
            result: Synthesis result to verify
            segment_data: Original segment data
            tolerance: Numerical tolerance for verification

        Returns:
            Dictionary of constraint satisfaction flags
        """
        verification = {}

        if not result.success:
            logger.warning(f"Cannot verify failed segment {result.segment_idx}")
            return {"all_satisfied": False}

        P = result.P
        L = result.L
        K = result.K
        nu = result.nu

        # 1. Check P ≻ 0
        eigs_P = np.linalg.eigvalsh(P)
        verification["P_positive_definite"] = np.all(eigs_P > tolerance)

        # 2. Check nu > 0
        verification["nu_positive"] = nu > tolerance

        # 3. Check P ⪰ P_min
        eigs_diff = np.linalg.eigvalsh(P - segment_data.P_min)
        verification["P_exceeds_P_min"] = np.all(eigs_diff > -tolerance)

        # 4. Check input constraint (Schur complement)
        # [R_max  L^T] ⪰ 0  ⟺  R_max - L^T P^{-1} L ⪰ 0
        # [L      P  ]
        try:
            input_schur = segment_data.R_max - L.T @ np.linalg.solve(P, L)
            eigs_input = np.linalg.eigvalsh(input_schur)
            verification["input_constraint_satisfied"] = np.all(eigs_input > -tolerance)
        except np.linalg.LinAlgError:
            verification["input_constraint_satisfied"] = False

        # 5. Check K = L P^{-1}
        try:
            K_computed = L @ np.linalg.inv(P)
            verification["K_consistent"] = np.allclose(K, K_computed, atol=tolerance)
        except np.linalg.LinAlgError:
            verification["K_consistent"] = False

        verification["all_satisfied"] = all(verification.values())

        if verification["all_satisfied"]:
            logger.info(f"Segment {result.segment_idx}: All constraints verified ✓")
        else:
            logger.warning(f"Segment {result.segment_idx}: Constraint verification failed!")
            for constraint, satisfied in verification.items():
                if not satisfied and constraint != "all_satisfied":
                    logger.warning(f"  {constraint}: FAILED")

        return verification

    def extract_funnel_sequence(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract the sequence of funnels (P_i) and controllers (K_i).

        Returns:
            Tuple of (funnels, controllers) where:
                funnels: List of P_i matrices
                controllers: List of K_i matrices
        """
        if not self.results:
            logger.warning("No results available. Run synthesis first.")
            return [], []

        funnels = [r.P for r in self.results if r.success]
        controllers = [r.K for r in self.results if r.success]

        logger.info(f"Extracted {len(funnels)} funnels and {len(controllers)} controllers")

        return funnels, controllers

    def get_synthesis_summary(self) -> Dict:
        """
        Get summary statistics of synthesis results.

        Returns:
            Dictionary of summary statistics
        """
        if not self.results:
            return {"n_segments": 0, "n_success": 0, "success_rate": 0.0}

        n_segments = len(self.results)
        n_success = sum(1 for r in self.results if r.success)
        success_rate = n_success / n_segments

        successful_results = [r for r in self.results if r.success]

        summary = {
            "n_segments": n_segments,
            "n_success": n_success,
            "n_failed": n_segments - n_success,
            "success_rate": success_rate,
            "total_solve_time": sum(r.solve_time for r in self.results),
            "avg_solve_time": np.mean([r.solve_time for r in self.results]),
        }

        if successful_results:
            summary.update(
                {
                    "avg_objective": np.mean([r.objective_value for r in successful_results]),
                    "avg_lambda1": np.mean([r.lambda1 for r in successful_results]),
                    "avg_lambda2": np.mean([r.lambda2 for r in successful_results]),
                    "avg_nu": np.mean([r.nu for r in successful_results]),
                    "avg_P_trace": np.mean([np.trace(r.P) for r in successful_results]),
                    "avg_K_norm": np.mean([np.linalg.norm(r.K) for r in successful_results]),
                }
            )

        return summary


def create_synthesizer(
    n_states: int,
    n_controls: int,
    data_length: int,
    alpha: float = 0.97,
    nu_lower_bound: float = 1e-6,
    solver: str = "SCS",
    verbose: bool = True,
) -> FunnelLMISynthesizer:
    """
    Factory function to create a funnel synthesizer with custom config.

    Args:
        n_states: State dimension
        n_controls: Control dimension
        data_length: Data length
        alpha: Lyapunov decay rate
        nu_lower_bound: Lower bound for nu
        solver: CVXPY solver name
        verbose: Solver verbosity

    Returns:
        Configured FunnelLMISynthesizer
    """
    config = FunnelSynthesisConfig(alpha=alpha, nu_lower_bound=nu_lower_bound, solver=solver, verbose=verbose)

    return FunnelLMISynthesizer(n_states, n_controls, data_length, config)
