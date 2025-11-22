"""LMI (Linear Matrix Inequality) builder for Phase 5 funnel synthesis.

This module constructs the LMI constraints for robust funnel synthesis,
including the main stability LMI with uncertainty bounds from Phase 3
and data-driven terms from Phase 2.
"""

import logging

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from ddfs.data_collection.hankel import SegmentHankelMatrices
from ddfs.uncertainty.constants import UncertaintyConstants

logger = logging.getLogger(__name__)


class LMIBuilder:
    """Builds LMI constraints for funnel synthesis.

    Constructs the main stability LMI:
        S(P_i, L_i, nu) - lambda1 * N1_tilde - lambda2 * N2_tilde ≻ 0

    where:
    - P_i ∈ R^(n, n): Shape matrix (symmetric PSD)
    - L_i ∈ R^(m, n): Control gain times P_i (L_i = K_i P_i)
    - nu: Lyapunov decrease rate
    - λ₁, λ₂: S-procedure multipliers
    - alpha: Lyapunov decay rate (fixed scalar)

    The LMI ensures:
    1. Lyapunov stability with decay rate alpha
    2. Data-driven uncertainty bounds (Ñ₁ term)
    3. Linearization uncertainty bounds (Ñ₂ term)

    Matrix dimensions:
    - S: (4n+2m) x (4n+2m)
    - Ñ₁, Ñ₂: (4n+2m) x (4n+2m) (padded from (3n+2m) x (3n+2m))

    Example:
        >>> from ddfs.synthesis import LMIBuilder
        >>> from ddfs.uncertainty import UncertaintyConstants
        >>>
        >>> # Setup
        >>> n, m = 3, 2
        >>> constants = UncertaintyConstants(...)
        >>> hankel = SegmentHankelMatrices(...)
        >>>
        >>> # Build LMI
        >>> builder = LMIBuilder(n, m, alpha=0.95)
        >>> lmi = builder.build_stability_lmi(
        ...     P_i=P_var,
        ...     L_i=L_var,
        ...     nu=nu_var,
        ...     lambda1=lambda1_var,
        ...     lambda2=lambda2_var,
        ...     segment_idx=0,
        ...     constants=constants,
        ...     hankel=hankel,
        ... )
    """

    def __init__(
        self,
        n: int,
        m: int,
        alpha: float = 0.95,
    ):
        """Initialize LMI builder.

        Args:
            n: State dimension
            m: Control dimension
            alpha: Lyapunov decay rate (0 < alpha < 1)
        """
        self.n = n
        self.m = m
        self.alpha = alpha

        # Matrix dimensions
        self.S_dim = 4 * n + 2 * m
        self.N_dim = 3 * n + 2 * m
        self.N_tilde_dim = 4 * n + 2 * m  # Padded

        logger.info(f"Initialized LMIBuilder (n={n}, m={m}, alpha={alpha})")
        logger.info(f"  S dimension: {self.S_dim}x{self.S_dim}")
        logger.info(f"  N dimension (before padding): {self.N_dim}x{self.N_dim}")
        logger.info(f"  Ñ dimension (after padding): {self.N_tilde_dim}x{self.N_tilde_dim}")

    def build_S_matrix(
        self,
        P_i: cp.Variable,
        L_i: cp.Variable,
        nu: cp.Variable,
    ) -> cp.Expression:
        """Build S(P_i, L_i, nu) matrix.

        S has block structure (4n+2m) x (4n+2m):

        [alpha * P_i - nu * I,    0,      0,      0,      0,      0   ]
        [0,        -P_i,  -L_i^T,  -P_i,  -L_i^T,   0   ]
        [0,        -L_i,    0,     -L_i,    0,     L_i  ]
        [0,        -P_i,  -L_i^T,  -P_i,  -L_i^T,   0   ]
        [0,        -L_i,    0,     -L_i,    0,     L_i  ]
        [0,         0,    L_i^T,    0,    L_i^T,   P_i  ]

        Block dimensions:
        - Row/Col 1: n, n
        - Row/Col 2: n, n
        - Row/Col 3: m, m
        - Row/Col 4: n, n
        - Row/Col 5: m, m
        - Row/Col 6: n, n

        Args:
            P_i: Shape matrix (n, n, symmetric PSD)
            L_i: Control gain matrix (m, n)
            nu: Lyapunov decrease rate (scalar)

        Returns:
            S matrix as CVXPY expression
        """
        n, m = self.n, self.m

        # Create zero blocks
        zero_nn = np.zeros((n, n))
        zero_nm = np.zeros((n, m))
        zero_mn = np.zeros((m, n))
        zero_mm = np.zeros((m, m))

        # Build S matrix using cp.bmat
        S = cp.bmat(
            [
                # Row 1: [alpha * P_i - nu * I, 0, 0, 0, 0, 0]
                [self.alpha * P_i - nu * np.eye(n), zero_nn, zero_nm, zero_nn, zero_nm, zero_nn],
                # Row 2: [0, -P_i, -L_i^T, -P_i, -L_i^T, 0]
                [zero_nn, -P_i, -L_i.T, -P_i, -L_i.T, zero_nn],
                # Row 3: [0, -L_i, 0, -L_i, 0, L_i]
                [zero_mn, -L_i, zero_mm, -L_i, zero_mm, L_i],
                # Row 4: [0, -P_i, -L_i^T, -P_i, -L_i^T, 0]
                [zero_nn, -P_i, -L_i.T, -P_i, -L_i.T, zero_nn],
                # Row 5: [0, -L_i, 0, -L_i, 0, L_i]
                [zero_mn, -L_i, zero_mm, -L_i, zero_mm, L_i],
                # Row 6: [0, 0, L_i^T, 0, L_i^T, P_i]
                [zero_nn, zero_nn, L_i.T, zero_nn, L_i.T, P_i],
            ]
        )

        return S

    def build_N1_tilde(
        self,
        segment_idx: int,
        constants: UncertaintyConstants,
        hankel: SegmentHankelMatrices,
    ) -> NDArray[np.float64]:
        """Build Ñ₁ matrix for data-driven uncertainty.

        N₁ = M @ D @ M^T

        where:
        M = [I_n,  H⁺     ]    (3n+2m) x (n+L)
            [0,   -H      ]
            [0,   -Ξ      ]
            [0,    0      ]
            [0,    0      ]

        D = [β I_n,  0    ]    (n+L) x (n+L)
            [0,     -I_L  ]

        Then pad to get Ñ₁:
        Ñ₁ = [N₁,  0 ]    (4n+2m) x (4n+2m)
             [0,   0 ]

        Args:
            segment_idx: Segment index
            constants: Uncertainty constants (for β_i)
            hankel: Hankel matrices (for H, H⁺, Ξ)

        Returns:
            Ñ₁ as numpy array (4n+2m) x (4n+2m)
        """
        n, m = self.n, self.m

        # Get data from hankel
        H = hankel.H  # (n, L)
        H_plus = hankel.H_plus  # (n, L)
        Xi = hankel.Xi  # (m, L)
        L = hankel.L

        # Get β for this segment
        beta_i = constants.get_beta(segment_idx)

        logger.debug(f"Building N₁ for segment {segment_idx}")
        logger.debug(f"  H shape: {H.shape}, H⁺ shape: {H_plus.shape}, Ξ shape: {Xi.shape}")
        logger.debug(f"  β_{segment_idx} = {beta_i:.6e}")

        # Build M matrix: (3n+2m) x (n+L)
        M = np.block(
            [
                [np.eye(n), H_plus],  # n x (n+L)
                [np.zeros((n, n)), -H],  # n x (n+L)
                [np.zeros((m, n)), -Xi],  # m x (n+L)
                [np.zeros((n, n + L))],  # n x (n+L)
                [np.zeros((m, n + L))],  # m x (n+L)
            ]
        )

        # Build D matrix: (n+L) x (n+L)
        D = np.block(
            [
                [beta_i * np.eye(n), np.zeros((n, L))],
                [np.zeros((L, n)), -np.eye(L)],
            ]
        )

        # Compute N₁ = M @ D @ M^T
        N1 = M @ D @ M.T  # (3n+2m) x (3n+2m)

        # Pad with zeros to get Ñ₁: (4n+2m) x (4n+2m)
        N1_tilde = np.block(
            [
                [N1, np.zeros((self.N_dim, n))],
                [np.zeros((n, self.N_dim)), np.zeros((n, n))],
            ]
        )

        logger.debug(f"  N₁ shape: {N1.shape}")
        logger.debug(f"  Ñ₁ shape: {N1_tilde.shape}")

        return N1_tilde

    def build_N2_tilde(
        self,
        T: int,
        constants: UncertaintyConstants,
    ) -> NDArray[np.float64]:
        """Build Ñ₂ matrix for linearization uncertainty.

        N₂ = M @ D @ M^T

        where:
        M = [I_n,  0,  0 ]    (3n+2m) x (2n+m)
            [0,    0,  0 ]
            [0,    0,  0 ]
            [0,   I_n, 0 ]
            [0,    0, I_m]

        D = [C² T̃² I_n,  0,    0  ]    (2n+m) x (2n+m)
            [0,         -I_n,  0  ]
            [0,          0,  -I_m ]

        where T̃ = 2T - 1 and C is from constants.

        Then pad to get Ñ₂:
        Ñ₂ = [N₂,  0 ]    (4n+2m) x (4n+2m)
             [0,   0 ]

        Args:
            T: Segment length
            constants: Uncertainty constants (for C)

        Returns:
            Ñ₂ as numpy array (4n+2m) x (4n+2m)
        """
        n, m = self.n, self.m

        # Compute T̃ = 2T - 1
        T_tilde = 2 * T - 1

        # Get C from constants
        C = constants.C

        logger.debug("Building N₂")
        logger.debug(f"  T = {T}, T̃ = {T_tilde}")
        logger.debug(f"  C = {C:.6e}")

        # Build M matrix: (3n+2m) x (2n+m)
        M = np.block(
            [
                [np.eye(n), np.zeros((n, n)), np.zeros((n, m))],  # n x (2n+m)
                [np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, m))],  # n x (2n+m)
                [np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, m))],  # m x (2n+m)
                [np.zeros((n, n)), np.eye(n), np.zeros((n, m))],  # n x (2n+m)
                [np.zeros((m, n)), np.zeros((m, n)), np.eye(m)],  # m x (2n+m)
            ]
        )

        # Build D matrix: (2n+m) x (2n+m)
        D = np.block(
            [
                [C**2 * T_tilde**2 * np.eye(n), np.zeros((n, n)), np.zeros((n, m))],
                [np.zeros((n, n)), -np.eye(n), np.zeros((n, m))],
                [np.zeros((m, n)), np.zeros((m, n)), -np.eye(m)],
            ]
        )

        # Compute N₂ = M @ D @ M^T
        N2 = M @ D @ M.T  # (3n+2m) x (3n+2m)

        # Pad with zeros to get Ñ₂: (4n+2m) x (4n+2m)
        N2_tilde = np.block(
            [
                [N2, np.zeros((self.N_dim, n))],
                [np.zeros((n, self.N_dim)), np.zeros((n, n))],
            ]
        )

        logger.debug(f"  N₂ shape: {N2.shape}")
        logger.debug(f"  Ñ₂ shape: {N2_tilde.shape}")

        return N2_tilde

    def build_stability_lmi(
        self,
        P_i: cp.Variable,
        L_i: cp.Variable,
        nu: cp.Variable,
        lambda1: cp.Variable,
        lambda2: cp.Variable,
        segment_idx: int,
        constants: UncertaintyConstants,
        hankel: SegmentHankelMatrices,
        T: int,
    ) -> cp.Constraint:
        """Build the main stability LMI constraint.

        S(P_i, L_i, nu) - λ₁ Ñ₁ - λ₂ Ñ₂ ≻ 0

        This ensures:
        1. Lyapunov stability with decay rate self.alpha
        2. Robustness to data-driven uncertainty (Ñ₁)
        3. Robustness to linearization uncertainty (Ñ₂)

        Args:
            P_i: Shape matrix variable (n, n)
            L_i: Control gain matrix variable (m, n)
            nu: S-lemma multiplier
            lambda1: S-procedure multiplier for Ñ₁ (scalar)
            lambda2: S-procedure multiplier for Ñ₂ (scalar)
            segment_idx: Segment index
            constants: Uncertainty constants
            hankel: Hankel matrices for this segment
            T: Segment length

        Returns:
            LMI constraint: S - λ₁ Ñ₁ - λ₂ Ñ₂ ≻ 0
        """
        logger.info(f"Building stability LMI for segment {segment_idx}")

        # Build S matrix
        S = self.build_S_matrix(P_i, L_i, nu)

        # Build Ñ₁ matrix (data-driven uncertainty)
        N1_tilde = self.build_N1_tilde(segment_idx, constants, hankel)

        # Build Ñ₂ matrix (linearization uncertainty)
        N2_tilde = self.build_N2_tilde(T, constants)

        # Build LMI: S - λ₁ Ñ₁ - λ₂ Ñ₂ ≻ 0
        lmi_matrix = S - lambda1 * N1_tilde - lambda2 * N2_tilde

        # Return PSD constraint (≻ 0 means >> 0 in CVXPY)
        constraint = lmi_matrix >> 0

        logger.info(f"  LMI dimension: {self.S_dim}x{self.S_dim}")
        logger.info(f"  Built LMI for segment {segment_idx}")

        return constraint

    def build_auxiliary_constraints(
        self,
        P_i: cp.Variable,
        lambda1: cp.Variable,
        lambda2: cp.Variable,
        nu: cp.Variable,
    ) -> list[cp.Constraint]:
        """Build auxiliary constraints for the optimization.

        Additional constraints needed:
        1. P_i ≻ 0 (positive definite)
        2. λ₁ ≥ 0 (S-procedure multiplier)
        3. λ₂ ≥ 0 (S-procedure multiplier)
        4. nu ≥ 0

        Args:
            P_i: Shape matrix variable
            lambda1: S-procedure multiplier 1
            lambda2: S-procedure multiplier 2
            nu

        Returns:
            List of constraints
        """
        constraints = []

        # P_i must be positive definite
        constraints.append(P_i >> 0)

        # S-procedure multipliers must be non-negative
        constraints.append(lambda1 >= 0)
        constraints.append(lambda2 >= 0)

        # Lyapunov decrease rate must be non-negative
        constraints.append(nu >= 0)

        logger.debug("Built auxiliary constraints")

        return constraints

    def validate_dimensions(
        self,
        P_i: cp.Variable,
        L_i: cp.Variable,
    ) -> bool:
        """Validate that variable dimensions match expectations.

        Args:
            P_i: Shape matrix (should be n, n)
            L_i: Control gain matrix (should be m, n)

        Returns:
            True if dimensions are correct

        Raises:
            ValueError: If dimensions don't match
        """
        if P_i.shape != (self.n, self.n):
            raise ValueError(f"P_i has shape {P_i.shape}, expected ({self.n}, {self.n})")

        if L_i.shape != (self.m, self.n):
            raise ValueError(f"L_i has shape {L_i.shape}, expected ({self.m}, {self.n})")

        logger.debug("Variable dimensions validated")
        return True
