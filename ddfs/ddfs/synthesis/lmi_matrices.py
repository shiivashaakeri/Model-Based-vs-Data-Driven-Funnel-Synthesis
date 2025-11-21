"""
LMI Matrix Construction for Data-Driven Funnel Synthesis

This module implements the core matrix construction functions for the LMI-based
funnel synthesis optimization problem. All matrices are constructed according to
the paper's specifications with proper dimensions and numerical stability considerations.

Key matrices:
- S: (4n+2m) x (4n+2m) stability matrix
- N₁: (3n+2m) x (3n+2m) data informativity constraint
- N₂: (3n+2m) x (3n+2m) uncertainty bound constraint
- Ñ₁, Ñ₂: (4n+2m) x (4n+2m) padded versions

Note: This module constructs matrices where nu can be a CVXPY variable.
The SDP optimization problem with variables (P_i, L_i, λ₁, λ₂, nu) is defined in funnel_lmi.py.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Union

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LMIMatrixDimensions:
    """
    Tracks all relevant dimensions for LMI matrix construction.

    Attributes:
        n: State dimension
        m: Control input dimension
        L: Data length (Hankel matrix columns)
        S_dim: S-matrix dimension (4n+2m)
        N_dim: N₁, N₂ dimension (3n+2m)
    """

    n: int  # State dimension
    m: int  # Control dimension
    L: int  # Data length

    @property
    def S_dim(self) -> int:
        """S-matrix dimension: 4n+2m"""
        return 4 * self.n + 2 * self.m

    @property
    def N_dim(self) -> int:
        """N₁, N₂ matrix dimension: 3n+2m"""
        return 3 * self.n + 2 * self.m

    def validate(self):
        """Validate dimension consistency"""
        assert self.n > 0, "State dimension must be positive"
        assert self.m > 0, "Control dimension must be positive"
        assert self.L > 0, "Data length must be positive"
        assert self.n + self.m <= self.L, "Data length should be sufficient for informativity"


class LMIMatrixConstructor:
    """
    Constructs all LMI matrices needed for funnel synthesis optimization.

    This class handles the construction of:
    - S(P_i, L_i, nu): Stability matrix
    - N₁: Data informativity constraint (from Hankel matrices)
    - N₂: Uncertainty bound constraint
    - Ñ₁, Ñ₂: Padded versions for dimensional compatibility

    The actual SDP optimization problem is defined in funnel_lmi.py with variables:
    - P_i ∈ ℝⁿˣⁿ: State Lyapunov matrix (symmetric PSD)
    - L_i ∈ ℝᵐˣⁿ: Controller parameterization (K_i = L_i P_i⁻¹)
    - λ₁ ≥ 0: Data informativity multiplier
    - λ₂ ≥ 0: Uncertainty bound multiplier
    - nu > 0: Slack variable
    """

    def __init__(self, n_states: int, n_controls: int, data_length: int):
        """
        Initialize LMI matrix constructor.

        Args:
            n_states: State dimension (n)
            n_controls: Control dimension (m)
            data_length: Length of data trajectory (L)
        """
        self.dims = LMIMatrixDimensions(n=n_states, m=n_controls, L=data_length)
        self.dims.validate()

        logger.info(f"Initialized LMI constructor: n={self.dims.n}, m={self.dims.m}, L={self.dims.L}")
        logger.info(f"Matrix dimensions: S={self.dims.S_dim}x{self.dims.S_dim}, N={self.dims.N_dim}x{self.dims.N_dim}")

    def construct_S_matrix(
        self, P_i: cp.Variable, L_i: cp.Variable, nu: Union[cp.Variable, cp.Parameter, float], alpha: float
    ) -> cp.Expression:
        """
        Construct the S stability matrix.

        S(P_i, L_i, nu) = [
            alpha * P_i - nu * I_n    0         0        0         0        0
            0          -P_i     -L_i^T    -P_i     -L_i^T     0
            0          -L_i       0       -L_i       0       L_i
            0          -P_i     -L_i^T    -P_i     -L_i^T     0
            0          -L_i       0       -L_i       0       L_i
            0           0       L_i^T      0       L_i^T     P_i
        ]

        Dimension: (4n+2m) x (4n+2m)

        Args:
            P_i: State Lyapunov matrix (nxn, symmetric, PSD)
            L_i: Controller parameterization matrix (mxn)
            nu: Slack parameter (> 0) - can be Variable, Parameter, or float
            alpha: Lyapunov decay rate (0 < alpha < 1)

        Returns:
            S matrix as CVXPY expression
        """
        n, m = self.dims.n, self.dims.m

        # Validate alpha
        assert 0 < alpha < 1, f"Alpha must be in (0,1), got {alpha}"

        # Build blocks
        # Block (0,0): alpha * P_i - nu * I_n
        S_00 = alpha * P_i - nu * np.eye(n)

        # Block (1,1): -P_i
        S_11 = -P_i

        # Block (1,2): -L_i^T (nxm)
        S_12 = -L_i.T

        # Block (1,3): -P_i
        S_13 = -P_i

        # Block (1,4): -L_i^T
        S_14 = -L_i.T

        # Block (2,1): -L_i (mxn)
        S_21 = -L_i

        # Block (2,3): -L_i
        S_23 = -L_i

        # Block (2,5): L_i
        S_25 = L_i

        # Block (3,1): -P_i
        S_31 = -P_i

        # Block (3,2): -L_i^T
        S_32 = -L_i.T

        # Block (3,3): -P_i
        S_33 = -P_i

        # Block (3,4): -L_i^T
        S_34 = -L_i.T

        # Block (4,1): -L_i
        S_41 = -L_i

        # Block (4,3): -L_i
        S_43 = -L_i

        # Block (4,5): L_i
        S_45 = L_i

        # Block (5,2): L_i^T
        S_52 = L_i.T

        # Block (5,4): L_i^T
        S_54 = L_i.T

        # Block (5,5): P_i
        S_55 = P_i

        # Zero blocks
        zero_nn = np.zeros((n, n))
        zero_nm = np.zeros((n, m))
        zero_mn = np.zeros((m, n))
        zero_mm = np.zeros((m, m))

        # Assemble using cp.bmat (block matrix)
        # Row 0: [alpha * P_i - nu * I_n, 0, 0, 0, 0, 0]
        # Row 1: [0, -P_i, -L_i^T, -P_i, -L_i^T, 0]
        # Row 2: [0, -L_i, 0, -L_i, 0, L_i]
        # Row 3: [0, -P_i, -L_i^T, -P_i, -L_i^T, 0]
        # Row 4: [0, -L_i, 0, -L_i, 0, L_i]
        # Row 5: [0, 0, L_i^T, 0, L_i^T, P_i]

        S = cp.bmat(
            [
                [S_00, zero_nn, zero_nm, zero_nn, zero_nm, zero_nn],
                [zero_nn, S_11, S_12, S_13, S_14, zero_nn],
                [zero_mn, S_21, zero_mm, S_23, zero_mm, S_25],
                [zero_nn, S_31, S_32, S_33, S_34, zero_nn],
                [zero_mn, S_41, zero_mm, S_43, zero_mm, S_45],
                [zero_nn, zero_nn, S_52, zero_nn, S_54, S_55],
            ]
        )

        return S

    def construct_N1_matrix(
        self, H_prev: np.ndarray, H_plus_prev: np.ndarray, Xi_prev: np.ndarray, beta_prev: float
    ) -> np.ndarray:
        """
        Construct the N₁ data informativity constraint matrix.

        N₁ = G₁ @ M₁ @ G₁^T

        where:
        G₁ = [I_n    H⁺_{i-1}]  (3n+2m) x (n+L)
             [0     -H_{i-1} ]
             [0     -Ξ_{i-1} ]
             [0        0     ]
             [0        0     ]

        M₁ = [β_{i-1}I_n    0  ]  (n+L) x (n+L)
             [    0        -I_L]

        Dimension: (3n+2m) x (3n+2m)

        Args:
            H_prev: Hankel matrix H_{i-1} (nxL)
            H_plus_prev: Hankel matrix H⁺_{i-1} (nxL)
            Xi_prev: Control Hankel matrix Ξ_{i-1} (mxL)
            beta_prev: Informativity constant β_{i-1} > 0

        Returns:
            N₁ matrix as numpy array
        """
        n, m, L = self.dims.n, self.dims.m, self.dims.L

        # Validate inputs
        assert H_prev.shape == (n, L), f"H_prev shape mismatch: expected {(n, L)}, got {H_prev.shape}"
        assert H_plus_prev.shape == (n, L), f"H_plus_prev shape mismatch: expected {(n, L)}, got {H_plus_prev.shape}"
        assert Xi_prev.shape == (m, L), f"Xi_prev shape mismatch: expected {(m, L)}, got {Xi_prev.shape}"
        assert beta_prev > 0, f"Beta must be positive, got {beta_prev}"

        # Build G₁ matrix: (3n+2m) x (n+L)
        G1 = np.zeros((self.dims.N_dim, n + L))

        # Row block 0 (n rows): [I_n, H⁺_{i-1}]
        G1[0:n, 0:n] = np.eye(n)
        G1[0:n, n : n + L] = H_plus_prev

        # Row block 1 (n rows): [0, -H_{i-1}]
        G1[n : 2 * n, n : n + L] = -H_prev

        # Row block 2 (m rows): [0, -Ξ_{i-1}]
        G1[2 * n : 2 * n + m, n : n + L] = -Xi_prev

        # Row blocks 3-4 (n+m rows): all zeros (already initialized)

        # Build M₁ matrix: (n+L) x (n+L)
        M1 = np.zeros((n + L, n + L))
        M1[0:n, 0:n] = beta_prev * np.eye(n)
        M1[n : n + L, n : n + L] = -np.eye(L)

        # Compute N₁ = G₁ @ M₁ @ G₁^T
        N1 = G1 @ M1 @ G1.T

        # Verify symmetry (should be symmetric by construction)
        assert np.allclose(N1, N1.T, atol=1e-10), "N1 is not symmetric!"

        logger.debug(f"Constructed N1: shape={N1.shape}, norm={np.linalg.norm(N1):.4f}")

        return N1

    def construct_N2_matrix(self, C: float, T_segment: float) -> np.ndarray:
        """
        Construct the N₂ uncertainty bound constraint matrix.

        N₂ = G₂ @ M₂ @ G₂^T

        where:
        G₂ = [I_n  0   0 ]  (3n+2m) x (2n+m)
             [0    0   0 ]
             [0    0   0 ]
             [0   I_n  0 ]
             [0    0  I_m]

        M₂ = [C²T̃ᵢ²I_n    0      0  ]  (2n+m) x (2n+m)
             [   0       -I_n    0  ]
             [   0        0     -I_m]

        Dimension: (3n+2m) x (3n+2m)

        Args:
            C: Increment bound constant > 0
            T_segment: Segment duration T̃ᵢ > 0

        Returns:
            N₂ matrix as numpy array
        """
        n, m = self.dims.n, self.dims.m

        # Validate inputs
        assert C > 0, f"C must be positive, got {C}"
        assert T_segment > 0, f"T_segment must be positive, got {T_segment}"

        # Build G₂ matrix: (3n+2m) x (2n+m)
        G2 = np.zeros((self.dims.N_dim, 2 * n + m))

        # Row block 0 (n rows): [I_n, 0, 0]
        G2[0:n, 0:n] = np.eye(n)

        # Row blocks 1-2 (n+m rows): all zeros (already initialized)

        # Row block 3 (n rows): [0, I_n, 0]
        G2[2 * n + m : 3 * n + m, n : 2 * n] = np.eye(n)

        # Row block 4 (m rows): [0, 0, I_m]
        G2[3 * n + m : 3 * n + 2 * m, 2 * n : 2 * n + m] = np.eye(m)

        # Build M₂ matrix: (2n+m) x (2n+m)
        M2 = np.zeros((2 * n + m, 2 * n + m))
        M2[0:n, 0:n] = (C**2) * (T_segment**2) * np.eye(n)
        M2[n : 2 * n, n : 2 * n] = -np.eye(n)
        M2[2 * n : 2 * n + m, 2 * n : 2 * n + m] = -np.eye(m)

        # Compute N₂ = G₂ @ M₂ @ G₂^T
        N2 = G2 @ M2 @ G2.T

        # Verify symmetry
        assert np.allclose(N2, N2.T, atol=1e-10), "N2 is not symmetric!"

        logger.debug(f"Constructed N2: shape={N2.shape}, norm={np.linalg.norm(N2):.4f}")

        return N2

    def pad_to_S_dimension(self, N_matrix: np.ndarray) -> np.ndarray:
        """
        Pad N₁ or N₂ matrix to match S-matrix dimension.

        Ñ = [N   0]  (4n+2m) x (4n+2m)
            [0   0]

        where N is (3n+2m) x (3n+2m)

        Args:
            N_matrix: Either N₁ or N₂ matrix (3n+2m) x (3n+2m)

        Returns:
            Padded matrix Ñ of size (4n+2m) x (4n+2m)
        """
        assert N_matrix.shape == (self.dims.N_dim, self.dims.N_dim), (
            f"Input must be {self.dims.N_dim}x{self.dims.N_dim}, got {N_matrix.shape}"
        )

        # Create padded matrix
        N_tilde = np.zeros((self.dims.S_dim, self.dims.S_dim))

        # Copy N into top-left block
        N_tilde[0 : self.dims.N_dim, 0 : self.dims.N_dim] = N_matrix

        # Bottom-right block is already zeros

        logger.debug(f"Padded matrix from {N_matrix.shape} to {N_tilde.shape}")

        return N_tilde

    def construct_all_lmi_matrices(
        self,
        P_i: cp.Variable,
        L_i: cp.Variable,
        nu: Union[cp.Variable, cp.Parameter, float],
        alpha: float,
        H_prev: np.ndarray,
        H_plus_prev: np.ndarray,
        Xi_prev: np.ndarray,
        beta_prev: float,
        C: float,
        T_segment: float,
    ) -> Tuple[cp.Expression, np.ndarray, np.ndarray]:
        """
        Construct all LMI matrices in one call.

        Args:
            P_i: State Lyapunov matrix variable
            L_i: Controller parameterization variable
            nu: Slack parameter (Variable, Parameter, or float)
            alpha: Lyapunov decay rate
            H_prev: Previous segment Hankel matrix
            H_plus_prev: Previous segment Hankel matrix H⁺
            Xi_prev: Previous segment control Hankel matrix
            beta_prev: Previous segment informativity constant
            C: Increment bound
            T_segment: Segment duration

        Returns:
            Tuple of (S, Ñ₁, Ñ₂) matrices
        """
        # Construct S matrix (CVXPY expression)
        S = self.construct_S_matrix(P_i, L_i, nu, alpha)

        # Construct N₁ and N₂ (numpy arrays)
        N1 = self.construct_N1_matrix(H_prev, H_plus_prev, Xi_prev, beta_prev)
        N2 = self.construct_N2_matrix(C, T_segment)

        # Pad to S dimension
        N1_tilde = self.pad_to_S_dimension(N1)
        N2_tilde = self.pad_to_S_dimension(N2)

        logger.info("Constructed all LMI matrices successfully")

        return S, N1_tilde, N2_tilde

    def verify_matrix_properties(
        self, N1_tilde: np.ndarray, N2_tilde: np.ndarray, check_condition: bool = True
    ) -> dict:
        """
        Verify numerical properties of constructed matrices.

        Args:
            N1_tilde: Padded N₁ matrix
            N2_tilde: Padded N₂ matrix
            check_condition: Whether to compute condition numbers

        Returns:
            Dictionary of matrix properties
        """
        properties = {}

        # Check symmetry
        properties["N1_symmetric"] = np.allclose(N1_tilde, N1_tilde.T, atol=1e-10)
        properties["N2_symmetric"] = np.allclose(N2_tilde, N2_tilde.T, atol=1e-10)

        # Check dimensions
        properties["N1_shape"] = N1_tilde.shape
        properties["N2_shape"] = N2_tilde.shape
        properties["shapes_correct"] = N1_tilde.shape == (self.dims.S_dim, self.dims.S_dim) and N2_tilde.shape == (
            self.dims.S_dim,
            self.dims.S_dim,
        )

        # Compute norms
        properties["N1_norm"] = np.linalg.norm(N1_tilde)
        properties["N2_norm"] = np.linalg.norm(N2_tilde)

        # Condition numbers (expensive, optional)
        if check_condition:
            try:
                properties["N1_condition"] = np.linalg.cond(N1_tilde)
                properties["N2_condition"] = np.linalg.cond(N2_tilde)
            except np.linalg.LinAlgError:
                properties["N1_condition"] = np.inf
                properties["N2_condition"] = np.inf

        # Log results
        logger.info(f"Matrix verification: {properties}")

        return properties


def create_lmi_constructor(n_states: int, n_controls: int, data_length: int) -> LMIMatrixConstructor:
    """
    Factory function to create LMI matrix constructor.

    Args:
        n_states: State dimension
        n_controls: Control dimension
        data_length: Data trajectory length

    Returns:
        Configured LMIMatrixConstructor instance
    """
    return LMIMatrixConstructor(n_states, n_controls, data_length)
