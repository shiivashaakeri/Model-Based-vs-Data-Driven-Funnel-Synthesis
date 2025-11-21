"""
Uncertainty quantification for DDFS.

This module computes all uncertainty constants required for robust funnel synthesis:
- gamma: Maximum plant-twin mismatch along nominal trajectory
- L_r: Linearization error Lipschitz constant (via finite differences)
- L_J: Jacobian Lipschitz constant (via sampling)
- C: Increment bound (C = L_J * v_max)
- β_i: Per-segment uncertainty bounds from data

These constants are used in the LMI-based funnel synthesis (Phase 6).
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UncertaintyConstants:
    """
    Container for all uncertainty constants in DDFS.

    Attributes:
        gamma: Maximum plant-twin mismatch gamma = max ||f_plant - f_twin||
        L_r: Linearization error Lipschitz constant
        L_J: Jacobian Lipschitz constant
        C: Increment bound C = L_J * v_max
        beta_i: Per-segment uncertainty bounds (list of β_i for each segment)

        # Metadata
        n_samples_gamma: Number of samples used for gamma computation
        n_samples_L_J: Number of samples used for L_J computation
        epsilon_fd: Finite difference epsilon for L_r
        v_max: Maximum velocity bound used for C
    """

    # Core constants
    gamma: float = 0.0
    L_r: float = 0.0
    L_J: float = 0.0
    C: float = 0.0
    beta_i: List[float] = field(default_factory=list)

    # Metadata
    n_samples_gamma: int = 0
    n_samples_L_J: int = 0  # noqa: N815
    epsilon_fd: float = 1e-6
    v_max: float = 0.0

    # Detailed results (optional, for analysis)
    gamma_per_timestep: Optional[np.ndarray] = None
    L_r_per_state: Optional[np.ndarray] = None
    L_J_samples: Optional[np.ndarray] = None

    def save(self, filepath: Path):
        """Save constants to pickle file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"✓ Saved uncertainty constants to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "UncertaintyConstants":
        """Load constants from pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def summary(self) -> str:
        """Generate a summary string of all constants."""
        lines = [
            "=" * 60,
            "UNCERTAINTY CONSTANTS SUMMARY",
            "=" * 60,
            f"gamma (plant-twin mismatch):     {self.gamma:.6f}",
            f"L_r (linearization error):   {self.L_r:.6f}",
            f"L_J (Jacobian Lipschitz):    {self.L_J:.6f}",
            f"C (increment bound):         {self.C:.6f}",
            f"v_max (velocity bound):      {self.v_max:.6f}",
            "",
            f"Number of segments:          {len(self.beta_i)}",
            f"β_i range: [{min(self.beta_i):.6f}, {max(self.beta_i):.6f}]" if self.beta_i else "β_i: (empty)",
            "",
            "Sampling:",
            f"  - gamma samples:  {self.n_samples_gamma}",
            f"  - L_J samples: {self.n_samples_L_J}",
            f"  - FD epsilon:  {self.epsilon_fd:.2e}",
            "=" * 60,
        ]
        return "\n".join(lines)


class UncertaintyQuantifier:
    """
    Computes all uncertainty constants for DDFS.

    This class implements the uncertainty quantification algorithms from
    the DDFS paper, computing:
    1. gamma: Plant-twin mismatch along nominal
    2. L_r: Linearization error Lipschitz constant
    3. L_J: Jacobian Lipschitz constant
    4. C: Increment bound
    5. β_i: Per-segment data-driven bounds
    """

    def __init__(
        self,
        plant,
        twin,
        n_states: int = 3,
        n_controls: int = 2,
        epsilon_fd: float = 1e-6,
        n_samples_L_J: int = 10000,
        sampling_box: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize uncertainty quantifier.

        Args:
            plant: Physical plant dynamics (DynamicalSystem)
            twin: Twin model dynamics (DynamicalSystem)
            n_states: State dimension
            n_controls: Control dimension
            epsilon_fd: Finite difference step size for L_r
            n_samples_L_J: Number of samples for L_J computation
            sampling_box: Box constraints for sampling (dict with 'x_min', 'x_max', 'u_min', 'u_max')
        """
        self.plant = plant
        self.twin = twin
        self.n_states = n_states
        self.n_controls = n_controls
        self.epsilon_fd = epsilon_fd
        self.n_samples_L_J = n_samples_L_J

        # Default sampling box (can be overridden)
        if sampling_box is None:
            self.sampling_box = {
                "x_min": np.array([-10, -10, -2 * np.pi]),
                "x_max": np.array([10, 10, 2 * np.pi]),
                "u_min": np.array([-2.0, -2.0]),
                "u_max": np.array([2.0, 2.0]),
            }
        else:
            self.sampling_box = sampling_box

    def compute_all(
        self, nominal_trajectory: Dict, collected_data: Dict, v_max: float = 1.0, verbose: bool = True
    ) -> UncertaintyConstants:
        """
        Compute all uncertainty constants.

        Args:
            nominal_trajectory: Nominal trajectory dict with 'X', 'U', 'T'
            collected_data: Collected offline data with segments
            v_max: Maximum velocity bound for increment bound C
            verbose: Print progress

        Returns:
            UncertaintyConstants object with all computed values
        """
        constants = UncertaintyConstants(epsilon_fd=self.epsilon_fd, v_max=v_max)

        if verbose:
            print("\n" + "=" * 70)
            print("PHASE 5: UNCERTAINTY QUANTIFICATION")
            print("=" * 70)

        # 1. Compute gamma (plant-twin mismatch)
        if verbose:
            print("\n📊 Step 1: Computing gamma (plant-twin mismatch)...")
        constants.gamma, constants.gamma_per_timestep, constants.n_samples_gamma = self._compute_gamma(
            nominal_trajectory, verbose
        )

        # 2. Compute L_r (linearization error)
        if verbose:
            print("\n📊 Step 2: Computing L_r (linearization error)...")
        constants.L_r, constants.L_r_per_state = self._compute_L_r(nominal_trajectory, verbose)

        # 3. Compute L_J (Jacobian Lipschitz constant)
        if verbose:
            print("\n📊 Step 3: Computing L_J (Jacobian Lipschitz)...")
        constants.L_J, constants.L_J_samples, constants.n_samples_L_J = self._compute_L_J(verbose)

        # 4. Compute C (increment bound)
        if verbose:
            print("\n📊 Step 4: Computing C (increment bound)...")
        constants.C = self._compute_C(constants.L_J, v_max, verbose)

        # 5. Compute β_i (per-segment bounds)
        if verbose:
            print("\n📊 Step 5: Computing β_i (per-segment bounds)...")
        constants.beta_i = self._compute_beta_i(collected_data, verbose)

        if verbose:
            print("\n" + constants.summary())

        return constants

    def _compute_gamma(self, nominal_trajectory: Dict, verbose: bool = True) -> Tuple[float, np.ndarray, int]:
        """
        Compute gamma: maximum plant-twin mismatch along nominal trajectory.

        gamma = max_t ||f_plant(x_t, u_t) - f_twin(x_t, u_t)||

        Args:
            nominal_trajectory: Dict with 'X' (N x n), 'U' (N x m), 'T' (N,)
            verbose: Print progress

        Returns:
            gamma: Maximum mismatch
            gamma_per_timestep: Mismatch at each timestep (N,)
            n_samples: Number of timesteps evaluated
        """
        X = nominal_trajectory["X"]  # (N, n)
        U = nominal_trajectory["U"]  # (N, m)
        N = X.shape[0]

        gamma_per_timestep = np.zeros(N)

        for t in range(N):
            x_t = X[t]
            u_t = U[t]

            # Evaluate both dynamics
            f_plant = self.plant.dynamics(x_t, u_t)
            f_twin = self.twin.dynamics(x_t, u_t)

            # Compute mismatch
            mismatch = np.linalg.norm(f_plant - f_twin)
            gamma_per_timestep[t] = mismatch

        gamma = np.max(gamma_per_timestep)

        if verbose:
            print(f"   gamma = {gamma:.6f}")
            print(f"   Mean mismatch: {np.mean(gamma_per_timestep):.6f}")
            print(f"   Std mismatch:  {np.std(gamma_per_timestep):.6f}")
            print(f"   Evaluated at {N} timesteps")

        return gamma, gamma_per_timestep, N

    def _compute_L_r(self, nominal_trajectory: Dict, verbose: bool = True) -> Tuple[float, np.ndarray]:
        """
        Compute L_r: Lipschitz constant of linearization error via finite differences.

        The linearization error is:
        r(x, u, δx, δu) = f(x+δx, u+δu) - f(x, u) - A*δx - B*δu

        We estimate L_r such that ||r|| ≤ L_r * ||(δx, δu)||

        Args:
            nominal_trajectory: Dict with 'X', 'U', 'T'
            verbose: Print progress

        Returns:
            L_r: Lipschitz constant
            L_r_per_state: L_r computed at each nominal state (N,)
        """
        X = nominal_trajectory["X"]
        U = nominal_trajectory["U"]
        N = X.shape[0]

        L_r_per_state = np.zeros(N)
        epsilon = self.epsilon_fd

        for t in range(N):
            x_t = X[t]
            u_t = U[t]

            # Compute Jacobians A, B at (x_t, u_t)
            A_t, B_t = self._compute_jacobians(x_t, u_t)

            # Sample perturbations
            max_L_r_at_t = 0.0
            n_perturbations = 10  # Sample a few perturbations per state

            for _ in range(n_perturbations):
                # Random perturbation
                delta_x = np.random.randn(self.n_states) * epsilon
                delta_u = np.random.randn(self.n_controls) * epsilon

                # Evaluate nonlinear dynamics
                f_perturbed = self.twin.dynamics(x_t + delta_x, u_t + delta_u)
                f_nominal = self.twin.dynamics(x_t, u_t)

                # Linearization
                f_linear = f_nominal + A_t @ delta_x + B_t @ delta_u

                # Residual
                residual = f_perturbed - f_linear
                residual_norm = np.linalg.norm(residual)

                # Perturbation norm
                perturbation_norm = np.linalg.norm(np.concatenate([delta_x, delta_u]))

                # Lipschitz constant estimate
                if perturbation_norm > 1e-10:
                    L_r_estimate = residual_norm / perturbation_norm
                    max_L_r_at_t = max(max_L_r_at_t, L_r_estimate)

            L_r_per_state[t] = max_L_r_at_t

        L_r = np.max(L_r_per_state)

        if verbose:
            print(f"   L_r = {L_r:.6f}")
            print(f"   Mean L_r: {np.mean(L_r_per_state):.6f}")
            print(f"   Evaluated at {N} nominal states")

        return L_r, L_r_per_state

    def _compute_L_J(self, verbose: bool = True) -> Tuple[float, np.ndarray, int]:
        """
        Compute L_J: Lipschitz constant of the Jacobian via sampling.

        L_J measures how much the linearization (A, B) changes across the state-control space:
        L_J = max_{(x1,u1), (x2,u2)} ||[A1, B1] - [A2, B2]||_F / ||(x1-x2, u1-u2)||

        Args:
            verbose: Print progress

        Returns:
            L_J: Jacobian Lipschitz constant
            L_J_samples: Array of L_J estimates from pairs (n_samples,)
            n_samples: Number of sample pairs evaluated
        """
        n_samples = self.n_samples_L_J
        L_J_samples = []

        # Sample random pairs of (x, u)
        for _ in range(n_samples):
            # Sample two random points
            x1 = np.random.uniform(self.sampling_box["x_min"], self.sampling_box["x_max"])
            u1 = np.random.uniform(self.sampling_box["u_min"], self.sampling_box["u_max"])

            x2 = np.random.uniform(self.sampling_box["x_min"], self.sampling_box["x_max"])
            u2 = np.random.uniform(self.sampling_box["u_min"], self.sampling_box["u_max"])

            # Compute Jacobians
            A1, B1 = self._compute_jacobians(x1, u1)
            A2, B2 = self._compute_jacobians(x2, u2)

            # Jacobian difference (Frobenius norm)
            J1 = np.hstack([A1, B1])  # (n, n+m)
            J2 = np.hstack([A2, B2])
            J_diff_norm = np.linalg.norm(J1 - J2, "fro")

            # State-control difference
            xu1 = np.concatenate([x1, u1])
            xu2 = np.concatenate([x2, u2])
            xu_diff_norm = np.linalg.norm(xu1 - xu2)

            # Lipschitz estimate
            if xu_diff_norm > 1e-10:
                L_J_estimate = J_diff_norm / xu_diff_norm
                L_J_samples.append(L_J_estimate)

        L_J_samples = np.array(L_J_samples)
        L_J = np.max(L_J_samples)

        if verbose:
            print(f"   L_J = {L_J:.6f}")
            print(f"   Mean L_J: {np.mean(L_J_samples):.6f}")
            print(f"   95th percentile: {np.percentile(L_J_samples, 95):.6f}")
            print(f"   Evaluated {n_samples} sample pairs")

        return L_J, L_J_samples, n_samples

    def _compute_C(self, L_J: float, v_max: float, verbose: bool = True) -> float:
        """
        Compute C: increment bound.

        C = L_J * v_max

        where v_max is the maximum velocity (norm of ẋ) in the system.

        Args:
            L_J: Jacobian Lipschitz constant
            v_max: Maximum velocity bound
            verbose: Print progress

        Returns:
            C: Increment bound
        """
        C = L_J * v_max

        if verbose:
            print(f"   C = L_J * v_max = {L_J:.6f} * {v_max:.4f} = {C:.6f}")

        return C

    def _compute_beta_i(self, collected_data: Dict, verbose: bool = True) -> List[float]:
        """
        Compute β_i: per-segment uncertainty bounds from data.

        For each segment i, β_i quantifies the data-driven uncertainty:
        β_i is computed from the Hankel matrices and represents the
        "noise" or uncertainty in the data for that segment.

        In practice, β_i can be estimated as:
        β_i = ||Ξ_i||_F / ||H_i||_F

        where Ξ_i contains the residuals and H_i is the Hankel matrix.

        Args:
            collected_data: Dict with 'segments' containing Hankel matrices
            verbose: Print progress

        Returns:
            List of β_i values (one per segment)
        """
        segments = collected_data.get("segments", [])
        n_segments = len(segments)

        if n_segments == 0:
            if verbose:
                print("   ⚠️  No segments found in collected data!")
            return []

        beta_i = []

        for i, segment in enumerate(segments):
            # Extract Hankel matrices
            H_i = segment.get("H_i")  # Past data Hankel matrix
            Xi_i = segment.get("Xi_i")  # Residual/noise matrix

            if H_i is None or Xi_i is None:
                if verbose:
                    print(f"   ⚠️  Segment {i}: Missing Hankel matrices, using default β_i = 0.1")
                beta_i.append(0.1)
                continue

            # Compute β_i as ratio of Frobenius norms
            H_i_norm = np.linalg.norm(H_i, "fro")
            Xi_i_norm = np.linalg.norm(Xi_i, "fro")

            if H_i_norm < 1e-10:
                if verbose:
                    print(f"   ⚠️  Segment {i}: H_i has zero norm, using default β_i = 0.1")
                beta_i.append(0.1)
            else:
                beta_val = Xi_i_norm / H_i_norm
                beta_i.append(beta_val)

        if verbose:
            print(f"   Computed β_i for {n_segments} segments")
            print(f"   β_i range: [{min(beta_i):.6f}, {max(beta_i):.6f}]")
            print(f"   Mean β_i: {np.mean(beta_i):.6f}")

        return beta_i

    def _compute_jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobians A = ∂f/∂x and B = ∂f/∂u via finite differences.

        Args:
            x: State (n,)
            u: Control (m,)

        Returns:
            A: State Jacobian (n, n)
            B: Control Jacobian (n, m)
        """
        epsilon = self.epsilon_fd
        f_nominal = self.twin.dynamics(x, u)

        # Compute A = ∂f/∂x
        A = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            x_perturbed = x.copy()
            x_perturbed[i] += epsilon
            f_perturbed = self.twin.dynamics(x_perturbed, u)
            A[:, i] = (f_perturbed - f_nominal) / epsilon

        # Compute B = ∂f/∂u
        B = np.zeros((self.n_states, self.n_controls))
        for i in range(self.n_controls):
            u_perturbed = u.copy()
            u_perturbed[i] += epsilon
            f_perturbed = self.twin.dynamics(x, u_perturbed)
            B[:, i] = (f_perturbed - f_nominal) / epsilon

        return A, B


def main():
    """
    Example usage of UncertaintyQuantifier.

    This would typically be called from a script that loads:
    1. The nominal trajectory (from Phase 3)
    2. The collected offline data (from Phase 4)
    3. The plant and twin models
    """
    # This is a placeholder - you'd load actual data
    print("=" * 70)
    print("UncertaintyQuantifier - Example Usage")
    print("=" * 70)
    print()
    print("To use this module:")
    print("1. Load nominal trajectory from Phase 3")
    print("2. Load collected data from Phase 4")
    print("3. Initialize plant and twin models")
    print("4. Create UncertaintyQuantifier and call compute_all()")
    print()
    print("See scripts/03_compute_uncertainty.py for full implementation")
    print("=" * 70)


if __name__ == "__main__":
    main()
