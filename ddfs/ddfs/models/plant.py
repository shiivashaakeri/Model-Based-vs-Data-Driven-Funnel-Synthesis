"""
Physical plant model with model-plant mismatch.

This module implements the true physical system dynamics, which differs from the digital twin
due to parameter uncertainties, unmodeled dynamics, and disturbances.

The relationship between plant and twin is:
    f_plant(x, u) = f_twin(x, u) + Δ(x, u)

where Δ(x, u) represents the additive mismatch.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .base import DynamicalSystem
from .unicycle import UnicycleModel


class PlantModel(DynamicalSystem):
    """
    Physical plant model with parameter mismatch.

    The plant dynamics differ from the digital twin due to:
    - Parameter uncertainties (e.g., slightly different kinematics)
    - Unmodeled effects (e.g., slip, skid, terrain interaction)
    - External disturbances
    """

    def __init__(
        self,
        twin: UnicycleModel,
        parameter_mismatch: Optional[Dict[str, float]] = None,
        x0: Optional[np.ndarray] = None,
        xf: Optional[np.ndarray] = None,
    ):
        """
        Initialize plant model with mismatch relative to digital twin.

        Args:
            twin: Reference to the digital twin model
            parameter_mismatch: Dict of parameter perturbations
                For unicycle, can include:
                - 'velocity_scale': Multiplier on commanded velocity (e.g., 0.95 = 5% slower)
                - 'angular_rate_scale': Multiplier on angular rate (e.g., 1.03 = 3% faster turning)
                - 'slip_coefficient': Lateral slip factor (0 = no slip, >0 = more slip)
            x0: Initial state (defaults to twin's x0)
            xf: Desired state (defaults to twin's xf)
        """
        super().__init__(n_states=3, n_inputs=2)

        # Store reference to twin
        self.twin = twin

        # Default parameter mismatch: slight deviations
        self.param_mismatch = (
            parameter_mismatch
            if parameter_mismatch is not None
            else {
                "velocity_scale": 0.95,  # Plant is 5% slower than twin
                "angular_rate_scale": 1.03,  # Plant turns 3% faster than twin
                "slip_coefficient": 0.02,  # Small lateral slip
            }
        )

        # Initial and desired states
        self.x0 = x0 if x0 is not None else twin.get_initial_state()
        self.xf = xf if xf is not None else twin.get_desired_state()

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Physical plant dynamics with parameter mismatch.

        The plant has slightly different behavior than the twin due to:
        - Scaled velocity/angular rate
        - Slip effects

        Args:
            x: State [px, py, theta]
            u: Input [v, omega]

        Returns:
            xdot: Time derivative with mismatch effects
        """
        # Extract states and inputs
        px, py, theta = x
        v, omega = u

        # Apply parameter mismatch
        v_actual = v * self.param_mismatch["velocity_scale"]
        omega_actual = omega * self.param_mismatch["angular_rate_scale"]

        # Slip effect: small lateral drift perpendicular to heading
        slip_coeff = self.param_mismatch["slip_coefficient"]

        # Modified dynamics with mismatch
        xdot = np.array(
            [
                v_actual * np.cos(theta) - slip_coeff * v_actual * np.sin(theta),  # Lateral slip
                v_actual * np.sin(theta) + slip_coeff * v_actual * np.cos(theta),  # Lateral slip
                omega_actual,
            ]
        )

        return xdot

    def linearize(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize plant dynamics around (x, u).

        Computes Jacobians A = ∂f_plant/∂x and B = ∂f_plant/∂u analytically.

        Args:
            x: State at linearization point [px, py, theta]
            u: Input at linearization point [v, omega]

        Returns:
            A: State Jacobian, shape (3, 3)
            B: Input Jacobian, shape (3, 2)
        """
        # Extract states and inputs
        px, py, theta = x
        v, omega = u

        # Get parameter scales
        v_scale = self.param_mismatch["velocity_scale"]
        omega_scale = self.param_mismatch["angular_rate_scale"]
        slip = self.param_mismatch["slip_coefficient"]

        # Actual velocity with scaling
        v_actual = v * v_scale

        # State Jacobian: A = ∂f_plant/∂x
        A = np.array(
            [
                [0, 0, -v_actual * np.sin(theta) - slip * v_actual * np.cos(theta)],
                [0, 0, v_actual * np.cos(theta) - slip * v_actual * np.sin(theta)],
                [0, 0, 0],
            ]
        )

        # Input Jacobian: B = ∂f_plant/∂u
        B = np.array(
            [
                [v_scale * (np.cos(theta) - slip * np.sin(theta)), 0],
                [v_scale * (np.sin(theta) + slip * np.cos(theta)), 0],
                [0, omega_scale],
            ]
        )

        return A, B

    def compute_mismatch(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute additive mismatch Δ(x, u) between plant and twin.

        Δ(x, u) = f_plant(x, u) - f_twin(x, u)

        This is the discrepancy that the data-driven controller must handle.

        Args:
            x: State [px, py, theta]
            u: Input [v, omega]

        Returns:
            delta: Mismatch vector, shape (3,)
        """
        f_plant = self.dynamics(x, u)
        f_twin = self.twin.dynamics(x, u)
        delta = f_plant - f_twin
        return delta

    def compute_mismatch_norm(self, x: np.ndarray, u: np.ndarray) -> float:
        """
        Compute norm of mismatch: ||Δ(x, u)||

        Args:
            x: State [px, py, theta]
            u: Input [v, omega]

        Returns:
            norm: Euclidean norm of mismatch
        """
        delta = self.compute_mismatch(x, u)
        return np.linalg.norm(delta)

    def compute_max_mismatch_on_trajectory(self, x_traj: np.ndarray, u_traj: np.ndarray) -> Tuple[float, int]:
        """
        Compute maximum mismatch along a trajectory (used for computing gamma).

        gamma = max_k ||Δ(x(k), u(k))||

        Args:
            x_traj: State trajectory, shape (N+1, 3)
            u_traj: Input trajectory, shape (N, 2)

        Returns:
            gamma: Maximum mismatch norm
            max_idx: Index where maximum occurs
        """
        N = u_traj.shape[0]
        mismatch_norms = np.zeros(N)

        for k in range(N):
            mismatch_norms[k] = self.compute_mismatch_norm(x_traj[k], u_traj[k])

        max_idx = np.argmax(mismatch_norms)
        gamma = mismatch_norms[max_idx]

        return gamma, max_idx

    def set_parameter_mismatch(self, param_mismatch: Dict[str, float]):
        """
        Update parameter mismatch values.

        Args:
            param_mismatch: Dict of parameter perturbations
        """
        self.param_mismatch.update(param_mismatch)

    def get_parameter_mismatch(self) -> Dict[str, float]:
        """Get current parameter mismatch."""
        return self.param_mismatch.copy()

    def set_initial_state(self, x0: np.ndarray):
        """Set initial state."""
        assert len(x0) == 3, "Initial state must be 3-dimensional [px, py, theta]"
        self.x0 = x0.copy()

    def set_desired_state(self, xf: np.ndarray):
        """Set desired/goal state."""
        assert len(xf) == 3, "Desired state must be 3-dimensional [px, py, theta]"
        self.xf = xf.copy()

    def get_initial_state(self) -> np.ndarray:
        """Get initial state."""
        return self.x0.copy()

    def get_desired_state(self) -> np.ndarray:
        """Get desired state."""
        return self.xf.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"PlantModel(n_states={self.n_states}, n_inputs={self.n_inputs}, param_mismatch={self.param_mismatch})"
