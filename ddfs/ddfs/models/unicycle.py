"""
Unicycle model (digital twin).

This module implements the kinematic unicycle model for mobile robot navigation.

Dynamics:
    State: x = [px, py, theta]^T
        - px, py: Position in 2D workspace
        - theta: Heading angle

    Input: u = [v, omega]^T
        - v: Linear velocity
        - omega: Angular velocity

    Continuous-time dynamics:
        xdot = [v * cos(theta)
                v * sin(theta)
                omega]
"""

from typing import Optional, Tuple

import numpy as np

from .base import DynamicalSystem


class UnicycleModel(DynamicalSystem):
    """
    Kinematic unicycle model (digital twin).

    This represents the nominal/ideal dynamics of the system without any uncertainties.
    """

    def __init__(self, x0: Optional[np.ndarray] = None, xf: Optional[np.ndarray] = None):
        """
        Initialize the unicycle model.

        Args:
            x0: Initial state (3,)
            xf: Final state (3,)
        """
        super().__init__(n_states=3, n_inputs=2)
        self.x0 = x0 if x0 is not None else np.zeros(3)
        self.xf = xf if xf is not None else np.zeros(3)

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time unicycle dynamics: xdot = f(x, u)\

        Args:
            x: State [px, py, theta]
            u: Input [v, omega]

        Returns:
            xdot: Time derivative of state [vx, vy, thetadot]
        """
        # Extract states
        px, py, theta = x

        # Extract inputs
        v, omega = u

        # Dynamics
        xdot = np.array([v * np.cos(theta), v * np.sin(theta), omega])
        return xdot

    def linearize(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lineaize unicycle dynamics around (x,u).

        Computes jacobians A and B analytically.

        Args:
            x: State at linearization point [px, py, theta]
            u: Input at linearization point [v, omega]

        Returns:
            A: State Jacobian, shape (3,3)
            B: Input Jacobian, shape (3,2)
        """
        # Extract states and inputs
        px, py, theta = x
        v, omega = u

        # State Jacobian: A = d/dx f(x,u)
        A = np.array([[0, 0, -v * np.sin(theta)], [0, 0, v * np.cos(theta)], [0, 0, 0]])

        # Input Jacobian: B = d/du f(x,u)
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])

        return A, B

    def set_initial_state(self, x0: np.ndarray):
        """Set initial state."""
        assert x0.shape == (3,), "Initial state must be a 3-vector"
        self.x0 = x0.copy()

    def set_desired_state(self, xf: np.ndarray):
        """Set desired state."""
        assert xf.shape == (3,), "Desired state must be a 3-vector"
        self.xf = xf.copy()

    def get_initial_state(self) -> np.ndarray:
        """Get initial state."""
        return self.x0.copy()

    def get_desired_state(self) -> np.ndarray:
        """Get desired state."""
        return self.xf.copy()

    def distance_to_goal(self, x: np.ndarray) -> float:
        """
        Compute Euclidean distance from current state to desired state. (position only)

        Args:
            x: Current state [px, py, theta]

        Returns:
            distance: Euclidean distance to desired state
        """
        return np.linalg.norm(x[:2] - self.xf[:2])

    def is_goal_reached(self, x: np.ndarray, position_tol: float = 0.1, angle_tol: float = 0.1) -> bool:
        """
        Check if the current state is close to the desired state.

        Args:
            x: Current state [px, py, theta]
            position_tol: Position tolerance
            angle_tol: Angle tolerance

        Returns:
            reached: True if goal is reached, False otherwise
        """
        pos_error = np.linalg.norm(x[:2] - self.xf[:2])
        angle_error = np.abs(self._angle_diff(x[2], self.xf[2]))

        return pos_error < position_tol and angle_error < angle_tol

    @staticmethod
    def _angle_diff(theta1: float, theta2: float) -> float:
        """
        Compute signed difference between two angles in radians. (wraps to [-pi, pi]).

        Args:
            theta1: First angle in radians
            theta2: Second angle in radians

        Returns:
            diff: Signed difference in [-pi, pi]
        """
        diff = theta1 - theta2

        # Wrap to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi

        return diff

    def __repr__(self) -> str:
        """String representation of the unicycle model."""
        return f"UnicycleModel(n_states={self.n_states}, n_inputs={self.n_inputs}, x0={self.x0}, xf={self.xf})"
