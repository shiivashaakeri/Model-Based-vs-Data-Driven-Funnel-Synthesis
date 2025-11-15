"""
Abstract base class for dynamic system models.

This module defines the interface that all system models (digital twin, plant, etc.) must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class DynamicalSystem(ABC):
    """
    Abstract base class for dynamical systems.

    All concrete system models should inherit from this class
    and implement the required methods.
    """

    def __init__(self, n_states: int, n_inputs: int):
        """
        initialize the dynamical system.

        Args:
            n_states: Dimension of state space
            n_inputs: Dimension of input space
        """
        self.n_states = n_states
        self.n_inputs = n_inputs

    @abstractmethod
    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time dynamics: x_dot = f(x, u)

        Args:
            x: State vector (n_states,)
            u: Input vector (n_inputs,)

        Returns:
            xdot: Time derivative of state (n_states,)
        """
        pass

    @abstractmethod
    def linearize(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around (x, u): xdot = f(x*, u*) + A(x-x*) + B(u-u*)

        Args:
            x: State at linearization point (n_states,)
            u: Input at linearization point (n_inputs,)

        Returns:
            A: Jacobian of dynamics with respect to state (n_states, n_states)
            B: Jacobian of dynamics with respect to input (n_states, n_inputs)
        """
        pass

    def discrete_dynamics(self, x: np.ndarray, u: np.ndarray, dt: float, method: str = "rk4") -> np.ndarray:
        """
        Discretize contiuous dynamics using numerical integration.

        Args:
            x: State vector (n_states,)
            u: Input vector (n_inputs,)
            dt: Time step
            method: Integration method ('rk4' for Runge-Kutta 4th order)

        Returns:
            x_next: Next state vector (n_states,)
        """
        if method == "euler":
            return self._euler_step(x, u, dt)
        elif method == "rk4":
            return self._rk4_step(x, u, dt)
        else:
            raise ValueError(f"Invalid integration method: {method}")

    def _euler_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Forward Euler integration step.

        Args:
            x: Current state vector (n_states,)
            u: Current input vector (n_inputs,)
            dt: Time step

        Returns:
            x_next: Next state vector (n_states,)
        """
        return x + dt * self.dynamics(x, u)

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Runge-Kutta 4th order integration step.

        Args:
            x: Current state vector (n_states,)
            u: Current input vector (n_inputs,)
            dt: Time step

        Returns:
            x_next: Next state vector (n_states,)
        """
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5 * dt * k1, u)
        k3 = self.dynamics(x + 0.5 * dt * k2, u)
        k4 = self.dynamics(x + dt * k3, u)
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def discrete_linearization(
        self, x: np.ndarray, u: np.ndarray, dt: float, method: str = "rk4"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute linearization of the discrete time dynamics via finite difference approximation.

        x+ = f_d(x, u) = f_d(x*, u*) + A_d(x-x*) + B_d(u-u*)

        Args:
            x: State at linearization point (n_states,)
            u: Input at linearization point (n_inputs,)
            dt: Time step
            method: Integration method ('rk4' for Runge-Kutta 4th order)

        Returns:
            A_d: Jacobian of discrete dynamics with respect to state (n_states, n_states)
            B_d: Jacobian of discrete dynamics with respect to input (n_states, n_inputs)
        """
        eps = 1e-7  # finite difference step size

        # Nominal next state
        x_nom_next = self.discrete_dynamics(x, u, dt, method)

        # State Jacobian A_d via finite difference
        A_d = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            x_pert = x.copy()
            x_pert[i] += eps
            x_pert_next = self.discrete_dynamics(x_pert, u, dt, method)
            A_d[:, i] = (x_pert_next - x_nom_next) / eps

        # Input Jacobian B_d via finite difference
        B_d = np.zeros((self.n_states, self.n_inputs))
        for j in range(self.n_inputs):
            u_pert = u.copy()
            u_pert[j] += eps
            u_pert_next = self.discrete_dynamics(x, u_pert, dt, method)
            B_d[:, j] = (u_pert_next - x_nom_next) / eps

        return A_d, B_d

    def simulate(self, x0: np.ndarray, u_traj: np.ndarray, dt: float, method: str = "rk4") -> np.ndarray:
        """
        Simulate system forward in time given initial state and input trajectory.

        Args:
            x0: Initial state vector (n_states,)
            u_traj: Input trajectory (n_inputs, T)
            dt: Time step
            method: Integration method ('rk4' for Runge-Kutta 4th order)

        Returns:
            x_traj: State trajectory (n_states, T)
        """
        N = u_traj.shape[0]
        x_traj = np.zeros((N + 1, self.n_states))
        x_traj[0] = x0

        for k in range(N):
            x_traj[k + 1] = self.discrete_dynamics(x_traj[k], u_traj[k], dt, method)

        return x_traj

    @property
    def state_dim(self) -> int:
        """Get state dimension."""
        return self.n_states

    @property
    def input_dim(self) -> int:
        """Get input dimension."""
        return self.n_inputs
