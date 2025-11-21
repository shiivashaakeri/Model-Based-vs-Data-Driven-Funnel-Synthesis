# ddfs/ddfs/models/base.py

"""
Base classes for dynamics models.

This module provides abstract base classes for system dynamics models
used in the DDFS pipeline. All models use JAX for automatic differentiation
and high-performance computation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import jacfwd, jit


class DynamicsModel(ABC):
    """
    Abstract base class for continuous-time dynamics models.

    Implements:
        - Continuous dynamics: ẋ = f(x, u)
        - Discrete-time integration (RK4)
        - Jacobian computation via JAX autodiff
        - State and input dimension properties
    """

    def __init__(self, dt: float = 0.1):
        """
        Initialize dynamics model.

        Args:
            dt: Discretization timestep (seconds)
        """
        self.dt = dt

        # JIT-compile core methods for performance
        self._dynamics_jit = jit(self._dynamics)
        self._rk4_step_jit = jit(self._rk4_step)

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """State dimension n."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Input dimension m."""
        pass

    @abstractmethod
    def _dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Continuous-time dynamics: ẋ = f(x, u).

        Args:
            x: State vector (n,)
            u: Input vector (m,)

        Returns:
            State derivative ẋ (n,)
        """
        pass

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate continuous-time dynamics (JIT-compiled).

        Args:
            x: State vector (n,)
            u: Input vector (m,)

        Returns:
            State derivative ẋ (n,)
        """
        return self._dynamics_jit(x, u)

    def _rk4_step(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Single RK4 integration step.

        Args:
            x: Current state (n,)
            u: Control input (m,)
            dt: Timestep

        Returns:
            Next state x(t+dt) (n,)
        """
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * dt * k1, u)
        k3 = self._dynamics(x + 0.5 * dt * k2, u)
        k4 = self._dynamics(x + dt * k3, u)

        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def step(self, x: jnp.ndarray, u: jnp.ndarray, dt: Optional[float] = None) -> jnp.ndarray:
        """
        Discrete-time step: x(t+dt) = x(t) + ∫[t, t+dt] f(x, u) dτ.

        Uses 4th-order Runge-Kutta integration (RK4).

        Args:
            x: Current state (n,)
            u: Control input (m,)
            dt: Timestep (uses self.dt if None)

        Returns:
            Next state x(t+dt) (n,)
        """
        if dt is None:
            dt = self.dt
        return self._rk4_step_jit(x, u, dt)

    def jacobian_state(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Jacobian w.r.t. state: ∂f/∂x.

        Uses JAX automatic differentiation.

        Args:
            x: State vector (n,)
            u: Input vector (m,)

        Returns:
            Jacobian matrix A = ∂f/∂x (n, n)
        """
        jac_fn = jacfwd(self._dynamics, argnums=0)
        return jac_fn(x, u)

    def jacobian_input(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Jacobian w.r.t. input: ∂f/∂u.

        Uses JAX automatic differentiation.

        Args:
            x: State vector (n,)
            u: Input vector (m,)

        Returns:
            Jacobian matrix B = ∂f/∂u (n, m)
        """
        jac_fn = jacfwd(self._dynamics, argnums=1)
        return jac_fn(x, u)

    def jacobians(self, x: jnp.ndarray, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute both Jacobians: A = ∂f/∂x, B = ∂f/∂u.

        Args:
            x: State vector (n,)
            u: Input vector (m,)

        Returns:
            (A, B): Jacobian matrices (n, n) and (n, m)
        """
        A = self.jacobian_state(x, u)
        B = self.jacobian_input(x, u)
        return A, B

    def linearize(self, x_bar: jnp.ndarray, u_bar: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Linearize dynamics around operating point (x̄, ū).

        Linear approximation: δẋ ≈ A δx + B δu
        where A = ∂f/∂x|(x̄,ū), B = ∂f/∂u|(x̄,ū)

        Args:
            x_bar: Operating point state (n,)
            u_bar: Operating point input (m,)

        Returns:
            (A, B): Linearized dynamics matrices
        """
        return self.jacobians(x_bar, u_bar)

    def simulate_trajectory(self, x0: jnp.ndarray, u_traj: jnp.ndarray, dt: Optional[float] = None) -> jnp.ndarray:
        """
        Simulate full trajectory from initial state.

        Args:
            x0: Initial state (n,)
            u_traj: Control trajectory (N, m)
            dt: Timestep (uses self.dt if None)

        Returns:
            State trajectory (N+1, n)
        """
        if dt is None:
            dt = self.dt

        N = u_traj.shape[0]
        x_traj = jnp.zeros((N + 1, self.state_dim))
        x_traj = x_traj.at[0].set(x0)

        for k in range(N):
            x_traj = x_traj.at[k + 1].set(self.step(x_traj[k], u_traj[k], dt))

        return x_traj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(state_dim={self.state_dim}, input_dim={self.input_dim}, dt={self.dt})"


class TwinModel(DynamicsModel):
    """
    Digital twin model with approximate/nominal parameters.

    This is the model used for planning (Phase 1). It may not
    perfectly match the real plant dynamics.
    """

    pass


class PlantModel(DynamicsModel):
    """
    Plant model representing the real system.

    In simulation, this includes model mismatch relative to the twin.
    In reality, this would be the actual robot hardware.
    """

    def __init__(self, twin: TwinModel, mismatch_params: Optional[Dict[str, Any]] = None):
        """
        Initialize plant model with mismatch relative to twin.

        Args:
            twin: Digital twin model
            mismatch_params: Parameters defining plant-twin mismatch
        """
        super().__init__(dt=twin.dt)
        self.twin = twin
        self.mismatch_params = mismatch_params or {}

    @property
    def state_dim(self) -> int:
        return self.twin.state_dim

    @property
    def input_dim(self) -> int:
        return self.twin.input_dim

    @abstractmethod
    def _apply_mismatch(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Apply model mismatch to dynamics.

        Args:
            x: State vector (n,)
            u: Input vector (m,)

        Returns:
            State derivative with mismatch applied (n,)
        """
        pass

    def _dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Plant dynamics with mismatch.

        Args:
            x: State vector (n,)
            u: Input vector (m,)

        Returns:
            State derivative ẋ (n,)
        """
        return self._apply_mismatch(x, u)

    def compute_mismatch(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """
        Compute plant-twin mismatch at (x, u).

        gamma(x, u) = ||f_plant(x, u) - f_twin(x, u)||

        Args:
            x: State vector (n,)
            u: Input vector (m,)

        Returns:
            Mismatch magnitude gamma
        """
        f_plant = self._dynamics(x, u)
        f_twin = self.twin._dynamics(x, u)
        return jnp.linalg.norm(f_plant - f_twin)


def validate_state_input_dims(x: jnp.ndarray, u: jnp.ndarray, expected_state_dim: int, expected_input_dim: int) -> None:
    """
    Validate state and input dimensions.

    Args:
        x: State vector
        u: Input vector
        expected_state_dim: Expected state dimension
        expected_input_dim: Expected input dimension

    Raises:
        ValueError: If dimensions don't match
    """
    if x.shape[-1] != expected_state_dim:
        raise ValueError(f"State dimension mismatch: got {x.shape[-1]}, expected {expected_state_dim}")
    if u.shape[-1] != expected_input_dim:
        raise ValueError(f"Input dimension mismatch: got {u.shape[-1]}, expected {expected_input_dim}")
