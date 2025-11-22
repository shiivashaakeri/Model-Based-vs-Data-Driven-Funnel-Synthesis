# ddfs/ddfs/models/base.py

"""
Base classes for dynamics models.

This module provides abstract base classes for system dynamics models
used in the DDFS pipeline. All models use JAX for automatic differentiation
and high-performance computation.

Key Classes
-----------
DynamicsModel : Abstract base for all dynamics models
    - Implements continuous dynamics f(x, u)
    - Provides RK4 integration for discrete-time stepping
    - Auto-differentiates Jacobians using JAX

TwinModel : Digital twin (nominal model for planning)
    - Used in Phase 1 for trajectory planning
    - Approximate model with nominal parameters

PlantModel : Real system (actual dynamics with mismatch)
    - Used in Phase 2 for data collection
    - Includes model mismatch relative to twin
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import jacfwd, jit


class DynamicsModel(ABC):
    """
    Abstract base class for continuous-time dynamics models.

    All dynamics models implement:
        ẋ = f(x, u)

    And provide:
        - Discrete-time integration (RK4)
        - Jacobian computation via JAX autodiff
        - State and input dimension properties

    Parameters
    ----------
    dt : float
        Discretization timestep (seconds)

    Attributes
    ----------
    dt : float
        Timestep for discrete-time integration
    """

    def __init__(self, dt: float = 0.1):
        """
        Initialize dynamics model.

        Parameters
        ----------
        dt : float, optional
            Discretization timestep (seconds), by default 0.1
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

        This is the core dynamics function that must be implemented
        by all subclasses.

        Parameters
        ----------
        x : jnp.ndarray
            State vector, shape (n,)
        u : jnp.ndarray
            Input vector, shape (m,)

        Returns
        -------
        x_dot : jnp.ndarray
            State derivative ẋ, shape (n,)
        """
        pass

    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate continuous-time dynamics (JIT-compiled).

        Parameters
        ----------
        x : jnp.ndarray
            State vector, shape (n,)
        u : jnp.ndarray
            Input vector, shape (m,)

        Returns
        -------
        x_dot : jnp.ndarray
            State derivative ẋ, shape (n,)
        """
        return self._dynamics_jit(x, u)

    def _rk4_step(self, x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Single RK4 integration step.

        Implements 4th-order Runge-Kutta integration:
            k1 = f(x, u)
            k2 = f(x + dt/2 * k1, u)
            k3 = f(x + dt/2 * k2, u)
            k4 = f(x + dt * k3, u)
            x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        Parameters
        ----------
        x : jnp.ndarray
            Current state, shape (n,)
        u : jnp.ndarray
            Control input, shape (m,)
        dt : float
            Timestep

        Returns
        -------
        x_next : jnp.ndarray
            Next state x(t+dt), shape (n,)
        """
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * dt * k1, u)
        k3 = self._dynamics(x + 0.5 * dt * k2, u)
        k4 = self._dynamics(x + dt * k3, u)

        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def step(self, x: jnp.ndarray, u: jnp.ndarray, dt: Optional[float] = None) -> jnp.ndarray:
        """
        Discrete-time step: x(t+dt) = integrate[f(x,u)] from t to t+dt.

        Uses 4th-order Runge-Kutta integration for high accuracy.

        Parameters
        ----------
        x : jnp.ndarray
            Current state, shape (n,)
        u : jnp.ndarray
            Control input, shape (m,)
        dt : float, optional
            Timestep (uses self.dt if None)

        Returns
        -------
        x_next : jnp.ndarray
            Next state x(t+dt), shape (n,)
        """
        if dt is None:
            dt = self.dt
        return self._rk4_step_jit(x, u, dt)

    def jacobian_state(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Jacobian w.r.t. state: A = ∂f/∂x.

        Uses JAX automatic differentiation.

        Parameters
        ----------
        x : jnp.ndarray
            State vector, shape (n,)
        u : jnp.ndarray
            Input vector, shape (m,)

        Returns
        -------
        A : jnp.ndarray
            Jacobian matrix A = ∂f/∂x, shape (n, n)
        """
        jac_fn = jacfwd(self._dynamics, argnums=0)
        return jac_fn(x, u)

    def jacobian_input(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Jacobian w.r.t. input: B = ∂f/∂u.

        Uses JAX automatic differentiation.

        Parameters
        ----------
        x : jnp.ndarray
            State vector, shape (n,)
        u : jnp.ndarray
            Input vector, shape (m,)

        Returns
        -------
        B : jnp.ndarray
            Jacobian matrix B = ∂f/∂u, shape (n, m)
        """
        jac_fn = jacfwd(self._dynamics, argnums=1)
        return jac_fn(x, u)

    def jacobians(self, x: jnp.ndarray, u: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute both Jacobians: A = ∂f/∂x, B = ∂f/∂u.

        Parameters
        ----------
        x : jnp.ndarray
            State vector, shape (n,)
        u : jnp.ndarray
            Input vector, shape (m,)

        Returns
        -------
        A : jnp.ndarray
            State Jacobian, shape (n, n)
        B : jnp.ndarray
            Input Jacobian, shape (n, m)
        """
        A = self.jacobian_state(x, u)
        B = self.jacobian_input(x, u)
        return A, B

    def linearize(self, x_bar: jnp.ndarray, u_bar: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Linearize dynamics around operating point (x̄, ū).

        Linear approximation: δẋ ≈ A δx + B δu
        where A = ∂f/∂x|(x̄,ū), B = ∂f/∂u|(x̄,ū)

        Parameters
        ----------
        x_bar : jnp.ndarray
            Operating point state, shape (n,)
        u_bar : jnp.ndarray
            Operating point input, shape (m,)

        Returns
        -------
        A : jnp.ndarray
            Linearized state matrix, shape (n, n)
        B : jnp.ndarray
            Linearized input matrix, shape (n, m)
        """
        return self.jacobians(x_bar, u_bar)

    def simulate_trajectory(self, x0: jnp.ndarray, u_traj: jnp.ndarray, dt: Optional[float] = None) -> jnp.ndarray:
        """
        Simulate full trajectory from initial state.

        Parameters
        ----------
        x0 : jnp.ndarray
            Initial state, shape (n,)
        u_traj : jnp.ndarray
            Control trajectory, shape (N, m)
        dt : float, optional
            Timestep (uses self.dt if None)

        Returns
        -------
        x_traj : jnp.ndarray
            State trajectory, shape (N+1, n)
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
        """String representation."""
        return f"{self.__class__.__name__}(state_dim={self.state_dim}, input_dim={self.input_dim}, dt={self.dt})"


class TwinModel(DynamicsModel):
    """
    Digital twin model with nominal/approximate parameters.

    This is the model used for planning in Phase 1. It represents
    our best understanding of the system dynamics, but may not
    perfectly match the real plant.

    The twin is used for:
        - Phase 1: Trajectory planning (SCvx)
        - Phase 3: Uncertainty quantification (computing mismatch)
        - Phase 4: Funnel synthesis (linearization)

    Notes
    -----
    Subclasses must implement:
        - state_dim property
        - input_dim property
        - _dynamics(x, u) method
    """

    pass


class PlantModel(DynamicsModel):
    """
    Plant model representing the real system.

    In simulation, this includes model mismatch relative to the twin.
    In reality, this would be the actual robot hardware.

    The plant is used for:
        - Phase 2: Data collection (generating trajectories)
        - Phase 6: Deployment simulation (closed-loop testing)

    Parameters
    ----------
    twin : TwinModel
        Associated digital twin model
    mismatch_params : dict, optional
        Parameters defining plant-twin mismatch

    Attributes
    ----------
    twin : TwinModel
        Reference to the digital twin
    mismatch_params : dict
        Mismatch parameters (system-specific)
    """

    def __init__(self, twin: TwinModel, mismatch_params: Optional[dict] = None):
        """
        Initialize plant model with mismatch relative to twin.

        Parameters
        ----------
        twin : TwinModel
            Digital twin model
        mismatch_params : dict, optional
            Parameters defining plant-twin mismatch
        """
        super().__init__(dt=twin.dt)
        self.twin = twin
        self.mismatch_params = mismatch_params or {}

    @property
    def state_dim(self) -> int:
        """State dimension (inherited from twin)."""
        return self.twin.state_dim

    @property
    def input_dim(self) -> int:
        """Input dimension (inherited from twin)."""
        return self.twin.input_dim

    @abstractmethod
    def _apply_mismatch(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Apply model mismatch to dynamics.

        This method must be implemented by subclasses to define
        how the plant differs from the twin.

        Parameters
        ----------
        x : jnp.ndarray
            State vector, shape (n,)
        u : jnp.ndarray
            Input vector, shape (m,)

        Returns
        -------
        x_dot : jnp.ndarray
            State derivative with mismatch applied, shape (n,)
        """
        pass

    def _dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Plant dynamics with mismatch.

        Calls the subclass-specific _apply_mismatch method.

        Parameters
        ----------
        x : jnp.ndarray
            State vector, shape (n,)
        u : jnp.ndarray
            Input vector, shape (m,)

        Returns
        -------
        x_dot : jnp.ndarray
            State derivative ẋ, shape (n,)
        """
        return self._apply_mismatch(x, u)

    def compute_mismatch(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """
        Compute plant-twin mismatch at (x, u).

        Mismatch magnitude:
            gamma(x, u) = ||f_plant(x, u) - f_twin(x, u)||

        This is used in Phase 3 for uncertainty quantification.

        Parameters
        ----------
        x : jnp.ndarray
            State vector, shape (n,)
        u : jnp.ndarray
            Input vector, shape (m,)

        Returns
        -------
        gamma : float
            Mismatch magnitude gamma
        """
        f_plant = self._dynamics(x, u)
        f_twin = self.twin._dynamics(x, u)
        return float(jnp.linalg.norm(f_plant - f_twin))


def validate_state_input_dims(x: jnp.ndarray, u: jnp.ndarray, expected_state_dim: int, expected_input_dim: int) -> None:
    """
    Validate state and input dimensions.

    Parameters
    ----------
    x : jnp.ndarray
        State vector
    u : jnp.ndarray
        Input vector
    expected_state_dim : int
        Expected state dimension
    expected_input_dim : int
        Expected input dimension

    Raises
    ------
    ValueError
        If dimensions don't match expected values
    """
    if x.shape[-1] != expected_state_dim:
        raise ValueError(f"State dimension mismatch: got {x.shape[-1]}, expected {expected_state_dim}")
    if u.shape[-1] != expected_input_dim:
        raise ValueError(f"Input dimension mismatch: got {u.shape[-1]}, expected {expected_input_dim}")
