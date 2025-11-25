"""
Abstract Base Model Class for DDFS.

This module defines the interface that all dynamical system models must implement,
including continuous/discrete dynamics, Jacobian computation, and constraint handling.

The base class supports:
- Continuous-time dynamics: dx/dt = f(x, u)
- Discrete-time dynamics: x_{k+1} = f_d(x_k, u_k)
- Analytical and numerical Jacobian computation
- Plant-twin model relationships with mismatch quantification
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ddfs.utils.logging_utils import get_logger
from ddfs.utils.math_utils import numerical_jacobian_xu

logger = get_logger(__name__)


# =============================================================================
# Model Parameters Dataclass
# =============================================================================


@dataclass
class ModelParameters:
    """
    Container for model physical parameters.

    This dataclass stores physical parameters that define a specific
    instance of a dynamical system (e.g., mass, inertia, lengths).

    Parameters
    ----------
    params : dict
        Dictionary of parameter names to values.
    name : str
        Descriptive name for this parameter set.
    """

    params: Dict[str, Any] = field(default_factory=dict)
    name: str = "default"

    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access to parameters."""
        if key in ("params", "name") or key.startswith("_"):
            return super().__getattribute__(key)
        try:
            return self.params[key]
        except KeyError:
            raise AttributeError(f"Parameter '{key}' not found")

    def __setattr__(self, key: str, value: Any) -> None:
        """Allow attribute-style setting of parameters."""
        if key in ("params", "name"):
            super().__setattr__(key, value)
        else:
            self.params[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter with default value."""
        return self.params.get(key, default)

    def update(self, **kwargs) -> None:
        """Update multiple parameters."""
        self.params.update(kwargs)

    def copy(self) -> "ModelParameters":
        """Create a copy of parameters."""
        return ModelParameters(
            params=self.params.copy(),
            name=self.name,
        )

    def with_modifications(self, **kwargs) -> "ModelParameters":
        """Create a modified copy of parameters."""
        new_params = self.params.copy()
        new_params.update(kwargs)
        return ModelParameters(params=new_params, name=f"{self.name}_modified")

    def __repr__(self) -> str:
        return f"ModelParameters(name='{self.name}', params={self.params})"


# =============================================================================
# Abstract Base Model
# =============================================================================


class BaseModel(ABC):
    """
    Abstract base class for dynamical system models.

    All system models (unicycle, quadrotor, etc.) must inherit from this class
    and implement the required abstract methods.

    The class provides:
    - Interface for continuous and discrete dynamics
    - Jacobian computation (analytical or numerical)
    - Integration methods for discretization
    - State/input dimension properties
    - Constraint handling interface

    Parameters
    ----------
    params : ModelParameters or dict
        Physical parameters of the model.
    dt : float
        Discretization timestep [s].
    integration_method : str
        Integration method: 'euler', 'rk4', or 'rk2'.
    name : str
        Model name identifier.
    """

    def __init__(
        self,
        params: Union[ModelParameters, Dict[str, Any], None] = None,
        dt: float = 0.02,
        integration_method: str = "rk4",
        name: str = "base_model",
    ):
        """Initialize the base model."""
        # Handle parameters
        if params is None:
            self._params = ModelParameters(name=name)
        elif isinstance(params, dict):
            self._params = ModelParameters(params=params, name=name)
        else:
            self._params = params

        self._dt = dt
        self._integration_method = integration_method
        self._name = name

        # Validate integration method
        valid_methods = ["euler", "rk2", "rk4"]
        if integration_method not in valid_methods:
            raise ValueError(f"integration_method must be one of {valid_methods}, got '{integration_method}'")

        # Cache for Jacobians (optional optimization)
        self._jacobian_cache: Dict[str, Any] = {}

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def name(self) -> str:
        """Model name identifier."""
        return self._name

    @property
    def params(self) -> ModelParameters:
        """Model parameters."""
        return self._params

    @property
    def dt(self) -> float:
        """Discretization timestep [s]."""
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        """Set discretization timestep."""
        if value <= 0:
            raise ValueError(f"dt must be positive, got {value}")
        self._dt = value
        self._jacobian_cache.clear()  # Invalidate cache

    @property
    def integration_method(self) -> str:
        """Integration method name."""
        return self._integration_method

    @integration_method.setter
    def integration_method(self, value: str) -> None:
        """Set integration method."""
        valid_methods = ["euler", "rk2", "rk4"]
        if value not in valid_methods:
            raise ValueError(f"integration_method must be one of {valid_methods}")
        self._integration_method = value
        self._jacobian_cache.clear()

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of state dimensions."""
        pass

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        """Number of input dimensions."""
        pass

    @property
    def state_dim(self) -> int:
        """Alias for n_states."""
        return self.n_states

    @property
    def input_dim(self) -> int:
        """Alias for n_inputs."""
        return self.n_inputs

    @property
    def state_labels(self) -> List[str]:
        """
        Labels for state variables.

        Override in subclasses for meaningful names.
        """
        return [f"x_{i}" for i in range(self.n_states)]

    @property
    def input_labels(self) -> List[str]:
        """
        Labels for input variables.

        Override in subclasses for meaningful names.
        """
        return [f"u_{i}" for i in range(self.n_inputs)]

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def continuous_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """
        Compute continuous-time dynamics: dx/dt = f(x, u).

        Parameters
        ----------
        x : np.ndarray
            State vector of shape (n_states,).
        u : np.ndarray
            Input vector of shape (n_inputs,).

        Returns
        -------
        np.ndarray
            State derivative dx/dt of shape (n_states,).
        """
        pass

    @abstractmethod
    def get_default_parameters(self) -> ModelParameters:
        """
        Get default physical parameters for the model.

        Returns
        -------
        ModelParameters
            Default parameter set.
        """
        pass

    # =========================================================================
    # Discrete Dynamics (with numerical integration)
    # =========================================================================

    def discrete_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute discrete-time dynamics: x_{k+1} = f_d(x_k, u_k).

        Uses numerical integration of continuous dynamics.

        Parameters
        ----------
        x : np.ndarray
            Current state vector of shape (n_states,).
        u : np.ndarray
            Input vector of shape (n_inputs,).
        dt : float, optional
            Timestep (uses self.dt if not provided).

        Returns
        -------
        np.ndarray
            Next state x_{k+1} of shape (n_states,).
        """
        dt = dt if dt is not None else self._dt

        if self._integration_method == "euler":
            return self._integrate_euler(x, u, dt)
        elif self._integration_method == "rk2":
            return self._integrate_rk2(x, u, dt)
        else:  # rk4
            return self._integrate_rk4(x, u, dt)

    def _integrate_euler(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Euler integration."""
        x_dot = self.continuous_dynamics(x, u)
        return x + dt * x_dot

    def _integrate_rk2(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Second-order Runge-Kutta (midpoint method)."""
        k1 = self.continuous_dynamics(x, u)
        k2 = self.continuous_dynamics(x + 0.5 * dt * k1, u)
        return x + dt * k2

    def _integrate_rk4(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Fourth-order Runge-Kutta integration."""
        k1 = self.continuous_dynamics(x, u)
        k2 = self.continuous_dynamics(x + 0.5 * dt * k1, u)
        k3 = self.continuous_dynamics(x + 0.5 * dt * k2, u)
        k4 = self.continuous_dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Alias for discrete_dynamics.

        Parameters
        ----------
        x : np.ndarray
            Current state.
        u : np.ndarray
            Input.
        dt : float, optional
            Timestep.

        Returns
        -------
        np.ndarray
            Next state.
        """
        return self.discrete_dynamics(x, u, dt)

    def __call__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Make model callable for discrete dynamics.

        Allows: x_next = model(x, u)
        """
        return self.discrete_dynamics(x, u, dt)

    # =========================================================================
    # Jacobian Computation
    # =========================================================================

    def continuous_jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobians of continuous dynamics.

        Returns df/dx and df/du where dx/dt = f(x, u).

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray
            Input vector.

        Returns
        -------
        A_c : np.ndarray
            State Jacobian df/dx of shape (n_states, n_states).
        B_c : np.ndarray
            Input Jacobian df/du of shape (n_states, n_inputs).
        """
        # Try analytical Jacobians first
        if self.has_analytical_jacobians:
            return self._analytical_continuous_jacobians(x, u)

        # Fall back to numerical
        return self._numerical_continuous_jacobians(x, u)

    def discrete_jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobians of discrete dynamics.

        Returns A and B where x_{k+1} = f_d(x_k, u_k).

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray
            Input vector.
        dt : float, optional
            Timestep.

        Returns
        -------
        A : np.ndarray
            State Jacobian df_d/dx of shape (n_states, n_states).
        B : np.ndarray
            Input Jacobian df_d/du of shape (n_states, n_inputs).
        """
        dt = dt if dt is not None else self._dt

        # Try analytical discrete Jacobians first
        if self.has_analytical_discrete_jacobians:
            return self._analytical_discrete_jacobians(x, u, dt)

        # Numerical Jacobians of discrete dynamics
        return self._numerical_discrete_jacobians(x, u, dt)

    @property
    def has_analytical_jacobians(self) -> bool:
        """
        Whether model provides analytical continuous Jacobians.

        Override in subclass and return True if _analytical_continuous_jacobians
        is implemented.
        """
        return False

    @property
    def has_analytical_discrete_jacobians(self) -> bool:
        """
        Whether model provides analytical discrete Jacobians.

        Override in subclass and return True if _analytical_discrete_jacobians
        is implemented.
        """
        return False

    def _analytical_continuous_jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute analytical Jacobians of continuous dynamics.

        Override in subclass to provide analytical Jacobians.

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray
            Input vector.

        Returns
        -------
        A_c : np.ndarray
            State Jacobian.
        B_c : np.ndarray
            Input Jacobian.
        """
        raise NotImplementedError(
            "Analytical continuous Jacobians not implemented. "
            "Set has_analytical_jacobians = False or implement this method."
        )

    def _analytical_discrete_jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute analytical Jacobians of discrete dynamics.

        Override in subclass to provide analytical Jacobians.

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray
            Input vector.
        dt : float
            Timestep.

        Returns
        -------
        A : np.ndarray
            State Jacobian.
        B : np.ndarray
            Input Jacobian.
        """
        raise NotImplementedError(
            "Analytical discrete Jacobians not implemented. "
            "Set has_analytical_discrete_jacobians = False or implement this method."
        )

    def _numerical_continuous_jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute numerical Jacobians of continuous dynamics.

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray
            Input vector.
        eps : float
            Finite difference step size.

        Returns
        -------
        A_c : np.ndarray
            State Jacobian.
        B_c : np.ndarray
            Input Jacobian.
        """
        return numerical_jacobian_xu(self.continuous_dynamics, x, u, eps)

    def _numerical_discrete_jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute numerical Jacobians of discrete dynamics.

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray
            Input vector.
        dt : float, optional
            Timestep.
        eps : float
            Finite difference step size.

        Returns
        -------
        A : np.ndarray
            State Jacobian.
        B : np.ndarray
            Input Jacobian.
        """
        dt = dt if dt is not None else self._dt

        def f_discrete(x_in, u_in):
            return self.discrete_dynamics(x_in, u_in, dt)

        return numerical_jacobian_xu(f_discrete, x, u, eps)

    def linearize(
        self,
        x: np.ndarray,
        u: np.ndarray,
        continuous: bool = False,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around operating point.

        Parameters
        ----------
        x : np.ndarray
            State operating point.
        u : np.ndarray
            Input operating point.
        continuous : bool
            If True, return continuous-time Jacobians.
            If False, return discrete-time Jacobians.
        dt : float, optional
            Timestep for discrete linearization.

        Returns
        -------
        A : np.ndarray
            State matrix.
        B : np.ndarray
            Input matrix.
        """
        if continuous:
            return self.continuous_jacobians(x, u)
        else:
            return self.discrete_jacobians(x, u, dt)

    # =========================================================================
    # Trajectory Simulation
    # =========================================================================

    def simulate(
        self,
        x0: np.ndarray,
        u_trajectory: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simulate system forward given initial state and input trajectory.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state of shape (n_states,).
        u_trajectory : np.ndarray
            Input trajectory of shape (N, n_inputs).
        dt : float, optional
            Timestep.

        Returns
        -------
        np.ndarray
            State trajectory of shape (N+1, n_states).
        """
        N = len(u_trajectory)
        x_trajectory = np.zeros((N + 1, self.n_states))
        x_trajectory[0] = x0

        for k in range(N):
            x_trajectory[k + 1] = self.discrete_dynamics(x_trajectory[k], u_trajectory[k], dt)

        return x_trajectory

    def simulate_with_controller(
        self,
        x0: np.ndarray,
        controller: Callable[[np.ndarray, int], np.ndarray],
        N: int,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate system with feedback controller.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state.
        controller : callable
            Controller function: u = controller(x, k).
        N : int
            Number of steps.
        dt : float, optional
            Timestep.

        Returns
        -------
        x_trajectory : np.ndarray
            State trajectory of shape (N+1, n_states).
        u_trajectory : np.ndarray
            Input trajectory of shape (N, n_inputs).
        """
        x_trajectory = np.zeros((N + 1, self.n_states))
        u_trajectory = np.zeros((N, self.n_inputs))

        x_trajectory[0] = x0

        for k in range(N):
            u_trajectory[k] = controller(x_trajectory[k], k)
            x_trajectory[k + 1] = self.discrete_dynamics(x_trajectory[k], u_trajectory[k], dt)

        return x_trajectory, u_trajectory

    # =========================================================================
    # State Normalization (optional, for quaternions etc.)
    # =========================================================================

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize state if needed (e.g., quaternion normalization).

        Override in subclass if state normalization is required.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Normalized state vector.
        """
        return x

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def zero_state(self) -> np.ndarray:
        """Return zero state vector."""
        return np.zeros(self.n_states)

    def zero_input(self) -> np.ndarray:
        """Return zero input vector."""
        return np.zeros(self.n_inputs)

    def random_state(
        self,
        low: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate random state within bounds.

        Parameters
        ----------
        low : np.ndarray, optional
            Lower bounds (default: -1).
        high : np.ndarray, optional
            Upper bounds (default: +1).

        Returns
        -------
        np.ndarray
            Random state vector.
        """
        if low is None:
            low = -np.ones(self.n_states)
        if high is None:
            high = np.ones(self.n_states)
        return np.random.uniform(low, high)

    def random_input(
        self,
        low: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate random input within bounds.

        Parameters
        ----------
        low : np.ndarray, optional
            Lower bounds (default: -1).
        high : np.ndarray, optional
            Upper bounds (default: +1).

        Returns
        -------
        np.ndarray
            Random input vector.
        """
        if low is None:
            low = -np.ones(self.n_inputs)
        if high is None:
            high = np.ones(self.n_inputs)
        return np.random.uniform(low, high)

    def validate_state(self, x: np.ndarray) -> bool:
        """
        Validate state vector dimensions.

        Parameters
        ----------
        x : np.ndarray
            State vector to validate.

        Returns
        -------
        bool
            True if valid.
        """
        return x.shape == (self.n_states,)

    def validate_input(self, u: np.ndarray) -> bool:
        """
        Validate input vector dimensions.

        Parameters
        ----------
        u : np.ndarray
            Input vector to validate.

        Returns
        -------
        bool
            True if valid.
        """
        return u.shape == (self.n_inputs,)

    def info(self) -> Dict[str, Any]:
        """
        Get model information dictionary.

        Returns
        -------
        dict
            Model information.
        """
        return {
            "name": self.name,
            "n_states": self.n_states,
            "n_inputs": self.n_inputs,
            "state_labels": self.state_labels,
            "input_labels": self.input_labels,
            "dt": self.dt,
            "integration_method": self.integration_method,
            "has_analytical_jacobians": self.has_analytical_jacobians,
            "has_analytical_discrete_jacobians": self.has_analytical_discrete_jacobians,
            "parameters": self.params.params,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  name='{self.name}',\n"
            f"  n_states={self.n_states},\n"
            f"  n_inputs={self.n_inputs},\n"
            f"  dt={self.dt},\n"
            f"  integration_method='{self.integration_method}'\n"
            f")"
        )


# =============================================================================
# Plant-Twin Model Pair
# =============================================================================


class PlantTwinPair:
    """
    Container for plant and twin model pair.

    Manages the relationship between the physical plant (unknown/uncertain)
    and its digital twin (known model used for planning).

    Parameters
    ----------
    twin : BaseModel
        Digital twin model (known, used for planning).
    plant : BaseModel
        Physical plant model (represents true system).
    """

    def __init__(
        self,
        twin: BaseModel,
        plant: BaseModel,
    ):
        """Initialize plant-twin pair."""
        self.twin = twin
        self.plant = plant

        # Validate compatibility
        if twin.n_states != plant.n_states:
            raise ValueError(f"State dimension mismatch: twin={twin.n_states}, plant={plant.n_states}")
        if twin.n_inputs != plant.n_inputs:
            raise ValueError(f"Input dimension mismatch: twin={twin.n_inputs}, plant={plant.n_inputs}")

    @property
    def n_states(self) -> int:
        """Number of state dimensions."""
        return self.twin.n_states

    @property
    def n_inputs(self) -> int:
        """Number of input dimensions."""
        return self.twin.n_inputs

    @property
    def dt(self) -> float:
        """Timestep (from twin)."""
        return self.twin.dt

    @dt.setter
    def dt(self, value: float) -> None:
        """Set timestep for both models."""
        self.twin.dt = value
        self.plant.dt = value

    def mismatch(
        self,
        x: np.ndarray,
        u: np.ndarray,
        continuous: bool = True,
    ) -> np.ndarray:
        """
        Compute mismatch between plant and twin: Δ(x, u) = f_plant - f_twin.

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray
            Input vector.
        continuous : bool
            If True, compute continuous dynamics mismatch.
            If False, compute discrete dynamics mismatch.

        Returns
        -------
        np.ndarray
            Mismatch vector Δ(x, u).
        """
        if continuous:
            f_plant = self.plant.continuous_dynamics(x, u)
            f_twin = self.twin.continuous_dynamics(x, u)
        else:
            f_plant = self.plant.discrete_dynamics(x, u)
            f_twin = self.twin.discrete_dynamics(x, u)

        return f_plant - f_twin

    def estimate_mismatch_bound(
        self,
        x_samples: np.ndarray,
        u_samples: np.ndarray,
        continuous: bool = True,
    ) -> float:
        """
        Estimate uniform mismatch bound gamma from samples.

        Computes max ||Δ(x, u)|| over samples.

        Parameters
        ----------
        x_samples : np.ndarray
            State samples of shape (n_samples, n_states).
        u_samples : np.ndarray
            Input samples of shape (n_samples, n_inputs).
        continuous : bool
            Whether to use continuous or discrete dynamics.

        Returns
        -------
        float
            Estimated mismatch bound gamma.
        """
        max_mismatch = 0.0

        for x, u in zip(x_samples, u_samples):
            delta = self.mismatch(x, u, continuous)
            mismatch_norm = np.linalg.norm(delta)
            max_mismatch = max(max_mismatch, mismatch_norm)

        return max_mismatch

    def estimate_mismatch_bound_along_trajectory(
        self,
        x_trajectory: np.ndarray,
        u_trajectory: np.ndarray,
        continuous: bool = False,
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate mismatch bound along a trajectory.

        Parameters
        ----------
        x_trajectory : np.ndarray
            State trajectory of shape (N+1, n_states).
        u_trajectory : np.ndarray
            Input trajectory of shape (N, n_inputs).
        continuous : bool
            Whether to use continuous or discrete dynamics.

        Returns
        -------
        gamma : float
            Maximum mismatch bound along trajectory.
        mismatch_norms : np.ndarray
            Mismatch norms at each point.
        """
        N = len(u_trajectory)
        mismatch_norms = np.zeros(N)

        for k in range(N):
            delta = self.mismatch(x_trajectory[k], u_trajectory[k], continuous)
            mismatch_norms[k] = np.linalg.norm(delta)

        return np.max(mismatch_norms), mismatch_norms

    def simulate_both(
        self,
        x0: np.ndarray,
        u_trajectory: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate both plant and twin with same initial condition and inputs.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state.
        u_trajectory : np.ndarray
            Input trajectory.

        Returns
        -------
        x_twin : np.ndarray
            Twin state trajectory.
        x_plant : np.ndarray
            Plant state trajectory.
        """
        x_twin = self.twin.simulate(x0, u_trajectory)
        x_plant = self.plant.simulate(x0, u_trajectory)
        return x_twin, x_plant

    def __repr__(self) -> str:
        return (
            f"PlantTwinPair(\n"
            f"  twin={self.twin.name},\n"
            f"  plant={self.plant.name},\n"
            f"  n_states={self.n_states},\n"
            f"  n_inputs={self.n_inputs}\n"
            f")"
        )


# =============================================================================
# Model Factory Protocol
# =============================================================================


class ModelFactory:
    """
    Factory for creating plant-twin model pairs.

    Subclass this to create factory classes for specific systems
    (e.g., UnicycleFactory, QuadrotorFactory).
    """

    @staticmethod
    def create_twin(**kwargs) -> BaseModel:
        """Create twin model. Override in subclass."""
        raise NotImplementedError

    @staticmethod
    def create_plant(**kwargs) -> BaseModel:
        """Create plant model. Override in subclass."""
        raise NotImplementedError

    @classmethod
    def create_pair(cls, **kwargs) -> PlantTwinPair:
        """Create plant-twin pair."""
        twin = cls.create_twin(**kwargs)
        plant = cls.create_plant(**kwargs)
        return PlantTwinPair(twin=twin, plant=plant)
