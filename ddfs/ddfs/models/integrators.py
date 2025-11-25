"""
Numerical Integration Module for DDFS.

This module provides numerical integration schemes for discretizing
continuous-time dynamics, including:
- Euler method (1st order)
- Midpoint method / RK2 (2nd order)
- Runge-Kutta 4 (4th order)
- Jacobian propagation through integrators

Each integrator can be used standalone or as part of a model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple, Union

import numpy as np

from ddfs.utils.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

# Dynamics function signature: f(x, u) -> x_dot
DynamicsFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]

# Jacobian function signature: f(x, u) -> (A, B) where A = df/dx, B = df/du
JacobianFunction = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


# =============================================================================
# Integration Method Enum
# =============================================================================


class IntegrationMethod(Enum):
    """Available numerical integration methods."""

    EULER = "euler"
    RK2 = "rk2"
    MIDPOINT = "midpoint"  # Alias for RK2
    RK4 = "rk4"

    @classmethod
    def from_string(cls, name: str) -> "IntegrationMethod":
        """Create from string name."""
        name_lower = name.lower()
        if name_lower in ("euler", "forward_euler"):
            return cls.EULER
        elif name_lower in ("rk2", "midpoint", "heun"):
            return cls.RK2
        elif name_lower in ("rk4", "runge_kutta", "runge-kutta"):
            return cls.RK4
        else:
            raise ValueError(f"Unknown integration method: {name}. Available: euler, rk2, rk4")


# =============================================================================
# Abstract Integrator Base Class
# =============================================================================


class Integrator(ABC):
    """
    Abstract base class for numerical integrators.

    Parameters
    ----------
    dt : float
        Default integration timestep.
    """

    def __init__(self, dt: float = 0.01):
        if dt <= 0:
            raise ValueError(f"Timestep dt must be positive, got {dt}")
        self._dt = dt

    @property
    def dt(self) -> float:
        """Default timestep."""
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"Timestep dt must be positive, got {value}")
        self._dt = value

    @property
    @abstractmethod
    def order(self) -> int:
        """Order of accuracy of the integrator."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the integration method."""
        pass

    @abstractmethod
    def step(
        self,
        f: DynamicsFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Perform one integration step.

        Parameters
        ----------
        f : callable
            Dynamics function f(x, u) -> x_dot.
        x : np.ndarray
            Current state.
        u : np.ndarray
            Input (held constant during step).
        dt : float, optional
            Timestep (uses default if not provided).

        Returns
        -------
        np.ndarray
            Next state.
        """
        pass

    def integrate(
        self,
        f: DynamicsFunction,
        x0: np.ndarray,
        u_trajectory: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Integrate over a trajectory of inputs.

        Parameters
        ----------
        f : callable
            Dynamics function.
        x0 : np.ndarray
            Initial state.
        u_trajectory : np.ndarray
            Input trajectory of shape (N, n_inputs).
        dt : float, optional
            Timestep.

        Returns
        -------
        np.ndarray
            State trajectory of shape (N+1, n_states).
        """
        dt = dt if dt is not None else self._dt
        N = len(u_trajectory)
        n_states = len(x0)

        x_trajectory = np.zeros((N + 1, n_states))
        x_trajectory[0] = x0

        for k in range(N):
            x_trajectory[k + 1] = self.step(f, x_trajectory[k], u_trajectory[k], dt)

        return x_trajectory

    def __call__(
        self,
        f: DynamicsFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Make integrator callable."""
        return self.step(f, x, u, dt)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dt={self.dt}, order={self.order})"


# =============================================================================
# Euler Integrator
# =============================================================================


class EulerIntegrator(Integrator):
    """
    Forward Euler integration (1st order).

    x_{k+1} = x_k + dt * f(x_k, u_k)

    Simple but low accuracy. Suitable for smooth dynamics with small timesteps.
    """

    @property
    def order(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "euler"

    def step(
        self,
        f: DynamicsFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Euler step."""
        dt = dt if dt is not None else self._dt
        x_dot = f(x, u)
        return x + dt * x_dot

    def step_with_derivative(
        self,
        f: DynamicsFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Euler step returning both next state and derivative.

        Returns
        -------
        x_next : np.ndarray
            Next state.
        x_dot : np.ndarray
            State derivative at current point.
        """
        dt = dt if dt is not None else self._dt
        x_dot = f(x, u)
        x_next = x + dt * x_dot
        return x_next, x_dot


# =============================================================================
# RK2 / Midpoint Integrator
# =============================================================================


class RK2Integrator(Integrator):
    """
    Second-order Runge-Kutta (Midpoint method).

    k1 = f(x_k, u_k)
    k2 = f(x_k + 0.5*dt*k1, u_k)
    x_{k+1} = x_k + dt * k2

    Better accuracy than Euler with modest computational increase.
    """

    @property
    def order(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "rk2"

    def step(
        self,
        f: DynamicsFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """RK2 step."""
        dt = dt if dt is not None else self._dt

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)

        return x + dt * k2

    def step_with_stages(
        self,
        f: DynamicsFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RK2 step returning intermediate stages.

        Returns
        -------
        x_next : np.ndarray
            Next state.
        k1 : np.ndarray
            First stage derivative.
        k2 : np.ndarray
            Second stage derivative.
        """
        dt = dt if dt is not None else self._dt

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        x_next = x + dt * k2

        return x_next, k1, k2


# =============================================================================
# RK4 Integrator
# =============================================================================


class RK4Integrator(Integrator):
    """
    Fourth-order Runge-Kutta integration.

    k1 = f(x_k, u_k)
    k2 = f(x_k + 0.5*dt*k1, u_k)
    k3 = f(x_k + 0.5*dt*k2, u_k)
    k4 = f(x_k + dt*k3, u_k)
    x_{k+1} = x_k + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    High accuracy, the standard choice for most applications.
    """

    @property
    def order(self) -> int:
        return 4

    @property
    def name(self) -> str:
        return "rk4"

    def step(
        self,
        f: DynamicsFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """RK4 step."""
        dt = dt if dt is not None else self._dt

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)

        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step_with_stages(
        self,
        f: DynamicsFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        RK4 step returning intermediate stages.

        Returns
        -------
        x_next : np.ndarray
            Next state.
        k1, k2, k3, k4 : np.ndarray
            Stage derivatives.
        """
        dt = dt if dt is not None else self._dt

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return x_next, k1, k2, k3, k4


# =============================================================================
# Integrator with Jacobian Propagation
# =============================================================================


class IntegratorWithJacobian:
    """
    Integrator that also computes Jacobians of the discrete dynamics.

    Propagates Jacobians through the integration steps to compute
    the sensitivity of x_{k+1} with respect to x_k and u_k.

    Parameters
    ----------
    integrator : Integrator
        Base integrator to use.
    """

    def __init__(self, integrator: Integrator):
        self.integrator = integrator

    @property
    def dt(self) -> float:
        return self.integrator.dt

    @dt.setter
    def dt(self, value: float) -> None:
        self.integrator.dt = value

    def step_with_jacobians(
        self,
        f: DynamicsFunction,
        f_jacobian: JacobianFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform integration step and compute Jacobians.

        Parameters
        ----------
        f : callable
            Dynamics function f(x, u) -> x_dot.
        f_jacobian : callable
            Jacobian function returning (df/dx, df/du).
        x : np.ndarray
            Current state.
        u : np.ndarray
            Input.
        dt : float, optional
            Timestep.

        Returns
        -------
        x_next : np.ndarray
            Next state.
        A : np.ndarray
            Jacobian dx_{k+1}/dx_k.
        B : np.ndarray
            Jacobian dx_{k+1}/du_k.
        """
        dt = dt if dt is not None else self.integrator.dt

        if isinstance(self.integrator, EulerIntegrator):
            return self._euler_jacobians(f, f_jacobian, x, u, dt)
        elif isinstance(self.integrator, RK2Integrator):
            return self._rk2_jacobians(f, f_jacobian, x, u, dt)
        elif isinstance(self.integrator, RK4Integrator):
            return self._rk4_jacobians(f, f_jacobian, x, u, dt)
        else:
            raise NotImplementedError(f"Jacobian propagation not implemented for {type(self.integrator)}")

    def _euler_jacobians(
        self,
        f: DynamicsFunction,
        f_jacobian: JacobianFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Euler step with Jacobian computation.

        x_{k+1} = x_k + dt * f(x_k, u_k)

        A = I + dt * df/dx
        B = dt * df/du
        """
        n = len(x)

        # Compute next state
        x_dot = f(x, u)
        x_next = x + dt * x_dot

        # Compute Jacobians
        A_c, B_c = f_jacobian(x, u)

        A = np.eye(n) + dt * A_c
        B = dt * B_c

        return x_next, A, B

    def _rk2_jacobians(
        self,
        f: DynamicsFunction,
        f_jacobian: JacobianFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RK2 step with Jacobian computation.

        k1 = f(x, u)
        x_mid = x + 0.5*dt*k1
        k2 = f(x_mid, u)
        x_{k+1} = x + dt * k2

        Using chain rule:
        A = I + dt * df/dx|_mid * (I + 0.5*dt * df/dx|_x)
        B = dt * (df/du|_mid + df/dx|_mid * 0.5*dt * df/du|_x)
        """
        n = len(x)

        # Forward pass
        k1 = f(x, u)
        x_mid = x + 0.5 * dt * k1
        k2 = f(x_mid, u)
        x_next = x + dt * k2

        # Jacobians at each point
        A1, B1 = f_jacobian(x, u)  # At x
        A2, B2 = f_jacobian(x_mid, u)  # At x_mid

        # Chain rule
        # dx_mid/dx = I + 0.5*dt*A1
        # dx_mid/du = 0.5*dt*B1
        dx_mid_dx = np.eye(n) + 0.5 * dt * A1
        dx_mid_du = 0.5 * dt * B1

        # dk2/dx = A2 * dx_mid/dx
        # dk2/du = B2 + A2 * dx_mid/du
        dk2_dx = A2 @ dx_mid_dx
        dk2_du = B2 + A2 @ dx_mid_du

        # dx_next/dx = I + dt * dk2/dx
        # dx_next/du = dt * dk2/du
        A = np.eye(n) + dt * dk2_dx
        B = dt * dk2_du

        return x_next, A, B

    def _rk4_jacobians(
        self,
        f: DynamicsFunction,
        f_jacobian: JacobianFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RK4 step with Jacobian computation.

        Uses chain rule through all four stages.
        """
        n = len(x)

        # Forward pass - compute stages
        k1 = f(x, u)
        x2 = x + 0.5 * dt * k1
        k2 = f(x2, u)
        x3 = x + 0.5 * dt * k2
        k3 = f(x3, u)
        x4 = x + dt * k3
        k4 = f(x4, u)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Jacobians at each evaluation point
        A1, B1 = f_jacobian(x, u)
        A2, B2 = f_jacobian(x2, u)
        A3, B3 = f_jacobian(x3, u)
        A4, B4 = f_jacobian(x4, u)

        I = np.eye(n)

        # Stage derivatives using chain rule
        # dk1/dx = A1, dk1/du = B1

        # x2 = x + 0.5*dt*k1
        # dx2/dx = I + 0.5*dt*A1
        # dx2/du = 0.5*dt*B1
        dx2_dx = I + 0.5 * dt * A1
        dx2_du = 0.5 * dt * B1

        # dk2/dx = A2 @ dx2/dx
        # dk2/du = B2 + A2 @ dx2/du
        dk2_dx = A2 @ dx2_dx
        dk2_du = B2 + A2 @ dx2_du

        # x3 = x + 0.5*dt*k2
        # dx3/dx = I + 0.5*dt*dk2/dx
        # dx3/du = 0.5*dt*dk2/du
        dx3_dx = I + 0.5 * dt * dk2_dx
        dx3_du = 0.5 * dt * dk2_du

        # dk3/dx = A3 @ dx3/dx
        # dk3/du = B3 + A3 @ dx3/du
        dk3_dx = A3 @ dx3_dx
        dk3_du = B3 + A3 @ dx3_du

        # x4 = x + dt*k3
        # dx4/dx = I + dt*dk3/dx
        # dx4/du = dt*dk3/du
        dx4_dx = I + dt * dk3_dx
        dx4_du = dt * dk3_du

        # dk4/dx = A4 @ dx4/dx
        # dk4/du = B4 + A4 @ dx4/du
        dk4_dx = A4 @ dx4_dx
        dk4_du = B4 + A4 @ dx4_du

        # x_next = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        # A = dx_next/dx = I + (dt/6)*(dk1/dx + 2*dk2/dx + 2*dk3/dx + dk4/dx)
        # B = dx_next/du = (dt/6)*(dk1/du + 2*dk2/du + 2*dk3/du + dk4/du)
        A = I + (dt / 6.0) * (A1 + 2.0 * dk2_dx + 2.0 * dk3_dx + dk4_dx)
        B = (dt / 6.0) * (B1 + 2.0 * dk2_du + 2.0 * dk3_du + dk4_du)

        return x_next, A, B


# =============================================================================
# Numerical Jacobian Computation for Discrete Dynamics
# =============================================================================


def numerical_discrete_jacobians(
    f: DynamicsFunction,
    integrator: Integrator,
    x: np.ndarray,
    u: np.ndarray,
    dt: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute discrete Jacobians numerically using finite differences.

    Parameters
    ----------
    f : callable
        Continuous dynamics function.
    integrator : Integrator
        Integrator to use.
    x : np.ndarray
        State vector.
    u : np.ndarray
        Input vector.
    dt : float, optional
        Timestep.
    eps : float
        Finite difference step.

    Returns
    -------
    A : np.ndarray
        State Jacobian dx_{k+1}/dx_k.
    B : np.ndarray
        Input Jacobian dx_{k+1}/du_k.
    """
    dt = dt if dt is not None else integrator.dt
    n_x = len(x)
    n_u = len(u)

    # Reference point
    x_next_ref = integrator.step(f, x, u, dt)  # noqa: F841

    # State Jacobian
    A = np.zeros((n_x, n_x))
    for i in range(n_x):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps

        x_next_plus = integrator.step(f, x_plus, u, dt)
        x_next_minus = integrator.step(f, x_minus, u, dt)

        A[:, i] = (x_next_plus - x_next_minus) / (2 * eps)

    # Input Jacobian
    B = np.zeros((n_x, n_u))
    for i in range(n_u):
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[i] += eps
        u_minus[i] -= eps

        x_next_plus = integrator.step(f, x, u_plus, dt)
        x_next_minus = integrator.step(f, x, u_minus, dt)

        B[:, i] = (x_next_plus - x_next_minus) / (2 * eps)

    return A, B


# =============================================================================
# Factory Functions
# =============================================================================


def create_integrator(
    method: Union[str, IntegrationMethod] = "rk4",
    dt: float = 0.01,
) -> Integrator:
    """
    Create an integrator by name.

    Parameters
    ----------
    method : str or IntegrationMethod
        Integration method name.
    dt : float
        Timestep.

    Returns
    -------
    Integrator
        Created integrator instance.
    """
    if isinstance(method, str):
        method = IntegrationMethod.from_string(method)

    if method == IntegrationMethod.EULER:
        return EulerIntegrator(dt=dt)
    elif method in (IntegrationMethod.RK2, IntegrationMethod.MIDPOINT):
        return RK2Integrator(dt=dt)
    elif method == IntegrationMethod.RK4:
        return RK4Integrator(dt=dt)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def create_integrator_with_jacobian(
    method: Union[str, IntegrationMethod] = "rk4",
    dt: float = 0.01,
) -> IntegratorWithJacobian:
    """
    Create an integrator with Jacobian propagation capability.

    Parameters
    ----------
    method : str or IntegrationMethod
        Integration method name.
    dt : float
        Timestep.

    Returns
    -------
    IntegratorWithJacobian
        Integrator with Jacobian computation.
    """
    base_integrator = create_integrator(method, dt)
    return IntegratorWithJacobian(base_integrator)


# =============================================================================
# Integrator Comparison Utility
# =============================================================================


@dataclass
class IntegrationResult:
    """Result of integration comparison."""

    method: str
    trajectory: np.ndarray
    final_state: np.ndarray
    n_evaluations: int


def compare_integrators(
    f: DynamicsFunction,
    x0: np.ndarray,
    u: np.ndarray,
    t_final: float,
    dt: float,
    methods: Optional[list] = None,
) -> dict:
    """
    Compare different integration methods on the same problem.

    Parameters
    ----------
    f : callable
        Dynamics function.
    x0 : np.ndarray
        Initial state.
    u : np.ndarray
        Constant input.
    t_final : float
        Final time.
    dt : float
        Timestep.
    methods : list, optional
        Methods to compare (default: all).

    Returns
    -------
    dict
        Dictionary mapping method names to IntegrationResult.
    """
    if methods is None:
        methods = ["euler", "rk2", "rk4"]

    N = int(t_final / dt)
    u_trajectory = np.tile(u, (N, 1))

    results = {}

    for method in methods:
        integrator = create_integrator(method, dt)
        trajectory = integrator.integrate(f, x0, u_trajectory, dt)

        # Count function evaluations per step
        evals_per_step = {"euler": 1, "rk2": 2, "rk4": 4}
        n_evals = N * evals_per_step.get(method, 1)

        results[method] = IntegrationResult(
            method=method,
            trajectory=trajectory,
            final_state=trajectory[-1],
            n_evaluations=n_evals,
        )

    return results


# =============================================================================
# Variable Step Size Integration (Optional)
# =============================================================================


class AdaptiveRK45Integrator:
    """
    Adaptive step-size Runge-Kutta-Fehlberg (RK45) integrator.

    Automatically adjusts step size to maintain error within tolerance.
    Useful for stiff systems or when high accuracy is needed.

    Parameters
    ----------
    dt_init : float
        Initial timestep guess.
    dt_min : float
        Minimum allowed timestep.
    dt_max : float
        Maximum allowed timestep.
    atol : float
        Absolute error tolerance.
    rtol : float
        Relative error tolerance.
    """

    def __init__(
        self,
        dt_init: float = 0.01,
        dt_min: float = 1e-6,
        dt_max: float = 1.0,
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        self.dt_init = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.atol = atol
        self.rtol = rtol

        # RK45 Butcher tableau coefficients
        self._a = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
        self._b = np.array(
            [
                [0, 0, 0, 0, 0],
                [1 / 4, 0, 0, 0, 0],
                [3 / 32, 9 / 32, 0, 0, 0],
                [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
                [439 / 216, -8, 3680 / 513, -845 / 4104, 0],
                [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
            ]
        )
        # 5th order weights
        self._c5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
        # 4th order weights
        self._c4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])

    def integrate_to_time(
        self,
        f: DynamicsFunction,
        x0: np.ndarray,
        u: np.ndarray,
        t_final: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate from t=0 to t=t_final with adaptive stepping.

        Parameters
        ----------
        f : callable
            Dynamics function.
        x0 : np.ndarray
            Initial state.
        u : np.ndarray
            Input (held constant).
        t_final : float
            Final time.

        Returns
        -------
        t_history : np.ndarray
            Time points.
        x_history : np.ndarray
            State history.
        """
        t = 0.0
        x = x0.copy()
        dt = self.dt_init

        t_history = [t]
        x_history = [x.copy()]

        while t < t_final:
            # Don't overshoot
            dt = min(dt, t_final - t)

            # Compute RK45 step
            x_new, error = self._rk45_step(f, x, u, dt)

            # Error estimate
            scale = self.atol + self.rtol * np.maximum(np.abs(x), np.abs(x_new))
            err_ratio = np.max(np.abs(error) / scale)

            if err_ratio <= 1.0:
                # Accept step
                t += dt
                x = x_new
                t_history.append(t)
                x_history.append(x.copy())

            # Adjust step size
            dt_new = 0.9 * dt * (1.0 / err_ratio) ** 0.2 if err_ratio > 0 else 2.0 * dt

            dt = np.clip(dt_new, self.dt_min, self.dt_max)

        return np.array(t_history), np.array(x_history)

    def _rk45_step(
        self,
        f: DynamicsFunction,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one RK45 step.

        Returns
        -------
        x_new : np.ndarray
            New state (5th order).
        error : np.ndarray
            Error estimate (difference between 4th and 5th order).
        """
        k = np.zeros((6, len(x)))

        k[0] = f(x, u)
        k[1] = f(x + dt * self._b[1, 0] * k[0], u)
        k[2] = f(x + dt * (self._b[2, 0] * k[0] + self._b[2, 1] * k[1]), u)
        k[3] = f(x + dt * (self._b[3, 0] * k[0] + self._b[3, 1] * k[1] + self._b[3, 2] * k[2]), u)
        k[4] = f(
            x + dt * (self._b[4, 0] * k[0] + self._b[4, 1] * k[1] + self._b[4, 2] * k[2] + self._b[4, 3] * k[3]), u
        )
        k[5] = f(
            x
            + dt
            * (
                self._b[5, 0] * k[0]
                + self._b[5, 1] * k[1]
                + self._b[5, 2] * k[2]
                + self._b[5, 3] * k[3]
                + self._b[5, 4] * k[4]
            ),
            u,
        )

        # 5th order solution
        x5 = x + dt * np.sum(self._c5[:, None] * k, axis=0)

        # 4th order solution
        x4 = x + dt * np.sum(self._c4[:, None] * k, axis=0)

        # Error estimate
        error = x5 - x4

        return x5, error
