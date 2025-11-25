"""
Unicycle Model Implementation for DDFS.

This module implements the unicycle (differential-drive robot) model with:
- Continuous-time dynamics
- Analytical Jacobians
- Plant and twin variants with configurable mismatch
- Factory for creating plant-twin pairs

State: x = [px, py, theta]^T
  - px: x-position [m]
  - py: y-position [m]
  - theta: heading angle [rad]

Input: u = [v, omega]^T
  - v: linear velocity [m/s]
  - omega: angular velocity [rad/s]

Continuous dynamics (basic):
  px_dot = v * cos(theta)
  py_dot = v * sin(theta)
  theta_dot = omega
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ddfs.models.base_model import BaseModel, ModelFactory, ModelParameters, PlantTwinPair
from ddfs.utils.logging_utils import get_logger
from ddfs.utils.math_utils import wrap_angle

logger = get_logger(__name__)


# =============================================================================
# Unicycle Parameters
# =============================================================================


@dataclass
class UnicycleParameters(ModelParameters):
    """
    Parameters for unicycle model.

    Twin Parameters (nominal model):
    - No additional parameters for basic unicycle

    Plant Mismatch Parameters:
    - velocity_scale: Multiplicative factor on velocity (default: 1.0)
    - heading_bias: Additive bias on heading rate [rad/s] (default: 0.0)
    - lateral_slip: Lateral slip coefficient (default: 0.0)
    - velocity_bias: Additive bias on velocity [m/s] (default: 0.0)
    """

    def __init__(
        self,
        velocity_scale: float = 1.0,
        heading_bias: float = 0.0,
        lateral_slip: float = 0.0,
        velocity_bias: float = 0.0,
        name: str = "unicycle_params",
    ):
        params = {
            "velocity_scale": velocity_scale,
            "heading_bias": heading_bias,
            "lateral_slip": lateral_slip,
            "velocity_bias": velocity_bias,
        }
        super().__init__(params=params, name=name)

    @property
    def velocity_scale(self) -> float:
        return self.params["velocity_scale"]

    @property
    def heading_bias(self) -> float:
        return self.params["heading_bias"]

    @property
    def lateral_slip(self) -> float:
        return self.params["lateral_slip"]

    @property
    def velocity_bias(self) -> float:
        return self.params["velocity_bias"]

    @classmethod
    def twin_default(cls) -> "UnicycleParameters":
        """Default parameters for twin (nominal model)."""
        return cls(
            velocity_scale=1.0,
            heading_bias=0.0,
            lateral_slip=0.0,
            velocity_bias=0.0,
            name="twin_params",
        )

    @classmethod
    def plant_default(
        cls,
        velocity_scale: float = 1.05,
        heading_bias: float = 0.02,
        lateral_slip: float = 0.03,
        velocity_bias: float = 0.0,
    ) -> "UnicycleParameters":
        """Default parameters for plant (with mismatch)."""
        return cls(
            velocity_scale=velocity_scale,
            heading_bias=heading_bias,
            lateral_slip=lateral_slip,
            velocity_bias=velocity_bias,
            name="plant_params",
        )


# =============================================================================
# Unicycle Model Base Class
# =============================================================================


class UnicycleModel(BaseModel):
    """
    Unicycle (differential-drive robot) model.

    State: x = [px, py, theta]^T
    Input: u = [v, omega]^T

    Parameters
    ----------
    params : UnicycleParameters or dict, optional
        Model parameters.
    dt : float
        Discretization timestep [s].
    integration_method : str
        Integration method: 'euler', 'rk2', or 'rk4'.
    name : str
        Model name identifier.
    """

    # State indices
    PX = 0  # x-position
    PY = 1  # y-position
    THETA = 2  # heading angle

    # Input indices
    V = 0  # linear velocity
    OMEGA = 1  # angular velocity

    def __init__(
        self,
        params: Optional[Union[UnicycleParameters, Dict[str, Any]]] = None,
        dt: float = 0.02,
        integration_method: str = "rk4",
        name: str = "unicycle",
    ):
        # Handle parameters
        if params is None:
            params = UnicycleParameters.twin_default()
        elif isinstance(params, dict):
            params = UnicycleParameters(**params)

        super().__init__(
            params=params,
            dt=dt,
            integration_method=integration_method,
            name=name,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_states(self) -> int:
        return 3

    @property
    def n_inputs(self) -> int:
        return 2

    @property
    def state_labels(self) -> list:
        return ["px", "py", "theta"]

    @property
    def input_labels(self) -> list:
        return ["v", "omega"]

    @property
    def position_indices(self) -> list:
        """Indices of position states."""
        return [self.PX, self.PY]

    @property
    def has_analytical_jacobians(self) -> bool:
        return True

    @property
    def has_analytical_discrete_jacobians(self) -> bool:
        # We provide analytical continuous Jacobians
        # Discrete Jacobians computed via chain rule or numerically
        return False

    # =========================================================================
    # Continuous Dynamics
    # =========================================================================

    def continuous_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """
        Compute continuous-time dynamics: dx/dt = f(x, u).

        Basic unicycle (twin):
            px_dot = v * cos(theta)
            py_dot = v * sin(theta)
            theta_dot = omega

        With mismatch (plant):
            px_dot = (velocity_scale * v + velocity_bias) * cos(theta)
                     - lateral_slip * v * sin(theta)
            py_dot = (velocity_scale * v + velocity_bias) * sin(theta)
                     + lateral_slip * v * cos(theta)
            theta_dot = omega + heading_bias

        Parameters
        ----------
        x : np.ndarray
            State [px, py, theta].
        u : np.ndarray
            Input [v, omega].

        Returns
        -------
        np.ndarray
            State derivative [px_dot, py_dot, theta_dot].
        """
        # Extract state
        theta = x[self.THETA]

        # Extract input
        v = u[self.V]
        omega = u[self.OMEGA]

        # Get parameters
        velocity_scale = self._params.get("velocity_scale", 1.0)
        heading_bias = self._params.get("heading_bias", 0.0)
        lateral_slip = self._params.get("lateral_slip", 0.0)
        velocity_bias = self._params.get("velocity_bias", 0.0)

        # Effective velocity
        v_eff = velocity_scale * v + velocity_bias

        # Trigonometric terms
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Compute derivatives
        px_dot = v_eff * cos_theta - lateral_slip * v * sin_theta
        py_dot = v_eff * sin_theta + lateral_slip * v * cos_theta
        theta_dot = omega + heading_bias

        return np.array([px_dot, py_dot, theta_dot])

    # =========================================================================
    # Analytical Jacobians
    # =========================================================================

    def _analytical_continuous_jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute analytical Jacobians of continuous dynamics.

        Returns df/dx (A_c) and df/du (B_c).
        """
        # Extract state and input
        theta = x[self.THETA]
        v = u[self.V]

        # Get parameters
        velocity_scale = self._params.get("velocity_scale", 1.0)
        velocity_bias = self._params.get("velocity_bias", 0.0)
        lateral_slip = self._params.get("lateral_slip", 0.0)

        v_eff = velocity_scale * v + velocity_bias

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # A_c = df/dx (3x3)
        # df/d(px) = 0, df/d(py) = 0
        # df/d(theta):
        #   d(px_dot)/d(theta) = -v_eff * sin(theta) - lateral_slip * v * cos(theta)
        #   d(py_dot)/d(theta) = v_eff * cos(theta) - lateral_slip * v * sin(theta)
        #   d(theta_dot)/d(theta) = 0

        A_c = np.zeros((3, 3))
        A_c[self.PX, self.THETA] = -v_eff * sin_theta - lateral_slip * v * cos_theta
        A_c[self.PY, self.THETA] = v_eff * cos_theta - lateral_slip * v * sin_theta
        # A_c[self.THETA, :] = 0

        # B_c = df/du (3x2)
        # df/d(v):
        #   d(px_dot)/d(v) = velocity_scale * cos(theta) - lateral_slip * sin(theta)
        #   d(py_dot)/d(v) = velocity_scale * sin(theta) + lateral_slip * cos(theta)
        #   d(theta_dot)/d(v) = 0
        # df/d(omega):
        #   d(px_dot)/d(omega) = 0
        #   d(py_dot)/d(omega) = 0
        #   d(theta_dot)/d(omega) = 1

        B_c = np.zeros((3, 2))
        B_c[self.PX, self.V] = velocity_scale * cos_theta - lateral_slip * sin_theta
        B_c[self.PY, self.V] = velocity_scale * sin_theta + lateral_slip * cos_theta
        B_c[self.THETA, self.OMEGA] = 1.0

        return A_c, B_c

    # =========================================================================
    # State Normalization
    # =========================================================================

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize state by wrapping heading angle to [-pi, pi].

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Normalized state.
        """
        x_normalized = x.copy()
        x_normalized[self.THETA] = wrap_angle(x[self.THETA])
        return x_normalized

    # =========================================================================
    # Default Parameters
    # =========================================================================

    def get_default_parameters(self) -> ModelParameters:
        """Get default parameters for the model."""
        return UnicycleParameters.twin_default()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_position(self, x: np.ndarray) -> np.ndarray:
        """Extract position from state."""
        return x[self.position_indices]

    def get_heading(self, x: np.ndarray) -> float:
        """Extract heading from state."""
        return x[self.THETA]

    def state_from_pose(
        self,
        px: float,
        py: float,
        theta: float,
    ) -> np.ndarray:
        """Create state vector from pose components."""
        return np.array([px, py, theta])

    def compute_forward_velocity(self, x: np.ndarray, x_dot: np.ndarray) -> float:
        """
        Compute forward velocity from state and state derivative.

        v = px_dot * cos(theta) + py_dot * sin(theta)
        """
        theta = x[self.THETA]
        px_dot = x_dot[self.PX]
        py_dot = x_dot[self.PY]
        return px_dot * np.cos(theta) + py_dot * np.sin(theta)


# =============================================================================
# Specialized Twin and Plant Models
# =============================================================================


class UnicycleTwin(UnicycleModel):
    """
    Unicycle twin model (nominal, no mismatch).

    This represents the known digital twin used for planning.
    """

    def __init__(
        self,
        dt: float = 0.02,
        integration_method: str = "rk4",
    ):
        super().__init__(
            params=UnicycleParameters.twin_default(),
            dt=dt,
            integration_method=integration_method,
            name="unicycle_twin",
        )


class UnicyclePlant(UnicycleModel):
    """
    Unicycle plant model (with mismatch).

    This represents the true physical system with unmodeled dynamics.

    Parameters
    ----------
    velocity_scale : float
        Multiplicative factor on velocity (>1 means faster than twin).
    heading_bias : float
        Additive bias on heading rate [rad/s].
    lateral_slip : float
        Lateral slip coefficient.
    velocity_bias : float
        Additive bias on velocity [m/s].
    dt : float
        Discretization timestep.
    integration_method : str
        Integration method.
    """

    def __init__(
        self,
        velocity_scale: float = 1.05,
        heading_bias: float = 0.02,
        lateral_slip: float = 0.03,
        velocity_bias: float = 0.0,
        dt: float = 0.02,
        integration_method: str = "rk4",
    ):
        params = UnicycleParameters.plant_default(
            velocity_scale=velocity_scale,
            heading_bias=heading_bias,
            lateral_slip=lateral_slip,
            velocity_bias=velocity_bias,
        )
        super().__init__(
            params=params,
            dt=dt,
            integration_method=integration_method,
            name="unicycle_plant",
        )

    @property
    def mismatch_params(self) -> dict:
        """Get mismatch parameters."""
        return {
            "velocity_scale": self._params.get("velocity_scale", 1.0),
            "heading_bias": self._params.get("heading_bias", 0.0),
            "lateral_slip": self._params.get("lateral_slip", 0.0),
            "velocity_bias": self._params.get("velocity_bias", 0.0),
        }


# =============================================================================
# Factory Class
# =============================================================================


class UnicycleFactory(ModelFactory):
    """
    Factory for creating unicycle plant-twin pairs.

    Example
    -------
    >>> pair = UnicycleFactory.create_pair(
    ...     velocity_scale=1.05,
    ...     heading_bias=0.02,
    ...     dt=0.02,
    ... )
    >>> twin = pair.twin
    >>> plant = pair.plant
    """

    @staticmethod
    def create_twin(
        dt: float = 0.02,
        integration_method: str = "rk4",
    ) -> UnicycleTwin:
        """Create unicycle twin model."""
        return UnicycleTwin(dt=dt, integration_method=integration_method)

    @staticmethod
    def create_plant(
        velocity_scale: float = 1.05,
        heading_bias: float = 0.02,
        lateral_slip: float = 0.03,
        velocity_bias: float = 0.0,
        dt: float = 0.02,
        integration_method: str = "rk4",
    ) -> UnicyclePlant:
        """Create unicycle plant model."""
        return UnicyclePlant(
            velocity_scale=velocity_scale,
            heading_bias=heading_bias,
            lateral_slip=lateral_slip,
            velocity_bias=velocity_bias,
            dt=dt,
            integration_method=integration_method,
        )

    @classmethod
    def create_pair(
        cls,
        velocity_scale: float = 1.05,
        heading_bias: float = 0.02,
        lateral_slip: float = 0.03,
        velocity_bias: float = 0.0,
        dt: float = 0.02,
        integration_method: str = "rk4",
    ) -> PlantTwinPair:
        """
        Create plant-twin pair with specified mismatch.

        Parameters
        ----------
        velocity_scale : float
            Plant velocity scale factor.
        heading_bias : float
            Plant heading bias [rad/s].
        lateral_slip : float
            Plant lateral slip coefficient.
        velocity_bias : float
            Plant velocity bias [m/s].
        dt : float
            Discretization timestep.
        integration_method : str
            Integration method.

        Returns
        -------
        PlantTwinPair
            Paired plant and twin models.
        """
        twin = cls.create_twin(dt=dt, integration_method=integration_method)
        plant = cls.create_plant(
            velocity_scale=velocity_scale,
            heading_bias=heading_bias,
            lateral_slip=lateral_slip,
            velocity_bias=velocity_bias,
            dt=dt,
            integration_method=integration_method,
        )
        return PlantTwinPair(twin=twin, plant=plant)

    @classmethod
    def from_config(cls, config) -> PlantTwinPair:
        """
        Create plant-twin pair from configuration.

        Parameters
        ----------
        config : Config
            Configuration object with system parameters.

        Returns
        -------
        PlantTwinPair
            Paired plant and twin models.
        """
        dt = config.simulation.dt
        mismatch = config.system.mismatch

        return cls.create_pair(
            velocity_scale=mismatch.get("velocity_scale", 1.05),
            heading_bias=mismatch.get("heading_bias", 0.02),
            lateral_slip=mismatch.get("lateral_slip", 0.03),
            velocity_bias=mismatch.get("velocity_bias", 0.0),
            dt=dt,
            integration_method="rk4",
        )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_unicycle_mismatch_bound(
    plant: UnicyclePlant,
    twin: UnicycleTwin,
    v_range: Tuple[float, float] = (-2.0, 2.0),
    omega_range: Tuple[float, float] = (-2.0, 2.0),
    n_samples: int = 1000,
) -> float:
    """
    Estimate mismatch bound gamma for unicycle models.

    Samples the state-input space and computes maximum mismatch.

    Parameters
    ----------
    plant : UnicyclePlant
        Plant model.
    twin : UnicycleTwin
        Twin model.
    v_range : tuple
        Range of linear velocity.
    omega_range : tuple
        Range of angular velocity.
    n_samples : int
        Number of samples.

    Returns
    -------
    float
        Estimated mismatch bound gamma.
    """
    max_mismatch = 0.0

    for _ in range(n_samples):
        # Sample state (position doesn't affect dynamics difference, only theta matters)
        theta = np.random.uniform(-np.pi, np.pi)
        x = np.array([0.0, 0.0, theta])

        # Sample input
        v = np.random.uniform(v_range[0], v_range[1])
        omega = np.random.uniform(omega_range[0], omega_range[1])
        u = np.array([v, omega])

        # Compute mismatch
        f_plant = plant.continuous_dynamics(x, u)
        f_twin = twin.continuous_dynamics(x, u)
        mismatch_norm = np.linalg.norm(f_plant - f_twin)

        max_mismatch = max(max_mismatch, mismatch_norm)

    return max_mismatch


def analytical_mismatch_bound(
    velocity_scale: float,
    heading_bias: float,
    lateral_slip: float,
    velocity_bias: float,
    v_max: float,
) -> float:
    """
    Compute analytical upper bound on mismatch.

    For unicycle, the mismatch is:
        Δ = [(velocity_scale - 1) * v + velocity_bias] * [cos(theta), sin(theta)]
            + lateral_slip * v * [-sin(theta), cos(theta)]
            + [0, 0, heading_bias]

    The bound is:
        ||Δ|| <= sqrt([(velocity_scale-1)*v_max + velocity_bias]^2 + [lateral_slip*v_max]^2)
                 + |heading_bias|

    Parameters
    ----------
    velocity_scale : float
        Velocity scale factor.
    heading_bias : float
        Heading bias.
    lateral_slip : float
        Lateral slip coefficient.
    velocity_bias : float
        Velocity bias.
    v_max : float
        Maximum velocity magnitude.

    Returns
    -------
    float
        Analytical mismatch bound.
    """
    # Position mismatch bound
    v_mismatch = abs(velocity_scale - 1) * v_max + abs(velocity_bias)
    slip_mismatch = abs(lateral_slip) * v_max
    position_bound = np.sqrt(v_mismatch**2 + slip_mismatch**2)

    # Heading mismatch bound
    heading_bound = abs(heading_bias)

    # Total bound (using 2-norm of stacked vector)
    return np.sqrt(position_bound**2 + heading_bound**2)


def create_unicycle_trajectory(  # noqa: C901
    model: UnicycleModel,
    x_init: np.ndarray,
    x_final: np.ndarray,
    N: int,
    method: str = "straight_line",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple reference trajectory for unicycle.

    Parameters
    ----------
    model : UnicycleModel
        Unicycle model instance.
    x_init : np.ndarray
        Initial state [px, py, theta].
    x_final : np.ndarray
        Final state [px, py, theta].
    N : int
        Number of steps.
    method : str
        Trajectory generation method: 'straight_line' or 'smooth'.

    Returns
    -------
    x_ref : np.ndarray
        Reference state trajectory (N+1, 3).
    u_ref : np.ndarray
        Reference input trajectory (N, 2).
    """
    dt = model.dt

    if method == "straight_line":
        # Simple straight-line interpolation with heading adjustment
        x_ref = np.zeros((N + 1, 3))
        u_ref = np.zeros((N, 2))

        # Interpolate position
        for i in range(N + 1):
            alpha = i / N
            x_ref[i, 0] = (1 - alpha) * x_init[0] + alpha * x_final[0]
            x_ref[i, 1] = (1 - alpha) * x_init[1] + alpha * x_final[1]

        # Compute heading as direction of motion
        for i in range(N):
            dx = x_ref[i + 1, 0] - x_ref[i, 0]
            dy = x_ref[i + 1, 1] - x_ref[i, 1]
            x_ref[i, 2] = np.arctan2(dy, dx)

        x_ref[N, 2] = x_final[2]  # Final heading

        # Compute reference inputs
        for i in range(N):
            # Velocity from position change
            dx = x_ref[i + 1, 0] - x_ref[i, 0]
            dy = x_ref[i + 1, 1] - x_ref[i, 1]
            v = np.sqrt(dx**2 + dy**2) / dt

            # Angular velocity from heading change
            dtheta = wrap_angle(x_ref[i + 1, 2] - x_ref[i, 2])
            omega = dtheta / dt

            u_ref[i] = np.array([v, omega])

        return x_ref, u_ref

    elif method == "smooth":
        # Smooth trajectory using minimum-jerk profile
        # Time vector
        t = np.linspace(0, 1, N + 1)

        # Minimum-jerk profile: s(t) = 10*t^3 - 15*t^4 + 6*t^5
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        s_dot = (30 * t**2 - 60 * t**3 + 30 * t**4) / (N * dt)

        x_ref = np.zeros((N + 1, 3))
        u_ref = np.zeros((N, 2))

        # Interpolate position
        delta_pos = x_final[:2] - x_init[:2]
        for i in range(N + 1):
            x_ref[i, :2] = x_init[:2] + s[i] * delta_pos

        # Compute heading
        heading_to_goal = np.arctan2(delta_pos[1], delta_pos[0])
        for i in range(N + 1):
            x_ref[i, 2] = heading_to_goal

        x_ref[N, 2] = x_final[2]

        # Compute reference inputs
        dist = np.linalg.norm(delta_pos)
        for i in range(N):
            u_ref[i, 0] = s_dot[i] * dist  # velocity
            u_ref[i, 1] = 0.0  # angular velocity (constant heading)

        # Adjust final steps for heading change
        if N > 10:
            final_dtheta = wrap_angle(x_final[2] - heading_to_goal)
            for i in range(N - 10, N):
                alpha = (i - (N - 10)) / 10
                x_ref[i, 2] = heading_to_goal + alpha * final_dtheta
                u_ref[i, 1] = final_dtheta / (10 * dt)

        return x_ref, u_ref

    else:
        raise ValueError(f"Unknown trajectory method: {method}")
