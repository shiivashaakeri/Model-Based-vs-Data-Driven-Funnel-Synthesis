# ddfs/ddfs/core/constraints.py

"""
Unified constraint definitions for all systems.

This module provides constraint classes for state and input bounds
for all supported systems (unicycle, quadrotor).

Constraints are used in:
    - Phase 1: Planning (trajectory optimization bounds)
    - Phase 4: Funnel synthesis (ellipsoid computation)
    - Phase 6: Deployment (safety monitoring)

Key Classes
-----------
SystemConstraints : Abstract base class for all constraints
UnicycleConstraints : State and input constraints for unicycle
QuadrotorConstraints : State and input constraints for quadrotor
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import jax.numpy as jnp


class SystemConstraints(ABC):
    """
    Abstract base class for system constraints.

    All constraint classes must implement:
        - check_state(x): Verify state satisfies constraints
        - check_input(u): Verify input satisfies constraints
        - clip_input(u): Clip input to bounds
        - to_dict(): Convert to dictionary format
        - from_config(): Create from configuration dictionary
    """

    @abstractmethod
    def check_state(self, x: jnp.ndarray) -> bool:
        """
        Check if state satisfies constraints.

        Parameters
        ----------
        x : jnp.ndarray
            State vector

        Returns
        -------
        valid : bool
            True if state is within bounds
        """
        pass

    @abstractmethod
    def check_input(self, u: jnp.ndarray) -> bool:
        """
        Check if input satisfies constraints.

        Parameters
        ----------
        u : jnp.ndarray
            Input vector

        Returns
        -------
        valid : bool
            True if input is within bounds
        """
        pass

    @abstractmethod
    def clip_input(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Clip input to satisfy constraints.

        Parameters
        ----------
        u : jnp.ndarray
            Input vector

        Returns
        -------
        u_clipped : jnp.ndarray
            Clipped input
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert constraints to dictionary format.

        Returns
        -------
        config : dict
            Dictionary with constraint parameters
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "SystemConstraints":
        """
        Create constraints from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration with 'state_bounds' and 'input_bounds'

        Returns
        -------
        constraints : SystemConstraints
            Constraint object
        """
        pass


class UnicycleConstraints(SystemConstraints):
    """
    State and input constraints for unicycle.

    State constraints:
        - Workspace bounds: x ∈ [x_min, x_max], y ∈ [y_min, y_max]
        - Heading: θ ∈ [-π, π] (always satisfied by normalization)

    Input constraints:
        - Velocity: v ∈ [v_min, v_max]
        - Angular velocity: ω ∈ [-ω_max, ω_max]

    Parameters
    ----------
    x_min : float, optional
        Minimum x position (m), by default -10.0
    x_max : float, optional
        Maximum x position (m), by default 10.0
    y_min : float, optional
        Minimum y position (m), by default -10.0
    y_max : float, optional
        Maximum y position (m), by default 10.0
    v_min : float, optional
        Minimum linear velocity (m/s), by default 0.0
    v_max : float, optional
        Maximum linear velocity (m/s), by default 2.0
    omega_max : float, optional
        Maximum angular velocity magnitude (rad/s), by default 2.0

    Attributes
    ----------
    x_min : jnp.ndarray
        Lower state bounds [x_min, y_min, -π], shape (3,)
    x_max : jnp.ndarray
        Upper state bounds [x_max, y_max, π], shape (3,)
    u_min : jnp.ndarray
        Lower input bounds [v_min, -ω_max], shape (2,)
    u_max : jnp.ndarray
        Upper input bounds [v_max, ω_max], shape (2,)

    Examples
    --------
    >>> from ddfs.core.constraints import UnicycleConstraints
    >>> import jax.numpy as jnp
    >>>
    >>> constraints = UnicycleConstraints(
    ...     x_min=0.0, x_max=12.0,
    ...     y_min=0.0, y_max=8.0,
    ...     v_min=0.0, v_max=2.0,
    ...     omega_max=2.0
    ... )
    >>>
    >>> x = jnp.array([5.0, 4.0, 0.5])  # Valid state
    >>> print(constraints.check_state(x))
    True
    >>>
    >>> u = jnp.array([3.0, 1.0])  # v exceeds v_max
    >>> u_clipped = constraints.clip_input(u)
    >>> print(u_clipped)
    [2.0, 1.0]
    """

    def __init__(
        self,
        x_min: float = -10.0,
        x_max: float = 10.0,
        y_min: float = -10.0,
        y_max: float = 10.0,
        v_min: float = 0.0,
        v_max: float = 2.0,
        omega_max: float = 2.0,
    ):
        """
        Initialize unicycle constraints.

        Parameters
        ----------
        x_min : float, optional
            Minimum x position (m)
        x_max : float, optional
            Maximum x position (m)
        y_min : float, optional
            Minimum y position (m)
        y_max : float, optional
            Maximum y position (m)
        v_min : float, optional
            Minimum linear velocity (m/s)
        v_max : float, optional
            Maximum linear velocity (m/s)
        omega_max : float, optional
            Maximum angular velocity magnitude (rad/s)
        """
        # State bounds [x, y, θ]
        self.x_min = jnp.array([x_min, y_min, -jnp.pi])
        self.x_max = jnp.array([x_max, y_max, jnp.pi])

        # Input bounds [v, ω]
        self.u_min = jnp.array([v_min, -omega_max])
        self.u_max = jnp.array([v_max, omega_max])

    def check_state(self, x: jnp.ndarray) -> bool:
        """
        Check if state satisfies constraints.

        Parameters
        ----------
        x : jnp.ndarray
            State [x, y, θ], shape (3,)

        Returns
        -------
        valid : bool
            True if x is within bounds
        """
        return bool(jnp.all(x >= self.x_min) and jnp.all(x <= self.x_max))

    def check_input(self, u: jnp.ndarray) -> bool:
        """
        Check if input satisfies constraints.

        Parameters
        ----------
        u : jnp.ndarray
            Input [v, ω], shape (2,)

        Returns
        -------
        valid : bool
            True if u is within bounds
        """
        return bool(jnp.all(u >= self.u_min) and jnp.all(u <= self.u_max))

    def clip_state(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Clip state to satisfy constraints.

        Parameters
        ----------
        x : jnp.ndarray
            State [x, y, θ], shape (3,)

        Returns
        -------
        x_clipped : jnp.ndarray
            Clipped state, shape (3,)
        """
        return jnp.clip(x, self.x_min, self.x_max)

    def clip_input(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Clip input to satisfy constraints.

        Parameters
        ----------
        u : jnp.ndarray
            Input [v, ω], shape (2,)

        Returns
        -------
        u_clipped : jnp.ndarray
            Clipped input, shape (2,)
        """
        return jnp.clip(u, self.u_min, self.u_max)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert constraints to dictionary format.

        Returns
        -------
        config : dict
            Dictionary with constraint parameters
        """
        return {
            "state_bounds": {
                "x_min": float(self.x_min[0]),
                "x_max": float(self.x_max[0]),
                "y_min": float(self.x_min[1]),
                "y_max": float(self.x_max[1]),
                "theta_min": float(self.x_min[2]),
                "theta_max": float(self.x_max[2]),
            },
            "input_bounds": {
                "v_min": float(self.u_min[0]),
                "v_max": float(self.u_max[0]),
                "omega_min": float(self.u_min[1]),
                "omega_max": float(self.u_max[1]),
            },
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "UnicycleConstraints":
        """
        Create constraints from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration with 'state_bounds' and 'input_bounds'

        Returns
        -------
        constraints : UnicycleConstraints
            Constraint object

        Examples
        --------
        >>> config = {
        ...     'state_bounds': {
        ...         'x_min': 0.0, 'x_max': 10.0,
        ...         'y_min': 0.0, 'y_max': 8.0
        ...     },
        ...     'input_bounds': {
        ...         'v_min': 0.0, 'v_max': 2.0,
        ...         'omega_max': 2.0
        ...     }
        ... }
        >>> constraints = UnicycleConstraints.from_config(config)
        """
        state_bounds = config.get("state_bounds", {})
        input_bounds = config.get("input_bounds", {})

        return cls(
            x_min=state_bounds.get("x_min", -10.0),
            x_max=state_bounds.get("x_max", 10.0),
            y_min=state_bounds.get("y_min", -10.0),
            y_max=state_bounds.get("y_max", 10.0),
            v_min=input_bounds.get("v_min", 0.0),
            v_max=input_bounds.get("v_max", 2.0),
            omega_max=input_bounds.get("omega_max", 2.0),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnicycleConstraints("
            f"x∈[{self.x_min[0]:.1f}, {self.x_max[0]:.1f}], "
            f"y∈[{self.x_min[1]:.1f}, {self.x_max[1]:.1f}], "
            f"v∈[{self.u_min[0]:.1f}, {self.u_max[0]:.1f}], "
            f"ω∈[{self.u_min[1]:.1f}, {self.u_max[1]:.1f}])"
        )


class QuadrotorConstraints(SystemConstraints):
    """
    State and input constraints for quadrotor.

    State constraints:
        - Position: p ∈ [p_min, p_max]
        - Velocity: ||v|| ≤ v_max
        - Angular velocity: ||ω|| ≤ ω_max
        - Quaternion: always normalized (no explicit bounds)

    Input constraints:
        - Thrust: T ∈ [T_min, T_max]
        - Torques: τ ∈ [τ_min, τ_max]

    Parameters
    ----------
    x_min : float, optional
        Minimum x position (m), by default -5.0
    x_max : float, optional
        Maximum x position (m), by default 10.0
    y_min : float, optional
        Minimum y position (m), by default -5.0
    y_max : float, optional
        Maximum y position (m), by default 10.0
    z_min : float, optional
        Minimum z position (m, NED: negative is up), by default -5.0
    z_max : float, optional
        Maximum z position (m), by default 0.5
    v_max : float, optional
        Maximum velocity magnitude (m/s), by default 5.0
    omega_max : float, optional
        Maximum angular velocity magnitude (rad/s), by default 5.0
    T_min : float, optional
        Minimum thrust (N), by default 0.0
    T_max : float, optional
        Maximum thrust (N), by default 1.0
    tau_max : float, optional
        Maximum torque magnitude (N·m), by default 0.1

    Attributes
    ----------
    p_min : jnp.ndarray
        Minimum position bounds, shape (3,)
    p_max : jnp.ndarray
        Maximum position bounds, shape (3,)
    v_max : float
        Maximum velocity magnitude
    omega_max : float
        Maximum angular velocity magnitude
    u_min : jnp.ndarray
        Lower input bounds [T_min, -τ_max, -τ_max, -τ_max], shape (4,)
    u_max : jnp.ndarray
        Upper input bounds [T_max, τ_max, τ_max, τ_max], shape (4,)

    Examples
    --------
    >>> from ddfs.core.constraints import QuadrotorConstraints
    >>> import jax.numpy as jnp
    >>>
    >>> constraints = QuadrotorConstraints(
    ...     x_min=0.0, x_max=8.0,
    ...     y_min=0.0, y_max=8.0,
    ...     z_min=-5.0, z_max=0.5,
    ...     v_max=5.0, omega_max=5.0,
    ...     T_min=0.0, T_max=1.0, tau_max=0.1
    ... )
    >>>
    >>> # Check a valid state
    >>> x = jnp.zeros(13)
    >>> x = x.at[0:3].set(jnp.array([4.0, 4.0, -2.0]))  # Position
    >>> x = x.at[6].set(1.0)  # Identity quaternion
    >>> print(constraints.check_state(x))
    True
    """

    def __init__(
        self,
        x_min: float = -5.0,
        x_max: float = 10.0,
        y_min: float = -5.0,
        y_max: float = 10.0,
        z_min: float = -5.0,
        z_max: float = 0.5,
        v_max: float = 5.0,
        omega_max: float = 5.0,
        T_min: float = 0.0,
        T_max: float = 1.0,
        tau_max: float = 0.1,
    ):
        """
        Initialize quadrotor constraints.

        Parameters
        ----------
        x_min, x_max : float
            Position bounds in x (m)
        y_min, y_max : float
            Position bounds in y (m)
        z_min, z_max : float
            Position bounds in z (m) (NED: negative is up)
        v_max : float
            Maximum velocity magnitude (m/s)
        omega_max : float
            Maximum angular velocity magnitude (rad/s)
        T_min, T_max : float
            Thrust bounds (N)
        tau_max : float
            Maximum torque magnitude (N·m)
        """
        # State bounds (13D: [p, v, q, ω])
        # Note: quaternion is always normalized, so no explicit bounds
        self.p_min = jnp.array([x_min, y_min, z_min])
        self.p_max = jnp.array([x_max, y_max, z_max])
        self.v_max = v_max
        self.omega_max = omega_max

        # Input bounds (4D: [T, τx, τy, τz])
        self.u_min = jnp.array([T_min, -tau_max, -tau_max, -tau_max])
        self.u_max = jnp.array([T_max, tau_max, tau_max, tau_max])

    def check_state(self, x: jnp.ndarray) -> bool:
        """
        Check if state satisfies constraints.

        Parameters
        ----------
        x : jnp.ndarray
            State [p, v, q, ω], shape (13,)

        Returns
        -------
        valid : bool
            True if state is within bounds
        """
        pos = x[0:3]
        vel = x[3:6]
        omega = x[10:13]

        pos_ok = jnp.all(pos >= self.p_min) and jnp.all(pos <= self.p_max)
        vel_ok = jnp.linalg.norm(vel) <= self.v_max
        omega_ok = jnp.linalg.norm(omega) <= self.omega_max

        return bool(pos_ok and vel_ok and omega_ok)

    def check_input(self, u: jnp.ndarray) -> bool:
        """
        Check if input satisfies constraints.

        Parameters
        ----------
        u : jnp.ndarray
            Input [T, τx, τy, τz], shape (4,)

        Returns
        -------
        valid : bool
            True if input is within bounds
        """
        return bool(jnp.all(u >= self.u_min) and jnp.all(u <= self.u_max))

    def clip_input(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Clip input to satisfy constraints.

        Parameters
        ----------
        u : jnp.ndarray
            Input [T, τx, τy, τz], shape (4,)

        Returns
        -------
        u_clipped : jnp.ndarray
            Clipped input, shape (4,)
        """
        return jnp.clip(u, self.u_min, self.u_max)

    def to_dict(self) -> Dict[str, Any]:
        """Convert constraints to dictionary format."""
        return {
            "state_bounds": {
                "x_min": float(self.p_min[0]),
                "x_max": float(self.p_max[0]),
                "y_min": float(self.p_min[1]),
                "y_max": float(self.p_max[1]),
                "z_min": float(self.p_min[2]),
                "z_max": float(self.p_max[2]),
                "v_max": float(self.v_max),
                "omega_max": float(self.omega_max),
            },
            "input_bounds": {
                "T_min": float(self.u_min[0]),
                "T_max": float(self.u_max[0]),
                "tau_max": float(self.u_max[1]),
            },
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuadrotorConstraints":
        """Create constraints from configuration dictionary."""
        state_bounds = config.get("state_bounds", {})
        input_bounds = config.get("input_bounds", {})

        return cls(
            x_min=state_bounds.get("x_min", -5.0),
            x_max=state_bounds.get("x_max", 10.0),
            y_min=state_bounds.get("y_min", -5.0),
            y_max=state_bounds.get("y_max", 10.0),
            z_min=state_bounds.get("z_min", -5.0),
            z_max=state_bounds.get("z_max", 0.5),
            v_max=state_bounds.get("v_max", 5.0),
            omega_max=state_bounds.get("omega_max", 5.0),
            T_min=input_bounds.get("T_min", 0.0),
            T_max=input_bounds.get("T_max", 1.0),
            tau_max=input_bounds.get("tau_max", 0.1),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QuadrotorConstraints("
            f"p∈[{self.p_min}, {self.p_max}], "
            f"v_max={self.v_max:.1f}, "
            f"ω_max={self.omega_max:.1f}, "
            f"T∈[{self.u_min[0]:.2f}, {self.u_max[0]:.2f}])"
        )
