# ddfs/ddfs/models/unicycle.py

"""
Unicycle dynamics model.

This module implements the kinematic unicycle model used as the digital twin
for trajectory planning and controller synthesis.

State: x = [x, y, θ]
    - x, y: position in 2D plane (meters)
    - θ: heading angle (radians)

Input: u = [v, ω]
    - v: linear velocity (m/s)
    - ω: angular velocity (rad/s)

Dynamics:
    ẋ = v cos(θ)
    ẏ = v sin(θ)
    θ̇ = ω

Notes
-----
- This is a KINEMATIC model (no mass, no inertia)
- Constraints are defined in ddfs.core.constraints
- Plant mismatch is defined in ddfs.models.plant
"""

import jax.numpy as jnp

from ddfs.models.base import TwinModel


class UnicycleTwin(TwinModel):
    """
    Kinematic unicycle model (digital twin).

    This is the nominal model used for planning in Phase 1.
    The actual plant may have model mismatch (velocity scaling, slip, etc.).

    State dimension: n = 3
    Input dimension: m = 2

    Parameters
    ----------
    dt : float, optional
        Discretization timestep (seconds), by default 0.1

    Examples
    --------
    >>> from ddfs.models.unicycle import UnicycleTwin
    >>> import jax.numpy as jnp
    >>>
    >>> twin = UnicycleTwin(dt=0.1)
    >>> x = jnp.array([0.0, 0.0, 0.0])  # At origin, facing right
    >>> u = jnp.array([1.0, 0.5])       # Moving forward with turning
    >>> x_next = twin.step(x, u)
    >>> print(x_next)
    """

    def __init__(self, dt: float = 0.1):
        """
        Initialize unicycle twin model.

        Parameters
        ----------
        dt : float, optional
            Discretization timestep (seconds), by default 0.1
        """
        super().__init__(dt=dt)

    @property
    def state_dim(self) -> int:
        """State dimension n = 3."""
        return 3

    @property
    def input_dim(self) -> int:
        """Input dimension m = 2."""
        return 2

    def _dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Kinematic unicycle dynamics: ẋ = f(x, u).

        Implements the standard kinematic unicycle model:
            ẋ = v cos(θ)
            ẏ = v sin(θ)
            θ̇ = ω

        Parameters
        ----------
        x : jnp.ndarray
            State [x, y, θ], shape (3,)
        u : jnp.ndarray
            Input [v, ω], shape (2,)

        Returns
        -------
        x_dot : jnp.ndarray
            State derivative [ẋ, ẏ, θ̇], shape (3,)
        """
        # Extract state
        theta = x[2]

        # Extract input
        v = u[0]  # linear velocity
        omega = u[1]  # angular velocity

        # Kinematic unicycle equations
        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)
        theta_dot = omega

        return jnp.array([x_dot, y_dot, theta_dot])

    def normalize_state(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize state by wrapping angle to [-π, π].

        Parameters
        ----------
        x : jnp.ndarray
            State [x, y, θ], shape (3,)

        Returns
        -------
        x_normalized : jnp.ndarray
            Normalized state with θ ∈ [-π, π], shape (3,)

        Examples
        --------
        >>> twin = UnicycleTwin()
        >>> x = jnp.array([1.0, 2.0, 4.0])  # θ > π
        >>> x_norm = twin.normalize_state(x)
        >>> print(x_norm[2])  # Should be wrapped to [-π, π]
        """
        x_norm = x.at[2].set(jnp.arctan2(jnp.sin(x[2]), jnp.cos(x[2])))
        return x_norm

    def state_distance(self, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
        """
        Compute distance between two states.

        Uses Euclidean distance for position and angular difference for heading:
            d = √[(x₁-x₂)² + (y₁-y₂)² + (θ₁-θ₂)²]

        where angular difference is wrapped to [-π, π].

        Parameters
        ----------
        x1 : jnp.ndarray
            First state [x, y, θ], shape (3,)
        x2 : jnp.ndarray
            Second state [x, y, θ], shape (3,)

        Returns
        -------
        distance : float
            Distance between states

        Examples
        --------
        >>> twin = UnicycleTwin()
        >>> x1 = jnp.array([0.0, 0.0, 0.0])
        >>> x2 = jnp.array([1.0, 1.0, jnp.pi/4])
        >>> dist = twin.state_distance(x1, x2)
        """
        # Position distance
        pos_diff = x1[:2] - x2[:2]
        pos_dist = jnp.linalg.norm(pos_diff)

        # Angular distance (wrapped to [-π, π])
        theta_diff = jnp.arctan2(jnp.sin(x1[2] - x2[2]), jnp.cos(x1[2] - x2[2]))

        # Combined distance
        return float(jnp.sqrt(pos_dist**2 + theta_diff**2))

    def __repr__(self) -> str:
        """String representation."""
        return f"UnicycleTwin(state_dim={self.state_dim}, input_dim={self.input_dim}, dt={self.dt})"


def create_unicycle_example() -> dict:
    """
    Create example unicycle configuration matching specifications.

    Based on typical unicycle problem:
        n = 3, m = 2
        tf = 8.0 seconds
        N = 61 timesteps
        dt = tf/N ≈ 0.131 seconds
        x_0 = [1.0, 1.0, 0]
        x_des = [10, 5.5, 0]
        v ∈ [0, 2] m/s
        ω ∈ [-2, 2] rad/s
        workspace: x ∈ [0, 12], y ∈ [0, 8]
        obstacles: 2 circles at [4, 3] and [8, 3], radius 1.0

    Returns
    -------
    config : dict
        Configuration dictionary with system, planning, constraints, and obstacles

    Examples
    --------
    >>> from ddfs.models.unicycle import create_unicycle_example
    >>> config = create_unicycle_example()
    >>> print(config['system']['dt'])
    0.13114754098360656
    """
    return {
        "system": {
            "name": "unicycle",
            "state_dim": 3,
            "input_dim": 2,
            "dt": 8.0 / 61,  # ≈ 0.131 seconds
        },
        "planning": {
            "tf": 8.0,
            "N": 61,
            "x0": [1.0, 1.0, 0.0],
            "xf": [10.0, 5.5, 0.0],
        },
        "constraints": {
            "state_bounds": {
                "x_min": 0.0,
                "x_max": 12.0,
                "y_min": 0.0,
                "y_max": 8.0,
                "theta_min": -3.141592653589793,
                "theta_max": 3.141592653589793,
            },
            "input_bounds": {
                "v_min": 0.0,
                "v_max": 2.0,
                "omega_min": -2.0,
                "omega_max": 2.0,
            },
        },
        "obstacles": [
            {
                "id": "obs_1",
                "type": "circle",
                "center": [4.0, 3.0],
                "radius": 1.0,
                "safety_margin": 0.25,
            },
            {
                "id": "obs_2",
                "type": "circle",
                "center": [8.0, 3.0],
                "radius": 1.0,
                "safety_margin": 0.25,
            },
        ],
        "plant_mismatch": {
            "velocity_scale": 0.95,  # Plant moves 5% slower
            "angular_scale": 1.03,  # Plant turns 3% faster
            "slip_coefficient": 0.02,  # Lateral slip
        },
    }
