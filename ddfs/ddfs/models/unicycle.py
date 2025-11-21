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
"""

from typing import Any, Dict

import jax.numpy as jnp

from .base import TwinModel


class UnicycleTwin(TwinModel):
    """
    Kinematic unicycle model (digital twin).

    This is the approximate model used for planning in Phase 1.
    The actual plant may have model mismatch (velocity scaling, slip, etc.).

    State dimension: n = 3
    Input dimension: m = 2

    Constraints (typical):
        State: x ∈ workspace, θ ∈ [-π, π]
        Input: v ∈ [v_min, v_max], ω ∈ [-ω_max, ω_max]
    """

    def __init__(self, dt: float = 0.1):
        """
        Initialize unicycle twin model.

        Args:
            dt: Discretization timestep (seconds)
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

        Args:
            x: State [x, y, θ] (3,)
            u: Input [v, ω] (2,)

        Returns:
            State derivative [ẋ, ẏ, θ̇] (3,)
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

        Args:
            x: State [x, y, θ] (3,)

        Returns:
            Normalized state with θ ∈ [-π, π] (3,)
        """
        x_norm = x.at[2].set(jnp.arctan2(jnp.sin(x[2]), jnp.cos(x[2])))
        return x_norm

    def state_distance(self, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
        """
        Compute distance between two states.

        Uses Euclidean distance for position and angular difference for heading:
        d = √[(x₁-x₂)² + (y₁-y₂)² + (θ₁-θ₂)²]

        where angular difference wraps around ±π.

        Args:
            x1: First state [x, y, θ] (3,)
            x2: Second state [x, y, θ] (3,)

        Returns:
            Distance between states
        """
        # Position distance
        pos_diff = x1[:2] - x2[:2]
        pos_dist = jnp.linalg.norm(pos_diff)

        # Angular distance (wrapped)
        theta_diff = jnp.arctan2(jnp.sin(x1[2] - x2[2]), jnp.cos(x1[2] - x2[2]))

        # Combined distance
        return jnp.sqrt(pos_dist**2 + theta_diff**2)

    def __repr__(self) -> str:
        return f"UnicycleTwin(state_dim={self.state_dim}, input_dim={self.input_dim}, dt={self.dt})"


class UnicycleConstraints:
    """
    State and input constraints for unicycle.

    Typical constraints:
        - Workspace bounds: x ∈ [x_min, x_max], y ∈ [y_min, y_max]
        - Heading: θ ∈ [-π, π] (always satisfied by normalization)
        - Velocity: v ∈ [v_min, v_max]
        - Angular velocity: ω ∈ [-ω_max, ω_max]
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

        Args:
            x_min: Minimum x position (m)
            x_max: Maximum x position (m)
            y_min: Minimum y position (m)
            y_max: Maximum y position (m)
            v_min: Minimum linear velocity (m/s)
            v_max: Maximum linear velocity (m/s)
            omega_max: Maximum angular velocity magnitude (rad/s)
        """
        # State bounds
        self.x_min = jnp.array([x_min, y_min, -jnp.pi])
        self.x_max = jnp.array([x_max, y_max, jnp.pi])

        # Input bounds
        self.u_min = jnp.array([v_min, -omega_max])
        self.u_max = jnp.array([v_max, omega_max])

    def check_state(self, x: jnp.ndarray) -> bool:
        """
        Check if state satisfies constraints.

        Args:
            x: State [x, y, θ] (3,)

        Returns:
            True if x is within bounds
        """
        return jnp.all(x >= self.x_min) and jnp.all(x <= self.x_max)

    def check_input(self, u: jnp.ndarray) -> bool:
        """
        Check if input satisfies constraints.

        Args:
            u: Input [v, ω] (2,)

        Returns:
            True if u is within bounds
        """
        return jnp.all(u >= self.u_min) and jnp.all(u <= self.u_max)

    def clip_state(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Clip state to satisfy constraints.

        Args:
            x: State [x, y, θ] (3,)

        Returns:
            Clipped state (3,)
        """
        return jnp.clip(x, self.x_min, self.x_max)

    def clip_input(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Clip input to satisfy constraints.

        Args:
            u: Input [v, ω] (2,)

        Returns:
            Clipped input (2,)
        """
        return jnp.clip(u, self.u_min, self.u_max)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert constraints to dictionary format.

        Returns:
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

        Args:
            config: Configuration with 'state_bounds' and 'input_bounds'

        Returns:
            UnicycleConstraints instance

        Example config:
            {
                'state_bounds': {
                    'x_min': 0.0, 'x_max': 10.0,
                    'y_min': 0.0, 'y_max': 8.0
                },
                'input_bounds': {
                    'v_min': 0.0, 'v_max': 2.0,
                    'omega_max': 2.0
                }
            }
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
        return (
            f"UnicycleConstraints("
            f"x∈[{self.x_min[0]:.1f}, {self.x_max[0]:.1f}], "
            f"y∈[{self.x_min[1]:.1f}, {self.x_max[1]:.1f}], "
            f"v∈[{self.u_min[0]:.1f}, {self.u_max[0]:.1f}], "
            f"ω∈[{self.u_min[1]:.1f}, {self.u_max[1]:.1f}])"
        )


def create_unicycle_example() -> Dict[str, Any]:
    """
    Create example unicycle configuration matching your specifications.

    Based on:
        n = 3, m = 2
        tf = 8, T = 61, dt = tf/T
        x_0 = [1.0, 1.0, 0]
        x_des = [10, 5.5, 0]
        u1_max = 2, u1_min = 0, u2_max = 2
        num_obs = 2
        obs = [[4, 3], [8, 3]]
        obs_r = 1

    Returns:
        Configuration dictionary
    """
    return {
        "system": {
            "name": "unicycle",
            "state_dim": 3,
            "input_dim": 2,
            "dt": 8.0 / 61,  # ≈ 0.131 seconds
        },
        "planning": {"tf": 8.0, "N": 61, "x0": [1.0, 1.0, 0.0], "xf": [10.0, 5.5, 0.0]},
        "constraints": {
            "state_bounds": {"x_min": 0.0, "x_max": 12.0, "y_min": 0.0, "y_max": 8.0},
            "input_bounds": {"v_min": 0.0, "v_max": 2.0, "omega_max": 2.0},
        },
        "obstacles": [
            {"id": "obs_1", "type": "circle", "center": [4.0, 3.0], "radius": 1.0},
            {"id": "obs_2", "type": "circle", "center": [8.0, 3.0], "radius": 1.0},
        ],
        "plant_mismatch": {"velocity_scale": 0.95, "angular_scale": 1.03, "slip_coefficient": 0.02},
    }
