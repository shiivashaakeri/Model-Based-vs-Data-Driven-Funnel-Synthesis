# ddfs/ddfs/models/quadrotor.py
"""
Quadrotor dynamics model.

This module implements the full 3D quadrotor model with quaternion-based
attitude representation used as the digital twin for trajectory planning
and controller synthesis.

State: x = [p, v, q, ω] (13D)
    - p: position [x, y, z] in inertial frame (3,)
    - v: velocity [vx, vy, vz] in inertial frame (3,)
    - q: quaternion [qw, qx, qy, qz] for orientation (4,)
    - ω: angular velocity [ωx, ωy, ωz] in body frame (3,)

Input: u = [T, τ] (4D)
    - T: total thrust (N)
    - τ: torques [τx, τy, τz] in body frame (N⋅m) (3,)

Dynamics:
    ṗ = v
    v̇ = (R(q) * [0, 0, -T]ᵀ + [0, 0, mg]ᵀ) / m
    q̇ = 0.5 * Ω(ω) * q
    ω̇ = J⁻¹ * (τ - ω x (J * ω))

where R(q) is the rotation matrix from body to inertial frame.
"""

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp

from .base import TwinModel


class QuadrotorTwin(TwinModel):
    """
    Full 3D quadrotor model with quaternion attitude (digital twin).

    This is the approximate model used for planning in Phase 1.
    The actual plant may have model mismatch (mass, inertia, drag, etc.).

    State dimension: n = 13
    Input dimension: m = 4

    Convention: NED (North-East-Down) frame
        - z axis points downward
        - gravity is positive in z direction

    Parameters:
        - m: mass (kg)
        - J: inertia tensor (kg⋅m²) - 3x3 diagonal matrix
        - g: gravitational acceleration (m/s²)
    """

    def __init__(
        self,
        mass: float = 0.0293,  # kg (from your spec)
        inertia: Optional[jnp.ndarray] = None,
        gravity: float = 9.81,
        dt: float = 0.1,
    ):
        """
        Initialize quadrotor twin model.

        Args:
            mass: Quadrotor mass (kg)
            inertia: Inertia tensor (3x3 diagonal matrix). If None, uses default.
            gravity: Gravitational acceleration (m/s²)
            dt: Discretization timestep (seconds)
        """
        super().__init__(dt=dt)

        self.m = mass
        self.g = gravity

        # Default inertia from your spec (scaled by 100)
        if inertia is None:
            self.J = jnp.diag(jnp.array([1.8203e-5, 1.8186e-5, 3.4484e-5])) * 100.0
        else:
            self.J = inertia

        # Precompute inverse inertia for efficiency
        self.J_inv = jnp.linalg.inv(self.J)

    @property
    def state_dim(self) -> int:
        """State dimension n = 13."""
        return 13

    @property
    def input_dim(self) -> int:
        """Input dimension m = 4."""
        return 4

    def _dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Quadrotor dynamics: ẋ = f(x, u).

        Args:
            x: State [p, v, q, ω] (13,)
                - p: position (3,)
                - v: velocity (3,)
                - q: quaternion (4,)
                - ω: angular velocity (3,)
            u: Input [T, τ] (4,)
                - T: thrust (scalar)
                - τ: torques (3,)

        Returns:
            State derivative (13,)
        """
        # Extract state components
        vel = x[3:6]
        q = x[6:10]
        omega = x[10:13]

        # Extract input components
        T = u[0]  # thrust
        tau = u[1:4]  # torques

        # --- Translational dynamics ---
        # Gravity in inertial frame (NED: z points down, so gravity is positive)
        f_g_i = jnp.array([0.0, 0.0, self.g * self.m])

        # Thrust in body frame (pointing up in body frame = negative z)
        f_T_b = jnp.array([0.0, 0.0, -T])

        # Rotate thrust to inertial frame
        f_T_i = self._quat_rotate(q, f_T_b)

        # Net force and acceleration
        f_net_i = f_T_i + f_g_i
        v_dot_i = f_net_i / self.m

        # Position derivative
        pos_dot_i = vel

        # --- Rotational dynamics ---
        # Euler's equation: J ω̇ = τ - ω x (J ω)
        # => ω̇ = J⁻¹ (τ - ω x (J ω))
        omega_dot_b = self.J_inv @ (tau - jnp.cross(omega, self.J @ omega))

        # Quaternion kinematics: q̇ = 0.5 * Ω(ω) * q
        # where Ω(ω) is the skew-symmetric matrix for quaternion multiplication
        Omega = jnp.array(
            [
                [0, -omega[0], -omega[1], -omega[2]],
                [omega[0], 0, omega[2], -omega[1]],
                [omega[1], -omega[2], 0, omega[0]],
                [omega[2], omega[1], -omega[0], 0],
            ]
        )
        q_dot_i = 0.5 * Omega @ q

        # Combine all derivatives
        x_dot = jnp.hstack([pos_dot_i, v_dot_i, q_dot_i, omega_dot_b])

        return x_dot

    @staticmethod
    def _quat_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Rotate vector v from body frame to inertial frame using quaternion q.

        Quaternion convention: q = [qw, qx, qy, qz]
        Rotation: v_i = R(q) * v_b

        Args:
            q: Quaternion [qw, qx, qy, qz] (4,)
            v: Vector in body frame (3,)

        Returns:
            Vector in inertial frame (3,)
        """
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        # Rotation matrix from quaternion (body to inertial)
        R = jnp.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
            ]
        )

        return R @ v

    def normalize_state(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize state by ensuring quaternion has unit norm.

        Args:
            x: State [p, v, q, ω] (13,)

        Returns:
            Normalized state with ||q|| = 1 (13,)
        """
        q = x[6:10]
        q_norm = q / jnp.linalg.norm(q)
        x_norm = x.at[6:10].set(q_norm)
        return x_norm

    def state_distance(self, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
        """
        Compute distance between two states.

        Uses Euclidean distance for position/velocity/angular velocity
        and geodesic distance for quaternion.

        Args:
            x1: First state (13,)
            x2: Second state (13,)

        Returns:
            Weighted distance between states
        """
        # Position and velocity distance
        pos_diff = jnp.linalg.norm(x1[0:3] - x2[0:3])
        vel_diff = jnp.linalg.norm(x1[3:6] - x2[3:6])

        # Quaternion geodesic distance
        q1 = x1[6:10]
        q2 = x2[6:10]
        q_dot = jnp.abs(jnp.dot(q1, q2))
        q_dist = 2.0 * jnp.arccos(jnp.clip(q_dot, 0.0, 1.0))

        # Angular velocity distance
        omega_diff = jnp.linalg.norm(x1[10:13] - x2[10:13])

        # Weighted combination
        return jnp.sqrt(pos_diff**2 + vel_diff**2 + q_dist**2 + omega_diff**2)

    def quaternion_to_euler(self, q: jnp.ndarray) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).

        Args:
            q: Quaternion [qw, qx, qy, qz] (4,)

        Returns:
            (roll, pitch, yaw) in radians
        """
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        # Roll (x-axis rotation)
        roll = jnp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

        # Pitch (y-axis rotation)
        pitch = jnp.arcsin(jnp.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))

        # Yaw (z-axis rotation)
        yaw = jnp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        return roll, pitch, yaw

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> jnp.ndarray:
        """
        Convert Euler angles to quaternion.

        Args:
            roll: Roll angle (radians)
            pitch: Pitch angle (radians)
            yaw: Yaw angle (radians)

        Returns:
            Quaternion [qw, qx, qy, qz] (4,)
        """
        cy = jnp.cos(yaw * 0.5)
        sy = jnp.sin(yaw * 0.5)
        cp = jnp.cos(pitch * 0.5)
        sp = jnp.sin(pitch * 0.5)
        cr = jnp.cos(roll * 0.5)
        sr = jnp.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return jnp.array([qw, qx, qy, qz])

    def __repr__(self) -> str:
        return f"QuadrotorTwin(m={self.m:.4f}kg, state_dim={self.state_dim}, input_dim={self.input_dim}, dt={self.dt})"


class QuadrotorConstraints:
    """
    State and input constraints for quadrotor.

    Typical constraints:
        - Position: p ∈ [p_min, p_max]
        - Velocity: ||v|| ≤ v_max
        - Angular velocity: ||ω|| ≤ ω_max
        - Thrust: T ∈ [T_min, T_max]
        - Torques: ||τ|| ≤ τ_max
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

        Args:
            x_min, x_max: Position bounds in x (m)
            y_min, y_max: Position bounds in y (m)
            z_min, z_max: Position bounds in z (m) (NED: negative is up)
            v_max: Maximum velocity magnitude (m/s)
            omega_max: Maximum angular velocity magnitude (rad/s)
            T_min, T_max: Thrust bounds (N)
            tau_max: Maximum torque magnitude (N⋅m)
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

        Args:
            x: State [p, v, q, ω] (13,)

        Returns:
            True if state is within bounds
        """
        pos = x[0:3]
        vel = x[3:6]
        omega = x[10:13]

        pos_ok = jnp.all(pos >= self.p_min) and jnp.all(pos <= self.p_max)
        vel_ok = jnp.linalg.norm(vel) <= self.v_max
        omega_ok = jnp.linalg.norm(omega) <= self.omega_max

        return pos_ok and vel_ok and omega_ok

    def check_input(self, u: jnp.ndarray) -> bool:
        """
        Check if input satisfies constraints.

        Args:
            u: Input [T, τx, τy, τz] (4,)

        Returns:
            True if input is within bounds
        """
        return jnp.all(u >= self.u_min) and jnp.all(u <= self.u_max)

    def clip_input(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Clip input to satisfy constraints.

        Args:
            u: Input [T, τx, τy, τz] (4,)

        Returns:
            Clipped input (4,)
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


def create_quadrotor_example() -> Dict[str, Any]:
    """
    Create example quadrotor configuration matching your specifications.

    Based on:
        n = 13, m = 4
        m = 0.0293 kg
        J = diag([1.8203e-5, 1.8186e-5, 3.4484e-5]) * 100
        g = 9.81
        tf = 4, T = 51, dt = tf/T
        x_0 = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        x_des = [5, 5, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        num_obs = 2
        obs = [[2, 2, -1.5], [4, 4, -3.5]]
        obs_r = 0.5

    Returns:
        Configuration dictionary
    """
    return {
        "system": {
            "name": "quadrotor",
            "state_dim": 13,
            "input_dim": 4,
            "dt": 4.0 / 51,  # ≈ 0.078 seconds
            "mass": 0.0293,
            "inertia": [1.8203e-3, 1.8186e-3, 3.4484e-3],  # scaled by 100
            "gravity": 9.81,
        },
        "planning": {
            "tf": 4.0,
            "N": 51,
            "x0": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "xf": [5.0, 5.0, -4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "constraints": {
            "state_bounds": {
                "x_min": 0.0,
                "x_max": 8.0,
                "y_min": 0.0,
                "y_max": 8.0,
                "z_min": -5.0,
                "z_max": 0.5,
                "v_max": 5.0,
                "omega_max": 5.0,
            },
            "input_bounds": {"T_min": 0.0, "T_max": 1.0, "tau_max": 0.1},
        },
        "obstacles": [
            {"id": "obs_1", "type": "sphere", "center": [2.0, 2.0, -1.5], "radius": 0.5},
            {"id": "obs_2", "type": "sphere", "center": [4.0, 4.0, -3.5], "radius": 0.5},
        ],
        "plant_mismatch": {
            "mass_scale": 0.98,
            "inertia_scale": 1.02,
            "drag_coefficient": 0.01,
            "thrust_efficiency": 0.95,
        },
    }
