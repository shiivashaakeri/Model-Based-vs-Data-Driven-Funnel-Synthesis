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
    - τ: torques [τx, τy, τz] in body frame (N·m) (3,)

Dynamics:
    ṗ = v
    v̇ = (R(q) * [0, 0, -T]ᵀ + [0, 0, mg]ᵀ) / m
    q̇ = 0.5 * Ω(ω) * q
    ω̇ = J⁻¹ * (τ - ω x (J * ω))

where R(q) is the rotation matrix from body to inertial frame.

Convention: NED (North-East-Down) frame
    - z axis points downward
    - gravity is positive in z direction

Notes
-----
- Constraints are defined in ddfs.core.constraints
- Plant mismatch is defined in ddfs.models.plant
"""

from typing import Tuple

import jax.numpy as jnp

from ddfs.models.base import TwinModel


class QuadrotorTwin(TwinModel):
    """
    Full 3D quadrotor model with quaternion attitude (digital twin).

    This is the nominal model used for planning in Phase 1.
    The actual plant may have model mismatch (mass, inertia, drag, etc.).

    State dimension: n = 13
    Input dimension: m = 4

    Parameters
    ----------
    mass : float, optional
        Quadrotor mass (kg), by default 0.0293
    inertia : jnp.ndarray, optional
        Inertia tensor (3x3 diagonal matrix, kg·m²), by default scaled values
    gravity : float, optional
        Gravitational acceleration (m/s²), by default 9.81
    dt : float, optional
        Discretization timestep (seconds), by default 0.1

    Attributes
    ----------
    m : float
        Mass (kg)
    J : jnp.ndarray
        Inertia tensor (3x3)
    J_inv : jnp.ndarray
        Inverse inertia tensor (3x3)
    g : float
        Gravitational acceleration (m/s²)

    Examples
    --------
    >>> from ddfs.models.quadrotor import QuadrotorTwin
    >>> import jax.numpy as jnp
    >>>
    >>> twin = QuadrotorTwin(mass=0.0293, dt=0.078)
    >>> # Hover at origin: p=[0,0,0], v=[0,0,0], q=[1,0,0,0], ω=[0,0,0]
    >>> x = jnp.zeros(13)
    >>> x = x.at[6].set(1.0)  # qw = 1 (identity quaternion)
    >>> u = jnp.array([0.0293 * 9.81, 0.0, 0.0, 0.0])  # Hover thrust
    >>> x_next = twin.step(x, u)
    """

    def __init__(
        self,
        mass: float = 0.0293,
        inertia: jnp.ndarray = None,
        gravity: float = 9.81,
        dt: float = 0.1,
    ):
        """
        Initialize quadrotor twin model.

        Parameters
        ----------
        mass : float, optional
            Quadrotor mass (kg), by default 0.0293
        inertia : jnp.ndarray, optional
            Inertia tensor (3x3 diagonal), by default scaled from spec
        gravity : float, optional
            Gravitational acceleration (m/s²), by default 9.81
        dt : float, optional
            Discretization timestep (seconds), by default 0.1
        """
        super().__init__(dt=dt)

        self.m = mass
        self.g = gravity

        # Default inertia from spec (scaled by 100)
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

        Parameters
        ----------
        x : jnp.ndarray
            State [p, v, q, ω], shape (13,)
                - p: position (3,)
                - v: velocity (3,)
                - q: quaternion (4,)
                - ω: angular velocity (3,)
        u : jnp.ndarray
            Input [T, τ], shape (4,)
                - T: thrust (scalar)
                - τ: torques (3,)

        Returns
        -------
        x_dot : jnp.ndarray
            State derivative, shape (13,)
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

        Parameters
        ----------
        q : jnp.ndarray
            Quaternion [qw, qx, qy, qz], shape (4,)
        v : jnp.ndarray
            Vector in body frame, shape (3,)

        Returns
        -------
        v_i : jnp.ndarray
            Vector in inertial frame, shape (3,)
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

        Parameters
        ----------
        x : jnp.ndarray
            State [p, v, q, ω], shape (13,)

        Returns
        -------
        x_normalized : jnp.ndarray
            Normalized state with ||q|| = 1, shape (13,)

        Examples
        --------
        >>> twin = QuadrotorTwin()
        >>> x = jnp.zeros(13)
        >>> x = x.at[6:10].set(jnp.array([0.7, 0.7, 0.0, 0.0]))  # Non-unit quaternion
        >>> x_norm = twin.normalize_state(x)
        >>> print(jnp.linalg.norm(x_norm[6:10]))  # Should be 1.0
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

        Parameters
        ----------
        x1 : jnp.ndarray
            First state, shape (13,)
        x2 : jnp.ndarray
            Second state, shape (13,)

        Returns
        -------
        distance : float
            Weighted distance between states

        Examples
        --------
        >>> twin = QuadrotorTwin()
        >>> x1 = jnp.zeros(13)
        >>> x1 = x1.at[6].set(1.0)  # Identity quaternion
        >>> x2 = jnp.array([1.0, 1.0, -1.0] + [0]*10)
        >>> x2 = x2.at[6].set(1.0)
        >>> dist = twin.state_distance(x1, x2)
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
        return float(jnp.sqrt(pos_diff**2 + vel_diff**2 + q_dist**2 + omega_diff**2))

    def quaternion_to_euler(self, q: jnp.ndarray) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).

        Parameters
        ----------
        q : jnp.ndarray
            Quaternion [qw, qx, qy, qz], shape (4,)

        Returns
        -------
        roll : float
            Roll angle (radians)
        pitch : float
            Pitch angle (radians)
        yaw : float
            Yaw angle (radians)

        Examples
        --------
        >>> twin = QuadrotorTwin()
        >>> q = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity
        >>> roll, pitch, yaw = twin.quaternion_to_euler(q)
        >>> print(roll, pitch, yaw)  # Should be ~0, 0, 0
        """
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        # Roll (x-axis rotation)
        roll = jnp.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

        # Pitch (y-axis rotation)
        pitch = jnp.arcsin(jnp.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))

        # Yaw (z-axis rotation)
        yaw = jnp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        return float(roll), float(pitch), float(yaw)

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> jnp.ndarray:
        """
        Convert Euler angles to quaternion.

        Parameters
        ----------
        roll : float
            Roll angle (radians)
        pitch : float
            Pitch angle (radians)
        yaw : float
            Yaw angle (radians)

        Returns
        -------
        q : jnp.ndarray
            Quaternion [qw, qx, qy, qz], shape (4,)

        Examples
        --------
        >>> from ddfs.models.quadrotor import QuadrotorTwin
        >>> q = QuadrotorTwin.euler_to_quaternion(0.0, 0.0, 0.0)
        >>> print(q)  # Should be [1, 0, 0, 0]
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
        """String representation."""
        return f"QuadrotorTwin(m={self.m:.4f}kg, state_dim={self.state_dim}, input_dim={self.input_dim}, dt={self.dt})"


def create_quadrotor_example() -> dict:
    """
    Create example quadrotor configuration matching specifications.

    Based on typical quadrotor problem:
        n = 13, m = 4
        m = 0.0293 kg
        J = diag([1.8203e-5, 1.8186e-5, 3.4484e-5]) * 100
        g = 9.81
        tf = 4.0 seconds
        N = 51 timesteps
        dt = tf/N ≈ 0.078 seconds
        x_0 = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        x_des = [5, 5, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        workspace: x ∈ [0, 8], y ∈ [0, 8], z ∈ [-5, 0.5]
        obstacles: 2 spheres at [2, 2, -1.5] and [4, 4, -3.5], radius 0.5

    Returns
    -------
    config : dict
        Configuration dictionary with system, planning, constraints, and obstacles

    Examples
    --------
    >>> from ddfs.models.quadrotor import create_quadrotor_example
    >>> config = create_quadrotor_example()
    >>> print(config['system']['mass'])
    0.0293
    """
    return {
        "system": {
            "name": "quadrotor",
            "state_dim": 13,
            "input_dim": 4,
            "dt": 4.0 / 51,  # ≈ 0.078 seconds
            "mass": 0.0293,
            "inertia": [1.8203e-3, 1.8186e-3, 3.4484e-3],  # Scaled by 100
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
            "input_bounds": {
                "T_min": 0.0,
                "T_max": 1.0,
                "tau_x_min": -0.1,
                "tau_x_max": 0.1,
                "tau_y_min": -0.1,
                "tau_y_max": 0.1,
                "tau_z_min": -0.1,
                "tau_z_max": 0.1,
            },
        },
        "obstacles": [
            {
                "id": "obs_1",
                "type": "sphere",
                "center": [2.0, 2.0, -1.5],
                "radius": 0.5,
                "safety_margin": 0.2,
            },
            {
                "id": "obs_2",
                "type": "sphere",
                "center": [4.0, 4.0, -3.5],
                "radius": 0.5,
                "safety_margin": 0.2,
            },
        ],
        "plant_mismatch": {
            "mass_scale": 0.98,  # Plant is 2% lighter
            "inertia_scale": 1.02,  # Inertia is 2% higher
            "drag_coefficient": 0.01,  # Aerodynamic drag
            "thrust_efficiency": 0.95,  # 5% thrust loss
        },
    }
