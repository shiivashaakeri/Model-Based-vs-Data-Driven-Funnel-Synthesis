# ddfs/ddfs/models/plant.py

"""
Plant models with model mismatch.

This module implements plant models that represent the real system
with parameters that differ from the digital twin.

The plant is used for:
    - Phase 2: Data collection (generating trajectories)
    - Phase 6: Deployment simulation (closed-loop testing)

The mismatch between plant and twin is quantified in Phase 3 and
used for robust funnel synthesis in Phase 4.
"""

import jax.numpy as jnp
from jax import jit

from ddfs.models.base import PlantModel, TwinModel


class UnicyclePlant(PlantModel):
    """
    Unicycle plant with model mismatch.

    Mismatch types:
        - velocity_scale: Actual velocity differs from commanded
        - angular_scale: Actual angular velocity differs from commanded
        - slip_coefficient: Lateral slip/drift

    State: x = [x, y, θ] (position and heading)
    Input: u = [v, ω] (linear and angular velocity)

    Twin dynamics:
        ẋ = v cos(θ)
        ẏ = v sin(θ)
        θ̇ = ω

    Plant dynamics (with mismatch):
        ẋ = rho_v · v cos(θ) + slip_y
        ẏ = rho_v · v sin(θ) - slip_x
        θ̇ = rho_omega · ω

    where slip introduces lateral drift proportional to velocity.

    Parameters
    ----------
    twin : TwinModel
        Digital twin model
    velocity_scale : float, optional
        Velocity scaling factor rho_v (e.g., 0.95 = 5% slower), by default 1.0
    angular_scale : float, optional
        Angular velocity scaling rho_omega (e.g., 1.03 = 3% faster), by default 1.0
    slip_coefficient : float, optional
        Lateral slip coefficient (0 = no slip), by default 0.0

    Examples
    --------
    >>> from ddfs.models.unicycle import UnicycleTwin
    >>> from ddfs.models.plant import UnicyclePlant
    >>> import jax.numpy as jnp
    >>>
    >>> twin = UnicycleTwin(dt=0.1)
    >>> plant = UnicyclePlant(twin, velocity_scale=0.95, slip_coefficient=0.02)
    >>>
    >>> x = jnp.array([0.0, 0.0, 0.0])
    >>> u = jnp.array([1.0, 0.5])
    >>>
    >>> x_twin = twin.step(x, u)
    >>> x_plant = plant.step(x, u)
    >>> mismatch = plant.compute_mismatch(x, u)
    >>> print(f"Mismatch: {mismatch:.6f}")
    """

    def __init__(
        self,
        twin: TwinModel,
        velocity_scale: float = 1.0,
        angular_scale: float = 1.0,
        slip_coefficient: float = 0.0,
    ):
        """
        Initialize unicycle plant with mismatch parameters.

        Parameters
        ----------
        twin : TwinModel
            Digital twin model
        velocity_scale : float, optional
            Velocity scaling factor rho_v (e.g., 0.95 = 5% slower)
        angular_scale : float, optional
            Angular velocity scaling rho_omega (e.g., 1.03 = 3% faster)
        slip_coefficient : float, optional
            Lateral slip coefficient (0 = no slip)
        """
        mismatch_params = {
            "velocity_scale": velocity_scale,
            "angular_scale": angular_scale,
            "slip_coefficient": slip_coefficient,
        }
        super().__init__(twin, mismatch_params)

        self.velocity_scale = velocity_scale
        self.angular_scale = angular_scale
        self.slip_coefficient = slip_coefficient

        # JIT-compile mismatch application
        self._apply_mismatch_jit = jit(self._apply_mismatch)

    def _apply_mismatch(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Apply mismatch to unicycle dynamics.

        Parameters
        ----------
        x : jnp.ndarray
            State [x, y, θ], shape (3,)
        u : jnp.ndarray
            Input [v, ω], shape (2,)

        Returns
        -------
        x_dot : jnp.ndarray
            State derivative with mismatch, shape (3,)
        """
        # Extract state and input
        theta = x[2]
        v = u[0]
        omega = u[1]

        # Apply velocity and angular scaling
        v_actual = self.velocity_scale * v
        omega_actual = self.angular_scale * omega

        # Compute slip (lateral drift perpendicular to heading)
        # Slip is proportional to velocity and perpendicular to motion
        slip_x = self.slip_coefficient * v_actual * jnp.sin(theta)
        slip_y = self.slip_coefficient * v_actual * jnp.cos(theta)

        # Plant dynamics with mismatch
        x_dot = v_actual * jnp.cos(theta) + slip_y
        y_dot = v_actual * jnp.sin(theta) - slip_x
        theta_dot = omega_actual

        return jnp.array([x_dot, y_dot, theta_dot])

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnicyclePlant("
            f"velocity_scale={self.velocity_scale:.3f}, "
            f"angular_scale={self.angular_scale:.3f}, "
            f"slip_coefficient={self.slip_coefficient:.4f})"
        )


class QuadrotorPlant(PlantModel):
    """
    Quadrotor plant with model mismatch.

    Mismatch types:
        - mass_scale: Actual mass differs from nominal
        - inertia_scale: Inertia tensor scaling
        - drag_coefficient: Aerodynamic drag
        - thrust_efficiency: Motor efficiency factor

    State: x = [p, v, q, ω] (13D)
        - p: position [x, y, z] in inertial frame (3,)
        - v: velocity [vx, vy, vz] in inertial frame (3,)
        - q: quaternion [qw, qx, qy, qz] for orientation (4,)
        - ω: angular velocity [ωx, ωy, ωz] in body frame (3,)

    Input: u = [T, τ] (4D)
        - T: total thrust (scalar)
        - τ: torques [τx, τy, τz] in body frame (3,)

    Parameters
    ----------
    twin : TwinModel
        Digital twin model
    mass_scale : float, optional
        Mass scaling factor (e.g., 0.95 = 5% lighter), by default 1.0
    inertia_scale : float, optional
        Inertia scaling factor (e.g., 1.05 = 5% higher), by default 1.0
    drag_coefficient : float, optional
        Aerodynamic drag coefficient, by default 0.0
    thrust_efficiency : float, optional
        Thrust efficiency factor (e.g., 0.9 = 10% loss), by default 1.0

    Attributes
    ----------
    m_actual : float
        Actual mass with mismatch (kg)
    J_actual : jnp.ndarray
        Actual inertia tensor with mismatch (3x3)

    Examples
    --------
    >>> from ddfs.models.quadrotor import QuadrotorTwin
    >>> from ddfs.models.plant import QuadrotorPlant
    >>> import jax.numpy as jnp
    >>>
    >>> twin = QuadrotorTwin(mass=0.0293, dt=0.078)
    >>> plant = QuadrotorPlant(twin, mass_scale=0.98, drag_coefficient=0.01)
    >>>
    >>> x = jnp.zeros(13)
    >>> x = x.at[6].set(1.0)  # Identity quaternion
    >>> u = jnp.array([0.0293 * 9.81, 0.0, 0.0, 0.0])
    >>>
    >>> mismatch = plant.compute_mismatch(x, u)
    >>> print(f"Mismatch: {mismatch:.6f}")
    """

    def __init__(
        self,
        twin: TwinModel,
        mass_scale: float = 1.0,
        inertia_scale: float = 1.0,
        drag_coefficient: float = 0.0,
        thrust_efficiency: float = 1.0,
    ):
        """
        Initialize quadrotor plant with mismatch parameters.

        Parameters
        ----------
        twin : TwinModel
            Digital twin model
        mass_scale : float, optional
            Mass scaling factor (e.g., 0.95 = 5% lighter)
        inertia_scale : float, optional
            Inertia scaling factor (e.g., 1.05 = 5% higher)
        drag_coefficient : float, optional
            Aerodynamic drag coefficient
        thrust_efficiency : float, optional
            Thrust efficiency factor (e.g., 0.9 = 10% loss)
        """
        mismatch_params = {
            "mass_scale": mass_scale,
            "inertia_scale": inertia_scale,
            "drag_coefficient": drag_coefficient,
            "thrust_efficiency": thrust_efficiency,
        }
        super().__init__(twin, mismatch_params)

        self.mass_scale = mass_scale
        self.inertia_scale = inertia_scale
        self.drag_coefficient = drag_coefficient
        self.thrust_efficiency = thrust_efficiency

        # Get nominal parameters from twin
        self.m_nominal = twin.m
        self.J_nominal = twin.J

        # Compute actual parameters with mismatch
        self.m_actual = self.m_nominal * mass_scale
        self.J_actual = self.J_nominal * inertia_scale

        # JIT-compile mismatch application
        self._apply_mismatch_jit = jit(self._apply_mismatch)

    def _apply_mismatch(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Apply mismatch to quadrotor dynamics.

        Parameters
        ----------
        x : jnp.ndarray
            State [p, v, q, ω], shape (13,)
        u : jnp.ndarray
            Input [T, τ], shape (4,)

        Returns
        -------
        x_dot : jnp.ndarray
            State derivative with mismatch, shape (13,)
        """
        # Extract state
        vel = x[3:6]
        q = x[6:10]
        omega = x[10:13]

        # Extract input with thrust efficiency mismatch
        T = u[0] * self.thrust_efficiency
        tau = u[1:4]

        # Gravity in inertial frame (NED convention: z points down)
        g = 9.81
        f_g_i = jnp.array([0.0, 0.0, g * self.m_actual])

        # Thrust in body frame, then rotate to inertial frame
        f_T_b = jnp.array([0.0, 0.0, -T])
        f_T_i = self._quat_rotate(q, f_T_b)

        # Aerodynamic drag (in inertial frame, opposing velocity)
        f_drag_i = -self.drag_coefficient * vel * jnp.linalg.norm(vel)

        # Net force
        f_net_i = f_T_i + f_g_i + f_drag_i

        # Translational dynamics (Newton's second law)
        v_dot_i = f_net_i / self.m_actual
        pos_dot_i = vel

        # Rotational dynamics (Euler's equation)
        # ω̇ = J^(-1) (τ - ω x (J ω))
        J_inv = jnp.linalg.inv(self.J_actual)
        omega_dot_b = J_inv @ (tau - jnp.cross(omega, self.J_actual @ omega))

        # Quaternion dynamics
        # q̇ = 0.5 * Ω(ω) * q
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

        Uses the formula: v_i = q * v_b * q^*

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

        # Rotation matrix from quaternion
        R = jnp.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
            ]
        )

        return R @ v

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QuadrotorPlant("
            f"mass_scale={self.mass_scale:.3f}, "
            f"inertia_scale={self.inertia_scale:.3f}, "
            f"drag={self.drag_coefficient:.4f}, "
            f"thrust_eff={self.thrust_efficiency:.3f})"
        )


def create_plant_from_config(twin: TwinModel, config: dict) -> PlantModel:
    """
    Factory function to create plant from configuration.

    Parameters
    ----------
    twin : TwinModel
        Digital twin model
    config : dict
        Configuration dictionary with mismatch parameters

    Returns
    -------
    plant : PlantModel
        Plant model with configured mismatch

    Examples
    --------
    Example config for unicycle:
    >>> config = {
    ...     'velocity_scale': 0.95,
    ...     'angular_scale': 1.03,
    ...     'slip_coefficient': 0.02
    ... }

    Example config for quadrotor:
    >>> config = {
    ...     'mass_scale': 0.98,
    ...     'inertia_scale': 1.02,
    ...     'drag_coefficient': 0.01,
    ...     'thrust_efficiency': 0.95
    ... }

    Usage:
    >>> from ddfs.models.unicycle import UnicycleTwin
    >>> from ddfs.models.plant import create_plant_from_config
    >>>
    >>> twin = UnicycleTwin(dt=0.1)
    >>> config = {'velocity_scale': 0.95, 'angular_scale': 1.03, 'slip_coefficient': 0.02}
    >>> plant = create_plant_from_config(twin, config)
    """
    # Determine plant type from twin class name
    twin_type = twin.__class__.__name__

    if "Unicycle" in twin_type:
        return UnicyclePlant(
            twin=twin,
            velocity_scale=config.get("velocity_scale", 1.0),
            angular_scale=config.get("angular_scale", 1.0),
            slip_coefficient=config.get("slip_coefficient", 0.0),
        )
    elif "Quadrotor" in twin_type:
        return QuadrotorPlant(
            twin=twin,
            mass_scale=config.get("mass_scale", 1.0),
            inertia_scale=config.get("inertia_scale", 1.0),
            drag_coefficient=config.get("drag_coefficient", 0.0),
            thrust_efficiency=config.get("thrust_efficiency", 1.0),
        )
    else:
        raise ValueError(f"Unknown twin type: {twin_type}")
