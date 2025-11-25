"""
Quadrotor Model Implementation for DDFS.

This module implements the 6-DoF quadrotor model with:
- Full rotational dynamics using quaternions
- Continuous-time dynamics
- Analytical Jacobians
- Plant and twin variants with configurable mismatch
- Factory for creating plant-twin pairs

State: x = [pos, vel, quat, omega]^T (13 states)
  - pos: position in NED frame [px, py, pz] [m]
  - vel: velocity in NED frame [vx, vy, vz] [m/s]
  - quat: attitude quaternion [qw, qx, qy, qz] (scalar-first)
  - omega: angular velocity in body frame [wx, wy, wz] [rad/s]

Input: u = [T, tau_x, tau_y, tau_z]^T (4 inputs)
  - T: total thrust [N]
  - tau: torques in body frame [N*m]

Note: NED frame means z-axis points DOWN (positive z is below origin)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ddfs.models.base_model import BaseModel, ModelFactory, ModelParameters, PlantTwinPair
from ddfs.utils.logging_utils import get_logger
from ddfs.utils.math_utils import (
    quat_normalize,
    quat_rotate,
    quat_to_euler,
    quat_to_rotation_matrix,
)

logger = get_logger(__name__)


# =============================================================================
# Quadrotor Parameters
# =============================================================================


@dataclass
class QuadrotorParameters(ModelParameters):
    """
    Parameters for quadrotor model.

    Physical Parameters:
    - mass: Total mass [kg]
    - gravity: Gravitational acceleration [m/s^2]
    - inertia: Diagonal inertia matrix elements [Ixx, Iyy, Izz] [kg*m^2]
    - arm_length: Distance from center to motor [m]

    Mismatch Parameters (for plant):
    - mass_scale: Multiplicative factor on mass (default: 1.0)
    - inertia_scale: Multiplicative factor on inertia (default: 1.0)
    - drag_coefficient: Linear aerodynamic drag coefficient (default: 0.0)
    - thrust_scale: Multiplicative factor on thrust (default: 1.0)
    - torque_scale: Multiplicative factor on torques (default: 1.0)
    - com_offset: Center of mass offset in body frame [m] (default: [0,0,0])
    """

    def __init__(
        self,
        mass: float = 0.0293,
        gravity: float = 9.81,
        inertia_xx: float = 1.8203e-3,
        inertia_yy: float = 1.8186e-3,
        inertia_zz: float = 3.4484e-3,
        arm_length: float = 0.046,
        mass_scale: float = 1.0,
        inertia_scale: float = 1.0,
        drag_coefficient: float = 0.0,
        thrust_scale: float = 1.0,
        torque_scale: float = 1.0,
        com_offset: Optional[np.ndarray] = None,
        name: str = "quadrotor_params",
    ):
        if com_offset is None:
            com_offset = np.zeros(3)

        params = {
            "mass": mass,
            "gravity": gravity,
            "inertia_xx": inertia_xx,
            "inertia_yy": inertia_yy,
            "inertia_zz": inertia_zz,
            "arm_length": arm_length,
            "mass_scale": mass_scale,
            "inertia_scale": inertia_scale,
            "drag_coefficient": drag_coefficient,
            "thrust_scale": thrust_scale,
            "torque_scale": torque_scale,
            "com_offset": com_offset,
        }
        super().__init__(params=params, name=name)

    @property
    def mass(self) -> float:
        return self.params["mass"]

    @property
    def gravity(self) -> float:
        return self.params["gravity"]

    @property
    def inertia_matrix(self) -> np.ndarray:
        """Get the diagonal inertia matrix."""
        return np.diag(
            [
                self.params["inertia_xx"],
                self.params["inertia_yy"],
                self.params["inertia_zz"],
            ]
        )

    @property
    def inertia_vector(self) -> np.ndarray:
        """Get inertia diagonal elements as vector."""
        return np.array(
            [
                self.params["inertia_xx"],
                self.params["inertia_yy"],
                self.params["inertia_zz"],
            ]
        )

    @property
    def arm_length(self) -> float:
        return self.params["arm_length"]

    @property
    def effective_mass(self) -> float:
        """Mass with mismatch scaling applied."""
        return self.params["mass"] * self.params["mass_scale"]

    @property
    def effective_inertia_matrix(self) -> np.ndarray:
        """Inertia matrix with mismatch scaling applied."""
        return self.inertia_matrix * self.params["inertia_scale"]

    @property
    def effective_inertia_vector(self) -> np.ndarray:
        """Inertia diagonal with mismatch scaling applied."""
        return self.inertia_vector * self.params["inertia_scale"]

    @classmethod
    def twin_default(cls) -> "QuadrotorParameters":
        """Default parameters for twin (nominal model)."""
        return cls(
            mass=0.0293,
            gravity=9.81,
            inertia_xx=1.8203e-3,  # Scaled by 100 as specified
            inertia_yy=1.8186e-3,
            inertia_zz=3.4484e-3,
            arm_length=0.046,
            mass_scale=1.0,
            inertia_scale=1.0,
            drag_coefficient=0.0,
            thrust_scale=1.0,
            torque_scale=1.0,
            com_offset=np.zeros(3),
            name="twin_params",
        )

    @classmethod
    def plant_default(
        cls,
        mass_scale: float = 1.08,
        inertia_scale: float = 1.05,
        drag_coefficient: float = 0.01,
        thrust_scale: float = 1.0,
        torque_scale: float = 1.0,
    ) -> "QuadrotorParameters":
        """Default parameters for plant (with mismatch)."""
        return cls(
            mass=0.0293,
            gravity=9.81,
            inertia_xx=1.8203e-3,
            inertia_yy=1.8186e-3,
            inertia_zz=3.4484e-3,
            arm_length=0.046,
            mass_scale=mass_scale,
            inertia_scale=inertia_scale,
            drag_coefficient=drag_coefficient,
            thrust_scale=thrust_scale,
            torque_scale=torque_scale,
            com_offset=np.zeros(3),
            name="plant_params",
        )


# =============================================================================
# Quaternion and Rotation Utilities (Local)
# =============================================================================


def _quat_to_omega_matrix(omega: np.ndarray) -> np.ndarray:
    """
    Create the Omega matrix for quaternion derivative.

    q_dot = 0.5 * Omega(omega) @ q

    Parameters
    ----------
    omega : np.ndarray
        Angular velocity [wx, wy, wz].

    Returns
    -------
    np.ndarray
        4x4 Omega matrix.
    """
    wx, wy, wz = omega
    return np.array(
        [
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0],
        ]
    )


def _skew(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.

    Parameters
    ----------
    v : np.ndarray
        3D vector.

    Returns
    -------
    np.ndarray
        3x3 skew-symmetric matrix.
    """
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


# =============================================================================
# Quadrotor Model Base Class
# =============================================================================


class QuadrotorModel(BaseModel):
    """
    6-DoF Quadrotor model with quaternion attitude representation.

    State: x = [pos, vel, quat, omega]^T (13 states)
    Input: u = [T, tau_x, tau_y, tau_z]^T (4 inputs)

    Frame conventions:
    - Inertial frame: NED (North-East-Down)
    - Body frame: FRD (Forward-Right-Down)
    - Quaternion: scalar-first [qw, qx, qy, qz]

    Parameters
    ----------
    params : QuadrotorParameters or dict, optional
        Model parameters.
    dt : float
        Discretization timestep [s].
    integration_method : str
        Integration method: 'euler', 'rk2', or 'rk4'.
    name : str
        Model name identifier.
    """

    # State indices
    PX, PY, PZ = 0, 1, 2  # Position
    VX, VY, VZ = 3, 4, 5  # Velocity
    QW, QX, QY, QZ = 6, 7, 8, 9  # Quaternion
    WX, WY, WZ = 10, 11, 12  # Angular velocity

    # Input indices
    THRUST = 0
    TAU_X, TAU_Y, TAU_Z = 1, 2, 3

    # Slice definitions for convenience
    POS_SLICE = slice(0, 3)
    VEL_SLICE = slice(3, 6)
    QUAT_SLICE = slice(6, 10)
    OMEGA_SLICE = slice(10, 13)

    def __init__(
        self,
        params: Optional[Union[QuadrotorParameters, Dict[str, Any]]] = None,
        dt: float = 0.02,
        integration_method: str = "rk4",
        name: str = "quadrotor",
    ):
        # Handle parameters
        if params is None:
            params = QuadrotorParameters.twin_default()
        elif isinstance(params, dict):
            params = QuadrotorParameters(**params)

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
        return 13

    @property
    def n_inputs(self) -> int:
        return 4

    @property
    def state_labels(self) -> list:
        return [
            "px",
            "py",
            "pz",
            "vx",
            "vy",
            "vz",
            "qw",
            "qx",
            "qy",
            "qz",
            "wx",
            "wy",
            "wz",
        ]

    @property
    def input_labels(self) -> list:
        return ["T", "tau_x", "tau_y", "tau_z"]

    @property
    def position_indices(self) -> list:
        """Indices of position states."""
        return [self.PX, self.PY, self.PZ]

    @property
    def has_analytical_jacobians(self) -> bool:
        return True

    @property
    def has_analytical_discrete_jacobians(self) -> bool:
        return False

    # =========================================================================
    # State Extraction Utilities
    # =========================================================================

    def get_position(self, x: np.ndarray) -> np.ndarray:
        """Extract position from state."""
        return x[self.POS_SLICE].copy()

    def get_velocity(self, x: np.ndarray) -> np.ndarray:
        """Extract velocity from state."""
        return x[self.VEL_SLICE].copy()

    def get_quaternion(self, x: np.ndarray) -> np.ndarray:
        """Extract quaternion from state."""
        return x[self.QUAT_SLICE].copy()

    def get_angular_velocity(self, x: np.ndarray) -> np.ndarray:
        """Extract angular velocity from state."""
        return x[self.OMEGA_SLICE].copy()

    def get_euler_angles(self, x: np.ndarray) -> Tuple[float, float, float]:
        """Extract Euler angles (roll, pitch, yaw) from state."""
        q = self.get_quaternion(x)
        return quat_to_euler(q)

    def get_rotation_matrix(self, x: np.ndarray) -> np.ndarray:
        """Get rotation matrix from body to inertial frame."""
        q = self.get_quaternion(x)
        return quat_to_rotation_matrix(q)

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

        Equations of motion:
            pos_dot = vel
            vel_dot = (1/m) * (R @ f_thrust + f_gravity + f_drag)
            quat_dot = 0.5 * Omega(omega) @ quat
            omega_dot = J^{-1} @ (tau - omega x (J @ omega))

        Parameters
        ----------
        x : np.ndarray
            State vector (13,).
        u : np.ndarray
            Input vector (4,).

        Returns
        -------
        np.ndarray
            State derivative (13,).
        """
        # Extract state components
        pos = x[self.POS_SLICE]  # noqa: F841
        vel = x[self.VEL_SLICE]
        quat = x[self.QUAT_SLICE]
        omega = x[self.OMEGA_SLICE]

        # Extract inputs
        T = u[self.THRUST]
        tau = u[self.TAU_X : self.TAU_Z + 1]

        # Get effective parameters (with mismatch if plant)
        mass = self._params.get("mass", 0.0293) * self._params.get("mass_scale", 1.0)
        gravity = self._params.get("gravity", 9.81)
        J = self._get_effective_inertia()
        J_inv = np.linalg.inv(J)
        drag_coeff = self._params.get("drag_coefficient", 0.0)
        thrust_scale = self._params.get("thrust_scale", 1.0)
        torque_scale = self._params.get("torque_scale", 1.0)

        # Apply thrust/torque scaling
        T_eff = T * thrust_scale
        tau_eff = tau * torque_scale

        # =====================================================================
        # Position derivative: pos_dot = vel
        # =====================================================================
        pos_dot = vel

        # =====================================================================
        # Velocity derivative: vel_dot = (1/m) * sum(forces)
        # =====================================================================

        # Gravity force in NED frame (points down, +z direction)
        f_gravity = np.array([0.0, 0.0, mass * gravity])

        # Thrust force in body frame (points up, -z direction in body)
        f_thrust_body = np.array([0.0, 0.0, -T_eff])

        # Rotate thrust to inertial frame
        f_thrust_inertial = quat_rotate(quat, f_thrust_body)

        # Aerodynamic drag (simple linear model in inertial frame)
        f_drag = -drag_coeff * vel

        # Total force and acceleration
        f_total = f_thrust_inertial + f_gravity + f_drag
        vel_dot = f_total / mass

        # =====================================================================
        # Quaternion derivative: quat_dot = 0.5 * Omega(omega) @ quat
        # =====================================================================
        Omega = _quat_to_omega_matrix(omega)
        quat_dot = 0.5 * Omega @ quat

        # =====================================================================
        # Angular velocity derivative: Euler's equation
        # omega_dot = J^{-1} @ (tau - omega x (J @ omega))
        # =====================================================================
        J_omega = J @ omega
        omega_cross_J_omega = np.cross(omega, J_omega)
        omega_dot = J_inv @ (tau_eff - omega_cross_J_omega)

        # Assemble state derivative
        x_dot = np.zeros(13)
        x_dot[self.POS_SLICE] = pos_dot
        x_dot[self.VEL_SLICE] = vel_dot
        x_dot[self.QUAT_SLICE] = quat_dot
        x_dot[self.OMEGA_SLICE] = omega_dot

        return x_dot

    def _get_effective_inertia(self) -> np.ndarray:
        """Get effective inertia matrix with scaling."""
        inertia_scale = self._params.get("inertia_scale", 1.0)
        return np.diag(
            [
                self._params.get("inertia_xx", 1.8203e-3) * inertia_scale,
                self._params.get("inertia_yy", 1.8186e-3) * inertia_scale,
                self._params.get("inertia_zz", 3.4484e-3) * inertia_scale,
            ]
        )

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
        # Extract state
        vel = x[self.VEL_SLICE]  # noqa: F841
        quat = x[self.QUAT_SLICE]
        omega = x[self.OMEGA_SLICE]
        qw, qx, qy, qz = quat

        # Extract input
        T = u[self.THRUST]

        # Get parameters
        mass = self._params.get("mass", 0.0293) * self._params.get("mass_scale", 1.0)
        J = self._get_effective_inertia()
        J_inv = np.linalg.inv(J)
        drag_coeff = self._params.get("drag_coefficient", 0.0)
        thrust_scale = self._params.get("thrust_scale", 1.0)
        torque_scale = self._params.get("torque_scale", 1.0)

        T_eff = T * thrust_scale
        Jxx, Jyy, Jzz = J[0, 0], J[1, 1], J[2, 2]
        wx, wy, wz = omega

        # Initialize Jacobians
        A = np.zeros((13, 13))
        B = np.zeros((13, 4))

        # =====================================================================
        # d(pos_dot)/d(state) - Position derivative depends on velocity
        # =====================================================================
        # pos_dot = vel
        A[self.POS_SLICE, self.VEL_SLICE] = np.eye(3)

        # =====================================================================
        # d(vel_dot)/d(state) - Velocity derivative
        # =====================================================================

        # d(vel_dot)/d(vel) - from drag
        A[self.VEL_SLICE, self.VEL_SLICE] = -drag_coeff / mass * np.eye(3)

        # d(vel_dot)/d(quat) - from thrust rotation
        # f_thrust_inertial = R(q) @ [0, 0, -T]
        # Need derivative of quaternion rotation

        # Rotation of [0, 0, -T] by quaternion q
        # Using the formula: R(q) @ v where R is rotation matrix from q
        # d(R@v)/dq can be computed analytically

        # For v = [0, 0, -T_eff]:
        # R @ v = 2*T_eff * [qw*qy + qx*qz, qy*qz - qw*qx, 0.5 - qx^2 - qy^2] (for -z thrust)
        # Actually, let's compute it properly

        # The rotated thrust is:
        # f_x = 2*T_eff*(qx*qz - qw*qy)
        # f_y = 2*T_eff*(qy*qz + qw*qx)
        # f_z = T_eff*(qw^2 - qx^2 - qy^2 + qz^2) but for -T in body z
        # Let's use: R @ [0,0,-T] = -T * R[:,2]

        # R[:,2] (third column of rotation matrix):
        # [2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), qw^2 - qx^2 - qy^2 + qz^2]

        # So f_thrust_inertial = -T_eff * R[:,2]

        # Derivatives w.r.t quaternion:
        df_thrust_dq = np.zeros((3, 4))
        df_thrust_dq[0, :] = -T_eff * 2 * np.array([qy, qz, qw, qx])  # d/dq of 2*(qx*qz + qw*qy)
        df_thrust_dq[1, :] = -T_eff * 2 * np.array([-qx, -qw, qz, qy])  # d/dq of 2*(qy*qz - qw*qx)
        df_thrust_dq[2, :] = -T_eff * 2 * np.array([qw, -qx, -qy, qz])  # d/dq of (qw^2 - qx^2 - qy^2 + qz^2)

        A[self.VEL_SLICE, self.QUAT_SLICE] = df_thrust_dq / mass

        # =====================================================================
        # d(quat_dot)/d(state) - Quaternion derivative
        # =====================================================================
        # quat_dot = 0.5 * Omega(omega) @ quat

        # d(quat_dot)/d(quat)
        Omega = _quat_to_omega_matrix(omega)
        A[self.QUAT_SLICE, self.QUAT_SLICE] = 0.5 * Omega

        # d(quat_dot)/d(omega)
        # d(Omega @ q)/d(omega)
        # Omega @ q = [[-wx*qx - wy*qy - wz*qz],
        #              [wx*qw + wz*qy - wy*qz],
        #              [wy*qw - wz*qx + wx*qz],
        #              [wz*qw + wy*qx - wx*qy]]

        dquat_dot_domega = 0.5 * np.array(
            [
                [-qx, -qy, -qz],
                [qw, -qz, qy],
                [qz, qw, -qx],
                [-qy, qx, qw],
            ]
        )
        A[self.QUAT_SLICE, self.OMEGA_SLICE] = dquat_dot_domega

        # =====================================================================
        # d(omega_dot)/d(state) - Angular velocity derivative
        # =====================================================================
        # omega_dot = J^{-1} @ (tau - omega x (J @ omega))

        # d(omega_dot)/d(omega)
        # Let's compute d(omega x (J @ omega))/d(omega)
        # omega x (J @ omega) = [wy*Jzz*wz - wz*Jyy*wy,
        #                        wz*Jxx*wx - wx*Jzz*wz,
        #                        wx*Jyy*wy - wy*Jxx*wx]

        # This is: [wy*wz*(Jzz - Jyy), wz*wx*(Jxx - Jzz), wx*wy*(Jyy - Jxx)]

        d_cross_domega = np.array(
            [
                [0, wz * (Jzz - Jyy), wy * (Jzz - Jyy)],
                [wz * (Jxx - Jzz), 0, wx * (Jxx - Jzz)],
                [wy * (Jyy - Jxx), wx * (Jyy - Jxx), 0],
            ]
        )

        A[self.OMEGA_SLICE, self.OMEGA_SLICE] = -J_inv @ d_cross_domega

        # =====================================================================
        # d(x_dot)/d(u) - Input Jacobian
        # =====================================================================

        # d(vel_dot)/d(T)
        # vel_dot depends on T through thrust
        # f_thrust_inertial = R @ [0, 0, -T*thrust_scale]
        R_col2 = np.array(
            [
                2 * (qx * qz + qw * qy),
                2 * (qy * qz - qw * qx),
                qw**2 - qx**2 - qy**2 + qz**2,
            ]
        )
        B[self.VEL_SLICE, self.THRUST] = -thrust_scale * R_col2 / mass

        # d(omega_dot)/d(tau)
        B[self.OMEGA_SLICE, self.TAU_X : self.TAU_Z + 1] = torque_scale * J_inv

        return A, B

    # =========================================================================
    # State Normalization
    # =========================================================================

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize state by ensuring unit quaternion.

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
        quat = x[self.QUAT_SLICE]
        x_normalized[self.QUAT_SLICE] = quat_normalize(quat)
        return x_normalized

    def discrete_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Override discrete dynamics to ensure quaternion normalization.
        """
        x_next = super().discrete_dynamics(x, u, dt)
        return self.normalize_state(x_next)

    # =========================================================================
    # Default Parameters
    # =========================================================================

    def get_default_parameters(self) -> ModelParameters:
        """Get default parameters for the model."""
        return QuadrotorParameters.twin_default()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def hover_thrust(self) -> float:
        """
        Compute thrust required for hover.

        Returns
        -------
        float
            Hover thrust [N].
        """
        mass = self._params.get("mass", 0.0293) * self._params.get("mass_scale", 1.0)
        gravity = self._params.get("gravity", 9.81)
        thrust_scale = self._params.get("thrust_scale", 1.0)
        return mass * gravity / thrust_scale

    def hover_input(self) -> np.ndarray:
        """
        Get input for hover (level flight).

        Returns
        -------
        np.ndarray
            Hover input [T, 0, 0, 0].
        """
        return np.array([self.hover_thrust(), 0.0, 0.0, 0.0])

    def hover_state(self, position: np.ndarray = None) -> np.ndarray:
        """
        Get state for hover at given position.

        Parameters
        ----------
        position : np.ndarray, optional
            Desired position [px, py, pz]. Default is origin.

        Returns
        -------
        np.ndarray
            Hover state (13,).
        """
        if position is None:
            position = np.zeros(3)

        x = np.zeros(13)
        x[self.POS_SLICE] = position
        # Velocity = 0
        x[self.QUAT_SLICE] = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        # Angular velocity = 0
        return x

    def state_from_components(
        self,
        position: np.ndarray,
        velocity: np.ndarray = None,
        quaternion: np.ndarray = None,
        angular_velocity: np.ndarray = None,
    ) -> np.ndarray:
        """
        Create state vector from components.

        Parameters
        ----------
        position : np.ndarray
            Position [px, py, pz].
        velocity : np.ndarray, optional
            Velocity [vx, vy, vz]. Default is zero.
        quaternion : np.ndarray, optional
            Quaternion [qw, qx, qy, qz]. Default is identity.
        angular_velocity : np.ndarray, optional
            Angular velocity [wx, wy, wz]. Default is zero.

        Returns
        -------
        np.ndarray
            State vector (13,).
        """
        x = np.zeros(13)
        x[self.POS_SLICE] = position

        if velocity is not None:
            x[self.VEL_SLICE] = velocity

        if quaternion is not None:
            x[self.QUAT_SLICE] = quat_normalize(quaternion)
        else:
            x[self.QUAT_SLICE] = np.array([1.0, 0.0, 0.0, 0.0])

        if angular_velocity is not None:
            x[self.OMEGA_SLICE] = angular_velocity

        return x


# =============================================================================
# Specialized Twin and Plant Models
# =============================================================================


class QuadrotorTwin(QuadrotorModel):
    """
    Quadrotor twin model (nominal, no mismatch).

    This represents the known digital twin used for planning.
    """

    def __init__(
        self,
        dt: float = 0.02,
        integration_method: str = "rk4",
    ):
        super().__init__(
            params=QuadrotorParameters.twin_default(),
            dt=dt,
            integration_method=integration_method,
            name="quadrotor_twin",
        )


class QuadrotorPlant(QuadrotorModel):
    """
    Quadrotor plant model (with mismatch).

    This represents the true physical system with unmodeled dynamics.

    Parameters
    ----------
    mass_scale : float
        Multiplicative factor on mass.
    inertia_scale : float
        Multiplicative factor on inertia.
    drag_coefficient : float
        Linear aerodynamic drag coefficient.
    thrust_scale : float
        Multiplicative factor on thrust effectiveness.
    torque_scale : float
        Multiplicative factor on torque effectiveness.
    dt : float
        Discretization timestep.
    integration_method : str
        Integration method.
    """

    def __init__(
        self,
        mass_scale: float = 1.08,
        inertia_scale: float = 1.05,
        drag_coefficient: float = 0.01,
        thrust_scale: float = 1.0,
        torque_scale: float = 1.0,
        dt: float = 0.02,
        integration_method: str = "rk4",
    ):
        params = QuadrotorParameters.plant_default(
            mass_scale=mass_scale,
            inertia_scale=inertia_scale,
            drag_coefficient=drag_coefficient,
            thrust_scale=thrust_scale,
            torque_scale=torque_scale,
        )
        super().__init__(
            params=params,
            dt=dt,
            integration_method=integration_method,
            name="quadrotor_plant",
        )

    @property
    def mismatch_params(self) -> dict:
        """Get mismatch parameters."""
        return {
            "mass_scale": self._params.get("mass_scale", 1.0),
            "inertia_scale": self._params.get("inertia_scale", 1.0),
            "drag_coefficient": self._params.get("drag_coefficient", 0.0),
            "thrust_scale": self._params.get("thrust_scale", 1.0),
            "torque_scale": self._params.get("torque_scale", 1.0),
        }


# =============================================================================
# Factory Class
# =============================================================================


class QuadrotorFactory(ModelFactory):
    """
    Factory for creating quadrotor plant-twin pairs.

    Example
    -------
    >>> pair = QuadrotorFactory.create_pair(
    ...     mass_scale=1.08,
    ...     inertia_scale=1.05,
    ...     drag_coefficient=0.01,
    ...     dt=0.02,
    ... )
    >>> twin = pair.twin
    >>> plant = pair.plant
    """

    @staticmethod
    def create_twin(
        dt: float = 0.02,
        integration_method: str = "rk4",
    ) -> QuadrotorTwin:
        """Create quadrotor twin model."""
        return QuadrotorTwin(dt=dt, integration_method=integration_method)

    @staticmethod
    def create_plant(
        mass_scale: float = 1.08,
        inertia_scale: float = 1.05,
        drag_coefficient: float = 0.01,
        thrust_scale: float = 1.0,
        torque_scale: float = 1.0,
        dt: float = 0.02,
        integration_method: str = "rk4",
    ) -> QuadrotorPlant:
        """Create quadrotor plant model."""
        return QuadrotorPlant(
            mass_scale=mass_scale,
            inertia_scale=inertia_scale,
            drag_coefficient=drag_coefficient,
            thrust_scale=thrust_scale,
            torque_scale=torque_scale,
            dt=dt,
            integration_method=integration_method,
        )

    @classmethod
    def create_pair(
        cls,
        mass_scale: float = 1.08,
        inertia_scale: float = 1.05,
        drag_coefficient: float = 0.01,
        thrust_scale: float = 1.0,
        torque_scale: float = 1.0,
        dt: float = 0.02,
        integration_method: str = "rk4",
    ) -> PlantTwinPair:
        """
        Create plant-twin pair with specified mismatch.

        Parameters
        ----------
        mass_scale : float
            Plant mass scale factor.
        inertia_scale : float
            Plant inertia scale factor.
        drag_coefficient : float
            Plant aerodynamic drag coefficient.
        thrust_scale : float
            Plant thrust effectiveness scale.
        torque_scale : float
            Plant torque effectiveness scale.
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
            mass_scale=mass_scale,
            inertia_scale=inertia_scale,
            drag_coefficient=drag_coefficient,
            thrust_scale=thrust_scale,
            torque_scale=torque_scale,
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
            mass_scale=mismatch.get("mass_scale", 1.08),
            inertia_scale=mismatch.get("inertia_scale", 1.05),
            drag_coefficient=mismatch.get("drag_coefficient", 0.01),
            thrust_scale=mismatch.get("thrust_scale", 1.0),
            torque_scale=mismatch.get("torque_scale", 1.0),
            dt=dt,
            integration_method="rk4",
        )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_quadrotor_mismatch_bound(
    plant: QuadrotorPlant,
    twin: QuadrotorTwin,
    n_samples: int = 2000,
    pos_range: Tuple[float, float] = (-5.0, 10.0),
    vel_range: Tuple[float, float] = (-3.0, 3.0),
    omega_range: Tuple[float, float] = (-5.0, 5.0),
    thrust_range: Tuple[float, float] = (0.0, 1.0),
    torque_range: Tuple[float, float] = (-0.01, 0.01),
) -> float:
    """
    Estimate mismatch bound gamma for quadrotor models.

    Samples the state-input space and computes maximum mismatch.

    Parameters
    ----------
    plant : QuadrotorPlant
        Plant model.
    twin : QuadrotorTwin
        Twin model.
    n_samples : int
        Number of samples.
    pos_range : tuple
        Position sampling range.
    vel_range : tuple
        Velocity sampling range.
    omega_range : tuple
        Angular velocity sampling range.
    thrust_range : tuple
        Thrust sampling range.
    torque_range : tuple
        Torque sampling range.

    Returns
    -------
    float
        Estimated mismatch bound gamma.
    """
    max_mismatch = 0.0

    for _ in range(n_samples):
        # Sample state
        pos = np.random.uniform(pos_range[0], pos_range[1], 3)
        vel = np.random.uniform(vel_range[0], vel_range[1], 3)

        # Random quaternion (uniform on unit sphere)
        quat = np.random.randn(4)
        quat = quat / np.linalg.norm(quat)
        if quat[0] < 0:  # Ensure positive scalar part for consistency
            quat = -quat

        omega = np.random.uniform(omega_range[0], omega_range[1], 3)

        x = np.concatenate([pos, vel, quat, omega])

        # Sample input
        T = np.random.uniform(thrust_range[0], thrust_range[1])
        tau = np.random.uniform(torque_range[0], torque_range[1], 3)
        u = np.concatenate([[T], tau])

        # Compute mismatch
        f_plant = plant.continuous_dynamics(x, u)
        f_twin = twin.continuous_dynamics(x, u)
        mismatch_norm = np.linalg.norm(f_plant - f_twin)

        max_mismatch = max(max_mismatch, mismatch_norm)

    return max_mismatch


def create_quadrotor_hover_trajectory(
    model: QuadrotorModel,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    N: int,
    method: str = "minimum_snap",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple reference trajectory for quadrotor.

    Parameters
    ----------
    model : QuadrotorModel
        Quadrotor model instance.
    start_pos : np.ndarray
        Start position [px, py, pz].
    end_pos : np.ndarray
        End position [px, py, pz].
    N : int
        Number of steps.
    method : str
        Trajectory method: 'linear', 'minimum_snap', or 'hover'.

    Returns
    -------
    x_ref : np.ndarray
        Reference state trajectory (N+1, 13).
    u_ref : np.ndarray
        Reference input trajectory (N, 4).
    """
    dt = model.dt

    x_ref = np.zeros((N + 1, 13))
    u_ref = np.zeros((N, 4))

    # Get hover thrust
    T_hover = model.hover_thrust()

    if method == "hover":
        # Just hover at start position
        for i in range(N + 1):
            x_ref[i] = model.hover_state(start_pos)
        u_ref[:, 0] = T_hover
        return x_ref, u_ref

    elif method == "linear":
        # Linear interpolation of position
        for i in range(N + 1):
            alpha = i / N
            pos = (1 - alpha) * start_pos + alpha * end_pos
            x_ref[i] = model.hover_state(pos)

            # Compute velocity from position change
            if i > 0:
                x_ref[i, model.VEL_SLICE] = (x_ref[i, model.POS_SLICE] - x_ref[i - 1, model.POS_SLICE]) / dt

        # Simple feedforward: hover thrust + small corrections
        u_ref[:, 0] = T_hover
        return x_ref, u_ref

    elif method == "minimum_snap":
        # Smooth trajectory using minimum-jerk/snap profile
        t_normalized = np.linspace(0, 1, N + 1)

        # Minimum-jerk: s(t) = 10*t^3 - 15*t^4 + 6*t^5
        s = 10 * t_normalized**3 - 15 * t_normalized**4 + 6 * t_normalized**5
        s_dot = (30 * t_normalized**2 - 60 * t_normalized**3 + 30 * t_normalized**4) / (N * dt)
        s_ddot = (60 * t_normalized - 180 * t_normalized**2 + 120 * t_normalized**3) / (N * dt) ** 2

        delta_pos = end_pos - start_pos

        for i in range(N + 1):
            pos = start_pos + s[i] * delta_pos
            vel = s_dot[i] * delta_pos if i < N else np.zeros(3)
            x_ref[i] = model.hover_state(pos)
            x_ref[i, model.VEL_SLICE] = vel

        # Feedforward thrust (accounting for acceleration)
        mass = model._params.get("mass", 0.0293) * model._params.get("mass_scale", 1.0)
        gravity = model._params.get("gravity", 9.81)

        for i in range(N):
            accel = s_ddot[i] * delta_pos
            # Required thrust: T = m * (g - az) for NED frame
            # In NED, +z is down, so hovering requires thrust = mg
            # Additional z acceleration requires additional thrust
            T_required = mass * (gravity - accel[2])
            u_ref[i, 0] = max(0.0, T_required)  # Thrust must be positive

        return x_ref, u_ref

    else:
        raise ValueError(f"Unknown trajectory method: {method}")
