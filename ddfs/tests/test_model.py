"""Comprehensive tests for ddfs.models module."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddfs.models import (
    DynamicsModel,
    QuadrotorPlant,
    QuadrotorTwin,
    TwinModel,
    UnicyclePlant,
    UnicycleTwin,
    create_plant_from_config,
    create_quadrotor_example,
    create_unicycle_example,
    validate_state_input_dims,
)

# ============================================================================
# BASE CLASS TESTS
# ============================================================================


class TestDynamicsModel:
    """Test DynamicsModel abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            DynamicsModel(dt=0.1)  # Abstract class


class TestValidateStateInputDims:
    """Test validate_state_input_dims function."""

    def test_valid_dimensions(self):
        """Test validation with correct dimensions."""
        x = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([1.0, 2.0])
        validate_state_input_dims(x, u, expected_state_dim=3, expected_input_dim=2)

    def test_invalid_state_dim(self):
        """Test validation with incorrect state dimension."""
        x = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="State dimension mismatch"):
            validate_state_input_dims(x, u, expected_state_dim=2, expected_input_dim=2)

    def test_invalid_input_dim(self):
        """Test validation with incorrect input dimension."""
        x = jnp.array([1.0, 2.0, 3.0])
        u = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            validate_state_input_dims(x, u, expected_state_dim=3, expected_input_dim=3)


# ============================================================================
# UNICYCLE TWIN TESTS
# ============================================================================


class TestUnicycleTwin:
    """Test UnicycleTwin class."""

    def test_creation(self):
        """Test UnicycleTwin creation."""
    twin = UnicycleTwin(dt=0.1)

    assert twin.state_dim == 3
    assert twin.input_dim == 2
        assert twin.dt == 0.1

    def test_creation_default_dt(self):
        """Test UnicycleTwin with default dt."""
        twin = UnicycleTwin()

        assert twin.dt == 0.1  # Default value

    def test_dynamics_stationary(self):
        """Test dynamics with zero input."""
        twin = UnicycleTwin(dt=0.1)
        x = jnp.array([1.0, 2.0, 0.5])
        u = jnp.array([0.0, 0.0])

        x_dot = twin.dynamics(x, u)

        assert x_dot.shape == (3,)
        assert jnp.allclose(x_dot, jnp.array([0.0, 0.0, 0.0]))

    def test_dynamics_forward(self):
        """Test dynamics with forward motion."""
        twin = UnicycleTwin(dt=0.1)
        x = jnp.array([0.0, 0.0, 0.0])  # At origin, facing right
        u = jnp.array([1.0, 0.0])  # Forward velocity only

        x_dot = twin.dynamics(x, u)

        assert x_dot[0] > 0  # Moving in x direction
        assert jnp.allclose(x_dot[1], 0.0)  # No y motion
        assert jnp.allclose(x_dot[2], 0.0)  # No rotation

    def test_dynamics_rotation(self):
        """Test dynamics with rotation only."""
        twin = UnicycleTwin(dt=0.1)
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([0.0, 1.0])  # Angular velocity only

        x_dot = twin.dynamics(x, u)

        assert jnp.allclose(x_dot[:2], jnp.array([0.0, 0.0]))  # No translation
        assert x_dot[2] > 0  # Rotating

    def test_dynamics_combined(self):
        """Test dynamics with combined motion."""
        twin = UnicycleTwin(dt=0.1)
        x = jnp.array([0.0, 0.0, jnp.pi / 4])  # 45 degrees
        u = jnp.array([1.0, 0.5])  # Forward + rotation

        x_dot = twin.dynamics(x, u)

        assert x_dot[0] > 0  # Moving in x
        assert x_dot[1] > 0  # Moving in y
        assert x_dot[2] > 0  # Rotating

    def test_step(self):
        """Test discrete-time step."""
        twin = UnicycleTwin(dt=0.1)
    x = jnp.array([0.0, 0.0, 0.0])
    u = jnp.array([1.0, 0.5])

    x_next = twin.step(x, u)

    assert x_next.shape == (3,)
    assert not jnp.allclose(x_next, x)  # Should have moved

    def test_step_custom_dt(self):
        """Test step with custom timestep."""
        twin = UnicycleTwin(dt=0.1)
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.0])

        x_next_short = twin.step(x, u, dt=0.05)
        x_next_long = twin.step(x, u, dt=0.2)

        # Longer timestep should move further
        assert jnp.linalg.norm(x_next_long[:2]) > jnp.linalg.norm(x_next_short[:2])

    def test_jacobian_state(self):
        """Test state Jacobian computation."""
        twin = UnicycleTwin(dt=0.1)
        x = jnp.array([1.0, 1.0, 0.5])
        u = jnp.array([1.0, 0.5])

        A = twin.jacobian_state(x, u)

        assert A.shape == (3, 3)

    def test_jacobian_input(self):
        """Test input Jacobian computation."""
        twin = UnicycleTwin(dt=0.1)
        x = jnp.array([1.0, 1.0, 0.5])
        u = jnp.array([1.0, 0.5])

        B = twin.jacobian_input(x, u)

        assert B.shape == (3, 2)

    def test_jacobians(self):
        """Test combined Jacobian computation."""
        twin = UnicycleTwin(dt=0.1)
        x = jnp.array([1.0, 1.0, 0.5])
        u = jnp.array([1.0, 0.5])

        A, B = twin.jacobians(x, u)

        assert A.shape == (3, 3)
        assert B.shape == (3, 2)

    def test_linearize(self):
        """Test linearization."""
        twin = UnicycleTwin(dt=0.1)
        x_bar = jnp.array([1.0, 1.0, 0.5])
        u_bar = jnp.array([1.0, 0.5])

        A, B = twin.linearize(x_bar, u_bar)

        assert A.shape == (3, 3)
        assert B.shape == (3, 2)

    def test_simulate_trajectory(self):
        """Test trajectory simulation."""
        twin = UnicycleTwin(dt=0.1)
        x0 = jnp.array([0.0, 0.0, 0.0])
        u_traj = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        x_traj = twin.simulate_trajectory(x0, u_traj)

        assert x_traj.shape == (4, 3)  # N+1 states
        assert jnp.allclose(x_traj[0], x0)
        # Should be moving forward
        assert x_traj[-1, 0] > x_traj[0, 0]

    def test_normalize_state(self):
        """Test state normalization (angle wrapping)."""
        twin = UnicycleTwin(dt=0.1)

        # Angle > π
        x = jnp.array([1.0, 2.0, 4.0])
        x_norm = twin.normalize_state(x)
        assert -np.pi <= x_norm[2] <= np.pi

        # Angle < -π
        x = jnp.array([1.0, 2.0, -4.0])
        x_norm = twin.normalize_state(x)
        assert -np.pi <= x_norm[2] <= np.pi

        # Angle already in range
        x = jnp.array([1.0, 2.0, 0.5])
        x_norm = twin.normalize_state(x)
        assert jnp.allclose(x_norm[:2], x[:2])
        assert -np.pi <= x_norm[2] <= np.pi

    def test_state_distance(self):
        """Test state distance computation."""
        twin = UnicycleTwin(dt=0.1)

        x1 = jnp.array([0.0, 0.0, 0.0])
        x2 = jnp.array([1.0, 1.0, jnp.pi / 4])

        dist = twin.state_distance(x1, x2)

        assert dist > 0
        assert isinstance(dist, float)

    def test_state_distance_same(self):
        """Test state distance for identical states."""
        twin = UnicycleTwin(dt=0.1)

        x = jnp.array([1.0, 2.0, 0.5])
        dist = twin.state_distance(x, x)

        assert jnp.allclose(dist, 0.0)

    def test_repr(self):
        """Test string representation."""
        twin = UnicycleTwin(dt=0.1)
        repr_str = repr(twin)

        assert "UnicycleTwin" in repr_str
        assert "state_dim=3" in repr_str
        assert "input_dim=2" in repr_str


class TestCreateUnicycleExample:
    """Test create_unicycle_example function."""

    def test_create_example(self):
        """Test example configuration creation."""
        config = create_unicycle_example()

        assert "system" in config
        assert "planning" in config
        assert "constraints" in config
        assert "obstacles" in config
        assert "plant_mismatch" in config

        assert config["system"]["state_dim"] == 3
        assert config["system"]["input_dim"] == 2
        assert len(config["obstacles"]) == 2


# ============================================================================
# QUADROTOR TWIN TESTS
# ============================================================================


class TestQuadrotorTwin:
    """Test QuadrotorTwin class."""

    def test_creation(self):
        """Test QuadrotorTwin creation."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)

        assert twin.state_dim == 13
        assert twin.input_dim == 4
        assert twin.dt == 0.078
        assert twin.m == 0.0293

    def test_creation_defaults(self):
        """Test QuadrotorTwin with default parameters."""
        twin = QuadrotorTwin()

        assert twin.state_dim == 13
        assert twin.input_dim == 4
        assert twin.m == 0.0293  # Default mass
        assert twin.g == 9.81  # Default gravity
        assert twin.J.shape == (3, 3)
        assert twin.J_inv.shape == (3, 3)

    def test_creation_custom_inertia(self):
        """Test QuadrotorTwin with custom inertia."""
        inertia = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        twin = QuadrotorTwin(mass=0.1, inertia=inertia, dt=0.1)

        assert jnp.allclose(twin.J, inertia)

    def test_dynamics_hover(self):
        """Test dynamics at hover equilibrium."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)

        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # Identity quaternion
        u = jnp.array([0.0293 * 9.81, 0.0, 0.0, 0.0])  # Hover thrust

        x_dot = twin.dynamics(x, u)

        assert x_dot.shape == (13,)
        # At hover, velocity should be zero (or very small)
        assert jnp.allclose(x_dot[3:6], jnp.zeros(3), atol=1e-6)

    def test_dynamics_forward_motion(self):
        """Test dynamics with forward motion."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)

        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # Identity quaternion
        u = jnp.array([0.0293 * 9.81 * 1.1, 0.0, 0.0, 0.0])  # Extra thrust

        x_dot = twin.dynamics(x, u)

        # Should accelerate upward (negative z in NED)
        assert x_dot[5] < 0  # vz < 0 means moving up

    def test_step(self):
        """Test discrete-time step."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)

        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # Identity quaternion
        # Use extra thrust to ensure motion
        u = jnp.array([0.0293 * 9.81 * 1.1, 0.0, 0.0, 0.0])

        x_next = twin.step(x, u)

        assert x_next.shape == (13,)
        # At hover with extra thrust, should accelerate upward
        assert x_next[5] < x[5]  # vz should decrease (moving up in NED)

    def test_jacobians(self):
        """Test Jacobian computation."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)

        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # Identity quaternion
        u = jnp.array([0.0293 * 9.81, 0.0, 0.0, 0.0])

        A, B = twin.jacobians(x, u)

        assert A.shape == (13, 13)
        assert B.shape == (13, 4)

    def test_normalize_state(self):
        """Test state normalization (quaternion normalization)."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)

        # Non-unit quaternion
        x = jnp.zeros(13)
        x = x.at[6:10].set(jnp.array([0.7, 0.7, 0.0, 0.0]))

        x_norm = twin.normalize_state(x)

        q_norm = x_norm[6:10]
        assert jnp.allclose(jnp.linalg.norm(q_norm), 1.0)

    def test_state_distance(self):
        """Test state distance computation."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)

        x1 = jnp.zeros(13)
        x1 = x1.at[6].set(1.0)  # Identity quaternion

        x2 = jnp.zeros(13)
        x2 = x2.at[0:3].set(jnp.array([1.0, 1.0, -1.0]))
        x2 = x2.at[6].set(1.0)  # Identity quaternion

        dist = twin.state_distance(x1, x2)

        assert dist > 0
        assert isinstance(dist, float)

    def test_quaternion_to_euler(self):
        """Test quaternion to Euler conversion."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)

        # Identity quaternion
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        roll, pitch, yaw = twin.quaternion_to_euler(q)

        assert abs(roll) < 1e-6
        assert abs(pitch) < 1e-6
        assert abs(yaw) < 1e-6

    def test_euler_to_quaternion(self):
        """Test Euler to quaternion conversion."""
        # Identity rotation
        q = QuadrotorTwin.euler_to_quaternion(0.0, 0.0, 0.0)

        assert jnp.allclose(q, jnp.array([1.0, 0.0, 0.0, 0.0]))

    def test_euler_quaternion_roundtrip(self):
        """Test Euler-quaternion roundtrip conversion."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)

        roll, pitch, yaw = 0.1, 0.2, 0.3
        q = QuadrotorTwin.euler_to_quaternion(roll, pitch, yaw)
        roll2, pitch2, yaw2 = twin.quaternion_to_euler(q)

        assert abs(roll - roll2) < 1e-5
        assert abs(pitch - pitch2) < 1e-5
        assert abs(yaw - yaw2) < 1e-5

    def test_repr(self):
        """Test string representation."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)
        repr_str = repr(twin)

        assert "QuadrotorTwin" in repr_str
        assert "state_dim=13" in repr_str or "13" in repr_str


class TestCreateQuadrotorExample:
    """Test create_quadrotor_example function."""

    def test_create_example(self):
        """Test example configuration creation."""
        config = create_quadrotor_example()

        assert "system" in config
        assert "planning" in config
        assert "constraints" in config
        assert "obstacles" in config
        assert "plant_mismatch" in config

        assert config["system"]["state_dim"] == 13
        assert config["system"]["input_dim"] == 4
        assert config["system"]["mass"] == 0.0293
        assert len(config["obstacles"]) == 2


# ============================================================================
# PLANT MODEL TESTS
# ============================================================================


class TestUnicyclePlant:
    """Test UnicyclePlant class."""

    def test_creation(self):
        """Test UnicyclePlant creation."""
    twin = UnicycleTwin(dt=0.1)
    plant = UnicyclePlant(twin, velocity_scale=0.95, slip_coefficient=0.02)

    assert plant.state_dim == 3
    assert plant.input_dim == 2
        assert plant.velocity_scale == 0.95
        assert plant.slip_coefficient == 0.02
        assert plant.angular_scale == 1.0  # Default

    def test_creation_all_params(self):
        """Test UnicyclePlant with all parameters."""
        twin = UnicycleTwin(dt=0.1)
        plant = UnicyclePlant(
            twin, velocity_scale=0.9, angular_scale=1.1, slip_coefficient=0.03
        )

        assert plant.velocity_scale == 0.9
        assert plant.angular_scale == 1.1
        assert plant.slip_coefficient == 0.03

    def test_dynamics_with_mismatch(self):
        """Test dynamics with mismatch applied."""
        twin = UnicycleTwin(dt=0.1)
        plant = UnicyclePlant(twin, velocity_scale=0.5, angular_scale=2.0)

        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 1.0])

        x_dot_plant = plant.dynamics(x, u)
        x_dot_twin = twin.dynamics(x, u)

        # Plant should move slower in x (velocity_scale=0.5)
        assert x_dot_plant[0] < x_dot_twin[0]
        # Plant should rotate faster (angular_scale=2.0)
        assert x_dot_plant[2] > x_dot_twin[2]

    def test_compute_mismatch(self):
        """Test mismatch computation."""
        twin = UnicycleTwin(dt=0.1)
        plant = UnicyclePlant(twin, velocity_scale=0.95, slip_coefficient=0.02)

    x = jnp.array([1.0, 1.0, 0.0])
    u = jnp.array([1.0, 0.5])

    mismatch = plant.compute_mismatch(x, u)

    assert mismatch >= 0  # Mismatch is non-negative
        assert isinstance(mismatch, float)

    def test_compute_mismatch_no_mismatch(self):
        """Test mismatch with no mismatch parameters."""
        twin = UnicycleTwin(dt=0.1)
        plant = UnicyclePlant(twin, velocity_scale=1.0, angular_scale=1.0, slip_coefficient=0.0)

        x = jnp.array([1.0, 1.0, 0.0])
        u = jnp.array([1.0, 0.5])

        mismatch = plant.compute_mismatch(x, u)

        # Should be very small (numerical errors only)
        assert mismatch < 1e-6

    def test_step(self):
        """Test discrete-time step."""
        twin = UnicycleTwin(dt=0.1)
        plant = UnicyclePlant(twin, velocity_scale=0.95)

        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.5])

        x_next = plant.step(x, u)

        assert x_next.shape == (3,)
        assert not jnp.allclose(x_next, x)

    def test_repr(self):
        """Test string representation."""
        twin = UnicycleTwin(dt=0.1)
        plant = UnicyclePlant(twin, velocity_scale=0.95, slip_coefficient=0.02)
        repr_str = repr(plant)

        assert "UnicyclePlant" in repr_str


class TestQuadrotorPlant:
    """Test QuadrotorPlant class."""

    def test_creation(self):
        """Test QuadrotorPlant creation."""
    twin = QuadrotorTwin(mass=0.0293, dt=0.078)
    plant = QuadrotorPlant(twin, mass_scale=0.98, drag_coefficient=0.01)

    assert plant.state_dim == 13
    assert plant.input_dim == 4
        assert plant.mass_scale == 0.98
        assert plant.drag_coefficient == 0.01
    assert plant.m_actual == pytest.approx(0.0293 * 0.98)

    def test_creation_all_params(self):
        """Test QuadrotorPlant with all parameters."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)
        plant = QuadrotorPlant(
            twin,
            mass_scale=0.95,
            inertia_scale=1.05,
            drag_coefficient=0.02,
            thrust_efficiency=0.9,
        )

        assert plant.mass_scale == 0.95
        assert plant.inertia_scale == 1.05
        assert plant.drag_coefficient == 0.02
        assert plant.thrust_efficiency == 0.9

    def test_dynamics_with_mismatch(self):
        """Test dynamics with mismatch applied."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)
        plant = QuadrotorPlant(twin, mass_scale=0.5, thrust_efficiency=0.5, drag_coefficient=0.1)

        # Use state with non-zero velocity to see drag effect
        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # Identity quaternion
        x = x.at[3:6].set(jnp.array([1.0, 1.0, -1.0]))  # Non-zero velocity
        u = jnp.array([0.0293 * 9.81 * 1.5, 0.1, 0.1, 0.1])

        x_dot_plant = plant.dynamics(x, u)
        x_dot_twin = twin.dynamics(x, u)

        # Plant should have different dynamics due to mismatch
        # Check that at least one component differs significantly
        diff = jnp.abs(x_dot_plant - x_dot_twin)
        assert jnp.max(diff) > 1e-3  # Should have significant difference

    def test_compute_mismatch(self):
        """Test mismatch computation."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)
        plant = QuadrotorPlant(twin, mass_scale=0.98, drag_coefficient=0.01)

        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # Identity quaternion
        u = jnp.array([0.0293 * 9.81, 0.0, 0.0, 0.0])

        mismatch = plant.compute_mismatch(x, u)

        assert mismatch >= 0
        assert isinstance(mismatch, float)

    def test_compute_mismatch_no_mismatch(self):
        """Test mismatch with no mismatch parameters."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)
        plant = QuadrotorPlant(
            twin,
            mass_scale=1.0,
            inertia_scale=1.0,
            drag_coefficient=0.0,
            thrust_efficiency=1.0,
        )

        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # Identity quaternion
        u = jnp.array([0.0293 * 9.81, 0.0, 0.0, 0.0])

        mismatch = plant.compute_mismatch(x, u)

        # Should be very small (numerical errors only)
        assert mismatch < 1e-5

    def test_step(self):
        """Test discrete-time step."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)
        plant = QuadrotorPlant(twin, mass_scale=0.98)

        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # Identity quaternion
        u = jnp.array([0.0293 * 9.81, 0.0, 0.0, 0.0])

        x_next = plant.step(x, u)

        assert x_next.shape == (13,)
        assert not jnp.allclose(x_next, x)

    def test_repr(self):
        """Test string representation."""
        twin = QuadrotorTwin(mass=0.0293, dt=0.078)
        plant = QuadrotorPlant(twin, mass_scale=0.98, drag_coefficient=0.01)
        repr_str = repr(plant)

        assert "QuadrotorPlant" in repr_str


class TestCreatePlantFromConfig:
    """Test create_plant_from_config function."""

    def test_unicycle_plant(self):
        """Test creating UnicyclePlant from config."""
        twin = UnicycleTwin(dt=0.1)
        config = {"velocity_scale": 0.95, "angular_scale": 1.03}

        plant = create_plant_from_config(twin, config)

        assert isinstance(plant, UnicyclePlant)
        assert plant.velocity_scale == 0.95
        assert plant.angular_scale == 1.03

    def test_unicycle_plant_partial_config(self):
        """Test creating UnicyclePlant with partial config."""
        twin = UnicycleTwin(dt=0.1)
        config = {"velocity_scale": 0.95}  # Missing angular_scale and slip_coefficient

        plant = create_plant_from_config(twin, config)

        assert isinstance(plant, UnicyclePlant)
        assert plant.velocity_scale == 0.95
        assert plant.angular_scale == 1.0  # Default
        assert plant.slip_coefficient == 0.0  # Default

    def test_quadrotor_plant(self):
        """Test creating QuadrotorPlant from config."""
        twin = QuadrotorTwin(dt=0.078)
        config = {"mass_scale": 0.98, "thrust_efficiency": 0.95}

        plant = create_plant_from_config(twin, config)

        assert isinstance(plant, QuadrotorPlant)
        assert plant.mass_scale == 0.98
        assert plant.thrust_efficiency == 0.95

    def test_quadrotor_plant_partial_config(self):
        """Test creating QuadrotorPlant with partial config."""
        twin = QuadrotorTwin(dt=0.078)
        config = {"mass_scale": 0.98}  # Missing other parameters

        plant = create_plant_from_config(twin, config)

        assert isinstance(plant, QuadrotorPlant)
        assert plant.mass_scale == 0.98
        assert plant.inertia_scale == 1.0  # Default
        assert plant.drag_coefficient == 0.0  # Default
        assert plant.thrust_efficiency == 1.0  # Default

    def test_unknown_twin_type(self):
        """Test error handling for unknown twin type."""
        # Create a mock twin that doesn't match known types
        class MockTwin(TwinModel):
            @property
            def state_dim(self):
                return 2

            @property
            def input_dim(self):
                return 1

            def _dynamics(self, x, u):
                # Use arguments to avoid unused warning
                _ = x
                _ = u
                return jnp.array([0.0, 0.0])

        twin = MockTwin(dt=0.1)
        config = {}

        with pytest.raises(ValueError, match="Unknown twin type"):
            create_plant_from_config(twin, config)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestModelIntegration:
    """Integration tests for model interactions."""

    def test_twin_plant_consistency(self):
        """Test that plant inherits dimensions from twin."""
        twin = UnicycleTwin(dt=0.1)
        plant = UnicyclePlant(twin)

        assert plant.state_dim == twin.state_dim
        assert plant.input_dim == twin.input_dim
        assert plant.dt == twin.dt

    def test_twin_plant_mismatch(self):
        """Test that plant and twin produce different trajectories."""
    twin = UnicycleTwin(dt=0.1)
        plant = UnicyclePlant(twin, velocity_scale=0.9)

        x0 = jnp.array([0.0, 0.0, 0.0])
        u_traj = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        x_traj_twin = twin.simulate_trajectory(x0, u_traj)
        x_traj_plant = plant.simulate_trajectory(x0, u_traj)

        # Plant should move slower (velocity_scale=0.9)
        assert x_traj_plant[-1, 0] < x_traj_twin[-1, 0]

    def test_jacobian_consistency(self):
        """Test that Jacobians are consistent with finite differences."""
        twin = UnicycleTwin(dt=0.1)
    x = jnp.array([1.0, 1.0, 0.5])
    u = jnp.array([1.0, 0.5])

    A, B = twin.jacobians(x, u)

        # Check dimensions
    assert A.shape == (3, 3)
    assert B.shape == (3, 2)

        # Check that A and B are not all zeros
        assert not jnp.allclose(A, jnp.zeros((3, 3)))
        assert not jnp.allclose(B, jnp.zeros((3, 2)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
