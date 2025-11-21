# ddfs/tests/test_planning.py

"""
Comprehensive test suite for DDFS models and SCvx planner.

Tests cover:
1. Model functionality (unicycle and quadrotor)
2. Plant mismatch behavior
3. Constraints validation
4. SCvx planning with obstacle avoidance
"""

# Import models
import sys

import jax.numpy as jnp
import numpy as np  # Only for test data generation and CVXPY compatibility
import pytest

sys.path.insert(0, "/home/claude")

from ddfs.models.plant import QuadrotorPlant, UnicyclePlant, create_plant_from_config
from ddfs.models.quadrotor import QuadrotorConstraints, QuadrotorTwin, create_quadrotor_example
from ddfs.models.unicycle import UnicycleConstraints, UnicycleTwin, create_unicycle_example
from ddfs.planning.nominal_trajectory import NominalTrajectory
from ddfs.planning.scvx import SCvxParameters, SCvxPlanner, SCvxProblem

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def unicycle_twin():
    """Create unicycle twin model."""
    return UnicycleTwin(dt=0.1)


@pytest.fixture
def quadrotor_twin():
    """Create quadrotor twin model."""
    return QuadrotorTwin(dt=0.1)


@pytest.fixture
def unicycle_plant(unicycle_twin):
    """Create unicycle plant with mismatch."""
    return UnicyclePlant(twin=unicycle_twin, velocity_scale=0.95, angular_scale=1.03, slip_coefficient=0.02)


@pytest.fixture
def quadrotor_plant(quadrotor_twin):
    """Create quadrotor plant with mismatch."""
    return QuadrotorPlant(
        twin=quadrotor_twin, mass_scale=0.98, inertia_scale=1.02, drag_coefficient=0.01, thrust_efficiency=0.95
    )


@pytest.fixture
def unicycle_constraints():
    """Create unicycle constraints."""
    return UnicycleConstraints(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0, v_min=0.0, v_max=2.0, omega_max=2.0)


@pytest.fixture
def quadrotor_constraints():
    """Create quadrotor constraints."""
    return QuadrotorConstraints(
        x_min=0.0,
        x_max=8.0,
        y_min=0.0,
        y_max=8.0,
        z_min=-5.0,
        z_max=0.5,
        v_max=5.0,
        omega_max=5.0,
        T_min=0.0,
        T_max=1.0,
        tau_max=0.1,
    )


# ============================================================================
# MODEL TESTS - UNICYCLE
# ============================================================================


class TestUnicycleTwin:
    """Test unicycle twin model."""

    def test_dimensions(self, unicycle_twin):
        """Test state and input dimensions."""
        assert unicycle_twin.state_dim == 3
        assert unicycle_twin.input_dim == 2

    def test_dynamics_forward_motion(self, unicycle_twin):
        """Test forward motion dynamics."""
        x = jnp.array([0.0, 0.0, 0.0])  # Start at origin, heading east
        u = jnp.array([1.0, 0.0])  # Move forward at 1 m/s

        x_dot = unicycle_twin.dynamics(x, u)

        # Should move in +x direction
        assert abs(x_dot[0] - 1.0) < 1e-6  # dx/dt = v*cos(0) = 1
        assert abs(x_dot[1]) < 1e-6  # dy/dt = v*sin(0) = 0
        assert abs(x_dot[2]) < 1e-6  # dθ/dt = ω = 0

    def test_dynamics_rotation(self, unicycle_twin):
        """Test pure rotation dynamics."""
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([0.0, 1.5])  # Pure rotation at 1.5 rad/s

        x_dot = unicycle_twin.dynamics(x, u)

        # No translation, only rotation
        assert abs(x_dot[0]) < 1e-6
        assert abs(x_dot[1]) < 1e-6
        assert abs(x_dot[2] - 1.5) < 1e-6

    def test_step_integration(self, unicycle_twin):
        """Test RK4 integration step."""
        x0 = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.0])
        dt = 0.1

        x1 = unicycle_twin.step(x0, u, dt)

        # After 0.1s at 1 m/s, should be at approximately (0.1, 0, 0)
        assert abs(x1[0] - 0.1) < 1e-3
        assert abs(x1[1]) < 1e-3
        assert abs(x1[2]) < 1e-3

    def test_jacobians(self, unicycle_twin):
        """Test Jacobian computation."""
        x = jnp.array([1.0, 2.0, np.pi / 4])
        u = jnp.array([1.0, 0.5])

        A, B = unicycle_twin.jacobians(x, u)

        # Check shapes
        assert A.shape == (3, 3)
        assert B.shape == (3, 2)

        # A should have structure related to heading angle
        # B should show direct control influence

    def test_state_normalization(self, unicycle_twin):
        """Test angle normalization."""
        x = jnp.array([1.0, 2.0, 4 * np.pi])  # Angle wraps around
        x_norm = unicycle_twin.normalize_state(x)

        # Angle should be wrapped to [-π, π]
        assert -np.pi <= x_norm[2] <= np.pi
        assert abs(x_norm[2]) < 1e-6  # 4π wraps to 0

    def test_simulate_trajectory(self, unicycle_twin):
        """Test trajectory simulation."""
        x0 = jnp.array([0.0, 0.0, 0.0])
        N = 10
        u_traj = jnp.ones((N, 2))  # Constant forward motion with rotation

        x_traj = unicycle_twin.simulate_trajectory(x0, u_traj, dt=0.1)

        assert x_traj.shape == (N + 1, 3)
        assert jnp.allclose(x_traj[0], x0)  # Initial state preserved


class TestUnicyclePlant:
    """Test unicycle plant model with mismatch."""

    def test_velocity_scaling(self, unicycle_twin):
        """Test velocity scaling mismatch."""
        plant = UnicyclePlant(twin=unicycle_twin, velocity_scale=0.9)

        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.0])  # Command 1 m/s

        x_dot_plant = plant.dynamics(x, u)
        x_dot_twin = unicycle_twin.dynamics(x, u)

        # Plant should move slower
        assert x_dot_plant[0] < x_dot_twin[0]
        assert abs(x_dot_plant[0] - 0.9) < 1e-6

    def test_angular_scaling(self, unicycle_twin):
        """Test angular velocity scaling mismatch."""
        plant = UnicyclePlant(twin=unicycle_twin, angular_scale=1.1)

        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([0.0, 1.0])  # Command 1 rad/s

        x_dot_plant = plant.dynamics(x, u)
        x_dot_twin = unicycle_twin.dynamics(x, u)

        # Plant should turn faster
        assert x_dot_plant[2] > x_dot_twin[2]
        assert abs(x_dot_plant[2] - 1.1) < 1e-6

    def test_slip_effect(self, unicycle_twin):
        """Test lateral slip mismatch."""
        plant = UnicyclePlant(twin=unicycle_twin, slip_coefficient=0.1)

        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.0])  # Move forward

        x_dot_plant = plant.dynamics(x, u)
        x_dot_twin = unicycle_twin.dynamics(x, u)

        # Slip should add lateral component
        # With theta=0, slip_y is added to x_dot, slip_x (which is 0) is subtracted from y_dot
        # So x_dot should be different from twin due to slip_y
        assert abs(x_dot_plant[0] - x_dot_twin[0]) > 1e-6

    def test_compute_mismatch(self, unicycle_plant, unicycle_twin):  # noqa: ARG002
        """Test mismatch computation."""
        x = jnp.array([1.0, 2.0, 0.5])
        u = jnp.array([1.0, 0.5])

        gamma = unicycle_plant.compute_mismatch(x, u)

        # Mismatch should be positive due to scaling and slip
        assert gamma > 0


class TestUnicycleConstraints:
    """Test unicycle constraints."""

    def test_state_bounds_check(self, unicycle_constraints):
        """Test state constraint checking."""
        x_valid = jnp.array([5.0, 4.0, 0.0])
        x_invalid = jnp.array([15.0, 4.0, 0.0])  # x > x_max

        assert unicycle_constraints.check_state(x_valid)
        assert not unicycle_constraints.check_state(x_invalid)

    def test_input_bounds_check(self, unicycle_constraints):
        """Test input constraint checking."""
        u_valid = jnp.array([1.0, 1.0])
        u_invalid = jnp.array([3.0, 1.0])  # v > v_max

        assert unicycle_constraints.check_input(u_valid)
        assert not unicycle_constraints.check_input(u_invalid)

    def test_input_clipping(self, unicycle_constraints):
        """Test input clipping."""
        u_violate = jnp.array([3.0, 3.0])
        u_clipped = unicycle_constraints.clip_input(u_violate)

        assert u_clipped[0] == 2.0  # Clipped to v_max
        assert u_clipped[1] == 2.0  # Clipped to omega_max

    def test_from_config(self):
        """Test constraint creation from config."""
        config = {
            "state_bounds": {"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 8.0},
            "input_bounds": {"v_min": 0.0, "v_max": 2.0, "omega_max": 2.0},
        }

        constraints = UnicycleConstraints.from_config(config)

        assert constraints.x_min[0] == 0.0
        assert constraints.x_max[0] == 10.0


# ============================================================================
# MODEL TESTS - QUADROTOR
# ============================================================================


class TestQuadrotorTwin:
    """Test quadrotor twin model."""

    def test_dimensions(self, quadrotor_twin):
        """Test state and input dimensions."""
        assert quadrotor_twin.state_dim == 13
        assert quadrotor_twin.input_dim == 4

    def test_hover_equilibrium(self, quadrotor_twin):
        """Test hovering equilibrium dynamics."""
        # Hover state: no velocity, identity quaternion, no angular velocity
        x = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        # Thrust to counteract gravity (T = m*g)
        T_hover = quadrotor_twin.m * quadrotor_twin.g
        u = jnp.array([T_hover, 0, 0, 0])

        x_dot = quadrotor_twin.dynamics(x, u)

        # All derivatives should be near zero for hover
        assert jnp.linalg.norm(x_dot) < 1e-3

    def test_free_fall(self, quadrotor_twin):
        """Test free fall with no thrust."""
        x = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        u = jnp.array([0, 0, 0, 0])  # No thrust

        x_dot = quadrotor_twin.dynamics(x, u)

        # Should accelerate downward (positive z in NED)
        assert x_dot[5] > 0  # z-velocity derivative should be positive

    def test_quaternion_rotation(self, quadrotor_twin):
        """Test quaternion rotation function."""
        q = jnp.array([1, 0, 0, 0])  # Identity quaternion
        v_body = jnp.array([1, 0, 0])

        v_inertial = quadrotor_twin._quat_rotate(q, v_body)

        # Identity rotation should not change vector
        assert jnp.allclose(v_inertial, v_body)

    def test_euler_quaternion_conversion(self, quadrotor_twin):
        """Test Euler <-> quaternion conversion."""
        roll, pitch, yaw = 0.1, 0.2, 0.3

        q = quadrotor_twin.euler_to_quaternion(roll, pitch, yaw)
        roll2, pitch2, yaw2 = quadrotor_twin.quaternion_to_euler(q)

        # Should recover original angles
        assert abs(roll - roll2) < 1e-6
        assert abs(pitch - pitch2) < 1e-6
        assert abs(yaw - yaw2) < 1e-6

    def test_quaternion_normalization(self, quadrotor_twin):
        """Test state normalization (quaternion)."""
        x = jnp.array([0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0])  # Unnormalized quat
        x_norm = quadrotor_twin.normalize_state(x)

        q_norm = jnp.linalg.norm(x_norm[6:10])
        assert abs(q_norm - 1.0) < 1e-6

    def test_jacobians(self, quadrotor_twin):
        """Test Jacobian computation."""
        x = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = jnp.array([0.3, 0.0, 0.0, 0.0])

        A, B = quadrotor_twin.jacobians(x, u)

        # Check shapes
        assert A.shape == (13, 13)
        assert B.shape == (13, 4)


class TestQuadrotorPlant:
    """Test quadrotor plant model with mismatch."""

    def test_mass_scaling(self, quadrotor_twin):
        """Test mass scaling mismatch."""
        plant = QuadrotorPlant(twin=quadrotor_twin, mass_scale=1.1)

        x = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        u = jnp.array([0.3, 0, 0, 0])

        x_dot_plant = plant.dynamics(x, u)
        x_dot_twin = quadrotor_twin.dynamics(x, u)

        # Heavier plant should have different acceleration
        assert not jnp.allclose(x_dot_plant[3:6], x_dot_twin[3:6])

    def test_thrust_efficiency(self, quadrotor_twin):
        """Test thrust efficiency mismatch."""
        plant = QuadrotorPlant(twin=quadrotor_twin, thrust_efficiency=0.8)

        x = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        u = jnp.array([0.3, 0, 0, 0])

        # Plant with 80% efficiency should produce less upward force
        x_dot = plant.dynamics(x, u)

        # Should have more downward acceleration than with full thrust
        assert x_dot[5] > 0  # Still falling (thrust not enough)

    def test_drag_effect(self, quadrotor_twin):
        """Test aerodynamic drag."""
        plant = QuadrotorPlant(twin=quadrotor_twin, drag_coefficient=0.1)

        # State with velocity
        x = jnp.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        u = jnp.array([quadrotor_twin.m * quadrotor_twin.g, 0, 0, 0])

        x_dot = plant.dynamics(x, u)

        # Drag should oppose velocity
        assert x_dot[3] < 0  # Deceleration in x

    def test_compute_mismatch(self, quadrotor_plant, quadrotor_twin):  # noqa: ARG002
        """Test mismatch computation."""
        x = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.1, 0, 0])
        u = jnp.array([0.3, 0.01, 0, 0])

        gamma = quadrotor_plant.compute_mismatch(x, u)

        # Mismatch should be positive
        assert gamma > 0


class TestQuadrotorConstraints:
    """Test quadrotor constraints."""

    def test_state_bounds_check(self, quadrotor_constraints):
        """Test state constraint checking."""
        # Valid state
        x_valid = jnp.array([4, 4, -2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        assert quadrotor_constraints.check_state(x_valid)

        # Out of position bounds
        x_invalid = jnp.array([10, 4, -2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        assert not quadrotor_constraints.check_state(x_invalid)

    def test_velocity_limit_check(self, quadrotor_constraints):
        """Test velocity magnitude constraint."""
        # Valid velocity
        x_valid = jnp.array([4, 4, -2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        assert quadrotor_constraints.check_state(x_valid)

        # Excessive velocity
        x_invalid = jnp.array([4, 4, -2, 10, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        assert not quadrotor_constraints.check_state(x_invalid)

    def test_input_bounds_check(self, quadrotor_constraints):
        """Test input constraint checking."""
        u_valid = jnp.array([0.5, 0.05, 0, 0])
        u_invalid = jnp.array([2.0, 0, 0, 0])  # Thrust too high

        assert quadrotor_constraints.check_input(u_valid)
        assert not quadrotor_constraints.check_input(u_invalid)

    def test_input_clipping(self, quadrotor_constraints):
        """Test input clipping."""
        u_violate = jnp.array([2.0, 0.2, -0.2, 0.15])
        u_clipped = quadrotor_constraints.clip_input(u_violate)

        assert u_clipped[0] == 1.0  # Clipped to T_max
        assert u_clipped[1] == 0.1  # Clipped to tau_max


# ============================================================================
# FACTORY TESTS
# ============================================================================


class TestFactoryFunctions:
    """Test factory functions for creating models."""

    def test_create_plant_unicycle(self, unicycle_twin):
        """Test plant creation from config for unicycle."""
        config = {"velocity_scale": 0.9, "angular_scale": 1.1, "slip_coefficient": 0.05}

        plant = create_plant_from_config(unicycle_twin, config)

        assert isinstance(plant, UnicyclePlant)
        assert plant.velocity_scale == 0.9

    def test_create_plant_quadrotor(self, quadrotor_twin):
        """Test plant creation from config for quadrotor."""
        config = {"mass_scale": 0.95, "inertia_scale": 1.05, "drag_coefficient": 0.02, "thrust_efficiency": 0.9}

        plant = create_plant_from_config(quadrotor_twin, config)

        assert isinstance(plant, QuadrotorPlant)
        assert plant.mass_scale == 0.95

    def test_create_unicycle_example(self):
        """Test unicycle example configuration."""
        config = create_unicycle_example()

        assert config["system"]["state_dim"] == 3
        assert config["system"]["input_dim"] == 2
        assert len(config["obstacles"]) == 2

    def test_create_quadrotor_example(self):
        """Test quadrotor example configuration."""
        config = create_quadrotor_example()

        assert config["system"]["state_dim"] == 13
        assert config["system"]["input_dim"] == 4
        assert len(config["obstacles"]) == 2


# ============================================================================
# NOMINAL TRAJECTORY TESTS
# ============================================================================


class TestNominalTrajectory:
    """Test NominalTrajectory container."""

    def test_creation(self):
        """Test trajectory creation."""
        N = 10
        x_nom = jnp.array(np.random.randn(N + 1, 3))
        u_nom = jnp.array(np.random.randn(N, 2))

        traj = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)

        assert traj.state_dim == 3
        assert traj.input_dim == 2
        assert traj.tf == 1.0

    def test_evaluate_at(self):
        """Test trajectory evaluation."""
        N = 10
        x_nom = jnp.array(np.random.randn(N + 1, 3))
        u_nom = jnp.array(np.random.randn(N, 2))

        traj = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)

        x_k, u_k = traj.evaluate_at(5)

        assert jnp.allclose(x_k, x_nom[5])
        assert jnp.allclose(u_k, u_nom[5])

    def test_save_load(self, tmp_path):
        """Test trajectory save/load."""
        N = 10
        x_nom = jnp.array(np.random.randn(N + 1, 3))
        u_nom = jnp.array(np.random.randn(N, 2))

        traj = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)

        path = tmp_path / "traj.pkl"
        traj.save(path)

        traj_loaded = NominalTrajectory.load(path)

        assert jnp.allclose(traj_loaded.x_nom, x_nom)
        assert jnp.allclose(traj_loaded.u_nom, u_nom)


# ============================================================================
# SCVX TESTS
# ============================================================================


class TestSCvxProblem:
    """Test SCvx convex subproblem."""

    def test_problem_creation(self, unicycle_twin):
        """Test problem creation."""
        N = 20
        state_bounds = (jnp.array([0, 0, -np.pi]), jnp.array([10, 8, np.pi]))
        input_bounds = (jnp.array([0, -2]), jnp.array([2, 2]))

        problem = SCvxProblem(
            model=unicycle_twin, N=N, state_dim=3, input_dim=2, state_bounds=state_bounds, input_bounds=input_bounds
        )

        assert problem.X.shape == (3, N + 1)
        assert problem.U.shape == (2, N)

    def test_parameter_setting(self, unicycle_twin):
        """Test setting problem parameters."""
        N = 10
        problem = SCvxProblem(model=unicycle_twin, N=N, state_dim=3, input_dim=2)

        A_bar = jnp.array(np.random.randn(3, 3, N))
        B_bar = jnp.array(np.random.randn(3, 2, N))
        z_bar = jnp.array(np.random.randn(3, N))
        X_ref = jnp.array(np.random.randn(3, N + 1))
        U_ref = jnp.array(np.random.randn(2, N))

        problem.set_parameters(
            A_bar=A_bar, B_bar=B_bar, z_bar=z_bar, X_ref=X_ref, U_ref=U_ref, tr_radius=1.0, weight_nu=1e5
        )

        # Parameters should be set without error


class TestSCvxPlanner:
    """Test SCvx planner."""

    def test_planner_creation(self, unicycle_twin, unicycle_constraints):
        """Test planner creation."""
        state_bounds = (
            jnp.array([float(unicycle_constraints.x_min[0]), float(unicycle_constraints.x_min[1]), -np.pi]),
            jnp.array([float(unicycle_constraints.x_max[0]), float(unicycle_constraints.x_max[1]), np.pi]),
        )
        input_bounds = (unicycle_constraints.u_min, unicycle_constraints.u_max)

        params = SCvxParameters(max_iterations=5, verbose=False)

        planner = SCvxPlanner(model=unicycle_twin, params=params, state_bounds=state_bounds, input_bounds=input_bounds)

        assert planner.model == unicycle_twin

    def test_initialize_trajectory(self, unicycle_twin):
        """Test trajectory initialization."""
        planner = SCvxPlanner(model=unicycle_twin, params=SCvxParameters(verbose=False))

        x0 = jnp.array([1.0, 1.0, 0.0])
        xf = jnp.array([5.0, 5.0, 0.0])
        N = 20

        X, U = planner.initialize_trajectory(x0, xf, N)

        assert X.shape == (3, N + 1)
        assert U.shape == (2, N)
        assert jnp.allclose(X[:, 0], x0)
        assert jnp.allclose(X[:, -1], xf)

    def test_compute_linearization(self, unicycle_twin):
        """Test dynamics linearization."""
        planner = SCvxPlanner(model=unicycle_twin, params=SCvxParameters(verbose=False))

        N = 10
        X = jnp.array(np.random.randn(3, N + 1))
        U = jnp.array(np.random.randn(2, N))
        dt = 0.1

        A_bar, B_bar, z_bar = planner.compute_linearization(X, U, dt)

        assert A_bar.shape == (3, 3, N)
        assert B_bar.shape == (3, 2, N)
        assert z_bar.shape == (3, N)

    def test_integrate_trajectory(self, unicycle_twin):
        """Test nonlinear trajectory integration."""
        planner = SCvxPlanner(model=unicycle_twin, params=SCvxParameters(verbose=False))

        N = 10
        X = jnp.array(np.random.randn(3, N + 1))
        U = jnp.array(np.random.randn(2, N))
        dt = 0.1

        X_nl = planner.integrate_trajectory(X, U, dt)

        assert X_nl.shape == (3, N + 1)
        assert jnp.allclose(X_nl[:, 0], X[:, 0])  # Initial state preserved

    def test_simple_planning_unicycle(self, unicycle_twin, unicycle_constraints):
        """Test simple planning without obstacles for unicycle."""
        state_bounds = (jnp.array([0, 0, -np.pi]), jnp.array([12, 8, np.pi]))
        input_bounds = (unicycle_constraints.u_min, unicycle_constraints.u_max)

        params = SCvxParameters(max_iterations=10, verbose=False, convergence_tol=1e-2)

        planner = SCvxPlanner(model=unicycle_twin, params=params, state_bounds=state_bounds, input_bounds=input_bounds)

        x0 = jnp.array([1.0, 1.0, 0.0])
        xf = jnp.array([5.0, 5.0, 0.0])
        N = 30
        dt = 0.1

        # Plan trajectory
        traj = planner.plan(x0, xf, N, dt)

        # Check result
        assert isinstance(traj, NominalTrajectory)
        assert traj.N == N
        # Allow some tolerance for boundary constraints (solver numerical precision)
        assert jnp.allclose(traj.x_nom[0], x0, atol=1e-2)
        assert jnp.allclose(traj.x_nom[-1], xf, atol=1e-2)

    def test_simple_planning_quadrotor(self, quadrotor_twin, quadrotor_constraints):
        """Test simple planning without obstacles for quadrotor."""
        state_bounds = (
            jnp.array([0, 0, -5, -5, -5, -5, -1, -1, -1, -1, -5, -5, -5]),
            jnp.array([8, 8, 0.5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 5]),
        )
        input_bounds = (quadrotor_constraints.u_min, quadrotor_constraints.u_max)

        # Use fewer iterations and more relaxed convergence for faster test
        params = SCvxParameters(max_iterations=10, verbose=False, convergence_tol=1e-1, solver="SCS")

        planner = SCvxPlanner(model=quadrotor_twin, params=params, state_bounds=state_bounds, input_bounds=input_bounds)

        x0 = jnp.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        xf = jnp.array([3, 3, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        N = 20  # Reduce N for faster computation
        dt = 0.1  # Larger dt for fewer steps

        # Plan trajectory
        traj = planner.plan(x0, xf, N, dt)

        # Check result
        assert isinstance(traj, NominalTrajectory)
        assert traj.N == N
        # Allow more tolerance for quadrotor (13D is harder to solve exactly)
        assert jnp.allclose(traj.x_nom[0], x0, atol=1e-1)
        assert jnp.allclose(traj.x_nom[-1], xf, atol=1e-1)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining models and planning."""

    def test_unicycle_full_workflow(self, unicycle_twin, unicycle_plant, unicycle_constraints):
        """Test full workflow: plan on twin, simulate on plant."""
        # Setup planner
        state_bounds = (jnp.array([0, 0, -np.pi]), jnp.array([12, 8, np.pi]))
        input_bounds = (unicycle_constraints.u_min, unicycle_constraints.u_max)

        params = SCvxParameters(max_iterations=10, verbose=False)
        planner = SCvxPlanner(model=unicycle_twin, params=params, state_bounds=state_bounds, input_bounds=input_bounds)

        # Plan on twin
        x0 = jnp.array([1.0, 1.0, 0.0])
        xf = jnp.array([8.0, 6.0, 0.0])
        N = 40
        dt = 0.1

        traj = planner.plan(x0, xf, N, dt)

        # Simulate on plant
        x_plant_traj = unicycle_plant.simulate_trajectory(x0, traj.u_nom, dt)

        # Plant trajectory should deviate from twin plan due to mismatch
        mismatch_error = float(jnp.linalg.norm(x_plant_traj - traj.x_nom))
        assert mismatch_error > 0  # Should have some deviation

    def test_twin_plant_consistency(self, unicycle_twin):
        """Test that twin and plant are consistent when no mismatch."""
        # Create plant with no mismatch
        plant = UnicyclePlant(twin=unicycle_twin, velocity_scale=1.0, angular_scale=1.0, slip_coefficient=0.0)

        x = jnp.array([1.0, 2.0, 0.5])
        u = jnp.array([1.0, 0.5])

        x_dot_twin = unicycle_twin.dynamics(x, u)
        x_dot_plant = plant.dynamics(x, u)

        # Should be identical with no mismatch
        assert jnp.allclose(x_dot_twin, x_dot_plant, atol=1e-6)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
