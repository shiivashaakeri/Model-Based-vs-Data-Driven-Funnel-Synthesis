# ddfs/tests/test_models.py

# ddfs/tests/test_models.py

"""
Test suite for dynamics models.

Tests:
- Base class functionality
- Unicycle twin and plant models
- Quadrotor twin and plant models
- Constraint checking
- Jacobian computation
- Integration accuracy
"""

import unittest

import jax.numpy as jnp
import numpy as np

from ddfs.models import (
    QuadrotorConstraints,
    QuadrotorPlant,
    QuadrotorTwin,
    UnicycleConstraints,
    UnicyclePlant,
    UnicycleTwin,
    create_plant_from_config,
    create_quadrotor_example,
    create_unicycle_example,
)


class TestUnicycleTwin(unittest.TestCase):
    """Test unicycle twin model."""

    def setUp(self):
        """Create unicycle twin instance."""
        self.dt = 0.1
        self.twin = UnicycleTwin(dt=self.dt)

    def test_dimensions(self):
        """Test state and input dimensions."""
        self.assertEqual(self.twin.state_dim, 3)
        self.assertEqual(self.twin.input_dim, 2)

    def test_dynamics_straight(self):
        """Test straight-line motion."""
        x = jnp.array([0.0, 0.0, 0.0])  # Origin, heading east
        u = jnp.array([1.0, 0.0])  # 1 m/s forward, no turning

        x_dot = self.twin.dynamics(x, u)

        self.assertAlmostEqual(float(x_dot[0]), 1.0, places=5)  # ẋ = v
        self.assertAlmostEqual(float(x_dot[1]), 0.0, places=5)  # ẏ = 0
        self.assertAlmostEqual(float(x_dot[2]), 0.0, places=5)  # θ̇ = 0

    def test_dynamics_turning(self):
        """Test circular motion."""
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 1.0])  # 1 m/s forward, 1 rad/s turning

        x_dot = self.twin.dynamics(x, u)

        self.assertAlmostEqual(float(x_dot[0]), 1.0, places=5)
        self.assertAlmostEqual(float(x_dot[1]), 0.0, places=5)
        self.assertAlmostEqual(float(x_dot[2]), 1.0, places=5)

    def test_step_integration(self):
        """Test RK4 integration."""
        x0 = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.0])

        x1 = self.twin.step(x0, u)

        # After 0.1s at 1 m/s, should move ~0.1m forward
        self.assertAlmostEqual(float(x1[0]), 0.1, places=3)
        self.assertAlmostEqual(float(x1[1]), 0.0, places=3)
        self.assertAlmostEqual(float(x1[2]), 0.0, places=3)

    def test_jacobians(self):
        """Test Jacobian computation."""
        x = jnp.array([1.0, 1.0, 0.5])
        u = jnp.array([1.0, 0.5])

        A, B = self.twin.jacobians(x, u)

        # Check dimensions
        self.assertEqual(A.shape, (3, 3))
        self.assertEqual(B.shape, (3, 2))

        # Verify against finite differences
        eps = 1e-5
        A_fd = np.zeros((3, 3))
        for i in range(3):
            x_plus = x.at[i].add(eps)
            x_minus = x.at[i].add(-eps)
            A_fd[:, i] = (self.twin.dynamics(x_plus, u) - self.twin.dynamics(x_minus, u)) / (2 * eps)

        # Use slightly relaxed tolerance due to finite difference numerical errors
        np.testing.assert_allclose(A, A_fd, atol=2e-3, rtol=1e-3)

    def test_normalize_state(self):
        """Test angle normalization."""
        x = jnp.array([1.0, 1.0, 7.0])  # θ > 2π
        x_norm = self.twin.normalize_state(x)

        # θ should be wrapped to [-π, π]
        self.assertGreaterEqual(float(x_norm[2]), -np.pi)
        self.assertLessEqual(float(x_norm[2]), np.pi)

    def test_trajectory_simulation(self):
        """Test full trajectory simulation."""
        x0 = jnp.array([0.0, 0.0, 0.0])
        N = 10
        u_traj = jnp.ones((N, 2)) * jnp.array([1.0, 0.0])

        x_traj = self.twin.simulate_trajectory(x0, u_traj)

        self.assertEqual(x_traj.shape, (N + 1, 3))
        # Should move forward ~1.0m
        self.assertAlmostEqual(float(x_traj[-1, 0]), 1.0, places=1)


class TestUnicyclePlant(unittest.TestCase):
    """Test unicycle plant with mismatch."""

    def setUp(self):
        """Create twin and plant."""
        self.twin = UnicycleTwin(dt=0.1)
        self.plant = UnicyclePlant(twin=self.twin, velocity_scale=0.95, angular_scale=1.05, slip_coefficient=0.02)

    def test_dimensions(self):
        """Test dimensions match twin."""
        self.assertEqual(self.plant.state_dim, self.twin.state_dim)
        self.assertEqual(self.plant.input_dim, self.twin.input_dim)

    def test_mismatch_exists(self):
        """Test plant differs from twin."""
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.5])

        f_plant = self.plant.dynamics(x, u)
        f_twin = self.twin.dynamics(x, u)

        # Should be different due to mismatch
        self.assertGreater(float(jnp.linalg.norm(f_plant - f_twin)), 0.01)

    def test_compute_mismatch(self):
        """Test mismatch computation."""
        x = jnp.array([1.0, 1.0, 0.5])
        u = jnp.array([1.0, 0.5])

        gamma = self.plant.compute_mismatch(x, u)

        # Should be non-zero positive
        self.assertGreater(float(gamma), 0.0)
        self.assertLess(float(gamma), 1.0)  # Reasonable bound

    def test_velocity_scaling(self):
        """Test velocity mismatch effect."""
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.0, 0.0])

        x_plant = self.plant.step(x, u)
        x_twin = self.twin.step(x, u)

        # Plant should move less (0.95 scale)
        self.assertLess(float(x_plant[0]), float(x_twin[0]))


class TestUnicycleConstraints(unittest.TestCase):
    """Test unicycle constraints."""

    def setUp(self):
        """Create constraints."""
        self.constraints = UnicycleConstraints(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, v_min=0.0, v_max=2.0, omega_max=2.0
        )

    def test_state_checking(self):
        """Test state constraint checking."""
        x_valid = jnp.array([5.0, 5.0, 0.0])
        x_invalid = jnp.array([15.0, 5.0, 0.0])

        self.assertTrue(self.constraints.check_state(x_valid))
        self.assertFalse(self.constraints.check_state(x_invalid))

    def test_input_checking(self):
        """Test input constraint checking."""
        u_valid = jnp.array([1.0, 1.0])
        u_invalid = jnp.array([3.0, 1.0])

        self.assertTrue(self.constraints.check_input(u_valid))
        self.assertFalse(self.constraints.check_input(u_invalid))

    def test_input_clipping(self):
        """Test input clipping."""
        u_large = jnp.array([5.0, 5.0])
        u_clipped = self.constraints.clip_input(u_large)

        self.assertLessEqual(float(u_clipped[0]), 2.0)
        self.assertLessEqual(float(u_clipped[1]), 2.0)

    def test_config_round_trip(self):
        """Test to_dict and from_config."""
        config = self.constraints.to_dict()
        constraints2 = UnicycleConstraints.from_config(config)

        np.testing.assert_array_equal(self.constraints.x_min, constraints2.x_min)


class TestQuadrotorTwin(unittest.TestCase):
    """Test quadrotor twin model."""

    def setUp(self):
        """Create quadrotor twin."""
        self.dt = 0.05
        self.twin = QuadrotorTwin(mass=0.5, inertia=jnp.diag(jnp.array([0.01, 0.01, 0.02])), dt=self.dt)

    def test_dimensions(self):
        """Test dimensions."""
        self.assertEqual(self.twin.state_dim, 13)
        self.assertEqual(self.twin.input_dim, 4)

    def test_hover_equilibrium(self):
        """Test hover at origin."""
        # Hovering state: zero position/velocity, identity quaternion
        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # qw = 1 (identity quaternion)

        # Thrust balances gravity
        u = jnp.array([self.twin.m * self.twin.g, 0.0, 0.0, 0.0])

        x_dot = self.twin.dynamics(x, u)

        # Should be nearly stationary (small numerical errors ok)
        self.assertLess(float(jnp.linalg.norm(x_dot)), 0.1)

    def test_free_fall(self):
        """Test free fall dynamics."""
        x = jnp.zeros(13)
        x = x.at[6].set(1.0)  # Identity quaternion
        u = jnp.zeros(4)  # No thrust

        x_dot = self.twin.dynamics(x, u)

        # Should accelerate downward at ~9.81 m/s²
        self.assertAlmostEqual(float(x_dot[5]), self.twin.g, places=1)

    def test_quaternion_normalization(self):
        """Test quaternion normalization."""
        x = jnp.zeros(13)
        x = x.at[6:10].set(jnp.array([1.0, 1.0, 1.0, 1.0]))

        x_norm = self.twin.normalize_state(x)
        q_norm = jnp.linalg.norm(x_norm[6:10])

        self.assertAlmostEqual(float(q_norm), 1.0, places=6)

    def test_euler_quaternion_conversion(self):
        """Test Euler <-> quaternion conversion."""
        roll, pitch, yaw = 0.1, 0.2, 0.3

        q = QuadrotorTwin.euler_to_quaternion(roll, pitch, yaw)
        roll2, pitch2, yaw2 = self.twin.quaternion_to_euler(q)

        self.assertAlmostEqual(roll, roll2, places=5)
        self.assertAlmostEqual(pitch, pitch2, places=5)
        self.assertAlmostEqual(yaw, yaw2, places=5)

    def test_step_integration(self):
        """Test integration step."""
        x = jnp.zeros(13)
        x = x.at[6].set(1.0)
        u = jnp.array([self.twin.m * self.twin.g, 0.0, 0.0, 0.0])

        x1 = self.twin.step(x, u)

        # Should remain near hover
        self.assertLess(float(jnp.linalg.norm(x1[:3])), 0.01)


class TestQuadrotorPlant(unittest.TestCase):
    """Test quadrotor plant with mismatch."""

    def setUp(self):
        """Create twin and plant."""
        self.twin = QuadrotorTwin(mass=0.5, dt=0.05)
        self.plant = QuadrotorPlant(
            twin=self.twin, mass_scale=0.95, inertia_scale=1.05, drag_coefficient=0.01, thrust_efficiency=0.95
        )

    def test_dimensions(self):
        """Test dimensions match twin."""
        self.assertEqual(self.plant.state_dim, self.twin.state_dim)
        self.assertEqual(self.plant.input_dim, self.twin.input_dim)

    def test_mismatch_exists(self):
        """Test plant differs from twin."""
        x = jnp.zeros(13)
        x = x.at[6].set(1.0)
        u = jnp.array([5.0, 0.1, 0.1, 0.1])

        gamma = self.plant.compute_mismatch(x, u)

        # Should be non-zero
        self.assertGreater(float(gamma), 0.01)

    def test_thrust_efficiency(self):
        """Test thrust efficiency effect."""
        # Start above ground with some velocity to see drag effect
        x = jnp.zeros(13)
        x = x.at[2].set(-2.0)  # z = -2.0 (2m above ground in NED)
        x = x.at[5].set(-1.0)  # vz = -1.0 (moving up)
        x = x.at[6].set(1.0)  # Identity quaternion

        # Thrust command (hover thrust for nominal mass)
        u = jnp.array([self.twin.m * self.twin.g, 0.0, 0.0, 0.0])

        # Check dynamics directly - plant should have different acceleration
        f_twin = self.twin.dynamics(x, u)
        f_plant = self.plant.dynamics(x, u)

        # Plant should have different dynamics due to drag and efficiency
        # Check z-acceleration (index 5 in dynamics output)
        # With drag, plant should decelerate upward motion more
        self.assertNotAlmostEqual(float(f_plant[5]), float(f_twin[5]), places=3)


class TestQuadrotorConstraints(unittest.TestCase):
    """Test quadrotor constraints."""

    def setUp(self):
        """Create constraints."""
        self.constraints = QuadrotorConstraints(
            x_min=-5.0,
            x_max=5.0,
            y_min=-5.0,
            y_max=5.0,
            z_min=-5.0,
            z_max=0.5,
            v_max=3.0,
            omega_max=2.0,
            T_min=0.0,
            T_max=10.0,
            tau_max=1.0,
        )

    def test_state_checking(self):
        """Test state constraint checking."""
        x_valid = jnp.zeros(13)
        x_valid = x_valid.at[6].set(1.0)

        x_invalid = jnp.zeros(13)
        x_invalid = x_invalid.at[0].set(10.0)  # Outside bounds
        x_invalid = x_invalid.at[6].set(1.0)

        self.assertTrue(self.constraints.check_state(x_valid))
        self.assertFalse(self.constraints.check_state(x_invalid))

    def test_input_checking(self):
        """Test input constraint checking."""
        u_valid = jnp.array([5.0, 0.5, 0.5, 0.5])
        u_invalid = jnp.array([15.0, 0.5, 0.5, 0.5])

        self.assertTrue(self.constraints.check_input(u_valid))
        self.assertFalse(self.constraints.check_input(u_invalid))

    def test_input_clipping(self):
        """Test input clipping."""
        u_large = jnp.array([20.0, 5.0, 5.0, 5.0])
        u_clipped = self.constraints.clip_input(u_large)

        self.assertLessEqual(float(u_clipped[0]), 10.0)
        self.assertLessEqual(float(jnp.max(jnp.abs(u_clipped[1:]))), 1.0)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory and example creation functions."""

    def test_create_unicycle_example(self):
        """Test unicycle example creation."""
        config = create_unicycle_example()

        self.assertIn("system", config)
        self.assertIn("planning", config)
        self.assertIn("constraints", config)
        self.assertEqual(config["system"]["state_dim"], 3)
        self.assertEqual(config["system"]["input_dim"], 2)

    def test_create_quadrotor_example(self):
        """Test quadrotor example creation."""
        config = create_quadrotor_example()

        self.assertIn("system", config)
        self.assertIn("planning", config)
        self.assertIn("constraints", config)
        self.assertEqual(config["system"]["state_dim"], 13)
        self.assertEqual(config["system"]["input_dim"], 4)

    def test_create_plant_unicycle(self):
        """Test plant creation from config - unicycle."""
        twin = UnicycleTwin(dt=0.1)
        config = {"velocity_scale": 0.95, "angular_scale": 1.05, "slip_coefficient": 0.02}

        plant = create_plant_from_config(twin, config)

        self.assertIsInstance(plant, UnicyclePlant)
        self.assertEqual(plant.velocity_scale, 0.95)

    def test_create_plant_quadrotor(self):
        """Test plant creation from config - quadrotor."""
        twin = QuadrotorTwin(mass=0.5)
        config = {"mass_scale": 0.95, "inertia_scale": 1.05, "drag_coefficient": 0.01, "thrust_efficiency": 0.95}

        plant = create_plant_from_config(twin, config)

        self.assertIsInstance(plant, QuadrotorPlant)
        self.assertEqual(plant.mass_scale, 0.95)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases."""

    def test_unicycle_large_angle(self):
        """Test unicycle with large angle."""
        twin = UnicycleTwin(dt=0.1)
        x = jnp.array([0.0, 0.0, 100.0])  # Large angle
        u = jnp.array([1.0, 0.5])

        x_dot = twin.dynamics(x, u)

        # Should still compute finite values
        self.assertTrue(jnp.all(jnp.isfinite(x_dot)))

    def test_quadrotor_zero_inertia_protection(self):
        """Test quadrotor doesn't divide by zero."""
        # Very small inertia (but not zero)
        twin = QuadrotorTwin(mass=0.5, inertia=jnp.diag(jnp.array([1e-8, 1e-8, 1e-8])))

        x = jnp.zeros(13)
        x = x.at[6].set(1.0)
        u = jnp.array([5.0, 0.1, 0.1, 0.1])

        x_dot = twin.dynamics(x, u)

        # Should compute without NaN/Inf
        self.assertTrue(jnp.all(jnp.isfinite(x_dot)))

    def test_long_simulation_stability(self):
        """Test long simulation doesn't explode."""
        twin = UnicycleTwin(dt=0.1)
        x0 = jnp.array([0.0, 0.0, 0.0])
        u_traj = jnp.ones((100, 2)) * jnp.array([1.0, 0.1])

        x_traj = twin.simulate_trajectory(x0, u_traj)

        # Should remain finite throughout
        self.assertTrue(jnp.all(jnp.isfinite(x_traj)))


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
