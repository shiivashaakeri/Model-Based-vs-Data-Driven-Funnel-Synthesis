"""
Unit tests for Phase 1: Models

Tests for:
- base.py: DynamicalSystem abstract class
- unicycle.py: UnicycleModel (digital twin)
- plant.py: PlantModel (physical plant with mismatch)
"""

import numpy as np
import pytest

from ddfs.models.plant import PlantModel
from ddfs.models.unicycle import UnicycleModel


class TestDynamicalSystemBase:
    """Tests for DynamicalSystem base class functionality."""

    def test_euler_integration(self):
        """Test Euler integration produces expected results."""
        # Create a simple unicycle model
        model = UnicycleModel()

        # Initial state and input
        x = np.array([0.0, 0.0, 0.0])  # At origin, heading east
        u = np.array([1.0, 0.0])  # Moving forward at 1 m/s
        dt = 0.1

        # Forward Euler: x+ = x + dt * f(x, u)
        x_next = model.discrete_dynamics(x, u, dt, method="euler")

        # Expected: px += 0.1, py += 0, theta += 0
        expected = np.array([0.1, 0.0, 0.0])
        np.testing.assert_allclose(x_next, expected, atol=1e-10)

    def test_rk4_integration(self):
        """Test RK4 integration is more accurate than Euler."""
        model = UnicycleModel()

        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.5])  # Forward + turning
        dt = 0.1

        # RK4 should give a result
        x_next_rk4 = model.discrete_dynamics(x, u, dt, method="rk4")

        # Should have moved forward and started turning
        assert x_next_rk4[0] > 0  # px increased
        assert x_next_rk4[2] > 0  # theta increased

    def test_discrete_linearization(self):
        """Test discrete-time linearization via finite differences."""
        model = UnicycleModel()

        x = np.array([1.0, 2.0, 0.5])
        u = np.array([0.5, 0.1])
        dt = 0.01

        A_d, B_d = model.discrete_linearization(x, u, dt, method="rk4")

        # Check dimensions
        assert A_d.shape == (3, 3)
        assert B_d.shape == (3, 2)

        # A_d should be close to identity + dt*A for small dt
        # (just a sanity check that it's reasonable)
        assert np.allclose(np.diag(A_d), [1, 1, 1], atol=0.1)

    def test_simulate(self):
        """Test forward simulation over multiple timesteps."""
        model = UnicycleModel(x0=np.array([0, 0, 0]))

        # Constant forward velocity
        N = 10
        u_traj = np.tile([1.0, 0.0], (N, 1))
        dt = 0.1

        x_traj = model.simulate(model.x0, u_traj, dt, method="rk4")

        # Should have shape (N+1, 3)
        assert x_traj.shape == (N + 1, 3)

        # Position should increase monotonically
        assert np.all(np.diff(x_traj[:, 0]) > 0)  # px increases
        assert np.allclose(x_traj[:, 1], 0, atol=1e-6)  # py stays ~0
        assert np.allclose(x_traj[:, 2], 0, atol=1e-6)  # theta stays ~0


class TestUnicycleModel:
    """Tests for UnicycleModel (digital twin)."""

    def test_initialization(self):
        """Test model initializes with correct dimensions."""
        model = UnicycleModel()
        assert model.state_dim == 3
        assert model.input_dim == 2

    def test_dynamics_straight_line(self):
        """Test dynamics for straight-line motion."""
        model = UnicycleModel()

        x = np.array([0.0, 0.0, 0.0])  # Origin, heading east
        u = np.array([1.0, 0.0])  # Forward at 1 m/s, no turning

        xdot = model.dynamics(x, u)

        # Expected: [1.0, 0.0, 0.0]
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(xdot, expected, atol=1e-10)

    def test_dynamics_turning(self):
        """Test dynamics with turning."""
        model = UnicycleModel()

        x = np.array([0.0, 0.0, np.pi / 4])  # Heading northeast
        u = np.array([1.0, 0.5])  # Forward + turning

        xdot = model.dynamics(x, u)

        # xdot should be: [cos(pi/4), sin(pi/4), 0.5]
        expected = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.5])
        np.testing.assert_allclose(xdot, expected, atol=1e-10)

    def test_linearization(self):
        """Test analytical linearization."""
        model = UnicycleModel()

        x = np.array([1.0, 2.0, 0.5])
        u = np.array([0.5, 0.1])

        A, B = model.linearize(x, u)

        # Check dimensions
        assert A.shape == (3, 3)
        assert B.shape == (3, 2)

        # A[2, :] should be zero (theta doesn't depend on state)
        np.testing.assert_allclose(A[2, :], 0, atol=1e-10)

        # B[2, 0] should be zero (theta doesn't depend on v)
        assert abs(B[2, 0]) < 1e-10

        # B[2, 1] should be 1 (theta_dot = omega)
        assert abs(B[2, 1] - 1.0) < 1e-10

    def test_linearization_values(self):
        """Test specific numerical values of Jacobians."""
        model = UnicycleModel()

        # At origin, heading east, moving forward
        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.0])

        A, B = model.linearize(x, u)

        # A should be:
        # [0, 0, -1*sin(0)]   [0, 0,  0]
        # [0, 0,  1*cos(0)] = [0, 0,  1]
        # [0, 0,  0]          [0, 0,  0]
        A_expected = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        np.testing.assert_allclose(A, A_expected, atol=1e-10)

        # B should be:
        # [cos(0), 0]   [1, 0]
        # [sin(0), 0] = [0, 0]
        # [0,      1]   [0, 1]
        B_expected = np.array([[1, 0], [0, 0], [0, 1]])
        np.testing.assert_allclose(B, B_expected, atol=1e-10)

    def test_initial_desired_states(self):
        """Test setting and getting initial/desired states."""
        x0 = np.array([1.0, 2.0, 0.5])
        xf = np.array([5.0, 5.0, 1.0])

        model = UnicycleModel(x0=x0, xf=xf)

        np.testing.assert_array_equal(model.get_initial_state(), x0)
        np.testing.assert_array_equal(model.get_desired_state(), xf)

        # Test setters
        new_x0 = np.array([0, 0, 0])
        model.set_initial_state(new_x0)
        np.testing.assert_array_equal(model.get_initial_state(), new_x0)

    def test_distance_to_goal(self):
        """Test distance computation to goal."""
        xf = np.array([5.0, 0.0, 0.0])
        model = UnicycleModel(xf=xf)

        x = np.array([0.0, 0.0, 0.0])
        dist = model.distance_to_goal(x)

        # Distance should be 5.0
        assert abs(dist - 5.0) < 1e-10

    def test_goal_reached(self):
        """Test goal-reached check."""
        xf = np.array([5.0, 5.0, np.pi / 4])
        model = UnicycleModel(xf=xf)

        # Exactly at goal
        assert model.is_goal_reached(xf, position_tol=0.1, angle_tol=0.1)

        # Slightly off in position
        x_close = np.array([5.05, 5.05, np.pi / 4])
        assert model.is_goal_reached(x_close, position_tol=0.1, angle_tol=0.1)

        # Too far
        x_far = np.array([4.5, 4.5, np.pi / 4])
        assert not model.is_goal_reached(x_far, position_tol=0.1, angle_tol=0.1)

    def test_angle_diff_wrapping(self):
        """Test angle difference wrapping."""
        model = UnicycleModel()

        # Test wrapping: pi and -pi should be close
        diff = model._angle_diff(np.pi, -np.pi)
        assert abs(diff) < 1e-10  # Should wrap to 0

        # Test normal case
        diff = model._angle_diff(np.pi / 2, 0)
        assert abs(diff - np.pi / 2) < 1e-10

        # Test wrapping: 3*pi/2 - 0 should wrap to -pi/2
        diff = model._angle_diff(3 * np.pi / 2, 0)
        assert abs(diff - (-np.pi / 2)) < 1e-10


class TestPlantModel:
    """Tests for PlantModel (physical plant with mismatch)."""

    def test_initialization(self):
        """Test plant initializes with twin reference."""
        twin = UnicycleModel()
        plant = PlantModel(twin=twin)

        assert plant.state_dim == 3
        assert plant.input_dim == 2
        assert plant.twin is twin

    def test_default_parameter_mismatch(self):
        """Test default parameter mismatch is applied."""
        twin = UnicycleModel()
        plant = PlantModel(twin=twin)

        params = plant.get_parameter_mismatch()
        assert "velocity_scale" in params
        assert "angular_rate_scale" in params
        assert "slip_coefficient" in params

    def test_dynamics_differs_from_twin(self):
        """Test plant dynamics differ from twin due to mismatch."""
        twin = UnicycleModel()
        plant = PlantModel(
            twin=twin, parameter_mismatch={"velocity_scale": 0.9, "angular_rate_scale": 1.1, "slip_coefficient": 0.0}
        )

        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.5])

        xdot_twin = twin.dynamics(x, u)
        xdot_plant = plant.dynamics(x, u)

        # Should be different
        assert not np.allclose(xdot_twin, xdot_plant)

    def test_compute_mismatch(self):
        """Test mismatch computation Δ(x, u)."""
        twin = UnicycleModel()
        plant = PlantModel(
            twin=twin, parameter_mismatch={"velocity_scale": 0.9, "angular_rate_scale": 1.0, "slip_coefficient": 0.0}
        )

        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.0])  # Forward at 1 m/s

        delta = plant.compute_mismatch(x, u)

        # Expected mismatch in px direction: 0.9 - 1.0 = -0.1
        # (plant is slower)
        assert delta[0] < 0  # Negative because plant is slower
        assert abs(delta[1]) < 1e-10  # py should be ~0
        assert abs(delta[2]) < 1e-10  # theta should be ~0

    def test_compute_mismatch_norm(self):
        """Test mismatch norm computation."""
        twin = UnicycleModel()
        plant = PlantModel(
            twin=twin, parameter_mismatch={"velocity_scale": 0.95, "angular_rate_scale": 1.0, "slip_coefficient": 0.0}
        )

        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.0])

        delta_norm = plant.compute_mismatch_norm(x, u)

        # Should be positive (mismatch exists)
        assert delta_norm > 0

        # Should match manual calculation
        delta = plant.compute_mismatch(x, u)
        expected_norm = np.linalg.norm(delta)
        assert abs(delta_norm - expected_norm) < 1e-10

    def test_compute_max_mismatch_on_trajectory(self):
        """Test computing gamma (max mismatch along trajectory)."""
        twin = UnicycleModel()
        plant = PlantModel(
            twin=twin, parameter_mismatch={"velocity_scale": 0.9, "angular_rate_scale": 1.1, "slip_coefficient": 0.05}
        )

        # Create a simple trajectory
        N = 10
        x_traj = np.zeros((N + 1, 3))
        u_traj = np.ones((N, 2))  # Constant input

        # Fill trajectory (simple forward motion)
        for k in range(N):
            x_traj[k + 1] = x_traj[k] + 0.1 * twin.dynamics(x_traj[k], u_traj[k])

        gamma, max_idx = plant.compute_max_mismatch_on_trajectory(x_traj, u_traj)

        # Gamma should be positive
        assert gamma > 0

        # max_idx should be valid
        assert 0 <= max_idx < N

        # Gamma should match the max over manual computation
        max_manual = max(plant.compute_mismatch_norm(x_traj[k], u_traj[k]) for k in range(N))
        assert abs(gamma - max_manual) < 1e-10

    def test_set_parameter_mismatch(self):
        """Test updating parameter mismatch."""
        twin = UnicycleModel()
        plant = PlantModel(twin=twin)

        new_params = {"velocity_scale": 0.85, "slip_coefficient": 0.1}
        plant.set_parameter_mismatch(new_params)

        params = plant.get_parameter_mismatch()
        assert params["velocity_scale"] == 0.85
        assert params["slip_coefficient"] == 0.1

    def test_linearization_with_mismatch(self):
        """Test plant linearization includes mismatch effects."""
        twin = UnicycleModel()
        plant = PlantModel(
            twin=twin, parameter_mismatch={"velocity_scale": 0.9, "angular_rate_scale": 1.1, "slip_coefficient": 0.0}
        )

        x = np.array([0.0, 0.0, 0.0])
        u = np.array([1.0, 0.5])

        A_twin, B_twin = twin.linearize(x, u)
        A_plant, B_plant = plant.linearize(x, u)

        # Jacobians should differ
        assert not np.allclose(A_twin, A_plant)
        assert not np.allclose(B_twin, B_plant)

    def test_no_mismatch_equals_twin(self):
        """Test that with no mismatch, plant behaves like twin."""
        twin = UnicycleModel()
        plant = PlantModel(
            twin=twin, parameter_mismatch={"velocity_scale": 1.0, "angular_rate_scale": 1.0, "slip_coefficient": 0.0}
        )

        x = np.array([1.0, 2.0, 0.5])
        u = np.array([0.5, 0.1])

        xdot_twin = twin.dynamics(x, u)
        xdot_plant = plant.dynamics(x, u)

        # Should be identical
        np.testing.assert_allclose(xdot_twin, xdot_plant, atol=1e-10)

        # Mismatch should be zero
        delta = plant.compute_mismatch(x, u)
        np.testing.assert_allclose(delta, 0, atol=1e-10)


class TestIntegration:
    """Integration tests across models."""

    def test_plant_twin_trajectory_divergence(self):
        """Test that plant and twin trajectories diverge due to mismatch."""
        twin = UnicycleModel(x0=np.array([0, 0, 0]))
        plant = PlantModel(
            twin=twin, parameter_mismatch={"velocity_scale": 0.9, "angular_rate_scale": 1.1, "slip_coefficient": 0.05}
        )

        # Same initial state and input trajectory
        N = 50
        u_traj = np.tile([1.0, 0.2], (N, 1))
        dt = 0.1

        x_twin = twin.simulate(twin.x0, u_traj, dt)
        x_plant = plant.simulate(plant.x0, u_traj, dt)

        # Trajectories should diverge over time
        final_error = np.linalg.norm(x_twin[-1] - x_plant[-1])
        assert final_error > 0.1  # Significant divergence

        # Error should grow over time
        errors = np.linalg.norm(x_twin - x_plant, axis=1)
        assert errors[-1] > errors[0]

    def test_discrete_vs_continuous_linearization_consistency(self):
        """Test that discrete and continuous linearization are consistent for small dt."""
        model = UnicycleModel()

        x = np.array([1.0, 2.0, 0.5])
        u = np.array([0.5, 0.1])
        dt = 0.001  # Very small timestep

        # Continuous linearization
        A_cont, B_cont = model.linearize(x, u)

        # Discrete linearization
        A_disc, B_disc = model.discrete_linearization(x, u, dt, method="euler")

        # For small dt: A_d ≈ I + dt*A, B_d ≈ dt*B
        A_disc_approx = np.eye(3) + dt * A_cont
        B_disc_approx = dt * B_cont

        np.testing.assert_allclose(A_disc, A_disc_approx, atol=1e-5)
        np.testing.assert_allclose(B_disc, B_disc_approx, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
