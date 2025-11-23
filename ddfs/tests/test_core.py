"""Comprehensive tests for ddfs.core module."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from ddfs.core import (
    CircleObstacle,
    DDFSConfig,
    QuadrotorConstraints,
    SphereObstacle,
    UnicycleConstraints,
    Workspace2D,
    Workspace3D,
    check_collision_free,
    create_obstacles_from_config,
    load_config,
    minimum_distance_to_obstacles,
)

# ============================================================================
# CONSTRAINTS TESTS
# ============================================================================


class TestUnicycleConstraints:
    """Test UnicycleConstraints class."""

    def test_creation(self):
        """Test UnicycleConstraints creation."""
    constraints = UnicycleConstraints(
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=8.0,
        v_min=0.0,
        v_max=2.0,
        omega_max=2.0,
    )

        assert constraints.x_min[0] == 0.0
        assert constraints.x_max[0] == 10.0
        assert constraints.u_min[0] == 0.0
        assert constraints.u_max[0] == 2.0

    def test_check_state_valid(self):
        """Test state validation with valid states."""
        constraints = UnicycleConstraints(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=8.0, v_min=0.0, v_max=2.0, omega_max=2.0
        )

    # Valid state
        x = jnp.array([5.0, 4.0, 0.5])
    assert constraints.check_state(x)

        # State at boundary
        x_boundary = jnp.array([0.0, 0.0, -jnp.pi])
        assert constraints.check_state(x_boundary)

    def test_check_state_invalid(self):
        """Test state validation with invalid states."""
        constraints = UnicycleConstraints(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=8.0, v_min=0.0, v_max=2.0, omega_max=2.0
        )

        # Invalid x
        x_bad_x = jnp.array([15.0, 4.0, 0.5])
        assert not constraints.check_state(x_bad_x)

        # Invalid y
        x_bad_y = jnp.array([5.0, 10.0, 0.5])
        assert not constraints.check_state(x_bad_y)

    def test_check_input_valid(self):
        """Test input validation with valid inputs."""
        constraints = UnicycleConstraints(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=8.0, v_min=0.0, v_max=2.0, omega_max=2.0
        )

    # Valid input
        u = jnp.array([1.0, 0.5])
    assert constraints.check_input(u)

        # Input at boundary
        u_boundary = jnp.array([2.0, 2.0])
        assert constraints.check_input(u_boundary)

    def test_check_input_invalid(self):
        """Test input validation with invalid inputs."""
        constraints = UnicycleConstraints(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=8.0, v_min=0.0, v_max=2.0, omega_max=2.0
        )

        # Invalid v
        u_bad_v = jnp.array([3.0, 0.5])
        assert not constraints.check_input(u_bad_v)

        # Invalid omega
        u_bad_omega = jnp.array([1.0, 3.0])
        assert not constraints.check_input(u_bad_omega)

    def test_clip_state(self):
        """Test state clipping."""
        constraints = UnicycleConstraints(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=8.0, v_min=0.0, v_max=2.0, omega_max=2.0
        )

        # State outside bounds
        x_outside = jnp.array([15.0, 10.0, 0.5])
        x_clipped = constraints.clip_state(x_outside)

        assert constraints.check_state(x_clipped)
        assert x_clipped[0] == 10.0  # Clipped to x_max
        assert x_clipped[1] == 8.0  # Clipped to y_max

    def test_clip_input(self):
        """Test input clipping."""
        constraints = UnicycleConstraints(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=8.0, v_min=0.0, v_max=2.0, omega_max=2.0
        )

        # Input outside bounds
        u_bad = jnp.array([3.0, 5.0])
    u_clipped = constraints.clip_input(u_bad)

    assert constraints.check_input(u_clipped)
        assert u_clipped[0] == 2.0  # Clipped to v_max
        assert u_clipped[1] == 2.0  # Clipped to omega_max

    def test_to_dict(self):
        """Test conversion to dictionary."""
        constraints = UnicycleConstraints(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=8.0, v_min=0.0, v_max=2.0, omega_max=2.0
        )

        config = constraints.to_dict()

        assert "state_bounds" in config
        assert "input_bounds" in config
        assert config["state_bounds"]["x_min"] == 0.0
        assert config["input_bounds"]["v_max"] == 2.0

    def test_from_config(self):
        """Test creation from configuration."""
        config = {
            "state_bounds": {
                "x_min": 0.0,
                "x_max": 10.0,
                "y_min": 0.0,
                "y_max": 8.0,
            },
            "input_bounds": {
                "v_min": 0.0,
                "v_max": 2.0,
                "omega_max": 2.0,
            },
        }

        constraints = UnicycleConstraints.from_config(config)

        assert constraints.x_min[0] == 0.0
        assert constraints.x_max[0] == 10.0
        assert constraints.u_max[0] == 2.0

    def test_defaults(self):
        """Test default parameter values."""
        constraints = UnicycleConstraints()

        # Check defaults are applied
        assert constraints.x_min[0] == -10.0
        assert constraints.x_max[0] == 10.0
        assert constraints.u_max[0] == 2.0


class TestQuadrotorConstraints:
    """Test QuadrotorConstraints class."""

    def test_creation(self):
        """Test QuadrotorConstraints creation."""
        constraints = QuadrotorConstraints(
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

        assert constraints.p_min[0] == 0.0
        assert constraints.p_max[0] == 8.0
        assert constraints.v_max == 5.0
        assert constraints.omega_max == 5.0

    def test_check_state_valid(self):
        """Test state validation with valid states."""
    constraints = QuadrotorConstraints(
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

        # Valid state
    x = np.zeros(13)
    x[:3] = [4.0, 4.0, -2.0]  # Position
    x[6] = 1.0  # Quaternion w
        assert constraints.check_state(jnp.array(x))

        # State at boundary
        x_boundary = np.zeros(13)
        x_boundary[:3] = [0.0, 0.0, -5.0]
        x_boundary[6] = 1.0
        assert constraints.check_state(jnp.array(x_boundary))

    def test_check_state_invalid(self):
        """Test state validation with invalid states."""
        constraints = QuadrotorConstraints(
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

        # Invalid position
        x_bad_pos = np.zeros(13)
        x_bad_pos[:3] = [10.0, 4.0, -2.0]
        x_bad_pos[6] = 1.0
        assert not constraints.check_state(jnp.array(x_bad_pos))

        # Invalid velocity
        x_bad_vel = np.zeros(13)
        x_bad_vel[:3] = [4.0, 4.0, -2.0]
        x_bad_vel[3:6] = [10.0, 0.0, 0.0]  # Exceeds v_max
        x_bad_vel[6] = 1.0
        assert not constraints.check_state(jnp.array(x_bad_vel))

        # Invalid angular velocity
        x_bad_omega = np.zeros(13)
        x_bad_omega[:3] = [4.0, 4.0, -2.0]
        x_bad_omega[10:13] = [10.0, 0.0, 0.0]  # Exceeds omega_max
        x_bad_omega[6] = 1.0
        assert not constraints.check_state(jnp.array(x_bad_omega))

    def test_check_input_valid(self):
        """Test input validation with valid inputs."""
        constraints = QuadrotorConstraints(
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

    # Valid input
        u = jnp.array([0.5, 0.0, 0.0, 0.0])
    assert constraints.check_input(u)

        # Input at boundary
        u_boundary = jnp.array([1.0, 0.1, 0.1, 0.1])
        assert constraints.check_input(u_boundary)

    def test_check_input_invalid(self):
        """Test input validation with invalid inputs."""
        constraints = QuadrotorConstraints(
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

        # Invalid thrust
        u_bad_T = jnp.array([2.0, 0.0, 0.0, 0.0])
        assert not constraints.check_input(u_bad_T)

        # Invalid torque
        u_bad_tau = jnp.array([0.5, 0.2, 0.0, 0.0])
        assert not constraints.check_input(u_bad_tau)

    def test_clip_input(self):
        """Test input clipping."""
        constraints = QuadrotorConstraints(
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

        # Input outside bounds
        u_bad = jnp.array([2.0, 0.2, 0.2, 0.2])
        u_clipped = constraints.clip_input(u_bad)

        assert constraints.check_input(u_clipped)
        assert u_clipped[0] == 1.0  # Clipped to T_max
        assert abs(u_clipped[1]) <= 0.1  # Clipped to tau_max

    def test_to_dict(self):
        """Test conversion to dictionary."""
        constraints = QuadrotorConstraints(
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

        config = constraints.to_dict()

        assert "state_bounds" in config
        assert "input_bounds" in config
        assert config["state_bounds"]["x_min"] == 0.0
        assert config["input_bounds"]["T_max"] == 1.0

    def test_from_config(self):
        """Test creation from configuration."""
        config = {
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
                "tau_max": 0.1,
            },
        }

        constraints = QuadrotorConstraints.from_config(config)

        assert constraints.p_min[0] == 0.0
        assert constraints.p_max[0] == 8.0
        assert constraints.u_max[0] == 1.0


# ============================================================================
# WORKSPACE TESTS
# ============================================================================


class TestWorkspace2D:
    """Test Workspace2D class."""

    def test_creation(self):
        """Test Workspace2D creation."""
        workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)

        assert workspace.x_min == 0.0
        assert workspace.x_max == 12.0
        assert workspace.width == 12.0
        assert workspace.height == 8.0
        assert workspace.area == 96.0

    def test_creation_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        with pytest.raises(ValueError):
            Workspace2D(x_min=10.0, x_max=5.0, y_min=0.0, y_max=8.0)

        with pytest.raises(ValueError):
            Workspace2D(x_min=0.0, x_max=12.0, y_min=8.0, y_max=0.0)

    def test_contains(self):
        """Test point containment checking."""
    workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)

        # Point inside
    x_inside = np.array([6.0, 4.0, 0.0])
    assert workspace.contains(x_inside)

        # Point outside
    x_outside = np.array([15.0, 4.0, 0.0])
    assert not workspace.contains(x_outside)

        # Point at boundary
        x_boundary = np.array([0.0, 0.0, 0.0])
        assert workspace.contains(x_boundary)

        # Point with margin
        x_margin = np.array([0.5, 0.5, 0.0])
        assert workspace.contains(x_margin, margin=0.5)
        assert not workspace.contains(x_margin, margin=1.0)

    def test_distance_to_boundary(self):
        """Test distance to boundary computation."""
        workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)

        # Point in center
        x_center = np.array([6.0, 4.0, 0.0])
        dist = workspace.distance_to_boundary(x_center)
        assert dist > 0  # Inside
        assert dist == pytest.approx(4.0)  # Distance to nearest boundary

        # Point at boundary
        x_boundary = np.array([0.0, 4.0, 0.0])
        dist = workspace.distance_to_boundary(x_boundary)
        assert dist == pytest.approx(0.0)

        # Point outside
        x_outside = np.array([15.0, 4.0, 0.0])
        dist = workspace.distance_to_boundary(x_outside)
        assert dist < 0  # Outside

    def test_clip_to_workspace(self):
        """Test clipping to workspace."""
        workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)

        # Point outside
        x_outside = np.array([15.0, 10.0, 0.5])
        x_clipped = workspace.clip_to_workspace(x_outside)

        assert x_clipped[0] == 12.0
        assert x_clipped[1] == 8.0
        assert x_clipped[2] == 0.5  # θ unchanged

    def test_sample_random_point(self):
        """Test random point sampling."""
        workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)

        # Sample multiple points
        for _ in range(10):
    point = workspace.sample_random_point()
    assert len(point) == 2
    assert workspace.contains(np.append(point, 0.0))

        # Sample with margin
        for _ in range(10):
            point = workspace.sample_random_point(margin=1.0)
            assert workspace.contains(np.append(point, 0.0), margin=1.0)

    def test_get_corners(self):
        """Test corner point retrieval."""
        workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)

        corners = workspace.get_corners()
        assert corners.shape == (4, 2)

        # Check all corners are at boundaries
        for corner in corners:
            assert workspace.contains(np.append(corner, 0.0))

    def test_bounds_property(self):
        """Test bounds property."""
        workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)

        bounds = workspace.bounds
        assert bounds == (0.0, 12.0, 0.0, 8.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)

        config = workspace.to_dict()
        assert config["x_min"] == 0.0
        assert config["x_max"] == 12.0

    def test_from_config(self):
        """Test creation from configuration."""
        config = {"x_min": 0.0, "x_max": 12.0, "y_min": 0.0, "y_max": 8.0}

        workspace = Workspace2D.from_config(config)
        assert workspace.x_min == 0.0
        assert workspace.x_max == 12.0


class TestWorkspace3D:
    """Test Workspace3D class."""

    def test_creation(self):
        """Test Workspace3D creation."""
        workspace = Workspace3D(
            x_min=0.0, x_max=8.0, y_min=0.0, y_max=8.0, z_min=-5.0, z_max=0.5
        )

        assert workspace.x_min == 0.0
        assert workspace.x_max == 8.0
        assert workspace.width == 8.0
        assert workspace.height == 8.0
        assert workspace.depth == 5.5
        assert workspace.volume == 8.0 * 8.0 * 5.5

    def test_creation_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        with pytest.raises(ValueError):
            Workspace3D(x_min=10.0, x_max=5.0, y_min=0.0, y_max=8.0, z_min=-5.0, z_max=0.5)

    def test_contains(self):
        """Test point containment checking."""
    workspace = Workspace3D(
            x_min=0.0, x_max=8.0, y_min=0.0, y_max=8.0, z_min=-5.0, z_max=0.5
        )

        # Point inside
    x_inside = np.zeros(13)
    x_inside[:3] = [4.0, 4.0, -2.0]
    assert workspace.contains(x_inside)

        # Point outside
    x_outside = np.zeros(13)
    x_outside[:3] = [10.0, 4.0, -2.0]
    assert not workspace.contains(x_outside)

    def test_distance_to_boundary(self):
        """Test distance to boundary computation."""
        workspace = Workspace3D(
            x_min=0.0, x_max=8.0, y_min=0.0, y_max=8.0, z_min=-5.0, z_max=0.5
        )

        # Point in center
        x_center = np.zeros(13)
        x_center[:3] = [4.0, 4.0, -2.25]
        dist = workspace.distance_to_boundary(x_center)
        assert dist > 0  # Inside

    def test_clip_to_workspace(self):
        """Test clipping to workspace."""
        workspace = Workspace3D(
            x_min=0.0, x_max=8.0, y_min=0.0, y_max=8.0, z_min=-5.0, z_max=0.5
        )

        # Point outside
        x_outside = np.zeros(13)
        x_outside[:3] = [10.0, 10.0, -10.0]
        x_clipped = workspace.clip_to_workspace(x_outside)

        assert x_clipped[0] == 8.0
        assert x_clipped[1] == 8.0
        assert x_clipped[2] == -5.0

    def test_sample_random_point(self):
        """Test random point sampling."""
        workspace = Workspace3D(
            x_min=0.0, x_max=8.0, y_min=0.0, y_max=8.0, z_min=-5.0, z_max=0.5
        )

        # Sample multiple points
        for _ in range(10):
            point = workspace.sample_random_point()
            assert len(point) == 3
            x_test = np.zeros(13)
            x_test[:3] = point
            assert workspace.contains(x_test)

    def test_bounds_property(self):
        """Test bounds property."""
        workspace = Workspace3D(
            x_min=0.0, x_max=8.0, y_min=0.0, y_max=8.0, z_min=-5.0, z_max=0.5
        )

        bounds = workspace.bounds
        assert bounds == (0.0, 8.0, 0.0, 8.0, -5.0, 0.5)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        workspace = Workspace3D(
            x_min=0.0, x_max=8.0, y_min=0.0, y_max=8.0, z_min=-5.0, z_max=0.5
        )

        config = workspace.to_dict()
        assert config["z_min"] == -5.0

    def test_from_config(self):
        """Test creation from configuration."""
        config = {
            "x_min": 0.0,
            "x_max": 8.0,
            "y_min": 0.0,
            "y_max": 8.0,
            "z_min": -5.0,
            "z_max": 0.5,
        }

        workspace = Workspace3D.from_config(config)
        assert workspace.z_min == -5.0


# ============================================================================
# OBSTACLES TESTS
# ============================================================================


class TestCircleObstacle:
    """Test CircleObstacle class."""

    def test_creation(self):
        """Test CircleObstacle creation."""
        obs = CircleObstacle("obs_1", center=[5.0, 5.0], radius=1.0, safety_margin=0.25)

        assert obs.id == "obs_1"
        assert obs.radius == 1.0
        assert obs.safety_margin == 0.25
        assert obs.effective_radius == 1.25

    def test_creation_invalid_dimension(self):
        """Test that invalid center dimension raises error."""
        with pytest.raises(ValueError):
            CircleObstacle("obs_1", center=[1.0, 2.0, 3.0], radius=1.0)

    def test_contains(self):
        """Test point containment checking."""
    obs = CircleObstacle("obs_1", center=[5.0, 5.0], radius=1.0, safety_margin=0.25)

    # Point inside
    point_inside = np.array([5.5, 5.0])
    assert obs.contains(point_inside)

    # Point outside
    point_outside = np.array([10.0, 10.0])
    assert not obs.contains(point_outside)

        # Point on boundary
        point_boundary = np.array([6.0, 5.0])  # Exactly at radius
        assert obs.contains(point_boundary)

        # Point with margin
        point_with_margin = np.array([6.2, 5.0])  # Inside with margin
        assert obs.contains(point_with_margin, include_margin=True)
        assert not obs.contains(point_with_margin, include_margin=False)

    def test_distance_to(self):
        """Test distance computation."""
        obs = CircleObstacle("obs_1", center=[5.0, 5.0], radius=1.0, safety_margin=0.25)

        # Point outside
        point_outside = np.array([10.0, 10.0])
    dist = obs.distance_to(point_outside)
    assert dist > 0  # Positive = safe

        # Point inside
        point_inside = np.array([5.5, 5.0])
        dist = obs.distance_to(point_inside)
        assert dist < 0  # Negative = collision

        # Point on boundary
        point_boundary = np.array([6.0, 5.0])
        dist = obs.distance_to(point_boundary, include_margin=False)
        assert dist == pytest.approx(0.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        obs = CircleObstacle("obs_1", center=[5.0, 5.0], radius=1.0, safety_margin=0.25)

        config = obs.to_dict()
        assert config["id"] == "obs_1"
        assert config["type"] == "circle"
        assert config["radius"] == 1.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        config = {
            "id": "obs_1",
            "type": "circle",
            "center": [5.0, 5.0],
            "radius": 1.0,
            "safety_margin": 0.25,
        }

        obs = CircleObstacle.from_dict(config)
        assert obs.id == "obs_1"
        assert obs.radius == 1.0


class TestSphereObstacle:
    """Test SphereObstacle class."""

    def test_creation(self):
        """Test SphereObstacle creation."""
        obs = SphereObstacle("obs_1", center=[2.0, 2.0, -1.5], radius=0.5, safety_margin=0.2)

        assert obs.id == "obs_1"
        assert obs.radius == 0.5
        assert obs.safety_margin == 0.2
        assert obs.effective_radius == 0.7

    def test_creation_invalid_dimension(self):
        """Test that invalid center dimension raises error."""
        with pytest.raises(ValueError):
            SphereObstacle("obs_1", center=[1.0, 2.0], radius=1.0)

    def test_contains(self):
        """Test point containment checking."""
        obs = SphereObstacle("obs_1", center=[2.0, 2.0, -1.5], radius=0.5, safety_margin=0.2)

        # Point inside
        point_inside = np.array([2.0, 2.0, -1.0])
        assert obs.contains(point_inside)

        # Point outside
        point_outside = np.array([10.0, 10.0, 0.0])
        assert not obs.contains(point_outside)

        # Point with full quadrotor state
        x_quadrotor = np.zeros(13)
        x_quadrotor[:3] = [2.0, 2.0, -1.0]
        assert obs.contains(x_quadrotor)

    def test_distance_to(self):
        """Test distance computation."""
        obs = SphereObstacle("obs_1", center=[2.0, 2.0, -1.5], radius=0.5, safety_margin=0.2)

        # Point outside
        point_outside = np.array([5.0, 5.0, 0.0])
        dist = obs.distance_to(point_outside)
        assert dist > 0

        # Point inside
        point_inside = np.array([2.0, 2.0, -1.0])
        dist = obs.distance_to(point_inside)
        assert dist < 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        obs = SphereObstacle("obs_1", center=[2.0, 2.0, -1.5], radius=0.5, safety_margin=0.2)

        config = obs.to_dict()
        assert config["type"] == "sphere"
        assert len(config["center"]) == 3

    def test_from_dict(self):
        """Test creation from dictionary."""
        config = {
            "id": "obs_1",
            "type": "sphere",
            "center": [2.0, 2.0, -1.5],
            "radius": 0.5,
            "safety_margin": 0.2,
        }

        obs = SphereObstacle.from_dict(config)
        assert obs.id == "obs_1"
        assert obs.radius == 0.5


class TestObstacleFunctions:
    """Test obstacle utility functions."""

    def test_check_collision_free(self):
    """Test collision-free checking."""
    obstacles = [
        CircleObstacle("obs_1", [4.0, 3.0], 1.0, 0.25),
        CircleObstacle("obs_2", [8.0, 3.0], 1.0, 0.25),
    ]

    # Safe point
    safe_point = np.array([1.0, 1.0])
    assert check_collision_free(safe_point, obstacles)

    # Unsafe point
    unsafe_point = np.array([4.0, 3.0])
    assert not check_collision_free(unsafe_point, obstacles)

        # Empty obstacles list
        assert check_collision_free(safe_point, [])

    def test_minimum_distance_to_obstacles(self):
        """Test minimum distance computation."""
        obstacles = [
            CircleObstacle("obs_1", [4.0, 3.0], 1.0, 0.25),
            CircleObstacle("obs_2", [8.0, 3.0], 1.0, 0.25),
        ]

        # Point far from obstacles
        point_far = np.array([1.0, 1.0])
        min_dist = minimum_distance_to_obstacles(point_far, obstacles)
        assert min_dist > 0

        # Point close to obstacle
        point_close = np.array([4.5, 3.0])
        min_dist = minimum_distance_to_obstacles(point_close, obstacles)
        assert min_dist < 0  # Inside obstacle

        # Empty obstacles list
        min_dist = minimum_distance_to_obstacles(point_far, [])
        assert min_dist == float("inf")

    def test_create_obstacles_from_config(self):
        """Test obstacle creation from configuration."""
        # Unicycle obstacles
        config_uni = [
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
        ]

        obstacles = create_obstacles_from_config(config_uni, "unicycle")
        assert len(obstacles) == 2
        assert all(isinstance(obs, CircleObstacle) for obs in obstacles)

        # Quadrotor obstacles
        config_quad = [
            {
                "id": "obs_1",
                "type": "sphere",
                "center": [2.0, 2.0, -1.5],
                "radius": 0.5,
                "safety_margin": 0.2,
            }
        ]

        obstacles = create_obstacles_from_config(config_quad, "quadrotor")
        assert len(obstacles) == 1
        assert isinstance(obstacles[0], SphereObstacle)

        # Invalid type
        with pytest.raises(ValueError):
            create_obstacles_from_config(config_uni, "quadrotor")

        # Unknown obstacle type
        with pytest.raises(ValueError):
            create_obstacles_from_config([{"type": "unknown"}], "unicycle")


# ============================================================================
# CONFIG TESTS
# ============================================================================


class TestDDFSConfig:
    """Test DDFSConfig class."""

    def _create_test_config(self, system_type: str = "unicycle") -> dict:
        """Create a test configuration dictionary."""
        if system_type == "unicycle":
            return {
                "experiment": {
                    "name": "test_experiment",
                    "description": "Test experiment",
                    "output_dir": "test_results",
                },
                "system": {
                    "active": "unicycle",
                    "unicycle": {
                        "state_dim": 3,
                        "input_dim": 2,
                        "dt": 0.1,
                        "state_bounds": {
                            "x_min": 0.0,
                            "x_max": 12.0,
                            "y_min": 0.0,
                            "y_max": 8.0,
                        },
                        "input_bounds": {
                            "v_min": 0.0,
                            "v_max": 2.0,
                            "omega_max": 2.0,
                        },
                    },
                },
                "planning": {
                    "unicycle": {
                        "tf": 8.0,
                        "N": 80,
                        "x0": [0.0, 0.0, 0.0],
                        "xf": [12.0, 8.0, 0.0],
                    }
                },
                "environment": {
                    "unicycle": {
                        "workspace": {
                            "x_min": 0.0,
                            "x_max": 12.0,
                            "y_min": 0.0,
                            "y_max": 8.0,
                        },
                        "obstacles": [
                            {
                                "id": "obs_1",
                                "type": "circle",
                                "center": [4.0, 3.0],
                                "radius": 1.0,
                                "safety_margin": 0.25,
                            }
                        ],
                    }
                },
            }
        else:  # quadrotor
            return {
                "experiment": {
                    "name": "test_experiment",
                    "description": "Test experiment",
                    "output_dir": "test_results",
                },
                "system": {
                    "active": "quadrotor",
                    "quadrotor": {
                        "state_dim": 13,
                        "input_dim": 4,
                        "dt": 0.078,
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
                            "tau_max": 0.1,
                        },
                    },
                },
                "planning": {
                    "quadrotor": {
                        "tf": 8.0,
                        "N": 103,
                        "x0": [0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "xf": [8.0, 8.0, -2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    }
                },
                "environment": {
                    "quadrotor": {
                        "workspace": {
                            "x_min": 0.0,
                            "x_max": 8.0,
                            "y_min": 0.0,
                            "y_max": 8.0,
                            "z_min": -5.0,
                            "z_max": 0.5,
                        },
                        "obstacles": [
                            {
                                "id": "obs_1",
                                "type": "sphere",
                                "center": [2.0, 2.0, -1.5],
                                "radius": 0.5,
                                "safety_margin": 0.2,
                            }
                        ],
                    }
                },
            }

    def test_creation_unicycle(self, tmp_path):
        """Test DDFSConfig creation for unicycle."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        assert config.system_type == "unicycle"

    def test_creation_quadrotor(self, tmp_path):
        """Test DDFSConfig creation for quadrotor."""
        config_dict = self._create_test_config("quadrotor")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        assert config.system_type == "quadrotor"

    def test_file_not_found(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            DDFSConfig("nonexistent_file.yaml")

    def test_validation_missing_section(self, tmp_path):
        """Test validation with missing sections."""
        config_dict = {"experiment": {}, "system": {"active": "unicycle"}}  # Missing planning, environment
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        with pytest.raises(ValueError, match="Missing required config section"):
            DDFSConfig(config_file)

    def test_validation_invalid_system_type(self, tmp_path):
        """Test validation with invalid system type."""
        config_dict = self._create_test_config("unicycle")
        config_dict["system"]["active"] = "invalid_system"
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        with pytest.raises(ValueError, match="Invalid system type"):
            DDFSConfig(config_file)

    def test_get_system_config(self, tmp_path):
        """Test get_system_config method."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        system_config = config.get_system_config()

        assert system_config["state_dim"] == 3
        assert system_config["input_dim"] == 2

    def test_get_planning_params(self, tmp_path):
        """Test get_planning_params method."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        planning_params = config.get_planning_params()

        assert planning_params["N"] == 80
        assert planning_params["tf"] == 8.0

    def test_get_environment_config(self, tmp_path):
        """Test get_environment_config method."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        env_config = config.get_environment_config()

        assert "workspace" in env_config
        assert "obstacles" in env_config

    def test_get_constraints(self, tmp_path):
        """Test get_constraints method."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        constraints = config.get_constraints()

        assert isinstance(constraints, UnicycleConstraints)
        # Test caching
        constraints2 = config.get_constraints()
        assert constraints is constraints2

    def test_get_workspace(self, tmp_path):
        """Test get_workspace method."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        workspace = config.get_workspace()

        assert isinstance(workspace, Workspace2D)
        # Test caching
        workspace2 = config.get_workspace()
        assert workspace is workspace2

    def test_get_obstacles(self, tmp_path):
        """Test get_obstacles method."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        obstacles = config.get_obstacles()

        assert len(obstacles) == 1
        assert isinstance(obstacles[0], CircleObstacle)
        # Test caching
        obstacles2 = config.get_obstacles()
        assert obstacles is obstacles2

    def test_get_plant_mismatch_params(self, tmp_path):
        """Test get_plant_mismatch_params method."""
        config_dict = self._create_test_config("unicycle")
        config_dict["system"]["unicycle"]["plant_mismatch"] = {"velocity_scale": 0.95}
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        mismatch = config.get_plant_mismatch_params()

        assert mismatch["velocity_scale"] == 0.95

    def test_get_data_collection_params(self, tmp_path):
        """Test get_data_collection_params method."""
        config_dict = self._create_test_config("unicycle")
        config_dict["data_collection"] = {"M": 50, "excitation": {"type": "gaussian"}}
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        data_params = config.get_data_collection_params()

        assert data_params["M"] == 50

    def test_get_uncertainty_params(self, tmp_path):
        """Test get_uncertainty_params method."""
        config_dict = self._create_test_config("unicycle")
        config_dict["uncertainty"] = {"n_samples": 1000}
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        uncertainty_params = config.get_uncertainty_params()

        assert uncertainty_params["n_samples"] == 1000

    def test_get_synthesis_params(self, tmp_path):
        """Test get_synthesis_params method."""
        config_dict = self._create_test_config("unicycle")
        config_dict["synthesis"] = {"alpha": 0.95, "mu": 1.1}
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        synthesis_params = config.get_synthesis_params()

        assert synthesis_params["alpha"] == 0.95

    def test_get_experiment_info(self, tmp_path):
        """Test get_experiment_info method."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        experiment_info = config.get_experiment_info()

        assert experiment_info["name"] == "test_experiment"

    def test_get_output_dir(self, tmp_path):
        """Test get_output_dir method."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        output_dir = config.get_output_dir()

        assert output_dir == Path("test_results") / "unicycle"

    def test_summary(self, tmp_path):
        """Test summary method."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        summary = config.summary()

        assert "DDFS CONFIGURATION" in summary
        assert "unicycle" in summary.lower()

    def test_repr(self, tmp_path):
        """Test string representation."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = DDFSConfig(config_file)
        repr_str = repr(config)

        assert "DDFSConfig" in repr_str
        assert "unicycle" in repr_str

    def test_load_config_function(self, tmp_path):
        """Test load_config convenience function."""
        config_dict = self._create_test_config("unicycle")
        config_file = tmp_path / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_file)
        assert isinstance(config, DDFSConfig)
        assert config.system_type == "unicycle"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
