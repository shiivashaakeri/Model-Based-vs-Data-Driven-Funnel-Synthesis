"""Minimal tests for ddfs.core module."""

import numpy as np
import pytest

from ddfs.core import (
    CircleObstacle,
    QuadrotorConstraints,
    UnicycleConstraints,
    Workspace2D,
    Workspace3D,
    check_collision_free,
)


def test_unicycle_constraints():
    """Test UnicycleConstraints creation and checking."""
    constraints = UnicycleConstraints(
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=8.0,
        v_min=0.0,
        v_max=2.0,
        omega_max=2.0,
    )

    # Valid state
    x = np.array([5.0, 4.0, 0.5])
    assert constraints.check_state(x)

    # Valid input
    u = np.array([1.0, 0.5])
    assert constraints.check_input(u)

    # Clip invalid input
    u_bad = np.array([3.0, 5.0])
    u_clipped = constraints.clip_input(u_bad)
    assert constraints.check_input(u_clipped)


def test_quadrotor_constraints():
    """Test QuadrotorConstraints creation and checking."""
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

    # Valid state (simplified)
    x = np.zeros(13)
    x[:3] = [4.0, 4.0, -2.0]  # Position
    x[6] = 1.0  # Quaternion w
    assert constraints.check_state(x)

    # Valid input
    u = np.array([0.5, 0.0, 0.0, 0.0])
    assert constraints.check_input(u)


def test_workspace_2d():
    """Test Workspace2D."""
    workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)

    # Check containment
    x_inside = np.array([6.0, 4.0, 0.0])
    assert workspace.contains(x_inside)

    x_outside = np.array([15.0, 4.0, 0.0])
    assert not workspace.contains(x_outside)

    # Sample random point
    point = workspace.sample_random_point()
    assert len(point) == 2
    assert workspace.contains(np.append(point, 0.0))


def test_workspace_3d():
    """Test Workspace3D."""
    workspace = Workspace3D(
        x_min=0.0,
        x_max=8.0,
        y_min=0.0,
        y_max=8.0,
        z_min=-5.0,
        z_max=0.5,
    )

    # Check containment
    x_inside = np.zeros(13)
    x_inside[:3] = [4.0, 4.0, -2.0]
    assert workspace.contains(x_inside)

    x_outside = np.zeros(13)
    x_outside[:3] = [10.0, 4.0, -2.0]
    assert not workspace.contains(x_outside)


def test_circle_obstacle():
    """Test CircleObstacle."""
    obs = CircleObstacle("obs_1", center=[5.0, 5.0], radius=1.0, safety_margin=0.25)

    # Point inside
    point_inside = np.array([5.5, 5.0])
    assert obs.contains(point_inside)

    # Point outside
    point_outside = np.array([10.0, 10.0])
    assert not obs.contains(point_outside)

    # Distance
    dist = obs.distance_to(point_outside)
    assert dist > 0  # Positive = safe


def test_collision_checking():
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
