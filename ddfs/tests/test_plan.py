"""Minimal tests for ddfs.planning module."""

import numpy as np
import pytest

from ddfs.planning import NominalTrajectory


def test_nominal_trajectory_creation():
    """Test NominalTrajectory creation and validation."""
    N = 10
    n, m = 3, 2

    x_nom = np.random.randn(N + 1, n)
    u_nom = np.random.randn(N, m)

    traj = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)

    assert traj.N == N
    assert traj.state_dim == n
    assert traj.input_dim == m
    assert traj.dt == 0.1


def test_nominal_trajectory_properties():
    """Test trajectory properties."""
    N = 20
    dt = 0.15
    x_nom = np.random.randn(N + 1, 3)
    u_nom = np.random.randn(N, 2)

    traj = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=dt)

    # Check final time
    assert traj.tf == pytest.approx(N * dt)

    # Check time vector
    t = traj.get_time_vector()
    assert len(t) == N + 1
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(traj.tf)


def test_nominal_trajectory_evaluation():
    """Test trajectory evaluation at specific timestep."""
    N = 10
    x_nom = np.random.randn(N + 1, 3)
    u_nom = np.random.randn(N, 2)

    traj = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)

    # Evaluate at timestep 5
    x_5, u_5 = traj.evaluate_at(5)

    assert np.allclose(x_5, x_nom[5])
    assert np.allclose(u_5, u_nom[5])


def test_nominal_trajectory_invalid():
    """Test that invalid dimensions raise errors."""
    N = 10

    # Wrong x_nom length
    with pytest.raises(ValueError):
        x_nom = np.random.randn(N, 3)  # Should be N+1
        u_nom = np.random.randn(N, 2)
        NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)

    # Wrong u_nom length
    with pytest.raises(ValueError):
        x_nom = np.random.randn(N + 1, 3)
        u_nom = np.random.randn(N - 1, 2)  # Should be N
        NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)

    # Invalid timestep in evaluate_at
    x_nom = np.random.randn(N + 1, 3)
    u_nom = np.random.randn(N, 2)
    traj = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)

    with pytest.raises(ValueError):
        traj.evaluate_at(N)  # Out of range


def test_nominal_trajectory_save_load(tmp_path):
    """Test saving and loading trajectories."""
    N = 10
    x_nom = np.random.randn(N + 1, 3)
    u_nom = np.random.randn(N, 2)

    traj = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)

    # Save
    path = tmp_path / "test_traj.pkl"
    traj.save(path)

    # Load
    loaded = NominalTrajectory.load(path)

    assert loaded.N == traj.N
    assert loaded.dt == traj.dt
    assert np.allclose(loaded.x_nom, traj.x_nom)
    assert np.allclose(loaded.u_nom, traj.u_nom)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
