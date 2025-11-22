"""Minimal tests for ddfs.data_collection module."""

import numpy as np
import pytest

from ddfs.data_collection import (
    ExcitationSignalGenerator,
    HankelMatrixBuilder,
    SegmentedData,
    Trajectory,
    TrajectorySegmenter,
)


def test_trajectory_creation():
    """Test Trajectory creation and validation."""
    N, n, m = 10, 3, 2
    x = np.random.randn(N + 1, n)
    u = np.random.randn(N, m)
    eta = np.random.randn(N + 1, n)
    xi = np.random.randn(N, m)

    traj = Trajectory(x=x, u=u, eta=eta, xi=xi, trial_id=1, x0=x[0])

    assert traj.N == N
    assert traj.state_dim == n
    assert traj.input_dim == m


def test_excitation_generator():
    """Test excitation signal generation."""
    gen = ExcitationSignalGenerator(signal_type="gaussian", amplitude=0.1, seed=42)

    epsilon = gen.generate(N=100, m=2)

    assert epsilon.shape == (100, 2)
    assert np.abs(epsilon).max() < 1.0  # Reasonable amplitude


def test_trajectory_segmenter():
    """Test trajectory segmentation."""
    # Create dummy trajectories
    N, n, m = 100, 3, 2
    M = 5
    trajectories = []

    for trial_id in range(1, M + 1):
        x = np.random.randn(N + 1, n)
        u = np.random.randn(N, m)
        traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
        trajectories.append(traj)

    # Segment
    segmenter = TrajectorySegmenter(T=50, L=30)
    segmented = segmenter.segment(trajectories, verbose=False)

    assert segmented.num_segments == 2  # 100 timesteps, T=50 → 2 segments
    assert segmented.num_trials == M
    assert segmented.L == 30
    assert segmented.T == 50


def test_hankel_matrices():
    """Test Hankel matrix building."""
    # Create segmented data
    _, n, m, L, M = 30, 3, 2, 30, 5

    segment_trajs = []
    for trial_id in range(1, M + 1):
        x = np.random.randn(L + 1, n)
        u = np.random.randn(L, m)
        eta = x * 0.1
        xi = u * 0.1
        traj = Trajectory(x=x, u=u, eta=eta, xi=xi, trial_id=trial_id, x0=x[0])
        segment_trajs.append(traj)

    # Build Hankel matrices
    builder = HankelMatrixBuilder(verbose=False)
    matrices = builder.build_segment_matrices(
        segment_trajectories=segment_trajs,
        segment_idx=0,
        k_start=0,
        k_end=L - 1,
    )

    # Check dimensions
    assert matrices.H.shape == (n, L * M)
    assert matrices.H_plus.shape == (n, L * M)
    assert matrices.Xi.shape == (m, L * M)

    # Check informativity
    is_informative, actual_rank, required_rank = matrices.check_informativity()
    assert required_rank == n + m


def test_segmented_data_properties():
    """Test SegmentedData properties."""
    # Create dummy data
    _, n, m, L = 30, 3, 2, 30

    segment_trajs = []
    for trial_id in range(1, 6):
        x = np.random.randn(L + 1, n)
        u = np.random.randn(L, m)
        traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
        segment_trajs.append(traj)

    segmented = SegmentedData(
        segments=[segment_trajs],
        segment_indices=[0],
        k_starts=[0],
        k_ends=[29],
        T=30,
        L=30,
    )

    assert segmented.num_segments == 1
    assert segmented.num_trials == 5
    assert len(segmented.get_segment(0)) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
