"""Comprehensive tests for ddfs.data_collection module."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddfs.data_collection import (
    DataCollector,
    ExcitationSignalGenerator,
    HankelMatrixBuilder,
    SegmentedData,
    SegmentHankelMatrices,
    Trajectory,
    TrajectorySegmenter,
)
from ddfs.models.base import PlantModel, TwinModel
from ddfs.planning.nominal_trajectory import NominalTrajectory

# ============================================================================
# TRAJECTORY TESTS
# ============================================================================


class TestTrajectory:
    """Test Trajectory class."""

    def test_creation(self):
        """Test Trajectory creation."""
    N, n, m = 10, 3, 2
    x = np.random.randn(N + 1, n)
    u = np.random.randn(N, m)
    eta = np.random.randn(N + 1, n)
    xi = np.random.randn(N, m)

    traj = Trajectory(x=x, u=u, eta=eta, xi=xi, trial_id=1, x0=x[0])

    assert traj.N == N
    assert traj.state_dim == n
    assert traj.input_dim == m
        assert traj.trial_id == 1

    def test_creation_invalid_dimensions(self):
        """Test that invalid dimensions raise errors."""
        N, n, m = 10, 3, 2

        # Wrong x length
        with pytest.raises(ValueError):
            x = np.random.randn(N, n)  # Should be N+1
            u = np.random.randn(N, m)
            Trajectory(x=x, u=u, eta=x, xi=u, trial_id=1, x0=x[0])

        # Wrong eta shape
        with pytest.raises(ValueError):
            x = np.random.randn(N + 1, n)
            u = np.random.randn(N, m)
            eta = np.random.randn(N, n)  # Wrong shape
            Trajectory(x=x, u=u, eta=eta, xi=u, trial_id=1, x0=x[0])

        # Wrong xi shape
        with pytest.raises(ValueError):
            x = np.random.randn(N + 1, n)
            u = np.random.randn(N, m)
            xi = np.random.randn(N + 1, m)  # Wrong shape
            Trajectory(x=x, u=u, eta=x, xi=xi, trial_id=1, x0=x[0])

    def test_properties(self):
        """Test Trajectory properties."""
        N, n, m = 20, 4, 3
        x = np.random.randn(N + 1, n)
        u = np.random.randn(N, m)
        traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=5, x0=x[0])

        assert traj.N == N
        assert traj.state_dim == n
        assert traj.input_dim == m

    def test_save_load(self, tmp_path):
        """Test saving and loading trajectories."""
        N, n, m = 10, 3, 2
        x = np.random.randn(N + 1, n)
        u = np.random.randn(N, m)
        traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=1, x0=x[0])

        # Save
        path = tmp_path / "trajectory.pkl"
        traj.save(path)

        # Load
        loaded = Trajectory.load(path)

        assert loaded.N == traj.N
        assert loaded.state_dim == traj.state_dim
        assert loaded.trial_id == traj.trial_id
        assert np.allclose(loaded.x, traj.x)
        assert np.allclose(loaded.u, traj.u)

    def test_repr(self):
        """Test string representation."""
        N, n, m = 10, 3, 2
        x = np.random.randn(N + 1, n)
        u = np.random.randn(N, m)
        traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=1, x0=x[0])

        repr_str = repr(traj)
        assert "Trajectory" in repr_str
        assert str(traj.trial_id) in repr_str


# ============================================================================
# EXCITATION SIGNAL GENERATOR TESTS
# ============================================================================


class TestExcitationSignalGenerator:
    """Test ExcitationSignalGenerator class."""

    def test_creation(self):
        """Test ExcitationSignalGenerator creation."""
        gen = ExcitationSignalGenerator(signal_type="gaussian", amplitude=0.1, seed=42)

        assert gen.signal_type == "gaussian"
        assert gen.amplitude == 0.1
        assert gen.seed == 42

    def test_gaussian(self):
        """Test Gaussian excitation signal."""
    gen = ExcitationSignalGenerator(signal_type="gaussian", amplitude=0.1, seed=42)

    epsilon = gen.generate(N=100, m=2)

    assert epsilon.shape == (100, 2)
    assert np.abs(epsilon).max() < 1.0  # Reasonable amplitude
        # Check amplitude scaling
        assert np.std(epsilon) < 0.2  # Should be around 0.1

    def test_chirp(self):
        """Test chirp excitation signal."""
        gen = ExcitationSignalGenerator(signal_type="chirp", amplitude=0.2, seed=None)

        epsilon = gen.generate(N=100, m=3)

        assert epsilon.shape == (100, 3)
        assert np.abs(epsilon).max() <= 0.2  # Bounded by amplitude
        # Chirp should have frequency sweep
        assert np.any(epsilon != 0)

    def test_multisine(self):
        """Test multisine excitation signal."""
        gen = ExcitationSignalGenerator(signal_type="multisine", amplitude=0.15, seed=123)

        epsilon = gen.generate(N=200, m=2)

        assert epsilon.shape == (200, 2)
        assert np.abs(epsilon).max() <= 0.15 * 2  # Amplitude scaling
        # Multisine should be periodic
        assert np.any(epsilon != 0)

    def test_prbs(self):
        """Test PRBS excitation signal."""
        gen = ExcitationSignalGenerator(signal_type="prbs", amplitude=0.1, seed=42)

        epsilon = gen.generate(N=100, m=2)

        assert epsilon.shape == (100, 2)
        # PRBS should only have values ±amplitude
        assert np.all(np.isin(epsilon, [-0.1, 0.1]))

    def test_unknown_signal_type(self):
        """Test that unknown signal type raises error."""
        gen = ExcitationSignalGenerator(signal_type="unknown", amplitude=0.1)

        with pytest.raises(ValueError, match="Unknown signal type"):
            gen.generate(N=100, m=2)

    def test_seed_reproducibility(self):
        """Test that seed ensures reproducibility."""
        # Set seed before each generation
        np.random.seed(42)
        gen1 = ExcitationSignalGenerator(signal_type="gaussian", amplitude=0.1, seed=42)
        epsilon1 = gen1.generate(N=100, m=2)

        np.random.seed(42)
        gen2 = ExcitationSignalGenerator(signal_type="gaussian", amplitude=0.1, seed=42)
        epsilon2 = gen2.generate(N=100, m=2)

        assert np.allclose(epsilon1, epsilon2)

    def test_defaults(self):
        """Test default parameters."""
        gen = ExcitationSignalGenerator()

        assert gen.signal_type == "gaussian"
        assert gen.amplitude == 0.1
        assert gen.seed is None

    def test_repr(self):
        """Test string representation."""
        gen = ExcitationSignalGenerator(signal_type="chirp", amplitude=0.2, seed=42)
        repr_str = repr(gen)

        assert "ExcitationSignalGenerator" in repr_str
        assert "chirp" in repr_str


# ============================================================================
# DATA COLLECTOR TESTS
# ============================================================================


class MockTwin(TwinModel):
    """Mock twin model for testing."""

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def input_dim(self) -> int:
        return 2

    def _dynamics(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Simple dynamics: x_dot = u."""
        return jnp.array([u[0] * jnp.cos(x[2]), u[0] * jnp.sin(x[2]), u[1]])


class MockPlant(PlantModel):
    """Mock plant model for testing."""

    def __init__(self, twin: TwinModel):
        super().__init__(twin)
        self.twin = twin

    @property
    def state_dim(self) -> int:
        return self.twin.state_dim

    @property
    def input_dim(self) -> int:
        return self.twin.input_dim

    def _apply_mismatch(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Apply small mismatch."""
        f_twin = self.twin._dynamics(x, u)
        # Add small mismatch
        mismatch = 0.05 * jnp.array([1.0, 1.0, 0.1])
        return f_twin + mismatch


class TestDataCollector:
    """Test DataCollector class."""

    def _create_nominal_trajectory(self, N: int = 50) -> NominalTrajectory:
        """Create a simple nominal trajectory for testing."""
        n, m = 3, 2
        dt = 0.1

        x_nom = np.zeros((N + 1, n))
        u_nom = np.ones((N, m)) * 0.5

        # Simple straight line trajectory
        for k in range(N + 1):
            x_nom[k] = [k * dt * 0.5, 0.0, 0.0]

        return NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=dt)

    def test_creation(self):
        """Test DataCollector creation."""
        twin = MockTwin(dt=0.1)
        plant = MockPlant(twin)
        nominal = self._create_nominal_trajectory()

        config = {"M": 5, "excitation": {"type": "gaussian", "amplitude": 0.1}}

        collector = DataCollector(plant, nominal, config)

        assert collector.M == 5
        assert collector.excitation_type == "gaussian"
        assert collector.excitation_amplitude == 0.1

    def test_collect_single_trial(self):
        """Test single trial collection."""
        twin = MockTwin(dt=0.1)
        plant = MockPlant(twin)
        nominal = self._create_nominal_trajectory(N=20)

        config = {"M": 1, "excitation": {"type": "gaussian", "amplitude": 0.05, "seed": 42}}

        collector = DataCollector(plant, nominal, config)
        traj = collector.collect_single_trial(trial_id=1, verbose=False)

        assert traj.N == nominal.N
        assert traj.state_dim == nominal.state_dim
        assert traj.input_dim == nominal.input_dim
        assert traj.trial_id == 1
        assert traj.x.shape == (nominal.N + 1, nominal.state_dim)
        assert traj.u.shape == (nominal.N, nominal.input_dim)

    def test_collect_trials(self):
        """Test multiple trial collection."""
        twin = MockTwin(dt=0.1)
        plant = MockPlant(twin)
        nominal = self._create_nominal_trajectory(N=20)

        config = {"M": 3, "excitation": {"type": "gaussian", "amplitude": 0.05, "seed": 42}}

        collector = DataCollector(plant, nominal, config)
        trajectories = collector.collect_trials(verbose=False)

        assert len(trajectories) == 3
        for i, traj in enumerate(trajectories):
            assert traj.trial_id == i + 1
            assert traj.N == nominal.N

    def test_default_config(self):
        """Test default configuration values."""
        twin = MockTwin(dt=0.1)
        plant = MockPlant(twin)
        nominal = self._create_nominal_trajectory()

        collector = DataCollector(plant, nominal, {})

        assert collector.M == 50  # Default
        assert collector.excitation_type == "gaussian"  # Default
        assert collector.excitation_amplitude == 0.1  # Default

    def test_repr(self):
        """Test string representation."""
        twin = MockTwin(dt=0.1)
        plant = MockPlant(twin)
        nominal = self._create_nominal_trajectory()

        collector = DataCollector(plant, nominal, {"M": 10})
        repr_str = repr(collector)

        assert "DataCollector" in repr_str
        assert "10" in repr_str


# ============================================================================
# SEGMENTED DATA TESTS
# ============================================================================


class TestSegmentedData:
    """Test SegmentedData class."""

    def test_creation(self):
        """Test SegmentedData creation."""
        n, m, L = 3, 2, 30
        M = 5

        segment_trajs = []
        for trial_id in range(1, M + 1):
            x = np.random.randn(L + 1, n)
            u = np.random.randn(L, m)
            traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
            segment_trajs.append(traj)

        segmented = SegmentedData(
            segments=[segment_trajs],
            segment_indices=[0],
            k_starts=[0],
            k_ends=[L - 1],
            T=30,
            L=30,
        )

        assert segmented.num_segments == 1
        assert segmented.num_trials == M
        assert segmented.T == 30
        assert segmented.L == 30

    def test_properties(self):
        """Test SegmentedData properties."""
        n, m, L = 3, 2, 30
        M = 5

        segment_trajs = []
        for trial_id in range(1, M + 1):
            x = np.random.randn(L + 1, n)
            u = np.random.randn(L, m)
            traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
            segment_trajs.append(traj)

        segmented = SegmentedData(
            segments=[segment_trajs, segment_trajs],  # 2 segments
            segment_indices=[0, 1],
            k_starts=[0, 30],
            k_ends=[29, 59],
            T=30,
            L=30,
        )

        assert segmented.num_segments == 2
        assert segmented.num_trials == M

    def test_get_segment(self):
        """Test get_segment method."""
        n, m, L = 3, 2, 30
        M = 5

        segment_trajs = []
        for trial_id in range(1, M + 1):
            x = np.random.randn(L + 1, n)
            u = np.random.randn(L, m)
            traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
            segment_trajs.append(traj)

        segmented = SegmentedData(
            segments=[segment_trajs],
            segment_indices=[0],
            k_starts=[0],
            k_ends=[L - 1],
            T=30,
            L=30,
        )

        seg = segmented.get_segment(0)
        assert len(seg) == M

        # Invalid index
        with pytest.raises(IndexError):
            segmented.get_segment(1)

    def test_empty_segments(self):
        """Test SegmentedData with empty segments."""
        segmented = SegmentedData(
            segments=[],
            segment_indices=[],
            k_starts=[],
            k_ends=[],
            T=30,
            L=30,
        )

        assert segmented.num_segments == 0
        assert segmented.num_trials == 0

    def test_save_load(self, tmp_path):
        """Test saving and loading SegmentedData."""
        n, m, L = 3, 2, 30
        M = 5

        segment_trajs = []
        for trial_id in range(1, M + 1):
            x = np.random.randn(L + 1, n)
            u = np.random.randn(L, m)
            traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
            segment_trajs.append(traj)

        segmented = SegmentedData(
            segments=[segment_trajs],
            segment_indices=[0],
            k_starts=[0],
            k_ends=[L - 1],
            T=30,
            L=30,
        )

        # Save
        path = tmp_path / "segmented.pkl"
        segmented.save(path)

        # Load
        loaded = SegmentedData.load(path)

        assert loaded.num_segments == segmented.num_segments
        assert loaded.num_trials == segmented.num_trials
        assert loaded.T == segmented.T
        assert loaded.L == segmented.L

    def test_repr(self):
        """Test string representation."""
        n, m, L = 3, 2, 30
        M = 5

        segment_trajs = []
        for trial_id in range(1, M + 1):
            x = np.random.randn(L + 1, n)
            u = np.random.randn(L, m)
            traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
            segment_trajs.append(traj)

        segmented = SegmentedData(
            segments=[segment_trajs],
            segment_indices=[0],
            k_starts=[0],
            k_ends=[L - 1],
            T=30,
            L=30,
        )

        repr_str = repr(segmented)
        assert "SegmentedData" in repr_str


# ============================================================================
# TRAJECTORY SEGMENTER TESTS
# ============================================================================


class TestTrajectorySegmenter:
    """Test TrajectorySegmenter class."""

    def test_creation(self):
        """Test TrajectorySegmenter creation."""
        segmenter = TrajectorySegmenter(T=50, L=30)

        assert segmenter.T == 50
        assert segmenter.L == 30
        assert segmenter.overlap == 0  # L < T, so gap
        assert segmenter.gap == 20

    def test_creation_overlapping(self):
        """Test segmenter with overlapping segments."""
        segmenter = TrajectorySegmenter(T=30, L=50)

        assert segmenter.T == 30
        assert segmenter.L == 50
        assert segmenter.overlap == 20
        assert segmenter.gap == 0

    def test_creation_non_overlapping(self):
        """Test segmenter with non-overlapping segments."""
        segmenter = TrajectorySegmenter(T=50, L=50)

        assert segmenter.T == 50
        assert segmenter.L == 50
        assert segmenter.overlap == 0
        assert segmenter.gap == 0

    def test_creation_invalid(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            TrajectorySegmenter(T=50, L=0)

        with pytest.raises(ValueError):
            TrajectorySegmenter(T=0, L=30)

    def test_compute_segments(self):
        """Test compute_segments method."""
        segmenter = TrajectorySegmenter(T=50, L=30)

        segments = segmenter.compute_segments(N=100)

        assert len(segments) == 2
        assert segments[0] == (0, 0, 29)
        assert segments[1] == (1, 50, 79)

    def test_compute_segments_overlapping(self):
        """Test compute_segments with overlapping segments."""
        segmenter = TrajectorySegmenter(T=30, L=50)

        segments = segmenter.compute_segments(N=100)

        # With T=30, L=50, N=100:
        # Segment 0: k=0, end=49 (0+50-1)
        # Segment 1: k=30, end=79 (30+50-1)
        # Segment 2: k=60, end=109 (60+50-1) but 60+50=110 > 100, so not included
        assert len(segments) == 2
        assert segments[0] == (0, 0, 49)
        assert segments[1] == (1, 30, 79)

    def test_compute_segments_no_segments(self):
        """Test compute_segments when no segments are possible."""
        segmenter = TrajectorySegmenter(T=50, L=30)

        segments = segmenter.compute_segments(N=20)  # N < L

        assert len(segments) == 0

    def test_segment(self):
    """Test trajectory segmentation."""
    N, n, m = 100, 3, 2
    M = 5
    trajectories = []

    for trial_id in range(1, M + 1):
        x = np.random.randn(N + 1, n)
        u = np.random.randn(N, m)
        traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
        trajectories.append(traj)

    segmenter = TrajectorySegmenter(T=50, L=30)
    segmented = segmenter.segment(trajectories, verbose=False)

        assert segmented.num_segments == 2
    assert segmented.num_trials == M
    assert segmented.L == 30
    assert segmented.T == 50

        # Check segment lengths
        for seg in segmented.segments:
            for traj in seg:
                assert traj.N == 30

    def test_segment_empty_trajectories(self):
        """Test segmentation with empty trajectory list."""
        segmenter = TrajectorySegmenter(T=50, L=30)

        with pytest.raises(ValueError, match="No trajectories"):
            segmenter.segment([], verbose=False)

    def test_segment_inconsistent_lengths(self):
        """Test segmentation with inconsistent trajectory lengths."""
        N, n, m = 100, 3, 2
        trajectories = []

        # Create trajectories with different lengths
        for trial_id in range(1, 3):
            N_traj = N if trial_id == 1 else N + 10
            x = np.random.randn(N_traj + 1, n)
            u = np.random.randn(N_traj, m)
            traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
            trajectories.append(traj)

        segmenter = TrajectorySegmenter(T=50, L=30)

        with pytest.raises(ValueError, match="has length"):
            segmenter.segment(trajectories, verbose=False)

    def test_get_coverage(self):
        """Test coverage computation."""
        segmenter = TrajectorySegmenter(T=50, L=30)

        covered, ratio = segmenter.get_coverage(N=100)

        assert covered > 0
        assert 0.0 <= ratio <= 1.0

    def test_get_coverage_no_segments(self):
        """Test coverage with no segments."""
        segmenter = TrajectorySegmenter(T=50, L=30)

        covered, ratio = segmenter.get_coverage(N=20)

        assert covered == 0
        assert ratio == 0.0

    def test_repr(self):
        """Test string representation."""
        segmenter = TrajectorySegmenter(T=50, L=30)
        repr_str = repr(segmenter)

        assert "TrajectorySegmenter" in repr_str
        assert "50" in repr_str
        assert "30" in repr_str


# ============================================================================
# HANKEL MATRICES TESTS
# ============================================================================


class TestSegmentHankelMatrices:
    """Test SegmentHankelMatrices class."""

    def test_creation(self):
        """Test SegmentHankelMatrices creation."""
        n, m, L, M = 3, 2, 30, 5

        H = np.random.randn(n, L * M)
        H_plus = np.random.randn(n, L * M)
        Xi = np.random.randn(m, L * M)

        matrices = SegmentHankelMatrices(
            segment_idx=0,
            H=H,
            H_plus=H_plus,
            Xi=Xi,
            k_start=0,
            k_end=L - 1,
            L=L,
            M=M,
        )

        assert matrices.segment_idx == 0
        assert matrices.state_dim == n
        assert matrices.input_dim == m
        assert matrices.L == L
        assert matrices.M == M

    def test_creation_invalid_dimensions(self):
        """Test that invalid dimensions raise errors."""
        n, m, L, M = 3, 2, 30, 5

        H = np.random.randn(n, L * M)
        H_plus = np.random.randn(n, L * M)
        Xi = np.random.randn(m, L * M)

        # Wrong H_plus shape
        with pytest.raises(ValueError):
            H_plus_bad = np.random.randn(n, L * M + 1)
            SegmentHankelMatrices(
                segment_idx=0,
                H=H,
                H_plus=H_plus_bad,
                Xi=Xi,
                k_start=0,
                k_end=L - 1,
                L=L,
                M=M,
            )

        # Wrong column count
        with pytest.raises(ValueError):
            H_bad = np.random.randn(n, L * M + 1)
            SegmentHankelMatrices(
                segment_idx=0,
                H=H_bad,
                H_plus=H_plus,
                Xi=Xi,
                k_start=0,
                k_end=L - 1,
                L=L,
                M=M,
            )

    def test_check_informativity(self):
        """Test informativity checking."""
        n, m, L, M = 3, 2, 30, 5

        # Create informative data (full rank)
        H = np.random.randn(n, L * M)
        H_plus = np.random.randn(n, L * M)
        Xi = np.random.randn(m, L * M)

        matrices = SegmentHankelMatrices(
            segment_idx=0,
            H=H,
            H_plus=H_plus,
            Xi=Xi,
            k_start=0,
            k_end=L - 1,
            L=L,
            M=M,
        )

        is_informative, actual_rank, required_rank = matrices.check_informativity()

        assert required_rank == n + m
        assert actual_rank >= 0
        # With random data, should likely be informative if L*M >> n+m
        if n + m <= L * M:
            assert actual_rank <= min(n + m, L * M)

    def test_compute_condition_number(self):
        """Test condition number computation."""
        n, m, L, M = 3, 2, 30, 5

        H = np.random.randn(n, L * M)
        H_plus = np.random.randn(n, L * M)
        Xi = np.random.randn(m, L * M)

        matrices = SegmentHankelMatrices(
            segment_idx=0,
            H=H,
            H_plus=H_plus,
            Xi=Xi,
            k_start=0,
            k_end=L - 1,
            L=L,
            M=M,
        )

        cond = matrices.compute_condition_number()
        assert cond > 0
        assert np.isfinite(cond)

    def test_compute_singular_values(self):
        """Test singular value computation."""
        n, m, L, M = 3, 2, 30, 5

        H = np.random.randn(n, L * M)
        H_plus = np.random.randn(n, L * M)
        Xi = np.random.randn(m, L * M)

        matrices = SegmentHankelMatrices(
            segment_idx=0,
            H=H,
            H_plus=H_plus,
            Xi=Xi,
            k_start=0,
            k_end=L - 1,
            L=L,
            M=M,
        )

        sigma = matrices.compute_singular_values()
        expected_len = min(n + m, L * M)
        assert len(sigma) == expected_len
        assert np.all(sigma >= 0)
        assert np.all(np.diff(sigma) <= 0)  # Descending order (noqa: YODA)

    def test_compute_minimum_singular_value(self):
        """Test minimum singular value computation."""
        n, m, L, M = 3, 2, 30, 5

        H = np.random.randn(n, L * M)
        H_plus = np.random.randn(n, L * M)
        Xi = np.random.randn(m, L * M)

        matrices = SegmentHankelMatrices(
            segment_idx=0,
            H=H,
            H_plus=H_plus,
            Xi=Xi,
            k_start=0,
            k_end=L - 1,
            L=L,
            M=M,
        )

        sigma_min = matrices.compute_minimum_singular_value()
        assert sigma_min >= 0
        assert np.isfinite(sigma_min)

    def test_get_data_matrix(self):
        """Test get_data_matrix method."""
        n, m, L, M = 3, 2, 30, 5

        H = np.random.randn(n, L * M)
        H_plus = np.random.randn(n, L * M)
        Xi = np.random.randn(m, L * M)

        matrices = SegmentHankelMatrices(
            segment_idx=0,
            H=H,
            H_plus=H_plus,
            Xi=Xi,
            k_start=0,
            k_end=L - 1,
            L=L,
            M=M,
        )

        data_matrix = matrices.get_data_matrix()
        assert data_matrix.shape == (n + m, L * M)
        assert np.allclose(data_matrix[:n], H)
        assert np.allclose(data_matrix[n:], Xi)

    def test_save_load(self, tmp_path):
        """Test saving and loading matrices."""
        n, m, L, M = 3, 2, 30, 5

        H = np.random.randn(n, L * M)
        H_plus = np.random.randn(n, L * M)
        Xi = np.random.randn(m, L * M)

        matrices = SegmentHankelMatrices(
            segment_idx=0,
            H=H,
            H_plus=H_plus,
            Xi=Xi,
            k_start=0,
            k_end=L - 1,
            L=L,
            M=M,
        )

        # Save
        path = tmp_path / "hankel.pkl"
        matrices.save(path)

        # Load
        loaded = SegmentHankelMatrices.load(path)

        assert loaded.segment_idx == matrices.segment_idx
        assert loaded.L == matrices.L
        assert loaded.M == matrices.M
        assert np.allclose(loaded.H, matrices.H)

    def test_repr(self):
        """Test string representation."""
        n, m, L, M = 3, 2, 30, 5

        H = np.random.randn(n, L * M)
        H_plus = np.random.randn(n, L * M)
        Xi = np.random.randn(m, L * M)

        matrices = SegmentHankelMatrices(
            segment_idx=0,
            H=H,
            H_plus=H_plus,
            Xi=Xi,
            k_start=0,
            k_end=L - 1,
            L=L,
            M=M,
        )

        repr_str = repr(matrices)
        assert "SegmentHankelMatrices" in repr_str


# ============================================================================
# HANKEL MATRIX BUILDER TESTS
# ============================================================================


class TestHankelMatrixBuilder:
    """Test HankelMatrixBuilder class."""

    def test_creation(self):
        """Test HankelMatrixBuilder creation."""
        builder = HankelMatrixBuilder(verbose=False)

        assert builder.verbose is False

    def test_build_segment_matrices(self):
        """Test building matrices for one segment."""
        n, m, L, M = 3, 2, 30, 5

    segment_trajs = []
    for trial_id in range(1, M + 1):
        x = np.random.randn(L + 1, n)
        u = np.random.randn(L, m)
        eta = x * 0.1
        xi = u * 0.1
        traj = Trajectory(x=x, u=u, eta=eta, xi=xi, trial_id=trial_id, x0=x[0])
        segment_trajs.append(traj)

    builder = HankelMatrixBuilder(verbose=False)
    matrices = builder.build_segment_matrices(
        segment_trajectories=segment_trajs,
        segment_idx=0,
        k_start=0,
        k_end=L - 1,
    )

    assert matrices.H.shape == (n, L * M)
    assert matrices.H_plus.shape == (n, L * M)
    assert matrices.Xi.shape == (m, L * M)
        assert matrices.segment_idx == 0
        assert matrices.L == L
        assert matrices.M == M

    def test_build_segment_matrices_empty(self):
        """Test building matrices with empty segment."""
        builder = HankelMatrixBuilder(verbose=False)

        with pytest.raises(ValueError, match="No trajectories"):
            builder.build_segment_matrices(
                segment_trajectories=[],
                segment_idx=0,
                k_start=0,
                k_end=29,
            )

    def test_build_segment_matrices_inconsistent_dimensions(self):
        """Test building matrices with inconsistent dimensions."""
        n, m, L, M = 3, 2, 30, 5

    segment_trajs = []
        for trial_id in range(1, M + 1):
            # Mix dimensions
            n_traj = n if trial_id == 1 else n + 1
            x = np.random.randn(L + 1, n_traj)
        u = np.random.randn(L, m)
        traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
        segment_trajs.append(traj)

        builder = HankelMatrixBuilder(verbose=False)

        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            builder.build_segment_matrices(
                segment_trajectories=segment_trajs,
                segment_idx=0,
                k_start=0,
                k_end=L - 1,
            )

    def test_build_all_segments(self):
        """Test building matrices for all segments."""
        N, n, m = 100, 3, 2
        M = 5

        # Create trajectories
        trajectories = []
        for trial_id in range(1, M + 1):
            x = np.random.randn(N + 1, n)
            u = np.random.randn(N, m)
            traj = Trajectory(x=x, u=u, eta=x * 0.1, xi=u * 0.1, trial_id=trial_id, x0=x[0])
            trajectories.append(traj)

        # Segment
        segmenter = TrajectorySegmenter(T=50, L=30)
        segmented = segmenter.segment(trajectories, verbose=False)

        # Build Hankel matrices
        builder = HankelMatrixBuilder(verbose=False)
        all_matrices = builder.build_all_segments(segmented)

        assert len(all_matrices) == segmented.num_segments
        for matrices in all_matrices:
            assert matrices.H.shape == (n, segmented.L * M)
            assert matrices.H_plus.shape == (n, segmented.L * M)
            assert matrices.Xi.shape == (m, segmented.L * M)

    def test_repr(self):
        """Test string representation."""
        builder = HankelMatrixBuilder(verbose=True)
        repr_str = repr(builder)

        assert "HankelMatrixBuilder" in repr_str
        assert "True" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
