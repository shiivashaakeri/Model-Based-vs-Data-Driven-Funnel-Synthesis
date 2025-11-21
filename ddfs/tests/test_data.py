"""
Unit tests for Phase 4: Data Collection

Tests for:
- collector.py: OfflineDataCollector
- segmenter.py: TrajectorySegmenter
- hankel.py: HankelMatrixBuilder
- informativity.py: InformativityChecker
"""

import numpy as np
import pytest

from ddfs.data.collector import OfflineDataCollector
from ddfs.data.hankel import HankelMatrixBuilder
from ddfs.data.informativity import InformativityChecker
from ddfs.data.segmenter import TrajectorySegmenter
from ddfs.models.plant import PlantModel
from ddfs.models.unicycle import UnicycleModel


class TestOfflineDataCollector:
    """Tests for OfflineDataCollector."""

    @pytest.fixture
    def setup_collector(self):
        """Create test setup with nominal trajectory and plant."""
        # Create simple nominal trajectory
        N = 50
        dt = 0.1
        n = 3
        m = 2

        x_nom = np.zeros((N + 1, n))
        u_nom = np.ones((N, m)) * 0.5

        # Simple forward motion
        for k in range(N):
            x_nom[k + 1] = x_nom[k] + dt * np.array([u_nom[k, 0], 0, u_nom[k, 1]])

        # Create plant
        twin = UnicycleModel(x0=x_nom[0], xf=x_nom[-1])
        plant = PlantModel(twin=twin)

        collector = OfflineDataCollector(
            plant=plant, nominal_x=x_nom, nominal_u=u_nom, dt=dt, excitation_magnitude=0.1, seed=42
        )

        return collector, x_nom, u_nom, N, dt, n, m

    def test_initialization(self, setup_collector):
        """Test collector initializes correctly."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        assert collector.N == N
        assert collector.n == n
        assert collector.m == m
        assert collector.dt == dt
        assert len(collector.trajectories) == 0

    def test_sample_initial_states_shape(self, setup_collector):
        """Test initial state sampling produces correct shape."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        n_samples = 10
        semi_axes = np.array([0.1, 0.1, 0.05])

        x0_samples = collector.sample_initial_states(n_samples, semi_axes)

        assert x0_samples.shape == (n_samples, n)

    def test_sample_initial_states_within_ellipsoid(self, setup_collector):
        """Test sampled states are within ellipsoid."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        n_samples = 100
        semi_axes = np.array([0.2, 0.2, 0.1])
        center = x_nom[0]

        x0_samples = collector.sample_initial_states(n_samples, semi_axes, center)

        # All samples should be within ellipsoid: (x-c)^T diag(1/a^2) (x-c) <= 1
        for x0 in x0_samples:
            diff = x0 - center
            scaled_diff = diff / semi_axes
            dist = np.dot(scaled_diff, scaled_diff)
            assert dist <= 1.0 + 1e-6  # Small tolerance for numerical errors

    def test_sample_initial_states_distribution(self, setup_collector):
        """Test samples are roughly uniformly distributed."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        n_samples = 1000
        semi_axes = np.array([0.2, 0.2, 0.1])
        center = np.zeros(n)

        x0_samples = collector.sample_initial_states(n_samples, semi_axes, center)

        # Mean should be close to center
        mean = np.mean(x0_samples, axis=0)
        np.testing.assert_allclose(mean, center, atol=0.05)

        # Check distribution in each dimension
        for i in range(n):
            # Should cover range roughly [-semi_axes[i], +semi_axes[i]]
            assert np.min(x0_samples[:, i]) < -0.5 * semi_axes[i]
            assert np.max(x0_samples[:, i]) > 0.5 * semi_axes[i]

    def test_generate_excitation_signal_shape(self, setup_collector):
        """Test excitation signal has correct shape."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        excitation = collector.generate_excitation_signal(N)

        assert excitation.shape == (N, m)

    def test_generate_excitation_signal_bounded(self, setup_collector):
        """Test excitation is bounded by magnitude."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        excitation = collector.generate_excitation_signal(N)

        # All values should be within [-magnitude, +magnitude]
        assert np.all(np.abs(excitation) <= collector.excitation_magnitude)

    def test_collect_single_trajectory(self, setup_collector):
        """Test collecting a single trajectory."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        x0 = x_nom[0] + np.array([0.1, 0.1, 0.05])
        traj_data = collector.collect_single_trajectory(x0, verbose=False)

        # Check all required keys present
        assert "x" in traj_data
        assert "u" in traj_data
        assert "eta" in traj_data
        assert "xi" in traj_data
        assert "eta_next" in traj_data
        assert "x0" in traj_data
        assert "excitation" in traj_data

        # Check shapes
        assert traj_data["x"].shape == (N + 1, n)
        assert traj_data["u"].shape == (N, m)
        assert traj_data["eta"].shape == (N + 1, n)
        assert traj_data["xi"].shape == (N, m)
        assert traj_data["eta_next"].shape == (N, n)

    def test_collect_single_trajectory_deviations(self, setup_collector):
        """Test deviation computations are correct."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        x0 = x_nom[0] + np.array([0.1, 0.1, 0.05])
        traj_data = collector.collect_single_trajectory(x0, verbose=False)

        # Initial deviation should match
        np.testing.assert_allclose(traj_data["eta"][0], x0 - x_nom[0], atol=1e-10)

        # Check consistency: eta[k] = x[k] - x_nom[k]
        for k in range(N + 1):
            expected_eta = traj_data["x"][k] - x_nom[k]
            np.testing.assert_allclose(traj_data["eta"][k], expected_eta, atol=1e-10)

        # Check consistency: xi[k] = u[k] - u_nom[k]
        for k in range(N):
            expected_xi = traj_data["u"][k] - u_nom[k]
            np.testing.assert_allclose(traj_data["xi"][k], expected_xi, atol=1e-8)

    def test_collect_dataset(self, setup_collector):
        """Test collecting full dataset."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        n_samples = 5
        semi_axes = np.array([0.1, 0.1, 0.05])

        dataset = collector.collect_dataset(n_samples, semi_axes, verbose=False)

        assert len(dataset) == n_samples
        assert len(collector.trajectories) == n_samples

        # Each trajectory should have proper structure
        for traj_data in dataset:
            assert traj_data["x"].shape == (N + 1, n)
            assert traj_data["u"].shape == (N, m)

    def test_get_statistics(self, setup_collector):
        """Test dataset statistics computation."""
        collector, x_nom, u_nom, N, dt, n, m = setup_collector

        # Collect some data first
        n_samples = 10
        semi_axes = np.array([0.1, 0.1, 0.05])
        collector.collect_dataset(n_samples, semi_axes, verbose=False)

        stats = collector.get_statistics()

        # Check all expected keys
        assert "n_trajectories" in stats
        assert "n_timesteps" in stats
        assert "n_datapoints" in stats
        assert "eta_mean" in stats
        assert "eta_std" in stats
        assert "eta_max" in stats
        assert "xi_mean" in stats
        assert "xi_std" in stats
        assert "xi_max" in stats

        # Check values are reasonable
        assert stats["n_trajectories"] == n_samples
        assert stats["n_timesteps"] == N
        assert stats["n_datapoints"] == n_samples * N
        assert stats["eta_max"] > 0
        assert stats["xi_max"] > 0


class TestTrajectorySegmenter:
    """Tests for TrajectorySegmenter."""

    def test_initialization(self):
        """Test segmenter initializes correctly."""
        N = 100
        segment_length = 20
        overlap = 0

        segmenter = TrajectorySegmenter(N, segment_length, overlap)

        assert segmenter.N == N
        assert segmenter.segment_length == segment_length
        assert segmenter.overlap == overlap
        assert segmenter.n_segments > 0

    def test_compute_segments_no_overlap(self):
        """Test segment computation without overlap."""
        N = 100
        segment_length = 20

        segmenter = TrajectorySegmenter(N, segment_length, overlap=0)

        segments = segmenter.get_segment_boundaries()

        # Should have 5 segments: [0,20), [20,40), [40,60), [60,80), [80,100)
        assert len(segments) == 5

        # Check each segment
        expected = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        for i, (start, end) in enumerate(segments):
            assert start == expected[i][0]
            assert end == expected[i][1]

    def test_compute_segments_with_overlap(self):
        """Test segment computation with overlap."""
        N = 100
        segment_length = 30
        overlap = 10

        segmenter = TrajectorySegmenter(N, segment_length, overlap)

        segments = segmenter.get_segment_boundaries()

        # With overlap, stride = 30 - 10 = 20
        # Segments: [0,30), [20,50), [40,70), [60,90), [80,100)
        assert len(segments) == 5

        # Check overlap
        for i in range(len(segments) - 1):
            _, end_i = segments[i]
            start_next, _ = segments[i + 1]
            # Overlap should be present
            assert end_i > start_next

    def test_segment_single_trajectory(self):
        """Test segmenting a single trajectory."""
        N = 50
        n = 3
        m = 2
        segment_length = 20

        # Create dummy trajectory
        traj_data = {
            "eta": np.random.randn(N + 1, n),
            "xi": np.random.randn(N, m),
            "eta_next": np.random.randn(N, n),
        }

        segmenter = TrajectorySegmenter(N, segment_length, overlap=0)
        segmented = segmenter.segment_single_trajectory(traj_data)

        # Should have 3 segments: [0,20), [20,40), [40,50)
        assert len(segmented) == 3

        # Check each segment
        for seg_data in segmented:
            assert "segment_idx" in seg_data
            assert "k_start" in seg_data
            assert "k_end" in seg_data
            assert "length" in seg_data
            assert "eta" in seg_data
            assert "xi" in seg_data
            assert "eta_next" in seg_data

    def test_segment_dataset(self):
        """Test segmenting full dataset."""
        N = 60
        n = 3
        m = 2
        n_traj = 10
        segment_length = 20

        # Create dummy dataset
        trajectories = []
        for _ in range(n_traj):
            traj_data = {
                "eta": np.random.randn(N + 1, n),
                "xi": np.random.randn(N, m),
                "eta_next": np.random.randn(N, n),
            }
            trajectories.append(traj_data)

        segmenter = TrajectorySegmenter(N, segment_length, overlap=0)
        segmented_dataset = segmenter.segment_dataset(trajectories, verbose=False)

        # Should have n_traj trajectories, each with 3 segments
        assert len(segmented_dataset) == n_traj
        assert all(len(traj_segs) == 3 for traj_segs in segmented_dataset)

    def test_get_segment_data_by_index(self):
        """Test extracting data for a specific segment."""
        N = 60
        n = 3
        m = 2
        n_traj = 5
        segment_length = 20

        # Create and segment dataset
        trajectories = []
        for _ in range(n_traj):
            traj_data = {
                "eta": np.random.randn(N + 1, n),
                "xi": np.random.randn(N, m),
                "eta_next": np.random.randn(N, n),
            }
            trajectories.append(traj_data)

        segmenter = TrajectorySegmenter(N, segment_length, overlap=0)
        segmented_dataset = segmenter.segment_dataset(trajectories, verbose=False)

        # Get segment 0 data
        seg_0_data = segmenter.get_segment_data_by_index(segmented_dataset, 0)

        assert len(seg_0_data) == n_traj
        assert all(seg["segment_idx"] == 0 for seg in seg_0_data)

    def test_aggregate_segment_data(self):
        """Test aggregating segment data."""
        N = 60
        n = 3
        m = 2
        n_traj = 5
        segment_length = 20

        # Create segment data
        segment_data = []
        for _ in range(n_traj):
            seg_data = {
                "length": segment_length,
                "eta": np.random.randn(segment_length + 1, n),
                "xi": np.random.randn(segment_length, m),
                "eta_next": np.random.randn(segment_length, n),
            }
            segment_data.append(seg_data)

        segmenter = TrajectorySegmenter(N, segment_length, overlap=0)
        aggregated = segmenter.aggregate_segment_data(segment_data)

        # Check shapes
        assert aggregated["eta_all"].shape == (segment_length + 1, n * n_traj)
        assert aggregated["xi_all"].shape == (segment_length, m * n_traj)
        assert aggregated["eta_next_all"].shape == (segment_length, n * n_traj)
        assert aggregated["n_trajectories"] == n_traj
        assert aggregated["length"] == segment_length

    def test_validate_segmentation(self):
        """Test segmentation validation."""
        N = 100
        segment_length = 25

        segmenter = TrajectorySegmenter(N, segment_length, overlap=0)

        valid, message = segmenter.validate_segmentation()

        # Should be valid (covers entire trajectory)
        assert valid
        assert "valid" in message.lower()


class TestHankelMatrixBuilder:
    """Tests for HankelMatrixBuilder."""

    @pytest.fixture
    def setup_builder(self):
        """Create test setup for Hankel matrix builder."""
        n = 3
        m = 2
        builder = HankelMatrixBuilder(n, m)

        # Create dummy segment data
        n_traj = 5
        segment_length = 20

        segment_data = []
        for _ in range(n_traj):
            seg_data = {
                "length": segment_length,
                "eta": np.random.randn(segment_length + 1, n) * 0.1,
                "xi": np.random.randn(segment_length, m) * 0.1,
                "eta_next": np.random.randn(segment_length, n) * 0.1,
            }
            segment_data.append(seg_data)

        return builder, segment_data, n, m, n_traj, segment_length

    def test_initialization(self):
        """Test builder initializes correctly."""
        n = 3
        m = 2
        builder = HankelMatrixBuilder(n, m)

        assert builder.n == n
        assert builder.m == m

    def test_build_segment_matrices(self, setup_builder):
        """Test building matrices for a single segment."""
        builder, segment_data, n, m, n_traj, segment_length = setup_builder

        matrices = builder.build_segment_matrices(segment_data)

        # Check all required keys
        assert "H" in matrices
        assert "H_plus" in matrices
        assert "Xi" in matrices
        assert "L" in matrices
        assert "n_trajectories" in matrices

        # Check shapes
        L = n_traj * segment_length
        assert matrices["H"].shape == (n + m, L)
        assert matrices["H_plus"].shape == (n, L)
        assert matrices["Xi"].shape == (m, L)
        assert matrices["L"] == L

    def test_hankel_matrix_structure(self, setup_builder):
        """Test Hankel matrix has correct structure."""
        builder, segment_data, n, m, n_traj, segment_length = setup_builder

        matrices = builder.build_segment_matrices(segment_data)

        H = matrices["H"]
        H_plus = matrices["H_plus"]  # noqa: F841
        Xi = matrices["Xi"]

        # H should contain [eta; xi]
        # Check that top n rows correspond to eta
        # and bottom m rows correspond to xi
        assert H.shape[0] == n + m

        # Xi should match bottom m rows of H
        np.testing.assert_allclose(H[n:, :], Xi, atol=1e-10)

    def test_data_equation_consistency(self, setup_builder):
        """Test that data matrices are internally consistent."""
        builder, segment_data, n, m, n_traj, segment_length = setup_builder

        matrices = builder.build_segment_matrices(segment_data)

        H = matrices["H"]
        H_plus = matrices["H_plus"]
        L = matrices["L"]

        # Each column should correspond to one timestep
        assert H.shape[1] == L
        assert H_plus.shape[1] == L

        # H_plus should contain next states
        # Verify first few columns match the segment data
        for traj_idx, seg_data in enumerate(segment_data):
            for k in range(min(3, segment_length)):  # Check first 3 timesteps
                # Column index: each trajectory takes segment_length columns
                col_idx = traj_idx * segment_length + k

                # H contains [eta(k); xi(k)]
                expected_h = np.concatenate([seg_data["eta"][k], seg_data["xi"][k]])
                np.testing.assert_allclose(H[:, col_idx], expected_h, atol=1e-10)

                # H_plus contains eta(k+1)
                expected_hplus = seg_data["eta_next"][k]
                np.testing.assert_allclose(H_plus[:, col_idx], expected_hplus, atol=1e-10)

    def test_check_data_sufficiency(self, setup_builder):
        """Test data sufficiency checking."""
        builder, segment_data, n, m, n_traj, segment_length = setup_builder

        matrices = builder.build_segment_matrices(segment_data)
        H = matrices["H"]

        sufficient, info = builder.check_data_sufficiency(H)

        # Check info keys
        assert "rank" in info
        assert "required_rank" in info
        assert "sufficient" in info
        assert "singular_values" in info
        assert "condition_number" in info

        # With random data, should likely be sufficient
        # (100 samples for 5-dimensional system)
        assert info["required_rank"] == n + m

    def test_compute_data_statistics(self, setup_builder):
        """Test data statistics computation."""
        builder, segment_data, n, m, n_traj, segment_length = setup_builder

        matrices = builder.build_segment_matrices(segment_data)
        stats = builder.compute_data_statistics(matrices)

        # Check all expected statistics are present
        expected_keys = [
            "H_norm_frobenius",
            "H_norm_max",
            "H_mean",
            "H_std",
            "H_plus_norm_frobenius",
            "H_plus_norm_max",
            "H_plus_mean",
            "H_plus_std",
            "Xi_norm_frobenius",
            "Xi_norm_max",
            "Xi_mean",
            "Xi_std",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))


class TestInformativityChecker:
    """Tests for InformativityChecker."""

    @pytest.fixture
    def setup_checker(self):
        """Create test setup for informativity checker."""
        n = 3
        m = 2
        checker = InformativityChecker(n, m)

        # Create well-conditioned data matrices
        L = 100  # Number of samples
        H = np.random.randn(n + m, L) * 0.1
        Xi = np.random.randn(m, L) * 0.1

        return checker, H, Xi, n, m, L

    def test_initialization(self):
        """Test checker initializes correctly."""
        n = 3
        m = 2
        checker = InformativityChecker(n, m)

        assert checker.n == n
        assert checker.m == m
        assert checker.rank_threshold > 0

    def test_check_rank_condition(self, setup_checker):
        """Test rank condition checking."""
        checker, H, Xi, n, m, L = setup_checker

        informative, info = checker.check_rank_condition(H, Xi)

        # Check info keys
        assert "rank" in info
        assert "required_rank" in info
        assert "informative" in info
        assert "singular_values" in info
        assert "condition_number" in info

        # Should be informative with random full-rank data
        assert info["required_rank"] == n + m

    def test_rank_deficient_data(self):
        """Test detection of rank-deficient data."""
        n = 3
        m = 2
        checker = InformativityChecker(n, m)

        # Create rank-deficient data (all zeros)
        L = 50
        H = np.zeros((n + m, L))
        Xi = np.zeros((m, L))

        informative, info = checker.check_rank_condition(H, Xi)

        # Should not be informative
        assert not informative
        assert info["rank"] < info["required_rank"]
        assert info["rank_deficiency"] > 0

    def test_check_segment_informativity(self, setup_checker):
        """Test segment informativity checking."""
        checker, H, Xi, n, m, L = setup_checker

        matrices = {"H": H, "Xi": Xi}
        informative, info = checker.check_segment_informativity(matrices)

        # Check additional info
        assert "n_samples" in info
        assert "min_required_samples" in info
        assert "sufficient_samples" in info

        assert info["n_samples"] == L
        assert info["min_required_samples"] == n + m

    def test_suggest_improvements(self):
        """Test improvement suggestions for non-informative data."""
        n = 3
        m = 2
        checker = InformativityChecker(n, m)

        # Create barely sufficient data
        info = {
            "informative": False,
            "rank_deficiency": 2,
            "sample_margin": 5,
            "condition_number": 1e12,
            "min_singular_value": 1e-9,
        }

        suggestions = checker.suggest_improvements(info)

        # Should have multiple suggestions
        assert len(suggestions) > 0
        assert any("rank" in s.lower() or "excitation" in s.lower() for s in suggestions)

    def test_compute_excitation_energy(self, setup_checker):
        """Test excitation energy computation."""
        checker, H, Xi, n, m, L = setup_checker

        energy = checker.compute_excitation_energy(Xi)

        # Check all expected metrics
        assert "total_energy" in energy
        assert "per_input_energy" in energy
        assert "mean_col_norm" in energy
        assert "max_col_norm" in energy

        # Energy should be positive
        assert energy["total_energy"] > 0
        assert all(e > 0 for e in energy["per_input_energy"])

    def test_analyze_data_diversity(self, setup_checker):
        """Test data diversity analysis."""
        checker, H, Xi, n, m, L = setup_checker

        diversity = checker.analyze_data_diversity(H)

        # Check metrics
        assert "covariance_eigenvalues" in diversity
        assert "trace" in diversity
        assert "determinant" in diversity
        assert "condition_number" in diversity

        # Eigenvalues should be positive
        assert all(eig >= 0 for eig in diversity["covariance_eigenvalues"])


class TestIntegration:
    """Integration tests across data collection modules."""

    def test_full_pipeline(self):
        """Test complete data collection pipeline."""
        # 1. Create nominal trajectory
        N = 50
        dt = 0.1
        n = 3
        m = 2

        x_nom = np.zeros((N + 1, n))
        u_nom = np.ones((N, m)) * 0.5

        for k in range(N):
            x_nom[k + 1] = x_nom[k] + dt * np.array([u_nom[k, 0], 0, u_nom[k, 1]])

        # 2. Create plant and collector
        twin = UnicycleModel(x0=x_nom[0], xf=x_nom[-1])
        plant = PlantModel(twin=twin)

        collector = OfflineDataCollector(
            plant=plant, nominal_x=x_nom, nominal_u=u_nom, dt=dt, excitation_magnitude=0.1, seed=42
        )

        # 3. Collect data
        n_samples = 10
        semi_axes = np.array([0.1, 0.1, 0.05])
        dataset = collector.collect_dataset(n_samples, semi_axes, verbose=False)

        assert len(dataset) == n_samples

        # 4. Segment trajectories
        segment_length = 15
        segmenter = TrajectorySegmenter(N, segment_length, overlap=0)
        segmented_data = segmenter.segment_dataset(dataset, verbose=False)

        assert len(segmented_data) == n_samples

        # 5. Build Hankel matrices
        builder = HankelMatrixBuilder(n, m)
        all_matrices = builder.build_all_segments(segmented_data, segmenter, verbose=False)

        assert len(all_matrices) > 0

        # 6. Check informativity
        checker = InformativityChecker(n, m)
        results = checker.check_all_segments(all_matrices, verbose=False)

        assert "all_informative" in results
        assert "n_segments" in results
        assert results["n_segments"] == len(all_matrices)

    def test_data_consistency_through_pipeline(self):
        """Test data remains consistent through segmentation and matrix building."""
        # Create simple test data
        N = 40
        n = 3
        m = 2
        n_traj = 5

        # Create trajectories
        trajectories = []
        for i in range(n_traj):
            traj = {
                "eta": np.ones((N + 1, n)) * (i + 1),  # Constant value per trajectory
                "xi": np.ones((N, m)) * (i + 1) * 0.1,
                "eta_next": np.ones((N, n)) * (i + 1),
            }
            trajectories.append(traj)

        # Segment
        segment_length = 20
        segmenter = TrajectorySegmenter(N, segment_length, overlap=0)
        segmented_data = segmenter.segment_dataset(trajectories, verbose=False)

        # Build matrices for first segment
        builder = HankelMatrixBuilder(n, m)
        segment_0_data = segmenter.get_segment_data_by_index(segmented_data, 0)
        matrices = builder.build_segment_matrices(segment_0_data)

        H = matrices["H"]  # noqa: F841
        Xi = matrices["Xi"]

        # Verify data came from correct trajectories
        # Each trajectory contributes segment_length columns
        for traj_idx in range(n_traj):
            col_start = traj_idx * segment_length
            col_end = (traj_idx + 1) * segment_length

            # Check Xi values match trajectory
            expected_xi = (traj_idx + 1) * 0.1
            np.testing.assert_allclose(Xi[:, col_start:col_end], expected_xi, atol=1e-10)

    def test_informativity_with_insufficient_data(self):
        """Test that insufficient data is correctly detected."""
        n = 3
        m = 2

        # Create data with too few samples
        L = n + m - 1  # One sample short
        H = np.random.randn(n + m, L)
        Xi = np.random.randn(m, L)

        checker = InformativityChecker(n, m)
        matrices = {"H": H, "Xi": Xi}

        informative, info = checker.check_segment_informativity(matrices)

        # Should detect insufficient samples
        assert not info["sufficient_samples"]
        assert info["sample_margin"] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
