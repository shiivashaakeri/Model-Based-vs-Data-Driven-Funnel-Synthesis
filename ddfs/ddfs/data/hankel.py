"""
Hankel matrix construction for DDFS.

This module builds Hankel-like data matrices from segmented trajectory data:
- H_i: Past state-input data matrix
- H+_i: Future state data matrix
- Ξ_i: Past input deviation matrix

These matrices are used in the data-driven funnel synthesis LMIs.

Data equation: H+_i = A_i H_i + B_i Ξ_i + W_i
where W_i captures model uncertainty.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class HankelMatrixBuilder:
    """
    Build Hankel-like data matrices for funnel synthesis.

    For each segment, constructs matrices from collected trajectory data.
    """

    def __init__(self, n: int, m: int):
        """
        Initialize Hankel matrix builder.

        Args:
            n: State dimension
            m: Input dimension
        """
        self.n = n
        self.m = m

    def build_segment_matrices(self, segment_data: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Build data matrices for a single segment.

        Args:
            segment_data: List of segment dictionaries from all trajectories

        Returns:
            matrices: Dictionary containing:
                - 'H': Past state-input matrix (n+m, L)
                - 'H_plus': Future state matrix (n, L)
                - 'Xi': Past input deviation matrix (m, L)
                - 'L': Number of data samples
                - 'n_trajectories': Number of trajectories used
        """
        n_traj = len(segment_data)
        segment_length = segment_data[0]["length"]  # Number of timesteps in segment

        # Total number of data samples
        L = n_traj * segment_length

        # Allocate matrices
        H = np.zeros((self.n + self.m, L))
        H_plus = np.zeros((self.n, L))
        Xi = np.zeros((self.m, L))

        # Fill matrices by stacking data from each trajectory
        col_idx = 0
        for seg_data in segment_data:
            eta = seg_data["eta"]  # (length+1, n)
            xi = seg_data["xi"]  # (length, m)
            eta_next = seg_data["eta_next"]  # (length, n)

            for k in range(segment_length):
                # Past state-input: [η(k); ξ(k)]
                H[: self.n, col_idx] = eta[k, :]
                H[self.n :, col_idx] = xi[k, :]

                # Future state: η(k+1)
                H_plus[:, col_idx] = eta_next[k, :]

                # Past input deviation: ξ(k)
                Xi[:, col_idx] = xi[k, :]

                col_idx += 1

        matrices = {
            "H": H,
            "H_plus": H_plus,
            "Xi": Xi,
            "L": L,
            "n_trajectories": n_traj,
            "segment_length": segment_length,
        }

        return matrices

    def build_all_segments(
        self, segmented_dataset: List[List[Dict]], segmenter, verbose: bool = True
    ) -> List[Dict[str, np.ndarray]]:
        """
        Build data matrices for all segments.

        Args:
            segmented_dataset: Segmented dataset from TrajectorySegmenter
            segmenter: TrajectorySegmenter instance
            verbose: Print progress

        Returns:
            all_matrices: List of matrix dictionaries, one per segment
        """
        n_segments = segmenter.n_segments

        if verbose:
            print("=" * 70)
            print("HANKEL MATRIX CONSTRUCTION")
            print("=" * 70)
            print(f"Number of segments: {n_segments}")
            print(f"State dimension:    {self.n}")
            print(f"Input dimension:    {self.m}")
            print("=" * 70)

        all_matrices = []

        for seg_idx in range(n_segments):
            # Get data for this segment across all trajectories
            segment_data = segmenter.get_segment_data_by_index(segmented_dataset, seg_idx)

            # Build matrices
            matrices = self.build_segment_matrices(segment_data)

            # Add segment info
            matrices["segment_idx"] = seg_idx
            seg_info = segmenter.get_segment_info(seg_idx)
            matrices["k_start"] = seg_info["k_start"]
            matrices["k_end"] = seg_info["k_end"]

            all_matrices.append(matrices)

            if verbose:
                print(f"\nSegment {seg_idx}:")
                print(f"  Time steps:    [{matrices['k_start']}, {matrices['k_end']}]")
                print(f"  H shape:       {matrices['H'].shape}")
                print(f"  H+ shape:      {matrices['H_plus'].shape}")
                print(f"  Ξ shape:       {matrices['Xi'].shape}")
                print(f"  Data samples:  {matrices['L']}")

        if verbose:
            print("\n" + "=" * 70)
            print(f"✓ Built Hankel matrices for {n_segments} segments")
            print("=" * 70)

        return all_matrices

    def check_data_sufficiency(self, H: np.ndarray, threshold: Optional[float] = None) -> Tuple[bool, Dict]:
        """
        Check if data matrix H has sufficient rank.

        For informativity, we need rank(H) = n + m.

        Args:
            H: Past state-input matrix (n+m, L)
            threshold: Singular value threshold for rank determination
                       (defaults to 1e-10)

        Returns:
            sufficient: True if rank is sufficient
            info: Dictionary with rank information
        """
        if threshold is None:
            threshold = 1e-10

        # Compute SVD
        singular_values = np.linalg.svd(H, compute_uv=False)

        # Determine rank
        rank = np.sum(singular_values > threshold)

        # Required rank
        required_rank = self.n + self.m

        # Check sufficiency
        sufficient = rank >= required_rank

        info = {
            "rank": int(rank),
            "required_rank": required_rank,
            "sufficient": sufficient,
            "singular_values": singular_values,
            "condition_number": float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else np.inf,
            "min_singular_value": float(singular_values[-1]) if len(singular_values) > 0 else 0.0,
            "max_singular_value": float(singular_values[0]) if len(singular_values) > 0 else 0.0,
        }

        return sufficient, info

    def validate_data_equation(self, matrices: Dict[str, np.ndarray], A: np.ndarray, B: np.ndarray) -> Dict:
        """
        Validate data equation: H+ ≈ A H + B Ξ + W

        Computes residual W = H+ - A H - B Ξ and its statistics.

        Args:
            matrices: Dictionary with H, H_plus, Xi
            A: Linearized state dynamics (n, n)
            B: Linearized input dynamics (n, m)

        Returns:
            validation: Dictionary with residual statistics
        """
        H = matrices["H"]
        H_plus = matrices["H_plus"]
        Xi = matrices["Xi"]

        # Extract state and input parts from H
        H_eta = H[: self.n, :]  # State deviations
        H_xi = H[self.n :, :]  # Input deviations  # noqa: F841

        # Compute prediction
        H_plus_pred = A @ H_eta + B @ Xi

        # Compute residual
        W = H_plus - H_plus_pred

        # Statistics
        frobenius_norm = np.linalg.norm(W, ord="fro")
        max_norm = np.max(np.abs(W))
        mean_norm = np.mean(np.linalg.norm(W, axis=0))

        validation = {
            "W": W,
            "frobenius_norm": float(frobenius_norm),
            "max_norm": float(max_norm),
            "mean_col_norm": float(mean_norm),
            "shape": W.shape,
        }

        return validation

    def compute_data_statistics(self, matrices: Dict[str, np.ndarray]) -> Dict:
        """
        Compute statistics of data matrices.

        Args:
            matrices: Dictionary with H, H_plus, Xi

        Returns:
            stats: Dictionary with statistics
        """
        H = matrices["H"]
        H_plus = matrices["H_plus"]
        Xi = matrices["Xi"]

        stats = {
            # H statistics
            "H_norm_frobenius": float(np.linalg.norm(H, ord="fro")),
            "H_norm_max": float(np.max(np.abs(H))),
            "H_mean": float(np.mean(H)),
            "H_std": float(np.std(H)),
            # H+ statistics
            "H_plus_norm_frobenius": float(np.linalg.norm(H_plus, ord="fro")),
            "H_plus_norm_max": float(np.max(np.abs(H_plus))),
            "H_plus_mean": float(np.mean(H_plus)),
            "H_plus_std": float(np.std(H_plus)),
            # Xi statistics
            "Xi_norm_frobenius": float(np.linalg.norm(Xi, ord="fro")),
            "Xi_norm_max": float(np.max(np.abs(Xi))),
            "Xi_mean": float(np.mean(Xi)),
            "Xi_std": float(np.std(Xi)),
        }

        return stats

    def print_matrix_summary(self, matrices: Dict[str, np.ndarray], seg_idx: int):
        """
        Print summary of matrices for a segment.

        Args:
            matrices: Dictionary with matrix data
            seg_idx: Segment index
        """
        print(f"\n{'=' * 70}")
        print(f"SEGMENT {seg_idx} DATA MATRICES")
        print("=" * 70)
        print(f"Time steps:     [{matrices['k_start']}, {matrices['k_end']}]")
        print(f"Segment length: {matrices['segment_length']}")
        print(f"Trajectories:   {matrices['n_trajectories']}")
        print(f"Data samples:   {matrices['L']}")

        print("\nMatrix shapes:")
        print(f"  H  (past):   {matrices['H'].shape}")
        print(f"  H+ (future): {matrices['H_plus'].shape}")
        print(f"  Ξ  (input):  {matrices['Xi'].shape}")

        # Check rank
        sufficient, rank_info = self.check_data_sufficiency(matrices["H"])
        print("\nRank analysis:")
        print(f"  Rank:          {rank_info['rank']}")
        print(f"  Required:      {rank_info['required_rank']}")
        print(f"  Sufficient:    {'✓ Yes' if sufficient else '✗ No'}")
        print(f"  Condition:     {rank_info['condition_number']:.2e}")
        print(f"  σ_min:         {rank_info['min_singular_value']:.2e}")  # noqa: RUF001
        print(f"  σ_max:         {rank_info['max_singular_value']:.2e}")  # noqa: RUF001

        # Statistics
        stats = self.compute_data_statistics(matrices)
        print("\nData statistics:")
        print(f"  ||H||_F:       {stats['H_norm_frobenius']:.4f}")
        print(f"  ||H+||_F:      {stats['H_plus_norm_frobenius']:.4f}")
        print(f"  ||Ξ||_F:       {stats['Xi_norm_frobenius']:.4f}")
        print("=" * 70)

    def __repr__(self) -> str:
        return f"HankelMatrixBuilder(n={self.n}, m={self.m})"
