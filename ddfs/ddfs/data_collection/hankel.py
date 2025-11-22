# ddfs/ddfs/data_collection/hankel.py

"""
Hankel matrix builder for Phase 2 data collection.

This module builds Hankel matrices from segmented trajectory data
for data-driven control and uncertainty quantification.

Key components:
- SegmentHankelMatrices: Container for Hankel matrices of one segment
- HankelMatrixBuilder: Builds Hankel matrices from trajectories
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from ddfs.data_collection.collector import Trajectory
from ddfs.data_collection.segmenter import SegmentedData


@dataclass
class SegmentHankelMatrices:
    """
    Hankel matrices for one segment.

    For segment i with data window [k_i, k_i+1, ..., k_i+L-1],
    stacks data from all M trajectories.

    The matrices are:
        H_i: State deviations η(k)
        H_i+: Next state deviations η(k+1)
        Ξ_i: Input deviations ξ(k)

    Stacked horizontally across all trials and timesteps:
        H_i = [η(k_i)^(1), ..., η(k_i+L-1)^(1), η(k_i)^(2), ..., η(k_i+L-1)^(M)]

    Attributes
    ----------
    segment_idx : int
        Segment index i
    H : np.ndarray
        Stacked state deviations η(k), shape (n, L*M)
    H_plus : np.ndarray
        Stacked next state deviations η(k+1), shape (n, L*M)
    Xi : np.ndarray
        Stacked input deviations ξ(k), shape (m, L*M)
    k_start : int
        Start timestep of segment
    k_end : int
        End timestep of segment (inclusive)
    L : int
        Data window length
    M : int
        Number of trials

    Examples
    --------
    >>> from ddfs.data_collection import SegmentHankelMatrices
    >>>
    >>> # Check informativity
    >>> matrices = SegmentHankelMatrices(...)
    >>> is_informative, rank, required = matrices.check_informativity()
    >>> print(f"Informative: {is_informative}, rank={rank}/{required}")
    """

    segment_idx: int
    H: np.ndarray  # (n, L*M)
    H_plus: np.ndarray  # (n, L*M)
    Xi: np.ndarray  # (m, L*M)
    k_start: int
    k_end: int
    L: int
    M: int

    def __post_init__(self):
        """Validate dimensions."""
        n, LM = self.H.shape
        m, LM2 = self.Xi.shape

        if self.H_plus.shape != (n, LM):
            raise ValueError(f"H_plus shape mismatch: expected ({n}, {LM}), got {self.H_plus.shape}")
        if LM2 != LM:
            raise ValueError(f"Xi width mismatch: expected {LM}, got {LM2}")
        if LM != self.L * self.M:
            raise ValueError(f"Expected {self.L}*{self.M}={self.L * self.M} columns, got {LM}")

    @property
    def state_dim(self) -> int:
        """State dimension n."""
        return self.H.shape[0]

    @property
    def input_dim(self) -> int:
        """Input dimension m."""
        return self.Xi.shape[0]

    def check_informativity(self) -> Tuple[bool, int, int]:
        """
        Check persistence of excitation condition.

        The data is informative if:
            rank([H; Xi]) = n + m

        This is the fundamental condition for data-driven control.

        Returns
        -------
        is_informative : bool
            True if persistence of excitation is satisfied
        actual_rank : int
            Actual rank of [H; Xi]
        required_rank : int
            Required rank (n + m)
        """
        n = self.H.shape[0]
        m = self.Xi.shape[0]

        # Stack H and Xi vertically: [H; Xi]
        data_matrix = np.vstack([self.H, self.Xi])  # (n+m, L*M)

        # Compute rank using SVD
        actual_rank = np.linalg.matrix_rank(data_matrix)
        required_rank = n + m

        is_informative = actual_rank == required_rank

        return is_informative, actual_rank, required_rank

    def compute_condition_number(self) -> float:
        """
        Compute condition number of [H; Xi].

        Lower is better (well-conditioned).
        Values > 1e10 indicate numerical issues.
        Values > 1e15 suggest near rank-deficiency.

        Returns
        -------
        cond : float
            Condition number κ([H; Xi])
        """
        data_matrix = np.vstack([self.H, self.Xi])
        cond = np.linalg.cond(data_matrix)
        return float(cond)

    def compute_singular_values(self) -> np.ndarray:
        """
        Compute singular values of [H; Xi].

        Useful for analyzing data quality and informativity.

        Returns
        -------
        sigma : np.ndarray
            Singular values in descending order
        """
        data_matrix = np.vstack([self.H, self.Xi])
        sigma = np.linalg.svd(data_matrix, compute_uv=False)
        return sigma

    def compute_minimum_singular_value(self) -> float:
        """
        Compute minimum singular value of [H; Xi].

        This is related to the "strength" of excitation.
        Larger values indicate better excitation.

        Returns
        -------
        sigma_min : float
            Minimum singular value
        """
        sigma = self.compute_singular_values()
        return float(sigma[-1])

    def get_data_matrix(self) -> np.ndarray:
        """
        Get the full data matrix [H; Xi].

        Returns
        -------
        data_matrix : np.ndarray
            Stacked data matrix, shape (n+m, L*M)
        """
        return np.vstack([self.H, self.Xi])

    def save(self, path: Union[Path, str]):
        """Save matrices to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[Path, str]) -> "SegmentHankelMatrices":
        """Load matrices from file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SegmentHankelMatrices("
            f"segment={self.segment_idx}, "
            f"k=[{self.k_start}:{self.k_end}], "
            f"H: {self.H.shape}, "
            f"L={self.L}, M={self.M})"
        )


class HankelMatrixBuilder:
    """
    Builds Hankel matrices from segmented trajectory data.

    For segment i with M trajectories, stacks data into:
    - H_i: State deviations η(k)
    - H_i+: Next state deviations η(k+1)
    - Ξ_i: Input deviations ξ(k)

    The stacking follows the pattern:
        H_i = [η(k_i)^(1), ..., η(k_i+L-1)^(1),  # Trial 1
               η(k_i)^(2), ..., η(k_i+L-1)^(2),  # Trial 2
               ...,
               η(k_i)^(M), ..., η(k_i+L-1)^(M)]  # Trial M

    This results in matrices of dimension (n x L*M).

    Parameters
    ----------
    verbose : bool, optional
        Print progress and diagnostics, by default True

    Examples
    --------
    >>> from ddfs.data_collection import HankelMatrixBuilder
    >>>
    >>> builder = HankelMatrixBuilder(verbose=True)
    >>> all_matrices = builder.build_all_segments(segmented_data)
    >>> print(len(all_matrices))
    5
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize builder.

        Parameters
        ----------
        verbose : bool, optional
            Print progress and diagnostics
        """
        self.verbose = verbose

    def build_segment_matrices(
        self,
        segment_trajectories: List[Trajectory],
        segment_idx: int,
        k_start: int,
        k_end: int,
    ) -> SegmentHankelMatrices:
        """
        Build H_i, H_i+, Ξ_i for segment i.

        Stacks data from all M trajectories in this segment:

        H_i = [η(k_i)^(1), ..., η(k_i+L-1)^(1),
               η(k_i)^(2), ..., η(k_i+L-1)^(2),
               ...,
               η(k_i)^(M), ..., η(k_i+L-1)^(M)]

        Dimension: (n x L*M)

        Similarly for H_i+ (next states) and Ξ_i (inputs).

        Parameters
        ----------
        segment_trajectories : List[Trajectory]
            Trajectories for this segment (length M)
        segment_idx : int
            Segment index
        k_start : int
            Segment start timestep (in original trajectory)
        k_end : int
            Segment end timestep (in original trajectory)

        Returns
        -------
        matrices : SegmentHankelMatrices
            Hankel matrices for this segment
        """
        M = len(segment_trajectories)

        if M == 0:
            raise ValueError("No trajectories in segment")

        # Get dimensions from first trajectory
        n = segment_trajectories[0].state_dim
        m = segment_trajectories[0].input_dim
        L = segment_trajectories[0].N  # Data window length

        # Validate all trajectories have same dimensions
        for traj in segment_trajectories:
            if traj.state_dim != n or traj.input_dim != m:
                raise ValueError(
                    f"Inconsistent dimensions: expected ({n}, {m}), got ({traj.state_dim}, {traj.input_dim})"
                )
            if traj.N != L:
                raise ValueError(f"Inconsistent window length: expected {L}, got {traj.N}")

        # Pre-allocate matrices
        H = np.zeros((n, L * M))
        H_plus = np.zeros((n, L * M))
        Xi = np.zeros((m, L * M))

        # Stack data from all trajectories
        for trial_idx, traj in enumerate(segment_trajectories):
            # Column indices for this trial
            col_start = trial_idx * L
            col_end = (trial_idx + 1) * L

            # H: η(k) for k = 0, ..., L-1 (in segment-local coordinates)
            # Shape: (L, n) -> transpose to (n, L)
            H[:, col_start:col_end] = traj.eta[:L].T

            # H+: η(k+1) for k = 0, ..., L-1
            # Shape: (L, n) -> transpose to (n, L)
            H_plus[:, col_start:col_end] = traj.eta[1 : L + 1].T

            # Ξ: ξ(k) for k = 0, ..., L-1
            # Shape: (L, m) -> transpose to (m, L)
            Xi[:, col_start:col_end] = traj.xi[:L].T

        # Create matrices object
        matrices = SegmentHankelMatrices(
            segment_idx=segment_idx,
            H=H,
            H_plus=H_plus,
            Xi=Xi,
            k_start=k_start,
            k_end=k_end,
            L=L,
            M=M,
        )

        # Print diagnostics if verbose
        if self.verbose:
            is_informative, actual_rank, required_rank = matrices.check_informativity()
            cond = matrices.compute_condition_number()
            sigma_min = matrices.compute_minimum_singular_value()

            print(f"  Segment {segment_idx}:")
            print(f"    Matrices: H {H.shape}, H+ {H_plus.shape}, Ξ {Xi.shape}")
            print(f"    Informativity: {'✓' if is_informative else '✗ FAILED'} (rank={actual_rank}/{required_rank})")
            print(f"    Condition number: {cond:.2e}")
            print(f"    Min singular value: {sigma_min:.2e}")

            if not is_informative:
                print("    ⚠ WARNING: Data is NOT informative! Need more excitation.")
            elif cond > 1e10:
                print("    ⚠ WARNING: Poor conditioning (κ > 1e10). Consider more excitation.")

        return matrices

    def build_all_segments(self, segmented_data: SegmentedData) -> List[SegmentHankelMatrices]:
        """
        Build Hankel matrices for all segments.

        Parameters
        ----------
        segmented_data : SegmentedData
            Segmented trajectory data

        Returns
        -------
        all_matrices : List[SegmentHankelMatrices]
            Hankel matrices for each segment
        """
        if self.verbose:
            print(f"\nBuilding Hankel matrices for {segmented_data.num_segments} segments...")

        all_matrices = []

        for i in range(segmented_data.num_segments):
            matrices = self.build_segment_matrices(
                segment_trajectories=segmented_data.segments[i],
                segment_idx=segmented_data.segment_indices[i],
                k_start=segmented_data.k_starts[i],
                k_end=segmented_data.k_ends[i],
            )
            all_matrices.append(matrices)

        if self.verbose:
            # Summary statistics
            print(f"\n{'=' * 60}")
            print("Hankel Matrix Summary")
            print(f"{'=' * 60}")

            num_informative = sum(1 for mat in all_matrices if mat.check_informativity()[0])

            print(f"Total segments: {len(all_matrices)}")
            print(f"Informative: {num_informative}/{len(all_matrices)}")

            if num_informative < len(all_matrices):
                print(f"⚠ WARNING: {len(all_matrices) - num_informative} segments are NOT informative!")
                print("  Consider: increasing M, increasing excitation amplitude, or different excitation type")
            else:
                print("✓ All segments are informative")

            # Condition number statistics
            conds = [mat.compute_condition_number() for mat in all_matrices]
            print("\nCondition numbers:")
            print(f"  Min: {np.min(conds):.2e}")
            print(f"  Max: {np.max(conds):.2e}")
            print(f"  Mean: {np.mean(conds):.2e}")

            print(f"{'=' * 60}\n")

        return all_matrices

    def __repr__(self) -> str:
        """String representation."""
        return f"HankelMatrixBuilder(verbose={self.verbose})"
