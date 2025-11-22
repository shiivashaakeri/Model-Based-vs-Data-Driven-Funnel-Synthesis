# ddfs/ddfs/data_collection/segmenter.py

"""
Trajectory segmentation for Phase 2 data collection.

This module segments trajectories into time windows for building Hankel matrices.

Key components:
- SegmentedData: Container for segmented trajectories
- TrajectorySegmenter: Segments trajectories into overlapping or non-overlapping windows
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ddfs.data_collection.collector import Trajectory


@dataclass
class SegmentedData:
    """
    Segmented trajectory data container.

    Contains lists of trajectories for each segment.

    For a trajectory of length N, segments are created with:
    - Segment spacing T (distance between segment start points)
    - Data window length L (actual data used from each segment)

    Segments can overlap if L > T.

    Attributes
    ----------
    segments : List[List[Trajectory]]
        segments[i] contains M trajectories for segment i
    segment_indices : List[int]
        Segment indices [0, 1, 2, ...]
    k_starts : List[int]
        Start timesteps for each segment
    k_ends : List[int]
        End timesteps for each segment (inclusive)
    T : int
        Segment spacing/length
    L : int
        Data window length

    Examples
    --------
    >>> from ddfs.data_collection import SegmentedData
    >>>
    >>> # Access segment data
    >>> seg_data = SegmentedData(...)
    >>> print(seg_data.num_segments)
    5
    >>> print(seg_data.num_trials)
    50
    >>>
    >>> # Get specific segment
    >>> segment_0_trajs = seg_data.get_segment(0)
    """

    segments: List[List[Trajectory]]
    segment_indices: List[int]
    k_starts: List[int]
    k_ends: List[int]
    T: int
    L: int

    @property
    def num_segments(self) -> int:
        """Number of segments."""
        return len(self.segments)

    @property
    def num_trials(self) -> int:
        """Number of trials (M) in each segment."""
        if self.num_segments == 0:
            return 0
        return len(self.segments[0])

    def get_segment(self, idx: int) -> List[Trajectory]:
        """
        Get trajectories for a specific segment.

        Parameters
        ----------
        idx : int
            Segment index

        Returns
        -------
        trajectories : List[Trajectory]
            Trajectories for this segment
        """
        if idx < 0 or idx >= self.num_segments:
            raise IndexError(f"Segment index {idx} out of range [0, {self.num_segments})")
        return self.segments[idx]

    def save(self, path: Path | str):
        """Save segmented data to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path | str) -> "SegmentedData":
        """Load segmented data from file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        """String representation."""
        return f"SegmentedData(segments={self.num_segments}, trials={self.num_trials}, T={self.T}, L={self.L})"


class TrajectorySegmenter:
    """
    Segments trajectories into time windows.

    For a trajectory of length N, creates segments with:
    - Segment spacing T: Distance between segment start points
    - Data window L: Amount of data extracted from each segment

    Segmentation modes:
    - Non-overlapping: L = T (segments don't overlap)
    - Overlapping: L > T (segments share data)
    - Sparse: L < T (gaps between segments)

    Parameters
    ----------
    T : int
        Segment spacing (e.g., 100 timesteps between segment starts)
    L : int
        Data window length (e.g., 60 timesteps of data per segment)

    Notes
    -----
    - If L = T: Non-overlapping segments
    - If L > T: Overlapping segments (overlap = L - T)
    - If L < T: Sparse segments (gap = T - L)

    Examples
    --------
    >>> from ddfs.data_collection import TrajectorySegmenter
    >>>
    >>> # Create segmenter
    >>> segmenter = TrajectorySegmenter(T=100, L=60)
    >>>
    >>> # Segment trajectories
    >>> segmented_data = segmenter.segment(trajectories)
    >>> print(segmented_data.num_segments)
    5
    """

    def __init__(self, T: int, L: int):
        """
        Initialize segmenter.

        Parameters
        ----------
        T : int
            Segment spacing
        L : int
            Data window length

        Raises
        ------
        ValueError
            If T or L are invalid
        """
        self.T = T
        self.L = L

        if L <= 0:
            raise ValueError(f"Data window length L must be positive, got {L}")
        if T <= 0:
            raise ValueError(f"Segment spacing T must be positive, got {T}")

        # Calculate overlap/gap
        if L > T:
            self.overlap = L - T
            self.gap = 0
        else:
            self.overlap = 0
            self.gap = T - L

    def compute_segments(self, N: int) -> List[Tuple[int, int, int]]:
        """
        Compute segment boundaries for horizon of length N.

        Parameters
        ----------
        N : int
            Horizon length (number of control intervals)

        Returns
        -------
        segments : List[Tuple[int, int, int]]
            List of (segment_idx, k_start, k_end) tuples
            where k_start and k_end are inclusive

        Notes
        -----
        We need L *control* intervals, which means L+1 states.
        For a trajectory with N control intervals, we have N+1 states.

        Examples
        --------
        >>> segmenter = TrajectorySegmenter(T=50, L=60)
        >>> segments = segmenter.compute_segments(N=100)
        >>> print(segments)
        [(0, 0, 59), (1, 50, 99)]
        """
        segments = []
        k = 0
        segment_idx = 0

        while k + self.L <= N:
            k_start = k
            k_end = k + self.L - 1  # Inclusive end
            segments.append((segment_idx, k_start, k_end))

            k += self.T
            segment_idx += 1

        return segments

    def segment(  # noqa: C901, PLR0912
        self,
        trajectories: List[Trajectory],
        verbose: bool = True,
    ) -> SegmentedData:
        """
        Segment all M trajectories.

        For each segment i:
            Extract data window from all trajectories
            at times [k_i, k_i+1, ..., k_i+L-1]

        Parameters
        ----------
        trajectories : List[Trajectory]
            List of M trajectories to segment
        verbose : bool, optional
            Print progress, by default True

        Returns
        -------
        segmented_data : SegmentedData
            Segmented trajectory data

        Raises
        ------
        ValueError
            If trajectories list is empty or trajectories have inconsistent lengths
        """
        if len(trajectories) == 0:
            raise ValueError("No trajectories to segment")

        # Validate all trajectories have same length
        N = trajectories[0].N
        for i, traj in enumerate(trajectories):
            if traj.N != N:
                raise ValueError(f"Trajectory {i} has length {traj.N}, expected {N}")

        # Compute segment boundaries
        segment_info = self.compute_segments(N)

        if len(segment_info) == 0:
            raise ValueError(f"No segments possible with N={N}, T={self.T}, L={self.L}. Need N >= L.")

        if verbose:
            print(f"\nSegmenting {len(trajectories)} trajectories...")
            print(f"  Trajectory horizon: N={N}")
            print(f"  Segment spacing: T={self.T}")
            print(f"  Data window length: L={self.L}")

            if self.overlap > 0:
                print(f"  Mode: Overlapping (overlap={self.overlap} timesteps)")
            elif self.gap > 0:
                print(f"  Mode: Sparse (gap={self.gap} timesteps)")
            else:
                print("  Mode: Non-overlapping")

            print(f"  Number of segments: {len(segment_info)}")

        # Segment trajectories
        segments = []
        segment_indices = []
        k_starts = []
        k_ends = []

        for seg_idx, k_start, k_end in segment_info:
            segment_trajs = []

            for traj in trajectories:
                # Extract segment from this trajectory
                # We need L control intervals, which means:
                # - States: x[k_start] to x[k_start+L] (L+1 states)
                # - Controls: u[k_start] to u[k_start+L-1] (L controls)
                # - Deviations: same as above

                seg_traj = Trajectory(
                    x=traj.x[k_start : k_start + self.L + 1].copy(),
                    u=traj.u[k_start : k_start + self.L].copy(),
                    eta=traj.eta[k_start : k_start + self.L + 1].copy(),
                    xi=traj.xi[k_start : k_start + self.L].copy(),
                    trial_id=traj.trial_id,
                    x0=traj.x[k_start].copy(),
                )
                segment_trajs.append(seg_traj)

            segments.append(segment_trajs)
            segment_indices.append(seg_idx)
            k_starts.append(k_start)
            k_ends.append(k_end)

            if verbose:
                print(
                    f"  Segment {seg_idx}: k=[{k_start}:{k_end}], {len(segment_trajs)} trajectories, {self.L} timesteps"
                )

        # Create segmented data object
        segmented_data = SegmentedData(
            segments=segments,
            segment_indices=segment_indices,
            k_starts=k_starts,
            k_ends=k_ends,
            T=self.T,
            L=self.L,
        )

        if verbose:
            print(f"✓ Segmented into {segmented_data.num_segments} segments")
            print(f"  Each segment has {segmented_data.num_trials} trials")

            # Validate segment lengths
            for i, seg in enumerate(segments):
                for j, traj in enumerate(seg):
                    if traj.N != self.L:
                        print(f"  ⚠ WARNING: Segment {i}, trial {j} has length {traj.N}, expected {self.L}")

        return segmented_data

    def get_coverage(self, N: int) -> Tuple[int, float]:
        """
        Compute how much of the trajectory is covered by segments.

        Parameters
        ----------
        N : int
            Trajectory horizon length

        Returns
        -------
        covered_timesteps : int
            Number of unique timesteps covered by at least one segment
        coverage_ratio : float
            Fraction of trajectory covered (0 to 1)
        """
        segments = self.compute_segments(N)

        if len(segments) == 0:
            return 0, 0.0

        # Mark which timesteps are covered
        covered = np.zeros(N, dtype=bool)

        for _, k_start, k_end in segments:
            covered[k_start : k_end + 1] = True

        covered_timesteps = np.sum(covered)
        coverage_ratio = covered_timesteps / N

        return int(covered_timesteps), coverage_ratio

    def __repr__(self) -> str:
        """String representation."""
        mode = "Overlapping" if self.overlap > 0 else ("Sparse" if self.gap > 0 else "Non-overlapping")
        return f"TrajectorySegmenter(T={self.T}, L={self.L}, mode={mode})"
