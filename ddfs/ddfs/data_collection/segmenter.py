# ddfs/ddfs/data_collection/segmenter.py

"""
Trajectory Segmenter for Phase 2: Data Collection

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
    Segmented trajectory data.

    Contains lists of trajectories for each segment.

    For a trajectory of length N, segments are created with:
    - Segment length T (spacing between segments)
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

    def save(self, path: Path):
        """Save segmented data to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "SegmentedData":
        """Load segmented data from file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
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

    Example with N=100, T=50, L=60:
    - Segment 0: k=[0, 59] (60 points)
    - Segment 1: k=[50, 109] → capped to k=[50, 99] (50 points)
    - Overlap: 10 timesteps between segments 0 and 1
    """

    def __init__(self, T: int, L: int):
        """
        Initialize segmenter.

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
        N=100, T=50, L=60:
        - Segment 0: k_start=0, k_end=59 (60 controls, 61 states)
        - Segment 1: k_start=50, k_end=99 (50 controls, 51 states)

        N=100, T=100, L=60:
        - Segment 0: k_start=0, k_end=59 (only one segment fits)
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

    def segment(self, trajectories: List[Trajectory], verbose: bool = True) -> SegmentedData:  # noqa: C901, PLR0912
        """
        Segment all M trajectories.

        For each segment i:
            Extract data window from all trajectories
            at times [k_i, k_i+1, ..., k_i+L-1]

        Parameters
        ----------
        trajectories : List[Trajectory]
            List of M trajectories to segment
        verbose : bool
            Print progress

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
            segments=segments, segment_indices=segment_indices, k_starts=k_starts, k_ends=k_ends, T=self.T, L=self.L
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

    def validate_segmentation(self, segmented_data: SegmentedData) -> Tuple[bool, List[str]]:  # noqa: C901, PLR0912
        """
        Validate segmented data for consistency.

        Parameters
        ----------
        segmented_data : SegmentedData
            Segmented data to validate

        Returns
        -------
        is_valid : bool
            True if all checks pass
        issues : List[str]
            List of issues found (empty if valid)
        """
        issues = []

        # Check each segment has same number of trials
        M = segmented_data.num_trials
        for i, seg in enumerate(segmented_data.segments):
            if len(seg) != M:
                issues.append(f"Segment {i} has {len(seg)} trials, expected {M}")

        # Check each trajectory in each segment has correct length
        for i, seg in enumerate(segmented_data.segments):
            for j, traj in enumerate(seg):
                if traj.N != self.L:
                    issues.append(f"Segment {i}, trial {j} has length {traj.N}, expected {self.L}")

        # Check dimensions are consistent
        for i, seg in enumerate(segmented_data.segments):
            n = seg[0].state_dim
            m = seg[0].input_dim

            for j, traj in enumerate(seg):
                if traj.state_dim != n:
                    issues.append(f"Segment {i}, trial {j} has state_dim={traj.state_dim}, expected {n}")
                if traj.input_dim != m:
                    issues.append(f"Segment {i}, trial {j} has input_dim={traj.input_dim}, expected {m}")

        # Check segment boundaries are consistent
        if len(segmented_data.k_starts) != segmented_data.num_segments:
            issues.append(
                f"Number of k_starts ({len(segmented_data.k_starts)}) != num_segments ({segmented_data.num_segments})"
            )

        if len(segmented_data.k_ends) != segmented_data.num_segments:
            issues.append(
                f"Number of k_ends ({len(segmented_data.k_ends)}) != num_segments ({segmented_data.num_segments})"
            )

        # Check k_end - k_start = L - 1
        for i in range(segmented_data.num_segments):
            expected_length = self.L - 1
            actual_length = segmented_data.k_ends[i] - segmented_data.k_starts[i]
            if actual_length != expected_length:
                issues.append(f"Segment {i}: k_end - k_start = {actual_length}, expected {expected_length}")

        is_valid = len(issues) == 0

        return is_valid, issues

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
        mode = "Overlapping" if self.overlap > 0 else ("Sparse" if self.gap > 0 else "Non-overlapping")
        return f"TrajectorySegmenter(T={self.T}, L={self.L}, mode={mode})"
