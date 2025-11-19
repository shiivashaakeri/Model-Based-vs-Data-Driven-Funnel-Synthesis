"""
Trajectory segmentation for DDFS.

This module partitions collected trajectories into time segments for:
1. Building separate data matrices per segment
2. Computing segment-wise uncertainty bounds
3. Synthesizing segment-wise funnels

Each segment T_i = {k_i, k_i+1, ..., k_i + T_seg - 1} contains T_seg timesteps.
"""

from typing import Dict, List, Tuple

import numpy as np


class TrajectorySegmenter:
    """
    Segment trajectories into fixed-length time windows.

    Divides the nominal trajectory into M segments, and extracts
    corresponding data from each collected trajectory.
    """

    def __init__(self, N: int, segment_length: int, overlap: int = 0):
        """
        Initialize trajectory segmenter.

        Args:
            N: Total horizon length (number of timesteps)
            segment_length: Length of each segment (T_seg)
            overlap: Number of overlapping timesteps between segments
        """
        self.N = N
        self.segment_length = segment_length
        self.overlap = overlap

        # Compute segment boundaries
        self.segments = self._compute_segments()
        self.n_segments = len(self.segments)

    def _compute_segments(self) -> List[Tuple[int, int]]:
        """
        Compute segment boundaries.

        Returns:
            segments: List of (start_idx, end_idx) tuples for each segment
        """
        segments = []
        stride = self.segment_length - self.overlap

        k_start = 0
        while k_start < self.N:
            k_end = min(k_start + self.segment_length, self.N)
            segments.append((k_start, k_end))

            # Move to next segment
            k_start += stride

            # Stop if we've covered the whole trajectory
            if k_end >= self.N:
                break

        return segments

    def segment_single_trajectory(self, trajectory_data: Dict) -> List[Dict]:
        """
        Segment a single trajectory into windows.

        Args:
            trajectory_data: Trajectory dictionary with keys:
                - 'eta': State deviations (N+1, n)
                - 'xi': Input deviations (N, m)
                - 'eta_next': Next state deviations (N, n)

        Returns:
            segmented_data: List of segment dictionaries, one per segment
        """
        eta = trajectory_data["eta"]
        xi = trajectory_data["xi"]
        eta_next = trajectory_data["eta_next"]

        segmented_data = []

        for seg_idx, (k_start, k_end) in enumerate(self.segments):
            # Extract data for this segment
            # Note: eta has N+1 states, xi and eta_next have N timesteps
            segment_data = {
                "segment_idx": seg_idx,
                "k_start": k_start,
                "k_end": k_end,
                "length": k_end - k_start,
                "eta": eta[k_start : k_end + 1, :],  # States at k_start, ..., k_end
                "xi": xi[k_start:k_end, :],  # Inputs at k_start, ..., k_end-1
                "eta_next": eta_next[k_start:k_end, :],  # Next states at k_start+1, ..., k_end
            }

            segmented_data.append(segment_data)

        return segmented_data

    def segment_dataset(self, trajectories: List[Dict], verbose: bool = True) -> List[List[Dict]]:
        """
        Segment all trajectories in dataset.

        Args:
            trajectories: List of trajectory dictionaries
            verbose: Print progress

        Returns:
            segmented_dataset: List of lists, where outer list is over trajectories
                               and inner list is over segments
        """
        if verbose:
            print("=" * 70)
            print("TRAJECTORY SEGMENTATION")
            print("=" * 70)
            print(f"Total horizon:    {self.N} timesteps")
            print(f"Segment length:   {self.segment_length}")
            print(f"Overlap:          {self.overlap}")
            print(f"Number of segments: {self.n_segments}")
            print(f"Trajectories:     {len(trajectories)}")
            print("=" * 70)

        segmented_dataset = []

        for traj_idx, traj_data in enumerate(trajectories):
            segmented_traj = self.segment_single_trajectory(traj_data)
            segmented_dataset.append(segmented_traj)

            if verbose and traj_idx % 10 == 0:
                print(f"  Segmented trajectory {traj_idx + 1}/{len(trajectories)}")

        if verbose:
            print("=" * 70)
            print(f"✓ Segmented {len(trajectories)} trajectories into {self.n_segments} segments each")
            print("=" * 70)

        return segmented_dataset

    def get_segment_data_by_index(self, segmented_dataset: List[List[Dict]], segment_idx: int) -> List[Dict]:
        """
        Extract all data for a specific segment across all trajectories.

        Args:
            segmented_dataset: Segmented dataset from segment_dataset()
            segment_idx: Index of segment to extract

        Returns:
            segment_data: List of segment dictionaries, one per trajectory
        """
        if segment_idx < 0 or segment_idx >= self.n_segments:
            raise ValueError(f"Segment index {segment_idx} out of range [0, {self.n_segments})")

        segment_data = [traj_segments[segment_idx] for traj_segments in segmented_dataset]

        return segment_data

    def aggregate_segment_data(self, segment_data: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Aggregate data from all trajectories for a single segment.

        Stacks data horizontally to form matrices for Hankel construction.

        Args:
            segment_data: List of segment dictionaries for one segment

        Returns:
            aggregated: Dictionary with:
                - 'eta_all': Stacked state deviations (length+1, n*n_traj)
                - 'xi_all': Stacked input deviations (length, m*n_traj)
                - 'eta_next_all': Stacked next state deviations (length, n*n_traj)
        """
        n_traj = len(segment_data)
        length = segment_data[0]["length"]

        # Dimensions
        n = segment_data[0]["eta"].shape[1]  # State dim
        m = segment_data[0]["xi"].shape[1]  # Input dim

        # Allocate arrays
        eta_all = np.zeros((length + 1, n * n_traj))
        xi_all = np.zeros((length, m * n_traj))
        eta_next_all = np.zeros((length, n * n_traj))

        # Stack data from each trajectory
        for traj_idx, seg_data in enumerate(segment_data):
            eta_all[:, traj_idx * n : (traj_idx + 1) * n] = seg_data["eta"]
            xi_all[:, traj_idx * m : (traj_idx + 1) * m] = seg_data["xi"]
            eta_next_all[:, traj_idx * n : (traj_idx + 1) * n] = seg_data["eta_next"]

        aggregated = {
            "eta_all": eta_all,
            "xi_all": xi_all,
            "eta_next_all": eta_next_all,
            "n_trajectories": n_traj,
            "length": length,
        }

        return aggregated

    def get_segment_boundaries(self) -> List[Tuple[int, int]]:
        """Get segment boundaries."""
        return self.segments.copy()

    def get_segment_info(self, segment_idx: int) -> Dict:
        """
        Get information about a specific segment.

        Args:
            segment_idx: Segment index

        Returns:
            info: Dictionary with segment information
        """
        if segment_idx < 0 or segment_idx >= self.n_segments:
            raise ValueError(f"Segment index {segment_idx} out of range [0, {self.n_segments})")

        k_start, k_end = self.segments[segment_idx]

        info = {
            "segment_idx": segment_idx,
            "k_start": k_start,
            "k_end": k_end,
            "length": k_end - k_start,
            "timesteps": list(range(k_start, k_end + 1)),
        }

        return info

    def print_segment_summary(self):
        """Print summary of segmentation."""
        print("\n" + "=" * 70)
        print("SEGMENTATION SUMMARY")
        print("=" * 70)
        print(f"Total horizon:      {self.N}")
        print(f"Segment length:     {self.segment_length}")
        print(f"Overlap:            {self.overlap}")
        print(f"Number of segments: {self.n_segments}")
        print("\nSegment boundaries:")
        for i, (k_start, k_end) in enumerate(self.segments):
            print(f"  Segment {i}: [{k_start}, {k_end}] ({k_end - k_start} timesteps)")
        print("=" * 70)

    def validate_segmentation(self) -> Tuple[bool, str]:
        """
        Validate segmentation covers entire trajectory.

        Returns:
            valid: True if valid
            message: Validation message
        """
        # Check coverage
        covered = set()
        for k_start, k_end in self.segments:
            covered.update(range(k_start, k_end))

        expected = set(range(self.N))

        if covered == expected:
            return True, "Segmentation valid: covers entire trajectory"
        elif covered.issuperset(expected):
            return True, "Segmentation valid: covers entire trajectory with overlap"
        else:
            missing = expected - covered
            return False, f"Segmentation invalid: missing timesteps {sorted(missing)}"

    def __repr__(self) -> str:
        return (
            f"TrajectorySegmenter(N={self.N}, segment_length={self.segment_length}, "
            f"overlap={self.overlap}, n_segments={self.n_segments})"
        )
