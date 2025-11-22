"""Funnel synthesis orchestrator for Phase 5.

This module provides the main FunnelSynthesizer class that orchestrates
the complete funnel synthesis pipeline, integrating LMI construction,
SDP solving, and funnel library generation.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ddfs.data_collection.hankel import SegmentHankelMatrices
from ddfs.feasibility.ellipsoid_solver import EllipsoidParams
from ddfs.synthesis.lmi_builder import LMIBuilder
from ddfs.synthesis.sdp_solver import SDPSolution, SDPSolver
from ddfs.uncertainty.constants import UncertaintyConstants

logger = logging.getLogger(__name__)


@dataclass
class FunnelSegment:
    """Single funnel segment container.

    Attributes:
        segment_idx: Segment index
        P: Shape matrix (nxn)
        K: Feedback gain matrix (mxn)
        c: Center point (n,)
        volume: Ellipsoid volume
        nu: Lyapunov decrease rate
        lambda1: S-procedure multiplier 1
        lambda2: S-procedure multiplier 2
    """

    segment_idx: int
    P: NDArray[np.float64]
    K: NDArray[np.float64]
    c: NDArray[np.float64]
    volume: float
    nu: float
    lambda1: float
    lambda2: float

    def contains(self, x: NDArray[np.float64]) -> bool:
        """Check if state x is inside the funnel.

        Args:
            x: State vector (n,)

        Returns:
            True if (x - c)^T P^{-1} (x - c) <= 1
        """
        diff = x - self.c
        P_inv = np.linalg.inv(self.P)
        return float(diff.T @ P_inv @ diff) <= 1.0

    def __repr__(self) -> str:
        return f"FunnelSegment(idx={self.segment_idx}, volume={self.volume:.6e})"


@dataclass
class FunnelLibrary:
    """Complete funnel library for trajectory tracking.

    Contains all funnel segments synthesized from Phase 5, ready for
    deployment in Phase 6 (tracking controller).

    Attributes:
        segments: List of FunnelSegment objects
        segment_indices: Segment indices
        n: State dimension
        m: Control dimension
        alpha: Lyapunov decay rate
        mu: Funnel growth bound
    """

    segments: List[FunnelSegment]
    segment_indices: List[int]
    n: int
    m: int
    alpha: float
    mu: float

    def __post_init__(self):
        """Validate library."""
        if len(self.segments) != len(self.segment_indices):
            raise ValueError(
                f"segments length {len(self.segments)} != segment_indices length {len(self.segment_indices)}"
            )

    @property
    def num_segments(self) -> int:
        """Number of segments."""
        return len(self.segments)

    def get_segment(self, idx: int) -> FunnelSegment:
        """Get funnel segment by index.

        Args:
            idx: Segment index

        Returns:
            FunnelSegment
        """
        pos = self.segment_indices.index(idx)
        return self.segments[pos]

    def get_gain(self, segment_idx: int) -> NDArray[np.float64]:
        """Get feedback gain K for a segment.

        Args:
            segment_idx: Segment index

        Returns:
            Feedback gain K (mxn)
        """
        return self.get_segment(segment_idx).K

    def summary(self) -> str:
        """Generate summary string.

        Returns:
            Human-readable summary
        """
        lines = [
            "=" * 70,
            "FUNNEL LIBRARY",
            "=" * 70,
            "",
            f"State dimension (n): {self.n}",
            f"Control dimension (m): {self.m}",
            f"Number of segments: {self.num_segments}",
            f"Lyapunov decay rate (alpha): {self.alpha}",
            f"Funnel growth bound (μ): {self.mu}",
            "",
            "Segment details:",
        ]

        volumes = [seg.volume for seg in self.segments]

        for seg in self.segments:
            lines.append(f"  Segment {seg.segment_idx}: volume={seg.volume:.6e}, nu={seg.nu:.6e}")

        lines.extend(
            [
                "",
                "Volume statistics:",
                f"  Min:  {np.min(volumes):.6e}",
                f"  Max:  {np.max(volumes):.6e}",
                f"  Mean: {np.mean(volumes):.6e}",
                "=" * 70,
            ]
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"FunnelLibrary(segments={self.num_segments}, n={self.n}, m={self.m})"


class FunnelSynthesizer:
    """Main orchestrator for funnel synthesis.

    Coordinates the complete Phase 5 pipeline:
    1. Initialize LMI builder and SDP solver
    2. Extract centers from nominal trajectory
    3. Solve SDP sequence for all segments
    4. Build funnel library for deployment

    Example:
        >>> from ddfs.synthesis import FunnelSynthesizer
        >>> from ddfs.core.config import DDFSConfig
        >>>
        >>> # Setup
        >>> config = DDFSConfig(...)
        >>> synthesizer = FunnelSynthesizer(
        ...     n=3, m=2,
        ...     alpha=0.95,
        ...     mu=1.1,
        ... )
        >>>
        >>> # Synthesize funnels
        >>> library = synthesizer.synthesize(
        ...     P_min_list=feasibility_ellipsoids,
        ...     R_max_list=control_constraints,
        ...     centers=nominal_trajectory_states,
        ...     constants=uncertainty_constants,
        ...     hankel_list=hankel_matrices,
        ...     T_list=segment_lengths,
        ... )
        >>>
        >>> # Use library
        >>> print(library.summary())
        >>> K_0 = library.get_gain(segment_idx=0)
    """

    def __init__(
        self,
        n: int,
        m: int,
        alpha: float = 0.95,
        mu: float = 1.1,
        solver_name: str = "SCS",
        verbose: bool = False,
    ):
        """Initialize funnel synthesizer.

        Args:
            n: State dimension
            m: Control dimension
            alpha: Lyapunov decay rate (0 < alpha < 1)
            mu: Funnel growth bound (typically > 1)
            solver_name: CVXPY solver (SCS, MOSEK, CVXOPT)
            verbose: Enable verbose output
        """
        self.n = n
        self.m = m
        self.alpha = alpha
        self.mu = mu
        self.solver_name = solver_name
        self.verbose = verbose

        # Create LMI builder
        self.lmi_builder = LMIBuilder(n, m, alpha)

        # Create SDP solver
        self.sdp_solver = SDPSolver(self.lmi_builder, solver_name, verbose)

        logger.info("Initialized FunnelSynthesizer")
        logger.info(f"  State dimension: {n}")
        logger.info(f"  Control dimension: {m}")
        logger.info(f"  Lyapunov decay rate alpha: {alpha}")
        logger.info(f"  Funnel growth bound μ: {mu}")
        logger.info(f"  SDP solver: {solver_name}")

    def synthesize(
        self,
        P_min_list: List[NDArray[np.float64]],
        R_max_list: List[NDArray[np.float64]],
        centers: List[NDArray[np.float64]],
        constants: UncertaintyConstants,
        hankel_list: List[SegmentHankelMatrices],
        T_list: List[int],
        segment_indices: Optional[List[int]] = None,
    ) -> FunnelLibrary:
        """Synthesize complete funnel library.

        Main method that orchestrates the entire synthesis pipeline:
        1. Validate inputs
        2. Solve SDP for all segments
        3. Build funnel library

        Args:
            P_min_list: Minimum feasible shape matrices from Phase 4 (one per segment)
            R_max_list: Control constraint matrices (one per segment)
            centers: Nominal trajectory center points (one per segment)
            constants: Uncertainty constants from Phase 3
            hankel_list: Hankel matrices from Phase 2 (one per segment)
            T_list: Segment lengths (one per segment)
            segment_indices: Optional explicit segment indices

        Returns:
            FunnelLibrary ready for deployment

        Example:
            >>> library = synthesizer.synthesize(
            ...     P_min_list=[P_min_0, P_min_1, ...],
            ...     R_max_list=[R_max_0, R_max_1, ...],
            ...     centers=[c_0, c_1, ...],
            ...     constants=uncertainty_constants,
            ...     hankel_list=[H_0, H_1, ...],
            ...     T_list=[100, 100, ...],
            ... )
        """
        logger.info("=" * 70)
        logger.info("PHASE 5: FUNNEL SYNTHESIS")
        logger.info("=" * 70)

        # Determine number of segments
        num_segments = len(P_min_list)

        # Validate inputs
        logger.info("\n[1/4] Validating inputs...")
        self._validate_inputs(P_min_list, R_max_list, centers, hankel_list, T_list, num_segments)
        logger.info("  ✓ All inputs validated")
        logger.info(f"  Number of segments: {num_segments}")

        # Generate segment indices if not provided
        if segment_indices is None:
            segment_indices = list(range(num_segments))

        # Solve SDP sequence
        logger.info("\n[2/4] Solving SDP sequence...")
        solutions = self.sdp_solver.solve_sequence(
            segment_indices=segment_indices,
            P_min_list=P_min_list,
            R_max_list=R_max_list,
            constants=constants,
            hankel_list=hankel_list,
            T_list=T_list,
            mu=self.mu,
        )
        logger.info(f"  ✓ Solved {len(solutions)} segments")

        # Build funnel segments
        logger.info("\n[3/4] Building funnel library...")
        funnel_segments = []

        for i, solution in enumerate(solutions):
            seg = FunnelSegment(
                segment_idx=solution.segment_idx,
                P=solution.P_i,
                K=solution.K_i,
                c=centers[i],
                volume=solution.volume,
                nu=solution.nu,
                lambda1=solution.lambda1,
                lambda2=solution.lambda2,
            )
            funnel_segments.append(seg)
            logger.debug(f"  Created FunnelSegment {solution.segment_idx}")

        # Create library
        library = FunnelLibrary(
            segments=funnel_segments,
            segment_indices=segment_indices,
            n=self.n,
            m=self.m,
            alpha=self.alpha,
            mu=self.mu,
        )

        logger.info("  ✓ Built funnel library with {library.num_segments} segments")

        # Validate solutions
        logger.info("\n[4/4] Validating solutions...")
        all_valid = self._validate_solutions(solutions, P_min_list, R_max_list)

        if all_valid:
            logger.info("  ✓ All solutions validated")
        else:
            logger.warning("  ⚠ Some solutions have constraint violations")

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("FUNNEL SYNTHESIS COMPLETE")
        logger.info("=" * 70)
        print("\n" + library.summary())

        return library

    def _validate_inputs(  # noqa: C901, PLR0912
        self,
        P_min_list: List[NDArray[np.float64]],
        R_max_list: List[NDArray[np.float64]],
        centers: List[NDArray[np.float64]],
        hankel_list: List[SegmentHankelMatrices],
        T_list: List[int],
        num_segments: int,
    ):
        """Validate all inputs have correct dimensions and lengths.

        Args:
            P_min_list: Minimum feasible matrices
            R_max_list: Control constraint matrices
            centers: Center points
            hankel_list: Hankel matrices
            T_list: Segment lengths
            num_segments: Expected number of segments

        Raises:
            ValueError: If validation fails
        """
        # Check lengths
        if len(R_max_list) != num_segments:
            raise ValueError(f"R_max_list length {len(R_max_list)} != {num_segments}")

        if len(centers) != num_segments:
            raise ValueError(f"centers length {len(centers)} != {num_segments}")

        if len(hankel_list) != num_segments:
            raise ValueError(f"hankel_list length {len(hankel_list)} != {num_segments}")

        if len(T_list) != num_segments:
            raise ValueError(f"T_list length {len(T_list)} != {num_segments}")

        # Check dimensions
        for i, P_min in enumerate(P_min_list):
            if P_min.shape != (self.n, self.n):
                raise ValueError(f"P_min_list[{i}] has shape {P_min.shape}, expected ({self.n}, {self.n})")

        for i, R_max in enumerate(R_max_list):
            if R_max.shape != (self.m, self.m):
                raise ValueError(f"R_max_list[{i}] has shape {R_max.shape}, expected ({self.m}, {self.m})")

        for i, center in enumerate(centers):
            if len(center) != self.n:
                raise ValueError(f"centers[{i}] has length {len(center)}, expected {self.n}")

        for i, hankel in enumerate(hankel_list):
            if hankel.state_dim != self.n:
                raise ValueError(f"hankel_list[{i}] state_dim {hankel.state_dim} != {self.n}")
            if hankel.input_dim != self.m:
                raise ValueError(f"hankel_list[{i}] input_dim {hankel.input_dim} != {self.m}")

    def _validate_solutions(
        self,
        solutions: List[SDPSolution],
        P_min_list: List[NDArray[np.float64]],
        R_max_list: List[NDArray[np.float64]],
    ) -> bool:
        """Validate all solutions satisfy constraints.

        Args:
            solutions: List of SDP solutions
            P_min_list: Minimum feasible matrices
            R_max_list: Control constraint matrices

        Returns:
            True if all solutions valid
        """
        all_valid = True
        P_prev = None

        for i, solution in enumerate(solutions):
            valid = self.sdp_solver.validate_solution(
                solution=solution,
                P_min_i=P_min_list[i],
                R_max_i=R_max_list[i],
                P_prev=P_prev,
                mu=self.mu,
            )

            if not valid:
                all_valid = False

            P_prev = solution.P_i

        return all_valid

    def synthesize_from_phase4(
        self,
        feasibility_ellipsoids: List[EllipsoidParams],
        R_max_list: List[NDArray[np.float64]],
        constants: UncertaintyConstants,
        hankel_list: List[SegmentHankelMatrices],
        T_list: List[int],
    ) -> FunnelLibrary:
        """Synthesize funnels directly from Phase 4 output.

        Convenience method that extracts P_min and centers from
        EllipsoidParams objects.

        Args:
            feasibility_ellipsoids: List of EllipsoidParams from Phase 4
            R_max_list: Control constraint matrices
            constants: Uncertainty constants from Phase 3
            hankel_list: Hankel matrices from Phase 2
            T_list: Segment lengths

        Returns:
            FunnelLibrary
        """
        # Extract P_min and centers from ellipsoids
        P_min_list = [ellipsoid.P for ellipsoid in feasibility_ellipsoids]
        centers = [ellipsoid.c for ellipsoid in feasibility_ellipsoids]
        segment_indices = [ellipsoid.segment_index for ellipsoid in feasibility_ellipsoids]

        return self.synthesize(
            P_min_list=P_min_list,
            R_max_list=R_max_list,
            centers=centers,
            constants=constants,
            hankel_list=hankel_list,
            T_list=T_list,
            segment_indices=segment_indices,
        )
