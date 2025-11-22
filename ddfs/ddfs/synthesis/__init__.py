"""Funnel synthesis module for DDFS Phase 5.

This module provides tools for synthesizing robust funnels through
SDP optimization with LMI constraints, integrating uncertainty bounds
from Phase 3 and data-driven matrices from Phase 2.

Key Components:
    - FunnelSynthesizer: Main orchestrator for funnel synthesis pipeline
    - SDPSolver: Solves SDP optimization problems
    - LMIBuilder: Constructs LMI constraints
    - FunnelLibrary: Container for synthesized funnel segments
    - FunnelSegment: Single funnel segment with shape matrix and feedback gain
    - SDPSolution: SDP optimization solution

Example:
    >>> from ddfs.synthesis import FunnelSynthesizer
    >>> from ddfs.uncertainty import UncertaintyConstants
    >>> from ddfs.data_collection import SegmentHankelMatrices
    >>>
    >>> # Setup synthesizer
    >>> synthesizer = FunnelSynthesizer(
    ...     n=3, m=2,
    ...     alpha=0.95,  # Lyapunov decay rate
    ...     mu=1.1,      # Funnel growth bound
    ... )
    >>>
    >>> # Synthesize funnels from Phase 2, 3, 4 data
    >>> library = synthesizer.synthesize(
    ...     P_min_list=feasibility_ellipsoids,  # From Phase 4
    ...     R_max_list=control_constraints,
    ...     centers=nominal_trajectory_centers,
    ...     constants=uncertainty_constants,    # From Phase 3
    ...     hankel_list=hankel_matrices,        # From Phase 2
    ...     T_list=segment_lengths,
    ... )
    >>>
    >>> # Use library for control
    >>> print(library.summary())
    >>> K_0 = library.get_gain(segment_idx=0)
    >>> u = K_0 @ (x - x_nom)  # Feedback control

Complete Pipeline Example:
    >>> from ddfs.core.config import DDFSConfig
    >>> from ddfs.models import UnicyclePlant, UnicycleTwin
    >>> from ddfs.planning import SCvxPlanner
    >>> from ddfs.data_collection import DataCollector, HankelMatrixBuilder
    >>> from ddfs.uncertainty import UncertaintyQuantifier
    >>> from ddfs.feasibility import EllipsoidSolver
    >>> from ddfs.synthesis import FunnelSynthesizer
    >>>
    >>> # Phase 1: Nominal planning
    >>> config = DDFSConfig(...)
    >>> planner = SCvxPlanner(config)
    >>> nominal_traj = planner.plan(x0, xf, obstacles)
    >>>
    >>> # Phase 2: Data collection
    >>> plant = UnicyclePlant(config)
    >>> twin = UnicycleTwin(config)
    >>> collector = DataCollector(plant, nominal_traj, config)
    >>> trajectories = collector.collect_trials()
    >>>
    >>> # Build Hankel matrices
    >>> hankel_builder = HankelMatrixBuilder()
    >>> hankel_list = hankel_builder.build_all_segments(segmented_data)
    >>>
    >>> # Phase 3: Uncertainty quantification
    >>> quantifier = UncertaintyQuantifier(config, plant, twin)
    >>> constants = quantifier.quantify_to_constants(
    ...     states, controls, segment_indices
    ... )
    >>>
    >>> # Phase 4: Feasibility envelopes
    >>> ellipsoid_solver = EllipsoidSolver(config, obstacles)
    >>> feasibility_ellipsoids = [
    ...     ellipsoid_solver.solve_mvie(P_0, P_init, beta_i)
    ...     for P_0, P_init, beta_i in zip(...)
    ... ]
    >>>
    >>> # Phase 5: Funnel synthesis
    >>> synthesizer = FunnelSynthesizer(n=3, m=2, alpha=0.95)
    >>> library = synthesizer.synthesize_from_phase4(
    ...     feasibility_ellipsoids=feasibility_ellipsoids,
    ...     R_max_list=control_constraints,
    ...     constants=constants,
    ...     hankel_list=hankel_list,
    ...     T_list=segment_lengths,
    ... )
    >>>
    >>> # Phase 6: Deploy with tracking controller
    >>> # (Controller implementation would use library.get_gain())
"""

from ddfs.synthesis.funnel_synthesizer import (
    FunnelLibrary,
    FunnelSegment,
    FunnelSynthesizer,
)
from ddfs.synthesis.lmi_builder import LMIBuilder
from ddfs.synthesis.sdp_solver import SDPSolution, SDPSolver

__all__ = [
    # Funnel containers
    "FunnelLibrary",
    "FunnelSegment",
    # Main synthesizer
    "FunnelSynthesizer",
    # Building blocks
    "LMIBuilder",
    "SDPSolution",
    "SDPSolver",
]
