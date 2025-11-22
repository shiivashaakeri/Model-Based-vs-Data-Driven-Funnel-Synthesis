"""Uncertainty quantification module for DDFS Phase 3.

This module provides tools for quantifying model uncertainty from collected
trajectory data, computing Lipschitz constants, and generating uncertainty
bounds for robust funnel synthesis.

Key Components:
    - UncertaintyQuantifier: Main class for computing uncertainty bounds
    - UncertaintyBounds: Container for computed bounds (legacy)
    - UncertaintyConstants: Complete container with all constants and diagnostics

Example:
    >>> from ddfs.uncertainty import UncertaintyQuantifier, UncertaintyConstants
    >>> from ddfs.core.config import DDFSConfig
    >>> from ddfs.models.unicycle import UnicyclePlant, UnicycleTwin
    >>>
    >>> config = DDFSConfig(...)
    >>> plant = UnicyclePlant(config)
    >>> twin = UnicycleTwin(config)
    >>> quantifier = UncertaintyQuantifier(config, plant, twin)
    >>>
    >>> # Compute uncertainty from collected data
    >>> constants = quantifier.quantify_to_constants(
    ...     states, controls, segment_indices
    ... )
    >>> print(constants.summary())
    >>> constants.save("uncertainty_constants.pkl")
"""

from ddfs.uncertainty.constants import UncertaintyConstants
from ddfs.uncertainty.quantifier import UncertaintyBounds, UncertaintyQuantifier

__all__ = [
    "UncertaintyBounds",
    "UncertaintyConstants",
    "UncertaintyQuantifier",
]
