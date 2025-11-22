"""Uncertainty constants and bounds.

This module provides the UncertaintyConstants dataclass for storing
uncertainty bounds, confidence levels, and other constants used in
uncertainty quantification.
"""

from dataclasses import dataclass


@dataclass
class UncertaintyConstants:
    """UncertaintyConstants dataclass for uncertainty quantification parameters.
    
    This dataclass stores constants and bounds used in uncertainty quantification,
    including confidence levels, uncertainty bounds per segment, and other
    parameters needed for robust control design.
    """
    pass

