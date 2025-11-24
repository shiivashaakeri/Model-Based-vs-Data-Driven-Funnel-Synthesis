"""
DDFS: Data-Driven Funnel Synthesis for Online Tracking Control
==============================================================

A Python package for robust data-driven control of nonlinear systems
using digital twins and quadratic funnels.

This package implements the methodology for:
- Nominal trajectory planning using digital twin models
- Online data-driven uncertainty quantification
- Quadratic funnel synthesis via LMI/SDP optimization
- Robust tracking control with safety certificates

Main Components
---------------
- models: System dynamics (unicycle, quadrotor, etc.)
- planning: Trajectory optimization (SCvx, LQR)
- feasibility: Constraint envelope computation (MVIE)
- data_collection: Segment management and data matrices
- uncertainty: Lipschitz estimation and uncertainty sets
- synthesis: Funnel synthesis and SDP solvers
- visualization: Plotting utilities

Example
-------
>>> import ddfs
>>> print(ddfs.__version__)
0.1.0
"""

__version__ = "0.1.0"
__author__ = "Shiva Shakeri"
__email__ = "sshakeri@uw.edu"

# Version tuple for programmatic comparison
VERSION = tuple(map(int, __version__.split(".")))  # noqa: RUF048

# Package-level imports (populated as modules are created)
# These will be added in later steps as modules are implemented

__all__ = [
    "VERSION",
    "__author__",
    "__version__",
]
