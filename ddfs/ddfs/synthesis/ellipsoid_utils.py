"""
Ellipsoid utilities for quadratic funnel synthesis.

This module provides tools for computing and manipulating ellipsoids in the
context of Data-Driven Funnel Synthesis (DDFS). Key functionalities include:

1. Computing maximum-volume inscribed ellipsoids (MVIE) in polytopes
2. Computing P_min(k) and R_max(k) for state/input constraints
3. Segment-wise envelope computation (P_min,i and R_max,i)
4. Ellipsoid containment checking
5. Visualization of ellipsoids

Mathematical Background:
------------------------
A quadratic funnel for segment i is parameterized by:
- P_i ≻ 0: Positive definite matrix defining state deviation ellipsoid
- K_i: Feedback gain matrix

The state ellipsoid is:
    E(P_i) = {η ∈ ℝⁿ | η^T P_i η ≤ 1}

Under linear control ξ = K_i η, the input ellipsoid is:
    E_u(R_i) = {ξ ∈ ℝᵐ | ξ^T R_i^(-1) ξ ≤ 1}
where R_i = K_i P_i^(-1) K_i^T.

For feasibility, we need:
    E(P_i) ⊆ E(P_min,i)    (state constraints)
    E_u(R_i) ⊆ E_u(R_max,i) (input constraints)

where P_min(k) and R_max(k) are the largest inscribable ellipsoids at time k.
"""

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

@dataclass