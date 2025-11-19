"""
State and input constraint handling for trajectory optimization.

This module provides utilities for:
- Box constraints on states and inputs
- Time-varying constraints
- Constraint tightening for robustness
- Constraint validation
"""

from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
from scipy.stats import norm


class BoxConstraints:
    """
    Box constraints: lower <= variable <= upper

    Used for state and input bounds.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray, name: str = "constraint"):
        """
        Initialize box constraints.

        Args:
            lower: Lower bounds (n,)
            upper: Upper bounds (n,)
            name: Name of constraint
        """
        assert len(lower) == len(upper), "Lower and upper bounds must have the same length"
        assert np.all(lower <= upper), "Lower bounds must be less than or equal to upper bounds"

        self.lower = np.array(lower, dtype=float)
        self.upper = np.array(upper, dtype=float)
        self.name = name
        self.dim = len(lower)

    def is_satisfied(self, x: np.ndarray, tolerance: float = 0.0) -> bool:
        """
        Check if point satisfies box constraints.

        Args:
            x: Point to check (n,)
            tolerance: Tolerance for floating point comparisons

        Returns:
            satisfied: True if point satisfies constraints, False otherwise
        """
        return np.all(x >= self.lower - tolerance) and np.all(x <= self.upper + tolerance)

    def project_onto_feasible(self, x: np.ndarray) -> np.ndarray:
        """
        Project poiny onto feasible region.

        Args:
            x: Point to project (n,)

        Returns:
            x_proj: Projected point (n,)
        """
        return np.clip(x, self.lower, self.upper)

    def build_cvxpy_constraints(self, var: cp.Variable, timestep: Optional[int] = None) -> List[cp.Constraint]:
        """
        Build CVXPY constraints.

        Args:
            var: Variable to apply constraints to
            timestep: Optional timestep index

        Returns:
            constraints: List of CVXPY constraints
        """
        if timestep is not None:
            return [
                var[timestep, :] >= self.lower,
                var[timestep, :] <= self.upper,
            ]
        else:
            return [
                var >= self.lower,
                var <= self.upper,
            ]

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds."""
        return self.lower.copy(), self.upper.copy()

    def compute_violation(self, x: np.ndarray) -> float:
        """
        Compute constraint violation.

        Args:
            x: Point to check (n,)

        Returns:
            violation: Maximum violation (0 if satisfied)
        """
        lower_violation = np.maximum(0, self.lower - x)
        upper_violation = np.maximum(0, x - self.upper)
        max_violation = np.maximum(np.max(lower_violation), np.max(upper_violation))
        return float(max_violation)

    def tighten(self, margin: np.ndarray) -> "BoxConstraints":
        """
        Tighten constraints by margin.

        Args:
            margin: Margin to tighten by (n,) or scalar

        Returns:
            tightened: New BoxConstraints object with tightened bounds
        """
        if np.isscalar(margin):
            margin = np.full(self.dim, margin)

        lower_tight = self.lower + margin
        upper_tight = self.upper - margin

        return BoxConstraints(lower_tight, upper_tight, name=f"{self.name}_tight")

    def __repr__(self) -> str:
        return f"BoxConstraints({self.name}, dim={self.dim})"


class TimeVaryingConstraints:
    """
    Time-varying constraints along a trajectory.

    Useful for state-dependent or time-dependent constraint tightening.
    """

    def __init__(self, constraints_sequence: List[BoxConstraints]):
        """
        Initialize time-varying constraints.

        Args:
            constraints_sequence: List of BoxConstraints objects, one for each timestep
        """
        self.constraints_sequence = constraints_sequence
        self.N = len(constraints_sequence) - 1

    def get_constraints_at_time(self, k: int) -> BoxConstraints:
        """
        Get constraints at timestep k.

        Args:
            k: Timestep index

        Returns:
            constraints: BoxConstraints object at timestep k
        """
        return self.constraints_sequence[k]

    def is_trajectory_satisfied(self, x_traj: np.ndarray, tolerance: float = 0.0) -> Tuple[bool, List[int]]:
        """
        Check if trajectory satifies time-varying constraints.

        Args:
            x_traj: Trajectory (N+1, n)
            tolerance: Tolerance for floating point comparisons

        Returns:
            satisfied: True if trajectory satisfies constraints, False otherwise
            violations: List of timesteps where constraints are violated
        """
        violations = []

        for k in range(len(self.constraints_sequence)):
            if not self.constraints_sequence[k].is_satisfied(x_traj[k], tolerance):
                violations.append(k)

        return len(violations) == 0, violations

    def build_cvxpy_constraints(self, var: cp.Variable) -> List[cp.Constraint]:
        """
        Build CVXPY constraints for entire trajectory.

        Args:
            var: Variable to apply constraints to

        Returns:
            constraints: List of all CVXPY constraints
        """
        constraints = []

        for k, box_constraints in enumerate(self.constraints_sequence):
            constraints.extend(box_constraints.build_cvxpy_constraints(var, timestep=k))

        return constraints


class ConstraintTightening:
    """
    Constraint tightening for robust trajectory optimization.

    Tighten constraints to account for:
        - Tracking error
        - Model uncertainty
        - Disturbances
    """

    @staticmethod
    def compute_tightening_from_deviation(
        x_nominal: np.ndarray, x_actual: np.ndarray, percentile: float = 95.0
    ) -> np.ndarray:
        """
        Compute constraint tightening based on observed deviations.

        Args:
            x_nominal: Nominal state trajectory (N+1, n)
            x_actual: Actual state trajectory (N+1, n)
            percentile: Percentile of deviations to use for tightening

        Returns:
            tightening: Tightening margins (n,)
        """
        deviations = np.abs(x_actual - x_nominal)
        tightening = np.percentile(deviations, percentile, axis=0)
        return tightening

    @staticmethod
    def compute_tightening_from_uncertainty(sigma: np.ndarray, confidence: float = 0.99) -> np.ndarray:
        """
        Compute constraint tightening from uncertainty bounds.

        Assumes Gaussian uncertainty with standard deviation sigma.
        Uses confidence level to determine tightening margin.

        Args:
            sigma: Standard deviation (n,)
            confidence: Confidence level (0-1)

        Returns:
            tightening: Tightening margins (n,)
        """

        z_score = norm.ppf((1 + confidence) / 2)
        return z_score * sigma

    @staticmethod
    def tighten_trajectory_constraints(
        base_constraints: BoxConstraints, tightening_margin: np.ndarray, N: int
    ) -> TimeVaryingConstraints:
        """
        Create time-varying tightened constraints.

        Args:
            base_constraints: Base BoxConstraints object
            tightening_margin: Tightening margins (n,)
            N: Number of timesteps

        Returns:
            time_varying: TimeVaryingConstraints object with tightened constraints
        """
        constraints_sequence = []

        for k in range(N + 1):
            tight_k = base_constraints.tighten(tightening_margin)
            constraints_sequence.append(tight_k)

        return TimeVaryingConstraints(constraints_sequence)


class ConstraintValidator:
    """
    Validate constraints along trajectories.
    """

    @staticmethod
    def validate_state_trajectory(
        x_traj: np.ndarray, constraints: BoxConstraints, tolerance: float = 0.0
    ) -> Tuple[bool, Dict]:
        """
        Validate state trajectory against box constraints.

        Args:
            x_traj: State trajectory (N+1, n)
            constraints: BoxConstraints object
            tolerance: Tolerance for floating point comparisons

        Returns:
            valid: True if trajectory satisfies constraints, False otherwise
            report: Validation report with violations
        """
        violations = []
        max_violation = 0.0

        for k in range(x_traj.shape[0]):
            viol = constraints.compute_violation(x_traj[k])
            if viol > tolerance:
                violations.append((k, viol))
                max_violation = max(max_violation, viol)

        report = {
            "valid": len(violations) == 0,
            "num_violations": len(violations),
            "violation_timesteps": [v[0] for v in violations],
            "max_violation": max_violation,
        }
        return len(violations) == 0, report

    @staticmethod
    def validate_input_trajectory(
        u_traj: np.ndarray, constraints: BoxConstraints, tolerance: float = 0.0
    ) -> Tuple[bool, Dict]:
        """
        Validate input trajectory against box constraints.

        Args:
            u_traj: Input trajectory (N, m)
            constraints: BoxConstraints object
            tolerance: Tolerance for floating point comparisons

        Returns:
            valid: True if trajectory satisfies constraints, False otherwise
            report: Validation report with violations
        """
        return ConstraintValidator.validate_state_trajectory(u_traj, constraints, tolerance)

    @staticmethod
    def compute_constraint_margin(x_traj: np.ndarray, constraints: BoxConstraints) -> Dict[str, np.ndarray]:
        """
        Compute margin to constraints along trajectory.

        Args:
            x_traj: State trajectory (N+1, n)
            constraints: BoxConstraints object

        Returns:
            margin: Dict with:
                - 'lower': Distance to lower bound (N+1, n)
                - 'upper': Distance to upper bound (N+1, n)
                - 'min': Minimum margin at each timestep (N+1,)
        """
        lower_bounds, upper_bounds = constraints.get_bounds()

        lower_margin = x_traj - lower_bounds
        upper_margin = upper_bounds - x_traj

        min_margin = np.minimum(
            np.min(lower_margin, axis=1),
            np.min(upper_margin, axis=1),
        )
        return {
            "lower": lower_margin,
            "upper": upper_margin,
            "min": min_margin,
        }


class ConstraintVisualization:
    """
    Visualization utilities for constraints.
    """

    @staticmethod
    def get_constraint_boundaries_2d(constraints: BoxConstraints) -> Dict[str, np.ndarray]:
        """
        Get 2D constraint boundaries for plotting.

        Args:
            constraints: BoxConstraints object

        Returns:
            boundaries: Dict with rectangle corners
        """
        lower, upper = constraints.get_bounds()

        # Get rectangle corners
        x_rect = [lower[0], upper[0], upper[0], lower[0], lower[0]]
        y_rect = [lower[1], lower[1], upper[1], upper[1], lower[1]]
        return {
            "x": np.array(x_rect),
            "y": np.array(y_rect),
        }
