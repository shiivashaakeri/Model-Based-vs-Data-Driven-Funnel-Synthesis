"""
Constraint Definitions for DDFS.

This module provides classes for representing:
- Box constraints on states and inputs
- Soft constraint wrappers with slack variables
- Constraint checking and violation computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from ddfs.utils.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Box Constraints
# =============================================================================


@dataclass
class BoxConstraint:
    """
    Axis-aligned box constraint: lb <= x <= ub.

    Parameters
    ----------
    lb : np.ndarray
        Lower bounds for each dimension.
    ub : np.ndarray
        Upper bounds for each dimension.
    labels : list of str, optional
        Names for each dimension (for logging/display).
    """

    lb: np.ndarray
    ub: np.ndarray
    labels: Optional[List[str]] = None

    def __post_init__(self):
        """Validate and convert bounds to numpy arrays."""
        self.lb = np.asarray(self.lb, dtype=np.float64)
        self.ub = np.asarray(self.ub, dtype=np.float64)

        if self.lb.shape != self.ub.shape:
            raise ValueError(f"Lower and upper bounds must have same shape: {self.lb.shape} vs {self.ub.shape}")

        if not np.all(self.lb <= self.ub):
            violations = np.where(self.lb > self.ub)[0]
            raise ValueError(f"Lower bounds must be <= upper bounds. Violations at indices: {violations}")

        if self.labels is not None and len(self.labels) != len(self.lb):
            raise ValueError(f"Number of labels ({len(self.labels)}) must match dimension ({len(self.lb)})")

    @property
    def dim(self) -> int:
        """Dimension of the constraint."""
        return len(self.lb)

    @property
    def center(self) -> np.ndarray:
        """Center of the box."""
        return 0.5 * (self.lb + self.ub)

    @property
    def half_widths(self) -> np.ndarray:
        """Half-widths of the box in each dimension."""
        return 0.5 * (self.ub - self.lb)

    @property
    def widths(self) -> np.ndarray:
        """Full widths of the box in each dimension."""
        return self.ub - self.lb

    @property
    def volume(self) -> float:
        """Volume (or area in 2D) of the box."""
        return np.prod(self.widths)

    def contains(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        """
        Check if point x is inside the box.

        Parameters
        ----------
        x : np.ndarray
            Point to check.
        tol : float
            Tolerance for boundary.

        Returns
        -------
        bool
            True if x is inside or on boundary.
        """
        return np.all(x >= self.lb - tol) and np.all(x <= self.ub + tol)

    def contains_batch(self, X: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """
        Check if multiple points are inside the box.

        Parameters
        ----------
        X : np.ndarray
            Points to check, shape (n_points, dim).
        tol : float
            Tolerance for boundary.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_points,).
        """
        above_lb = np.all(self.lb - tol <= X, axis=1)
        below_ub = np.all(self.ub + tol >= X, axis=1)
        return above_lb & below_ub

    def violation(self, x: np.ndarray) -> np.ndarray:
        """
        Compute constraint violation for each dimension.

        Positive values indicate violation.

        Parameters
        ----------
        x : np.ndarray
            Point to check.

        Returns
        -------
        np.ndarray
            Violation for each dimension (0 if satisfied).
        """
        lower_violation = np.maximum(self.lb - x, 0)
        upper_violation = np.maximum(x - self.ub, 0)
        return lower_violation + upper_violation

    def max_violation(self, x: np.ndarray) -> float:
        """
        Compute maximum constraint violation across all dimensions.

        Parameters
        ----------
        x : np.ndarray
            Point to check.

        Returns
        -------
        float
            Maximum violation (0 if satisfied).
        """
        return np.max(self.violation(x))

    def total_violation(self, x: np.ndarray) -> float:
        """
        Compute total (sum) constraint violation.

        Parameters
        ----------
        x : np.ndarray
            Point to check.

        Returns
        -------
        float
            Sum of violations (0 if satisfied).
        """
        return np.sum(self.violation(x))

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project point onto the box (clamp to bounds).

        Parameters
        ----------
        x : np.ndarray
            Point to project.

        Returns
        -------
        np.ndarray
            Projected point.
        """
        return np.clip(x, self.lb, self.ub)

    def distance_to_boundary(self, x: np.ndarray) -> float:
        """
        Compute signed distance to box boundary.

        Negative inside, positive outside.

        Parameters
        ----------
        x : np.ndarray
            Point to check.

        Returns
        -------
        float
            Signed distance to boundary.
        """
        # Distance to each face
        dist_to_lb = x - self.lb
        dist_to_ub = self.ub - x

        # Minimum distance to any face (negative if outside)
        min_dist_inside = np.min(np.minimum(dist_to_lb, dist_to_ub))

        if min_dist_inside >= 0:
            # Inside: return negative of distance to nearest face
            return -min_dist_inside
        else:
            # Outside: return positive distance
            return self.max_violation(x)

    def sample_uniform(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample uniformly from the box.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        np.ndarray
            Samples of shape (n_samples, dim).
        """
        return np.random.uniform(self.lb, self.ub, size=(n_samples, self.dim))

    def vertices(self) -> np.ndarray:
        """
        Get all vertices of the box.

        Returns
        -------
        np.ndarray
            Vertices of shape (2^dim, dim).
        """
        from itertools import product  # noqa: PLC0415

        vertices = []
        for corner in product([0, 1], repeat=self.dim):
            vertex = np.where(corner, self.ub, self.lb)
            vertices.append(vertex)
        return np.array(vertices)

    def shrink(self, margin: Union[float, np.ndarray]) -> "BoxConstraint":
        """
        Create a shrunk box with given margin.

        Parameters
        ----------
        margin : float or np.ndarray
            Margin to shrink by (scalar or per-dimension).

        Returns
        -------
        BoxConstraint
            Shrunk box constraint.
        """
        margin = np.asarray(margin)
        return BoxConstraint(
            lb=self.lb + margin,
            ub=self.ub - margin,
            labels=self.labels,
        )

    def expand(self, margin: Union[float, np.ndarray]) -> "BoxConstraint":
        """
        Create an expanded box with given margin.

        Parameters
        ----------
        margin : float or np.ndarray
            Margin to expand by (scalar or per-dimension).

        Returns
        -------
        BoxConstraint
            Expanded box constraint.
        """
        margin = np.asarray(margin)
        return BoxConstraint(
            lb=self.lb - margin,
            ub=self.ub + margin,
            labels=self.labels,
        )

    def to_polytope_form(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to polytope form: Ax <= b.

        Returns
        -------
        A : np.ndarray
            Constraint matrix of shape (2*dim, dim).
        b : np.ndarray
            Constraint vector of shape (2*dim,).
        """
        n = self.dim
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.hstack([self.ub, -self.lb])
        return A, b

    def __repr__(self) -> str:
        return f"BoxConstraint(dim={self.dim}, lb={self.lb}, ub={self.ub})"


# =============================================================================
# State and Input Constraints
# =============================================================================


@dataclass
class StateConstraint(BoxConstraint):
    """
    Box constraint specifically for state variables.

    Inherits all functionality from BoxConstraint with
    state-specific metadata.
    """

    n_states: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.n_states = self.dim


@dataclass
class InputConstraint(BoxConstraint):
    """
    Box constraint specifically for input variables.

    Inherits all functionality from BoxConstraint with
    input-specific metadata.
    """

    n_inputs: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.n_inputs = self.dim


# =============================================================================
# Combined State-Input Constraints
# =============================================================================


@dataclass
class StateInputConstraints:
    """
    Combined state and input box constraints.

    Parameters
    ----------
    state_constraint : StateConstraint or BoxConstraint
        Constraint on state variables.
    input_constraint : InputConstraint or BoxConstraint
        Constraint on input variables.
    """

    state_constraint: BoxConstraint
    input_constraint: BoxConstraint

    @property
    def n_states(self) -> int:
        """Number of state dimensions."""
        return self.state_constraint.dim

    @property
    def n_inputs(self) -> int:
        """Number of input dimensions."""
        return self.input_constraint.dim

    @property
    def x_min(self) -> np.ndarray:
        """State lower bounds."""
        return self.state_constraint.lb

    @property
    def x_max(self) -> np.ndarray:
        """State upper bounds."""
        return self.state_constraint.ub

    @property
    def u_min(self) -> np.ndarray:
        """Input lower bounds."""
        return self.input_constraint.lb

    @property
    def u_max(self) -> np.ndarray:
        """Input upper bounds."""
        return self.input_constraint.ub

    def check_state(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        """Check if state satisfies constraints."""
        return self.state_constraint.contains(x, tol)

    def check_input(self, u: np.ndarray, tol: float = 1e-9) -> bool:
        """Check if input satisfies constraints."""
        return self.input_constraint.contains(u, tol)

    def check(self, x: np.ndarray, u: np.ndarray, tol: float = 1e-9) -> bool:
        """Check if both state and input satisfy constraints."""
        return self.check_state(x, tol) and self.check_input(u, tol)

    def state_violation(self, x: np.ndarray) -> np.ndarray:
        """Compute state constraint violation."""
        return self.state_constraint.violation(x)

    def input_violation(self, u: np.ndarray) -> np.ndarray:
        """Compute input constraint violation."""
        return self.input_constraint.violation(u)

    def max_violation(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute maximum violation across state and input."""
        return max(
            self.state_constraint.max_violation(x),
            self.input_constraint.max_violation(u),
        )

    def project_state(self, x: np.ndarray) -> np.ndarray:
        """Project state onto feasible region."""
        return self.state_constraint.project(x)

    def project_input(self, u: np.ndarray) -> np.ndarray:
        """Project input onto feasible region."""
        return self.input_constraint.project(u)

    def shrink(
        self,
        state_margin: Union[float, np.ndarray] = 0.0,
        input_margin: Union[float, np.ndarray] = 0.0,
    ) -> "StateInputConstraints":
        """
        Create shrunk constraints with given margins.

        Parameters
        ----------
        state_margin : float or np.ndarray
            Margin to shrink state bounds.
        input_margin : float or np.ndarray
            Margin to shrink input bounds.

        Returns
        -------
        StateInputConstraints
            Shrunk constraints.
        """
        return StateInputConstraints(
            state_constraint=self.state_constraint.shrink(state_margin),
            input_constraint=self.input_constraint.shrink(input_margin),
        )

    @classmethod
    def from_bounds(
        cls,
        x_min: np.ndarray,
        x_max: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        state_labels: Optional[List[str]] = None,
        input_labels: Optional[List[str]] = None,
    ) -> "StateInputConstraints":
        """
        Create from explicit bounds arrays.

        Parameters
        ----------
        x_min, x_max : np.ndarray
            State bounds.
        u_min, u_max : np.ndarray
            Input bounds.
        state_labels : list of str, optional
            State variable names.
        input_labels : list of str, optional
            Input variable names.

        Returns
        -------
        StateInputConstraints
            Combined constraint object.
        """
        return cls(
            state_constraint=BoxConstraint(lb=x_min, ub=x_max, labels=state_labels),
            input_constraint=BoxConstraint(lb=u_min, ub=u_max, labels=input_labels),
        )

    def __repr__(self) -> str:
        return (
            f"StateInputConstraints(\n"
            f"  states: {self.n_states}D, x ∈ [{self.x_min}, {self.x_max}]\n"
            f"  inputs: {self.n_inputs}D, u ∈ [{self.u_min}, {self.u_max}]\n"
            f")"
        )


# =============================================================================
# Soft Constraints
# =============================================================================


@dataclass
class SoftBoxConstraint:
    """
    Soft box constraint with slack variables.

    Transforms hard constraint lb <= x <= ub into:
        lb - s_l <= x <= ub + s_u
        s_l >= 0, s_u >= 0

    with penalty weight on slack variables.

    Parameters
    ----------
    hard_constraint : BoxConstraint
        The underlying hard constraint.
    weight : float
        Penalty weight for constraint violation.
    slack_type : str
        Type of slack: 'l1' (linear), 'l2' (quadratic), or 'linf' (max).
    """

    hard_constraint: BoxConstraint
    weight: float = 1e3
    slack_type: str = "l2"  # 'l1', 'l2', or 'linf'

    def __post_init__(self):
        if self.slack_type not in ["l1", "l2", "linf"]:
            raise ValueError(f"slack_type must be 'l1', 'l2', or 'linf', got {self.slack_type}")

    @property
    def dim(self) -> int:
        """Dimension of the constraint."""
        return self.hard_constraint.dim

    @property
    def lb(self) -> np.ndarray:
        """Lower bounds (from hard constraint)."""
        return self.hard_constraint.lb

    @property
    def ub(self) -> np.ndarray:
        """Upper bounds (from hard constraint)."""
        return self.hard_constraint.ub

    def compute_slack(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute required slack variables for given point.

        Parameters
        ----------
        x : np.ndarray
            Point to evaluate.

        Returns
        -------
        s_lower : np.ndarray
            Slack for lower bound violations.
        s_upper : np.ndarray
            Slack for upper bound violations.
        """
        s_lower = np.maximum(self.lb - x, 0)
        s_upper = np.maximum(x - self.ub, 0)
        return s_lower, s_upper

    def penalty(self, x: np.ndarray) -> float:
        """
        Compute penalty for constraint violation.

        Parameters
        ----------
        x : np.ndarray
            Point to evaluate.

        Returns
        -------
        float
            Penalty value.
        """
        s_lower, s_upper = self.compute_slack(x)
        total_slack = np.concatenate([s_lower, s_upper])

        if self.slack_type == "l1":
            return self.weight * np.sum(total_slack)
        elif self.slack_type == "l2":
            return self.weight * np.sum(total_slack**2)
        else:  # linf
            return self.weight * np.max(total_slack)

    def penalty_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of penalty with respect to x.

        Parameters
        ----------
        x : np.ndarray
            Point to evaluate.

        Returns
        -------
        np.ndarray
            Gradient of penalty.
        """
        grad = np.zeros(self.dim)
        s_lower, s_upper = self.compute_slack(x)

        if self.slack_type == "l1":
            # Subgradient
            grad -= self.weight * (s_lower > 0).astype(float)
            grad += self.weight * (s_upper > 0).astype(float)
        elif self.slack_type == "l2":
            grad -= 2 * self.weight * s_lower
            grad += 2 * self.weight * s_upper
        else:  # linf
            total_slack = np.concatenate([s_lower, s_upper])
            max_idx = np.argmax(total_slack)
            if max_idx < self.dim:
                grad[max_idx] = -self.weight
            else:
                grad[max_idx - self.dim] = self.weight

        return grad

    def is_satisfied(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        """Check if hard constraint is satisfied."""
        return self.hard_constraint.contains(x, tol)


@dataclass
class SoftStateInputConstraints:
    """
    Soft constraints for both state and input.

    Parameters
    ----------
    hard_constraints : StateInputConstraints
        The underlying hard constraints.
    state_weight : float
        Penalty weight for state violations.
    input_weight : float
        Penalty weight for input violations.
    slack_type : str
        Type of slack penalty.
    """

    hard_constraints: StateInputConstraints
    state_weight: float = 1e3
    input_weight: float = 1e3
    slack_type: str = "l2"

    def __post_init__(self):
        self.soft_state = SoftBoxConstraint(
            hard_constraint=self.hard_constraints.state_constraint,
            weight=self.state_weight,
            slack_type=self.slack_type,
        )
        self.soft_input = SoftBoxConstraint(
            hard_constraint=self.hard_constraints.input_constraint,
            weight=self.input_weight,
            slack_type=self.slack_type,
        )

    def penalty(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute total penalty for state and input violations."""
        return self.soft_state.penalty(x) + self.soft_input.penalty(u)

    def state_penalty(self, x: np.ndarray) -> float:
        """Compute penalty for state violations only."""
        return self.soft_state.penalty(x)

    def input_penalty(self, u: np.ndarray) -> float:
        """Compute penalty for input violations only."""
        return self.soft_input.penalty(u)

    def is_satisfied(self, x: np.ndarray, u: np.ndarray, tol: float = 1e-9) -> bool:
        """Check if hard constraints are satisfied."""
        return self.soft_state.is_satisfied(x, tol) and self.soft_input.is_satisfied(u, tol)


# =============================================================================
# Constraint Factory Functions
# =============================================================================


def create_box_constraint(
    lb: Union[float, List[float], np.ndarray],
    ub: Union[float, List[float], np.ndarray],
    dim: Optional[int] = None,
    labels: Optional[List[str]] = None,
) -> BoxConstraint:
    """
    Create a box constraint with flexible input formats.

    Parameters
    ----------
    lb : float, list, or np.ndarray
        Lower bounds. If scalar and dim provided, broadcast to all dimensions.
    ub : float, list, or np.ndarray
        Upper bounds. If scalar and dim provided, broadcast to all dimensions.
    dim : int, optional
        Dimension (required if lb/ub are scalars).
    labels : list of str, optional
        Labels for each dimension.

    Returns
    -------
    BoxConstraint
        Created constraint.
    """
    # Handle scalar inputs
    if np.isscalar(lb):
        if dim is None:
            raise ValueError("dim must be provided if lb is scalar")
        lb = np.full(dim, lb)

    if np.isscalar(ub):
        if dim is None:
            raise ValueError("dim must be provided if ub is scalar")
        ub = np.full(dim, ub)

    return BoxConstraint(lb=np.asarray(lb), ub=np.asarray(ub), labels=labels)


def constraints_from_config(config) -> StateInputConstraints:
    """
    Create constraints from a configuration object.

    Parameters
    ----------
    config : Config
        Configuration object with system.bounds.

    Returns
    -------
    StateInputConstraints
        Created constraints.
    """
    bounds = config.system.bounds

    return StateInputConstraints.from_bounds(
        x_min=np.array(bounds.x_min),
        x_max=np.array(bounds.x_max),
        u_min=np.array(bounds.u_min),
        u_max=np.array(bounds.u_max),
        state_labels=config.system.state_labels,
        input_labels=config.system.input_labels,
    )
