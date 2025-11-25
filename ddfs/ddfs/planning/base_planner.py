"""
Base Planner and Optimization Problem Classes for DDFS.

This module provides abstract base classes for trajectory optimization:
- BasePlanner: Interface for trajectory planners
- OptimizationProblem: Single convex subproblem formulation
- Cost functions, constraints, and solver interfaces

Designed for CVXPY backend with support for:
- Hard and soft constraints
- Warm-starting
- Convergence tracking for iterative methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

from ddfs.core.constraints import StateInputConstraints
from ddfs.core.obstacles import ObstacleCollection
from ddfs.planning.trajectory import Trajectory
from ddfs.utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


# =============================================================================
# Enums and Status Types
# =============================================================================


class SolverStatus(Enum):
    """Optimization solver status."""

    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"
    INFEASIBLE = "infeasible"
    INFEASIBLE_INACCURATE = "infeasible_inaccurate"
    UNBOUNDED = "unbounded"
    UNBOUNDED_INACCURATE = "unbounded_inaccurate"
    SOLVER_ERROR = "solver_error"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"
    UNKNOWN = "unknown"

    @classmethod
    def from_cvxpy(cls, status: str) -> "SolverStatus":
        """Convert CVXPY status string to SolverStatus."""
        mapping = {
            cp.OPTIMAL: cls.OPTIMAL,
            cp.OPTIMAL_INACCURATE: cls.OPTIMAL_INACCURATE,
            cp.INFEASIBLE: cls.INFEASIBLE,
            cp.INFEASIBLE_INACCURATE: cls.INFEASIBLE_INACCURATE,
            cp.UNBOUNDED: cls.UNBOUNDED,
            cp.UNBOUNDED_INACCURATE: cls.UNBOUNDED_INACCURATE,
            cp.SOLVER_ERROR: cls.SOLVER_ERROR,
        }
        return mapping.get(status, cls.UNKNOWN)

    @property
    def is_optimal(self) -> bool:
        """Check if status indicates optimal solution."""
        return self in (SolverStatus.OPTIMAL, SolverStatus.OPTIMAL_INACCURATE)

    @property
    def is_feasible(self) -> bool:
        """Check if status indicates feasible solution."""
        return self in (
            SolverStatus.OPTIMAL,
            SolverStatus.OPTIMAL_INACCURATE,
            SolverStatus.MAX_ITERATIONS,
        )


class ConvergenceStatus(Enum):
    """Convergence status for iterative planners."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    INFEASIBLE = "infeasible"
    TRUST_REGION_FAILED = "trust_region_failed"
    COST_INCREASED = "cost_increased"
    NUMERICAL_ERROR = "numerical_error"
    NOT_STARTED = "not_started"


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass
class SubproblemResult:
    """Result from a single optimization subproblem."""

    status: SolverStatus
    cost: float
    solve_time: float
    x: Optional[np.ndarray] = None  # State trajectory (N+1, n)
    u: Optional[np.ndarray] = None  # Input trajectory (N, m)
    slack_state: Optional[np.ndarray] = None  # State constraint slack
    slack_input: Optional[np.ndarray] = None  # Input constraint slack
    slack_obstacle: Optional[np.ndarray] = None  # Obstacle constraint slack
    slack_dynamics: Optional[np.ndarray] = None  # Dynamics constraint slack
    dual_values: Optional[Dict[str, np.ndarray]] = None
    solver_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_solved(self) -> bool:
        """Check if problem was solved successfully."""
        return self.status.is_optimal and self.x is not None

    @property
    def total_slack(self) -> float:
        """Total slack violation."""
        total = 0.0
        if self.slack_state is not None:
            total += np.sum(np.abs(self.slack_state))
        if self.slack_input is not None:
            total += np.sum(np.abs(self.slack_input))
        if self.slack_obstacle is not None:
            total += np.sum(np.abs(self.slack_obstacle))
        if self.slack_dynamics is not None:
            total += np.sum(np.abs(self.slack_dynamics))
        return total


@dataclass
class PlannerResult:
    """Result from trajectory planner."""

    trajectory: Optional[Trajectory]
    status: ConvergenceStatus
    cost: float
    iterations: int
    total_time: float
    iteration_history: List[SubproblemResult] = field(default_factory=list)
    convergence_history: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if planning succeeded."""
        return self.status == ConvergenceStatus.CONVERGED and self.trajectory is not None

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Planner Result",
            "-" * 40,
            f"Status:      {self.status.value}",
            f"Cost:        {self.cost:.6f}",
            f"Iterations:  {self.iterations}",
            f"Total time:  {self.total_time:.3f} s",
        ]
        if self.trajectory is not None:
            lines.append(f"Trajectory:  N={self.trajectory.N}, dt={self.trajectory.dt}")
        return "\n".join(lines)


# =============================================================================
# Convergence Criteria
# =============================================================================


@dataclass
class ConvergenceCriteria:
    """
    Convergence criteria for iterative planners.

    Parameters
    ----------
    max_iterations : int
        Maximum number of iterations.
    cost_tolerance : float
        Relative cost change tolerance for convergence.
    constraint_tolerance : float
        Maximum constraint violation for convergence.
    trust_region_tolerance : float
        Minimum trust region size before failure.
    min_cost_improvement : float
        Minimum relative cost improvement per iteration.
    """

    max_iterations: int = 50
    cost_tolerance: float = 1e-6
    constraint_tolerance: float = 1e-4
    trust_region_tolerance: float = 1e-4
    min_cost_improvement: float = -1e-3  # Allow small increases

    def check_convergence(
        self,
        iteration: int,
        cost: float,
        cost_prev: float,
        constraint_violation: float,
        trust_region: float,
    ) -> Tuple[bool, ConvergenceStatus]:
        """
        Check if optimization has converged.

        Returns
        -------
        converged : bool
            Whether converged.
        status : ConvergenceStatus
            Convergence status.
        """
        # Max iterations
        if iteration >= self.max_iterations:
            return True, ConvergenceStatus.MAX_ITERATIONS

        # Trust region too small
        if trust_region < self.trust_region_tolerance:
            return True, ConvergenceStatus.TRUST_REGION_FAILED

        # Cost change
        relative_change = abs(cost - cost_prev) / cost_prev if cost_prev > 0 else abs(cost - cost_prev)

        # Check convergence
        if relative_change < self.cost_tolerance and constraint_violation < self.constraint_tolerance:
            return True, ConvergenceStatus.CONVERGED

        return False, ConvergenceStatus.NOT_STARTED


# =============================================================================
# Cost Function Classes
# =============================================================================


class CostFunction(ABC):
    """Abstract base class for cost functions."""

    @abstractmethod
    def running_cost(
        self,
        x: cp.Variable,
        u: cp.Variable,
        k: int,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ) -> cp.Expression:
        """
        Compute running cost at timestep k.

        Parameters
        ----------
        x : cp.Variable
            State variable at time k.
        u : cp.Variable
            Input variable at time k.
        k : int
            Timestep index.
        x_ref : np.ndarray, optional
            Reference state.
        u_ref : np.ndarray, optional
            Reference input.

        Returns
        -------
        cp.Expression
            Cost expression.
        """
        pass

    @abstractmethod
    def terminal_cost(
        self,
        x: cp.Variable,
        x_target: Optional[np.ndarray] = None,
    ) -> cp.Expression:
        """
        Compute terminal cost.

        Parameters
        ----------
        x : cp.Variable
            Terminal state variable.
        x_target : np.ndarray, optional
            Target terminal state.

        Returns
        -------
        cp.Expression
            Terminal cost expression.
        """
        pass


class QuadraticCost(CostFunction):
    """
    Quadratic cost function.

    J = x_N^T Q_f x_N + sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k)

    Can also track reference trajectories.
    """

    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        Q_f: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ):
        """
        Initialize quadratic cost.

        Parameters
        ----------
        Q : np.ndarray
            State cost matrix (n x n), positive semidefinite.
        R : np.ndarray
            Input cost matrix (m x m), positive definite.
        Q_f : np.ndarray, optional
            Terminal state cost matrix. Defaults to Q.
        x_ref : np.ndarray, optional
            Reference state (for tracking).
        u_ref : np.ndarray, optional
            Reference input (for tracking).
        """
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.Q_f = np.asarray(Q_f) if Q_f is not None else self.Q
        self._x_ref = x_ref
        self._u_ref = u_ref

        # Compute square roots for CVXPY
        self._Q_sqrt = self._matrix_sqrt(self.Q)
        self._R_sqrt = self._matrix_sqrt(self.R)
        self._Q_f_sqrt = self._matrix_sqrt(self.Q_f)

    def _matrix_sqrt(self, M: np.ndarray) -> np.ndarray:
        """Compute matrix square root for PSD matrix."""
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0)  # Ensure non-negative
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def running_cost(
        self,
        x: cp.Variable,
        u: cp.Variable,
        k: int,  # noqa: ARG002
        x_ref: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ) -> cp.Expression:
        """Compute running cost."""
        # Use provided reference or stored reference
        if x_ref is None:
            x_ref = self._x_ref if self._x_ref is not None else np.zeros(x.shape[0])
        if u_ref is None:
            u_ref = self._u_ref if self._u_ref is not None else np.zeros(u.shape[0])

        # Quadratic cost: ||Q_sqrt @ (x - x_ref)||^2 + ||R_sqrt @ (u - u_ref)||^2
        x_err = x - x_ref
        u_err = u - u_ref

        cost = cp.sum_squares(self._Q_sqrt @ x_err) + cp.sum_squares(self._R_sqrt @ u_err)
        return cost

    def terminal_cost(
        self,
        x: cp.Variable,
        x_target: Optional[np.ndarray] = None,
    ) -> cp.Expression:
        """Compute terminal cost."""
        if x_target is None:
            x_target = np.zeros(x.shape[0])

        x_err = x - x_target
        return cp.sum_squares(self._Q_f_sqrt @ x_err)


class MinimumTimeCost(CostFunction):
    """Cost function for minimum-time problems (penalize trajectory length)."""

    def __init__(self, time_weight: float = 1.0):
        self.time_weight = time_weight

    def running_cost(
        self,
        x: cp.Variable,  # noqa: ARG002
        u: cp.Variable,  # noqa: ARG002
        k: int,  # noqa: ARG002
        x_ref: Optional[np.ndarray] = None,  # noqa: ARG002
        u_ref: Optional[np.ndarray] = None,  # noqa: ARG002
    ) -> cp.Expression:
        """Running cost is constant (time penalty)."""
        return self.time_weight

    def terminal_cost(
        self,
        x: cp.Variable,  # noqa: ARG002
        x_target: Optional[np.ndarray] = None,  # noqa: ARG002
    ) -> cp.Expression:
        """No terminal cost."""
        return 0.0


class MinimumControlEffortCost(CostFunction):
    """Cost function minimizing control effort."""

    def __init__(
        self,
        R: np.ndarray,
        Q_f: Optional[np.ndarray] = None,
    ):
        self.R = np.asarray(R)
        self.Q_f = Q_f
        self._R_sqrt = self._matrix_sqrt(self.R)
        self._Q_f_sqrt = self._matrix_sqrt(Q_f) if Q_f is not None else None

    def _matrix_sqrt(self, M: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def running_cost(
        self,
        x: cp.Variable,  # noqa: ARG002
        u: cp.Variable,
        k: int,  # noqa: ARG002
        x_ref: Optional[np.ndarray] = None,  # noqa: ARG002
        u_ref: Optional[np.ndarray] = None,  # noqa: ARG002
    ) -> cp.Expression:
        """Running cost: ||R_sqrt @ u||^2."""
        return cp.sum_squares(self._R_sqrt @ u)

    def terminal_cost(
        self,
        x: cp.Variable,
        x_target: Optional[np.ndarray] = None,
    ) -> cp.Expression:
        """Terminal cost."""
        if self._Q_f_sqrt is None or x_target is None:
            return 0.0
        return cp.sum_squares(self._Q_f_sqrt @ (x - x_target))


# =============================================================================
# Constraint Classes
# =============================================================================


class ConstraintType(Enum):
    """Type of constraint (hard or soft)."""

    HARD = "hard"
    SOFT_L1 = "soft_l1"
    SOFT_L2 = "soft_l2"


@dataclass
class ConstraintConfig:
    """Configuration for a constraint."""

    constraint_type: ConstraintType = ConstraintType.HARD
    penalty_weight: float = 1e4
    slack_lower_bound: float = 0.0

    @classmethod
    def hard(cls) -> "ConstraintConfig":
        return cls(constraint_type=ConstraintType.HARD)

    @classmethod
    def soft_l1(cls, weight: float = 1e4) -> "ConstraintConfig":
        return cls(constraint_type=ConstraintType.SOFT_L1, penalty_weight=weight)

    @classmethod
    def soft_l2(cls, weight: float = 1e4) -> "ConstraintConfig":
        return cls(constraint_type=ConstraintType.SOFT_L2, penalty_weight=weight)


class ConstraintSet:
    """
    Manager for optimization constraints.

    Handles both hard and soft constraints with CVXPY.
    """

    def __init__(self):
        self._constraints: List[cp.Constraint] = []
        self._slack_variables: Dict[str, cp.Variable] = {}
        self._slack_penalties: List[cp.Expression] = []

    def add_hard_constraint(self, constraint: cp.Constraint) -> None:
        """Add a hard constraint."""
        self._constraints.append(constraint)

    def add_soft_constraint(
        self,
        expr: cp.Expression,
        bound: float,
        name: str,
        config: ConstraintConfig,
        is_upper_bound: bool = True,
    ) -> cp.Variable:
        """
        Add a soft constraint with slack variable.

        Parameters
        ----------
        expr : cp.Expression
            Expression to constrain.
        bound : float
            Constraint bound.
        name : str
            Name for slack variable.
        config : ConstraintConfig
            Constraint configuration.
        is_upper_bound : bool
            If True: expr <= bound + slack
            If False: expr >= bound - slack

        Returns
        -------
        cp.Variable
            Slack variable.
        """
        # Create slack variable
        slack = cp.Variable(nonneg=True, name=f"slack_{name}")
        self._slack_variables[name] = slack

        # Add constraint with slack
        if is_upper_bound:
            self._constraints.append(expr <= bound + slack)
        else:
            self._constraints.append(expr >= bound - slack)

        # Add penalty
        if config.constraint_type == ConstraintType.SOFT_L1:
            self._slack_penalties.append(config.penalty_weight * slack)
        elif config.constraint_type == ConstraintType.SOFT_L2:
            self._slack_penalties.append(config.penalty_weight * cp.square(slack))

        return slack

    def add_box_constraint(
        self,
        var: cp.Variable,
        lb: np.ndarray,
        ub: np.ndarray,
        config: ConstraintConfig = None,
        name: str = "box",
    ) -> Optional[Tuple[cp.Variable, cp.Variable]]:
        """
        Add box constraints: lb <= var <= ub.

        Parameters
        ----------
        var : cp.Variable
            Variable to constrain.
        lb : np.ndarray
            Lower bound.
        ub : np.ndarray
            Upper bound.
        config : ConstraintConfig, optional
            Constraint config. Default is hard constraint.
        name : str
            Name prefix for slack variables.

        Returns
        -------
        tuple or None
            (slack_lower, slack_upper) if soft, None if hard.
        """
        if config is None:
            config = ConstraintConfig.hard()

        if config.constraint_type == ConstraintType.HARD:
            self._constraints.append(var >= lb)
            self._constraints.append(var <= ub)
            return None
        else:
            # Soft constraints
            slack_lb = cp.Variable(var.shape, nonneg=True, name=f"slack_{name}_lb")
            slack_ub = cp.Variable(var.shape, nonneg=True, name=f"slack_{name}_ub")

            self._constraints.append(var >= lb - slack_lb)
            self._constraints.append(var <= ub + slack_ub)

            self._slack_variables[f"{name}_lb"] = slack_lb
            self._slack_variables[f"{name}_ub"] = slack_ub

            # Add penalties
            if config.constraint_type == ConstraintType.SOFT_L1:
                self._slack_penalties.append(config.penalty_weight * cp.sum(slack_lb))
                self._slack_penalties.append(config.penalty_weight * cp.sum(slack_ub))
            else:  # L2
                self._slack_penalties.append(config.penalty_weight * cp.sum_squares(slack_lb))
                self._slack_penalties.append(config.penalty_weight * cp.sum_squares(slack_ub))

            return slack_lb, slack_ub

    @property
    def constraints(self) -> List[cp.Constraint]:
        """Get all constraints."""
        return self._constraints

    @property
    def slack_penalty(self) -> cp.Expression:
        """Get total slack penalty."""
        if not self._slack_penalties:
            return 0.0
        return sum(self._slack_penalties)

    @property
    def slack_variables(self) -> Dict[str, cp.Variable]:
        """Get slack variables."""
        return self._slack_variables

    def get_slack_values(self) -> Dict[str, np.ndarray]:
        """Get slack variable values after solving."""
        return {name: var.value for name, var in self._slack_variables.items() if var.value is not None}

    def total_slack_violation(self) -> float:
        """Compute total slack violation after solving."""
        total = 0.0
        for var in self._slack_variables.values():
            if var.value is not None:
                total += np.sum(np.abs(var.value))
        return total


# =============================================================================
# Abstract Optimization Problem
# =============================================================================


class OptimizationProblem(ABC):
    """
    Abstract base class for a single optimization subproblem.

    Defines the interface for setting up and solving convex optimization
    problems using CVXPY.
    """

    def __init__(
        self,
        n_states: int,
        n_inputs: int,
        N: int,
        dt: float,
    ):
        """
        Initialize optimization problem.

        Parameters
        ----------
        n_states : int
            State dimension.
        n_inputs : int
            Input dimension.
        N : int
            Number of timesteps.
        dt : float
            Timestep.
        """
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.N = N
        self.dt = dt

        # CVXPY variables
        self._x: Optional[cp.Variable] = None
        self._u: Optional[cp.Variable] = None

        # Problem components
        self._objective: Optional[cp.Expression] = None
        self._constraints: ConstraintSet = ConstraintSet()
        self._problem: Optional[cp.Problem] = None

        # Solver settings
        self._solver = cp.MOSEK
        self._solver_options: Dict[str, Any] = {}

    @property
    def x(self) -> cp.Variable:
        """State trajectory variable (N+1, n_states)."""
        if self._x is None:
            self._x = cp.Variable((self.N + 1, self.n_states), name="x")
        return self._x

    @property
    def u(self) -> cp.Variable:
        """Input trajectory variable (N, n_inputs)."""
        if self._u is None:
            self._u = cp.Variable((self.N, self.n_inputs), name="u")
        return self._u

    def set_solver(
        self,
        solver: str = "MOSEK",
        **options,
    ) -> None:
        """
        Set solver and options.

        Parameters
        ----------
        solver : str
            Solver name (MOSEK, ECOS, SCS, OSQP).
        **options
            Solver-specific options.
        """
        solver_map = {
            "MOSEK": cp.MOSEK,
            "ECOS": cp.ECOS,
            "SCS": cp.SCS,
            "OSQP": cp.OSQP,
            "CLARABEL": cp.CLARABEL,
        }
        self._solver = solver_map.get(solver.upper(), cp.MOSEK)
        self._solver_options = options

    @abstractmethod
    def setup(
        self,
        x_init: np.ndarray,
        x_target: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Set up the optimization problem.

        Parameters
        ----------
        x_init : np.ndarray
            Initial state.
        x_target : np.ndarray
            Target state.
        **kwargs
            Additional problem-specific parameters.
        """
        pass

    def _build_problem(self) -> None:
        """Build the CVXPY problem."""
        if self._objective is None:
            raise ValueError("Objective not set. Call setup() first.")

        # Add slack penalty to objective
        total_objective = self._objective + self._constraints.slack_penalty

        self._problem = cp.Problem(
            cp.Minimize(total_objective),
            self._constraints.constraints,
        )

    def solve(
        self,
        warm_start: bool = True,
        verbose: bool = False,
    ) -> SubproblemResult:
        """
        Solve the optimization problem.

        Parameters
        ----------
        warm_start : bool
            Use warm start from previous solution.
        verbose : bool
            Print solver output.

        Returns
        -------
        SubproblemResult
            Solution result.
        """
        if self._problem is None:
            self._build_problem()

        with Timer("CVXPY solve") as timer:
            try:
                self._problem.solve(
                    solver=self._solver,
                    warm_start=warm_start,
                    verbose=verbose,
                    **self._solver_options,
                )
            except cp.SolverError as e:
                logger.warning(f"Solver error: {e}")
                return SubproblemResult(
                    status=SolverStatus.SOLVER_ERROR,
                    cost=np.inf,
                    solve_time=timer.elapsed,
                    solver_info={"error": str(e)},
                )

        status = SolverStatus.from_cvxpy(self._problem.status)

        result = SubproblemResult(
            status=status,
            cost=self._problem.value if self._problem.value is not None else np.inf,
            solve_time=timer.elapsed,
            solver_info={
                "cvxpy_status": self._problem.status,
                "solver": str(self._solver),
            },
        )

        if status.is_optimal:
            result.x = self._x.value
            result.u = self._u.value

            # Get slack values
            slack_values = self._constraints.get_slack_values()
            if "state_lb" in slack_values or "state_ub" in slack_values:
                result.slack_state = np.maximum(
                    slack_values.get("state_lb", 0),
                    slack_values.get("state_ub", 0),
                )
            if "input_lb" in slack_values or "input_ub" in slack_values:
                result.slack_input = np.maximum(
                    slack_values.get("input_lb", 0),
                    slack_values.get("input_ub", 0),
                )

        return result

    def warm_start_from(
        self,
        x_init: np.ndarray,
        u_init: np.ndarray,
    ) -> None:
        """
        Set warm start values.

        Parameters
        ----------
        x_init : np.ndarray
            Initial state trajectory guess.
        u_init : np.ndarray
            Initial input trajectory guess.
        """
        if self._x is not None:
            self._x.value = x_init
        if self._u is not None:
            self._u.value = u_init


# =============================================================================
# Abstract Base Planner
# =============================================================================


class BasePlanner(ABC):
    """
    Abstract base class for trajectory planners.

    Provides interface for single-shot and iterative planners.
    """

    def __init__(
        self,
        n_states: int,
        n_inputs: int,
        dt: float,
        N: int,
    ):
        """
        Initialize planner.

        Parameters
        ----------
        n_states : int
            State dimension.
        n_inputs : int
            Input dimension.
        dt : float
            Timestep.
        N : int
            Horizon length.
        """
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.dt = dt
        self.N = N

        # Constraints
        self._state_constraints: Optional[StateInputConstraints] = None
        self._obstacles: Optional[ObstacleCollection] = None

        # Cost function
        self._cost_function: Optional[CostFunction] = None

        # Solver settings
        self._solver = "MOSEK"
        self._solver_options: Dict[str, Any] = {}
        self._verbose = False

        # Convergence criteria
        self._convergence = ConvergenceCriteria()

        # History
        self._iteration_history: List[SubproblemResult] = []
        self._convergence_history: Dict[str, List[float]] = {
            "cost": [],
            "constraint_violation": [],
            "trust_region": [],
        }

    def set_constraints(
        self,
        constraints: StateInputConstraints,
        obstacles: Optional[ObstacleCollection] = None,
    ) -> None:
        """Set state/input constraints and obstacles."""
        self._state_constraints = constraints
        self._obstacles = obstacles

    def set_cost_function(self, cost_function: CostFunction) -> None:
        """Set cost function."""
        self._cost_function = cost_function

    def set_solver(self, solver: str = "MOSEK", **options) -> None:
        """Set solver and options."""
        self._solver = solver
        self._solver_options = options

    def set_convergence_criteria(self, criteria: ConvergenceCriteria) -> None:
        """Set convergence criteria."""
        self._convergence = criteria

    def set_verbose(self, verbose: bool) -> None:
        """Set verbose mode."""
        self._verbose = verbose

    @abstractmethod
    def plan(
        self,
        x_init: np.ndarray,
        x_target: np.ndarray,
        u_init_guess: Optional[np.ndarray] = None,
        x_init_guess: Optional[np.ndarray] = None,
        **kwargs,
    ) -> PlannerResult:
        """
        Plan a trajectory from initial to target state.

        Parameters
        ----------
        x_init : np.ndarray
            Initial state.
        x_target : np.ndarray
            Target state.
        u_init_guess : np.ndarray, optional
            Initial input trajectory guess.
        x_init_guess : np.ndarray, optional
            Initial state trajectory guess.
        **kwargs
            Additional planner-specific parameters.

        Returns
        -------
        PlannerResult
            Planning result.
        """
        pass

    def _create_trajectory(
        self,
        x: np.ndarray,
        u: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Trajectory:
        """Create trajectory from solution."""
        if metadata is None:
            metadata = {}
        return Trajectory(
            x=x,
            u=u,
            dt=self.dt,
            t0=0.0,
            metadata=metadata,
        )

    def _log_iteration(
        self,
        iteration: int,
        cost: float,
        violation: float,
        trust_region: float,
    ) -> None:
        """Log iteration info."""
        self._convergence_history["cost"].append(cost)
        self._convergence_history["constraint_violation"].append(violation)
        self._convergence_history["trust_region"].append(trust_region)

        if self._verbose:
            logger.info(f"Iter {iteration:3d}: cost={cost:.6e}, violation={violation:.6e}, tr={trust_region:.4f}")

    def reset_history(self) -> None:
        """Reset iteration history."""
        self._iteration_history = []
        self._convergence_history = {
            "cost": [],
            "constraint_violation": [],
            "trust_region": [],
        }


# =============================================================================
# Utility Functions
# =============================================================================


def create_quadratic_cost(
    Q: np.ndarray,
    R: np.ndarray,
    Q_f: Optional[np.ndarray] = None,
) -> QuadraticCost:
    """
    Create quadratic cost function.

    Parameters
    ----------
    Q : np.ndarray
        State cost matrix.
    R : np.ndarray
        Input cost matrix.
    Q_f : np.ndarray, optional
        Terminal cost matrix.

    Returns
    -------
    QuadraticCost
        Cost function.
    """
    return QuadraticCost(Q=Q, R=R, Q_f=Q_f)


def create_default_cost(
    n_states: int,
    n_inputs: int,
    state_weight: float = 1.0,
    input_weight: float = 0.1,
    terminal_weight: float = 10.0,
) -> QuadraticCost:
    """
    Create default quadratic cost with diagonal weights.

    Parameters
    ----------
    n_states : int
        State dimension.
    n_inputs : int
        Input dimension.
    state_weight : float
        Diagonal state cost weight.
    input_weight : float
        Diagonal input cost weight.
    terminal_weight : float
        Diagonal terminal cost weight.

    Returns
    -------
    QuadraticCost
        Cost function.
    """
    Q = state_weight * np.eye(n_states)
    R = input_weight * np.eye(n_inputs)
    Q_f = terminal_weight * np.eye(n_states)
    return QuadraticCost(Q=Q, R=R, Q_f=Q_f)
