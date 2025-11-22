# ddfs/ddfs/planning/scvx.py

"""
Successive Convexification (SCvx) trajectory planner.

This module provides the SCvxPlanner class for generating feasible
trajectories using successive convexification with obstacle avoidance.

The planner solves the trajectory optimization problem:
    minimize    cost(x, u)
    subject to  x(k+1) = f(x(k), u(k))          (dynamics)
                x(0) = x0                        (initial condition)
                x(N) = xf                        (terminal condition)
                x(k) ∈ X                         (state constraints)
                u(k) ∈ U                         (input constraints)
                ||x(k)[:d] - obs_i|| ≥ r_i      (obstacle avoidance)

Using successive convexification to handle non-convex constraints.
"""

import logging
from typing import Any, Dict, List, Optional

import cvxpy as cp
import numpy as np

from ddfs.core.constraints import SystemConstraints
from ddfs.core.obstacles import Obstacle
from ddfs.models.base import TwinModel
from ddfs.planning.nominal_trajectory import NominalTrajectory


class SCvxPlanner:
    """
    SCvx trajectory planner using successive convexification.

    This planner generates feasible trajectories from an initial state to
    a goal state while avoiding obstacles. The SCvx algorithm iteratively
    solves convex approximations of the non-convex trajectory optimization problem.

    Parameters
    ----------
    twin : TwinModel
        Digital twin model for dynamics
    constraints : SystemConstraints
        State and input constraints
    obstacles : List[Obstacle]
        List of obstacles to avoid
    config : Dict[str, Any], optional
        Planner configuration parameters

    Attributes
    ----------
    twin : TwinModel
        Digital twin dynamics
    constraints : SystemConstraints
        System constraints
    obstacles : List[Obstacle]
        Obstacles to avoid
    max_iterations : int
        Maximum SCvx iterations
    convergence_tol : float
        Convergence tolerance
    trust_region : float
        Trust region radius
    verbose : bool
        Print iteration info

    Examples
    --------
    >>> from ddfs.core import DDFSConfig
    >>> from ddfs.models import UnicycleTwin
    >>> from ddfs.planning import SCvxPlanner
    >>>
    >>> # Setup
    >>> config = DDFSConfig('config/ddfs_config.yaml')
    >>> twin = UnicycleTwin(dt=0.131)
    >>> constraints = config.get_constraints()
    >>> obstacles = config.get_obstacles()
    >>>
    >>> # Plan
    >>> planner = SCvxPlanner(twin, constraints, obstacles)
    >>> params = config.get_planning_params()
    >>> traj = planner.plan(
    ...     x0=params['x0'],
    ...     xf=params['xf'],
    ...     N=params['N']
    ... )
    """

    def __init__(
        self,
        twin: TwinModel,
        constraints: SystemConstraints,
        obstacles: List[Obstacle],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SCvx planner.

        Parameters
        ----------
        twin : TwinModel
            Digital twin model
        constraints : SystemConstraints
            System constraints
        obstacles : List[Obstacle]
            List of obstacles
        config : dict, optional
            Configuration parameters
        """
        self.twin = twin
        self.constraints = constraints
        self.obstacles = obstacles
        self.config = config or {}

        # Extract config parameters
        self.max_iterations = self.config.get("max_iterations", 20)
        self.convergence_tol = self.config.get("convergence_tol", 1e-3)
        self.trust_region = self.config.get("trust_region", 1.0)
        self.verbose = self.config.get("verbose", True)
        self.weight_state = self.config.get("weight_state", 1.0)
        self.weight_input = self.config.get("weight_input", 0.1)
        self.weight_virtual = self.config.get("weight_virtual", 1000.0)

        # Logging
        self.logger = logging.getLogger(__name__)

    def plan(  # noqa: C901
        self,
        x0: np.ndarray,
        xf: np.ndarray,
        N: int,
        x_guess: Optional[np.ndarray] = None,
        u_guess: Optional[np.ndarray] = None,
    ) -> NominalTrajectory:
        """
        Plan trajectory from x0 to xf.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state, shape (n,)
        xf : np.ndarray
            Goal state, shape (n,)
        N : int
            Planning horizon
        x_guess : np.ndarray, optional
            Initial guess for states, shape (N+1, n)
        u_guess : np.ndarray, optional
            Initial guess for inputs, shape (N, m)

        Returns
        -------
        trajectory : NominalTrajectory
            Planned nominal trajectory

        Raises
        ------
        RuntimeError
            If planning fails to converge
        """
        if self.verbose:
            self.logger.info("=" * 60)
            self.logger.info("SCvx TRAJECTORY PLANNING")
            self.logger.info("=" * 60)
            self.logger.info(f"Initial state: {x0}")
            self.logger.info(f"Goal state: {xf}")
            self.logger.info(f"Horizon: N={N}")
            self.logger.info(f"Obstacles: {len(self.obstacles)}")

        n = self.twin.state_dim
        m = self.twin.input_dim
        dt = self.twin.dt

        # Initialize guess (straight line in state space)
        if x_guess is None:
            x_guess = self._initialize_state_guess(x0, xf, N, n)
        if u_guess is None:
            u_guess = np.zeros((N, m))

        # SCvx iterations
        x_sol = x_guess.copy()
        u_sol = u_guess.copy()

        for iteration in range(self.max_iterations):
            if self.verbose:
                self.logger.info(f"\nIteration {iteration + 1}/{self.max_iterations}")

            # Linearize dynamics around current trajectory
            A_list, B_list, c_list = self._linearize_trajectory(x_sol, u_sol, dt)

            # Solve convex subproblem
            x_new, u_new, cost = self._solve_convex_subproblem(
                x0, xf, N, n, m, dt, x_sol, u_sol, A_list, B_list, c_list
            )

            # Check convergence
            delta_x = np.linalg.norm(x_new - x_sol)
            delta_u = np.linalg.norm(u_new - u_sol)

            if self.verbose:
                self.logger.info(f"  Cost: {cost:.6e}")
                self.logger.info(f"  Δx: {delta_x:.6e}, Δu: {delta_u:.6e}")

            # Update solution
            x_sol = x_new
            u_sol = u_new

            # Check convergence
            if delta_x < self.convergence_tol and delta_u < self.convergence_tol:
                if self.verbose:
                    self.logger.info(f"\n✓ Converged in {iteration + 1} iterations")
                break
        else:
            if self.verbose:
                self.logger.warning(f"\n⚠ Did not converge in {self.max_iterations} iterations")

        # Verify obstacle avoidance
        violations = self._check_obstacle_violations(x_sol)
        if violations:
            self.logger.warning(f"⚠ Obstacle violations detected: {violations}")

        # Create nominal trajectory
        trajectory = NominalTrajectory(x_nom=x_sol, u_nom=u_sol, N=N, dt=dt)

        if self.verbose:
            self.logger.info("=" * 60)

        return trajectory

    def _initialize_state_guess(self, x0: np.ndarray, xf: np.ndarray, N: int, n: int) -> np.ndarray:
        """
        Initialize state guess as straight line from x0 to xf.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state
        xf : np.ndarray
            Goal state
        N : int
            Horizon
        n : int
            State dimension

        Returns
        -------
        x_guess : np.ndarray
            Initial state guess, shape (N+1, n)
        """
        x_guess = np.zeros((N + 1, n))
        for i in range(N + 1):
            alpha = i / N
            x_guess[i] = (1 - alpha) * x0 + alpha * xf
        return x_guess

    def _linearize_trajectory(
        self, x_traj: np.ndarray, u_traj: np.ndarray, dt: float
    ) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Linearize dynamics along trajectory.

        For each timestep k, computes:
            x(k+1) ≈ A(k) x(k) + B(k) u(k) + c(k)

        where A(k), B(k) are Jacobians and c(k) is the affine term.

        Parameters
        ----------
        x_traj : np.ndarray
            State trajectory, shape (N+1, n)
        u_traj : np.ndarray
            Input trajectory, shape (N, m)
        dt : float
            Timestep

        Returns
        -------
        A_list : list of np.ndarray
            List of A matrices, each (n, n)
        B_list : list of np.ndarray
            List of B matrices, each (n, m)
        c_list : list of np.ndarray
            List of c vectors, each (n,)
        """
        N = u_traj.shape[0]
        A_list = []
        B_list = []
        c_list = []

        for k in range(N):
            x_k = x_traj[k]
            u_k = u_traj[k]

            # Get Jacobians from twin
            A_cont, B_cont = self.twin.jacobians(x_k, u_k)

            # Discretize: x(k+1) = x(k) + dt * (A x(k) + B u(k))
            # First-order approximation
            A_disc = np.eye(len(x_k)) + dt * A_cont
            B_disc = dt * B_cont

            # Compute affine term
            x_next_actual = self.twin.step(x_k, u_k, dt)
            x_next_linear = A_disc @ x_k + B_disc @ u_k
            c_k = x_next_actual - x_next_linear

            A_list.append(A_disc)
            B_list.append(B_disc)
            c_list.append(c_k)

        return A_list, B_list, c_list

    def _solve_convex_subproblem(  # noqa: C901
        self,
        x0: np.ndarray,
        xf: np.ndarray,
        N: int,
        n: int,
        m: int,
        dt: float,  # noqa: ARG002
        x_ref: np.ndarray,
        u_ref: np.ndarray,
        A_list: List[np.ndarray],
        B_list: List[np.ndarray],
        c_list: List[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Solve convex subproblem with linearized dynamics.

        Parameters
        ----------
        x0, xf : np.ndarray
            Initial and goal states
        N : int
            Horizon
        n, m : int
            State and input dimensions
        dt : float
            Timestep
        x_ref, u_ref : np.ndarray
            Reference trajectory for trust region
        A_list, B_list, c_list : list
            Linearized dynamics

        Returns
        -------
        x_sol : np.ndarray
            Optimal state trajectory, shape (N+1, n)
        u_sol : np.ndarray
            Optimal input trajectory, shape (N, m)
        cost : float
            Optimal cost
        """
        # Decision variables
        x = cp.Variable((N + 1, n))
        u = cp.Variable((N, m))
        nu = cp.Variable(N + 1)  # Virtual control for feasibility

        # Cost function: minimize deviation + input effort + virtual control penalty
        cost = 0
        for k in range(N + 1):
            cost += self.weight_state * cp.sum_squares(x[k] - xf)
            cost += self.weight_virtual * nu[k]
        for k in range(N):
            cost += self.weight_input * cp.sum_squares(u[k])

        # Constraints
        constraints = []

        # Initial condition
        constraints.append(x[0] == x0)

        # Linearized dynamics
        for k in range(N):
            constraints.append(x[k + 1] == A_list[k] @ x[k] + B_list[k] @ u[k] + c_list[k])

        # Terminal condition (with slack)
        constraints.append(cp.norm(x[N] - xf, "inf") <= nu[N])

        # Input constraints
        for k in range(N):
            u_clipped = self.constraints.clip_input(u_ref[k])  # noqa: F841
            # Simple box constraints (system-specific)
            if hasattr(self.constraints, "u_min") and hasattr(self.constraints, "u_max"):
                constraints.append(u[k] >= self.constraints.u_min)
                constraints.append(u[k] <= self.constraints.u_max)

        # Obstacle avoidance (linearized)
        for k in range(N + 1):
            for obs in self.obstacles:
                # Get position from state (first 2 or 3 elements)
                pos_dim = len(obs.center)
                pos_ref = x_ref[k, :pos_dim]

                # Linearize: ||pos - center|| ≥ r
                # At reference: a^T (pos - pos_ref) ≥ b
                dist_vec = pos_ref - obs.center
                dist = np.linalg.norm(dist_vec)

                if dist > 1e-6:  # Avoid division by zero
                    a = dist_vec / dist
                    b = obs.effective_radius - dist

                    # Linearized constraint with slack
                    constraints.append(a @ x[k, :pos_dim] >= a @ pos_ref + b - nu[k])

        # Trust region (limit deviation from reference)
        for k in range(N + 1):
            constraints.append(cp.norm(x[k] - x_ref[k]) <= self.trust_region)
        for k in range(N):
            constraints.append(cp.norm(u[k] - u_ref[k]) <= self.trust_region)

        # Non-negativity of virtual control
        constraints.append(nu >= 0)

        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")

        return x.value, u.value, problem.value

    def _check_obstacle_violations(self, x_traj: np.ndarray) -> List[str]:
        """
        Check for obstacle violations in trajectory.

        Parameters
        ----------
        x_traj : np.ndarray
            State trajectory

        Returns
        -------
        violations : list of str
            List of violation messages
        """
        violations = []
        for k, x_k in enumerate(x_traj):
            for obs in self.obstacles:
                if obs.contains(x_k, include_margin=True):
                    violations.append(f"Step {k}: inside {obs.id}")
        return violations

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SCvxPlanner("
            f"system={self.twin.__class__.__name__}, "
            f"obstacles={len(self.obstacles)}, "
            f"max_iter={self.max_iterations})"
        )
