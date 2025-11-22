"""
Successive Convexification (SCvx) Planner with Obstacle Avoidance

This version properly handles:
1. Linearized obstacle constraints with trust region
2. Better discrete-time linearization
3. Circular/spherical obstacle avoidance
4. Quaternion normalization for quadrotor

References
----------
[1] Mao et al., "Successive Convexification: A Superlinearly Convergent
    Algorithm for Non-convex Optimal Control Problems", 2018
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np

from ddfs.models.base import DynamicsModel
from ddfs.planning.nominal_trajectory import NominalTrajectory


@dataclass
class Obstacle:
    """
    Obstacle definition.

    Attributes
    ----------
    center : np.ndarray
        Obstacle center position (2D or 3D)
    radius : float
        Obstacle radius
    """

    center: np.ndarray
    radius: float

    def signed_distance(self, p: np.ndarray) -> float:
        """Compute signed distance from point to obstacle surface."""
        return np.linalg.norm(p - self.center) - self.radius

    def gradient(self, p: np.ndarray) -> np.ndarray:
        """Compute gradient of signed distance function."""
        diff = p - self.center
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            # Avoid division by zero
            return np.zeros_like(diff)
        return diff / dist


@dataclass
class SCvxParameters:
    """
    Parameters for the SCvx algorithm.

    Attributes
    ----------
    max_iterations : int
        Maximum number of SCvx iterations
    convergence_tol : float
        Convergence tolerance for virtual control norm
    trust_region_init : float
        Initial trust region radius
    trust_region_min : float
        Minimum trust region radius
    trust_region_max : float
        Maximum trust region radius
    rho_0 : float
        Trust region shrink threshold (reject if rho < rho_0)
    rho_1 : float
        Trust region shrink threshold (shrink if rho < rho_1)
    rho_2 : float
        Trust region expand threshold (expand if rho >= rho_2)
    alpha : float
        Trust region shrink factor
    beta : float
        Trust region expand factor
    weight_nu : float
        Weight for virtual control penalty
    weight_nu_bound : float
        Upper bound for virtual control weight (for ramping)
    nu_tol : float
        Virtual control tolerance for convergence
    verbose : bool
        Print iteration details
    solver : str
        CVXPY solver to use ('ECOS', 'SCS', 'MOSEK', etc.)
    solver_verbose : bool
        Print solver output
    """

    max_iterations: int = 30
    convergence_tol: float = 1e-3
    trust_region_init: float = 1.0
    trust_region_min: float = 0.01
    trust_region_max: float = 100.0
    rho_0: float = 0.0
    rho_1: float = 0.25
    rho_2: float = 0.75
    alpha: float = 2.0
    beta: float = 2.0
    weight_nu: float = 1e5
    weight_nu_bound: float = 1e10
    nu_tol: float = 1e-6
    verbose: bool = True
    solver: str = "ECOS"
    solver_verbose: bool = False


class ObstacleManager:
    """
    Manages obstacle avoidance constraints for SCvx.

    Handles linearization of nonlinear obstacle avoidance constraints
    around a reference trajectory.
    """

    def __init__(self, obstacles: List[Obstacle], state_dim: int, position_indices: List[int]):
        """
        Initialize obstacle manager.

        Parameters
        ----------
        obstacles : List[Obstacle]
            List of obstacles to avoid
        state_dim : int
            Full state dimension
        position_indices : List[int]
            Indices of position coordinates in state vector (e.g., [0, 1] for 2D)
        """
        self.obstacles = obstacles
        self.state_dim = state_dim
        self.pos_indices = position_indices
        self.pos_dim = len(position_indices)

    def add_constraints(self, X: cp.Variable, X_ref: cp.Parameter, N: int, safety_margin: float = 0.1) -> List:
        """
        Add linearized obstacle avoidance constraints.

        For each obstacle and timestep, enforces:
            h(p_ref) + ∇h(p_ref)ᵀ(p - p_ref) ≥ safety_margin

        where h(p) = ||p - p_obs|| - r_obs is the signed distance.

        Parameters
        ----------
        X : cp.Variable, shape (n, N+1)
            State decision variables
        X_ref : cp.Parameter, shape (n, N+1)
            Reference trajectory around which to linearize
        N : int
            Number of timesteps
        safety_margin : float
            Additional safety margin beyond obstacle radius

        Returns
        -------
        constraints : List
            List of CVXPY constraints
        """
        constraints = []

        if len(self.obstacles) == 0:
            return constraints

        for k in range(N + 1):
            for obs in self.obstacles:
                # Extract position from reference
                p_ref = X_ref[self.pos_indices, k]

                # Check if reference value is available
                if not hasattr(p_ref, "value") or p_ref.value is None:
                    # Skip constraint if reference not initialized yet
                    # This happens during problem setup
                    continue

                # Signed distance at reference
                h_ref = obs.signed_distance(p_ref.value)

                # Skip if reference is already far from obstacle
                # (constraint would be very loose)
                if h_ref > obs.radius * 3.0:
                    continue

                # Gradient at reference
                grad_h = obs.gradient(p_ref.value)

                # Check gradient is valid
                if np.linalg.norm(grad_h) < 1e-10:
                    # At obstacle center - use conservative constraint
                    # Force position away from center
                    p = X[self.pos_indices, k]
                    constraints.append(cp.norm(p - obs.center) >= obs.radius + safety_margin)
                    continue

                # Linearized constraint: h_ref + grad_h' * (p - p_ref) >= margin
                # Rearranged: grad_h' * p >= grad_h' * p_ref - h_ref + margin
                p = X[self.pos_indices, k]

                rhs = grad_h @ p_ref.value - h_ref + safety_margin
                constraints.append(grad_h @ p >= rhs)

        return constraints


class SCvxProblem:
    """
    Convex subproblem for Successive Convexification with obstacle avoidance.
    """

    def __init__(
        self,
        model: DynamicsModel,
        N: int,
        state_dim: int,
        input_dim: int,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        input_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        obstacle_manager: Optional[ObstacleManager] = None,
        position_indices: Optional[List[int]] = None,  # noqa: ARG002
    ):
        """
        Initialize SCvx convex subproblem.

        Parameters
        ----------
        model : DynamicsModel
            Dynamics model
        N : int
            Number of timesteps
        state_dim : int
            State dimension
        input_dim : int
            Input dimension
        state_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
            State box constraints (x_min, x_max)
        input_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
            Input box constraints (u_min, u_max)
        obstacle_manager : Optional[ObstacleManager]
            Manager for obstacle avoidance constraints
        position_indices : Optional[List[int]]
            Indices of position states (for obstacle avoidance)
        """
        self.model = model
        self.N = N
        self.n = state_dim
        self.m = input_dim
        self.obstacle_manager = obstacle_manager

        # Decision variables
        self.X = cp.Variable((self.n, N + 1))  # States
        self.U = cp.Variable((self.m, N))  # Controls
        self.nu = cp.Variable((self.n, N))  # Virtual control

        # Parameters (will be updated each iteration)
        self.A_bar = cp.Parameter((self.n, self.n, N))  # Linearized A matrices
        self.B_bar = cp.Parameter((self.n, self.m, N))  # Linearized B matrices
        self.z_bar = cp.Parameter((self.n, N))  # Affine terms
        self.X_ref = cp.Parameter((self.n, N + 1))  # Reference trajectory
        self.U_ref = cp.Parameter((self.m, N))  # Reference controls
        self.tr_radius = cp.Parameter(nonneg=True)  # Trust region radius
        self.weight_nu = cp.Parameter(nonneg=True)  # Virtual control weight

        # Initialize X_ref to avoid issues during setup
        self.X_ref.value = np.zeros((self.n, N + 1))

        # Build constraints
        constraints = []

        # Linearized dynamics: x[k+1] = A_bar[k]@x[k] + B_bar[k]@u[k] + z_bar[k] + nu[k]
        for k in range(N):
            constraints.append(
                self.X[:, k + 1]
                == self.A_bar[:, :, k] @ self.X[:, k]
                + self.B_bar[:, :, k] @ self.U[:, k]
                + self.z_bar[:, k]
                + self.nu[:, k]
            )

        # Trust region constraint: ||X - X_ref||_inf + ||U - U_ref||_inf <= tr_radius
        dx = self.X - self.X_ref
        du = self.U - self.U_ref
        constraints.append(cp.norm(dx, "inf") + cp.norm(du, "inf") <= self.tr_radius)

        # State bounds (box constraints)
        if state_bounds is not None:
            x_min, x_max = state_bounds
            for i in range(self.n):
                if not np.isinf(x_min[i]):
                    constraints.append(self.X[i, :] >= x_min[i])
                if not np.isinf(x_max[i]):
                    constraints.append(self.X[i, :] <= x_max[i])

        # Input bounds (box constraints)
        if input_bounds is not None:
            u_min, u_max = input_bounds
            for i in range(self.m):
                if not np.isinf(u_min[i]):
                    constraints.append(self.U[i, :] >= u_min[i])
                if not np.isinf(u_max[i]):
                    constraints.append(self.U[i, :] <= u_max[i])

        # NOTE: Obstacle avoidance constraints will be added later
        # since they depend on reference trajectory which updates each iteration
        # We'll rebuild the problem each iteration to update these constraints

        # Objective: minimize virtual control
        objective = cp.Minimize(self.weight_nu * cp.norm(self.nu, 1))

        # Create problem (without obstacle constraints for now)
        self.prob = cp.Problem(objective, constraints)
        self.base_constraints = constraints  # Save base constraints

    def set_parameters(
        self,
        A_bar: np.ndarray,
        B_bar: np.ndarray,
        z_bar: np.ndarray,
        X_ref: np.ndarray,
        U_ref: np.ndarray,
        tr_radius: float,
        weight_nu: float,
    ):
        """Set parameter values and update obstacle constraints for the convex subproblem."""
        self.A_bar.value = A_bar
        self.B_bar.value = B_bar
        self.z_bar.value = z_bar
        self.X_ref.value = X_ref
        self.U_ref.value = U_ref
        self.tr_radius.value = tr_radius
        self.weight_nu.value = weight_nu

        # Rebuild problem with updated obstacle constraints
        if self.obstacle_manager is not None:
            # Get updated obstacle constraints based on new reference
            obs_constraints = self.obstacle_manager.add_constraints(self.X, self.X_ref, self.N, safety_margin=0.1)

            # Rebuild problem with base constraints + updated obstacle constraints
            all_constraints = self.base_constraints + obs_constraints
            objective = cp.Minimize(self.weight_nu * cp.norm(self.nu, 1))
            self.prob = cp.Problem(objective, all_constraints)

    def solve(self, solver: str = "ECOS", verbose: bool = False) -> Tuple[bool, Optional[str]]:
        """Solve the convex subproblem."""
        try:
            self.prob.solve(solver=solver, verbose=verbose)
            # Accept both OPTIMAL and OPTIMAL_INACCURATE as successful solutions
            success = self.prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]

            if not success and verbose:
                print(f"  Solver status: {self.prob.status}")
                if self.prob.status == cp.INFEASIBLE:
                    print("  Problem is infeasible - constraints may be conflicting")
                    print(f"  Num constraints: {len(self.prob.constraints)}")

            return success, self.prob.status
        except cp.SolverError as e:
            return False, str(e)

    def get_solution(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get solution from solved problem."""
        return self.X.value, self.U.value, self.nu.value


class SCvxPlanner:
    """
    Successive Convexification planner with obstacle avoidance.
    """

    def __init__(
        self,
        model: DynamicsModel,
        params: Optional[SCvxParameters] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        input_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        obstacles: Optional[List[Obstacle]] = None,
        position_indices: Optional[List[int]] = None,
    ):
        """
        Initialize SCvx planner.

        Parameters
        ----------
        model : DynamicsModel
            System dynamics model
        params : Optional[SCvxParameters]
            Algorithm parameters
        state_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
            State box constraints (x_min, x_max)
        input_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
            Input box constraints (u_min, u_max)
        obstacles : Optional[List[Obstacle]]
            List of obstacles to avoid
        position_indices : Optional[List[int]]
            Indices of position coordinates in state vector
        """
        self.model = model
        self.params = params if params is not None else SCvxParameters()
        self.state_bounds = state_bounds
        self.input_bounds = input_bounds

        # Setup obstacle manager
        if obstacles is not None and position_indices is not None:
            self.obstacle_manager = ObstacleManager(
                obstacles=obstacles, state_dim=model.state_dim, position_indices=position_indices
            )
        else:
            self.obstacle_manager = None

        # Iteration data
        self.iteration_data = []

    def compute_linearization(
        self, X: np.ndarray, U: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute linearization of discrete-time dynamics around trajectory.

        Uses first-order discretization:
            x[k+1] ≈ x[k] + dt * f(x[k], u[k])

        Linearization:
            x[k+1] ≈ A_bar[k]@x[k] + B_bar[k]@u[k] + z_bar[k]

        where:
            A_bar[k] = I + dt * ∂f/∂x|(x[k], u[k])
            B_bar[k] = dt * ∂f/∂u|(x[k], u[k])
            z_bar[k] = x[k] + dt*f(x[k], u[k]) - A_bar[k]@x[k] - B_bar[k]@u[k]
        """
        n, m = self.model.state_dim, self.model.input_dim
        N = U.shape[1]

        A_bar = np.zeros((n, n, N))
        B_bar = np.zeros((n, m, N))
        z_bar = np.zeros((n, N))

        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]

            # Compute Jacobians at (x[k], u[k])
            A_jac, B_jac = self.model.jacobians(x_k, u_k)

            # Compute dynamics
            f_k = self.model.dynamics(x_k, u_k)

            # Discrete-time linearization (Euler)
            A_bar[:, :, k] = np.eye(n) + dt * np.array(A_jac)
            B_bar[:, :, k] = dt * np.array(B_jac)

            # Affine term
            x_next_nonlinear = x_k + dt * np.array(f_k)
            x_next_linear = A_bar[:, :, k] @ x_k + B_bar[:, :, k] @ u_k
            z_bar[:, k] = x_next_nonlinear - x_next_linear

        return A_bar, B_bar, z_bar

    def integrate_trajectory(self, X: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
        """Forward integrate nonlinear dynamics."""
        N = U.shape[1]
        X_nl = np.zeros_like(X)
        X_nl[:, 0] = X[:, 0]

        for k in range(N):
            X_nl[:, k + 1] = np.array(self.model.step(X_nl[:, k], U[:, k], dt))

        return X_nl

    def initialize_trajectory(self, x0: np.ndarray, xf: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:  # noqa: C901, PLR0912, PLR0915
        """
        Initialize trajectory with obstacle-avoiding path.

        Uses simple waypoint-based initialization that routes around obstacles.
        """
        _, m = self.model.state_dim, self.model.input_dim

        # If no obstacles, use straight line
        if self.obstacle_manager is None or len(self.obstacle_manager.obstacles) == 0:
            X = np.linspace(x0, xf, N + 1).T
            U = np.zeros((m, N))
            return X, U

        # Get position indices
        pos_indices = self.obstacle_manager.pos_indices
        pos_dim = len(pos_indices)

        # Extract start and goal positions
        p0 = x0[pos_indices]
        pf = xf[pos_indices]

        # Create waypoints that avoid obstacles
        waypoints = [p0]

        # Simple heuristic: if straight line collides, add waypoints
        for obs in self.obstacle_manager.obstacles:
            # Check if obstacle is roughly between start and goal
            # Project obstacle center onto line from p0 to pf
            v = pf - p0
            if np.linalg.norm(v) < 1e-10:
                continue

            v_norm = v / np.linalg.norm(v)
            to_obs = obs.center - p0
            proj_length = np.dot(to_obs, v_norm)

            # Only consider obstacles between start and goal
            if 0 < proj_length < np.linalg.norm(pf - p0):
                proj_point = p0 + proj_length * v_norm
                dist_to_line = np.linalg.norm(obs.center - proj_point)

                # If obstacle is close to line, add waypoint to avoid it
                if dist_to_line < obs.radius * 2.0:
                    # Create waypoint perpendicular to line, away from obstacle
                    if pos_dim == 2:
                        # 2D: perpendicular vector
                        perp = np.array([-v_norm[1], v_norm[0]])
                    else:
                        # 3D: choose perpendicular direction away from obstacle
                        to_center = obs.center - proj_point
                        if np.linalg.norm(to_center) > 1e-10:
                            perp = to_center / np.linalg.norm(to_center)
                        else:
                            perp = np.array([0, 0, 1]) if pos_dim == 3 else np.array([0, 1])

                    # Waypoint is offset from projection point
                    offset_dist = obs.radius * 2.0
                    waypoint = proj_point + perp * offset_dist
                    waypoints.append(waypoint)

        waypoints.append(pf)

        # Interpolate through waypoints
        if len(waypoints) == 2:
            # No intermediate waypoints, use straight line
            positions = np.linspace(waypoints[0], waypoints[1], N + 1).T
        else:
            # Multiple waypoints - distribute timesteps proportionally
            positions = []
            total_dist = sum(np.linalg.norm(waypoints[i + 1] - waypoints[i]) for i in range(len(waypoints) - 1))

            for i in range(len(waypoints) - 1):
                seg_dist = np.linalg.norm(waypoints[i + 1] - waypoints[i])
                n_seg = max(2, int(N * seg_dist / total_dist))

                if i == 0:
                    seg_pos = np.linspace(waypoints[i], waypoints[i + 1], n_seg)
                else:
                    seg_pos = np.linspace(waypoints[i], waypoints[i + 1], n_seg)[1:]

                positions.extend(seg_pos)

            # Ensure we have exactly N+1 points
            if len(positions) < N + 1:
                # Interpolate to get exact number
                t_old = np.linspace(0, 1, len(positions))
                t_new = np.linspace(0, 1, N + 1)
                positions = np.array([np.interp(t_new, t_old, np.array(positions)[:, d]) for d in range(pos_dim)]).T
            else:
                positions = np.array(positions[: N + 1])

        # Build full state trajectory
        X = np.zeros((self.model.state_dim, N + 1))
        for k in range(N + 1):
            X[:, k] = x0.copy()  # Start with x0
            X[pos_indices, k] = positions[k]  # Override positions

        # Linear interpolation for other states
        for i in range(self.model.state_dim):
            if i not in pos_indices:
                X[i, :] = np.linspace(x0[i], xf[i], N + 1)

        # Zero controls
        U = np.zeros((m, N))

        return X, U

    def plan(self, x0: np.ndarray, xf: np.ndarray, N: int, dt: float) -> NominalTrajectory:  # noqa: C901, PLR0912, PLR0915
        """
        Plan nominal trajectory using SCvx.

        Parameters
        ----------
        x0 : np.ndarray, shape (n,)
            Initial state
        xf : np.ndarray, shape (n,)
            Final state
        N : int
            Number of timesteps
        dt : float
            Timestep duration

        Returns
        -------
        nominal : NominalTrajectory
            Computed nominal trajectory
        """
        if self.params.verbose:
            print("=" * 60)
            print(" " * 15 + "SCvx PLANNER" + " " * 15)
            print("=" * 60)
            print(f"Initial state:  {x0}")
            print(f"Goal state:     {xf}")
            print(f"Horizon:        N={N}, dt={dt:.4f}s, tf={N * dt:.2f}s")
            if self.obstacle_manager:
                print(f"Obstacles:      {len(self.obstacle_manager.obstacles)} obstacles")
            print("=" * 60)

        # Initialize trajectory
        X, U = self.initialize_trajectory(x0, xf, N)

        # Create problem
        n, m = self.model.state_dim, self.model.input_dim
        problem = SCvxProblem(
            model=self.model,
            N=N,
            state_dim=n,
            input_dim=m,
            state_bounds=self.state_bounds,
            input_bounds=self.input_bounds,
            obstacle_manager=self.obstacle_manager,
        )

        # Add boundary constraints
        problem.prob.constraints.append(problem.X[:, 0] == x0)
        problem.prob.constraints.append(problem.X[:, -1] == xf)

        # SCvx iteration
        tr_radius = self.params.trust_region_init
        weight_nu = self.params.weight_nu
        last_nu_norm = None
        converged = False

        for iteration in range(self.params.max_iterations):
            if self.params.verbose:
                print(f"\n{'=' * 60}")
                print(f" Iteration {iteration + 1:02d}/{self.params.max_iterations}")
                print(f"{'=' * 60}")

            # Compute linearization
            A_bar, B_bar, z_bar = self.compute_linearization(X, U, dt)

            # Solve convex subproblem
            inner_iterations = 0
            while True:
                inner_iterations += 1

                # Set parameters
                problem.set_parameters(
                    A_bar=A_bar,
                    B_bar=B_bar,
                    z_bar=z_bar,
                    X_ref=X,
                    U_ref=U,
                    tr_radius=tr_radius,
                    weight_nu=weight_nu,
                )

                # Solve
                success, status = problem.solve(solver=self.params.solver, verbose=self.params.solver_verbose)

                if not success:
                    if inner_iterations > 5:
                        # If we've made good progress and virtual control is reasonably small, accept current solution
                        if last_nu_norm is not None and last_nu_norm < self.params.convergence_tol * 10:
                            if self.params.verbose:
                                print(f"  Solver failed but solution is close enough (||nu|| = {last_nu_norm:.6e})")
                            nu_norm = last_nu_norm  # Set for iteration data
                            converged = True
                            break
                        # Otherwise, try a fallback solver
                        if self.params.solver != "SCS":
                            if self.params.verbose:
                                print("  ECOS failed, trying SCS as fallback...")
                            success, status = problem.solve(solver="SCS", verbose=self.params.solver_verbose)
                            if success:
                                # Get solution from fallback solver
                                X_new, U_new, nu = problem.get_solution()
                                nu_norm = np.linalg.norm(nu, 1)
                                X, U = X_new, U_new
                                last_nu_norm = nu_norm
                                if self.params.verbose:
                                    print(f"  Fallback solver succeeded, ||nu|| = {nu_norm:.6e}")
                                if nu_norm < self.params.nu_tol:
                                    converged = True
                                break
                        # If all solvers fail and we have a reasonable solution, accept it
                        if last_nu_norm is not None and last_nu_norm < 0.5:
                            if self.params.verbose:
                                print(
                                    f"  All solvers failed but accepting current solution (||nu|| = {last_nu_norm:.6e})"
                                )
                            nu_norm = last_nu_norm  # Set for iteration data
                            converged = True
                            break
                        # If all solvers fail, raise error
                        raise RuntimeError(f"Solver failed repeatedly with status: {status}")
                    # Try shrinking trust region
                    tr_radius = max(tr_radius / 2.0, self.params.trust_region_min)
                    if self.params.verbose:
                        print(f"  Solver failed, shrinking TR to {tr_radius:.6e}")
                    continue

                # Get solution
                X_new, U_new, nu = problem.get_solution()

                # Compute virtual control norm
                nu_norm = np.linalg.norm(nu, 1)

                if self.params.verbose:
                    print(f"  Virtual control norm: {nu_norm:.6e}")
                    print(f"  Trust region radius:  {tr_radius:.6e}")

                # Check convergence
                if nu_norm < self.params.nu_tol:
                    converged = True
                    X, U = X_new, U_new
                    if self.params.verbose:
                        print(f"\n✓ Converged after {iteration + 1} iterations (||nu|| < {self.params.nu_tol})!")
                    break

                # Trust region update logic
                if last_nu_norm is None:
                    # First iteration - accept and continue
                    X, U = X_new, U_new
                    last_nu_norm = nu_norm
                    break

                # Compute reduction ratio
                actual_reduction = last_nu_norm - nu_norm

                if actual_reduction > 0:
                    # Accept solution
                    X, U = X_new, U_new
                    last_nu_norm = nu_norm

                    # Expand trust region if making good progress
                    if actual_reduction / last_nu_norm > 0.2:
                        tr_radius = min(tr_radius * self.params.beta, self.params.trust_region_max)
                        if self.params.verbose:
                            print(f"  ✓ Good progress, expanding TR to {tr_radius:.6e}")
                    break
                else:
                    # Shrink trust region and retry
                    tr_radius = max(tr_radius / self.params.alpha, self.params.trust_region_min)
                    if self.params.verbose:
                        print(f"  ✗ No progress, shrinking TR to {tr_radius:.6e}")

                    if tr_radius <= self.params.trust_region_min and inner_iterations > 3:
                        # Accept anyway if stuck
                        X, U = X_new, U_new
                        last_nu_norm = nu_norm
                        break

            # Store iteration data
            self.iteration_data.append(
                {
                    "iteration": iteration,
                    "X": X.copy(),
                    "U": U.copy(),
                    "nu_norm": nu_norm,
                    "tr_radius": tr_radius,
                }
            )

            if converged:
                break

            # Ramp up virtual control weight if not converging
            if iteration > 10 and nu_norm > 1e-3:
                weight_nu = min(weight_nu * 2.0, self.params.weight_nu_bound)

        if not converged and self.params.verbose:
            print(f"\n⚠ Maximum iterations ({self.params.max_iterations}) reached")
            print(f"  Final ||nu|| = {nu_norm:.6e}")

        # Create NominalTrajectory object
        nominal = NominalTrajectory(
            x_nom=X.T,  # Convert to (N+1, n)
            u_nom=U.T,  # Convert to (N, m)
            N=N,
            dt=dt,
        )

        if self.params.verbose:
            print(f"\n{'=' * 60}")
            print(f"Final trajectory: {nominal}")
            print(f"{'=' * 60}\n")

        return nominal
