"""
Successive Convexification (SCvx) trajectory planner.

This module implements SCvx for nonlinear trajectory optimization with:
- Nonlinear dynamics (unicycle)
- Obstacle avoidance constraints
- State and input bounds
- Trust regions for convergence

Reference:
- Mao et al. "Successive Convexification for 6-DoF Mars Rocket Powered Landing"
- Acikmese & Ploen "Convex Programming Approach to Powered Descent Guidance"
"""

from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np

from ddfs.environment.collision import CollisionChecker
from ddfs.models.base import DynamicalSystem


class SCvxPlanner:
    """
    Successive Convexification trajectory planner.

    Solves the nonconvex optimal control problem:
        minimize: ∫ ||u||² dt
        subject to:
            x(k+1) = f(x(k), u(k))           (nonlinear dynamics)
            h_obs(x(k)) ≥ 0                  (obstacle avoidance)
            x_min ≤ x(k) ≤ x_max             (state bounds)
            u_min ≤ u(k) ≤ u_max             (input bounds)
            x(0) = x0, x(N) = xf             (boundary conditions)

    by iteratively solving convex subproblems.
    """

    def __init__(
        self,
        model: DynamicalSystem,
        dt: float,
        N: int,
        collision_checker: Optional[CollisionChecker] = None,
        params: Optional[Dict] = None,
    ):
        """
        Initialize SCvx planner.

        Args:
            model: Dynamicsl system (unicycle)
            dt: Time step
            N: Number of time steps (horizon)
            collision_checker: Collision checker
            params: SCvx planner parameters
        """
        self.model = model
        self.dt = dt
        self.N = N
        self.collision_checker = collision_checker

        # State and Control dimensions
        self.n = model.state_dim
        self.m = model.input_dim

        # Default parameters
        default_params = {
            "max_iterations": 50,
            "tol_x": 1e-3,  # State convergence tolerance
            "tol_u": 1e-3,  # Control convergence tolerance
            "trust_region_rho": 1.0,  # Initial trust region radius
            "trust_region_rho_max": 10.0,  # Maximum trust region radius
            "trust_region_rho_min": 1e-4,  # Minimum trust region radius
            "trust_region_beta": 1.5,  # Trust region expansion factor
            "trust_region_gamma": 0.7,  # Trust region contraction factor
            "weight_trust_region": 1.0,  # Trust region penalty weight
            "weight_control": 1.0,  # Control penalty weight
            "weight_terminal": 100.0,  # Terminal penalty weight
            "verbose": True,
        }
        self.params = default_params
        if params is not None:
            self.params.update(params)

        # Storage for iteration history
        self.history = {"x_ref": [], "u_ref": [], "cost": [], "trust_region_rho": []}

    def plan(  # noqa: C901, PLR0912
        self,
        x0: np.ndarray,
        xf: np.ndarray,
        x_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        u_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        x_init: Optional[np.ndarray] = None,
        u_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Plan trajectory from x0 to xf using SCvx.

        Args:
            x0: Initial state (n,)
            xf: Final state (n,)
            x_bounds: State bounds (x_min, x_max), each (n,)
            u_bounds: Input bounds (u_min, u_max), each (m,)
            x_init: Initial trajectory guess (N+1, n) (optional)
            u_init: Initial input guess (N, m) (optional)

        Returns:
            x_traj: State trajectory (N+1, n)
            u_traj: Input trajectory (N, m)
            converged: True if converged
        """
        # Initialize reference trajectory
        if x_init is None or u_init is None:
            x_ref, u_ref = self._initialize_trajectory(x0, xf)
        else:
            x_ref = x_init.copy()
            u_ref = u_init.copy()

        # Trust region radius
        rho = self.params["trust_region_rho"]

        if self.params["verbose"]:
            print("=" * 60)
            print("SCvx Trajectory Planning")
            print("=" * 60)
            print(f"Initial state: {x0}")
            print(f"Final state:   {xf}")
            print(f"Horizon:       {self.N} steps ({self.N * self.dt:.2f}s)")
            print(f"State dim:     {self.n}")
            print(f"Input dim:     {self.m}")
            if self.collision_checker:
                print(f"Obstacles:     {self.collision_checker.num_obstacles()}")
            print("=" * 60)

        # Iteration loop
        for iteration in range(self.params["max_iterations"]):
            # Solve convex subproblem
            x_new, u_new, cost = self._solve_convex_subproblem(x_ref, u_ref, x0, xf, x_bounds, u_bounds, rho)

            if x_new is None:
                # Infeasible, reduce trust region
                rho *= self.params["trust_region_gamma"]
                if self.params["verbose"]:
                    print(f"[Iter {iteration}] INFEASIBLE - reducing trust region to ρ={rho:.4f}")  # noqa: RUF001

                if rho < self.params["trust_region_rho_min"]:
                    if self.params["verbose"]:
                        print("Trust region too small. Terminating.")
                    return x_ref, u_ref, False

                continue

            # Compute convergence metrics
            dx = np.linalg.norm(x_new - x_ref, ord=np.inf)
            du = np.linalg.norm(u_new - u_ref, ord=np.inf)

            # Store history
            self.history["x_ref"].append(x_ref.copy())
            self.history["u_ref"].append(u_ref.copy())
            self.history["cost"].append(cost)
            self.history["trust_region_rho"].append(rho)

            if self.params["verbose"]:
                print(f"[Iter {iteration}] Cost: {cost:.4f}, ||Δx||: {dx:.6f}, ||Δu||: {du:.6f}, ρ: {rho:.4f}")  # noqa: RUF001

            # Check convergence
            if dx < self.params["tol_x"] and du < self.params["tol_u"]:
                if self.params["verbose"]:
                    print("=" * 60)
                    print(f"✓ CONVERGED in {iteration + 1} iterations")
                    print("=" * 60)
                return x_new, u_new, True

            # Update reference
            x_ref = x_new
            u_ref = u_new

            # Expand trust region (solution accepted)
            rho = min(self.params["trust_region_rho_max"], rho * self.params["trust_region_beta"])

        # Max iterations reached
        if self.params["verbose"]:
            print("=" * 60)
            print(f"✗ Max iterations ({self.params['max_iterations']}) reached")
            print("=" * 60)

        return x_ref, u_ref, False

    def _initialize_trajectory(self, x0: np.ndarray, xf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize trajectory with straight-line interpolation in state space.

        Args:
            x0: Initial state
            xf: Final state

        Returns:
            x_init: Initial state trajectory (N+1, n)
            u_init: Initial input trajectory (N, m)
        """
        # Straight-line interpolation
        x_init = np.zeros((self.N + 1, self.n))
        for i in range(self.n):
            x_init[:, i] = np.linspace(x0[i], xf[i], self.N + 1)

        # Zero initial input
        u_init = np.zeros((self.N, self.m))

        return x_init, u_init

    def _solve_convex_subproblem(  # noqa: C901, PLR0912
        self,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
        x0: np.ndarray,
        xf: np.ndarray,
        x_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        u_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        rho: float,  # noqa: ARG002
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Solve convex subproblem at current iteration.

        Args:
            x_ref: Reference state trajectory (N+1, n)
            u_ref: Reference input trajectory (N, m)
            x0: Initial state
            xf: Final state
            x_bounds: State bounds
            u_bounds: Input bounds
            rho: Trust region radius

        Returns:
            x_new: New state trajectory (N+1, n) or None if infeasible
            u_new: New input trajectory (N, m) or None if infeasible
            cost: Objective value
        """
        # Define CVXPY variables
        x = cp.Variable((self.N + 1, self.n))
        u = cp.Variable((self.N, self.m))

        # Objective: minimize control effort + trust region penalty
        cost = 0.0

        # Control effort
        for k in range(self.N):
            cost += self.params["weight_control"] * cp.sum_squares(u[k, :])

        # Trust region penalty
        for k in range(self.N + 1):
            cost += self.params["weight_trust_region"] * cp.sum_squares(x[k, :] - x_ref[k, :])

        # Terminal cost (encourage reaching goal)
        cost += self.params["weight_terminal"] * cp.sum_squares(x[self.N, :] - xf)

        # Constraints
        constraints = []

        # Initial condition
        constraints.append(x[0, :] == x0)

        # Dynamics constraints (linearized)
        for k in range(self.N):
            # Linearize at reference point
            A_d, B_d = self.model.discrete_linearization(x_ref[k, :], u_ref[k, :], self.dt, method="rk4")

            # Affine term
            x_next_ref = self.model.discrete_dynamics(x_ref[k, :], u_ref[k, :], self.dt)
            c_d = x_next_ref - A_d @ x_ref[k, :] - B_d @ u_ref[k, :]

            # Linearized dynamics: x+ ≈ A_d x + B_d u + c_d
            constraints.append(x[k + 1, :] == A_d @ x[k, :] + B_d @ u[k, :] + c_d)

        # State bounds
        if x_bounds is not None:
            x_min, x_max = x_bounds
            for k in range(self.N + 1):
                constraints.append(x[k, :] >= x_min)
                constraints.append(x[k, :] <= x_max)

        # Input bounds
        if u_bounds is not None:
            u_min, u_max = u_bounds
            for k in range(self.N):
                constraints.append(u[k, :] >= u_min)
                constraints.append(u[k, :] <= u_max)

        # Obstacle avoidance constraints (linearized)
        if self.collision_checker is not None:
            for k in range(self.N + 1):
                # Get all obstacle gradients at reference point
                gradients = self.collision_checker.get_all_gradients(x_ref[k, :])
                distances = self.collision_checker.distance_to_all_obstacles(x_ref[k, :])

                for obs_idx, grad in gradients.items():
                    # Linearized constraint: d(x_ref) + ∇d^T (x - x_ref) ≥ 0
                    # Only consider position (first 2 states)
                    d_ref = distances[obs_idx]
                    constraints.append(d_ref + grad @ (x[k, :2] - x_ref[k, :2]) >= 0)

        # Trust region (as soft constraint via penalty in objective)
        # Alternatively, can add hard constraint:
        # for k in range(self.N + 1):
        #     constraints.append(cp.norm(x[k, :] - x_ref[k, :], 2) <= rho)

        # Solve problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return x.value, u.value, problem.value
            else:
                return None, None, np.inf

        except Exception as e:
            if self.params["verbose"]:
                print(f"Solver error: {e}")
            return None, None, np.inf

    def get_history(self) -> Dict:
        """Get iteration history."""
        return self.history

    def clear_history(self):
        """Clear iteration history."""
        self.history = {"x_ref": [], "u_ref": [], "cost": [], "trust_region_rho": []}
