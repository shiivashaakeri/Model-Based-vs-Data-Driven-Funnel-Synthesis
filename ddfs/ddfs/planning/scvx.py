# ddfs/ddfs/planning/scvx.py

"""
Successive Convexification (SCvx) Planner for Phase 1: Nominal Planning

This module implements the SCvx algorithm for trajectory optimization with
obstacle avoidance. It works with any system that implements the DynamicsModel
interface.

References
----------
[1] Mao et al., "Successive Convexification: A Superlinearly Convergent
    Algorithm for Non-convex Optimal Control Problems", 2018
[2] Szmuk & Açikmeşe, "Successive Convexification for 6-DoF Mars Rocket
    Powered Landing with Free-Final-Time", 2016
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cvxpy as cp
import jax.numpy as jnp
import numpy as np  # Only for CVXPY interface

from ddfs.models.base import DynamicsModel
from ddfs.planning.nominal_trajectory import NominalTrajectory


@dataclass
class SCvxParameters:
    """
    Parameters for the SCvx algorithm.

    Attributes
    ----------
    max_iterations : int
        Maximum number of SCvx iterations
    convergence_tol : float
        Convergence tolerance for predicted cost change
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
    verbose: bool = True
    solver: str = "ECOS"
    solver_verbose: bool = False


class SCvxProblem:
    """
    Convex subproblem for Successive Convexification.

    This class sets up and solves the convex optimization problem at each
    SCvx iteration. It handles the linearized dynamics, trust region
    constraints, and model-specific constraints.

    Parameters
    ----------
    model : DynamicsModel
        Dynamics model (must implement jacobians method)
    N : int
        Number of timesteps
    state_dim : int
        State dimension
    input_dim : int
        Input dimension
    state_bounds : Optional[Tuple[jnp.ndarray, jnp.ndarray]]
        State box constraints (x_min, x_max)
    input_bounds : Optional[Tuple[jnp.ndarray, jnp.ndarray]]
        Input box constraints (u_min, u_max)
    obstacle_constraint_fn : Optional[Callable]
        Function that adds obstacle avoidance constraints to CVXPY problem
    """

    def __init__(  # noqa: C901, PLR0912
        self,
        model: DynamicsModel,
        N: int,
        state_dim: int,
        input_dim: int,
        state_bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        input_bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        obstacle_constraint_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.N = N
        self.n = state_dim
        self.m = input_dim

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

        # Trust region constraint: ||X - X_ref||_1 + ||U - U_ref||_1 <= tr_radius
        dx = self.X - self.X_ref
        du = self.U - self.U_ref
        constraints.append(cp.norm(dx, 1) + cp.norm(du, 1) <= self.tr_radius)

        # State bounds (box constraints)
        # Convert JAX arrays to numpy for CVXPY
        if state_bounds is not None:
            x_min, x_max = state_bounds
            # Convert to numpy if needed
            if isinstance(x_min, jnp.ndarray):
                x_min = np.array(x_min)
            if isinstance(x_max, jnp.ndarray):
                x_max = np.array(x_max)
            for i in range(self.n):
                x_min_val = float(x_min[i])
                x_max_val = float(x_max[i])
                if not (np.isinf(x_min_val) or abs(x_min_val) > 1e10):
                    constraints.append(self.X[i, :] >= x_min_val)
                if not (np.isinf(x_max_val) or abs(x_max_val) > 1e10):
                    constraints.append(self.X[i, :] <= x_max_val)

        # Input bounds (box constraints)
        # Convert JAX arrays to numpy for CVXPY
        if input_bounds is not None:
            u_min, u_max = input_bounds
            # Convert to numpy if needed
            if isinstance(u_min, jnp.ndarray):
                u_min = np.array(u_min)
            if isinstance(u_max, jnp.ndarray):
                u_max = np.array(u_max)
            for i in range(self.m):
                u_min_val = float(u_min[i])
                u_max_val = float(u_max[i])
                if not (np.isinf(u_min_val) or abs(u_min_val) > 1e10):
                    constraints.append(self.U[i, :] >= u_min_val)
                if not (np.isinf(u_max_val) or abs(u_max_val) > 1e10):
                    constraints.append(self.U[i, :] <= u_max_val)

        # Obstacle avoidance constraints (if provided)
        if obstacle_constraint_fn is not None:
            constraints += obstacle_constraint_fn(self.X, self.X_ref)

        # Objective: minimize virtual control + optional model cost
        objective = cp.Minimize(self.weight_nu * cp.norm(self.nu, 1))

        # Create problem
        self.prob = cp.Problem(objective, constraints)

    def set_parameters(
        self,
        A_bar: jnp.ndarray,
        B_bar: jnp.ndarray,
        z_bar: jnp.ndarray,
        X_ref: jnp.ndarray,
        U_ref: jnp.ndarray,
        tr_radius: float,
        weight_nu: float,
    ):
        """
        Set parameter values for the convex subproblem.

        Parameters
        ----------
        A_bar : jnp.ndarray, shape (n, n, N)
            Linearized state-to-state matrices
        B_bar : jnp.ndarray, shape (n, m, N)
            Linearized input-to-state matrices
        z_bar : jnp.ndarray, shape (n, N)
            Affine terms from linearization
        X_ref : jnp.ndarray, shape (n, N+1)
            Reference state trajectory
        U_ref : jnp.ndarray, shape (m, N)
            Reference control trajectory
        tr_radius : float
            Trust region radius
        weight_nu : float
            Virtual control penalty weight
        """
        # Convert JAX arrays to numpy for CVXPY
        self.A_bar.value = np.array(A_bar)
        self.B_bar.value = np.array(B_bar)
        self.z_bar.value = np.array(z_bar)
        self.X_ref.value = np.array(X_ref)
        self.U_ref.value = np.array(U_ref)
        self.tr_radius.value = tr_radius
        self.weight_nu.value = weight_nu

    def solve(self, solver: str = "ECOS", verbose: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Solve the convex subproblem.

        Parameters
        ----------
        solver : str
            CVXPY solver name
        verbose : bool
            Print solver output

        Returns
        -------
        success : bool
            True if solver succeeded
        status : str
            Solver status message
        """
        try:
            self.prob.solve(solver=solver, verbose=verbose)
            # Accept both OPTIMAL and OPTIMAL_INACCURATE as success
            success = self.prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            return success, self.prob.status
        except cp.SolverError as e:
            return False, str(e)

    def get_solution(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get solution from solved problem.

        Returns
        -------
        X : jnp.ndarray, shape (n, N+1)
            Optimal state trajectory
        U : jnp.ndarray, shape (m, N)
            Optimal control trajectory
        nu : jnp.ndarray, shape (n, N)
            Virtual control values
        """
        # Convert numpy arrays from CVXPY to JAX arrays
        return jnp.array(self.X.value), jnp.array(self.U.value), jnp.array(self.nu.value)


class SCvxPlanner:
    """
    Successive Convexification planner for trajectory optimization.

    This planner computes a nominal trajectory from x0 to xf while avoiding
    obstacles and satisfying constraints. It works with any DynamicsModel.

    Parameters
    ----------
    model : DynamicsModel
        System dynamics model
    params : SCvxParameters
        Algorithm parameters
    state_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        State box constraints (x_min, x_max)
    input_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        Input box constraints (u_min, u_max)
    obstacle_constraint_fn : Optional[Callable]
        Function to add obstacle avoidance constraints
    """

    def __init__(
        self,
        model: DynamicsModel,
        params: Optional[SCvxParameters] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        input_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        obstacle_constraint_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.params = params if params is not None else SCvxParameters()
        self.state_bounds = state_bounds
        self.input_bounds = input_bounds
        self.obstacle_constraint_fn = obstacle_constraint_fn

        # Iteration data
        self.iteration_data = []

    def compute_linearization(
        self, X: jnp.ndarray, U: jnp.ndarray, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        Parameters
        ----------
        X : jnp.ndarray, shape (n, N+1)
            State trajectory
        U : jnp.ndarray, shape (m, N)
            Control trajectory
        dt : float
            Timestep

        Returns
        -------
        A_bar : jnp.ndarray, shape (n, n, N)
            Linearized state matrices
        B_bar : jnp.ndarray, shape (n, m, N)
            Linearized input matrices
        z_bar : jnp.ndarray, shape (n, N)
            Affine terms
        """
        n, m = self.model.state_dim, self.model.input_dim
        N = U.shape[1]

        A_bar = jnp.zeros((n, n, N))
        B_bar = jnp.zeros((n, m, N))
        z_bar = jnp.zeros((n, N))

        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]

            # Compute Jacobians at (x[k], u[k])
            A_jac, B_jac = self.model.jacobians(x_k, u_k)

            # Compute dynamics
            f_k = self.model.dynamics(x_k, u_k)

            # Discrete-time linearization (Euler)
            A_bar = A_bar.at[:, :, k].set(jnp.eye(n) + dt * A_jac)
            B_bar = B_bar.at[:, :, k].set(dt * B_jac)

            # Affine term
            x_next_nonlinear = x_k + dt * f_k
            x_next_linear = A_bar[:, :, k] @ x_k + B_bar[:, :, k] @ u_k
            z_bar = z_bar.at[:, k].set(x_next_nonlinear - x_next_linear)

        return A_bar, B_bar, z_bar

    def integrate_trajectory(self, X: jnp.ndarray, U: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Forward integrate nonlinear dynamics.

        Parameters
        ----------
        X : jnp.ndarray, shape (n, N+1)
            Initial state trajectory (only X[:, 0] is used)
        U : jnp.ndarray, shape (m, N)
            Control trajectory
        dt : float
            Timestep

        Returns
        -------
        X_nl : jnp.ndarray, shape (n, N+1)
            Nonlinear trajectory
        """
        N = U.shape[1]
        X_nl = jnp.zeros_like(X)
        X_nl = X_nl.at[:, 0].set(X[:, 0])

        for k in range(N):
            X_nl = X_nl.at[:, k + 1].set(self.model.step(X_nl[:, k], U[:, k], dt))

        return X_nl

    def initialize_trajectory(self, x0: jnp.ndarray, xf: jnp.ndarray, N: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize trajectory with straight-line interpolation.

        Parameters
        ----------
        x0 : jnp.ndarray, shape (n,)
            Initial state
        xf : jnp.ndarray, shape (n,)
            Final state
        N : int
            Number of timesteps

        Returns
        -------
        X : jnp.ndarray, shape (n, N+1)
            Initial state trajectory
        U : jnp.ndarray, shape (m, N)
            Initial control trajectory (zeros)
        """
        _, m = self.model.state_dim, self.model.input_dim

        # Linear interpolation for states
        # jnp.linspace doesn't support multi-dimensional, so use manual interpolation
        # Use vmap or manual loop with proper JAX operations
        n = len(x0)
        X = jnp.zeros((n, N + 1))
        for i in range(n):
            X = X.at[i, :].set(jnp.linspace(x0[i], xf[i], N + 1))

        # Zero controls
        U = jnp.zeros((m, N))

        return X, U

    def plan(self, x0: jnp.ndarray, xf: jnp.ndarray, N: int, dt: float) -> NominalTrajectory:  # noqa: C901, PLR0912, PLR0915
        """
        Plan nominal trajectory using SCvx.

        Parameters
        ----------
        x0 : jnp.ndarray, shape (n,)
            Initial state
        xf : jnp.ndarray, shape (n,)
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
        # Convert to numpy for CVXPY boundary constraints
        x0_np = np.array(x0)
        xf_np = np.array(xf)
        if self.params.verbose:
            print("=" * 60)
            print(" " * 15 + "SCvx PLANNER" + " " * 15)
            print("=" * 60)
            print(f"Initial state:  {x0}")
            print(f"Goal state:     {xf}")
            print(f"Horizon:        N={N}, dt={dt:.4f}s, tf={N * dt:.2f}s")
            print("=" * 60)

        # Initialize trajectory
        X, U = self.initialize_trajectory(x0, xf, N)

        # Add boundary conditions to problem
        n, m = self.model.state_dim, self.model.input_dim
        problem = SCvxProblem(
            model=self.model,
            N=N,
            state_dim=n,
            input_dim=m,
            state_bounds=self.state_bounds,
            input_bounds=self.input_bounds,
            obstacle_constraint_fn=self.obstacle_constraint_fn,
        )

        # Add boundary constraints (convert to numpy for CVXPY)
        problem.prob.constraints.append(problem.X[:, 0] == x0_np)
        problem.prob.constraints.append(problem.X[:, -1] == xf_np)

        # SCvx iteration
        tr_radius = self.params.trust_region_init
        last_nonlinear_cost = None
        converged = False

        for iteration in range(self.params.max_iterations):
            if self.params.verbose:
                print(f"\n{'=' * 60}")
                print(f" Iteration {iteration + 1:02d}/{self.params.max_iterations}")
                print(f"{'=' * 60}")

            # Compute linearization
            A_bar, B_bar, z_bar = self.compute_linearization(X, U, dt)

            # Solve convex subproblem
            max_trust_region_retries = 10  # Maximum retries per iteration
            trust_region_retries = 0

            while trust_region_retries < max_trust_region_retries:
                # Set parameters
                problem.set_parameters(
                    A_bar=A_bar,
                    B_bar=B_bar,
                    z_bar=z_bar,
                    X_ref=X,
                    U_ref=U,
                    tr_radius=tr_radius,
                    weight_nu=self.params.weight_nu,
                )

                # Solve
                success, status = problem.solve(solver=self.params.solver, verbose=self.params.solver_verbose)

                if not success:
                    # If solver failed, try with a more robust solver as fallback
                    if self.params.solver != "SCS":
                        if self.params.verbose:
                            print(f"  ⚠ Solver {self.params.solver} failed, trying SCS...")
                        success, status = problem.solve(solver="SCS", verbose=self.params.solver_verbose)

                    if not success:
                        raise RuntimeError(f"Solver failed with status: {status}")

                # Get solution
                X_new, U_new, nu = problem.get_solution()

                # Integrate nonlinear dynamics
                X_nl = self.integrate_trajectory(X_new, U_new, dt)

                # Compute costs (using JAX)
                linear_cost = float(self.params.weight_nu * jnp.linalg.norm(nu, ord=1))
                nonlinear_cost = float(jnp.linalg.norm(X_new - X_nl, ord=1))

                if self.params.verbose:
                    print(f"  Linear cost (virtual):  {linear_cost:.6e}")
                    print(f"  Nonlinear cost (model): {nonlinear_cost:.6e}")
                    print(f"  Trust region radius:    {tr_radius:.6e}")

                # First iteration
                if last_nonlinear_cost is None:
                    last_nonlinear_cost = nonlinear_cost
                    X, U = X_new, U_new
                    break

                # Compute changes
                actual_change = last_nonlinear_cost - nonlinear_cost
                predicted_change = last_nonlinear_cost - linear_cost

                if self.params.verbose:
                    print(f"  Actual change:          {actual_change:.6e}")
                    print(f"  Predicted change:       {predicted_change:.6e}")

                # Check convergence
                if abs(predicted_change) < self.params.convergence_tol:
                    converged = True
                    X, U = X_new, U_new
                    if self.params.verbose:
                        print(f"\n✓ Converged after {iteration + 1} iterations!")
                    break

                # Trust region update
                rho = actual_change / predicted_change if abs(predicted_change) > 1e-12 else 0.0

                if rho < self.params.rho_0:
                    # Reject solution, shrink trust region
                    old_tr_radius = tr_radius
                    tr_radius = max(tr_radius / self.params.alpha, self.params.trust_region_min)

                    # If trust region can't shrink further, accept solution anyway
                    if abs(tr_radius - old_tr_radius) < 1e-10 and tr_radius <= self.params.trust_region_min:
                        if self.params.verbose:
                            print(f"  ⚠ Trust region at minimum, accepting solution despite rho={rho:.4f}")
                        X, U = X_new, U_new
                        last_nonlinear_cost = nonlinear_cost
                        break

                    trust_region_retries += 1
                    if self.params.verbose:
                        print(f"  ✗ Solution rejected (rho={rho:.4f} < {self.params.rho_0})")
                        print(f"    Shrinking trust region to {tr_radius:.6e}")
                else:
                    # Accept solution
                    X, U = X_new, U_new
                    last_nonlinear_cost = nonlinear_cost

                    if rho < self.params.rho_1:
                        # Shrink trust region
                        tr_radius = max(tr_radius / self.params.alpha, self.params.trust_region_min)
                        if self.params.verbose:
                            print(f"  ✓ Solution accepted (rho={rho:.4f})")
                            print(f"    Shrinking trust region to {tr_radius:.6e}")
                    elif rho >= self.params.rho_2:
                        # Expand trust region
                        tr_radius = min(tr_radius * self.params.beta, self.params.trust_region_max)
                        if self.params.verbose:
                            print(f"  ✓ Solution accepted (rho={rho:.4f})")
                            print(f"    Expanding trust region to {tr_radius:.6e}")
                    # Keep trust region
                    elif self.params.verbose:
                        print(f"  ✓ Solution accepted (rho={rho:.4f})")

                    break

            # If we exhausted trust region retries, accept the last solution
            if trust_region_retries >= max_trust_region_retries:
                if self.params.verbose:
                    print("  Maximum trust region retries reached, accepting solution")
                X, U = X_new, U_new
                last_nonlinear_cost = nonlinear_cost

            # Store iteration data
            # Convert JAX arrays to numpy for storage (JAX arrays are immutable)
            self.iteration_data.append(
                {
                    "iteration": iteration,
                    "X": np.array(X),
                    "U": np.array(U),
                    "linear_cost": linear_cost,
                    "nonlinear_cost": nonlinear_cost,
                    "tr_radius": tr_radius,
                }
            )

            if converged:
                break

        if not converged and self.params.verbose:
            print(f"\n⚠ Maximum iterations ({self.params.max_iterations}) reached without convergence")

        # Enforce boundary conditions exactly (solver might have small numerical errors)
        X = X.at[:, 0].set(x0)
        X = X.at[:, -1].set(xf)
        # Normalize states if model has normalize_state method (for angles/quaternions)
        if hasattr(self.model, 'normalize_state'):
            for k in range(N + 1):
                x_k_norm = self.model.normalize_state(X[:, k])
                X = X.at[:, k].set(x_k_norm)

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
