"""
Successive Convexification (SCvx) Planner for DDFS.

This module implements the SCvx algorithm for generating nominal trajectories
on the digital twin model, including:
- Linearization of nonlinear dynamics around reference trajectory
- Trust region management (SOC constraints)
- Virtual control for dynamics constraint relaxation
- Linearized obstacle avoidance
- Convergence checking and warm-starting

Reference: Mao, Y., Szmuk, M., & Açikmeşe, B. (2016). Successive convexification
of non-convex optimal control problems and its convergence properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np

from ddfs.core.constraints import StateInputConstraints
from ddfs.core.obstacles import ObstacleCollection
from ddfs.models.base_model import BaseModel
from ddfs.planning.base_planner import (
    BasePlanner,
    ConstraintSet,
    ConvergenceCriteria,
    ConvergenceStatus,
    CostFunction,
    OptimizationProblem,
    PlannerResult,
    QuadraticCost,
    SubproblemResult,
)
from ddfs.utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)


# =============================================================================
# SCvx Configuration
# =============================================================================


@dataclass
class SCvxConfig:
    """
    Configuration for SCvx algorithm.

    Parameters
    ----------
    max_iterations : int
        Maximum number of SCvx iterations.
    cost_tolerance : float
        Relative cost change tolerance for convergence.
    constraint_tolerance : float
        Maximum constraint violation for convergence.
    trust_region_init : float
        Initial trust region radius.
    trust_region_min : float
        Minimum trust region radius (triggers failure).
    trust_region_max : float
        Maximum trust region radius.
    trust_region_shrink : float
        Trust region shrink factor (when step rejected).
    trust_region_expand : float
        Trust region expand factor (when step is good).
    rho_accept : float
        Minimum ratio for accepting a step.
    rho_good : float
        Ratio threshold for expanding trust region.
    virtual_control_weight : float
        Penalty weight for virtual control (dynamics slack).
    virtual_buffer_weight : float
        Penalty weight for virtual buffer (obstacle slack).
    state_constraint_weight : float
        Penalty weight for state constraint slack.
    input_constraint_weight : float
        Penalty weight for input constraint slack.
    use_soft_state_constraints : bool
        Use soft state constraints.
    use_soft_input_constraints : bool
        Use soft input constraints.
    use_soft_obstacle_constraints : bool
        Use soft obstacle constraints.
    obstacle_margin : float
        Safety margin for obstacle avoidance.
    """

    # Convergence
    max_iterations: int = 50
    cost_tolerance: float = 1e-6
    constraint_tolerance: float = 1e-4

    # Trust region
    trust_region_init: float = 1.0
    trust_region_min: float = 1e-4
    trust_region_max: float = 10.0
    trust_region_shrink: float = 0.5
    trust_region_expand: float = 1.5
    rho_accept: float = 0.1
    rho_good: float = 0.5

    # Penalty weights
    virtual_control_weight: float = 1e5
    virtual_buffer_weight: float = 1e5
    state_constraint_weight: float = 1e4
    input_constraint_weight: float = 1e4

    # Constraint handling
    use_soft_state_constraints: bool = True
    use_soft_input_constraints: bool = False
    use_soft_obstacle_constraints: bool = True
    obstacle_margin: float = 0.0

    def to_convergence_criteria(self) -> ConvergenceCriteria:
        """Convert to ConvergenceCriteria."""
        return ConvergenceCriteria(
            max_iterations=self.max_iterations,
            cost_tolerance=self.cost_tolerance,
            constraint_tolerance=self.constraint_tolerance,
            trust_region_tolerance=self.trust_region_min,
        )


# =============================================================================
# SCvx Subproblem
# =============================================================================


class SCvxSubproblem(OptimizationProblem):
    """
    Single convex subproblem for SCvx iteration.

    Solves the linearized optimal control problem with trust region constraints.
    """

    def __init__(
        self,
        n_states: int,
        n_inputs: int,
        N: int,
        dt: float,
        config: SCvxConfig,
    ):
        super().__init__(n_states, n_inputs, N, dt)
        self.config = config

        # Reference trajectory (linearization point)
        self._x_ref: Optional[np.ndarray] = None
        self._u_ref: Optional[np.ndarray] = None

        # Linearized dynamics matrices (A_k, B_k, c_k for each k)
        self._A: Optional[List[np.ndarray]] = None
        self._B: Optional[List[np.ndarray]] = None
        self._c: Optional[List[np.ndarray]] = None

        # Virtual control (dynamics slack)
        self._nu: Optional[cp.Variable] = None

        # Trust region radius
        self._trust_region: float = config.trust_region_init

        # Obstacle information
        self._obstacle_gradients: Optional[List[np.ndarray]] = None
        self._obstacle_values: Optional[List[float]] = None

    @property
    def nu(self) -> cp.Variable:
        """Virtual control variable (N, n_states)."""
        if self._nu is None:
            self._nu = cp.Variable((self.N, self.n_states), name="nu")
        return self._nu

    def set_reference(
        self,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
    ) -> None:
        """
        Set reference trajectory for linearization.

        Parameters
        ----------
        x_ref : np.ndarray
            Reference state trajectory (N+1, n_states).
        u_ref : np.ndarray
            Reference input trajectory (N, n_inputs).
        """
        self._x_ref = x_ref.copy()
        self._u_ref = u_ref.copy()

    def set_linearization(
        self,
        A: List[np.ndarray],
        B: List[np.ndarray],
        c: List[np.ndarray],
    ) -> None:
        """
        Set linearized dynamics matrices.

        Linearized dynamics: x_{k+1} = A_k @ x_k + B_k @ u_k + c_k

        Parameters
        ----------
        A : list of np.ndarray
            State matrices for each timestep (N elements).
        B : list of np.ndarray
            Input matrices for each timestep (N elements).
        c : list of np.ndarray
            Affine terms for each timestep (N elements).
        """
        self._A = A
        self._B = B
        self._c = c

    def set_obstacle_linearization(
        self,
        gradients: List[np.ndarray],
        values: List[float],
    ) -> None:
        """
        Set linearized obstacle constraints.

        Linearized constraint: grad^T @ (x - x_ref) + value >= 0

        Parameters
        ----------
        gradients : list of np.ndarray
            Signed distance gradients at reference points.
        values : list of float
            Signed distance values at reference points.
        """
        self._obstacle_gradients = gradients
        self._obstacle_values = values

    def set_trust_region(self, radius: float) -> None:
        """Set trust region radius."""
        self._trust_region = radius

    def setup(  # noqa: C901, PLR0912, PLR0915
        self,
        x_init: np.ndarray,
        x_target: np.ndarray,
        cost_function: CostFunction,
        state_constraints: Optional[StateInputConstraints] = None,
        position_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Set up the convex subproblem.

        Parameters
        ----------
        x_init : np.ndarray
            Initial state constraint.
        x_target : np.ndarray
            Target terminal state.
        cost_function : CostFunction
            Cost function to minimize.
        state_constraints : StateInputConstraints, optional
            State and input constraints.
        position_indices : list, optional
            Indices of position states for obstacle avoidance.
        """
        self._constraints = ConstraintSet()

        # Reset variables
        self._x = cp.Variable((self.N + 1, self.n_states), name="x")
        self._u = cp.Variable((self.N, self.n_inputs), name="u")
        self._nu = cp.Variable((self.N, self.n_states), name="nu")

        # =====================================================================
        # Cost function
        # =====================================================================
        cost = 0.0

        # Running cost
        for k in range(self.N):
            x_ref_k = self._x_ref[k] if self._x_ref is not None else None
            u_ref_k = self._u_ref[k] if self._u_ref is not None else None
            cost += cost_function.running_cost(self._x[k], self._u[k], k, x_ref_k, u_ref_k)

        # Terminal cost
        cost += cost_function.terminal_cost(self._x[self.N], x_target)

        # Virtual control penalty (L1 norm)
        cost += self.config.virtual_control_weight * cp.sum(cp.abs(self._nu))

        # =====================================================================
        # Initial state constraint
        # =====================================================================
        self._constraints.add_hard_constraint(self._x[0] == x_init)

        # =====================================================================
        # Linearized dynamics constraints with virtual control
        # =====================================================================
        if self._A is not None and self._B is not None and self._c is not None:
            for k in range(self.N):
                # x_{k+1} = A_k @ x_k + B_k @ u_k + c_k + nu_k
                dynamics = self._A[k] @ self._x[k] + self._B[k] @ self._u[k] + self._c[k] + self._nu[k]
                self._constraints.add_hard_constraint(self._x[k + 1] == dynamics)

        # =====================================================================
        # Trust region constraints (SOC)
        # =====================================================================
        if self._x_ref is not None and self._u_ref is not None:
            for k in range(self.N + 1):
                # ||x_k - x_ref_k||_2 <= trust_region
                self._constraints.add_hard_constraint(cp.norm(self._x[k] - self._x_ref[k], 2) <= self._trust_region)

            for k in range(self.N):
                # ||u_k - u_ref_k||_2 <= trust_region
                self._constraints.add_hard_constraint(cp.norm(self._u[k] - self._u_ref[k], 2) <= self._trust_region)

        # =====================================================================
        # State and input constraints
        # =====================================================================
        if state_constraints is not None:
            # StateInputConstraints uses state_constraint and input_constraint (singular)
            x_min = state_constraints.state_constraint.lb
            x_max = state_constraints.state_constraint.ub
            u_min = state_constraints.input_constraint.lb
            u_max = state_constraints.input_constraint.ub

            for k in range(self.N + 1):
                if self.config.use_soft_state_constraints:
                    # Soft state constraints with slack
                    slack_x_lb = cp.Variable(self.n_states, nonneg=True)
                    slack_x_ub = cp.Variable(self.n_states, nonneg=True)
                    self._constraints.add_hard_constraint(self._x[k] >= x_min - slack_x_lb)
                    self._constraints.add_hard_constraint(self._x[k] <= x_max + slack_x_ub)
                    cost += self.config.state_constraint_weight * (cp.sum(slack_x_lb) + cp.sum(slack_x_ub))
                else:
                    # Hard state constraints
                    self._constraints.add_hard_constraint(self._x[k] >= x_min)
                    self._constraints.add_hard_constraint(self._x[k] <= x_max)

            for k in range(self.N):
                if self.config.use_soft_input_constraints:
                    # Soft input constraints with slack
                    slack_u_lb = cp.Variable(self.n_inputs, nonneg=True)
                    slack_u_ub = cp.Variable(self.n_inputs, nonneg=True)
                    self._constraints.add_hard_constraint(self._u[k] >= u_min - slack_u_lb)
                    self._constraints.add_hard_constraint(self._u[k] <= u_max + slack_u_ub)
                    cost += self.config.input_constraint_weight * (cp.sum(slack_u_lb) + cp.sum(slack_u_ub))
                else:
                    # Hard input constraints
                    self._constraints.add_hard_constraint(self._u[k] >= u_min)
                    self._constraints.add_hard_constraint(self._u[k] <= u_max)

        # =====================================================================
        # Linearized obstacle constraints
        # =====================================================================
        if self._obstacle_gradients is not None and self._obstacle_values is not None and position_indices is not None:
            for k in range(self.N + 1):
                if k < len(self._obstacle_gradients):
                    grad = self._obstacle_gradients[k]
                    val = self._obstacle_values[k]

                    # Extract position from state
                    pos = self._x[k, position_indices]
                    pos_ref = self._x_ref[k, position_indices]

                    # Linearized constraint: grad^T @ (pos - pos_ref) + val >= margin
                    lhs = grad @ (pos - pos_ref) + val

                    if self.config.use_soft_obstacle_constraints:
                        slack_obs = cp.Variable(nonneg=True)
                        self._constraints.add_hard_constraint(lhs >= self.config.obstacle_margin - slack_obs)
                        cost += self.config.virtual_buffer_weight * slack_obs
                    else:
                        self._constraints.add_hard_constraint(lhs >= self.config.obstacle_margin)

        self._objective = cost
        self._build_problem()

    def solve(
        self,
        warm_start: bool = True,
        verbose: bool = False,
    ) -> SubproblemResult:
        """Solve the subproblem."""
        result = super().solve(warm_start=warm_start, verbose=verbose)

        # Add virtual control info
        if result.is_solved and self._nu.value is not None:
            result.slack_dynamics = self._nu.value
            result.solver_info["virtual_control_norm"] = np.sum(np.abs(self._nu.value))

        return result


# =============================================================================
# SCvx Planner
# =============================================================================


class SCvxPlanner(BasePlanner):
    """
    Successive Convexification (SCvx) trajectory planner.

    Generates optimal trajectories for nonlinear systems by iteratively
    solving convex subproblems with trust region constraints.
    """

    def __init__(
        self,
        model: BaseModel,
        config: Optional[SCvxConfig] = None,
    ):
        """
        Initialize SCvx planner.

        Parameters
        ----------
        model : BaseModel
            Dynamical system model (digital twin).
        config : SCvxConfig, optional
            SCvx configuration. Uses defaults if not provided.
        """
        super().__init__(
            n_states=model.n_states,
            n_inputs=model.n_inputs,
            dt=model.dt,
            N=0,  # Set during planning
        )

        self.model = model
        self.config = config or SCvxConfig()
        self._convergence = self.config.to_convergence_criteria()

        # Position indices for obstacle avoidance
        self._position_indices: Optional[List[int]] = None

    def set_position_indices(self, indices: List[int]) -> None:
        """Set position indices for obstacle avoidance."""
        self._position_indices = indices

    def plan(  # noqa: C901, PLR0912, PLR0915
        self,
        x_init: np.ndarray,
        x_target: np.ndarray,
        N: int,
        u_init_guess: Optional[np.ndarray] = None,
        x_init_guess: Optional[np.ndarray] = None,
        **kwargs,  # noqa: ARG002
    ) -> PlannerResult:
        """
        Plan trajectory using SCvx.

        Parameters
        ----------
        x_init : np.ndarray
            Initial state.
        x_target : np.ndarray
            Target state.
        N : int
            Number of timesteps.
        u_init_guess : np.ndarray, optional
            Initial input trajectory guess.
        x_init_guess : np.ndarray, optional
            Initial state trajectory guess.
        **kwargs
            Additional parameters.

        Returns
        -------
        PlannerResult
            Planning result with trajectory.
        """
        self.N = N
        self.reset_history()

        with Timer("SCvx total") as total_timer:
            # Generate initial guess if not provided
            x_ref, u_ref = self._generate_initial_guess(x_init, x_target, N, x_init_guess, u_init_guess)

            # Initialize trust region
            trust_region = self.config.trust_region_init

            # Best solution tracking
            best_x = x_ref.copy()
            best_u = u_ref.copy()
            best_cost = np.inf
            best_nonlinear_cost = np.inf

            # Main SCvx loop
            converged = False
            status = ConvergenceStatus.NOT_STARTED

            for iteration in range(self.config.max_iterations):
                # =============================================================
                # Step 1: Linearize dynamics around reference
                # =============================================================
                A, B, c = self._linearize_dynamics(x_ref, u_ref)

                # =============================================================
                # Step 2: Linearize obstacle constraints
                # =============================================================
                obs_grads, obs_vals = self._linearize_obstacles(x_ref)

                # =============================================================
                # Step 3: Setup and solve convex subproblem
                # =============================================================
                subproblem = SCvxSubproblem(
                    n_states=self.n_states,
                    n_inputs=self.n_inputs,
                    N=N,
                    dt=self.dt,
                    config=self.config,
                )

                subproblem.set_solver(self._solver, **self._solver_options)
                subproblem.set_reference(x_ref, u_ref)
                subproblem.set_linearization(A, B, c)
                subproblem.set_trust_region(trust_region)

                if obs_grads is not None:
                    subproblem.set_obstacle_linearization(obs_grads, obs_vals)

                subproblem.setup(
                    x_init=x_init,
                    x_target=x_target,
                    cost_function=self._cost_function,
                    state_constraints=self._state_constraints,
                    position_indices=self._position_indices,
                )

                # Warm start from previous solution
                subproblem.warm_start_from(x_ref, u_ref)

                # Debug: Check problem details before solving
                print(f"Iteration {iteration}:")
                print(f"  Num constraints: {len(subproblem._constraints._constraints)}")
                print(f"  Num variables: x={subproblem._x.shape}, u={subproblem._u.shape}")

                # Solve
                result = subproblem.solve(warm_start=True, verbose=False)
                self._iteration_history.append(result)

                # Debug: Check result after solving
                print(f"  Problem status: {result.status}")
                print(f"  Solver info: {result.solver_info}")

                if not result.is_solved:
                    logger.warning(f"Iteration {iteration}: Subproblem failed, shrinking trust region")
                    # Shrink trust region and retry WITH SAME REFERENCE
                    trust_region *= self.config.trust_region_shrink
                    if trust_region < self.config.trust_region_min:
                        status = ConvergenceStatus.TRUST_REGION_FAILED
                        break
                    # Continue with same x_ref, u_ref but smaller trust region
                    continue

                # =============================================================
                # Step 4: Evaluate actual cost improvement
                # =============================================================
                x_new = result.x
                u_new = result.u
                predicted_cost = result.cost

                # Compute actual nonlinear cost
                actual_cost = self._compute_nonlinear_cost(x_new, u_new, x_target)

                # Virtual control violation
                vc_violation = result.solver_info.get("virtual_control_norm", 0.0)

                # Log iteration
                self._log_iteration(iteration, actual_cost, vc_violation, trust_region)

                # =============================================================
                # Step 5: Trust region update (only after first iteration)
                # =============================================================
                accept_step = True

                if iteration > 0 and best_nonlinear_cost < np.inf:
                    # Compute improvement ratio
                    predicted_improvement = best_nonlinear_cost - predicted_cost
                    actual_improvement = best_nonlinear_cost - actual_cost

                    if abs(predicted_improvement) > 1e-10:
                        rho = actual_improvement / predicted_improvement
                    else:
                        rho = 1.0 if actual_improvement >= 0 else 0.0

                    # Debug: Print improvement metrics
                    print(f"  Predicted improvement: {predicted_improvement:.6e}")
                    print(f"  Actual improvement: {actual_improvement:.6e}")
                    print(f"  Rho: {rho:.4f}")
                    print(f"  Trust region before: {trust_region:.6f}")

                    # Update trust region based on ratio
                    if rho < self.config.rho_accept:
                        # Bad step - shrink trust region
                        trust_region *= self.config.trust_region_shrink
                        accept_step = False
                        logger.debug(
                            f"Iter {iteration}: Step rejected (rho={rho:.3f}), TR -> {trust_region:.4f}"
                        )

                        if trust_region < self.config.trust_region_min:
                            status = ConvergenceStatus.TRUST_REGION_FAILED
                            break
                        # Don't update reference, retry with same reference and smaller trust region
                        continue

                    elif rho > self.config.rho_good:
                        # Good step - expand trust region
                        trust_region = min(
                            trust_region * self.config.trust_region_expand,
                            self.config.trust_region_max,
                        )
                        logger.debug(f"Iter {iteration}: Good step, TR -> {trust_region:.4f}")

                # =============================================================
                # Step 6: Accept step and update reference
                # =============================================================
                if accept_step:
                    # Update reference
                    x_ref = x_new.copy()
                    u_ref = u_new.copy()

                    # Update best solution
                    if actual_cost < best_nonlinear_cost:
                        best_x = x_new.copy()
                        best_u = u_new.copy()
                        best_cost = predicted_cost  # noqa: F841
                        best_nonlinear_cost = actual_cost

                # =============================================================
                # Step 7: Check convergence
                # =============================================================
                converged, status = self._check_convergence(
                    iteration,
                    actual_cost,
                    best_nonlinear_cost if iteration > 0 else actual_cost * 2,
                    vc_violation,
                    trust_region,
                )

                if converged:
                    break

            # End of main loop
            if not converged and status == ConvergenceStatus.NOT_STARTED:
                status = ConvergenceStatus.MAX_ITERATIONS

        # Create result trajectory
        if best_x is not None and best_u is not None:
            trajectory = self._create_trajectory(
                best_x,
                best_u,
                metadata={
                    "planner": "SCvx",
                    "iterations": iteration + 1,
                    "final_cost": best_nonlinear_cost,
                    "status": status.value,
                },
            )
        else:
            trajectory = None

        return PlannerResult(
            trajectory=trajectory,
            status=status,
            cost=best_nonlinear_cost,
            iterations=iteration + 1,
            total_time=total_timer.elapsed,
            iteration_history=self._iteration_history,
            convergence_history=self._convergence_history,
            metadata={
                "config": self.config.__dict__,
                "final_trust_region": trust_region,
            },
        )

    def _generate_initial_guess(
        self,
        x_init: np.ndarray,
        x_target: np.ndarray,
        N: int,
        x_guess: Optional[np.ndarray] = None,
        u_guess: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial trajectory guess.

        Uses provided guesses or creates linear interpolation.
        """
        if x_guess is not None and u_guess is not None:
            return x_guess.copy(), u_guess.copy()

        # Linear interpolation for states
        if x_guess is None:
            x_guess = np.zeros((N + 1, self.n_states))
            for i in range(N + 1):
                alpha = i / N
                x_guess[i] = (1 - alpha) * x_init + alpha * x_target

        # Zero or small constant for inputs
        if u_guess is None:
            u_guess = np.zeros((N, self.n_inputs))

            # Try to compute a reasonable initial input
            # For hover-like systems, use equilibrium input
            if hasattr(self.model, "hover_input"):
                u_hover = self.model.hover_input()
                u_guess[:] = u_hover

        return x_guess, u_guess

    def _linearize_dynamics(
        self,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Linearize dynamics around reference trajectory.

        Returns A_k, B_k, c_k such that:
        x_{k+1} ≈ A_k @ x_k + B_k @ u_k + c_k

        where the linearization is:
        x_{k+1} ≈ f(x_ref_k, u_ref_k) + A_k @ (x_k - x_ref_k) + B_k @ (u_k - u_ref_k)
                = A_k @ x_k + B_k @ u_k + (f(x_ref_k, u_ref_k) - A_k @ x_ref_k - B_k @ u_ref_k)
        """
        A_list = []
        B_list = []
        c_list = []

        for k in range(self.N):
            x_k = x_ref[k]
            u_k = u_ref[k]

            # Get Jacobians at reference point
            A_k, B_k = self.model.discrete_jacobians(x_k, u_k)

            # Compute affine term
            # c_k = f(x_ref_k, u_ref_k) - A_k @ x_ref_k - B_k @ u_ref_k
            f_k = self.model.discrete_dynamics(x_k, u_k)
            c_k = f_k - A_k @ x_k - B_k @ u_k

            A_list.append(A_k)
            B_list.append(B_k)
            c_list.append(c_k)

        return A_list, B_list, c_list

    def _linearize_obstacles(
        self,
        x_ref: np.ndarray,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
        """
        Linearize obstacle constraints around reference trajectory.

        Returns gradients and values of signed distance function.
        """
        if self._obstacles is None or self._position_indices is None:
            return None, None

        if self._obstacles.n_obstacles == 0:
            return None, None

        gradients = []
        values = []

        for k in range(self.N + 1):
            pos = x_ref[k, self._position_indices]

            # Get minimum signed distance and its gradient
            min_dist, grad = self._compute_obstacle_gradient(pos)

            gradients.append(grad)
            values.append(min_dist)

        return gradients, values

    def _compute_obstacle_gradient(
        self,
        position: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute signed distance and gradient to nearest obstacle.

        Parameters
        ----------
        position : np.ndarray
            Position in workspace.

        Returns
        -------
        distance : float
            Signed distance (negative inside obstacle).
        gradient : np.ndarray
            Gradient of signed distance w.r.t. position.
        """
        min_dist = np.inf
        min_grad = np.zeros(len(position))

        for obstacle in self._obstacles.obstacles:
            dist = obstacle.signed_distance(position)

            if dist < min_dist:
                min_dist = dist
                # Numerical gradient of signed distance
                min_grad = obstacle.gradient_signed_distance(position)

        return min_dist, min_grad

    def _compute_nonlinear_cost(
        self,
        x: np.ndarray,
        u: np.ndarray,
        x_target: np.ndarray,
    ) -> float:
        """
        Compute actual nonlinear cost by simulating trajectory.

        This propagates through the true nonlinear dynamics to check
        the actual cost achieved.
        """
        if self._cost_function is None:
            return 0.0

        # Simulate to get actual trajectory
        x_sim = np.zeros_like(x)
        x_sim[0] = x[0]

        for k in range(self.N):
            x_sim[k + 1] = self.model.discrete_dynamics(x_sim[k], u[k])

        # Compute cost (using numpy, not cvxpy)
        cost = 0.0

        if isinstance(self._cost_function, QuadraticCost):
            Q = self._cost_function.Q
            R = self._cost_function.R
            Q_f = self._cost_function.Q_f

            for k in range(self.N):
                x_err = x_sim[k]
                u_err = u[k]
                cost += x_err @ Q @ x_err + u_err @ R @ u_err

            x_err_f = x_sim[self.N] - x_target
            cost += x_err_f @ Q_f @ x_err_f

        # Add dynamics violation penalty
        dynamics_violation = 0.0
        for k in range(self.N):
            x_next_actual = self.model.discrete_dynamics(x[k], u[k])
            dynamics_violation += np.sum(np.abs(x[k + 1] - x_next_actual))

        cost += self.config.virtual_control_weight * dynamics_violation

        return cost

    def _check_convergence(
        self,
        iteration: int,
        cost: float,
        cost_prev: float,
        constraint_violation: float,
        trust_region: float,
    ) -> Tuple[bool, ConvergenceStatus]:
        """
        Check convergence criteria.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        cost : float
            Current cost.
        cost_prev : float
            Previous cost.
        constraint_violation : float
            Virtual control norm (constraint violation).
        trust_region : float
            Current trust region size.

        Returns
        -------
        converged : bool
            True if converged.
        status : ConvergenceStatus
            Convergence status.
        """
        # Max iterations
        if iteration >= self.config.max_iterations - 1:
            return True, ConvergenceStatus.MAX_ITERATIONS

        # Trust region too small
        if trust_region < self.config.trust_region_min:
            return True, ConvergenceStatus.TRUST_REGION_FAILED

        # EARLY CONVERGENCE: If virtual control is essentially zero, we're done!
        # This means the linearized dynamics match the nonlinear dynamics well
        if constraint_violation < 1e-6:
            logger.info(f"Early convergence: virtual control norm = {constraint_violation:.2e}")
            return True, ConvergenceStatus.CONVERGED

        # Standard convergence check
        relative_change = abs(cost - cost_prev) / abs(cost_prev) if cost_prev > 1e-10 else abs(cost - cost_prev)

        if relative_change < self.config.cost_tolerance and constraint_violation < self.config.constraint_tolerance:
            return True, ConvergenceStatus.CONVERGED

        return False, ConvergenceStatus.NOT_STARTED


# =============================================================================
# Factory Functions
# =============================================================================


def create_scvx_planner(
    model: BaseModel,
    config: Optional[SCvxConfig] = None,
    cost_function: Optional[CostFunction] = None,
    constraints: Optional[StateInputConstraints] = None,
    obstacles: Optional[ObstacleCollection] = None,
    position_indices: Optional[List[int]] = None,
) -> SCvxPlanner:
    """
    Create and configure SCvx planner.

    Parameters
    ----------
    model : BaseModel
        Dynamical system model.
    config : SCvxConfig, optional
        SCvx configuration.
    cost_function : CostFunction, optional
        Cost function.
    constraints : StateInputConstraints, optional
        State and input constraints.
    obstacles : ObstacleCollection, optional
        Obstacles for avoidance.
    position_indices : list, optional
        Position state indices.

    Returns
    -------
    SCvxPlanner
        Configured planner.
    """
    planner = SCvxPlanner(model, config)

    if cost_function is not None:
        planner.set_cost_function(cost_function)

    if constraints is not None:
        planner.set_constraints(constraints, obstacles)

    if position_indices is not None:
        planner.set_position_indices(position_indices)

    return planner


def create_scvx_planner_from_config(
    model: BaseModel,
    config,  # System config from ddfs.core.config
) -> SCvxPlanner:
    """
    Create SCvx planner from system configuration.

    Parameters
    ----------
    model : BaseModel
        Dynamical system model.
    config : Config
        System configuration.

    Returns
    -------
    SCvxPlanner
        Configured planner.
    """
    from ddfs.core.workspace import workspace_from_config  # noqa: PLC0415

    # Create SCvx config from system config
    scvx_config = SCvxConfig(
        max_iterations=config.scvx.max_iterations,
        cost_tolerance=config.scvx.convergence_tolerance,
        constraint_tolerance=config.scvx.convergence_tolerance,
        trust_region_init=config.scvx.trust_region_init,
        trust_region_min=config.scvx.trust_region_min,
        trust_region_max=config.scvx.trust_region_max,
    )

    # Create workspace
    workspace = workspace_from_config(config)

    # Determine position indices
    if hasattr(model, "position_indices"):
        position_indices = model.position_indices
    else:
        position_indices = [0, 1] if model.n_states >= 2 else [0]

    # Create cost function
    Q = np.diag(config.cost.Q_diag)
    R = np.diag(config.cost.R_diag)
    Q_f = np.diag(config.cost.Qf_diag)
    cost_function = QuadraticCost(Q=Q, R=R, Q_f=Q_f)

    planner = SCvxPlanner(model, scvx_config)
    planner.set_cost_function(cost_function)
    planner.set_constraints(workspace.constraints, workspace.obstacles)
    planner.set_position_indices(position_indices)

    return planner
