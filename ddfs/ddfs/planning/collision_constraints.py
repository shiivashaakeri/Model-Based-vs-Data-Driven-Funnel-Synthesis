"""
Collision constraint linearization for SCvx.

This module provides utilities for linearizing nonconvex obstacle avoidance
constraints around reference trajectories for use in convex subproblems.
"""

from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

from ddfs.environment.collision import CollisionChecker


class CollisionConstraintLinearizer:
    """
    Linearize collision avoidance constraints for SCvx.

    For each obstacle with signed distance function d(x):
        d(x) >= 0  (nonconvex)

    Linearize around reference trajectory x_ref(k):
        d(x_ref(k)) + ∇d(x_ref(k))^T (x - x_ref(k)) >= 0 (convex, linear)
    """

    def __init__(self, collision_checker: CollisionChecker):
        """
        Initialize collision constraint linearizer.

        Args:
            collision_checker: Collision checker for obstacles
        """
        self.collision_checker = collision_checker
        self.num_obstacles = collision_checker.num_obstacles()

    def linearize_at_point(self, x: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Linearize all obstacle constraints at a single point.

        For obstacle i: d_i(x) = d_i(x_ref) + ∇d_i(x_ref)^T (x - x_ref)

        Args:
            x: State point [px, py, ...]

        Returns:
            linearizations: Dict mapping obstacle index to:
                - 'distance': d(x_ref)
                - 'gradient': ∇d(x_ref)
                - 'offset': d(x_ref) + ∇d(x_ref)^T (x_ref)
        """
        linearizations = {}

        # Get distances and gradients for all obstacles
        distances = self.collision_checker.distance_to_all_obstacles(x)
        gradients = self.collision_checker.get_all_gradients(x)

        for obs_idx in distances:
            d_ref = distances[obs_idx]
            grad = gradients[obs_idx]

            # Compute offset: b = d(x_ref) + ∇d(x_ref)^T (x_ref)
            # So that: d = ∇d^T x + b
            offset = d_ref - grad @ x[:2]

            linearizations[obs_idx] = {"distance": d_ref, "gradient": grad, "offset": offset}

        return linearizations

    def linearize_trajectory(self, x_traj: np.ndarray) -> List[Dict[int, Dict[str, np.ndarray]]]:
        """
        Linearize obstacle constraints along entire trajectory.

        Args:
            x_traj: State trajectory (N+1, n)

        Returns:
            linearizations: List of linearizations for each timestep
        """
        N = x_traj.shape[0] - 1
        trajectory_linearizations = []

        for k in range(N + 1):
            lin_k = self.linearize_at_point(x_traj[k])
            trajectory_linearizations.append(lin_k)

        return trajectory_linearizations

    def build_cvxpy_constraints(
        self,
        x_var: cp.Variable,
        x_ref: np.ndarray,
        timestep: int,
        linearizations: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    ) -> List:
        """
        Build CVXPY constraints for obstacle avoidance at a timestep.

        Args:
            x_var: CVXPY variable for state (N+1, n)
            x_ref: Reference state trajectory (N+1, n)
            timestep: Current timestep index
            linearizations: Precomputed linearizations (optional)
        Returns:
            constraints: List of CVXPY constraints
        """
        if linearizations is None:
            linearizations = self.linearize_at_point(x_ref[timestep])

        constraints = []

        for obs_idx, lin in linearizations.items():
            d_ref = lin["distance"]
            grad = lin["gradient"]

            constraints.append(grad @ x_var[:2] >= grad @ x_ref[timestep, :2] - d_ref)
        return constraints

    def check_constraint_violation(
        self, x_traj: np.ndarray, tolerance: float = 0.0
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Check if trajectory violates any obstacle constraints.

        Args:
            x_traj: State trajectory (N+1, n)
            tolerance: Safety tolerance (meters)

        Returns:
            violated: True if any constraints violated
            violations: List of (timestep, obstacle_index) tuples where violation occurred
        """
        violations = []

        for k in range(x_traj.shape[0]):
            # Check collision at each timestep
            colliding = self.collision_checker.get_colliding_obstacles(x_traj[k], tolerance)

            for obs_idx, _ in colliding:
                violations.append((k, obs_idx))

        return len(violations) > 0, violations

    def compute_minimum_clearance(self, x_traj: np.ndarray) -> Tuple[float, int, int]:
        """
        Compute minimum clearance to obstacles along trajectory.

        Args:
            x_traj: State trajectory (N+1, n)

        Returns:
            min_clearance: Minimum distance to nearest obstacle
            timestep: Index of timestep with minimum clearance
            obstacle_idx: Index of nearest obstacle
        """
        return self.collision_checker.get_trajectory_clearance(x_traj)

    def visualize_linearizations(
        self, x_ref: np.ndarray, obs_idx: int, grid_resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute linearized distance function on a grid for visualization.

        Args:
            x_ref: Reference state (n,)
            obs_idx: Obstacle index
            grid_resolution: Number of points in each dimension

        Returns:
            X: Grid x-coordinates (grid_resolution, grid_resolution)
            Y: Grid y-coordinates (grid_resolution, grid_resolution)
            D_linear: Linearized distances (grid_resolution, grid_resolution)
        """
        linearization = self.linearize_at_point(x_ref)

        if obs_idx not in linearization:
            raise ValueError(f"Obstacle {obs_idx} not found.")

        lin = linearization[obs_idx]
        d_ref = lin["distance"]
        grad = lin["gradient"]

        # Create grid
        obstacles = self.collision_checker.get_obstacles()
        lower, upper = obstacles[obs_idx].get_bounding_box()
        margin = 1.0
        x_range = np.linspace(lower[0] - margin, upper[0] + margin, grid_resolution)
        y_range = np.linspace(lower[1] - margin, upper[1] + margin, grid_resolution)
        X, Y = np.meshgrid(x_range, y_range)

        # Compute linearized distances
        D_linear = np.zeros_like(X)
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                x_grid = np.array([X[i, j], Y[i, j]])
                D_linear[i, j] = d_ref + grad @ (x_grid - x_ref[:2])

        return X, Y, D_linear


class AdaptiveCollisionMargin:
    """
    Adaptive safety margin for obstacle avoidance.

    Increases margin when clearance is low, decreases when trajectory is safe.
    """

    def __init__(
        self,
        margin_init: float = 0.1,
        margin_min: float = 0.05,
        margin_max: float = 0.5,
        clearance_threshold: float = 0.2,
    ):
        """
        Initialize adaptive margin manager.

        Args:
            margin_init: Initial safety margin (meters)
            margin_min: Minimum safety margin (meters)
            margin_max: Maximum safety margin (meters)
            clearance_threshold: Clearance below which margin is increased
        """

        self.margin = margin_init
        self.margin_min = margin_min
        self.margin_max = margin_max
        self.clearance_threshold = clearance_threshold

        self.history = []

    def update(self, min_clearance: float) -> float:
        """
        Update margin based on minimum clearance.

        Args:
            min_clearance: Minimum clearance along trajectory

        Returns:
            margin: Updated safety margin (meters)
        """
        if min_clearance < self.clearance_threshold:
            # Increase margin
            self.margin = min(self.margin_max, self.margin * 1.2)
        elif min_clearance > 2 * self.clearance_threshold:
            # Decrease margin
            self.margin = max(self.margin_min, self.margin * 0.9)

        self.history.append((min_clearance, self.margin))
        return self.margin

    def get_margin(self) -> float:
        """Get current safety margin."""
        return self.margin

    def reset(self):
        """Reset margin to initial value."""
        self.margin = 0.1
        self.history = []


class CollisionConstraintRelaxation:
    """
    Soft constraint relaxation for obstacle avoidance.

    Allows temporary constraint violations with penalty:
        d(x) + slack >= 0
        minimize: ... + penalty * ||slack||
    """

    @staticmethod
    def add_slack_variable(N: int, num_obstacles: int) -> cp.Variable:
        """
        Create slack variables for constraint relaxation.

        Args:
            N: Number of timesteps
            num_obstacles: Number of obstacles

        Returns:
            slack: Slack variables (N+1, num_obstacles)
        """
        return cp.Variable((N + 1, num_obstacles), nonneg=True)

    @staticmethod
    def build_relaxed_constraints(
        x_var: cp.Variable,
        slack_var: cp.Variable,
        x_ref: np.ndarray,
        timestep: int,
        obs_idx: int,
        linearizations: Dict[str, np.ndarray],
    ) -> cp.Constraint:
        """
        Build relaxed collision constraint with slack.

        Args:
            x_var: CVXPY variable for state (N+1, n)
            slack_var: Slack variables (N+1, num_obstacles)
            x_ref: Reference state trajectory (N+1, n)
            timestep: Current timestep index
            obs_idx: Obstacle index
            linearizations: Precomputed linearizations data (distance, gradient, offset)
        Returns:
            constraint: Relaxed collision constraint
        """
        d_ref = linearizations[obs_idx]["distance"]
        grad = linearizations[obs_idx]["gradient"]

        return grad @ x_var[:2] + slack_var[timestep, obs_idx] >= grad @ x_ref[timestep, :2] - d_ref

    @staticmethod
    def compute_slack_penalty(slack_var: cp.Variable, weight: float = 1e6) -> cp.Expression:
        """
        Compute slack penalty term for relaxed constraints.

        Args:
            slack_var: Slack variables (N+1, num_obstacles)
            weight: Penalty weight

        Returns:
            penalty: Penalty term for objective
        """
        return weight * cp.sum(slack_var)
