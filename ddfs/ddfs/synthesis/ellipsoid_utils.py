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
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse


@dataclass
class EllipsoidConstraints:
    """
    Container for ellipsoid constraint envelopes.

    Attributes:
        P_min_k: List of P_min ellipsoids for each timestep
        R_max_k: List of R_max ellipsoids for each timestep
        P_min_i: List of P_min ellipsoids for each segment
        R_max_i: List of R_max ellipsoids for each segment
        segment_times: List of times for each segment
        metadata: Additional information (obstacles, constraints, etc.)
    """

    P_min_k: List[np.ndarray]
    R_max_k: List[np.ndarray]
    P_min_i: List[np.ndarray]
    R_max_i: List[np.ndarray]
    segment_times: List[float]
    metadata: Dict = field(default_factory=dict)

    def get_segment_envelope(self, segment_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get P_min,i and R_max,i for a given segment."""
        return self.P_min_i[segment_idx], self.R_max_i[segment_idx]

    def save(self, filepath: Path):
        """Save constraints to pickle file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f" Saved ellipsoid constraints to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "EllipsoidConstraints":
        """Load constraints from pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class ObstacleDistanceComputer:
    """
    Utility for computing distances and gradients of obstacles.

    Supports:
        - Circular obstacles
        - Ellipsoidal obstacles
    """

    @staticmethod
    def distance_to_circle(
        x: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> float:
        """
        Compute signed distance from a point x to a circular obstacle.

        Positive distance means outside obstacle (safe).

        Args:
            x: Point [px, py, ...]
            center: Obstacle Center [cx, cy]
            radius: Obstacle Radius

        Returns:
            Signed signed distance (positive=safe)
        """
        return np.linalg.norm(x[:2] - center) - radius

    @staticmethod
    def gradient_distance_to_circle(
        x: np.ndarray,
        center: np.ndarray,
        radius: float,  # noqa: ARG004
    ) -> np.ndarray:
        """
        Compute gradient of distance to circular obstacle.

        nabla dist = (x-center) / ||x-center||

        Args:
            x: Point [px, py, ...]
            center: Obstacle Center [cx, cy]
            radius: Obstacle Radius

        Returns:
            Gradient [d/px, d/py, ...]
        """
        n = len(x)
        grad = np.zeros(n)

        diff = x[:2] - center
        norm_diff = np.linalg.norm(diff)

        if norm_diff > 1e-10:
            grad[:2] = diff / norm_diff

        return grad

    @staticmethod
    def distance_to_ellipsoid(
        x: np.ndarray,
        center: np.ndarray,
        semi_axes: np.ndarray,
        rotation: float = 0.0,
    ) -> float:
        """
        Compute approximate signed distance to ellipsoidal obstacle.

        Uses the algebraic distance as approximation:
        dist ≈ 1 - sqrt(η^T Q η)

        where η = R^T (x - center), Q = diag(1/a^2, 1/b^2), R is rotation.

        Args:
            x: Point (at least 2D)
            center: Ellipse center (2D)
            semi_axes: Semi-axes [a, b]
            rotation: Rotation angle (radians)

        Returns:
            Signed distance (positive = safe)
        """
        diff = x[:2] - center
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        eta = R.T @ diff

        scaled = eta / semi_axes
        algebraic_dist = np.linalg.norm(scaled)
        return algebraic_dist - 1.0

    @staticmethod
    def gradient_distance_to_ellipsoid(
        x: np.ndarray,
        center: np.ndarray,
        semi_axes: np.ndarray,
        rotation: float = 0.0,
    ) -> np.ndarray:
        """
        Compute gradient of distance to ellipsoidal obstacle.

        ∇dist ≈ 2 * R * Q * R^T * (x - center) / sqrt(...)

        Args:
            x: Point (at least 2D)
            center: Ellipse center (2D)
            semi_axes: Semi-axes [a, b]
            rotation: Rotation angle (radians)

        Returns:
            Gradient (n_states,)
        """
        n = len(x)
        grad = np.zeros(n)

        diff = x[:2] - center
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        eta = R.T @ diff

        Q = np.diag(1.0 / semi_axes**2)
        grad_eta = 2 * Q @ eta

        grad_xy = R @ grad_eta

        norm_grad = np.linalg.norm(grad_xy)
        if norm_grad > 1e-10:
            grad[:2] = grad_xy / norm_grad

        return grad


class EllipsoidUtility:
    """
    Utility class for computing and manipulating ellipsoids.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_controls: int = 2,
        x_max: float = 10.0,
        u_max: float = 2.0,
        solver: str = "ECOS",
        safety_margin: float = 0.1,
    ):
        """
        Initialize ellipsoid utility.

        Args:
            n_states: State dimension
            n_controls: Control dimension
            x_max: Maximum state value
            u_max: Maximum control value
            solver: Solver to use
            safety_margin: Safety margin for obstacles
        """
        self.n_states = n_states
        self.n_controls = n_controls
        self.x_max = x_max
        self.u_max = u_max
        self.solver = solver
        self.safety_margin = safety_margin
        self.distance_computer = ObstacleDistanceComputer()

    def compute_P_min(
        self,
        x_nom: np.ndarray,
        state_constraints: Dict,
        obstacles: Optional[List[Dict]] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute P_min(k): largest inscribed state ellipsoid at nominal x_nom.

        Supports obstacles in addition to box constraints.

        Args:
            x_nom: Nominal state [x, y, ...]
            state_constraints: Dict of state constraints
            obstacles: List of obstacle dictionaries
            verbose: Print solver details

        Returns:
            P_min: Positive definite matrix (n_states, n_states)
            success: True if successful
        """
        n = self.n_states

        # inearize all constraints at x_nom
        A_x, b_x = self.linearize_all_state_constraints(x_nom, state_constraints, obstacles)

        if A_x.shape[0] == 0:
            P_min = np.eye(n) / (self.x_max**2)
            return P_min, True

        Z = cp.Variable((n, n), symmetric=True)
        constraints = [Z >> 0]

        n_active = 0
        for i in range(A_x.shape[0]):
            a_j = A_x[i, :]
            b_j = b_x[i]
            if b_j > 1e-6:
                constraints.append(cp.norm(Z @ a_j, 2) <= b_j)
                n_active += 1
        if n_active == 0:
            warnings.warn("No active state constraints at x_nom. Using default ellipsoid.")
            P_min = np.eye(n) / (self.x_max**2)
            return P_min, True

        constraints.append(cp.trace(Z) <= self.x_max * np.eye(n))
        objective = cp.Maximize(cp.log_det(Z))

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self.solver, verbose=verbose)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                warnings.warn(f"P_min optimization failed with status: {problem.status}")
                P_min = np.eye(n) * 100.0  # Very small ellipsoid (conservative)
                return P_min, False

            Z_opt = Z.value
            Z_inv = np.linalg.inv(Z_opt)
            P_min = Z_inv.T @ Z_inv

            P_min = P_min / (P_min + P_min.T)
            eig_vals = np.linalg.eigvals(P_min)
            if np.min(eig_vals) <= 0:
                P_min += (abs(np.min(eig_vals)) + 1e-6) * np.eye(n)

            return P_min, True

        except Exception as e:
            warnings.warn(f"P_min optimization failed with error: {e}")
            P_min = np.eye(n) * 100.0  # Very small ellipsoid (conservative)
            return P_min, False

    def compute_R_max(
        self,
        u_nom: np.ndarray,
        input_constraints: Dict,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute R_max(k): largest inscribed input ellipsoid at nominal u_nom.

         (No changes from basic version - inputs don't have obstacle constraints)

        Args:
            u_nom: Nominal control (m,)
            input_constraints: Dict with constraint functions and bounds
            verbose: Print solver output

        Returns:
            R_max: Positive semi-definite matrix (m, m)
            success: Whether optimization succeeded
        """
        m = self.n_controls

        A_u, b_u = self.linearize_all_input_constraints(u_nom, input_constraints)

        if A_u.shape[0] == 0:
            R_max = np.eye(m) * (self.u_max**2)
            return R_max, True

        W = cp.Variable((m, m), symmetric=True)

        constraints = [W >> 0]

        n_active = 0
        for i in range(A_u.shape[0]):
            a_j = A_u[i, :]
            b_j = b_u[i]
            if b_j > 1e-6:
                constraints.append(cp.norm(W @ a_j, 2) <= b_j)
                n_active += 1
        if n_active == 0:
            warnings.warn("No active input constraints at u_nom. Using default ellipsoid.")
            R_max = np.eye(m) * (self.u_max**2)
            return R_max, True

        constraints.append(cp.trace(W) <= self.u_max * np.eye(m))

        objective = cp.Maximize(cp.log_det(W))
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self.solver, verbose=verbose)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                warnings.warn(f"R_max optimization failed with status: {problem.status}")
                R_max = np.eye(m) * 0.01
                return R_max, False

            W_opt = W.value
            R_max = W_opt.T @ W_opt
            R_max = R_max / (R_max + R_max.T)
            return R_max, True
        except Exception as e:
            warnings.warn(f"R_max optimization failed with error: {e}")
            R_max = np.eye(m) * 0.01
            return R_max, False

    def compute_envelopes_per_timestep(
        self,
        trajectory: Dict,
        state_constraints: Dict,
        input_constraints: Dict,
        obstacles: Optional[List[Dict]] = None,
        verbose: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute P_min(k) and R_max(k) for each timestep in trajectory.

        Supports obstacles in addition to box constraints that can be time-varying.

        Args:
            trajectory: Trajectory dictionary with keys:
                - 'X': State trajectory (N+1, n)
                - 'U': Control trajectory (N, m)
            state_constraints: Dict of state constraints
            input_constraints: Dict of input constraints
            obstacles: List of obstacle dictionaries
            verbose: Print solver output

        Returns:
            P_min_k: List of P_min(k) ellipsoids for each timestep
            R_max_k: List of R_max(k) ellipsoids for each timestep
        """
        X = trajectory["X"]
        U = trajectory["U"]
        N = X.shape[0]

        P_min_k = []
        R_max_k = []

        n_failed_P = 0
        n_failed_R = 0

        for k in range(N):
            # Get active obstacles at time k
            active_obstacles = self._get_active_obstacles_at_time(obstacles, k)

            # Compute P_min(k)
            P_min, success_P = self.compute_P_min(X[k], state_constraints, active_obstacles, verbose=False)
            P_min_k.append(P_min)
            if not success_P:
                n_failed_P += 1

            # Compute R_max(k)
            R_max, success_R = self.compute_R_max(U[k], input_constraints, verbose=False)
            R_max_k.append(R_max)
            if not success_R:
                n_failed_R += 1

        if verbose:
            print(f"    Computed P_min(k) for {N} timesteps ({n_failed_P} failed)")
            print(f"    Computed R_max(k) for {N} timesteps ({n_failed_R} failed)")
            if obstacles:
                print(f"    Considered {len(obstacles)} obstacles")

        return P_min_k, R_max_k

    def compute_segment_envelopes(
        self,
        P_min_k: List[np.ndarray],
        R_max_k: List[np.ndarray],
        segments: List[Dict],
        verbose: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute per-segment envelopes P_min,i and R_max,i.
        """
        n_segments = len(segments)
        P_min_i = []
        R_max_i = []

        for i, segment in enumerate(segments):
            time_indices = segment["time_indices"]

            # P_min,i = element-wise maximum
            P_segment = [P_min_k[k] for k in time_indices]
            P_min_seg = self._compute_matrix_envelope_max(P_segment)
            P_min_i.append(P_min_seg)

            # R_max,i = element-wise min
            R_segment = [R_max_k[k] for k in time_indices]
            R_max_seg = self._compute_matrix_envelope_min(R_segment)
            R_max_i.append(R_max_seg)

        if verbose:
            print(f"    Computed segment envelopes for {n_segments} segments")

        return P_min_i, R_max_i

    def compute_all_envelopes(
        self,
        trajectory: Dict,
        segments: List[Dict],
        state_constraints: Dict,
        input_constraints: Dict,
        obstacles: Optional[List[Dict]] = None,
        verbose: bool = True,
    ) -> EllipsoidConstraints:
        """
        Compute all ellipsoid constraint envelopes.

        Args:
            trajectory: Nominal trajectory
            segments: List of segment dictionaries
            state_constraints: Dict of state constraints
            input_constraints: Dict of input constraints
            obstacles: List of obstacle dictionaries
            verbose: Print solver output

        Returns:
            EllipsoidConstraints object with all envelopes
        """
        if verbose:
            print("\n Computing ellipsoid constraint envelopes...")

        # Step 1: Per-timestep envelopes
        if verbose:
            print("     Step 1: Computing P_min(k) and R_max(k)...")
        P_min_k, R_max_k = self.compute_envelopes_per_timestep(
            trajectory, state_constraints, input_constraints, obstacles, verbose=True
        )

        # Step 2: Per-segment envelopes
        if verbose:
            print("     Step 2: Computing P_min,i and R_max,i...")
        P_min_i, R_max_i = self.compute_segment_envelopes(P_min_k, R_max_k, segments, verbose=True)

        # Extract segment time indices
        segment_times = [seg["time_indices"] for seg in segments]

        metadata = {
            "n_timesteps": len(P_min_k),
            "n_segments": len(segments),
            "n_obstacles": len(obstacles) if obstacles else 0,
            "state_constraints": state_constraints,
            "input_constraints": input_constraints,
            "obstacles": obstacles,
        }

        constraints = EllipsoidConstraints(
            P_min_k=P_min_k,
            R_max_k=R_max_k,
            P_min_i=P_min_i,
            R_max_i=R_max_i,
            segment_times=segment_times,
            metadata=metadata,
        )
        if verbose:
            print(f"    Successfully computed envelopes for {len(segments)} segments")

        return constraints

    def visualize_feasibility_tube(
        self,
        trajectory: Dict,
        constraints: EllipsoidConstraints,
        obstacles: Optional[List[Dict]] = None,
        ax: Optional[plt.Axes] = None,
        n_ellipses: int = 20,
        alpha: float = 0.3,
    ) -> plt.Axes:
        """
        Visualize the feasibility tube around the nominal trajectory.

        Shows P_min(k) ellipsoids at selected timesteps along the trajectory,
        forming a tube of feasible deviations.

        Args:
            trajectory: Nominal trajectory
            constraints: EllipsoidConstraints object
            obstacles: List of obstacle dictionaries
            ax: Matplotlib axes
            n_ellipses: Number of ellipsoids to plot
            alpha: Transparency of ellipsoids

        Returns:
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))

        X = trajectory["X"]
        N = X.shape[0]

        # Plot nominal trajectory
        ax.plot(X[:, 0], X[:, 1], "k-", linewidth=2, label="Nominal trajectory", zorder=10)

        # Plot obstacles
        if obstacles:
            for obs in obstacles:
                if obs["type"] == "circle":
                    circle = Circle(
                        obs["center"],
                        obs["radius"],
                        fill=True,
                        facecolor="red",
                        alpha=0.3,
                        edgecolor="darkred",
                        linewidth=2,
                        label="Obstacle",
                    )
                    ax.add_patch(circle)
                elif obs["type"] == "ellipse":
                    # TODO: Add ellipse obstacle visualization
                    pass

        # Select timesteps to visualize
        step = max(1, N // n_ellipses)
        timesteps = range(0, N, step)

        # Plot P_min(k) ellipsoids
        for k in timesteps:
            P_min_k = constraints.P_min_k[k]
            self._plot_ellipsoid_2d(
                P_min_k, center=X[k], ax=ax, fill=True, facecolor="blue", alpha=alpha, edgecolor="blue", linewidth=0.5
            )

        ax.set_xlabel("x (m)", fontsize=12)
        ax.set_ylabel("y (m)", fontsize=12)
        ax.set_title("Feasibility Tube: P_min(k) Ellipsoids Along Trajectory", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        return ax

    def _linearize_all_state_constraints(
        self,
        x_nom: np.ndarray,
        state_constraints: Dict,
        obstacles: Optional[List[Dict]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize all state constraints including obstacles.

        Combines box constraints and obstacle constraints.

        Args:
            x_nom: Nominal state [x, y, ...]
            state_constraints: Dict of state constraints
            obstacles: List of obstacle dictionaries

        Returns:
            A_x: Constraint matrix (n_constraints, n_states)
            b_x: Constraint bounds (n_constraints,)
        """
        A_list = []
        b_list = []

        # 1. Box constraints
        A_box, b_box = self._linearize_box_constraints(x_nom, state_constraints)
        if A_box.shape[0] > 0:
            A_list.append(A_box)
            b_list.append(b_box)

        # 2. Obstacle constraints
        if obstacles:
            A_obs, b_obs = self._linearize_obstacle_constraints(x_nom, obstacles)
            if A_obs.shape[0] > 0:
                A_list.append(A_obs)
                b_list.append(b_obs)

        if len(A_list) > 0:
            A_x = np.vstack(A_list)
            b_x = np.concatenate(b_list)
        else:
            A_x = np.zeros((0, self.n_states))
            b_x = np.zeros(0)

        return A_x, b_x

    def _linearize_box_constraints(
        self,
        x_nom: np.ndarray,
        state_constraints: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize box constraints at a given point."""
        n = self.n_states

        if "box" not in state_constraints:
            return np.zeros((0, n)), np.zeros(0)

        box = state_constraints["box"]
        x_min = np.array(box.get("x_min", -np.inf * np.ones(n)))
        x_max = np.array(box.get("x_max", np.inf * np.ones(n)))

        A_list = []
        b_list = []

        # Lower bounds
        for i in range(n):
            if x_min[i] > -np.inf:
                a_j = np.zeros(n)
                a_j[i] = -1
                b_j = x_nom[i] - x_min[i]
                A_list.append(a_j)
                b_list.append(b_j)

        # Upper bounds
        for i in range(n):
            if x_max[i] < np.inf:
                a_j = np.zeros(n)
                a_j[i] = 1
                b_j = x_max[i] - x_nom[i]
                A_list.append(a_j)
                b_list.append(b_j)

        if len(A_list) > 0:
            A_x = np.array(A_list)
            b_x = np.array(b_list)
        else:
            A_x = np.zeros((0, n))
            b_x = np.zeros(0)

        return A_x, b_x

    def _linearize_obstacle_constraints(
        self,
        x_nom: np.ndarray,
        obstacles: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize obstacle avoidance constraints.

        For each obstacle, the constraints is:
        dist(x, obstacle) >= safety_margin

        Linearized as:
        grad(dist)^T . n >= dist(x_nom, obstacle) - safety_margin

        Args:
            x_nom: Nominal state [x, y, ...]
            obstacles: List of obstacle dictionaries

        Returns:
            A_obs: Constraint matrix (n_obstacles, n_states)
            b_obs: Constraint bounds (n_obstacles,)
        """
        n = self.n_states

        A_list = []
        b_list = []

        for obs in obstacles:
            obs_type = obs["type"]

            if obs_type == "circle":
                center = np.array(obs["center"])
                radius = obs["radius"]
                safety = obs.get("safety_margin", self.safety_margin)

                dist = self.distance_computer.distance_to_circle(x_nom, center, radius)
                grad_dist = self.distance_computer.gradient_distance_to_circle(x_nom, center, radius)

                a_j = -grad_dist
                b_j = safety - dist

                if dist < 3.0:
                    A_list.append(a_j)
                    b_list.append(max(0.0, b_j))

            elif obs_type == "ellipse":
                center = np.array(obs["center"])
                semi_axes = np.array(obs["semi_axes"])
                rotation = obs.get("rotation", 0.0)
                safety = obs.get("safety_margin", self.safety_margin)

                dist = self.distance_computer.distance_to_ellipsoid(x_nom, center, semi_axes, rotation)
                grad_dist = self.distance_computer.gradient_distance_to_ellipsoid(x_nom, center, semi_axes, rotation)

                a_j = -grad_dist
                b_j = safety - dist

                if dist < 3.0:
                    A_list.append(a_j)
                    b_list.append(max(0.0, b_j))

        if len(A_list) > 0:
            A_obs = np.array(A_list)
            b_obs = np.array(b_list)
        else:
            A_obs = np.zeros((0, n))
            b_obs = np.zeros(0)

        return A_obs, b_obs

    def _linearize_input_constraints(
        self,
        u_nom: np.ndarray,
        input_constraints: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize input constraints at a given point."""

        m = self.n_controls

        if "box" not in input_constraints:
            return np.zeros((0, m)), np.zeros(0)

        box = input_constraints["box"]
        u_min = np.array(box.get("u_min", -np.inf * np.ones(m)))
        u_max = np.array(box.get("u_max", np.inf * np.ones(m)))

        A_list = []
        b_list = []

        # Lower bounds
        for i in range(m):
            if u_min[i] > -np.inf:
                a_j = np.zeros(m)
                a_j[i] = -1
                b_j = u_nom[i] - u_min[i]
                A_list.append(a_j)
                b_list.append(b_j)

        # Upper bounds
        for i in range(m):
            if u_max[i] < np.inf:
                a_j = np.zeros(m)
                a_j[i] = 1
                b_j = u_max[i] - u_nom[i]
                A_list.append(a_j)
                b_list.append(b_j)

        if len(A_list) > 0:
            A_u = np.array(A_list)
            b_u = np.array(b_list)
        else:
            A_u = np.zeros((0, m))
            b_u = np.zeros(0)

        return A_u, b_u

    def _get_active_obstacles_at_time(
        self,
        obstacles: Optional[List[Dict]],
        k: int,
    ) -> Optional[List[Dict]]:
        """
        Get obstacles that are active at timestep k.

        Obstacles can have an 'active_timesteps' field to specify
        when they are active. If not specified, obstacles are always active.

        Args:
            obstacles: List of obstacle dictionaries
            k: Timestep index

        Returns:
            List of obstacle obstacles at time k
        """
        if obstacles is None:
            return None

        active = []
        for obs in obstacles:
            if "active_timesteps" in obs:
                if k in obs["active_timesteps"] or (
                    hasattr(obs["active_timesteps"], "__iter__")
                    and k >= obs["active_timesteps"][0]
                    and k <= obs["active_timesteps"][1]
                ):
                    active.append(obs)
            else:
                active.append(obs)
        return active if len(active) > 0 else None

    def _compute_matrix_envelope_max(self, matrices: List[np.ndarray]) -> np.ndarray:
        """Element-wise maximum (tightest P_min)"""
        if len(matrices) == 0:
            raise ValueError("No matrices to compute envelope from")

        result = matrices[0].copy()
        for M in matrices[1:]:
            result = np.maximum(result, M)

        result = 0.5 * (result + result.T)
        eig_vals = np.linalg.eigvals(result)
        if np.min(eig_vals) <= 0:
            result += (abs(np.min(eig_vals)) + 1e-6) * np.eye(result.shape[0])
        return result

    def _compute_matrix_envelope_min(self, matrices: List[np.ndarray]) -> np.ndarray:
        """Element-wise minimum (loosest R_max)"""
        if len(matrices) == 0:
            raise ValueError("No matrices to compute envelope from")

        result = matrices[0].copy()
        for M in matrices[1:]:
            result = np.minimum(result, M)

        result = 0.5 * (result + result.T)
        eig_vals = np.linalg.eigvals(result)
        if np.min(eig_vals) < 0:
            result += (abs(np.min(eig_vals)) + 1e-6) * np.eye(result.shape[0])
        return result

    def _plot_ellipsoid_2d(self, P: np.ndarray, center: np.ndarray, ax: plt.Axes, **kwargs):
        """Plot 2D ellipsoid (helper for visualization)."""
        P_2d = P[:2, :2]
        eig_vals, eig_vecs = np.linalg.eigh(P_2d)

        a = 1.0 / np.sqrt(eig_vals[0])
        b = 1.0 / np.sqrt(eig_vals[1])
        angle = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]) * 180 / np.pi

        ellipse = Ellipse(xy=center[:2], width=2 * a, height=2 * b, angle=angle, **kwargs)
        ax.add_patch(ellipse)
