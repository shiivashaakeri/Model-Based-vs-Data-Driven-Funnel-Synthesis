"""
Workspace Definitions for DDFS.

This module provides the Workspace class that combines:
- State and input constraints (from constraints.py)
- Obstacle collection (from obstacles.py)
- Position bounds derived from state constraints
- Unified feasibility checking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ddfs.core.constraints import (
    BoxConstraint,
    SoftStateInputConstraints,
    StateInputConstraints,
    constraints_from_config,
)
from ddfs.core.obstacles import (
    ObstacleCollection,
    obstacles_from_config,
)
from ddfs.utils.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Workspace Class
# =============================================================================


@dataclass
class Workspace:
    """
    Complete workspace definition including constraints and obstacles.

    Combines state/input constraints with obstacle avoidance for
    unified feasibility checking.

    Parameters
    ----------
    constraints : StateInputConstraints
        State and input box constraints.
    obstacles : ObstacleCollection
        Collection of obstacles.
    position_indices : list of int
        Indices of position states (for obstacle checking).
    safety_margin : float
        Default safety margin for obstacle avoidance.
    """

    constraints: StateInputConstraints
    obstacles: ObstacleCollection = field(default_factory=ObstacleCollection)
    position_indices: List[int] = field(default_factory=lambda: [0, 1])
    safety_margin: float = 0.0

    @property
    def n_states(self) -> int:
        """Number of state dimensions."""
        return self.constraints.n_states

    @property
    def n_inputs(self) -> int:
        """Number of input dimensions."""
        return self.constraints.n_inputs

    @property
    def n_obstacles(self) -> int:
        """Number of obstacles."""
        return self.obstacles.n_obstacles

    @property
    def position_dim(self) -> int:
        """Dimension of position space."""
        return len(self.position_indices)

    @property
    def x_min(self) -> np.ndarray:
        """State lower bounds."""
        return self.constraints.x_min

    @property
    def x_max(self) -> np.ndarray:
        """State upper bounds."""
        return self.constraints.x_max

    @property
    def u_min(self) -> np.ndarray:
        """Input lower bounds."""
        return self.constraints.u_min

    @property
    def u_max(self) -> np.ndarray:
        """Input upper bounds."""
        return self.constraints.u_max

    @property
    def position_bounds(self) -> BoxConstraint:
        """
        Get position bounds derived from state constraints.

        Returns
        -------
        BoxConstraint
            Box constraint on position variables only.
        """
        idx = self.position_indices
        return BoxConstraint(
            lb=self.x_min[idx],
            ub=self.x_max[idx],
        )

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        """
        Extract position from full state vector.

        Parameters
        ----------
        state : np.ndarray
            Full state vector.

        Returns
        -------
        np.ndarray
            Position vector.
        """
        return state[self.position_indices]

    # =========================================================================
    # Feasibility Checking
    # =========================================================================

    def check_state_bounds(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        """
        Check if state satisfies box constraints.

        Parameters
        ----------
        x : np.ndarray
            State vector.
        tol : float
            Tolerance.

        Returns
        -------
        bool
            True if state is within bounds.
        """
        return self.constraints.check_state(x, tol)

    def check_input_bounds(self, u: np.ndarray, tol: float = 1e-9) -> bool:
        """
        Check if input satisfies box constraints.

        Parameters
        ----------
        u : np.ndarray
            Input vector.
        tol : float
            Tolerance.

        Returns
        -------
        bool
            True if input is within bounds.
        """
        return self.constraints.check_input(u, tol)

    def check_collision(self, x: np.ndarray) -> bool:
        """
        Check if state is collision-free.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        bool
            True if collision-free.
        """
        if self.obstacles.is_empty:
            return True

        position = self.extract_position(x)
        return self.obstacles.is_collision_free(position)

    def is_feasible(
        self,
        x: np.ndarray,
        u: Optional[np.ndarray] = None,
        tol: float = 1e-9,
    ) -> bool:
        """
        Check if state (and optionally input) is fully feasible.

        Feasibility requires:
        1. State within box constraints
        2. Input within box constraints (if provided)
        3. No collision with obstacles

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray, optional
            Input vector.
        tol : float
            Tolerance for bound checking.

        Returns
        -------
        bool
            True if fully feasible.
        """
        # Check state bounds
        if not self.check_state_bounds(x, tol):
            return False

        # Check input bounds if provided
        if u is not None and not self.check_input_bounds(u, tol):
            return False

        # Check collision
        return self.check_collision(x)

    def feasibility_info(
        self,
        x: np.ndarray,
        u: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Get detailed feasibility information.

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray, optional
            Input vector.

        Returns
        -------
        dict
            Dictionary with feasibility details.
        """
        info = {
            "state_feasible": self.check_state_bounds(x),
            "state_violation": self.constraints.state_constraint.max_violation(x),
            "collision_free": self.check_collision(x),
            "min_obstacle_distance": self.min_obstacle_distance(x),
        }

        if u is not None:
            info["input_feasible"] = self.check_input_bounds(u)
            info["input_violation"] = self.constraints.input_constraint.max_violation(u)

        info["fully_feasible"] = (
            info["state_feasible"] and info["collision_free"] and (u is None or info["input_feasible"])
        )

        return info

    # =========================================================================
    # Distance Computations
    # =========================================================================

    def min_obstacle_distance(self, x: np.ndarray) -> float:
        """
        Compute minimum signed distance to any obstacle.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        float
            Minimum signed distance (negative if inside obstacle).
        """
        if self.obstacles.is_empty:
            return np.inf

        position = self.extract_position(x)
        return self.obstacles.min_signed_distance(position)

    def all_obstacle_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Compute signed distances to all obstacles.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Array of signed distances.
        """
        position = self.extract_position(x)
        return self.obstacles.all_signed_distances(position)

    def distance_to_state_boundary(self, x: np.ndarray) -> float:
        """
        Compute signed distance to state constraint boundary.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        float
            Signed distance (negative if outside bounds).
        """
        return self.constraints.state_constraint.distance_to_boundary(x)

    # =========================================================================
    # Trajectory Checking
    # =========================================================================

    def check_trajectory(
        self,
        x_traj: np.ndarray,
        u_traj: Optional[np.ndarray] = None,
    ) -> Tuple[bool, dict]:
        """
        Check if entire trajectory is feasible.

        Parameters
        ----------
        x_traj : np.ndarray
            State trajectory, shape (N+1, n_states).
        u_traj : np.ndarray, optional
            Input trajectory, shape (N, n_inputs).

        Returns
        -------
        is_feasible : bool
            True if entire trajectory is feasible.
        info : dict
            Detailed information about violations.
        """
        N = len(x_traj) - 1

        # Check state bounds
        state_feasible = np.array([self.check_state_bounds(x) for x in x_traj])

        # Check collisions
        collision_free = np.array([self.check_collision(x) for x in x_traj])

        # Check input bounds if provided
        if u_traj is not None:
            input_feasible = np.array([self.check_input_bounds(u) for u in u_traj])
        else:
            input_feasible = np.ones(N, dtype=bool)

        # Compute violations
        state_violations = np.array([self.constraints.state_constraint.max_violation(x) for x in x_traj])

        obstacle_distances = np.array([self.min_obstacle_distance(x) for x in x_traj])

        info = {
            "state_feasible": state_feasible,
            "collision_free": collision_free,
            "input_feasible": input_feasible,
            "state_violations": state_violations,
            "obstacle_distances": obstacle_distances,
            "max_state_violation": np.max(state_violations),
            "min_obstacle_distance": np.min(obstacle_distances),
            "n_state_violations": np.sum(~state_feasible),
            "n_collisions": np.sum(~collision_free),
        }

        if u_traj is not None:
            input_violations = np.array([self.constraints.input_constraint.max_violation(u) for u in u_traj])
            info["input_violations"] = input_violations
            info["max_input_violation"] = np.max(input_violations)
            info["n_input_violations"] = np.sum(~input_feasible)

        is_feasible = np.all(state_feasible) and np.all(collision_free) and np.all(input_feasible)

        return is_feasible, info

    # =========================================================================
    # Projection and Sampling
    # =========================================================================

    def project_state(self, x: np.ndarray) -> np.ndarray:
        """
        Project state onto feasible region (bounds only, not collision-free).

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Projected state.
        """
        return self.constraints.project_state(x)

    def project_input(self, u: np.ndarray) -> np.ndarray:
        """
        Project input onto feasible region.

        Parameters
        ----------
        u : np.ndarray
            Input vector.

        Returns
        -------
        np.ndarray
            Projected input.
        """
        return self.constraints.project_input(u)

    def sample_feasible_state(
        self,
        max_attempts: int = 1000,
    ) -> Optional[np.ndarray]:
        """
        Sample a random feasible state.

        Parameters
        ----------
        max_attempts : int
            Maximum sampling attempts.

        Returns
        -------
        np.ndarray or None
            Feasible state, or None if not found.
        """
        for _ in range(max_attempts):
            x = self.constraints.state_constraint.sample_uniform(1)[0]
            if self.check_collision(x):
                return x

        logger.warning(f"Could not find feasible state after {max_attempts} attempts")
        return None

    # =========================================================================
    # Soft Constraints
    # =========================================================================

    def get_soft_constraints(
        self,
        state_weight: float = 1e3,
        input_weight: float = 1e3,
        slack_type: str = "l2",
    ) -> SoftStateInputConstraints:
        """
        Get soft constraint wrapper.

        Parameters
        ----------
        state_weight : float
            Penalty weight for state violations.
        input_weight : float
            Penalty weight for input violations.
        slack_type : str
            Type of slack penalty ('l1', 'l2', 'linf').

        Returns
        -------
        SoftStateInputConstraints
            Soft constraint wrapper.
        """
        return SoftStateInputConstraints(
            hard_constraints=self.constraints,
            state_weight=state_weight,
            input_weight=input_weight,
            slack_type=slack_type,
        )

    def obstacle_penalty(
        self,
        x: np.ndarray,
        weight: float = 1e3,
        buffer: float = 0.0,
    ) -> float:
        """
        Compute penalty for obstacle proximity/collision.

        Parameters
        ----------
        x : np.ndarray
            State vector.
        weight : float
            Penalty weight.
        buffer : float
            Additional buffer distance.

        Returns
        -------
        float
            Penalty value (0 if collision-free with buffer).
        """
        if self.obstacles.is_empty:
            return 0.0

        min_dist = self.min_obstacle_distance(x)
        violation = -(min_dist - buffer)  # Positive if too close

        if violation <= 0:
            return 0.0

        return weight * violation**2

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def get_boundary_data(self) -> dict:
        """
        Get boundary data for visualization.

        Returns
        -------
        dict
            Dictionary with boundary information.
        """
        data = {
            "position_bounds": {
                "lb": self.position_bounds.lb,
                "ub": self.position_bounds.ub,
            },
            "state_bounds": {
                "lb": self.x_min,
                "ub": self.x_max,
            },
            "input_bounds": {
                "lb": self.u_min,
                "ub": self.u_max,
            },
            "obstacles": [],
        }

        for obs in self.obstacles:
            obs_data = {
                "center": obs.center.tolist(),
                "radius": obs.radius,
                "effective_radius": obs.effective_radius,
                "dim": obs.dim,
            }
            data["obstacles"].append(obs_data)

        return data

    def __repr__(self) -> str:
        return (
            f"Workspace(\n"
            f"  states: {self.n_states}D\n"
            f"  inputs: {self.n_inputs}D\n"
            f"  position_indices: {self.position_indices}\n"
            f"  obstacles: {self.n_obstacles}\n"
            f"  safety_margin: {self.safety_margin}\n"
            f")"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def workspace_from_config(
    config,
    safety_margin: float = 0.0,
) -> Workspace:
    """
    Create workspace from configuration.

    Parameters
    ----------
    config : Config
        Configuration object.
    safety_margin : float
        Safety margin for obstacles.

    Returns
    -------
    Workspace
        Created workspace.
    """
    # Create constraints
    constraints = constraints_from_config(config)

    # Create obstacles
    obstacles = obstacles_from_config(config, margin=safety_margin)

    # Determine position indices based on system
    system_name = config.system.name.lower()
    if system_name == "unicycle":
        position_indices = [0, 1]  # px, py
    elif system_name == "quadrotor":
        position_indices = [0, 1, 2]  # px, py, pz
    else:
        # Default: assume first 2 or 3 states are position
        n_states = config.system.n_states
        position_indices = list(range(min(3, n_states)))
        logger.warning(f"Unknown system '{system_name}', assuming position indices: {position_indices}")

    return Workspace(
        constraints=constraints,
        obstacles=obstacles,
        position_indices=position_indices,
        safety_margin=safety_margin,
    )


def create_workspace(
    x_min: np.ndarray,
    x_max: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    obstacles: Optional[List[dict]] = None,
    position_indices: Optional[List[int]] = None,
    safety_margin: float = 0.0,
) -> Workspace:
    """
    Create workspace from explicit parameters.

    Parameters
    ----------
    x_min, x_max : np.ndarray
        State bounds.
    u_min, u_max : np.ndarray
        Input bounds.
    obstacles : list of dict, optional
        List of obstacle specifications with 'center' and 'radius' keys.
    position_indices : list of int, optional
        Indices of position in state vector.
    safety_margin : float
        Safety margin for obstacles.

    Returns
    -------
    Workspace
        Created workspace.
    """
    # Create constraints
    constraints = StateInputConstraints.from_bounds(
        x_min=np.asarray(x_min),
        x_max=np.asarray(x_max),
        u_min=np.asarray(u_min),
        u_max=np.asarray(u_max),
    )

    # Create obstacles
    obstacle_collection = ObstacleCollection(default_margin=safety_margin)
    if obstacles is not None:
        from ddfs.core.obstacles import create_obstacle  # noqa: PLC0415

        for obs_spec in obstacles:
            obstacle = create_obstacle(
                center=obs_spec["center"],
                radius=obs_spec["radius"],
                margin=safety_margin,
            )
            obstacle_collection.add(obstacle)

    # Default position indices
    if position_indices is None:
        n_states = len(x_min)
        position_indices = list(range(min(2, n_states)))

    return Workspace(
        constraints=constraints,
        obstacles=obstacle_collection,
        position_indices=position_indices,
        safety_margin=safety_margin,
    )
