"""
Collision detection and distance computation utilities.

This module provides utilities for:
- Multi-obstacle collision checking
- Distance queries to nearest obstacle
- Trajectory validation
- Collision-free region queries
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .obstacles import Obstacle


class CollisionChecker:
    """
    Collision checker for multiple obstacles.

    Manage a collection of obstacles and provides efficient collision queries.
    """

    def __init__(self, obstacles: Optional[List[Obstacle]] = None):
        """
        Initialize collision checker.

        Args:
            obstacles: List of Obstacle objects (optional)
        """
        self.obstacles = obstacles if obstacles is not None else []

    def add_obstacle(self, obstacle: Obstacle):
        """
        Add an obstacle to the environment.

        Args:
            obstacle: Obstacle object
        """
        self.obstacles.append(obstacle)

    def remove_obstacle(self, index: int):
        """
        Remove an obstacle from the environment by index.

        Args:
            index: Index of the obstacle to remov
        """
        if 0 <= index < len(self.obstacles):
            del self.obstacles[index]

    def clear_obstacles(self):
        """
        Remove all obstacles from the environment.
        """
        self.obstacles = []

    def get_obstacles(self) -> List[Obstacle]:
        """
        Get the list of obstacles in the environment.
        """
        return self.obstacles.copy()

    def num_obstacles(self) -> int:
        """
        Get the number of obstacles in the environment.
        """
        return len(self.obstacles)

    def is_collision(self, x: np.ndarray, tolerance: float = 0.0) -> bool:
        """
        Check if point collides with any obstacle.

        Args:
            x: Point in workspace [px, py, ...]
            tolerance: Additional clearance (meters)

        Returns:
            collision: True if collision, False otherwise
        """
        return any(obs.is_collision(x, tolerance) for obs in self.obstacles)

    def is_collision_free(self, x: np.ndarray, tolerance: float = 0.0) -> bool:
        """
        Check if point is collision-free with all obstacles.

        Args:
            x: Point in workspace [px, py, ...]
            tolerance: Additional clearance (meters)

        Returns:
            free: True if collision-free, False otherwise
        """
        return not self.is_collision(x, tolerance)

    def get_colliding_obstacles(self, x: np.ndarray, tolerance: float = 0.0) -> List[Tuple[int, Obstacle]]:
        """
        Get list of all obstacles that collide with point.

        Args:
            x: Point in workspace [px, py, ...]
            tolerance: Additional clearance (meters)

        Returns:
            colliding: List of tuples (index, obstacle)
        """
        colliding = []
        for i, obs in enumerate(self.obstacles):
            if obs.is_collision(x, tolerance):
                colliding.append((i, obs))
        return colliding

    def distance_to_nearest_obstacle(self, x: np.ndarray) -> Tuple[float, Optional[int]]:
        """
        Compute distance to nearest obstacle.

        Args:
            x: Point in workspace [px, py, ...]

        Returns:
            distance: Distance to nearest obstacle
            index: Index of nearest obstacle, None if no obstacles
        """
        if len(self.obstacles) == 0:
            return np.inf, None

        distances = [obs.distance(x) for obs in self.obstacles]
        nearest_idx = np.argmin(distances)
        min_distance = distances[nearest_idx]

        return min_distance, nearest_idx

    def distance_to_all_obstacles(self, x: np.ndarray) -> Dict[int, float]:
        """
        Compute distance to all obstacles.

        Args:
            x: Point in workspace [px, py, ...]

        Returns:
            distances: Dictionary of obstacle indices and distances
        """
        return {i: obs.distance(x) for i, obs in enumerate(self.obstacles)}

    def check_trajectory_collision(
        self, x_traj: np.ndarray, tolerance: float = 0.0
    ) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Check if trajectory collides with any obstacle.

        Args:
            x_traj: Trajectory, shape (N, n) where n >= 2
            tolerance: Additional clearance (meters)

        Returns:
            collision: True if collision, False otherwise
            collision_idx: Index of first collision, None if no collision
            obstacle_idx: Index of first obstacle, None if no collision
        """
        for k, x in enumerate(x_traj):
            for i, obs in enumerate(self.obstacles):
                if obs.is_collision(x, tolerance):
                    return True, k, i
        return False, None, None

    def get_trajectory_clearance(self, x_traj: np.ndarray) -> Tuple[float, int, int]:
        """
        Check if trajectory has sufficient clearance with all obstacles.

        Args:
            x_traj: Trajectory, shape (N, n) where n >= 2

        Returns:
            clearance: Minimum clearance along trajectory
            timestep: Index of timestep with minimum clearance
            obstacle: Index of obstacle with minimum clearance
        """
        if len(self.obstacles) == 0:
            return np.inf, -1, -1

        min_clearance = np.inf
        min_timestep = -1
        min_obstacle = -1

        for k, x in enumerate(x_traj):
            dist, obs_idx = self.distance_to_nearest_obstacle(x)
            if dist < min_clearance:
                min_clearance = dist
                min_timestep = k
                min_obstacle = obs_idx

        return min_clearance, min_timestep, min_obstacle

    def validate_trajectory(self, x_traj: np.ndarray, min_clearance: float = 0.0) -> Tuple[bool, Optional[str]]:
        """
        Validate trajectory for collision-freeness.

        Args:
            x_traj: Trajectory, shape (N, n)
            min_clearance: Required minimum clearance

        Returns:
            valid: True if trajectory is valid
            message: Validation message (None if valid, error description if invalid)
        """
        collision, timestep, obs_idx = self.check_trajectory_collision(x_traj, tolerance=-min_clearance)

        if collision:
            return False, f"Collision at timestep {timestep} with obstacle {obs_idx}"

        clearance, timestep, obs_idx = self.get_trajectory_clearance(x_traj)

        if clearance < min_clearance:
            return False, (
                f"Insufficient clearance {clearance:.3f}m at timestep {timestep} (required {min_clearance:.3f}m)"
            )

        return True, None

    def get_gradient_nearest_obstacle(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
        """
        Get gradient of distance function to nearest obstacle.

        Useful for SCvx constraint linearization.

        Args:
            x: Point in workspace

        Returns:
            gradient: Gradient of nearest obstacle distance
            nearest_idx: Index of nearest obstacle
        """
        if len(self.obstacles) == 0:
            return np.zeros(2), None

        _, nearest_idx = self.distance_to_nearest_obstacle(x)
        gradient = self.obstacles[nearest_idx].gradient(x)

        return gradient, nearest_idx

    def get_all_gradients(self, x: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Get gradients of distance functions to all obstacles.

        Args:
            x: Point in workspace

        Returns:
            gradients: Dictionary of obstacle indices and gradients
        """
        return {i: obs.gradient(x) for i, obs in enumerate(self.obstacles)}

    def sample_collision_free_point(
        self, bounds: Tuple[np.ndarray, np.ndarray], max_attempts: int = 1000, tolerance: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Sample a random collision-free point within given bounds.

        Args:
            bounds: Tuple of lower and upper bounds [lower, upper]
            max_attempts: Maximum number of attempts to sample a collision-free point
            tolerance: Additional clearance (meters)

        Returns:
            point: Random collision-free point, None if no point found
        """
        lower, upper = bounds

        for _ in range(max_attempts):
            # sample radnom point within bounds
            point = np.random.uniform(lower, upper)

            # check if point is collision-free
            if self.is_collision_free(point, tolerance):
                return point

        return None

    def get_workspace_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box that contains all obstacles.

        Returns:
            lower: Lower corner [x_min, y_min]
            upper: Upper corner [x_max, y_max]
        """
        if len(self.obstacles) == 0:
            return np.array([0.0, 0.0]), np.array([1.0, 1.0])

        # Get bouynding box for each obstacle
        boxes = [obs.get_bounding_box() for obs in self.obstacles]
        lowers, uppers = zip(*boxes)

        # compute overall bounding box
        lower = np.min(lowers, axis=0)
        upper = np.max(uppers, axis=0)

        return lower, upper

    def __repr__(self) -> str:
        return f"CollisionChecker(num_obstacles={len(self.obstacles)})"
