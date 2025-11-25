"""
Obstacle Definitions for DDFS.

This module provides classes for representing:
- Circular obstacles (2D)
- Spherical obstacles (3D)
- Obstacle collections with collision checking
- Signed distance functions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from ddfs.utils.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Base Obstacle Class
# =============================================================================


class Obstacle(ABC):
    """
    Abstract base class for obstacles.

    All obstacles must implement signed distance and containment checking.
    """

    @abstractmethod
    def signed_distance(self, point: np.ndarray) -> float:
        """
        Compute signed distance from point to obstacle boundary.

        Negative inside, positive outside.

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        float
            Signed distance (negative if inside obstacle).
        """
        pass

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is inside the obstacle.

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        bool
            True if point is inside obstacle.
        """
        pass

    @abstractmethod
    def distance_to_surface(self, point: np.ndarray) -> float:
        """
        Compute unsigned distance from point to obstacle surface.

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        float
            Distance to nearest point on surface (always >= 0).
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the obstacle (2 or 3)."""
        pass


# =============================================================================
# Circular Obstacle (2D)
# =============================================================================


@dataclass
class CircularObstacle(Obstacle):
    """
    Circular obstacle in 2D.

    Parameters
    ----------
    center : np.ndarray
        Center of the circle [x, y].
    radius : float
        Radius of the circle.
    margin : float
        Safety margin (effective radius = radius + margin).
    """

    center: np.ndarray
    radius: float
    margin: float = 0.0

    def __post_init__(self):
        """Validate and convert center to numpy array."""
        self.center = np.asarray(self.center, dtype=np.float64)

        if len(self.center) != 2:
            raise ValueError(f"CircularObstacle center must be 2D, got shape {self.center.shape}")

        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")

        if self.margin < 0:
            raise ValueError(f"Margin must be non-negative, got {self.margin}")

    @property
    def dim(self) -> int:
        """Dimension (always 2 for circular)."""
        return 2

    @property
    def effective_radius(self) -> float:
        """Radius including safety margin."""
        return self.radius + self.margin

    def signed_distance(self, point: np.ndarray) -> float:
        """
        Compute signed distance from point to obstacle boundary.

        Parameters
        ----------
        point : np.ndarray
            Point to check [x, y] or [..., x, y].

        Returns
        -------
        float
            Signed distance (negative if inside).
        """
        point = np.asarray(point)

        # Handle higher-dimensional state by extracting position
        if len(point) > 2:
            point = point[:2]

        dist_to_center = np.linalg.norm(point - self.center)
        return dist_to_center - self.effective_radius

    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is inside the obstacle (including margin).

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        bool
            True if inside obstacle.
        """
        return self.signed_distance(point) < 0

    def distance_to_surface(self, point: np.ndarray) -> float:
        """
        Compute unsigned distance to obstacle surface.

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        float
            Distance to surface.
        """
        return abs(self.signed_distance(point))

    def closest_point_on_surface(self, point: np.ndarray) -> np.ndarray:
        """
        Find closest point on obstacle surface to given point.

        Parameters
        ----------
        point : np.ndarray
            Query point.

        Returns
        -------
        np.ndarray
            Closest point on surface.
        """
        point = np.asarray(point)
        if len(point) > 2:
            point = point[:2]

        direction = point - self.center
        dist = np.linalg.norm(direction)

        if dist < 1e-12:
            # Point at center, return arbitrary surface point
            return self.center + np.array([self.effective_radius, 0])

        return self.center + (direction / dist) * self.effective_radius

    def gradient_signed_distance(self, point: np.ndarray) -> np.ndarray:
        """
        Compute gradient of signed distance function.

        Parameters
        ----------
        point : np.ndarray
            Point to evaluate gradient.

        Returns
        -------
        np.ndarray
            Gradient vector (2D).
        """
        point = np.asarray(point)
        if len(point) > 2:
            point = point[:2]

        direction = point - self.center
        dist = np.linalg.norm(direction)

        if dist < 1e-12:
            # At center, gradient undefined, return zero
            return np.zeros(2)

        return direction / dist

    def sample_boundary(self, n_points: int = 100) -> np.ndarray:
        """
        Sample points on the obstacle boundary.

        Parameters
        ----------
        n_points : int
            Number of points to sample.

        Returns
        -------
        np.ndarray
            Points on boundary, shape (n_points, 2).
        """
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points = np.column_stack(
            [
                self.center[0] + self.effective_radius * np.cos(theta),
                self.center[1] + self.effective_radius * np.sin(theta),
            ]
        )
        return points

    def __repr__(self) -> str:
        return f"CircularObstacle(center={self.center}, radius={self.radius}, margin={self.margin})"


# =============================================================================
# Spherical Obstacle (3D)
# =============================================================================


@dataclass
class SphericalObstacle(Obstacle):
    """
    Spherical obstacle in 3D.

    Parameters
    ----------
    center : np.ndarray
        Center of the sphere [x, y, z].
    radius : float
        Radius of the sphere.
    margin : float
        Safety margin (effective radius = radius + margin).
    """

    center: np.ndarray
    radius: float
    margin: float = 0.0

    def __post_init__(self):
        """Validate and convert center to numpy array."""
        self.center = np.asarray(self.center, dtype=np.float64)

        if len(self.center) != 3:
            raise ValueError(f"SphericalObstacle center must be 3D, got shape {self.center.shape}")

        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")

        if self.margin < 0:
            raise ValueError(f"Margin must be non-negative, got {self.margin}")

    @property
    def dim(self) -> int:
        """Dimension (always 3 for spherical)."""
        return 3

    @property
    def effective_radius(self) -> float:
        """Radius including safety margin."""
        return self.radius + self.margin

    def signed_distance(self, point: np.ndarray) -> float:
        """
        Compute signed distance from point to obstacle boundary.

        Parameters
        ----------
        point : np.ndarray
            Point to check [x, y, z] or [..., x, y, z].

        Returns
        -------
        float
            Signed distance (negative if inside).
        """
        point = np.asarray(point)

        # Handle higher-dimensional state by extracting position
        if len(point) > 3:
            point = point[:3]

        dist_to_center = np.linalg.norm(point - self.center)
        return dist_to_center - self.effective_radius

    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is inside the obstacle (including margin).

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        bool
            True if inside obstacle.
        """
        return self.signed_distance(point) < 0

    def distance_to_surface(self, point: np.ndarray) -> float:
        """
        Compute unsigned distance to obstacle surface.

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        float
            Distance to surface.
        """
        return abs(self.signed_distance(point))

    def closest_point_on_surface(self, point: np.ndarray) -> np.ndarray:
        """
        Find closest point on obstacle surface to given point.

        Parameters
        ----------
        point : np.ndarray
            Query point.

        Returns
        -------
        np.ndarray
            Closest point on surface.
        """
        point = np.asarray(point)
        if len(point) > 3:
            point = point[:3]

        direction = point - self.center
        dist = np.linalg.norm(direction)

        if dist < 1e-12:
            # Point at center, return arbitrary surface point
            return self.center + np.array([self.effective_radius, 0, 0])

        return self.center + (direction / dist) * self.effective_radius

    def gradient_signed_distance(self, point: np.ndarray) -> np.ndarray:
        """
        Compute gradient of signed distance function.

        Parameters
        ----------
        point : np.ndarray
            Point to evaluate gradient.

        Returns
        -------
        np.ndarray
            Gradient vector (3D).
        """
        point = np.asarray(point)
        if len(point) > 3:
            point = point[:3]

        direction = point - self.center
        dist = np.linalg.norm(direction)

        if dist < 1e-12:
            return np.zeros(3)

        return direction / dist

    def sample_boundary(self, n_points: int = 100) -> np.ndarray:
        """
        Sample points on the obstacle boundary using Fibonacci sphere.

        Parameters
        ----------
        n_points : int
            Number of points to sample.

        Returns
        -------
        np.ndarray
            Points on boundary, shape (n_points, 3).
        """
        # Fibonacci sphere sampling for uniform distribution
        indices = np.arange(n_points)
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

        y = 1 - (indices / (n_points - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)

        theta = phi * indices

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        points = np.column_stack([x, y, z]) * self.effective_radius + self.center
        return points

    def __repr__(self) -> str:
        return f"SphericalObstacle(center={self.center}, radius={self.radius}, margin={self.margin})"


# =============================================================================
# Obstacle Collection
# =============================================================================


@dataclass
class ObstacleCollection:
    """
    Collection of obstacles with batch collision checking.

    Parameters
    ----------
    obstacles : list of Obstacle
        List of obstacles in the collection.
    default_margin : float
        Default safety margin to apply when not specified per-obstacle.
    """

    obstacles: List[Obstacle] = field(default_factory=list)
    default_margin: float = 0.0

    def __post_init__(self):
        """Validate obstacles."""
        if self.obstacles:
            dims = [obs.dim for obs in self.obstacles]
            if len(set(dims)) > 1:
                logger.warning(f"ObstacleCollection contains mixed dimensions: {set(dims)}")

    @property
    def n_obstacles(self) -> int:
        """Number of obstacles in collection."""
        return len(self.obstacles)

    @property
    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return len(self.obstacles) == 0

    def add(self, obstacle: Obstacle) -> None:
        """
        Add an obstacle to the collection.

        Parameters
        ----------
        obstacle : Obstacle
            Obstacle to add.
        """
        self.obstacles.append(obstacle)

    def add_circular(self, center: np.ndarray, radius: float, margin: Optional[float] = None) -> None:
        """
        Add a circular obstacle.

        Parameters
        ----------
        center : np.ndarray
            Center [x, y].
        radius : float
            Radius.
        margin : float, optional
            Safety margin (uses default if not specified).
        """
        margin = margin if margin is not None else self.default_margin
        self.obstacles.append(CircularObstacle(center, radius, margin))

    def add_spherical(self, center: np.ndarray, radius: float, margin: Optional[float] = None) -> None:
        """
        Add a spherical obstacle.

        Parameters
        ----------
        center : np.ndarray
            Center [x, y, z].
        radius : float
            Radius.
        margin : float, optional
            Safety margin (uses default if not specified).
        """
        margin = margin if margin is not None else self.default_margin
        self.obstacles.append(SphericalObstacle(center, radius, margin))

    def min_signed_distance(self, point: np.ndarray) -> float:
        """
        Compute minimum signed distance to any obstacle.

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        float
            Minimum signed distance (negative if inside any obstacle).
        """
        if self.is_empty:
            return np.inf

        distances = [obs.signed_distance(point) for obs in self.obstacles]
        return min(distances)

    def min_distance_to_surface(self, point: np.ndarray) -> float:
        """
        Compute minimum distance to any obstacle surface.

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        float
            Minimum distance to any surface.
        """
        if self.is_empty:
            return np.inf

        distances = [obs.distance_to_surface(point) for obs in self.obstacles]
        return min(distances)

    def is_collision_free(self, point: np.ndarray) -> bool:
        """
        Check if point is collision-free (not inside any obstacle).

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        bool
            True if collision-free.
        """
        return self.min_signed_distance(point) >= 0

    def check_trajectory(self, trajectory: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Check if trajectory is collision-free.

        Parameters
        ----------
        trajectory : np.ndarray
            State trajectory, shape (N, n_states).

        Returns
        -------
        is_valid : bool
            True if entire trajectory is collision-free.
        collision_mask : np.ndarray
            Boolean array indicating collision at each step.
        """
        if self.is_empty:
            return True, np.zeros(len(trajectory), dtype=bool)

        collision_mask = np.zeros(len(trajectory), dtype=bool)

        for i, state in enumerate(trajectory):
            if not self.is_collision_free(state):
                collision_mask[i] = True

        return not np.any(collision_mask), collision_mask

    def closest_obstacle(self, point: np.ndarray) -> Tuple[Optional[Obstacle], float]:
        """
        Find the closest obstacle to a point.

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        obstacle : Obstacle or None
            Closest obstacle (None if collection is empty).
        distance : float
            Signed distance to closest obstacle.
        """
        if self.is_empty:
            return None, np.inf

        distances = [obs.signed_distance(point) for obs in self.obstacles]
        min_idx = np.argmin(distances)
        return self.obstacles[min_idx], distances[min_idx]

    def all_signed_distances(self, point: np.ndarray) -> np.ndarray:
        """
        Compute signed distances to all obstacles.

        Parameters
        ----------
        point : np.ndarray
            Point to check.

        Returns
        -------
        np.ndarray
            Array of signed distances, shape (n_obstacles,).
        """
        if self.is_empty:
            return np.array([])

        return np.array([obs.signed_distance(point) for obs in self.obstacles])

    def get_boundary_points(self, n_points_per_obstacle: int = 100) -> np.ndarray:
        """
        Get boundary points for all obstacles (for visualization).

        Parameters
        ----------
        n_points_per_obstacle : int
            Number of points per obstacle.

        Returns
        -------
        np.ndarray
            Boundary points, shape (n_obstacles * n_points, dim).
        """
        if self.is_empty:
            return np.array([])

        all_points = []
        for obs in self.obstacles:
            all_points.append(obs.sample_boundary(n_points_per_obstacle))

        return np.vstack(all_points)

    def set_margin(self, margin: float) -> None:
        """
        Set margin for all obstacles.

        Parameters
        ----------
        margin : float
            Safety margin to set.
        """
        for obs in self.obstacles:
            obs.margin = margin

    def __iter__(self):
        """Iterate over obstacles."""
        return iter(self.obstacles)

    def __len__(self) -> int:
        """Number of obstacles."""
        return len(self.obstacles)

    def __getitem__(self, idx: int) -> Obstacle:
        """Get obstacle by index."""
        return self.obstacles[idx]

    def __repr__(self) -> str:
        return f"ObstacleCollection({self.n_obstacles} obstacles)"


# =============================================================================
# Factory Functions
# =============================================================================


def create_obstacle(
    center: Union[List[float], np.ndarray],
    radius: float,
    margin: float = 0.0,
) -> Obstacle:
    """
    Create an obstacle (circular or spherical) based on center dimension.

    Parameters
    ----------
    center : list or np.ndarray
        Center coordinates.
    radius : float
        Radius.
    margin : float
        Safety margin.

    Returns
    -------
    Obstacle
        CircularObstacle (2D) or SphericalObstacle (3D).
    """
    center = np.asarray(center)

    if len(center) == 2:
        return CircularObstacle(center, radius, margin)
    elif len(center) == 3:
        return SphericalObstacle(center, radius, margin)
    else:
        raise ValueError(f"Center must be 2D or 3D, got dimension {len(center)}")


def obstacles_from_config(config, margin: float = 0.0) -> ObstacleCollection:
    """
    Create obstacle collection from configuration.

    Parameters
    ----------
    config : Config
        Configuration object with system.obstacles.
    margin : float
        Default safety margin.

    Returns
    -------
    ObstacleCollection
        Collection of obstacles from config.
    """
    collection = ObstacleCollection(default_margin=margin)

    for obs_config in config.system.obstacles:
        obstacle = create_obstacle(
            center=obs_config.center,
            radius=obs_config.radius,
            margin=margin,
        )
        collection.add(obstacle)

    return collection
