"""
Obstacle representation for environment modeling.

This module defines obstacle classes used for collision avoidance in trajectory planning
and funnel synthesis. All obstacles provide:
- Signed distance functions
- Gradients for linearization in SCvx
- Collision checking
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class Obstacle(ABC):
    """
    Abstract base class for obstacles in the workspace.

    All obstacles must implement:
    - distance(x): Signed distance function (negative inside, positive outside)
    - gradient(x): Gradient of distance function (for constraint linearization)
    - is_collision(x): Check if point is in collision
    """

    def __init__(self, safety_margin: float = 0.0):
        """
        Initialize obstacle.

        Args:
            safety_margin: Additional clearance distance (meters)
        """
        self.safety_margin = safety_margin

    @abstractmethod
    def distance(self, x: np.ndarray) -> float:
        """
        Signed distance function from point to obstacle boundary.

        Convention:
            - Negative: Inside obstacle (collision)
            - Zero: On boundary
            - Positive: Outside obstacle (safe)

        Args:
            x: Point in workspace (at least [px, py])

        Returns:
            dist: Signed distance (includes safety margin)
        """
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of distance function at point x: nabla d(x)

        Used for linearizing collision constraints in SCvx:
            d(x) = d(x*) + nabla d(x*)^T (x - x*) >= 0

        Args:
            x: Point in workspace (at least [px, py])

        Returns:
            grad: Gradient of distance function
        """
        pass

    def is_collision(self, x: np.ndarray, tolerance: float = 0.0) -> bool:
        """
        Check if point collides with obstacle.

        Args:
            x: Point in workspace
            tolerance: Additional clearance (meters)

        Returns:
            collision: True if collision, False otherwise
        """
        return self.distance(x) < tolerance

    def check_trajectory_collision(self, x_traj: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Check if trajectory collides with obstacle.

        Args:
            x_traj: Trajectory, shape (N, n) where n >= 2

        Returns:
            collision: True if collision, False otherwise
            collision_idx: Index of first collision, None if no collision
        """
        for k, x in enumerate(x_traj):
            if self.is_collision(x):
                return True, k
        return False, None

    @abstractmethod
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of obstacle.

        Returns:
            lower: Lower corner [x_min, y_min]
            upper: Upper corner [x_max, y_max]
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of obstacle.
        """
        pass


class CircularObstacle(Obstacle):
    """
    Circular obstacle in 2D workspace.

    Defined by center (cx, cy) and radius (r).
    Distance function: d(x) = ||x - c|| - r
    """

    def __init__(self, center: np.ndarray, radius: float, safety_margin: float = 0.0):
        """
        Iniitialize circular obstacle.

        Args:
            center: Center position [cx, cy]
            radius: Radius of circle (meters)
            safety_margin: Additional clearance distance (meters)
        """
        super().__init__(safety_margin=safety_margin)

        assert len(center) == 2, "Center must be 2D [cx, cy]"
        assert radius > 0, "Radius must be positive"

        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self._effective_radius = self.radius + self.safety_margin

    def distance(self, x: np.ndarray) -> float:
        """
        Signed distance to circular boundary.

        Args:
            x: Point [px, py, ...] (only first two elements are used)

        Returns:
            dist: Signed distance
        """
        pos = x[:2]
        dist_to_center = np.linalg.norm(pos - self.center)
        return dist_to_center - self._effective_radius

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of distance function.

        nabla d(x) = (x - c) / ||x - c||

        Args:
            x: Point [px, py, ...]

        Returns:
            grad: Gradient [d/px d/py, ...]
        """
        pos = x[:2]
        diff = pos - self.center
        dist = np.linalg.norm(diff)

        if dist < 1e-10:
            # At center, gradient is undefined, use arbitrary unit vector
            return np.array([1.0, 0.0])

        return diff / dist

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of circle."""
        lower = self.center - self._effective_radius
        upper = self.center + self._effective_radius
        return lower, upper

    def get_center(self) -> np.ndarray:
        """Get center of circle."""
        return self.center.copy()

    def get_radius(self) -> float:
        """Get obstacle's radius of circle. (without safety margin)"""
        return self.radius

    def get_effective_radius(self) -> float:
        """Get effective radius (includes safety margin)."""
        return self._effective_radius

    def __repr__(self) -> str:
        return f"CircularObstacle(center={self.center}, radius={self.radius}, safety_margin={self.safety_margin})"


class EllipsoidalObstacle(Obstacle):
    f"""
    Ellipsoidal obstacle in 2D workspace.

    Defined by center (cx, cy), semi-axes (a, b), and rotation angle (theta).
    Distance function uses normalized ellipsoid coordinates.

    The ellipsoid is defined by:
        ((x-c)^T R^T Q R (x-c))^{1 / 2} - 1
    where Q = diag(1/a^2, 1/b^2) and R is the rotation matrix.
    """

    def __init__(self, center: np.ndarray, semi_axes: np.ndarray, rotation: float = 0.0, safety_margin: float = 0.0):
        """
        Initialize ellipsoidal obstacle.

        Args:
            center: Center position [cx, cy]
            semi_axes: Semi-axes lengths [a, b] where a, b > 0
            rotation: Rotation angle (radians, counterclockwise from x-axis)
            safety_margin: Additional clearance distance (meters)
        """
        super().__init__(safety_margin=safety_margin)

        assert len(center) == 2, "Center must be 2D [cx, cy]"
        assert len(semi_axes) == 2, "Semi-axes must be 2D [a, b]"
        assert np.all(np.array(semi_axes) > 0), "Semi-axes must be positive"

        self.center = np.array(center, dtype=float)
        self.semi_axes = np.array(semi_axes, dtype=float)
        self.rotation = float(rotation)

        # Effective semi-axes with safety margin
        self._effective_semi_axes = self.semi_axes + self.safety_margin

        # Precompute rotation matrix and its transpose
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        self.R = np.array([[c, -s], [s, c]], dtype=float)
        self.R_T = self.R.T

        # Precompute shape matrix Q = diag(1/a^2, 1/b^2)
        a_eff, b_eff = self._effective_semi_axes
        self.Q = np.diag([1.0 / a_eff**2, 1.0 / b_eff**2])

    def distance(self, x: np.ndarray) -> float:
        """
        Signed distance to ellipsoid boundary.

        Uses implicit function: d(x) = sqrt((x-c)^T R^T Q R (x-c)) - 1
        Then scale by harmonic mean of semi-axes for approximate metric distance.

        Args:
            x: Point [px, py, ...]

        Returns:
            dist: Signed distance (approximate for ellipse)
        """
        pos = x[:2]

        # Transform to ellipse coordinates
        diff = pos - self.center
        rotated = self.R_T @ diff

        # Compute normalized distance: sqrt(x^T Q x)
        normalized_dist = np.sqrt(rotated.T @ self.Q @ rotated)

        # Implicit level set: normalized dist - 1
        # Scale by harmonic mean of axes for metric distance approximation
        scale = 2.0 / (1.0 / self._effective_semi_axes[0] + 1.0 / self._effective_semi_axes[1])

        return scale * (normalized_dist - 1.0)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of distance function.

        nabla d(x) = (scale/norm) * R Q R^T (x - c)
        where norm = sqrt((x-c)^T R^T Q R (x-c))

        Args:
            x: Point [px, py, ...]

        Returns:
            grad: Gradient [d/px d/py, ...]
        """
        pos = x[:2]
        diff = pos - self.center
        rotated = self.R_T @ diff

        # Normalized distance
        normalized_dist = np.sqrt(rotated.T @ self.Q @ rotated)

        if normalized_dist < 1e-10:
            # At center, return arbitrary direction
            return np.array([1.0, 0.0])

        grad = (self.R @ self.Q @ rotated) / normalized_dist

        scale = 2.0 / (1.0 / self._effective_semi_axes[0] + 1.0 / self._effective_semi_axes[1])

        return scale * grad

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of rotated ellipse.

        For a rotated ellipse, the bounding box is determined by finding
        the maximum extent in each axis direction
        """
        a_eff, b_eff = self._effective_semi_axes
        c, s = np.cos(self.rotation), np.sin(self.rotation)

        # Max extent in x and y
        dx = np.sqrt((a_eff * c) ** 2 + (b_eff * s) ** 2)
        dy = np.sqrt((a_eff * s) ** 2 + (b_eff * c) ** 2)

        lower = self.center - np.array([dx, dy])
        upper = self.center + np.array([dx, dy])
        return lower, upper

    def get_center(self) -> np.ndarray:
        """Get center of ellipsoid."""
        return self.center.copy()

    def get_semi_axes(self) -> np.ndarray:
        """Get semi-axes lengths of ellipsoid. (without safety margin)"""
        return self.semi_axes.copy()

    def get_effective_semi_axes(self) -> np.ndarray:
        """Get effective semi-axes lengths (includes safety margin)."""
        return self._effective_semi_axes.copy()

    def get_rotation(self) -> float:
        """Get rotation angle of ellipsoid. (radians)"""
        return self.rotation

    def __repr__(self) -> str:
        return (
            f"EllipsoidalObstacle(center={self.center}, semi_axes={self.semi_axes}, "
            f"rotation={self.rotation}, safety_margin={self.safety_margin})"
        )
