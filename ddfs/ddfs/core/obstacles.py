# ddfs/ddfs/core/obstacles.py

"""
Unified obstacle definitions for all systems.

This module provides obstacle classes for collision avoidance
in trajectory planning and funnel synthesis.

Obstacles are used in:
    - Phase 1: Planning (trajectory optimization avoidance constraints)
    - Phase 4: Feasibility (ellipsoid computation with safety margins)
    - Visualization: Plotting environment

Key Classes
-----------
Obstacle : Abstract base class for all obstacles
CircleObstacle : 2D circular obstacle (for unicycle)
SphereObstacle : 3D spherical obstacle (for quadrotor)

Factory Functions
-----------------
create_obstacles_from_config : Create list of obstacles from configuration
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class Obstacle(ABC):
    """
    Abstract base class for obstacles.

    All obstacle classes must implement:
        - contains(point): Check if point is inside obstacle
        - distance_to(point): Compute distance from point to obstacle surface
        - to_dict(): Convert to dictionary format

    Attributes
    ----------
    id : str
        Unique identifier for this obstacle
    center : np.ndarray
        Center position of obstacle
    radius : float
        Radius of obstacle
    safety_margin : float
        Additional safety buffer around obstacle
    """

    def __init__(self, obstacle_id: str, center: np.ndarray, radius: float, safety_margin: float = 0.0):
        """
        Initialize obstacle.

        Parameters
        ----------
        obstacle_id : str
            Unique identifier
        center : np.ndarray
            Center position
        radius : float
            Obstacle radius
        safety_margin : float, optional
            Safety buffer (m), by default 0.0
        """
        self.id = obstacle_id
        self.center = np.array(center)
        self.radius = float(radius)
        self.safety_margin = float(safety_margin)

    @property
    def effective_radius(self) -> float:
        """
        Effective radius including safety margin.

        Returns
        -------
        r_eff : float
            radius + safety_margin
        """
        return self.radius + self.safety_margin

    @abstractmethod
    def contains(self, point: np.ndarray, include_margin: bool = True) -> bool:
        """
        Check if point is inside obstacle.

        Parameters
        ----------
        point : np.ndarray
            Point to check
        include_margin : bool, optional
            Include safety margin in check, by default True

        Returns
        -------
        inside : bool
            True if point is inside obstacle (+ margin if include_margin=True)
        """
        pass

    @abstractmethod
    def distance_to(self, point: np.ndarray, include_margin: bool = True) -> float:
        """
        Compute signed distance from point to obstacle surface.

        Distance convention:
            - Positive: point is outside obstacle
            - Negative: point is inside obstacle
            - Zero: point is on obstacle surface

        Parameters
        ----------
        point : np.ndarray
            Point to compute distance from
        include_margin : bool, optional
            Include safety margin, by default True

        Returns
        -------
        distance : float
            Signed distance (positive = safe, negative = collision)
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert obstacle to dictionary format.

        Returns
        -------
        config : dict
            Dictionary representation
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"id='{self.id}', "
            f"center={self.center}, "
            f"radius={self.radius:.2f}, "
            f"margin={self.safety_margin:.2f})"
        )


class CircleObstacle(Obstacle):
    """
    2D circular obstacle for unicycle planning.

    The obstacle occupies the region:
        ||[x, y] - center||₂ ≤ radius

    With safety margin, the keep-out region is:
        ||[x, y] - center||₂ ≤ radius + safety_margin

    Parameters
    ----------
    obstacle_id : str
        Unique identifier
    center : array-like
        Center position [x, y], shape (2,)
    radius : float
        Obstacle radius (m)
    safety_margin : float, optional
        Safety buffer (m), by default 0.0

    Examples
    --------
    >>> from ddfs.core.obstacles import CircleObstacle
    >>> import numpy as np
    >>>
    >>> obs = CircleObstacle("obs_1", center=[4.0, 3.0], radius=1.0, safety_margin=0.25)
    >>>
    >>> # Check if point is inside
    >>> point = np.array([4.5, 3.0])
    >>> print(obs.contains(point))
    True
    >>>
    >>> # Compute distance
    >>> point = np.array([6.0, 3.0])
    >>> dist = obs.distance_to(point)
    >>> print(f"Distance: {dist:.2f} m")
    Distance: 0.75 m
    """

    def __init__(
        self,
        obstacle_id: str,
        center: List[float],
        radius: float,
        safety_margin: float = 0.0,
    ):
        """
        Initialize 2D circular obstacle.

        Parameters
        ----------
        obstacle_id : str
            Unique identifier
        center : list
            Center position [x, y]
        radius : float
            Obstacle radius (m)
        safety_margin : float, optional
            Safety buffer (m)
        """
        if len(center) != 2:
            raise ValueError(f"Circle center must be 2D, got {len(center)}D")

        super().__init__(obstacle_id, np.array(center), radius, safety_margin)

    def contains(self, point: np.ndarray, include_margin: bool = True) -> bool:
        """
        Check if point is inside circular obstacle.

        Parameters
        ----------
        point : np.ndarray
            Point [x, y], shape (2,) or (3,) (θ ignored)
        include_margin : bool, optional
            Include safety margin, by default True

        Returns
        -------
        inside : bool
            True if point is inside obstacle
        """
        # Extract 2D position (ignore θ if present)
        pos = point[:2]

        # Compute distance from center
        dist = np.linalg.norm(pos - self.center)

        # Check against radius (with or without margin)
        r_check = self.effective_radius if include_margin else self.radius

        return dist <= r_check

    def distance_to(self, point: np.ndarray, include_margin: bool = True) -> float:
        """
        Compute signed distance from point to circle boundary.

        Parameters
        ----------
        point : np.ndarray
            Point [x, y], shape (2,) or (3,) (θ ignored)
        include_margin : bool, optional
            Include safety margin, by default True

        Returns
        -------
        distance : float
            Signed distance (positive = outside, negative = inside)
        """
        # Extract 2D position
        pos = point[:2]

        # Distance from center
        dist_from_center = np.linalg.norm(pos - self.center)

        # Radius to check against
        r_check = self.effective_radius if include_margin else self.radius

        # Signed distance (positive outside, negative inside)
        return float(dist_from_center - r_check)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns
        -------
        config : dict
            Dictionary with obstacle parameters
        """
        return {
            "id": self.id,
            "type": "circle",
            "center": self.center.tolist(),
            "radius": self.radius,
            "safety_margin": self.safety_margin,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CircleObstacle":
        """
        Create obstacle from dictionary.

        Parameters
        ----------
        config : dict
            Dictionary with obstacle parameters

        Returns
        -------
        obstacle : CircleObstacle
            Obstacle object
        """
        return cls(
            obstacle_id=config["id"],
            center=config["center"],
            radius=config["radius"],
            safety_margin=config.get("safety_margin", 0.0),
        )


class SphereObstacle(Obstacle):
    """
    3D spherical obstacle for quadrotor planning.

    The obstacle occupies the region:
        ||[x, y, z] - center||₂ ≤ radius

    With safety margin, the keep-out region is:
        ||[x, y, z] - center||₂ ≤ radius + safety_margin

    Parameters
    ----------
    obstacle_id : str
        Unique identifier
    center : array-like
        Center position [x, y, z], shape (3,)
    radius : float
        Obstacle radius (m)
    safety_margin : float, optional
        Safety buffer (m), by default 0.0

    Examples
    --------
    >>> from ddfs.core.obstacles import SphereObstacle
    >>> import numpy as np
    >>>
    >>> obs = SphereObstacle("obs_1", center=[2.0, 2.0, -1.5], radius=0.5, safety_margin=0.2)
    >>>
    >>> # Check if point is inside
    >>> point = np.array([2.0, 2.0, -1.0])
    >>> print(obs.contains(point))
    True
    >>>
    >>> # Compute distance (for quadrotor state, extract position)
    >>> x_quadrotor = np.zeros(13)  # Full state
    >>> x_quadrotor[:3] = [3.0, 3.0, -2.0]  # Position
    >>> dist = obs.distance_to(x_quadrotor[:3])
    >>> print(f"Distance: {dist:.2f} m")
    """

    def __init__(
        self,
        obstacle_id: str,
        center: List[float],
        radius: float,
        safety_margin: float = 0.0,
    ):
        """
        Initialize 3D spherical obstacle.

        Parameters
        ----------
        obstacle_id : str
            Unique identifier
        center : list
            Center position [x, y, z]
        radius : float
            Obstacle radius (m)
        safety_margin : float, optional
            Safety buffer (m)
        """
        if len(center) != 3:
            raise ValueError(f"Sphere center must be 3D, got {len(center)}D")

        super().__init__(obstacle_id, np.array(center), radius, safety_margin)

    def contains(self, point: np.ndarray, include_margin: bool = True) -> bool:
        """
        Check if point is inside spherical obstacle.

        Parameters
        ----------
        point : np.ndarray
            Point [x, y, z], shape (3,) or (13,) (for full quadrotor state)
        include_margin : bool, optional
            Include safety margin, by default True

        Returns
        -------
        inside : bool
            True if point is inside obstacle
        """
        # Extract 3D position (first 3 elements for quadrotor state)
        pos = point[:3]

        # Compute distance from center
        dist = np.linalg.norm(pos - self.center)

        # Check against radius (with or without margin)
        r_check = self.effective_radius if include_margin else self.radius

        return dist <= r_check

    def distance_to(self, point: np.ndarray, include_margin: bool = True) -> float:
        """
        Compute signed distance from point to sphere boundary.

        Parameters
        ----------
        point : np.ndarray
            Point [x, y, z], shape (3,) or (13,) (for full quadrotor state)
        include_margin : bool, optional
            Include safety margin, by default True

        Returns
        -------
        distance : float
            Signed distance (positive = outside, negative = inside)
        """
        # Extract 3D position
        pos = point[:3]

        # Distance from center
        dist_from_center = np.linalg.norm(pos - self.center)

        # Radius to check against
        r_check = self.effective_radius if include_margin else self.radius

        # Signed distance (positive outside, negative inside)
        return float(dist_from_center - r_check)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns
        -------
        config : dict
            Dictionary with obstacle parameters
        """
        return {
            "id": self.id,
            "type": "sphere",
            "center": self.center.tolist(),
            "radius": self.radius,
            "safety_margin": self.safety_margin,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SphereObstacle":
        """
        Create obstacle from dictionary.

        Parameters
        ----------
        config : dict
            Dictionary with obstacle parameters

        Returns
        -------
        obstacle : SphereObstacle
            Obstacle object
        """
        return cls(
            obstacle_id=config["id"],
            center=config["center"],
            radius=config["radius"],
            safety_margin=config.get("safety_margin", 0.0),
        )


def create_obstacles_from_config(obstacles_config: List[Dict[str, Any]], system_type: str) -> List[Obstacle]:
    """
    Factory function to create obstacles from configuration.

    Parameters
    ----------
    obstacles_config : list of dict
        List of obstacle configurations
    system_type : str
        System type: "unicycle" or "quadrotor"

    Returns
    -------
    obstacles : list of Obstacle
        List of obstacle objects

    Raises
    ------
    ValueError
        If obstacle type doesn't match system type

    Examples
    --------
    >>> config = [
    ...     {
    ...         'id': 'obs_1',
    ...         'type': 'circle',
    ...         'center': [4.0, 3.0],
    ...         'radius': 1.0,
    ...         'safety_margin': 0.25
    ...     },
    ...     {
    ...         'id': 'obs_2',
    ...         'type': 'circle',
    ...         'center': [8.0, 3.0],
    ...         'radius': 1.0,
    ...         'safety_margin': 0.25
    ...     }
    ... ]
    >>> obstacles = create_obstacles_from_config(config, system_type='unicycle')
    >>> print(len(obstacles))
    2
    """
    obstacles = []

    for obs_config in obstacles_config:
        obs_type = obs_config.get("type", "").lower()

        # Create appropriate obstacle type
        if obs_type == "circle":
            if system_type != "unicycle":
                raise ValueError(f"Circle obstacles only for unicycle, got system_type='{system_type}'")
            obstacle = CircleObstacle.from_dict(obs_config)

        elif obs_type == "sphere":
            if system_type != "quadrotor":
                raise ValueError(f"Sphere obstacles only for quadrotor, got system_type='{system_type}'")
            obstacle = SphereObstacle.from_dict(obs_config)

        else:
            raise ValueError(f"Unknown obstacle type: '{obs_type}'. Must be 'circle' or 'sphere'")

        obstacles.append(obstacle)

    return obstacles


def check_collision_free(point: np.ndarray, obstacles: List[Obstacle], include_margin: bool = True) -> bool:
    """
    Check if point is collision-free with respect to all obstacles.

    Parameters
    ----------
    point : np.ndarray
        Point to check
    obstacles : list of Obstacle
        List of obstacles
    include_margin : bool, optional
        Include safety margins, by default True

    Returns
    -------
    collision_free : bool
        True if point is outside all obstacles

    Examples
    --------
    >>> from ddfs.core.obstacles import CircleObstacle, check_collision_free
    >>> import numpy as np
    >>>
    >>> obstacles = [
    ...     CircleObstacle("obs_1", [4.0, 3.0], 1.0, 0.25),
    ...     CircleObstacle("obs_2", [8.0, 3.0], 1.0, 0.25)
    ... ]
    >>>
    >>> point = np.array([1.0, 1.0])
    >>> print(check_collision_free(point, obstacles))
    True
    >>>
    >>> point = np.array([4.0, 3.0])  # Inside obs_1
    >>> print(check_collision_free(point, obstacles))
    False
    """
    return all(not obs.contains(point, include_margin=include_margin) for obs in obstacles)


def minimum_distance_to_obstacles(point: np.ndarray, obstacles: List[Obstacle], include_margin: bool = True) -> float:
    """
    Compute minimum distance from point to any obstacle.

    Parameters
    ----------
    point : np.ndarray
        Point to compute distance from
    obstacles : list of Obstacle
        List of obstacles
    include_margin : bool, optional
        Include safety margins, by default True

    Returns
    -------
    min_distance : float
        Minimum signed distance to any obstacle
        (positive = safe, negative = collision)

    Examples
    --------
    >>> from ddfs.core.obstacles import CircleObstacle, minimum_distance_to_obstacles
    >>> import numpy as np
    >>>
    >>> obstacles = [
    ...     CircleObstacle("obs_1", [4.0, 3.0], 1.0, 0.25),
    ...     CircleObstacle("obs_2", [8.0, 3.0], 1.0, 0.25)
    ... ]
    >>>
    >>> point = np.array([1.0, 1.0])
    >>> min_dist = minimum_distance_to_obstacles(point, obstacles)
    >>> print(f"Min distance: {min_dist:.2f} m")
    """
    if not obstacles:
        return float("inf")

    distances = [obs.distance_to(point, include_margin=include_margin) for obs in obstacles]
    return float(np.min(distances))
