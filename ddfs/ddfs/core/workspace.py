# ddfs/ddfs/core/workspace.py

"""
Workspace definitions for environment boundaries.

This module implements workspace classes which define the physical
bounds of the operating environment.

The workspace is used in:
    - Phase 1: Planning (constraint on reachable states)
    - Phase 2: Data collection (bounds for safe exploration)
    - Phase 4: Feasibility (bounds for ellipsoid computation)
    - Phase 6: Deployment (verify robot stays in workspace)
    - Visualization: Setting plot limits

Key Classes
-----------
Workspace : Abstract base for all workspaces
Workspace2D : Rectangular workspace for unicycle (2D)
Workspace3D : Rectangular cuboid workspace for quadrotor (3D)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class Workspace(ABC):
    """
    Abstract base class for workspaces.

    All workspace classes must implement:
        - contains(point): Check if point is inside workspace
        - distance_to_boundary(point): Distance to nearest boundary
        - clip_to_workspace(point): Project point to workspace
        - sample_random_point(): Sample uniformly in workspace
        - to_dict(): Convert to dictionary format
        - from_config(): Create from configuration
    """

    @abstractmethod
    def contains(self, x: np.ndarray, margin: float = 0.0) -> bool:
        """
        Check if point is inside workspace.

        Parameters
        ----------
        x : np.ndarray
            State vector
        margin : float, optional
            Safety margin (point must be at least 'margin' from boundary)

        Returns
        -------
        inside : bool
            True if point is inside workspace (with margin)
        """
        pass

    @abstractmethod
    def distance_to_boundary(self, x: np.ndarray) -> float:
        """
        Compute distance from point to nearest workspace boundary.

        Parameters
        ----------
        x : np.ndarray
            State vector

        Returns
        -------
        distance : float
            Distance to nearest boundary (positive inside, negative outside)
        """
        pass

    @abstractmethod
    def clip_to_workspace(self, x: np.ndarray) -> np.ndarray:
        """
        Clip point to workspace bounds.

        Parameters
        ----------
        x : np.ndarray
            State vector

        Returns
        -------
        x_clipped : np.ndarray
            State with position clipped to workspace
        """
        pass

    @abstractmethod
    def sample_random_point(self, margin: float = 0.0) -> np.ndarray:
        """
        Sample uniformly random point in workspace.

        Parameters
        ----------
        margin : float, optional
            Keep samples at least 'margin' distance from boundaries

        Returns
        -------
        point : np.ndarray
            Random point in workspace
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workspace to dictionary format.

        Returns
        -------
        config : dict
            Dictionary representation
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "Workspace":
        """
        Create workspace from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary

        Returns
        -------
        workspace : Workspace
            Workspace object
        """
        pass


class Workspace2D(Workspace):
    """
    Rectangular workspace bounds in 2D.

    Defines the valid operating region as:
        x_min ≤ x ≤ x_max
        y_min ≤ y ≤ y_max

    Used for unicycle planning and visualization.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate (m)
    x_max : float
        Maximum x coordinate (m)
    y_min : float
        Minimum y coordinate (m)
    y_max : float
        Maximum y coordinate (m)

    Attributes
    ----------
    x_min, x_max, y_min, y_max : float
        Workspace boundaries
    width : float
        Width of workspace (x_max - x_min)
    height : float
        Height of workspace (y_max - y_min)
    center : np.ndarray
        Center point of workspace [x_c, y_c]
    area : float
        Area of workspace

    Examples
    --------
    >>> from ddfs.core.workspace import Workspace2D
    >>> import numpy as np
    >>>
    >>> workspace = Workspace2D(x_min=0.0, x_max=12.0, y_min=0.0, y_max=8.0)
    >>>
    >>> # Check if point is inside
    >>> x = np.array([5.0, 4.0, 0.5])  # [x, y, θ]
    >>> print(workspace.contains(x))
    True
    >>>
    >>> # Sample random point
    >>> point = workspace.sample_random_point(margin=0.5)
    >>> print(point)  # [x, y] within [0.5, 11.5] x [0.5, 7.5]
    """

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Initialize 2D rectangular workspace.

        Parameters
        ----------
        x_min : float
            Minimum x coordinate
        x_max : float
            Maximum x coordinate
        y_min : float
            Minimum y coordinate
        y_max : float
            Maximum y coordinate

        Raises
        ------
        ValueError
            If bounds are invalid (min >= max)
        """
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        # Validate bounds
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be less than y_max ({self.y_max})")

        # Compute derived properties
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.center = np.array([(self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2])
        self.area = self.width * self.height

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounds as tuple (x_min, x_max, y_min, y_max)."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def contains(self, x: np.ndarray, margin: float = 0.0) -> bool:
        """
        Check if point is inside workspace.

        Parameters
        ----------
        x : np.ndarray
            State vector (must have at least 2 elements for [x, y])
        margin : float, optional
            Safety margin (default: 0.0)
            Point must be at least 'margin' distance from boundary

        Returns
        -------
        inside : bool
            True if point is inside workspace (with margin)
        """
        pos = x[:2]  # Extract [x, y]

        in_x = (self.x_min + margin) <= pos[0] <= (self.x_max - margin)
        in_y = (self.y_min + margin) <= pos[1] <= (self.y_max - margin)

        return in_x and in_y

    def distance_to_boundary(self, x: np.ndarray) -> float:
        """
        Compute distance from point to nearest workspace boundary.

        Parameters
        ----------
        x : np.ndarray
            State vector

        Returns
        -------
        distance : float
            Distance to nearest boundary
            Positive if inside, negative if outside
        """
        pos = x[:2]

        # Distance to each boundary
        dist_to_x_min = pos[0] - self.x_min
        dist_to_x_max = self.x_max - pos[0]
        dist_to_y_min = pos[1] - self.y_min
        dist_to_y_max = self.y_max - pos[1]

        # Minimum distance (negative if outside)
        min_dist = min(dist_to_x_min, dist_to_x_max, dist_to_y_min, dist_to_y_max)

        return float(min_dist)

    def clip_to_workspace(self, x: np.ndarray) -> np.ndarray:
        """
        Clip point to workspace bounds.

        Parameters
        ----------
        x : np.ndarray
            State vector

        Returns
        -------
        x_clipped : np.ndarray
            State with position clipped to workspace
        """
        x_clipped = x.copy()
        x_clipped[0] = np.clip(x_clipped[0], self.x_min, self.x_max)
        x_clipped[1] = np.clip(x_clipped[1], self.y_min, self.y_max)

        return x_clipped

    def sample_random_point(self, margin: float = 0.0) -> np.ndarray:
        """
        Sample uniformly random point in workspace.

        Parameters
        ----------
        margin : float, optional
            Keep samples at least 'margin' distance from boundaries

        Returns
        -------
        point : np.ndarray
            Random point [x, y] in workspace
        """
        x = np.random.uniform(self.x_min + margin, self.x_max - margin)
        y = np.random.uniform(self.y_min + margin, self.y_max - margin)

        return np.array([x, y])

    def get_corners(self) -> np.ndarray:
        """
        Get corner points of workspace.

        Returns
        -------
        corners : np.ndarray
            Corner points, shape (4, 2)
            Order: [bottom-left, bottom-right, top-right, top-left]
        """
        return np.array(
            [
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max],
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workspace to dictionary format.

        Returns
        -------
        config : dict
            Dictionary with workspace parameters
        """
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Workspace2D":
        """
        Create workspace from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration with workspace bounds

        Returns
        -------
        workspace : Workspace2D
            Workspace object

        Examples
        --------
        >>> config = {
        ...     'x_min': 0.0, 'x_max': 12.0,
        ...     'y_min': 0.0, 'y_max': 8.0
        ... }
        >>> workspace = Workspace2D.from_config(config)
        """
        return cls(
            x_min=config["x_min"],
            x_max=config["x_max"],
            y_min=config["y_min"],
            y_max=config["y_max"],
        )

    def plot(self, ax, edgecolor="black", facecolor="none", linewidth=2, linestyle="-", alpha=1.0):
        """
        Plot workspace boundary on matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        edgecolor : str, optional
            Edge color (default: 'black')
        facecolor : str, optional
            Fill color (default: 'none')
        linewidth : float, optional
            Line width (default: 2)
        linestyle : str, optional
            Line style (default: '-')
        alpha : float, optional
            Transparency (default: 1.0)
        """
        from matplotlib.patches import Rectangle  # noqa: PLC0415

        rect = Rectangle(
            (self.x_min, self.y_min),
            self.width,
            self.height,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label="Workspace",
        )
        ax.add_patch(rect)

        # Set axis limits with small margin
        margin = 0.1 * max(self.width, self.height)
        ax.set_xlim(self.x_min - margin, self.x_max + margin)
        ax.set_ylim(self.y_min - margin, self.y_max + margin)
        ax.set_aspect("equal")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Workspace2D("
            f"x=[{self.x_min:.2f}, {self.x_max:.2f}], "
            f"y=[{self.y_min:.2f}, {self.y_max:.2f}], "
            f"area={self.area:.2f})"
        )


class Workspace3D(Workspace):
    """
    Rectangular cuboid workspace bounds in 3D.

    Defines the valid operating region as:
        x_min ≤ x ≤ x_max
        y_min ≤ y ≤ y_max
        z_min ≤ z ≤ z_max

    Used for quadrotor planning and visualization.

    Note: Uses NED convention (z points down, so z_min < 0 is "up")

    Parameters
    ----------
    x_min, x_max : float
        Bounds in x direction (m)
    y_min, y_max : float
        Bounds in y direction (m)
    z_min, z_max : float
        Bounds in z direction (m, NED: negative is up)

    Attributes
    ----------
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Workspace boundaries
    width, height, depth : float
        Dimensions of workspace
    center : np.ndarray
        Center point of workspace [x_c, y_c, z_c]
    volume : float
        Volume of workspace

    Examples
    --------
    >>> from ddfs.core.workspace import Workspace3D
    >>> import numpy as np
    >>>
    >>> workspace = Workspace3D(
    ...     x_min=0.0, x_max=8.0,
    ...     y_min=0.0, y_max=8.0,
    ...     z_min=-5.0, z_max=0.5
    ... )
    >>>
    >>> # Check if quadrotor state is inside
    >>> x = np.zeros(13)
    >>> x[:3] = [4.0, 4.0, -2.0]  # Position
    >>> x[6] = 1.0  # Identity quaternion
    >>> print(workspace.contains(x))
    True
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
    ):
        """
        Initialize 3D rectangular workspace.

        Parameters
        ----------
        x_min, x_max : float
            Bounds in x direction
        y_min, y_max : float
            Bounds in y direction
        z_min, z_max : float
            Bounds in z direction (NED: negative is up)

        Raises
        ------
        ValueError
            If bounds are invalid (min >= max)
        """
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

        # Validate bounds
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be less than y_max ({self.y_max})")
        if self.z_min >= self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be less than z_max ({self.z_max})")

        # Compute derived properties
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.depth = self.z_max - self.z_min
        self.center = np.array(
            [
                (self.x_min + self.x_max) / 2,
                (self.y_min + self.y_max) / 2,
                (self.z_min + self.z_max) / 2,
            ]
        )
        self.volume = self.width * self.height * self.depth

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Get bounds as tuple (x_min, x_max, y_min, y_max, z_min, z_max)."""
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def contains(self, x: np.ndarray, margin: float = 0.0) -> bool:
        """
        Check if point is inside workspace.

        Parameters
        ----------
        x : np.ndarray
            State vector (must have at least 3 elements for [x, y, z])
        margin : float, optional
            Safety margin (default: 0.0)

        Returns
        -------
        inside : bool
            True if point is inside workspace (with margin)
        """
        pos = x[:3]  # Extract [x, y, z]

        in_x = (self.x_min + margin) <= pos[0] <= (self.x_max - margin)
        in_y = (self.y_min + margin) <= pos[1] <= (self.y_max - margin)
        in_z = (self.z_min + margin) <= pos[2] <= (self.z_max - margin)

        return in_x and in_y and in_z

    def distance_to_boundary(self, x: np.ndarray) -> float:
        """
        Compute distance from point to nearest workspace boundary.

        Parameters
        ----------
        x : np.ndarray
            State vector

        Returns
        -------
        distance : float
            Distance to nearest boundary (positive inside, negative outside)
        """
        pos = x[:3]

        # Distance to each boundary
        dist_to_x_min = pos[0] - self.x_min
        dist_to_x_max = self.x_max - pos[0]
        dist_to_y_min = pos[1] - self.y_min
        dist_to_y_max = self.y_max - pos[1]
        dist_to_z_min = pos[2] - self.z_min
        dist_to_z_max = self.z_max - pos[2]

        # Minimum distance
        min_dist = min(dist_to_x_min, dist_to_x_max, dist_to_y_min, dist_to_y_max, dist_to_z_min, dist_to_z_max)

        return float(min_dist)

    def clip_to_workspace(self, x: np.ndarray) -> np.ndarray:
        """
        Clip point to workspace bounds.

        Parameters
        ----------
        x : np.ndarray
            State vector

        Returns
        -------
        x_clipped : np.ndarray
            State with position clipped to workspace
        """
        x_clipped = x.copy()
        x_clipped[0] = np.clip(x_clipped[0], self.x_min, self.x_max)
        x_clipped[1] = np.clip(x_clipped[1], self.y_min, self.y_max)
        x_clipped[2] = np.clip(x_clipped[2], self.z_min, self.z_max)

        return x_clipped

    def sample_random_point(self, margin: float = 0.0) -> np.ndarray:
        """
        Sample uniformly random point in workspace.

        Parameters
        ----------
        margin : float, optional
            Keep samples at least 'margin' distance from boundaries

        Returns
        -------
        point : np.ndarray
            Random point [x, y, z] in workspace
        """
        x = np.random.uniform(self.x_min + margin, self.x_max - margin)
        y = np.random.uniform(self.y_min + margin, self.y_max - margin)
        z = np.random.uniform(self.z_min + margin, self.z_max - margin)

        return np.array([x, y, z])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert workspace to dictionary format.

        Returns
        -------
        config : dict
            Dictionary with workspace parameters
        """
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Workspace3D":
        """
        Create workspace from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration with workspace bounds

        Returns
        -------
        workspace : Workspace3D
            Workspace object
        """
        return cls(
            x_min=config["x_min"],
            x_max=config["x_max"],
            y_min=config["y_min"],
            y_max=config["y_max"],
            z_min=config["z_min"],
            z_max=config["z_max"],
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Workspace3D("
            f"x=[{self.x_min:.2f}, {self.x_max:.2f}], "
            f"y=[{self.y_min:.2f}, {self.y_max:.2f}], "
            f"z=[{self.z_min:.2f}, {self.z_max:.2f}], "
            f"volume={self.volume:.2f})"
        )
