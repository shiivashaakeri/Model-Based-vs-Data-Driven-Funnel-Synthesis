"""
Workspace management and visualization.

This module provides:
- Workspace representation with bounds and obstacles
- Visualization utilities for trajectories, obstacles, and funnels
- Configuration and state management
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from .collision import CollisionChecker
from .obstacles import CircularObstacle, EllipsoidalObstacle, Obstacle


class Workspace:
    """
    2D workspace representation for trajectory planning and visualization.

    Manages:
    - Workspace bounds
    - Obstacles
    - Collision checking
    - Visualization
    """

    def __init__(self, bounds: Tuple[np.ndarray, np.ndarray], obstacles: Optional[List[Obstacle]] = None):
        """
        Initialize workspace.

        Args:
            bounds: (lower, upper) bounds [[x_min, y_min], [x_max, y_max]]
            obstacles: List of Obstacle objects (optional)
        """
        lower, upper = bounds
        assert len(lower) == 2 and len(upper) == 2, "Bounds must be 2D [x_min, y_min], [x_max, y_max]"
        assert np.all(upper > lower), "Upper bounds must be greater than lower bounds"

        self.lower_bounds = np.array(lower, dtype=float)
        self.upper_bounds = np.array(upper, dtype=float)

        # Initialize collision checker with obstacles
        self.collision_checker = CollisionChecker(obstacles)

    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the workspace."""
        self.collision_checker.add_obstacle(obstacle)

    def get_obstacles(self) -> List[Obstacle]:
        """Get all obstacles"""
        return self.collision_checker.get_obstacles()

    def add_obstacles(self, obstacles: List[Obstacle]):
        """Add an obstacle to the workspace."""
        self.collision_checker.add_obstacles(obstacles)

    def get_collision_checker(self) -> CollisionChecker:
        """Get collision checker instance."""
        return self.collision_checker

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get workspace bounds."""
        return self.lower_bounds.copy(), self.upper_bounds.copy()

    def is_in_bounds(self, x: np.ndarray) -> bool:
        """
        Check if point is within workspace bounds.

        Args:
            x: Point [x, y]

        Returns:
            in_bounds: True if point is within bounds, False otherwise
        """
        pos = x[:2]
        return np.all(pos >= self.lower_bounds) and np.all(pos <= self.upper_bounds)

    def is_valid_position(self, x: np.ndarray, tolerance: float = 0.0) -> bool:
        """
        Check if point is within workspace bounds and collision-free.

        Args:
            x: Point [x, y]
            tolerance: Additional clearance (meters)

        Returns:
            valid: True if point is within bounds and collision-free, False otherwise
        """
        return self.is_in_bounds(x) and self.collision_checker.is_collision_free(x, tolerance)

    def visualize(self, ax: Optional[plt.Axes] = None, show_grid: bool = True, title: Optional[str] = None) -> plt.Axes:
        """
        Visualize workspace with obstacles.

        Args:
            ax: Matplotlib axes (creates new if None)
            show_grid: Whether to show grid
            title: Plot title

        Returns:
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Set workspace bounds
        ax.set_xlim(self.lower_bounds[0], self.upper_bounds[0])
        ax.set_ylim(self.lower_bounds[1], self.upper_bounds[1])
        ax.set_aspect("equal", adjustable="box")

        # Draw obstacles
        for obs in self.collision_checker.get_obstacles():
            if isinstance(obs, CircularObstacle):
                circle = patches.Circle(
                    obs.get_center(),
                    obs.get_effective_radius(),
                    facecolor="red",
                    edgecolor="darkred",
                    alpha=0.3,
                    linewidth=2,
                )
                ax.add_patch(circle)

            elif isinstance(obs, EllipsoidalObstacle):
                center = obs.get_center()
                width, height = 2 * obs.get_effective_semi_axes()
                angle = np.degrees(obs.get_rotation())

                ellipse = patches.Ellipse(
                    center, width, height, angle=angle, facecolor="red", edgecolor="darkred", alpha=0.3, linewidth=2
                )
                ax.add_patch(ellipse)

        # Grid
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle="--")

        # Labels
        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("y [m]", fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)

        return ax

    def plot_trajectory(
        self,
        x_traj: np.ndarray,
        ax: Optional[plt.Axes] = None,
        color: str = "blue",
        label: Optional[str] = None,
        linewidth: float = 2.0,
        marker: Optional[str] = None,
        markersize: float = 6,
        alpha: float = 1.0,
        show_start_goal: bool = True,
    ) -> plt.Axes:
        """
        Plot trajectory in workspace.

        Args:
            x_traj: Trajectory, shape (N, n) where n >= 3
            ax: Matplotlib axes (creates new with workspace if None)
            color: Line color
            label: Legend label
            linewidth: Line width
            marker: Marker style (e.g., 'o', 'x')
            markersize: Marker size
            alpha: Transparency
            show_start_goal: Show start (green) and goal (red) markers

        Returns:
            ax: Matplotlib axes
        """
        if ax is None:
            ax = self.visualize()

        # Extract position
        px = x_traj[:, 0]
        py = x_traj[:, 1]

        # Plot trajectory
        ax.plot(
            px, py, color=color, label=label, linewidth=linewidth, marker=marker, markersize=markersize, alpha=alpha
        )

        # Start and goal markers
        if show_start_goal:
            ax.plot(px[0], py[0], "go", markersize=10, label="Start", zorder=10)
            ax.plot(px[-1], py[-1], "r*", markersize=15, label="Goal", zorder=10)

        if label:
            ax.legend()

        return ax

    def plot_trajectory_with_heading(
        self,
        x_traj: np.ndarray,
        ax: Optional[plt.Axes] = None,
        color: str = "blue",
        arrow_spacing: int = 5,
        arrow_length: float = 0.3,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot trajectory with heading arrows.

        Args:
            x_traj: Trajectory, shape (N, 3+)
            ax: Matplotlib axes
            color: Color for trajectory and arrows
            arrow_spacing: Plot arrow every N points
            arrow_length: Length of heading arrows
            **kwargs: Additional arguments for plot_trajectory

        Returns:
            ax: Matplotlib axes
        """
        ax = self.plot_trajectory(x_traj, ax=ax, color=color, **kwargs)

        # Plot heading arrows
        for k in range(0, len(x_traj), arrow_spacing):
            px, py, theta = x_traj[k, :3]
            dx = arrow_length * np.cos(theta)
            dy = arrow_length * np.sin(theta)
            ax.arrow(px, py, dx, dy, head_width=0.15, head_length=0.1, fc=color, ec=color, alpha=0.6)

        return ax

    def plot_point(
        self,
        x: np.ndarray,
        ax: Optional[plt.Axes] = None,
        color: str = "blue",
        marker: str = "o",
        markersize: float = 8,
        label: Optional[str] = None,
    ) -> plt.Axes:
        """
        Plot a single point in workspace.

        Args:
            x: Point [px, py, ...]
            ax: Matplotlib axes
            color: Marker color
            marker: Marker style
            markersize: Marker size
            label: Legend label

        Returns:
            ax: Matplotlib axes
        """
        if ax is None:
            ax = self.visualize()

        ax.plot(x[0], x[1], color=color, marker=marker, markersize=markersize, label=label)

        if label:
            ax.legend()

        return ax

    def plot_ellipse(
        self,
        center: np.ndarray,
        P: np.ndarray,
        ax: Optional[plt.Axes] = None,
        color: str = "blue",
        alpha: float = 0.2,
        linewidth: float = 2,
        label: Optional[str] = None,
    ) -> plt.Axes:
        """
        Plot ellipse defined by {x : (x-c)^T P (x-c) <= 1}.

        Useful for visualizing funnels in state space.

        Args:
            center: Center point [cx, cy]
            P: Positive definite matrix (2x2 or larger, uses top-left 2x2)
            ax: Matplotlib axes
            color: Ellipse color
            alpha: Transparency
            linewidth: Border width
            label: Legend label

        Returns:
            ax: Matplotlib axes
        """
        if ax is None:
            ax = self.visualize()

        # Extract 2x2 submatrix if larger
        P_2d = P[:2, :2]

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(P_2d)

        # Semi-axes: sqrt(1/eigenvalue)
        width = 2.0 / np.sqrt(eigvals[0])
        height = 2.0 / np.sqrt(eigvals[1])

        # Rotation angle
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        # Create ellipse patch
        ellipse = patches.Ellipse(
            center[:2],
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )
        ax.add_patch(ellipse)

        if label:
            ax.legend()

        return ax

    def plot_funnel(
        self,
        x_nom_traj: np.ndarray,
        P_sequence: List[np.ndarray],
        ax: Optional[plt.Axes] = None,
        color: str = "blue",
        alpha: float = 0.1,
        spacing: int = 5,
        show_nominal: bool = True,
    ) -> plt.Axes:
        """
        Plot funnel (sequence of ellipsoids) along nominal trajectory.

        Args:
            x_nom_traj: Nominal trajectory, shape (N+1, n)
            P_sequence: List of P matrices (one per segment or timestep)
            ax: Matplotlib axes
            color: Funnel color
            alpha: Transparency
            spacing: Plot ellipse every N points
            show_nominal: Whether to plot nominal trajectory

        Returns:
            ax: Matplotlib axes
        """
        if ax is None:
            ax = self.visualize()

        # Plot nominal trajectory
        if show_nominal:
            self.plot_trajectory(
                x_nom_traj, ax=ax, color="black", label="Nominal", linewidth=1.5, alpha=0.8, show_start_goal=False
            )

        # Plot ellipsoids
        for k in range(0, len(x_nom_traj), spacing):
            # Determine which P matrix to use
            # (This depends on your segmentation strategy)
            if len(P_sequence) == len(x_nom_traj):
                P = P_sequence[k]
            else:
                # Assume P_sequence corresponds to segments
                segment_idx = min(k // spacing, len(P_sequence) - 1)
                P = P_sequence[segment_idx]

            center = x_nom_traj[k, :2]
            self.plot_ellipse(center, P, ax=ax, color=color, alpha=alpha, linewidth=0.5)

        return ax

    def create_figure(self, figsize: Tuple[float, float] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a new figure with workspace visualization.

        Args:
            figsize: Figure size (width, height)

        Returns:
            fig: Matplotlib figure
            ax: Matplotlib axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.visualize(ax=ax)
        return fig, ax

    def __repr__(self) -> str:
        return (
            f"Workspace(bounds=[{self.lower_bounds}, {self.upper_bounds}], "
            f"num_obstacles={self.collision_checker.num_obstacles()})"
        )
