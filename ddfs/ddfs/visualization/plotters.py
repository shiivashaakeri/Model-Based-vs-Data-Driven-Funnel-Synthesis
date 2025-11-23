"""Generic plotting utilities for DDFS visualization.

This module provides system-agnostic plotting functions that work for
any vehicle type (unicycle, quadrotor, etc.). These utilities handle
common visualization tasks like trajectories, controls, and ellipsoids.

The functions in this module are dimension-flexible and make no assumptions
about the specific system being visualized.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# ==============================================================================
# PLOTTING STYLE
# ==============================================================================

# DDFS color scheme
COLORS = {
    "nominal": "#2E86AB",  # Blue - nominal trajectory
    "actual": "#A23B72",  # Purple - actual trajectory
    "funnel": "#F18F01",  # Orange - funnel ellipsoids
    "obstacle": "#C73E1D",  # Red - obstacles
    "workspace": "#6C757D",  # Gray - workspace boundary
    "start": "#06A77D",  # Green - start point
    "goal": "#D62828",  # Dark red - goal point
    "safe": "#06A77D",  # Green - safe/inside funnel
    "unsafe": "#D62828",  # Red - unsafe/outside funnel
}

# Default figure style
FIGURE_STYLE = {
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2,
    "font.size": 10,
}


def setup_figure(
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 100,
    style: Optional[Dict] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Setup figure with consistent DDFS styling.

    Args:
        figsize: Figure size (width, height) in inches
        dpi: Dots per inch for figure resolution
        style: Optional style overrides (updates FIGURE_STYLE)

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axes

    Examples:
        >>> fig, ax = setup_figure(figsize=(12, 8))
        >>> ax.plot([0, 1], [0, 1])
    """
    # Apply style
    plot_style = FIGURE_STYLE.copy()
    if style:
        plot_style.update(style)

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(plot_style)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def setup_figure_3d(
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 100,
    style: Optional[Dict] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Setup 3D figure with consistent DDFS styling.

    Args:
        figsize: Figure size (width, height) in inches
        dpi: Dots per inch for figure resolution
        style: Optional style overrides

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib 3D axes

    Examples:
        >>> fig, ax = setup_figure_3d()
        >>> ax.plot([0, 1], [0, 1], [0, 1])
    """
    # Apply style
    plot_style = FIGURE_STYLE.copy()
    if style:
        plot_style.update(style)

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(plot_style)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    return fig, ax


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = "tight",
    transparent: bool = False,
) -> None:
    """Save figure with consistent settings.

    Args:
        fig: Matplotlib figure to save
        path: Output path (creates parent directories if needed)
        dpi: Resolution for raster formats
        bbox_inches: Bounding box setting ('tight' removes whitespace)
        transparent: Use transparent background

    Examples:
        >>> fig, ax = setup_figure()
        >>> ax.plot([0, 1], [0, 1])
        >>> save_figure(fig, 'results/trajectory.png')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
    )
    print(f"✓ Saved figure: {path}")


# ==============================================================================
# TRAJECTORY PLOTTING
# ==============================================================================


def plot_trajectory(
    x_traj: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: str = "Trajectory",
    color: Optional[str] = None,
    alpha: float = 1.0,
    linewidth: float = 2.0,
    marker: Optional[str] = None,
    markevery: Optional[int] = None,
) -> plt.Axes:
    """Plot generic trajectory (auto-detects 2D or 3D).

    Args:
        x_traj: State trajectory, shape (N, n) where n >= 2
        ax: Matplotlib axes (creates new if None)
        label: Legend label
        color: Line color (uses default if None)
        alpha: Transparency (0=invisible, 1=opaque)
        linewidth: Line width
        marker: Marker style (e.g., 'o', 'x', '^')
        markevery: Show marker every N points

    Returns:
        ax: Matplotlib axes with trajectory plotted

    Examples:
        >>> x_traj = np.random.randn(100, 3)  # 100 timesteps, 3D state
        >>> ax = plot_trajectory(x_traj, label='Random walk')
    """
    if ax is None:
        if x_traj.shape[1] >= 3:
            _, ax = setup_figure_3d()
        else:
            _, ax = setup_figure()

    color = color or COLORS["nominal"]

    # 2D trajectory (plot x vs y)
    if x_traj.shape[1] == 2 or (x_traj.shape[1] > 2 and not hasattr(ax, "zaxis")):
        ax.plot(
            x_traj[:, 0],
            x_traj[:, 1],
            label=label,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            marker=marker,
            markevery=markevery,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # 3D trajectory
    elif x_traj.shape[1] >= 3 and hasattr(ax, "zaxis"):
        ax.plot(
            x_traj[:, 0],
            x_traj[:, 1],
            x_traj[:, 2],
            label=label,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            marker=marker,
            markevery=markevery,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    return ax


def plot_controls(
    u_traj: np.ndarray,
    time: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Control Inputs",
) -> plt.Axes:
    """Plot control inputs over time.

    Args:
        u_traj: Control trajectory, shape (N, m)
        time: Time vector, shape (N,). If None, uses indices
        labels: Control labels (e.g., ['v', 'ω']). If None, uses u_0, u_1, ...
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        ax: Matplotlib axes with controls plotted

    Examples:
        >>> u_traj = np.random.randn(100, 2)
        >>> time = np.linspace(0, 10, 100)
        >>> ax = plot_controls(u_traj, time, labels=['v', 'ω'])
    """
    if ax is None:
        _, ax = setup_figure()

    if time is None:
        time = np.arange(u_traj.shape[0])

    m = u_traj.shape[1]

    if labels is None:
        labels = [f"u_{i}" for i in range(m)]

    for i in range(m):
        ax.plot(time, u_traj[:, i], label=labels[i], linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_state_vs_time(
    x_traj: np.ndarray,
    time: Optional[np.ndarray] = None,
    state_labels: Optional[List[str]] = None,
    x_ref: Optional[np.ndarray] = None,
    fig: Optional[plt.Figure] = None,
    title: str = "State Evolution",
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot each state dimension vs time in subplots.

    Args:
        x_traj: State trajectory, shape (N, n)
        time: Time vector, shape (N,). If None, uses indices
        state_labels: State labels (e.g., ['x', 'y', 'θ'])
        x_ref: Reference trajectory, shape (N, n). If provided, plots comparison
        fig: Matplotlib figure (creates new if None)
        title: Overall title

    Returns:
        fig: Matplotlib figure
        axes: List of subplot axes

    Examples:
        >>> x_traj = np.random.randn(100, 3)
        >>> fig, axes = plot_state_vs_time(x_traj, state_labels=['x', 'y', 'θ'])
    """
    n = x_traj.shape[1]

    if time is None:
        time = np.arange(x_traj.shape[0])

    if state_labels is None:
        state_labels = [f"x_{i}" for i in range(n)]

    if fig is None:
        fig = plt.figure(figsize=(12, 3 * n))

    axes = []

    for i in range(n):
        ax = fig.add_subplot(n, 1, i + 1)

        # Plot actual
        ax.plot(time, x_traj[:, i], label="Actual", color=COLORS["actual"], linewidth=2)

        # Plot reference if provided
        if x_ref is not None:
            ax.plot(
                time, x_ref[:, i], label="Reference", color=COLORS["nominal"], linewidth=2, linestyle="--", alpha=0.7
            )
            ax.legend()

        ax.set_ylabel(state_labels[i])
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title(title)
        if i == n - 1:
            ax.set_xlabel("Time (s)")

        axes.append(ax)

    fig.tight_layout()

    return fig, axes


# ==============================================================================
# ELLIPSOID PLOTTING
# ==============================================================================


def plot_ellipsoid_2d(
    P: np.ndarray,
    c: np.ndarray,
    ax: plt.Axes,
    n_std: float = 1.0,
    color: Optional[str] = None,
    alpha: float = 0.3,
    edgecolor: Optional[str] = None,
    linewidth: float = 2,
    label: Optional[str] = None,
) -> plt.Axes:
    """Plot 2D ellipsoid on existing axes.

    Plots the ellipsoid {x | (x-c)^T P^{-1} (x-c) <= n_std^2}.

    Args:
        P: Shape matrix (2x2, positive definite)
        c: Center point (2,)
        ax: Matplotlib axes
        n_std: Number of standard deviations (radius multiplier)
        color: Fill color
        alpha: Fill transparency
        edgecolor: Edge color
        linewidth: Edge line width
        label: Legend label

    Returns:
        ax: Matplotlib axes with ellipsoid plotted

    Examples:
        >>> fig, ax = setup_figure()
        >>> P = np.diag([1.0, 0.5])
        >>> c = np.array([5.0, 3.0])
        >>> plot_ellipsoid_2d(P, c, ax, label='Funnel')
    """
    # Extract 2D components if P is larger
    P_2d = P[:2, :2]
    c_2d = c[:2]

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(P_2d)

    # Compute ellipse parameters
    # Semi-axes are sqrt(eigenvalues) * n_std
    width = 2 * n_std * np.sqrt(eigvals[0])
    height = 2 * n_std * np.sqrt(eigvals[1])

    # Angle of rotation (in degrees)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Create ellipse
    color = color or COLORS["funnel"]
    edgecolor = edgecolor or color

    ellipse = Ellipse(
        xy=c_2d,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
    )

    ax.add_patch(ellipse)

    return ax


def plot_ellipsoid_3d(
    P: np.ndarray,
    c: np.ndarray,
    ax: plt.Axes,
    n_std: float = 1.0,
    color: Optional[str] = None,
    alpha: float = 0.3,
    resolution: int = 20,
    label: Optional[str] = None,
) -> plt.Axes:
    """Plot 3D ellipsoid on existing axes.

    Plots the ellipsoid {x | (x-c)^T P^{-1} (x-c) <= n_std^2}.

    Args:
        P: Shape matrix (3x3, positive definite)
        c: Center point (3,)
        ax: Matplotlib 3D axes
        n_std: Number of standard deviations (radius multiplier)
        color: Surface color
        alpha: Surface transparency
        resolution: Number of points for mesh
        label: Legend label

    Returns:
        ax: Matplotlib 3D axes with ellipsoid plotted

    Examples:
        >>> fig, ax = setup_figure_3d()
        >>> P = np.diag([1.0, 0.5, 0.3])
        >>> c = np.array([5.0, 3.0, -2.0])
        >>> plot_ellipsoid_3d(P, c, ax, label='Funnel')
    """
    # Extract 3D components if P is larger
    P_3d = P[:3, :3]
    c_3d = c[:3]

    # Create sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    # Stack into points
    sphere_points = np.stack([x_sphere.ravel(), y_sphere.ravel(), z_sphere.ravel()], axis=0)

    # Transform sphere to ellipsoid: x = c + sqrt(P) * sphere_point * n_std
    try:
        P_sqrt = np.linalg.cholesky(P_3d)
    except np.linalg.LinAlgError:
        # If Cholesky fails, use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(P_3d)
        P_sqrt = eigvecs @ np.diag(np.sqrt(np.abs(eigvals))) @ eigvecs.T

    ellipsoid_points = c_3d[:, None] + n_std * P_sqrt @ sphere_points

    # Reshape for plotting
    x_ellipsoid = ellipsoid_points[0].reshape(resolution, resolution)
    y_ellipsoid = ellipsoid_points[1].reshape(resolution, resolution)
    z_ellipsoid = ellipsoid_points[2].reshape(resolution, resolution)

    # Plot surface
    color = color or COLORS["funnel"]

    ax.plot_surface(
        x_ellipsoid,
        y_ellipsoid,
        z_ellipsoid,
        color=color,
        alpha=alpha,
        edgecolor="none",
        label=label,
    )

    return ax


# ==============================================================================
# COMPARISON PLOTS
# ==============================================================================


def plot_trajectory_comparison(
    x_nom: np.ndarray,
    x_actual: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Trajectory Comparison",
) -> plt.Axes:
    """Plot nominal vs actual trajectory comparison.

    Args:
        x_nom: Nominal trajectory, shape (N, n)
        x_actual: Actual trajectory, shape (N, n)
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        ax: Matplotlib axes with both trajectories

    Examples:
        >>> x_nom = np.random.randn(100, 3)
        >>> x_actual = x_nom + 0.1 * np.random.randn(100, 3)
        >>> ax = plot_trajectory_comparison(x_nom, x_actual)
    """
    if ax is None:
        if x_nom.shape[1] >= 3:
            _, ax = setup_figure_3d()
        else:
            _, ax = setup_figure()

    # Plot nominal
    plot_trajectory(x_nom, ax=ax, label="Nominal", color=COLORS["nominal"], linewidth=2.5, alpha=0.8)

    # Plot actual
    plot_trajectory(x_actual, ax=ax, label="Actual", color=COLORS["actual"], linewidth=2, alpha=0.9)

    ax.set_title(title)
    ax.legend()

    return ax


def plot_tracking_error(
    x_nom: np.ndarray,
    x_actual: np.ndarray,
    time: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Tracking Error",
) -> plt.Axes:
    """Plot tracking error magnitude over time.

    Args:
        x_nom: Nominal trajectory, shape (N, n)
        x_actual: Actual trajectory, shape (N, n)
        time: Time vector, shape (N,). If None, uses indices
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        ax: Matplotlib axes with error plot

    Examples:
        >>> x_nom = np.random.randn(100, 3)
        >>> x_actual = x_nom + 0.1 * np.random.randn(100, 3)
        >>> ax = plot_tracking_error(x_nom, x_actual)
    """
    if ax is None:
        _, ax = setup_figure()

    if time is None:
        time = np.arange(x_nom.shape[0])

    # Compute error norm
    error = np.linalg.norm(x_actual - x_nom, axis=1)

    ax.plot(time, error, color=COLORS["actual"], linewidth=2)
    ax.fill_between(time, 0, error, alpha=0.3, color=COLORS["actual"])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_error = np.mean(error)
    max_error = np.max(error)
    ax.axhline(mean_error, color="gray", linestyle="--", alpha=0.5, label=f"Mean: {mean_error:.3f}")
    ax.text(
        0.02,
        0.98,
        f"Max: {max_error:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    ax.legend()

    return ax


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def add_start_goal_markers(
    ax: plt.Axes,
    x0: np.ndarray,
    xf: np.ndarray,
    is_3d: bool = False,
) -> plt.Axes:
    """Add start and goal markers to plot.

    Args:
        ax: Matplotlib axes
        x0: Start state
        xf: Goal state
        is_3d: Whether this is a 3D plot

    Returns:
        ax: Matplotlib axes with markers added
    """
    if is_3d:
        ax.scatter(
            x0[0],
            x0[1],
            x0[2],
            color=COLORS["start"],
            s=200,
            marker="o",
            edgecolors="black",
            linewidth=2,
            label="Start",
            zorder=10,
        )
        ax.scatter(
            xf[0],
            xf[1],
            xf[2],
            color=COLORS["goal"],
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Goal",
            zorder=10,
        )
    else:
        ax.scatter(
            x0[0],
            x0[1],
            color=COLORS["start"],
            s=200,
            marker="o",
            edgecolors="black",
            linewidth=2,
            label="Start",
            zorder=10,
        )
        ax.scatter(
            xf[0],
            xf[1],
            color=COLORS["goal"],
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Goal",
            zorder=10,
        )

    return ax


def set_equal_aspect(ax: plt.Axes, is_3d: bool = False) -> plt.Axes:
    """Set equal aspect ratio for better visualization.

    Args:
        ax: Matplotlib axes
        is_3d: Whether this is a 3D plot

    Returns:
        ax: Matplotlib axes with equal aspect
    """
    if is_3d:
        # For 3D, set equal aspect for all axes
        ax.set_box_aspect([1, 1, 1])
    else:
        ax.set_aspect("equal", adjustable="box")

    return ax
