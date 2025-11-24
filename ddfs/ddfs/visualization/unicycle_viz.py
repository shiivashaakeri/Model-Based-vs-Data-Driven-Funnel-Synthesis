"""Unicycle-specific visualization utilities.

This module provides 2D visualization functions for the unicycle system,
including workspace plots, obstacle visualization, trajectory tracking,
and funnel cross-sections.

The unicycle has state x = [x, y, θ] and operates in 2D space.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from ddfs.core.obstacles import Obstacle
from ddfs.core.workspace import Workspace2D
from ddfs.planning import NominalTrajectory
from ddfs.synthesis import FunnelLibrary
from ddfs.visualization.plotters import (
    COLORS,
    add_start_goal_markers,
    plot_ellipsoid_2d,
    plot_tracking_error,
    save_figure,
    set_equal_aspect,
    setup_figure,
)

# ==============================================================================
# WORKSPACE AND OBSTACLES
# ==============================================================================


def plot_workspace(
    workspace: Workspace2D,
    obstacles: List[Obstacle],
    ax: Optional[plt.Axes] = None,
    show_grid: bool = True,
) -> plt.Axes:
    """Plot 2D workspace with obstacles.

    Args:
        workspace: 2D workspace (rectangle)
        obstacles: List of circular obstacles
        ax: Matplotlib axes (creates new if None)
        show_grid: Whether to show grid

    Returns:
        ax: Matplotlib axes with workspace plotted

    Examples:
        >>> from ddfs.core import Workspace2D
        >>> from ddfs.core.obstacles import CircleObstacle
        >>>
        >>> workspace = Workspace2D(x_min=0, x_max=12, y_min=0, y_max=8)
        >>> obstacles = [CircleObstacle([4, 3], 1.0)]
        >>> ax = plot_workspace(workspace, obstacles)
    """
    if ax is None:
        _, ax = setup_figure(figsize=(10, 7))

    # Plot workspace boundary
    x_min, x_max, y_min, y_max = workspace.bounds

    ax.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color=COLORS["workspace"],
        linewidth=2.5,
        label="Workspace",
    )

    # Plot obstacles
    for i, obs in enumerate(obstacles):
        circle = Circle(
            obs.center[:2],
            obs.effective_radius,
            color=COLORS["obstacle"],
            alpha=0.4,
            edgecolor=COLORS["obstacle"],
            linewidth=2,
            label="Obstacle" if i == 0 else None,
        )
        ax.add_patch(circle)

        # Add label
        ax.text(obs.center[0], obs.center[1], obs.id, ha="center", va="center", fontsize=9, fontweight="bold")

    # Set limits with padding
    padding = 0.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(show_grid, alpha=0.3)
    ax.legend(loc="upper right")

    return ax


# ==============================================================================
# NOMINAL TRAJECTORY
# ==============================================================================


def plot_nominal_trajectory(
    nominal: NominalTrajectory,
    workspace: Workspace2D,
    obstacles: List[Obstacle],
    ax: Optional[plt.Axes] = None,
    show_heading: bool = True,
    heading_interval: int = 10,
    arrow_scale: float = 0.5,
) -> plt.Axes:
    """Plot nominal trajectory with workspace and obstacles.

    Args:
        nominal: Nominal trajectory
        workspace: 2D workspace
        obstacles: List of obstacles
        ax: Matplotlib axes (creates new if None)
        show_heading: Whether to show heading arrows
        heading_interval: Show heading every N timesteps
        arrow_scale: Scale factor for heading arrows

    Returns:
        ax: Matplotlib axes with nominal trajectory

    Examples:
        >>> from ddfs.planning import NominalTrajectory
        >>> import numpy as np
        >>>
        >>> x_nom = np.random.randn(61, 3)
        >>> u_nom = np.zeros((60, 2))
        >>> nominal = NominalTrajectory(x_nom, u_nom, N=60, dt=0.1)
        >>>
        >>> ax = plot_nominal_trajectory(nominal, workspace, obstacles)
    """
    if ax is None:
        _, ax = setup_figure(figsize=(12, 8))

    # Plot workspace and obstacles first
    plot_workspace(workspace, obstacles, ax=ax)

    # Extract position
    x_traj = nominal.x_nom[:, 0]
    y_traj = nominal.x_nom[:, 1]
    theta_traj = nominal.x_nom[:, 2]

    # Plot trajectory path
    ax.plot(x_traj, y_traj, color=COLORS["nominal"], linewidth=3, label="Nominal", zorder=5)

    # Add start and goal markers
    add_start_goal_markers(ax, nominal.x_nom[0], nominal.x_nom[-1], is_3d=False)

    # Show heading arrows
    if show_heading:
        for k in range(0, len(x_traj), heading_interval):
            x, y, theta = x_traj[k], y_traj[k], theta_traj[k]
            dx = arrow_scale * np.cos(theta)
            dy = arrow_scale * np.sin(theta)

            arrow = FancyArrowPatch(
                (x, y),
                (x + dx, y + dy),
                arrowstyle="->",
                mutation_scale=20,
                color=COLORS["nominal"],
                linewidth=2,
                alpha=0.6,
                zorder=6,
            )
            ax.add_patch(arrow)

    ax.set_title("Nominal Trajectory", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    return ax


# ==============================================================================
# DATA COLLECTION TRAJECTORIES
# ==============================================================================


def plot_collected_trajectories(
    trajectories: List,
    nominal: NominalTrajectory,
    workspace: Workspace2D,
    obstacles: List[Obstacle],
    ax: Optional[plt.Axes] = None,
    max_trajectories: Optional[int] = None,
    show_nominal: bool = True,
) -> plt.Axes:
    """Plot collected trajectories with nominal reference.

    Args:
        trajectories: List of Trajectory objects from data collection
        nominal: Nominal trajectory
        workspace: 2D workspace
        obstacles: List of obstacles
        ax: Matplotlib axes (creates new if None)
        max_trajectories: Maximum number of trajectories to plot (for clarity)
        show_nominal: Whether to show nominal trajectory

    Returns:
        ax: Matplotlib axes with trajectories

    Examples:
        >>> trajectories = collector.collect_trials()
        >>> ax = plot_collected_trajectories(
        ...     trajectories, nominal, workspace, obstacles
        ... )
    """
    if ax is None:
        _, ax = setup_figure(figsize=(12, 8))

    # Plot workspace and obstacles
    plot_workspace(workspace, obstacles, ax=ax)

    # Limit number of trajectories if specified
    traj_to_plot = trajectories[:max_trajectories] if max_trajectories else trajectories

    # Plot collected trajectories
    for i, traj in enumerate(traj_to_plot):
        x_traj = traj.x[:, 0]
        y_traj = traj.x[:, 1]

        ax.plot(
            x_traj,
            y_traj,
            color=COLORS["actual"],
            alpha=0.3,
            linewidth=1.5,
            label="Collected" if i == 0 else None,
            zorder=3,
        )

    # Plot nominal trajectory
    if show_nominal:
        x_nom = nominal.x_nom[:, 0]
        y_nom = nominal.x_nom[:, 1]
        ax.plot(x_nom, y_nom, color=COLORS["nominal"], linewidth=3, label="Nominal", zorder=5, linestyle="--")

    # Add start and goal
    add_start_goal_markers(ax, nominal.x_nom[0], nominal.x_nom[-1], is_3d=False)

    ax.set_title(f"Collected Trajectories (M={len(trajectories)})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    return ax


# ==============================================================================
# FUNNEL VISUALIZATION
# ==============================================================================


def plot_funnel_cross_sections(
    library: FunnelLibrary,
    nominal: NominalTrajectory,
    workspace: Workspace2D,
    obstacles: List[Obstacle],
    ax: Optional[plt.Axes] = None,
    show_centers: bool = True,
    n_std: float = 1.0,
    max_funnels: Optional[int] = None,
) -> plt.Axes:
    """Plot funnel cross-sections as ellipses along nominal trajectory.

    Args:
        library: Funnel library with segments
        nominal: Nominal trajectory
        workspace: 2D workspace
        obstacles: List of obstacles
        ax: Matplotlib axes (creates new if None)
        show_centers: Whether to show segment centers
        n_std: Number of standard deviations for ellipse size
        max_funnels: Maximum number of funnels to plot

    Returns:
        ax: Matplotlib axes with funnels

    Examples:
        >>> ax = plot_funnel_cross_sections(
        ...     library, nominal, workspace, obstacles
        ... )
    """
    if ax is None:
        _, ax = setup_figure(figsize=(12, 8))

    # Plot workspace and obstacles
    plot_workspace(workspace, obstacles, ax=ax)

    # Plot nominal trajectory
    x_nom = nominal.x_nom[:, 0]
    y_nom = nominal.x_nom[:, 1]
    ax.plot(x_nom, y_nom, color=COLORS["nominal"], linewidth=3, label="Nominal", zorder=5, linestyle="--", alpha=0.7)

    # Limit number of funnels if specified
    segments_to_plot = library.segments[:max_funnels] if max_funnels else library.segments

    # Plot funnel ellipses
    for i, segment in enumerate(segments_to_plot):
        # Get ellipse parameters (position only, first 2 dimensions)
        P = segment.P[:2, :2]
        c = segment.c[:2]

        # Color gradient based on segment index
        alpha = 0.2 + 0.3 * (i / len(segments_to_plot))

        plot_ellipsoid_2d(
            P,
            c,
            ax,
            n_std=n_std,
            color=COLORS["funnel"],
            alpha=alpha,
            edgecolor=COLORS["funnel"],
            linewidth=1.5,
            label="Funnel" if i == 0 else None,
        )

        # Show segment center
        if show_centers:
            ax.plot(c[0], c[1], "o", color=COLORS["funnel"], markersize=6, zorder=7)

    # Add start and goal
    add_start_goal_markers(ax, nominal.x_nom[0], nominal.x_nom[-1], is_3d=False)

    ax.set_title(f"Funnel Cross-Sections (n={library.num_segments})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    return ax


# ==============================================================================
# TRACKING RESULTS
# ==============================================================================


def plot_tracking_results(
    x_actual: np.ndarray,
    x_nominal: np.ndarray,
    u_actual: np.ndarray,
    u_nominal: np.ndarray,
    library: Optional[FunnelLibrary] = None,
    workspace: Optional[Workspace2D] = None,
    obstacles: Optional[List[Obstacle]] = None,
    time: Optional[np.ndarray] = None,
    fig: Optional[plt.Figure] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot comprehensive tracking results (4-panel figure).

    Creates a 2x2 figure with:
        - Top-left: 2D trajectory (x vs y)
        - Top-right: Heading vs time
        - Bottom-left: Controls (v, ω) vs time
        - Bottom-right: Tracking error vs time

    Args:
        x_actual: Actual state trajectory (N, 3)
        x_nominal: Nominal state trajectory (N, 3)
        u_actual: Actual control trajectory (N, 2)
        u_nominal: Nominal control trajectory (N, 2)
        library: Optional funnel library (shows funnels if provided)
        workspace: Optional workspace (shows bounds if provided)
        obstacles: Optional obstacles (shows obstacles if provided)
        time: Optional time vector
        fig: Optional figure (creates new if None)

    Returns:
        fig: Matplotlib figure
        axes: List of 4 subplot axes

    Examples:
        >>> fig, axes = plot_tracking_results(
        ...     x_actual, x_nominal, u_actual, u_nominal,
        ...     library=library, workspace=workspace, obstacles=obstacles
        ... )
    """
    if fig is None:
        fig = plt.figure(figsize=(16, 12))

    if time is None:
        time = np.arange(len(x_actual))

    axes = []

    # --- Panel 1: 2D Trajectory ---
    ax1 = fig.add_subplot(2, 2, 1)

    # Plot workspace and obstacles if provided
    if workspace and obstacles:
        plot_workspace(workspace, obstacles, ax=ax1, show_grid=False)

    # Plot funnels if provided
    if library:
        for segment in library.segments:
            P = segment.P[:2, :2]
            c = segment.c[:2]
            plot_ellipsoid_2d(P, c, ax1, n_std=1.0, color=COLORS["funnel"], alpha=0.2)

    # Plot trajectories
    ax1.plot(
        x_nominal[:, 0],
        x_nominal[:, 1],
        color=COLORS["nominal"],
        linewidth=3,
        label="Nominal",
        linestyle="--",
        alpha=0.7,
    )
    ax1.plot(x_actual[:, 0], x_actual[:, 1], color=COLORS["actual"], linewidth=2.5, label="Actual")

    add_start_goal_markers(ax1, x_actual[0], x_actual[-1], is_3d=False)

    ax1.set_xlabel("x (m)", fontsize=11)
    ax1.set_ylabel("y (m)", fontsize=11)
    ax1.set_title("Trajectory (x-y)", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    set_equal_aspect(ax1)
    axes.append(ax1)

    # --- Panel 2: Heading vs Time ---
    ax2 = fig.add_subplot(2, 2, 2)

    ax2.plot(time, x_nominal[:, 2], color=COLORS["nominal"], linewidth=2.5, label="Nominal", linestyle="--", alpha=0.7)
    ax2.plot(time, x_actual[:, 2], color=COLORS["actual"], linewidth=2, label="Actual")

    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Heading θ (rad)", fontsize=11)
    ax2.set_title("Heading vs Time", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    axes.append(ax2)

    # --- Panel 3: Controls vs Time ---
    ax3 = fig.add_subplot(2, 2, 3)

    # Linear velocity
    ax3.plot(
        time[:-1],
        u_nominal[:, 0],
        color=COLORS["nominal"],
        linewidth=2.5,
        label="v (nominal)",
        linestyle="--",
        alpha=0.7,
    )
    ax3.plot(time[:-1], u_actual[:, 0], color=COLORS["actual"], linewidth=2, label="v (actual)")

    # Angular velocity
    ax3.plot(
        time[:-1],
        u_nominal[:, 1],
        color=COLORS["nominal"],
        linewidth=2.5,
        label="ω (nominal)",
        linestyle=":",
        alpha=0.7,
    )
    ax3.plot(time[:-1], u_actual[:, 1], color=COLORS["actual"], linewidth=2, label="ω (actual)", linestyle=":")

    ax3.set_xlabel("Time (s)", fontsize=11)
    ax3.set_ylabel("Control Input", fontsize=11)
    ax3.set_title("Controls vs Time", fontsize=12, fontweight="bold")
    ax3.legend(ncol=2)
    ax3.grid(True, alpha=0.3)
    axes.append(ax3)

    # --- Panel 4: Tracking Error ---
    ax4 = fig.add_subplot(2, 2, 4)
    plot_tracking_error(x_nominal, x_actual, time, ax=ax4)
    axes.append(ax4)

    fig.suptitle("Tracking Results", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    return fig, axes


# ==============================================================================
# FUNNEL CONTAINMENT
# ==============================================================================


def plot_funnel_containment(
    x_actual: np.ndarray,
    library: FunnelLibrary,
    segment_indices: List[int],
    workspace: Workspace2D,
    obstacles: List[Obstacle],
    nominal: Optional[NominalTrajectory] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot trajectory colored by funnel containment status.

    Points are colored:
        - Green: Inside funnel (safe)
        - Red: Outside funnel (unsafe)

    Args:
        x_actual: Actual state trajectory (N, 3)
        library: Funnel library
        segment_indices: Segment index for each timestep (N,)
        workspace: 2D workspace
        obstacles: List of obstacles
        nominal: Optional nominal trajectory
        ax: Matplotlib axes (creates new if None)

    Returns:
        ax: Matplotlib axes with containment visualization

    Examples:
        >>> ax = plot_funnel_containment(
        ...     x_actual, library, segment_indices,
        ...     workspace, obstacles
        ... )
    """
    if ax is None:
        _, ax = setup_figure(figsize=(12, 8))

    # Plot workspace and obstacles
    plot_workspace(workspace, obstacles, ax=ax)

    # Plot funnels
    for segment in library.segments:
        P = segment.P[:2, :2]
        c = segment.c[:2]
        plot_ellipsoid_2d(P, c, ax, n_std=1.0, color=COLORS["funnel"], alpha=0.15)

    # Plot nominal if provided
    if nominal:
        ax.plot(
            nominal.x_nom[:, 0],
            nominal.x_nom[:, 1],
            color=COLORS["nominal"],
            linewidth=2.5,
            label="Nominal",
            linestyle="--",
            alpha=0.5,
            zorder=3,
        )

    # Check containment for each point
    colors = []
    for k in range(len(x_actual)):
        seg_idx = segment_indices[k]
        segment = library.get_segment(seg_idx)

        if segment.contains(x_actual[k]):
            colors.append(COLORS["safe"])
        else:
            colors.append(COLORS["unsafe"])

    # Plot trajectory with color coding
    x_traj = x_actual[:, 0]
    y_traj = x_actual[:, 1]

    for k in range(len(x_actual) - 1):
        ax.plot(
            [x_traj[k], x_traj[k + 1]], [y_traj[k], y_traj[k + 1]], color=colors[k], linewidth=2.5, alpha=0.8, zorder=4
        )

    # Add legend
    from matplotlib.lines import Line2D  # noqa: PLC0415

    legend_elements = [
        Line2D([0], [0], color=COLORS["safe"], linewidth=3, label="Inside Funnel"),
        Line2D([0], [0], color=COLORS["unsafe"], linewidth=3, label="Outside Funnel"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Compute and display statistics
    inside_count = sum(1 for c in colors if c == COLORS["safe"])
    total_count = len(colors)
    inside_pct = 100 * inside_count / total_count

    ax.text(
        0.02,
        0.98,
        f"Inside: {inside_count}/{total_count} ({inside_pct:.1f}%)",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title("Funnel Containment", fontsize=14, fontweight="bold")

    return ax


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================


def create_unicycle_report(
    nominal: NominalTrajectory,
    trajectories: List,
    x_actual: np.ndarray,
    u_actual: np.ndarray,
    library: FunnelLibrary,
    workspace: Workspace2D,
    obstacles: List[Obstacle],
    segment_indices: List[int],
    output_dir: str = "results/unicycle",
) -> None:
    """Create complete unicycle visualization report.

    Generates and saves multiple figures:
        1. nominal_trajectory.png
        2. collected_trajectories.png
        3. funnel_cross_sections.png
        4. tracking_results.png
        5. funnel_containment.png

    Args:
        nominal: Nominal trajectory
        trajectories: Collected trajectories
        x_actual: Actual tracking trajectory
        u_actual: Actual controls
        library: Funnel library
        workspace: 2D workspace
        obstacles: List of obstacles
        segment_indices: Segment indices for tracking
        output_dir: Output directory for figures

    Examples:
        >>> create_unicycle_report(
        ...     nominal, trajectories, x_actual, u_actual,
        ...     library, workspace, obstacles, segment_indices
        ... )
    """
    from pathlib import Path  # noqa: PLC0415

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating unicycle visualization report...")

    # 1. Nominal trajectory
    fig1, ax1 = setup_figure(figsize=(12, 8))
    plot_nominal_trajectory(nominal, workspace, obstacles, ax=ax1)
    save_figure(fig1, output_path / "nominal_trajectory.png")
    plt.close(fig1)

    # 2. Collected trajectories
    fig2, ax2 = setup_figure(figsize=(12, 8))
    plot_collected_trajectories(trajectories, nominal, workspace, obstacles, ax=ax2)
    save_figure(fig2, output_path / "collected_trajectories.png")
    plt.close(fig2)

    # 3. Funnel cross-sections
    fig3, ax3 = setup_figure(figsize=(12, 8))
    plot_funnel_cross_sections(library, nominal, workspace, obstacles, ax=ax3)
    save_figure(fig3, output_path / "funnel_cross_sections.png")
    plt.close(fig3)

    # 4. Tracking results
    time = nominal.get_time_vector()
    fig4, _ = plot_tracking_results(
        x_actual, nominal.x_nom, u_actual, nominal.u_nom, library, workspace, obstacles, time
    )
    save_figure(fig4, output_path / "tracking_results.png")
    plt.close(fig4)

    # 5. Funnel containment
    fig5, ax5 = setup_figure(figsize=(12, 8))
    plot_funnel_containment(x_actual, library, segment_indices, workspace, obstacles, nominal, ax=ax5)
    save_figure(fig5, output_path / "funnel_containment.png")
    plt.close(fig5)

    print(f"✓ Report saved to {output_path}/")
