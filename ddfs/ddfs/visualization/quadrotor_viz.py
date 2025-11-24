"""Quadrotor-specific visualization utilities.

This module provides 3D visualization functions for the quadrotor system,
including 3D workspace plots, attitude representation, trajectory tracking,
and funnel tubes.

The quadrotor has state x = [p, v, q, ω] (13D) and operates in 3D space.
Position is in NED (North-East-Down) frame where z points downward.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from ddfs.core.obstacles import Obstacle
from ddfs.core.workspace import Workspace3D
from ddfs.planning import NominalTrajectory
from ddfs.synthesis import FunnelLibrary
from ddfs.visualization.plotters import (
    COLORS,
    add_start_goal_markers,
    plot_ellipsoid_3d,
    plot_tracking_error,
    save_figure,
    setup_figure_3d,
)

# ==============================================================================
# WORKSPACE AND OBSTACLES
# ==============================================================================


def plot_workspace_3d(
    workspace: Workspace3D,
    obstacles: List[Obstacle],
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """Plot 3D workspace with spherical obstacles.

    Args:
        workspace: 3D workspace (cuboid)
        obstacles: List of spherical obstacles
        ax: Matplotlib 3D axes (creates new if None)

    Returns:
        ax: Matplotlib 3D axes with workspace plotted

    Examples:
        >>> from ddfs.core import Workspace3D
        >>> from ddfs.core.obstacles import SphereObstacle
        >>>
        >>> workspace = Workspace3D(
        ...     x_min=0, x_max=8, y_min=0, y_max=8, z_min=-5, z_max=0.5
        ... )
        >>> obstacles = [SphereObstacle([2, 2, -1.5], 0.5)]
        >>> ax = plot_workspace_3d(workspace, obstacles)
    """
    if ax is None:
        _, ax = setup_figure_3d(figsize=(12, 10))

    # Get workspace bounds
    x_min, x_max, y_min, y_max, z_min, z_max = workspace.bounds

    # Draw workspace edges
    # Bottom rectangle
    ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], color=COLORS["workspace"], linewidth=2, alpha=0.7)
    ax.plot([x_max, x_max], [y_min, y_max], [z_min, z_min], color=COLORS["workspace"], linewidth=2, alpha=0.7)
    ax.plot([x_max, x_min], [y_max, y_max], [z_min, z_min], color=COLORS["workspace"], linewidth=2, alpha=0.7)
    ax.plot([x_min, x_min], [y_max, y_min], [z_min, z_min], color=COLORS["workspace"], linewidth=2, alpha=0.7)

    # Top rectangle
    ax.plot([x_min, x_max], [y_min, y_min], [z_max, z_max], color=COLORS["workspace"], linewidth=2, alpha=0.7)
    ax.plot([x_max, x_max], [y_min, y_max], [z_max, z_max], color=COLORS["workspace"], linewidth=2, alpha=0.7)
    ax.plot([x_max, x_min], [y_max, y_max], [z_max, z_max], color=COLORS["workspace"], linewidth=2, alpha=0.7)
    ax.plot([x_min, x_min], [y_max, y_min], [z_max, z_max], color=COLORS["workspace"], linewidth=2, alpha=0.7)

    # Vertical edges
    ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], color=COLORS["workspace"], linewidth=2, alpha=0.7)
    ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], color=COLORS["workspace"], linewidth=2, alpha=0.7)
    ax.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], color=COLORS["workspace"], linewidth=2, alpha=0.7)
    ax.plot([x_min, x_min], [y_max, y_max], [z_min, z_max], color=COLORS["workspace"], linewidth=2, alpha=0.7)

    # Plot obstacles as spheres
    for i, obs in enumerate(obstacles):
        # Use plotters.py function for sphere
        P = obs.effective_radius**2 * np.eye(3)
        plot_ellipsoid_3d(
            P,
            obs.center[:3],
            ax,
            n_std=1.0,
            color=COLORS["obstacle"],
            alpha=0.4,
            resolution=15,
            label="Obstacle" if i == 0 else None,
        )

        # Add label
        ax.text(
            obs.center[0], obs.center[1], obs.center[2], obs.id, ha="center", va="center", fontsize=9, fontweight="bold"
        )

    # Set labels and limits
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_zlabel("z (m)", fontsize=11)

    # Set limits with padding
    padding = 0.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(z_min - padding, z_max + padding)

    # Equal aspect ratio
    ax.set_box_aspect(
        [
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        ]
    )

    ax.legend(loc="upper right")

    return ax


# ==============================================================================
# NOMINAL TRAJECTORY
# ==============================================================================


def plot_nominal_trajectory_3d(
    nominal: NominalTrajectory,
    workspace: Workspace3D,
    obstacles: List[Obstacle],
    ax: Optional[Axes3D] = None,
    show_velocity: bool = False,
    velocity_scale: float = 0.5,
    velocity_interval: int = 10,
) -> Axes3D:
    """Plot nominal 3D trajectory with workspace and obstacles.

    Args:
        nominal: Nominal trajectory
        workspace: 3D workspace
        obstacles: List of obstacles
        ax: Matplotlib 3D axes (creates new if None)
        show_velocity: Whether to show velocity arrows
        velocity_scale: Scale factor for velocity arrows
        velocity_interval: Show velocity every N timesteps

    Returns:
        ax: Matplotlib 3D axes with nominal trajectory

    Examples:
        >>> ax = plot_nominal_trajectory_3d(
        ...     nominal, workspace, obstacles
        ... )
    """
    if ax is None:
        _, ax = setup_figure_3d(figsize=(12, 10))

    # Plot workspace and obstacles first
    plot_workspace_3d(workspace, obstacles, ax=ax)

    # Extract position (first 3 components)
    p_traj = nominal.x_nom[:, :3]
    v_traj = nominal.x_nom[:, 3:6] if show_velocity else None

    # Plot trajectory path
    ax.plot(p_traj[:, 0], p_traj[:, 1], p_traj[:, 2], color=COLORS["nominal"], linewidth=3, label="Nominal", zorder=5)

    # Add start and goal markers
    add_start_goal_markers(ax, nominal.x_nom[0], nominal.x_nom[-1], is_3d=True)

    # Show velocity arrows
    if show_velocity and v_traj is not None:
        for k in range(0, len(p_traj), velocity_interval):
            p = p_traj[k]
            v = v_traj[k]

            # Draw velocity arrow
            ax.quiver(
                p[0],
                p[1],
                p[2],
                v[0],
                v[1],
                v[2],
                length=velocity_scale,
                color=COLORS["nominal"],
                alpha=0.6,
                arrow_length_ratio=0.3,
                linewidth=2,
            )

    ax.set_title("Nominal Trajectory (3D)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    return ax


# ==============================================================================
# DATA COLLECTION TRAJECTORIES
# ==============================================================================


def plot_collected_trajectories_3d(
    trajectories: List,
    nominal: NominalTrajectory,
    workspace: Workspace3D,
    obstacles: List[Obstacle],
    ax: Optional[Axes3D] = None,
    max_trajectories: Optional[int] = None,
    show_nominal: bool = True,
) -> Axes3D:
    """Plot collected 3D trajectories with nominal reference.

    Args:
        trajectories: List of Trajectory objects from data collection
        nominal: Nominal trajectory
        workspace: 3D workspace
        obstacles: List of obstacles
        ax: Matplotlib 3D axes (creates new if None)
        max_trajectories: Maximum number of trajectories to plot
        show_nominal: Whether to show nominal trajectory

    Returns:
        ax: Matplotlib 3D axes with trajectories

    Examples:
        >>> ax = plot_collected_trajectories_3d(
        ...     trajectories, nominal, workspace, obstacles
        ... )
    """
    if ax is None:
        _, ax = setup_figure_3d(figsize=(12, 10))

    # Plot workspace and obstacles
    plot_workspace_3d(workspace, obstacles, ax=ax)

    # Limit number of trajectories if specified
    traj_to_plot = trajectories[:max_trajectories] if max_trajectories else trajectories

    # Plot collected trajectories
    for i, traj in enumerate(traj_to_plot):
        p_traj = traj.x[:, :3]

        ax.plot(
            p_traj[:, 0],
            p_traj[:, 1],
            p_traj[:, 2],
            color=COLORS["actual"],
            alpha=0.3,
            linewidth=1.5,
            label="Collected" if i == 0 else None,
            zorder=3,
        )

    # Plot nominal trajectory
    if show_nominal:
        p_nom = nominal.x_nom[:, :3]
        ax.plot(
            p_nom[:, 0],
            p_nom[:, 1],
            p_nom[:, 2],
            color=COLORS["nominal"],
            linewidth=3,
            label="Nominal",
            zorder=5,
            linestyle="--",
        )

    # Add start and goal
    add_start_goal_markers(ax, nominal.x_nom[0], nominal.x_nom[-1], is_3d=True)

    ax.set_title(f"Collected Trajectories (M={len(trajectories)})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    return ax


# ==============================================================================
# FUNNEL VISUALIZATION
# ==============================================================================


def plot_funnel_tubes_3d(
    library: FunnelLibrary,
    nominal: NominalTrajectory,
    workspace: Workspace3D,
    obstacles: List[Obstacle],
    ax: Optional[Axes3D] = None,
    show_centers: bool = True,
    n_std: float = 1.0,
    max_funnels: Optional[int] = None,
    position_only: bool = True,
) -> Axes3D:
    """Plot 3D funnel tubes as ellipsoids along nominal trajectory.

    Args:
        library: Funnel library with segments
        nominal: Nominal trajectory
        workspace: 3D workspace
        obstacles: List of obstacles
        ax: Matplotlib 3D axes (creates new if None)
        show_centers: Whether to show segment centers
        n_std: Number of standard deviations for ellipsoid size
        max_funnels: Maximum number of funnels to plot
        position_only: If True, only plot position ellipsoids (3D)

    Returns:
        ax: Matplotlib 3D axes with funnels

    Examples:
        >>> ax = plot_funnel_tubes_3d(
        ...     library, nominal, workspace, obstacles
        ... )
    """
    if ax is None:
        _, ax = setup_figure_3d(figsize=(12, 10))

    # Plot workspace and obstacles
    plot_workspace_3d(workspace, obstacles, ax=ax)

    # Plot nominal trajectory
    p_nom = nominal.x_nom[:, :3]
    ax.plot(
        p_nom[:, 0],
        p_nom[:, 1],
        p_nom[:, 2],
        color=COLORS["nominal"],
        linewidth=3,
        label="Nominal",
        zorder=5,
        linestyle="--",
        alpha=0.7,
    )

    # Limit number of funnels if specified
    segments_to_plot = library.segments[:max_funnels] if max_funnels else library.segments

    # Plot funnel ellipsoids
    for i, segment in enumerate(segments_to_plot):
        # Get ellipsoid parameters (position only)
        if position_only:
            P = segment.P[:3, :3]
            c = segment.c[:3]
        else:
            # For full state visualization, project to 3D
            P = segment.P[:3, :3]
            c = segment.c[:3]

        # Color gradient based on segment index
        alpha = 0.15 + 0.15 * (i / len(segments_to_plot))

        plot_ellipsoid_3d(
            P,
            c,
            ax,
            n_std=n_std,
            color=COLORS["funnel"],
            alpha=alpha,
            resolution=15,
            label="Funnel" if i == 0 else None,
        )

        # Show segment center
        if show_centers:
            ax.scatter(c[0], c[1], c[2], color=COLORS["funnel"], s=50, zorder=7)

    # Add start and goal
    add_start_goal_markers(ax, nominal.x_nom[0], nominal.x_nom[-1], is_3d=True)

    ax.set_title(f"Funnel Tubes (n={library.num_segments})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    return ax


# ==============================================================================
# ATTITUDE VISUALIZATION
# ==============================================================================


def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        roll: Roll angle (rad)
        pitch: Pitch angle (rad)
        yaw: Yaw angle (rad)
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

    # Pitch (y-axis rotation)
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))

    # Yaw (z-axis rotation)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    return roll, pitch, yaw


def plot_attitude_tracking(
    x_actual: np.ndarray,
    x_nominal: np.ndarray,
    time: Optional[np.ndarray] = None,
    fig: Optional[plt.Figure] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot attitude tracking (roll, pitch, yaw).

    Args:
        x_actual: Actual state trajectory (N, 13)
        x_nominal: Nominal state trajectory (N, 13)
        time: Optional time vector
        fig: Optional figure (creates new if None)

    Returns:
        fig: Matplotlib figure
        axes: List of 3 subplot axes

    Examples:
        >>> fig, axes = plot_attitude_tracking(x_actual, x_nominal)
    """
    if fig is None:
        fig = plt.figure(figsize=(14, 10))

    if time is None:
        time = np.arange(len(x_actual))

    # Extract quaternions and convert to Euler angles
    N = len(x_actual)

    roll_nom = np.zeros(N)
    pitch_nom = np.zeros(N)
    yaw_nom = np.zeros(N)

    roll_act = np.zeros(N)
    pitch_act = np.zeros(N)
    yaw_act = np.zeros(N)

    for i in range(N):
        q_nom = x_nominal[i, 6:10]
        q_act = x_actual[i, 6:10]

        roll_nom[i], pitch_nom[i], yaw_nom[i] = quaternion_to_euler(q_nom)
        roll_act[i], pitch_act[i], yaw_act[i] = quaternion_to_euler(q_act)

    axes = []

    # Roll
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(time, roll_nom, color=COLORS["nominal"], linewidth=2.5, label="Nominal", linestyle="--", alpha=0.7)
    ax1.plot(time, roll_act, color=COLORS["actual"], linewidth=2, label="Actual")
    ax1.set_ylabel("Roll (rad)", fontsize=11)
    ax1.set_title("Attitude Tracking", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    axes.append(ax1)

    # Pitch
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(time, pitch_nom, color=COLORS["nominal"], linewidth=2.5, label="Nominal", linestyle="--", alpha=0.7)
    ax2.plot(time, pitch_act, color=COLORS["actual"], linewidth=2, label="Actual")
    ax2.set_ylabel("Pitch (rad)", fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    axes.append(ax2)

    # Yaw
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(time, yaw_nom, color=COLORS["nominal"], linewidth=2.5, label="Nominal", linestyle="--", alpha=0.7)
    ax3.plot(time, yaw_act, color=COLORS["actual"], linewidth=2, label="Actual")
    ax3.set_xlabel("Time (s)", fontsize=11)
    ax3.set_ylabel("Yaw (rad)", fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    axes.append(ax3)

    fig.tight_layout()

    return fig, axes


# ==============================================================================
# TRACKING RESULTS
# ==============================================================================


def plot_tracking_results_3d(  # noqa: C901, PLR0915
    x_actual: np.ndarray,
    x_nominal: np.ndarray,
    u_actual: np.ndarray,
    u_nominal: np.ndarray,
    library: Optional[FunnelLibrary] = None,
    workspace: Optional[Workspace3D] = None,
    obstacles: Optional[List[Obstacle]] = None,
    time: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot comprehensive 3D tracking results (multi-panel figure).

    Creates a large figure with 6 panels:
        1. 3D trajectory
        2. Position vs time (x, y, z)
        3. Velocity vs time (vx, vy, vz)
        4. Controls vs time (T, τx, τy, τz)
        5. Attitude vs time (roll, pitch, yaw)
        6. Tracking error vs time

    Args:
        x_actual: Actual state trajectory (N, 13)
        x_nominal: Nominal state trajectory (N, 13)
        u_actual: Actual control trajectory (N, 4)
        u_nominal: Nominal control trajectory (N, 4)
        library: Optional funnel library
        workspace: Optional workspace
        obstacles: Optional obstacles
        time: Optional time vector

    Returns:
        fig: Matplotlib figure
        axes: List of subplot axes

    Examples:
        >>> fig, axes = plot_tracking_results_3d(
        ...     x_actual, x_nominal, u_actual, u_nominal
        ... )
    """
    fig = plt.figure(figsize=(18, 14))

    if time is None:
        time = np.arange(len(x_actual))

    axes = []

    # --- Panel 1: 3D Trajectory ---
    ax1 = fig.add_subplot(3, 3, 1, projection="3d")

    # Plot workspace and obstacles if provided
    if workspace and obstacles:
        plot_workspace_3d(workspace, obstacles, ax=ax1)

    # Plot funnels if provided
    if library:
        for segment in library.segments[:10]:  # Limit to first 10 for clarity
            P = segment.P[:3, :3]
            c = segment.c[:3]
            plot_ellipsoid_3d(P, c, ax1, n_std=1.0, color=COLORS["funnel"], alpha=0.1, resolution=10)

    # Plot trajectories
    p_nom = x_nominal[:, :3]
    p_act = x_actual[:, :3]

    ax1.plot(
        p_nom[:, 0],
        p_nom[:, 1],
        p_nom[:, 2],
        color=COLORS["nominal"],
        linewidth=3,
        label="Nominal",
        linestyle="--",
        alpha=0.7,
    )
    ax1.plot(p_act[:, 0], p_act[:, 1], p_act[:, 2], color=COLORS["actual"], linewidth=2.5, label="Actual")

    add_start_goal_markers(ax1, x_actual[0], x_actual[-1], is_3d=True)

    ax1.set_xlabel("x (m)", fontsize=10)
    ax1.set_ylabel("y (m)", fontsize=10)
    ax1.set_zlabel("z (m)", fontsize=10)
    ax1.set_title("3D Trajectory", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    axes.append(ax1)

    # --- Panel 2: Position vs Time ---
    ax2 = fig.add_subplot(3, 3, 2)

    for i, label in enumerate(["x", "y", "z"]):
        ax2.plot(time, x_nominal[:, i], linestyle="--", alpha=0.7, linewidth=2, label=f"{label} (nom)")
        ax2.plot(time, x_actual[:, i], linewidth=1.5, label=f"{label} (act)")

    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Position (m)", fontsize=10)
    ax2.set_title("Position vs Time", fontsize=11, fontweight="bold")
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    axes.append(ax2)

    # --- Panel 3: Velocity vs Time ---
    ax3 = fig.add_subplot(3, 3, 3)

    for i, label in enumerate(["vx", "vy", "vz"]):
        ax3.plot(time, x_nominal[:, 3 + i], linestyle="--", alpha=0.7, linewidth=2, label=f"{label} (nom)")
        ax3.plot(time, x_actual[:, 3 + i], linewidth=1.5, label=f"{label} (act)")

    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_ylabel("Velocity (m/s)", fontsize=10)
    ax3.set_title("Velocity vs Time", fontsize=11, fontweight="bold")
    ax3.legend(ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)
    axes.append(ax3)

    # --- Panel 4: Controls vs Time ---
    ax4 = fig.add_subplot(3, 3, 4)

    # Thrust
    ax4.plot(
        time[:-1], u_nominal[:, 0], color=COLORS["nominal"], linestyle="--", linewidth=2, alpha=0.7, label="T (nom)"
    )
    ax4.plot(time[:-1], u_actual[:, 0], color=COLORS["actual"], linewidth=1.5, label="T (act)")

    ax4.set_xlabel("Time (s)", fontsize=10)
    ax4.set_ylabel("Thrust (N)", fontsize=10)
    ax4.set_title("Thrust vs Time", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    axes.append(ax4)

    # --- Panel 5: Torques vs Time ---
    ax5 = fig.add_subplot(3, 3, 5)

    for i, label in enumerate(["τx", "τy", "τz"]):
        ax5.plot(time[:-1], u_nominal[:, 1 + i], linestyle="--", alpha=0.7, linewidth=2, label=f"{label} (nom)")
        ax5.plot(time[:-1], u_actual[:, 1 + i], linewidth=1.5, label=f"{label} (act)")

    ax5.set_xlabel("Time (s)", fontsize=10)
    ax5.set_ylabel("Torque (N·m)", fontsize=10)
    ax5.set_title("Torques vs Time", fontsize=11, fontweight="bold")
    ax5.legend(ncol=2, fontsize=8)
    ax5.grid(True, alpha=0.3)
    axes.append(ax5)

    # --- Panel 6: Attitude ---
    ax6 = fig.add_subplot(3, 3, 6)

    # Convert quaternions to Euler angles
    N = len(x_actual)
    roll_nom, pitch_nom, yaw_nom = np.zeros(N), np.zeros(N), np.zeros(N)
    roll_act, pitch_act, yaw_act = np.zeros(N), np.zeros(N), np.zeros(N)

    for i in range(N):
        roll_nom[i], pitch_nom[i], yaw_nom[i] = quaternion_to_euler(x_nominal[i, 6:10])
        roll_act[i], pitch_act[i], yaw_act[i] = quaternion_to_euler(x_actual[i, 6:10])

    for nom, act, label in [(roll_nom, roll_act, "φ"), (pitch_nom, pitch_act, "θ"), (yaw_nom, yaw_act, "ψ")]:
        ax6.plot(time, nom, linestyle="--", alpha=0.7, linewidth=2)
        ax6.plot(time, act, linewidth=1.5, label=label)

    ax6.set_xlabel("Time (s)", fontsize=10)
    ax6.set_ylabel("Angle (rad)", fontsize=10)
    ax6.set_title("Attitude vs Time", fontsize=11, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    axes.append(ax6)

    # --- Panel 7: Angular Velocity ---
    ax7 = fig.add_subplot(3, 3, 7)

    for i, label in enumerate(["ωx", "ωy", "ωz"]):
        ax7.plot(time, x_nominal[:, 10 + i], linestyle="--", alpha=0.7, linewidth=2, label=f"{label} (nom)")
        ax7.plot(time, x_actual[:, 10 + i], linewidth=1.5, label=f"{label} (act)")

    ax7.set_xlabel("Time (s)", fontsize=10)
    ax7.set_ylabel("Angular Velocity (rad/s)", fontsize=10)
    ax7.set_title("Angular Velocity vs Time", fontsize=11, fontweight="bold")
    ax7.legend(ncol=2, fontsize=8)
    ax7.grid(True, alpha=0.3)
    axes.append(ax7)

    # --- Panel 8: Position Error ---
    ax8 = fig.add_subplot(3, 3, 8)

    pos_error = np.linalg.norm(x_actual[:, :3] - x_nominal[:, :3], axis=1)
    ax8.plot(time, pos_error, color=COLORS["actual"], linewidth=2)
    ax8.fill_between(time, 0, pos_error, alpha=0.3, color=COLORS["actual"])

    mean_error = np.mean(pos_error)
    max_error = np.max(pos_error)
    ax8.axhline(mean_error, color="gray", linestyle="--", alpha=0.5)
    ax8.text(
        0.02,
        0.98,
        f"Max: {max_error:.3f}\nMean: {mean_error:.3f}",
        transform=ax8.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=9,
    )

    ax8.set_xlabel("Time (s)", fontsize=10)
    ax8.set_ylabel("Position Error (m)", fontsize=10)
    ax8.set_title("Position Tracking Error", fontsize=11, fontweight="bold")
    ax8.grid(True, alpha=0.3)
    axes.append(ax8)

    # --- Panel 9: Total State Error ---
    ax9 = fig.add_subplot(3, 3, 9)
    plot_tracking_error(x_nominal, x_actual, time, ax=ax9)
    ax9.set_title("Total State Error", fontsize=11, fontweight="bold")
    axes.append(ax9)

    fig.suptitle("Quadrotor Tracking Results", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    return fig, axes


# ==============================================================================
# FUNNEL CONTAINMENT
# ==============================================================================


def plot_funnel_containment_3d(
    x_actual: np.ndarray,
    library: FunnelLibrary,
    segment_indices: List[int],
    workspace: Workspace3D,
    obstacles: List[Obstacle],
    nominal: Optional[NominalTrajectory] = None,
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """Plot 3D trajectory colored by funnel containment status.

    Points are colored:
        - Green: Inside funnel (safe)
        - Red: Outside funnel (unsafe)

    Args:
        x_actual: Actual state trajectory (N, 13)
        library: Funnel library
        segment_indices: Segment index for each timestep (N,)
        workspace: 3D workspace
        obstacles: List of obstacles
        nominal: Optional nominal trajectory
        ax: Matplotlib 3D axes (creates new if None)

    Returns:
        ax: Matplotlib 3D axes with containment visualization

    Examples:
        >>> ax = plot_funnel_containment_3d(
        ...     x_actual, library, segment_indices,
        ...     workspace, obstacles
        ... )
    """
    if ax is None:
        _, ax = setup_figure_3d(figsize=(12, 10))

    # Plot workspace and obstacles
    plot_workspace_3d(workspace, obstacles, ax=ax)

    # Plot funnels
    for segment in library.segments:
        P = segment.P[:3, :3]
        c = segment.c[:3]
        plot_ellipsoid_3d(P, c, ax, n_std=1.0, color=COLORS["funnel"], alpha=0.1, resolution=12)

    # Plot nominal if provided
    if nominal:
        p_nom = nominal.x_nom[:, :3]
        ax.plot(
            p_nom[:, 0],
            p_nom[:, 1],
            p_nom[:, 2],
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
    p_traj = x_actual[:, :3]

    for k in range(len(x_actual) - 1):
        ax.plot(
            [p_traj[k, 0], p_traj[k + 1, 0]],
            [p_traj[k, 1], p_traj[k + 1, 1]],
            [p_traj[k, 2], p_traj[k + 1, 2]],
            color=colors[k],
            linewidth=2.5,
            alpha=0.8,
            zorder=4,
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

    ax.text2D(
        0.02,
        0.98,
        f"Inside: {inside_count}/{total_count} ({inside_pct:.1f}%)",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title("Funnel Containment (3D)", fontsize=14, fontweight="bold")

    return ax


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================


def create_quadrotor_report(
    nominal: NominalTrajectory,
    trajectories: List,
    x_actual: np.ndarray,
    u_actual: np.ndarray,
    library: FunnelLibrary,
    workspace: Workspace3D,
    obstacles: List[Obstacle],
    segment_indices: List[int],
    output_dir: str = "results/quadrotor",
) -> None:
    """Create complete quadrotor visualization report.

    Generates and saves multiple figures:
        1. nominal_trajectory_3d.png
        2. collected_trajectories_3d.png
        3. funnel_tubes_3d.png
        4. tracking_results_3d.png
        5. attitude_tracking.png
        6. funnel_containment_3d.png

    Args:
        nominal: Nominal trajectory
        trajectories: Collected trajectories
        x_actual: Actual tracking trajectory
        u_actual: Actual controls
        library: Funnel library
        workspace: 3D workspace
        obstacles: List of obstacles
        segment_indices: Segment indices for tracking
        output_dir: Output directory for figures

    Examples:
        >>> create_quadrotor_report(
        ...     nominal, trajectories, x_actual, u_actual,
        ...     library, workspace, obstacles, segment_indices
        ... )
    """
    from pathlib import Path  # noqa: PLC0415

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating quadrotor visualization report...")

    # 1. Nominal trajectory
    fig1, ax1 = setup_figure_3d()
    plot_nominal_trajectory_3d(nominal, workspace, obstacles, ax=ax1)
    save_figure(fig1, output_path / "nominal_trajectory_3d.png")
    plt.close(fig1)

    # 2. Collected trajectories
    fig2, ax2 = setup_figure_3d()
    plot_collected_trajectories_3d(trajectories, nominal, workspace, obstacles, ax=ax2)
    save_figure(fig2, output_path / "collected_trajectories_3d.png")
    plt.close(fig2)

    # 3. Funnel tubes
    fig3, ax3 = setup_figure_3d()
    plot_funnel_tubes_3d(library, nominal, workspace, obstacles, ax=ax3)
    save_figure(fig3, output_path / "funnel_tubes_3d.png")
    plt.close(fig3)

    # 4. Tracking results
    time = nominal.get_time_vector()
    fig4, _ = plot_tracking_results_3d(
        x_actual, nominal.x_nom, u_actual, nominal.u_nom, library, workspace, obstacles, time
    )
    save_figure(fig4, output_path / "tracking_results_3d.png")
    plt.close(fig4)

    # 5. Attitude tracking
    fig5, _ = plot_attitude_tracking(x_actual, nominal.x_nom, time)
    save_figure(fig5, output_path / "attitude_tracking.png")
    plt.close(fig5)

    # 6. Funnel containment
    fig6, ax6 = setup_figure_3d()
    plot_funnel_containment_3d(x_actual, library, segment_indices, workspace, obstacles, nominal, ax=ax6)
    save_figure(fig6, output_path / "funnel_containment_3d.png")
    plt.close(fig6)

    print(f"✓ Report saved to {output_path}/")
