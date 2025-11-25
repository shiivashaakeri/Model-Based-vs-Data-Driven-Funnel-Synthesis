"""
Nominal Trajectory Design Example for DDFS.

This script demonstrates the complete workflow for designing nominal trajectories
using the digital twin model and SCvx planner for both unicycle and quadrotor systems.

Features:
- Load system configuration
- Build digital twin model
- Set up workspace with obstacles
- Design nominal trajectory using SCvx
- Analyze trajectory (Assumption 3 verification)
- Comprehensive visualization

Usage:
    python examples/nominal_traj_design.py --system unicycle
    python examples/nominal_traj_design.py --system quadrotor
    python examples/nominal_traj_design.py --system both
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# DDFS imports
from ddfs.core.config import load_system_config
from ddfs.core.workspace import Workspace, workspace_from_config
from ddfs.models.quadrotor import QuadrotorTwin
from ddfs.models.unicycle import UnicycleTwin
from ddfs.planning.base_planner import QuadraticCost
from ddfs.planning.scvx_planner import SCvxConfig, SCvxPlanner
from ddfs.planning.trajectory import Trajectory
from ddfs.planning.trajectory_analysis import (
    analyze_trajectory,
    compute_increment_bound,
    verify_assumption_3,
)
from ddfs.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_workspace_2d(
    workspace: Workspace,
    trajectory: Optional[Trajectory] = None,
    title: str = "Workspace",
    ax: Optional[plt.Axes] = None,
    position_indices: List[int] = [0, 1],
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot 2D workspace with obstacles and trajectory.

    Parameters
    ----------
    workspace : Workspace
        Workspace containing constraints and obstacles.
    trajectory : Trajectory, optional
        Trajectory to plot.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Existing axes to plot on.
    position_indices : list
        Indices of position states [x_idx, y_idx].
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Get workspace bounds
    x_min = workspace.x_min[position_indices[0]]
    x_max = workspace.x_max[position_indices[0]]
    y_min = workspace.x_min[position_indices[1]]
    y_max = workspace.x_max[position_indices[1]]

    # Plot workspace boundary
    rect = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        edgecolor="green",
        linewidth=2,
        linestyle="--",
        label="Workspace boundary",
    )
    ax.add_patch(rect)

    # Plot obstacles
    if workspace.obstacles is not None:
        for i, obstacle in enumerate(workspace.obstacles.obstacles):
            circle = plt.Circle(
                obstacle.center[:2],
                obstacle.radius,
                color="red",
                alpha=0.5,
                label="Obstacle" if i == 0 else None,
            )
            ax.add_patch(circle)

            # Obstacle with margin
            if hasattr(obstacle, "margin") and obstacle.margin > 0:
                circle_margin = plt.Circle(
                    obstacle.center[:2],
                    obstacle.effective_radius,
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    alpha=0.5,
                )
                ax.add_patch(circle_margin)

    # Plot trajectory
    if trajectory is not None:
        x_pos = trajectory.x[:, position_indices[0]]
        y_pos = trajectory.x[:, position_indices[1]]

        # Trajectory path
        ax.plot(x_pos, y_pos, "b-", linewidth=2, label="Trajectory")

        # Start and end markers
        ax.plot(x_pos[0], y_pos[0], "go", markersize=12, label="Start", zorder=5)
        ax.plot(x_pos[-1], y_pos[-1], "r*", markersize=15, label="Goal", zorder=5)

        # Direction arrows along trajectory
        n_arrows = 10
        indices = np.linspace(0, len(x_pos) - 1, n_arrows, dtype=int)
        for idx in indices[:-1]:
            if idx + 1 < len(x_pos):
                dx = x_pos[idx + 1] - x_pos[idx]
                dy = y_pos[idx + 1] - y_pos[idx]
                if np.sqrt(dx**2 + dy**2) > 1e-6:
                    ax.annotate(
                        "",
                        xy=(x_pos[idx] + dx * 0.5, y_pos[idx] + dy * 0.5),
                        xytext=(x_pos[idx], y_pos[idx]),
                        arrowprops={"arrowstyle": "->", "color": "blue", "alpha": 0.6},
                    )

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Set axis limits with padding
    padding = 0.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    return fig


def plot_workspace_3d(
    workspace: Workspace,
    trajectory: Optional[Trajectory] = None,
    title: str = "3D Workspace",
    ax: Optional[plt.Axes] = None,
    position_indices: List[int] = [0, 1, 2],
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot 3D workspace with obstacles and trajectory.

    Parameters
    ----------
    workspace : Workspace
        Workspace containing constraints and obstacles.
    trajectory : Trajectory, optional
        Trajectory to plot.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Existing 3D axes.
    position_indices : list
        Indices of position states [x_idx, y_idx, z_idx].
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Get workspace bounds
    x_min = workspace.x_min[position_indices[0]]
    x_max = workspace.x_max[position_indices[0]]
    y_min = workspace.x_min[position_indices[1]]
    y_max = workspace.x_max[position_indices[1]]
    z_min = workspace.x_min[position_indices[2]]
    z_max = workspace.x_max[position_indices[2]]

    # Plot workspace boundary (wireframe box)
    # Bottom face
    ax.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        [z_min, z_min, z_min, z_min, z_min],
        "g--",
        alpha=0.5,
    )
    # Top face
    ax.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        [z_max, z_max, z_max, z_max, z_max],
        "g--",
        alpha=0.5,
    )
    # Vertical edges
    for x, y in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
        ax.plot([x, x], [y, y], [z_min, z_max], "g--", alpha=0.5)

    # Plot spherical obstacles
    if workspace.obstacles is not None:
        for obstacle in workspace.obstacles.obstacles:
            center = obstacle.center
            radius = obstacle.radius

            # Create sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 15)
            xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            zs = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(xs, ys, zs, color="red", alpha=0.4)

    # Plot trajectory
    if trajectory is not None:
        x_pos = trajectory.x[:, position_indices[0]]
        y_pos = trajectory.x[:, position_indices[1]]
        z_pos = trajectory.x[:, position_indices[2]]

        # Trajectory path
        ax.plot(x_pos, y_pos, z_pos, "b-", linewidth=2, label="Trajectory")

        # Start and end markers
        ax.scatter(x_pos[0], y_pos[0], z_pos[0], c="green", s=100, marker="o", label="Start")
        ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], c="red", s=150, marker="*", label="Goal")

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_zlabel("z [m]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()

    return fig


def plot_states(
    trajectory: Trajectory,
    state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    state_labels: Optional[List[str]] = None,
    title: str = "State Trajectories",
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot state trajectories with bounds.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to plot.
    state_bounds : tuple, optional
        (lower_bounds, upper_bounds) arrays.
    state_labels : list, optional
        Labels for each state.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    n_states = trajectory.n_states
    t = trajectory.t

    if state_labels is None:
        state_labels = [f"$x_{{{i}}}$" for i in range(n_states)]

    # Determine subplot layout
    n_cols = min(3, n_states)
    n_rows = (n_states + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(n_states):
        ax = axes[i]

        # Plot state trajectory
        ax.plot(t, trajectory.x[:, i], "b-", linewidth=1.5, label=state_labels[i])

        # Plot bounds
        if state_bounds is not None:
            lb, ub = state_bounds
            ax.axhline(y=lb[i], color="r", linestyle="--", linewidth=1, alpha=0.7, label="Bounds")
            ax.axhline(y=ub[i], color="r", linestyle="--", linewidth=1, alpha=0.7)

            # Fill region outside bounds
            ax.fill_between(t, lb[i], ax.get_ylim()[0], alpha=0.1, color="red")
            ax.fill_between(t, ub[i], ax.get_ylim()[1], alpha=0.1, color="red")

        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel(state_labels[i], fontsize=10)
        ax.set_title(f"State: {state_labels[i]}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Hide unused subplots
    for i in range(n_states, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig


def plot_inputs(
    trajectory: Trajectory,
    input_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    input_labels: Optional[List[str]] = None,
    title: str = "Control Inputs",
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot control input trajectories with bounds.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to plot.
    input_bounds : tuple, optional
        (lower_bounds, upper_bounds) arrays.
    input_labels : list, optional
        Labels for each input.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    n_inputs = trajectory.n_inputs
    t = trajectory.t_inputs

    if input_labels is None:
        input_labels = [f"$u_{{{i}}}$" for i in range(n_inputs)]

    # Determine subplot layout
    n_cols = min(2, n_inputs)
    n_rows = (n_inputs + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (6 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(n_inputs):
        ax = axes[i]

        # Plot input trajectory (step plot for ZOH)
        ax.step(t, trajectory.u[:, i], "b-", linewidth=1.5, where="post", label=input_labels[i])

        # Plot bounds
        if input_bounds is not None:
            lb, ub = input_bounds
            ax.axhline(y=lb[i], color="r", linestyle="--", linewidth=1, alpha=0.7, label="Bounds")
            ax.axhline(y=ub[i], color="r", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel(input_labels[i], fontsize=10)
        ax.set_title(f"Input: {input_labels[i]}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Hide unused subplots
    for i in range(n_inputs, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig


def plot_convergence(
    convergence_history: Dict[str, List[float]],
    title: str = "SCvx Convergence",
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot SCvx convergence history.

    Parameters
    ----------
    convergence_history : dict
        Dictionary with 'cost', 'constraint_violation', 'trust_region' lists.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    iterations = range(len(convergence_history.get("cost", [])))

    # Cost
    if "cost" in convergence_history and len(convergence_history["cost"]) > 0:
        axes[0].semilogy(iterations, convergence_history["cost"], "b-o", markersize=4)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Cost")
        axes[0].set_title("Cost")
        axes[0].grid(True, alpha=0.3)

    # Constraint violation
    if "constraint_violation" in convergence_history and len(convergence_history["constraint_violation"]) > 0:
        axes[1].semilogy(iterations, convergence_history["constraint_violation"], "r-o", markersize=4)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Violation")
        axes[1].set_title("Constraint Violation")
        axes[1].grid(True, alpha=0.3)

    # Trust region
    if "trust_region" in convergence_history and len(convergence_history["trust_region"]) > 0:
        axes[2].plot(iterations, convergence_history["trust_region"], "g-o", markersize=4)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Radius")
        axes[2].set_title("Trust Region")
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig


def plot_trajectory_analysis(
    trajectory: Trajectory,
    title: str = "Trajectory Analysis",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot trajectory increment and smoothness analysis.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to analyze.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    # Compute analysis
    increment_analysis = compute_increment_bound(trajectory)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # State increments
    ax = axes[0, 0]
    t = trajectory.t[:-1]
    ax.plot(t, increment_analysis.state_increment_norms, "b-", linewidth=1.5)
    ax.axhline(y=increment_analysis.v_state, color="r", linestyle="--", label=f"Max = {increment_analysis.v_state:.4f}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("||Δx||")
    ax.set_title("State Increments")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Input increments
    ax = axes[0, 1]
    if len(increment_analysis.input_increment_norms) > 0:
        t_u = trajectory.t_inputs[:-1]
        ax.plot(t_u, increment_analysis.input_increment_norms, "b-", linewidth=1.5)
        ax.axhline(
            y=increment_analysis.v_input, color="r", linestyle="--", label=f"Max = {increment_analysis.v_input:.4f}"
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("||Δu||")
    ax.set_title("Input Increments")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined increments (for Assumption 3)
    ax = axes[1, 0]
    t_comb = trajectory.t[:-2]
    ax.plot(t_comb, increment_analysis.increment_norms, "b-", linewidth=1.5)
    ax.axhline(y=increment_analysis.v, color="r", linestyle="--", label=f"v = {increment_analysis.v:.4f}")
    ax.scatter(t_comb[increment_analysis.max_increment_index], increment_analysis.v, c="r", s=50, zorder=5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("||(Δx, Δu)||")
    ax.set_title("Combined Increments (Assumption 3)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram of increments
    ax = axes[1, 1]
    ax.hist(increment_analysis.increment_norms, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(x=increment_analysis.v, color="r", linestyle="--", linewidth=2, label=f"v = {increment_analysis.v:.4f}")
    ax.set_xlabel("||(Δx, Δu)||")
    ax.set_ylabel("Frequency")
    ax.set_title("Increment Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig


# =============================================================================
# System-Specific Design Functions
# =============================================================================


def design_unicycle_trajectory(  # noqa: PLR0915
    config_name: str = "unicycle",
    N: int = 500,
    save_plots: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[Trajectory, Dict]:
    """
    Design nominal trajectory for unicycle system.

    Parameters
    ----------
    config_name : str
        Configuration name.
    N : int
        Number of timesteps.
    save_plots : bool
        Whether to save plots.
    output_dir : Path, optional
        Output directory for plots.

    Returns
    -------
    trajectory : Trajectory
        Designed trajectory.
    results : dict
        Results dictionary.
    """
    logger.info("=" * 60)
    logger.info("UNICYCLE NOMINAL TRAJECTORY DESIGN")
    logger.info("=" * 60)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_system_config(config_name)

    # Create twin model
    logger.info("Creating twin model...")
    twin = UnicycleTwin(dt=config.simulation.dt)
    logger.info(f"  Model: {twin.name}")
    logger.info(f"  States: {twin.n_states}, Inputs: {twin.n_inputs}")
    logger.info(f"  dt: {twin.dt}")

    # Create workspace
    logger.info("Setting up workspace...")
    workspace = workspace_from_config(config, safety_margin=0.1)
    logger.info(f"  State bounds: [{workspace.x_min}, {workspace.x_max}]")
    logger.info(f"  Input bounds: [{workspace.u_min}, {workspace.u_max}]")
    logger.info(f"  Obstacles: {workspace.n_obstacles}")

    # Initial and target states
    x_init = np.array(config.system.x_init)
    x_target = np.array(config.system.x_final)
    logger.info(f"  Initial state: {x_init}")
    logger.info(f"  Target state: {x_target}")

    # Configure SCvx
    logger.info("Configuring SCvx planner...")
    scvx_config = SCvxConfig(
        max_iterations=config.solver.scvx_max_iters,
        cost_tolerance=config.solver.scvx_tol,
        trust_region_init=config.solver.trust_region_init,
        trust_region_min=config.solver.trust_region_min,
        trust_region_max=config.solver.trust_region_max,
        virtual_control_weight=1e5,
        state_constraint_weight=1e4,
        use_soft_state_constraints=True,
        use_soft_obstacle_constraints=True,
    )

    # Create cost function
    # Create cost function (use default weights)
    Q = np.diag([1.0, 1.0, 0.1]) if config.system.n_states == 3 else np.diag([1.0] * config.system.n_states)
    R = np.diag([0.1, 0.1]) if config.system.n_inputs == 2 else np.diag([0.1] * config.system.n_inputs)
    Q_f = np.diag([10.0, 10.0, 1.0]) if config.system.n_states == 3 else np.diag([10.0] * config.system.n_states)
    cost_function = QuadraticCost(Q=Q, R=R, Q_f=Q_f)

    # Create planner
    planner = SCvxPlanner(twin, scvx_config)
    planner.set_cost_function(cost_function)
    planner.set_constraints(workspace.constraints, workspace.obstacles)
    planner.set_position_indices(twin.position_indices)
    planner.set_solver(config.solver.name)

    # Plan trajectory
    logger.info(f"Planning trajectory (N={N})...")
    result = planner.plan(x_init=x_init, x_target=x_target, N=N)

    logger.info(f"  Status: {result.status.value}")
    logger.info(f"  Iterations: {result.iterations}")
    logger.info(f"  Cost: {result.cost:.4f}")
    logger.info(f"  Time: {result.total_time:.2f} s")

    if not result.success:
        logger.warning("Planning failed!")
        return None, {"status": "failed"}

    trajectory = result.trajectory

    # Analyze trajectory
    logger.info("Analyzing trajectory...")
    satisfied, increment_analysis = verify_assumption_3(trajectory)
    logger.info(f"  Assumption 3 satisfied: {satisfied}")
    logger.info(f"  Increment bound v: {increment_analysis.v:.6f}")

    analysis = analyze_trajectory(trajectory, twin, twin.position_indices)
    logger.info(f"  Path length: {analysis.smoothness.path_length:.4f}")
    logger.info(f"  Max velocity: {analysis.smoothness.max_velocity:.4f}")

    # Create plots
    logger.info("Generating plots...")

    # State labels and bounds
    state_labels = ["$p_x$ [m]", "$p_y$ [m]", "$\\theta$ [rad]"]
    input_labels = ["$v$ [m/s]", "$\\omega$ [rad/s]"]
    state_bounds = (workspace.x_min, workspace.x_max)
    input_bounds = (workspace.u_min, workspace.u_max)

    figures = {}

    # Workspace plot
    fig_ws = plot_workspace_2d(
        workspace,
        trajectory,
        title="Unicycle: Workspace and Trajectory",
        position_indices=twin.position_indices,
    )
    figures["workspace"] = fig_ws

    # State plots
    fig_states = plot_states(
        trajectory,
        state_bounds,
        state_labels,
        title="Unicycle: State Trajectories",
    )
    figures["states"] = fig_states

    # Input plots
    fig_inputs = plot_inputs(
        trajectory,
        input_bounds,
        input_labels,
        title="Unicycle: Control Inputs",
    )
    figures["inputs"] = fig_inputs

    # Convergence plot
    fig_conv = plot_convergence(
        result.convergence_history,
        title="Unicycle: SCvx Convergence",
    )
    figures["convergence"] = fig_conv

    # Analysis plot
    fig_analysis = plot_trajectory_analysis(
        trajectory,
        title="Unicycle: Trajectory Analysis",
    )
    figures["analysis"] = fig_analysis

    # Save plots
    if save_plots and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, fig in figures.items():
            filepath = output_dir / f"unicycle_{name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"  Saved: {filepath}")

        # Save trajectory
        traj_path = output_dir / "unicycle_trajectory.npz"
        trajectory.save(traj_path)
        logger.info(f"  Saved: {traj_path}")

    results = {
        "status": "success",
        "trajectory": trajectory,
        "result": result,
        "analysis": analysis,
        "increment_analysis": increment_analysis,
        "figures": figures,
    }

    return trajectory, results


def design_quadrotor_trajectory(  # noqa: PLR0915
    config_name: str = "quadrotor",
    N: int = 500,
    save_plots: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[Trajectory, Dict]:
    """
    Design nominal trajectory for quadrotor system.

    Parameters
    ----------
    config_name : str
        Configuration name.
    N : int
        Number of timesteps.
    save_plots : bool
        Whether to save plots.
    output_dir : Path, optional
        Output directory for plots.

    Returns
    -------
    trajectory : Trajectory
        Designed trajectory.
    results : dict
        Results dictionary.
    """
    logger.info("=" * 60)
    logger.info("QUADROTOR NOMINAL TRAJECTORY DESIGN")
    logger.info("=" * 60)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_system_config(config_name)

    # Create twin model
    logger.info("Creating twin model...")
    twin = QuadrotorTwin(dt=config.simulation.dt)
    logger.info(f"  Model: {twin.name}")
    logger.info(f"  States: {twin.n_states}, Inputs: {twin.n_inputs}")
    logger.info(f"  dt: {twin.dt}")
    logger.info(f"  Hover thrust: {twin.hover_thrust():.4f}")

    # Create workspace
    logger.info("Setting up workspace...")
    workspace = workspace_from_config(config, safety_margin=0.1)
    logger.info(f"  Obstacles: {workspace.n_obstacles}")

    # Initial and target states
    x_init = np.array(config.system.x_init)
    x_target = np.array(config.system.x_final)
    logger.info(f"  Initial position: {x_init[:3]}")
    logger.info(f"  Target position: {x_target[:3]}")

    # Configure SCvx
    logger.info("Configuring SCvx planner...")
    scvx_config = SCvxConfig(
        max_iterations=config.solver.scvx_max_iters,
        cost_tolerance=config.solver.scvx_tol,
        trust_region_init=config.solver.trust_region_init,
        trust_region_min=config.solver.trust_region_min,
        trust_region_max=config.solver.trust_region_max,
        virtual_control_weight=1e5,
        state_constraint_weight=1e4,
        use_soft_state_constraints=True,
        use_soft_obstacle_constraints=True,
    )

    # Create cost function
    # Create cost function (use default weights)
    Q = np.diag([1.0, 1.0, 0.1]) if config.system.n_states == 3 else np.diag([1.0] * config.system.n_states)
    R = np.diag([0.1, 0.1]) if config.system.n_inputs == 2 else np.diag([0.1] * config.system.n_inputs)
    Q_f = np.diag([10.0, 10.0, 1.0]) if config.system.n_states == 3 else np.diag([10.0] * config.system.n_states)
    cost_function = QuadraticCost(Q=Q, R=R, Q_f=Q_f)

    # Create planner
    planner = SCvxPlanner(twin, scvx_config)
    planner.set_cost_function(cost_function)
    planner.set_constraints(workspace.constraints, workspace.obstacles)
    planner.set_position_indices(twin.position_indices)
    planner.set_solver(config.solver.name)

    # Generate initial guess with hover thrust
    u_guess = np.zeros((N, twin.n_inputs))
    u_guess[:, 0] = twin.hover_thrust()

    # Plan trajectory
    logger.info(f"Planning trajectory (N={N})...")
    result = planner.plan(
        x_init=x_init,
        x_target=x_target,
        N=N,
        u_init_guess=u_guess,
    )

    logger.info(f"  Status: {result.status.value}")
    logger.info(f"  Iterations: {result.iterations}")
    logger.info(f"  Cost: {result.cost:.4f}")
    logger.info(f"  Time: {result.total_time:.2f} s")

    if not result.success:
        logger.warning("Planning failed!")
        return None, {"status": "failed"}

    trajectory = result.trajectory

    # Analyze trajectory
    logger.info("Analyzing trajectory...")
    satisfied, increment_analysis = verify_assumption_3(trajectory)
    logger.info(f"  Assumption 3 satisfied: {satisfied}")
    logger.info(f"  Increment bound v: {increment_analysis.v:.6f}")

    analysis = analyze_trajectory(trajectory, twin, twin.position_indices)
    logger.info(f"  Path length: {analysis.smoothness.path_length:.4f}")
    logger.info(f"  Max velocity: {analysis.smoothness.max_velocity:.4f}")

    # Create plots
    logger.info("Generating plots...")

    # State labels
    state_labels = [
        "$p_x$ [m]",
        "$p_y$ [m]",
        "$p_z$ [m]",
        "$v_x$ [m/s]",
        "$v_y$ [m/s]",
        "$v_z$ [m/s]",
        "$q_w$",
        "$q_x$",
        "$q_y$",
        "$q_z$",
        "$\\omega_x$ [rad/s]",
        "$\\omega_y$ [rad/s]",
        "$\\omega_z$ [rad/s]",
    ]
    input_labels = ["$T$ [N]", "$\\tau_x$ [Nm]", "$\\tau_y$ [Nm]", "$\\tau_z$ [Nm]"]
    state_bounds = (workspace.x_min, workspace.x_max)
    input_bounds = (workspace.u_min, workspace.u_max)

    figures = {}

    # 3D Workspace plot
    fig_ws = plot_workspace_3d(
        workspace,
        trajectory,
        title="Quadrotor: 3D Workspace and Trajectory",
        position_indices=twin.position_indices,
    )
    figures["workspace_3d"] = fig_ws

    # 2D projections
    fig_2d, axes_2d = plt.subplots(1, 3, figsize=(15, 5))

    # XY projection
    ax = axes_2d[0]
    ax.plot(trajectory.x[:, 0], trajectory.x[:, 1], "b-", linewidth=1.5)
    ax.plot(trajectory.x[0, 0], trajectory.x[0, 1], "go", markersize=10)
    ax.plot(trajectory.x[-1, 0], trajectory.x[-1, 1], "r*", markersize=12)
    if workspace.obstacles is not None:
        for obs in workspace.obstacles.obstacles:
            circle = plt.Circle(obs.center[:2], obs.radius, color="red", alpha=0.5)
            ax.add_patch(circle)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("XY Projection")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # XZ projection
    ax = axes_2d[1]
    ax.plot(trajectory.x[:, 0], trajectory.x[:, 2], "b-", linewidth=1.5)
    ax.plot(trajectory.x[0, 0], trajectory.x[0, 2], "go", markersize=10)
    ax.plot(trajectory.x[-1, 0], trajectory.x[-1, 2], "r*", markersize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("XZ Projection")
    ax.grid(True, alpha=0.3)

    # YZ projection
    ax = axes_2d[2]
    ax.plot(trajectory.x[:, 1], trajectory.x[:, 2], "b-", linewidth=1.5)
    ax.plot(trajectory.x[0, 1], trajectory.x[0, 2], "go", markersize=10)
    ax.plot(trajectory.x[-1, 1], trajectory.x[-1, 2], "r*", markersize=12)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("YZ Projection")
    ax.grid(True, alpha=0.3)

    fig_2d.suptitle("Quadrotor: Trajectory Projections", fontsize=14, fontweight="bold")
    fig_2d.tight_layout()
    figures["workspace_2d"] = fig_2d

    # State plots (position and velocity)
    fig_pos_vel, axes_pv = plt.subplots(2, 3, figsize=(15, 8))

    for i in range(3):
        # Position
        axes_pv[0, i].plot(trajectory.t, trajectory.x[:, i], "b-", linewidth=1.5)
        axes_pv[0, i].axhline(y=state_bounds[0][i], color="r", linestyle="--", alpha=0.7)
        axes_pv[0, i].axhline(y=state_bounds[1][i], color="r", linestyle="--", alpha=0.7)
        axes_pv[0, i].set_xlabel("Time [s]")
        axes_pv[0, i].set_ylabel(state_labels[i])
        axes_pv[0, i].set_title(f"Position: {['x', 'y', 'z'][i]}")
        axes_pv[0, i].grid(True, alpha=0.3)

        # Velocity
        axes_pv[1, i].plot(trajectory.t, trajectory.x[:, 3 + i], "b-", linewidth=1.5)
        axes_pv[1, i].axhline(y=state_bounds[0][3 + i], color="r", linestyle="--", alpha=0.7)
        axes_pv[1, i].axhline(y=state_bounds[1][3 + i], color="r", linestyle="--", alpha=0.7)
        axes_pv[1, i].set_xlabel("Time [s]")
        axes_pv[1, i].set_ylabel(state_labels[3 + i])
        axes_pv[1, i].set_title(f"Velocity: {['x', 'y', 'z'][i]}")
        axes_pv[1, i].grid(True, alpha=0.3)

    fig_pos_vel.suptitle("Quadrotor: Position and Velocity", fontsize=14, fontweight="bold")
    fig_pos_vel.tight_layout()
    figures["states_pos_vel"] = fig_pos_vel

    # Attitude plots
    fig_att, axes_att = plt.subplots(2, 4, figsize=(16, 8))

    # Quaternion
    for i in range(4):
        axes_att[0, i].plot(trajectory.t, trajectory.x[:, 6 + i], "b-", linewidth=1.5)
        axes_att[0, i].axhline(y=state_bounds[0][6 + i], color="r", linestyle="--", alpha=0.7)
        axes_att[0, i].axhline(y=state_bounds[1][6 + i], color="r", linestyle="--", alpha=0.7)
        axes_att[0, i].set_xlabel("Time [s]")
        axes_att[0, i].set_ylabel(state_labels[6 + i])
        axes_att[0, i].set_title(f"Quaternion: {['w', 'x', 'y', 'z'][i]}")
        axes_att[0, i].grid(True, alpha=0.3)

    # Angular velocity
    for i in range(3):
        axes_att[1, i].plot(trajectory.t, trajectory.x[:, 10 + i], "b-", linewidth=1.5)
        axes_att[1, i].axhline(y=state_bounds[0][10 + i], color="r", linestyle="--", alpha=0.7)
        axes_att[1, i].axhline(y=state_bounds[1][10 + i], color="r", linestyle="--", alpha=0.7)
        axes_att[1, i].set_xlabel("Time [s]")
        axes_att[1, i].set_ylabel(state_labels[10 + i])
        axes_att[1, i].set_title(f"Angular velocity: {['x', 'y', 'z'][i]}")
        axes_att[1, i].grid(True, alpha=0.3)

    axes_att[1, 3].set_visible(False)

    fig_att.suptitle("Quadrotor: Attitude", fontsize=14, fontweight="bold")
    fig_att.tight_layout()
    figures["states_attitude"] = fig_att

    # Input plots
    fig_inputs = plot_inputs(
        trajectory,
        input_bounds,
        input_labels,
        title="Quadrotor: Control Inputs",
    )
    figures["inputs"] = fig_inputs

    # Convergence plot
    fig_conv = plot_convergence(
        result.convergence_history,
        title="Quadrotor: SCvx Convergence",
    )
    figures["convergence"] = fig_conv

    # Analysis plot
    fig_analysis = plot_trajectory_analysis(
        trajectory,
        title="Quadrotor: Trajectory Analysis",
    )
    figures["analysis"] = fig_analysis

    # Save plots
    if save_plots and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, fig in figures.items():
            filepath = output_dir / f"quadrotor_{name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"  Saved: {filepath}")

        # Save trajectory
        traj_path = output_dir / "quadrotor_trajectory.npz"
        trajectory.save(traj_path)
        logger.info(f"  Saved: {traj_path}")

    results = {
        "status": "success",
        "trajectory": trajectory,
        "result": result,
        "analysis": analysis,
        "increment_analysis": increment_analysis,
        "figures": figures,
    }

    return trajectory, results


# =============================================================================
# Main Function
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Design nominal trajectories using SCvx")
    parser.add_argument(
        "--system",
        type=str,
        default="unicycle",
        choices=["unicycle", "quadrotor", "both"],
        help="System to design trajectory for",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=500,
        help="Number of timesteps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/nominal_trajectories",
        help="Output directory for plots and data",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    import logging  # noqa: PLC0415

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)

    save_plots = not args.no_save

    results = {}

    if args.system in ["unicycle", "both"]:
        # Use system-specific output directory: results/unicycle/nominal_trajectories
        output_dir = Path("results") / "unicycle" / "nominal_trajectories"
        trajectory, res = design_unicycle_trajectory(
            N=args.N,
            save_plots=save_plots,
            output_dir=output_dir,
        )
        results["unicycle"] = res

    if args.system in ["quadrotor", "both"]:
        # Use system-specific output directory: results/quadrotor/nominal_trajectories
        output_dir = Path("results") / "quadrotor" / "nominal_trajectories"
        trajectory, res = design_quadrotor_trajectory(
            N=args.N,
            save_plots=save_plots,
            output_dir=output_dir,
        )
        results["quadrotor"] = res

    if args.show:
        plt.show()

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
