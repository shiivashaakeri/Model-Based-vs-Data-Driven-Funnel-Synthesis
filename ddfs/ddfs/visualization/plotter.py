"""
Visualization module for DDFS pipeline.

Provides plotting utilities for:
- Nominal trajectories with obstacles
- State and input trajectories over time
- Constraint visualization
- Multi-phase comparison (to be extended)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from ddfs.planning.nominal_trajectory import NominalTrajectory


class DDFSPlotter:
    """
    Main plotter class for DDFS visualization.

    Handles plotting for all phases of the DDFS pipeline:
    - Phase 1: Nominal trajectory planning
    - Phase 2-6: (to be added later)

    Attributes
    ----------
    figsize : Tuple[int, int]
        Default figure size
    dpi : int
        Figure DPI for saving
    style : str
        Matplotlib style
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150, style: str = "default"):
        """
        Initialize plotter.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Default figure size (width, height)
        dpi : int
            DPI for saved figures
        style : str
            Matplotlib style to use
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        # Set style
        plt.style.use(style)

        # Color scheme
        self.colors = {
            "nominal": "#2E86AB",
            "actual": "#A23B72",
            "reference": "#F18F01",
            "obstacle": "#C73E1D",
            "start": "#06A77D",
            "goal": "#D62246",
            "constraint": "#666666",
        }

    # ========================================================================
    # PHASE 1: NOMINAL TRAJECTORY PLOTTING
    # ========================================================================

    def plot_nominal_2d(
        self,
        trajectory: NominalTrajectory,
        obstacles: List[Dict[str, Any]],
        workspace: Dict[str, float],
        save_path: Optional[Path] = None,
        show: bool = False,
        title: str = "Nominal Trajectory",
    ) -> plt.Figure:
        """
        Plot 2D nominal trajectory with obstacles.

        Parameters
        ----------
        trajectory : NominalTrajectory
            Nominal trajectory to plot
        obstacles : List[Dict[str, Any]]
            List of obstacle dictionaries with 'center', 'radius', 'type'
        workspace : Dict[str, float]
            Workspace bounds (x_min, x_max, y_min, y_max)
        save_path : Optional[Path]
            Path to save figure (if provided)
        show : bool
            Whether to display figure
        title : str
            Plot title

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract position trajectory (assume first 2 states are x, y)
        x_traj = trajectory.x_nom[:, 0]
        y_traj = trajectory.x_nom[:, 1]

        # Plot trajectory
        ax.plot(x_traj, y_traj, "-", color=self.colors["nominal"], linewidth=2.5, label="Nominal Trajectory", zorder=3)

        # Plot start and goal
        ax.plot(x_traj[0], y_traj[0], "o", color=self.colors["start"], markersize=12, label="Start", zorder=4)
        ax.plot(x_traj[-1], y_traj[-1], "*", color=self.colors["goal"], markersize=18, label="Goal", zorder=4)

        # Plot waypoints
        step = max(1, trajectory.N // 10)
        ax.plot(x_traj[::step], y_traj[::step], "o", color=self.colors["nominal"], markersize=4, alpha=0.5, zorder=2)

        # Plot obstacles
        for obs in obstacles:
            if obs["type"] == "circle":
                center = obs["center"]
                radius = obs["radius"]

                # Obstacle circle
                circle = Circle(center, radius, color=self.colors["obstacle"], alpha=0.3, zorder=1)
                ax.add_patch(circle)

                # Obstacle center
                ax.plot(center[0], center[1], "x", color=self.colors["obstacle"], markersize=10, markeredgewidth=2)

                # Safety margin (if provided)
                if "safety_margin" in obs:
                    margin_radius = radius + obs["safety_margin"]
                    circle_margin = Circle(
                        center,
                        margin_radius,
                        color=self.colors["obstacle"],
                        alpha=0.1,
                        linestyle="--",
                        fill=False,
                        zorder=1,
                    )
                    ax.add_patch(circle_margin)

        # Workspace bounds
        ax.set_xlim(workspace["x_min"], workspace["x_max"])
        ax.set_ylim(workspace["y_min"], workspace["y_max"])

        # Labels and formatting
        ax.set_xlabel("x (m)", fontsize=12)
        ax.set_ylabel("y (m)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal", adjustable="box")

        # Add info text
        info_text = f"N = {trajectory.N}\ndt = {trajectory.dt:.3f}s\ntf = {trajectory.tf:.2f}s"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"  ✓ Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_nominal_3d(  # noqa: PLR0915
        self,
        trajectory: NominalTrajectory,
        obstacles: List[Dict[str, Any]],
        workspace: Dict[str, float],
        save_path: Optional[Path] = None,
        show: bool = False,
        title: str = "Nominal Trajectory (3D)",
    ) -> plt.Figure:
        """
        Plot 3D nominal trajectory with spherical obstacles.

        Creates an INTERACTIVE 3D plot that you can rotate and zoom with mouse.

        Parameters
        ----------
        trajectory : NominalTrajectory
            Nominal trajectory to plot
        obstacles : List[Dict[str, Any]]
            List of obstacle dictionaries with 'center', 'radius', 'type'
        workspace : Dict[str, float]
            Workspace bounds (x_min, x_max, y_min, y_max, z_min, z_max)
        save_path : Optional[Path]
            Path to save figure
        show : bool
            Whether to display figure interactively
        title : str
            Plot title

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object

        Notes
        -----
        Interactive controls (when show=True):
        - Left mouse: Rotate view
        - Right mouse: Zoom
        - Middle mouse: Pan
        """
        # Use interactive backend for 3D
        import matplotlib  # noqa: PLC0415, ICN001

        if show:
            matplotlib.use("TkAgg")  # Interactive backend

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Extract position trajectory (assume first 3 states are x, y, z)
        x_traj = trajectory.x_nom[:, 0]
        y_traj = trajectory.x_nom[:, 1]
        z_traj = trajectory.x_nom[:, 2]

        # Plot trajectory
        ax.plot(
            x_traj,
            y_traj,
            z_traj,
            "-",
            color=self.colors["nominal"],
            linewidth=2.5,
            label="Nominal Trajectory",
            zorder=3,
        )

        # Plot start and goal
        ax.scatter(
            x_traj[0],
            y_traj[0],
            z_traj[0],
            color=self.colors["start"],
            s=150,
            marker="o",
            label="Start",
            zorder=4,
            edgecolors="black",
            linewidths=2,
        )
        ax.scatter(
            x_traj[-1],
            y_traj[-1],
            z_traj[-1],
            color=self.colors["goal"],
            s=200,
            marker="*",
            label="Goal",
            zorder=4,
            edgecolors="black",
            linewidths=2,
        )

        # Plot waypoints
        step = max(1, trajectory.N // 10)
        ax.scatter(
            x_traj[::step], y_traj[::step], z_traj[::step], color=self.colors["nominal"], s=20, alpha=0.5, zorder=2
        )

        # Plot obstacles (spheres)
        for obs in obstacles:
            if obs["type"] == "sphere":
                center = obs["center"]
                radius = obs["radius"]

                # Create sphere
                u = np.linspace(0, 2 * np.pi, 30)
                v = np.linspace(0, np.pi, 20)
                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

                ax.plot_surface(x_sphere, y_sphere, z_sphere, color=self.colors["obstacle"], alpha=0.3)

        # Workspace bounds
        ax.set_xlim(workspace["x_min"], workspace["x_max"])
        ax.set_ylim(workspace["y_min"], workspace["y_max"])
        ax.set_zlim(workspace["z_min"], workspace["z_max"])

        # Labels and formatting
        ax.set_xlabel("x (m)", fontsize=11)
        ax.set_ylabel("y (m)", fontsize=11)
        ax.set_zlabel("z (m)", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set initial viewing angle
        ax.view_init(elev=25, azim=45)

        # Add text with interaction instructions if showing interactively
        if show:
            info_text = "Interactive: Left-drag to rotate, Right-drag to zoom, Middle-drag to pan"
            fig.text(
                0.5, 0.02, info_text, ha="center", fontsize=9, bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}  # noqa: E501
            )

        plt.tight_layout()

        if save_path:
            # Save multiple views
            save_path = Path(save_path)

            # View 1: Default
            ax.view_init(elev=25, azim=45)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"  ✓ Saved: {save_path}")

            # View 2: Top-down
            stem = save_path.stem
            suffix = save_path.suffix
            top_view_path = save_path.parent / f"{stem}_top{suffix}"
            ax.view_init(elev=90, azim=0)
            fig.savefig(top_view_path, dpi=self.dpi, bbox_inches="tight")
            print(f"  ✓ Saved: {top_view_path} (top view)")

            # View 3: Side
            side_view_path = save_path.parent / f"{stem}_side{suffix}"
            ax.view_init(elev=0, azim=0)
            fig.savefig(side_view_path, dpi=self.dpi, bbox_inches="tight")
            print(f"  ✓ Saved: {side_view_path} (side view)")

            # Reset to default view
            ax.view_init(elev=25, azim=45)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_states_vs_time(  # noqa: C901
        self,
        trajectory: NominalTrajectory,
        state_bounds: Optional[Dict[str, Any]] = None,
        state_labels: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        show: bool = False,
        title: str = "State Trajectories",
    ) -> plt.Figure:
        """
        Plot state trajectories vs time with constraint bounds.

        Parameters
        ----------
        trajectory : NominalTrajectory
            Nominal trajectory to plot
        state_bounds : Optional[Dict[str, Any]]
            Dictionary of state bounds (x_min, x_max, y_min, etc.)
        state_labels : Optional[List[str]]
            Labels for each state
        save_path : Optional[Path]
            Path to save figure
        show : bool
            Whether to display figure
        title : str
            Plot title

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        """
        n = trajectory.state_dim
        time = np.linspace(0, trajectory.tf, trajectory.N + 1)

        # Default labels if not provided
        if state_labels is None:
            state_labels = [f"x_{i + 1}" for i in range(n)]

        # Create subplots
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))

        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i in range(n):
            ax = axes[i]

            # Plot state trajectory
            ax.plot(time, trajectory.x_nom[:, i], "-", color=self.colors["nominal"], linewidth=2, label="Nominal")

            # Plot bounds if provided
            if state_bounds is not None:

                # Check for min/max bounds
                min_key = f"{state_labels[i]}_min"
                max_key = f"{state_labels[i]}_max"

                if min_key in state_bounds and not np.isinf(state_bounds[min_key]):
                    ax.axhline(
                        state_bounds[min_key],
                        color=self.colors["constraint"],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Bounds",
                    )

                if max_key in state_bounds and not np.isinf(state_bounds[max_key]):
                    ax.axhline(
                        state_bounds[max_key], color=self.colors["constraint"], linestyle="--", linewidth=1.5, alpha=0.7
                    )

            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel(state_labels[i], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")

            if i == 0:
                ax.legend(loc="best", fontsize=9)

        # Hide extra subplots
        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"  ✓ Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_inputs_vs_time(  # noqa: C901
        self,
        trajectory: NominalTrajectory,
        input_bounds: Optional[Dict[str, Any]] = None,
        input_labels: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        show: bool = False,
        title: str = "Input Trajectories",
    ) -> plt.Figure:
        """
        Plot input trajectories vs time with constraint bounds.

        Parameters
        ----------
        trajectory : NominalTrajectory
            Nominal trajectory to plot
        input_bounds : Optional[Dict[str, Any]]
            Dictionary of input bounds (v_min, v_max, omega_min, etc.)
        input_labels : Optional[List[str]]
            Labels for each input
        save_path : Optional[Path]
            Path to save figure
        show : bool
            Whether to display figure
        title : str
            Plot title

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        """
        m = trajectory.input_dim
        time = np.linspace(0, trajectory.tf, trajectory.N + 1)
        time_input = time[:-1]  # Inputs are N timesteps

        # Default labels if not provided
        if input_labels is None:
            input_labels = [f"u_{i + 1}" for i in range(m)]

        # Create subplots
        ncols = min(3, m)
        nrows = int(np.ceil(m / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))

        if m == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i in range(m):
            ax = axes[i]

            # Plot input trajectory (step function)
            ax.step(
                time_input,
                trajectory.u_nom[:, i],
                where="post",
                color=self.colors["nominal"],
                linewidth=2,
                label="Nominal",
            )

            # Plot bounds if provided
            if input_bounds is not None:
                # Try to find bounds for this input
                min_key = f"{input_labels[i]}_min"
                max_key = f"{input_labels[i]}_max"

                if min_key in input_bounds and not np.isinf(input_bounds[min_key]):
                    ax.axhline(
                        input_bounds[min_key],
                        color=self.colors["constraint"],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Bounds",
                    )

                if max_key in input_bounds and not np.isinf(input_bounds[max_key]):
                    ax.axhline(
                        input_bounds[max_key], color=self.colors["constraint"], linestyle="--", linewidth=1.5, alpha=0.7
                    )

            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel(input_labels[i], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xlim(0, trajectory.tf)

            if i == 0:
                ax.legend(loc="best", fontsize=9)

        # Hide extra subplots
        for i in range(m, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"  ✓ Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_nominal_summary(
        self,
        trajectory: NominalTrajectory,
        obstacles: List[Dict[str, Any]],
        workspace: Dict[str, float],
        state_bounds: Optional[Dict[str, Any]] = None,
        input_bounds: Optional[Dict[str, Any]] = None,
        state_labels: Optional[List[str]] = None,
        input_labels: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        show: bool = False,
        system_name: str = "System",
    ) -> plt.Figure:
        """
        Create comprehensive summary plot for nominal trajectory.

        Combines spatial trajectory and time-domain plots in one figure.

        Parameters
        ----------
        trajectory : NominalTrajectory
            Nominal trajectory to plot
        obstacles : List[Dict[str, Any]]
            List of obstacles
        workspace : Dict[str, float]
            Workspace bounds
        state_bounds : Optional[Dict[str, Any]]
            State constraint bounds
        input_bounds : Optional[Dict[str, Any]]
            Input constraint bounds
        state_labels : Optional[List[str]]
            State labels
        input_labels : Optional[List[str]]
            Input labels
        save_path : Optional[Path]
            Path to save figure
        show : bool
            Whether to display figure
        system_name : str
            System name for title

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure object
        """
        # Determine if 2D or 3D
        is_3d = trajectory.state_dim >= 13  # Quadrotor

        if is_3d:
            # 3D system - simpler layout
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 3D trajectory (top left, span both rows)
            ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
            self._add_3d_trajectory_to_ax(ax_3d, trajectory, obstacles, workspace)

            # States (top right)
            ax_states = fig.add_subplot(gs[0, 1])
            self._add_states_summary_to_ax(ax_states, trajectory, state_bounds)

            # Inputs (bottom right)
            ax_inputs = fig.add_subplot(gs[1, 1])
            self._add_inputs_summary_to_ax(ax_inputs, trajectory, input_bounds)

        else:
            # 2D system - use gridspec for better layout
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 2D trajectory (left side)
            ax_2d = fig.add_subplot(gs[:, 0])
            self._add_2d_trajectory_to_ax(ax_2d, trajectory, obstacles, workspace)

            # States (top right)
            ax_states = fig.add_subplot(gs[0, 1])
            self._add_states_summary_to_ax(ax_states, trajectory, state_bounds, state_labels)

            # Inputs (bottom right)
            ax_inputs = fig.add_subplot(gs[1, 1])
            self._add_inputs_summary_to_ax(ax_inputs, trajectory, input_bounds, input_labels)

        fig.suptitle(f"{system_name} - Nominal Trajectory Summary", fontsize=16, fontweight="bold", y=0.98)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"  ✓ Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _add_2d_trajectory_to_ax(self, ax, trajectory, obstacles, workspace):
        """Helper to add 2D trajectory to existing axis."""
        x_traj = trajectory.x_nom[:, 0]
        y_traj = trajectory.x_nom[:, 1]

        ax.plot(x_traj, y_traj, "-", color=self.colors["nominal"], linewidth=2.5)
        ax.plot(x_traj[0], y_traj[0], "o", color=self.colors["start"], markersize=12)
        ax.plot(x_traj[-1], y_traj[-1], "*", color=self.colors["goal"], markersize=18)

        for obs in obstacles:
            if obs["type"] == "circle":
                circle = Circle(obs["center"], obs["radius"], color=self.colors["obstacle"], alpha=0.3)
                ax.add_patch(circle)

        ax.set_xlim(workspace["x_min"], workspace["x_max"])
        ax.set_ylim(workspace["y_min"], workspace["y_max"])
        ax.set_xlabel("x (m)", fontsize=11)
        ax.set_ylabel("y (m)", fontsize=11)
        ax.set_title("Spatial Trajectory", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    def _add_3d_trajectory_to_ax(self, ax, trajectory, obstacles, workspace):
        """Helper to add 3D trajectory to existing axis."""
        x_traj = trajectory.x_nom[:, 0]
        y_traj = trajectory.x_nom[:, 1]
        z_traj = trajectory.x_nom[:, 2]

        ax.plot(x_traj, y_traj, z_traj, "-", color=self.colors["nominal"], linewidth=2.5)
        ax.scatter(x_traj[0], y_traj[0], z_traj[0], color=self.colors["start"], s=150)
        ax.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color=self.colors["goal"], s=200, marker="*")

        for obs in obstacles:
            if obs["type"] == "sphere":
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 15)
                x = obs["center"][0] + obs["radius"] * np.outer(np.cos(u), np.sin(v))
                y = obs["center"][1] + obs["radius"] * np.outer(np.sin(u), np.sin(v))
                z = obs["center"][2] + obs["radius"] * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color=self.colors["obstacle"], alpha=0.3)

        ax.set_xlim(workspace["x_min"], workspace["x_max"])
        ax.set_ylim(workspace["y_min"], workspace["y_max"])
        ax.set_zlim(workspace["z_min"], workspace["z_max"])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Spatial Trajectory", fontsize=12, fontweight="bold")
        ax.view_init(elev=25, azim=45)

    def _add_states_summary_to_ax(self, ax, trajectory, state_bounds, labels=None):  # noqa: ARG002
        """Helper to add state summary to existing axis."""
        time = np.linspace(0, trajectory.tf, trajectory.N + 1)

        # Plot first 3-4 most important states
        n_plot = min(4, trajectory.state_dim)
        for i in range(n_plot):
            label = labels[i] if labels and i < len(labels) else f"$x_{i + 1}$"
            ax.plot(time, trajectory.x_nom[:, i], label=label, linewidth=1.5)

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("States", fontsize=10)
        ax.set_title("State Trajectories", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _add_inputs_summary_to_ax(self, ax, trajectory, input_bounds, labels=None):
        """Helper to add input summary to existing axis."""
        time = np.linspace(0, trajectory.tf, trajectory.N + 1)[:-1]

        for i in range(trajectory.input_dim):
            label = labels[i] if labels and i < len(labels) else f"$u_{i + 1}$"
            ax.step(time, trajectory.u_nom[:, i], where="post", label=label, linewidth=1.5)

            # Add bounds if available
            if input_bounds:
                min_key = f"{labels[i]}_min" if labels else f"u{i + 1}_min"
                max_key = f"{labels[i]}_max" if labels else f"u{i + 1}_max"

                if min_key in input_bounds:
                    ax.axhline(input_bounds[min_key], color="gray", linestyle="--", alpha=0.5, linewidth=1)
                if max_key in input_bounds:
                    ax.axhline(input_bounds[max_key], color="gray", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Inputs", fontsize=10)
        ax.set_title("Input Trajectories", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, trajectory.tf)
