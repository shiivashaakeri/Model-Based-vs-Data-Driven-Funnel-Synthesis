"""Generic plotting utilities for DDFS visualization.

This module provides system-agnostic plotting functions that work for
any vehicle type (unicycle, quadrotor, etc.). These utilities handle
common visualization tasks like trajectories, controls, and ellipsoids.

IMPORTANT: All ellipsoid functions now work with P-based EllipsoidParams
where E(P) = {η | η^T P η ≤ 1}. Larger P means smaller ellipsoid.
"""

from typing import Any, Dict, List, Optional, Tuple

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
    """Setup figure with consistent DDFS styling."""
    plot_style = FIGURE_STYLE.copy()
    if style:
        plot_style.update(style)

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(plot_style)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def plot_ellipse_from_P(
    P: np.ndarray,
    center: np.ndarray,
    ax: plt.Axes,
    n_std: float = 1.0,
    **kwargs,
) -> Ellipse:
    """Plot ellipse from P matrix where E(P) = {η | η^T P η ≤ 1}.

    Args:
        P: Shape matrix (positive definite), shape (2, 2) or larger
        center: Ellipse center, shape (2,) or larger (uses first 2 dims)
        ax: Matplotlib axes
        n_std: Scaling factor (1.0 means the boundary where η^T P η = 1)
        **kwargs: Additional arguments for Ellipse patch

    Returns:
        ellipse: Matplotlib Ellipse patch
    """
    # Extract 2D submatrix for x-y projection
    P_2d = P[:2, :2]
    center_2d = center[:2]

    # For E(P) = {η | η^T P η ≤ n_std^2}, we need the inverse for plotting
    # The covariance matrix for plotting is Σ = (1/n_std^2) * P^{-1}
    try:
        P_inv = np.linalg.inv(P_2d)
        Sigma = (1.0 / n_std**2) * P_inv
    except np.linalg.LinAlgError:
        # Fallback to identity if P is singular
        Sigma = np.eye(2) * 0.01

    # Compute eigenvalues and eigenvectors of Sigma
    eigvals, eigvecs = np.linalg.eigh(Sigma)

    # Ensure positive eigenvalues
    eigvals = np.maximum(eigvals, 1e-10)

    # Compute angle and width/height
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * np.sqrt(eigvals[0])
    height = 2 * np.sqrt(eigvals[1])

    # Create ellipse patch
    ellipse = Ellipse(
        xy=center_2d,
        width=width,
        height=height,
        angle=angle,
        **kwargs,
    )

    ax.add_patch(ellipse)
    return ellipse


def plot_ellipsoid_envelope_spatial(
    nominal,
    ellipsoids_dict,
    workspace,
    obstacles,
    segmented_data,
    ax: Optional[plt.Axes] = None,
    sample_every: int = 3,
) -> plt.Axes:
    """Plot spatial view of ellipsoid envelope with sampled segments.

    Shows the x-y projection of ellipsoids along the nominal trajectory.

    Args:
        nominal: NominalTrajectory object
        ellipsoids_dict: Dict with 'P_0_list', 'P_min_0_list', 'P_min_0_init_list', 'envelope_list'
        workspace: Workspace object
        obstacles: List of Obstacle objects
        segmented_data: SegmentedData with k_starts, k_ends
        ax: Matplotlib axes (creates new if None)
        sample_every: Show every Nth segment

    Returns:
        ax: Matplotlib axes with visualization
    """
    if ax is None:
        _, ax = setup_figure(figsize=(14, 10))

    # Try to use envelope_list first (new format)
    envelope_list = ellipsoids_dict.get("envelope_list", [])

    if envelope_list:
        # New format: extract from FeasibilityEnvelope objects
        P_0_list = [env.P_0 for env in envelope_list]
        P_min_0_list = [env.P_min_segment for env in envelope_list]
        P_min_0_init_list = [env.P_min_0_init for env in envelope_list]
    else:
        # Old format: extract directly from dict
        P_0_list = ellipsoids_dict.get("P_0_list", [])
        P_min_0_list = ellipsoids_dict.get("P_min_0_list", [])
        P_min_0_init_list = ellipsoids_dict.get("P_min_0_init_list", [])

    if not P_0_list:
        ax.text(0.5, 0.5, "No ellipsoid data", ha="center", va="center", transform=ax.transAxes)
        return ax

    # Plot workspace boundary
    if hasattr(workspace, "bounds"):
        bounds = workspace.bounds
        ax.plot(
            [bounds[0], bounds[1], bounds[1], bounds[0], bounds[0]],
            [bounds[2], bounds[2], bounds[3], bounds[3], bounds[2]],
            color=COLORS["workspace"],
            linewidth=2,
            linestyle="--",
            label="Workspace",
            zorder=1,
        )

    # Plot obstacles
    for obs in obstacles:
        if len(obs.center) >= 2:
            circle = plt.Circle(
                obs.center[:2],
                obs.effective_radius,
                color=COLORS["obstacle"],
                alpha=0.3,
                label="Obstacles" if obs == obstacles[0] else "",
                zorder=2,
            )
            ax.add_patch(circle)

    # Plot nominal trajectory
    ax.plot(
        nominal.x_nom[:, 0],
        nominal.x_nom[:, 1],
        color=COLORS["nominal"],
        linewidth=2.5,
        label="Nominal",
        zorder=5,
    )

    # Plot start and goal
    ax.plot(
        nominal.x_nom[0, 0],
        nominal.x_nom[0, 1],
        marker="o",
        markersize=12,
        color=COLORS["start"],
        label="Start",
        zorder=10,
    )
    ax.plot(
        nominal.x_nom[-1, 0],
        nominal.x_nom[-1, 1],
        marker="*",
        markersize=15,
        color=COLORS["goal"],
        label="Goal",
        zorder=10,
    )

    # Plot sampled ellipsoids
    sampled_indices = range(0, len(P_0_list), sample_every)

    for idx in sampled_indices:
        if idx >= len(P_0_list):
            break

        # Get segment start timestep
        k_start = segmented_data.k_starts[idx] if idx < len(segmented_data.k_starts) else 0
        center = nominal.x_nom[k_start]

        # Plot P_min_0_init (outermost, gray dashed)
        if idx < len(P_min_0_init_list):
            plot_ellipse_from_P(
                P_min_0_init_list[idx].P,
                center,
                ax,
                n_std=1.0,
                facecolor="none",
                edgecolor="gray",
                linewidth=1.5,
                linestyle="--",
                alpha=0.5,
                label="P_min_0_init" if idx == sampled_indices[0] else "",
                zorder=3,
            )

        # Plot P_min_0 (middle, blue)
        if idx < len(P_min_0_list):
            plot_ellipse_from_P(
                P_min_0_list[idx].P,
                center,
                ax,
                n_std=1.0,
                facecolor=COLORS["nominal"],
                edgecolor=COLORS["nominal"],
                linewidth=2,
                alpha=0.15,
                label="P_min_0" if idx == sampled_indices[0] else "",
                zorder=4,
            )

        # Plot P_0 (innermost, orange)
        if idx < len(P_0_list):
            plot_ellipse_from_P(
                P_0_list[idx].P,
                center,
                ax,
                n_std=1.0,
                facecolor=COLORS["funnel"],
                edgecolor=COLORS["funnel"],
                linewidth=2.5,
                alpha=0.25,
                label="P_0" if idx == sampled_indices[0] else "",
                zorder=5,
            )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title = f"Feasibility Envelope - Spatial View (showing {len(sampled_indices)}/{len(P_0_list)} segments)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    return ax


def plot_state_bounds_vs_time(
    nominal,
    ellipsoids_dict,
    workspace,  # noqa: ARG001
    obstacles,  # noqa: ARG001
    time: Optional[np.ndarray] = None,
    state_labels: Optional[List[str]] = None,
    fig: Optional[plt.Figure] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot state bounds over time from feasibility envelopes.

    Shows each state dimension separately with ellipsoid-derived bounds.

    Args:
        nominal: NominalTrajectory object
        ellipsoids_dict: Dict with ellipsoid lists
        workspace: Workspace object
        obstacles: List of obstacles
        time: Time vector for states (if None, uses dt from nominal)
        state_labels: List of state labels (e.g., ['x', 'y', 'θ'])
        fig: Matplotlib figure (creates new if None)

    Returns:
        fig: Matplotlib figure
        axes: List of subplot axes (one per state dimension)
    """
    # Create time vector for states
    if time is None:
        time = np.arange(nominal.N + 1) * nominal.dt

    # Extract ellipsoid lists
    envelope_list = ellipsoids_dict.get("envelope_list", [])

    if envelope_list:
        P_0_list = [env.P_0 for env in envelope_list]
        P_min_0_list = [env.P_min_segment for env in envelope_list]
        P_min_0_init_list = [env.P_min_0_init for env in envelope_list]
        segment_indices = ellipsoids_dict.get("segment_indices", [])
    else:
        P_0_list = ellipsoids_dict.get("P_0_list", [])
        P_min_0_list = ellipsoids_dict.get("P_min_0_list", [])
        P_min_0_init_list = ellipsoids_dict.get("P_min_0_init_list", [])
        segment_indices = ellipsoids_dict.get("segment_indices", [])

    # Get state dimension
    n = nominal.state_dim

    # Create subplots
    if fig is None:
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
        if n == 1:
            axes = [axes]
    else:
        axes = fig.get_axes()

    # Compute bounds for each state dimension
    # For E(P) = {η | η^T P η ≤ 1}, the bound in dimension i is ±1/sqrt(P_ii)
    def compute_bounds_from_P(P_list, segment_indices, N):
        """Compute ± bounds from P matrices."""
        upper = np.full((n, N + 1), np.nan)
        lower = np.full((n, N + 1), np.nan)

        for seg_idx, P_ell in enumerate(P_list):
            if seg_idx >= len(segment_indices):
                continue

            if isinstance(segment_indices[seg_idx], (list, tuple)):
                k_start = segment_indices[seg_idx][0]
                k_end = segment_indices[seg_idx][1]
            else:
                k_start = seg_idx * 20
                k_end = min(k_start + 20, N)

            # Extract P matrix
            P = P_ell.P if hasattr(P_ell, 'P') else P_ell
            center = P_ell.c if hasattr(P_ell, 'c') else nominal.x_nom[k_start]

            # For each dimension, bound is ±1/sqrt(P_ii)
            for i in range(n):
                if i < P.shape[0]:
                    bound = 1.0 / np.sqrt(max(P[i, i], 1e-10))
                    k_end_inclusive = k_end + 1
                    upper[i, k_start:k_end_inclusive] = center[i] + bound
                    lower[i, k_start:k_end_inclusive] = center[i] - bound

        return upper, lower

    # Compute bounds
    bounds_P_0_upper, bounds_P_0_lower = compute_bounds_from_P(P_0_list, segment_indices, nominal.N)
    bounds_P_min_0_upper, bounds_P_min_0_lower = compute_bounds_from_P(P_min_0_list, segment_indices, nominal.N)
    bounds_P_min_0_init_upper, bounds_P_min_0_init_lower = compute_bounds_from_P(P_min_0_init_list, segment_indices, nominal.N)

    # Plot each state dimension
    for i in range(n):
        ax = axes[i]

        # Plot nominal state
        ax.plot(
            time,
            nominal.x_nom[:, i],
            color=COLORS["nominal"],
            linewidth=2.5,
            label="Nominal",
            zorder=5,
        )

        # Plot P_0 bounds (orange shaded)
        ax.fill_between(
            time,
            bounds_P_0_lower[i, :],
            bounds_P_0_upper[i, :],
            color=COLORS["funnel"],
            alpha=0.3,
            label="P_0 bounds",
            zorder=1,
        )

        # Plot P_min_0 bounds (blue lines)
        ax.plot(
            time,
            bounds_P_min_0_upper[i, :],
            color=COLORS["nominal"],
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            label="P_min_0 bounds" if i == 0 else "",
            zorder=2,
        )
        ax.plot(
            time,
            bounds_P_min_0_lower[i, :],
            color=COLORS["nominal"],
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            zorder=2,
        )

        # Plot P_min_0_init bounds (gray dashed)
        ax.plot(
            time,
            bounds_P_min_0_init_upper[i, :],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            label="P_min_0_init" if i == 0 else "",
            zorder=3,
        )
        ax.plot(
            time,
            bounds_P_min_0_init_lower[i, :],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            zorder=3,
        )

        # Styling
        ax.set_ylabel(state_labels[i] if state_labels and i < len(state_labels) else f"x_{i}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        if i == 0:
            ax.set_title("State Evolution with Feasibility Bounds")
        if i == n - 1:
            ax.set_xlabel("Time (s)")

    plt.tight_layout()
    return fig, axes


def plot_funnel_envelope_detailed(
    nominal,
    ellipsoids,
    constants,  # noqa: ARG001
    workspace,
    obstacles,
    segmented_data,
    constraints,
    fig: Optional[plt.Figure] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Comprehensive feasibility envelope visualization dashboard.

    Creates multi-row figure showing:
    - Top: Spatial view with ellipsoids along trajectory
    - Middle: State bounds vs time (one subplot per state)
    - Bottom: Input bounds vs time (one subplot per input)

    Args:
        nominal: NominalTrajectory object
        ellipsoids: Dict with ellipsoid lists
        constants: UncertaintyConstants object
        workspace: Workspace object
        obstacles: List of obstacles
        segmented_data: SegmentedData object
        constraints: SystemConstraints object
        fig: Matplotlib figure (creates new if None)

    Returns:
        fig: Matplotlib figure
        axes_dict: Dict with keys 'spatial', 'states', 'inputs' containing axes
    """
    n = nominal.state_dim
    m = nominal.input_dim

    total_rows = 1 + n + m

    if fig is None:
        fig = plt.figure(figsize=(16, 4 + 3 * n + 3 * m))
    else:
        fig.clear()

    gs = fig.add_gridspec(
        total_rows,
        1,
        height_ratios=[4] + [3] * n + [3] * m,
        hspace=0.4,
    )

    axes_dict = {}

    # Top: Spatial view
    ax_spatial = fig.add_subplot(gs[0, 0])
    plot_ellipsoid_envelope_spatial(
        nominal,
        ellipsoids,
        workspace,
        obstacles,
        segmented_data,
        ax=ax_spatial,
        sample_every=max(1, segmented_data.num_segments // 8),
    )
    axes_dict["spatial"] = ax_spatial

    # Middle: State bounds
    state_axes = []
    for i in range(n):
        ax = fig.add_subplot(gs[1 + i, 0])
        state_axes.append(ax)

    # Call state bounds plotting
    _, state_axes = plot_state_bounds_vs_time(
        nominal,
        ellipsoids,
        workspace,
        obstacles,
        fig=fig,
    )
    axes_dict["states"] = state_axes

    # Bottom: Input bounds (simplified placeholder)
    input_axes = []
    for j in range(m):
        ax = fig.add_subplot(gs[1 + n + j, 0])

        time = np.arange(nominal.N) * nominal.dt
        ax.plot(time, nominal.u_nom[:, j], color=COLORS["nominal"], linewidth=2, label="Nominal")

        has_u_bounds = hasattr(constraints, "u_min") and hasattr(constraints, "u_max")
        if has_u_bounds and j < len(constraints.u_min):
            ax.axhline(constraints.u_min[j], color=COLORS["unsafe"], linestyle="--", linewidth=2, alpha=0.7)
            ax.axhline(constraints.u_max[j], color=COLORS["unsafe"], linestyle="--", linewidth=2, alpha=0.7)

        ax.set_ylabel(f"u_{j}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if j == m - 1:
            ax.set_xlabel("Time (s)")

        input_axes.append(ax)

    axes_dict["inputs"] = input_axes

    plt.tight_layout()
    return fig, axes_dict


def add_start_goal_markers(
    ax: plt.Axes,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    is_3d: bool = False,
) -> None:
    """Add start and goal markers to axes.

    Args:
        ax: Matplotlib axes
        x_start: Start state (first 2 or 3 elements used)
        x_goal: Goal state (first 2 or 3 elements used)
        is_3d: Whether this is a 3D plot
    """
    if is_3d:
        ax.plot([x_start[0]], [x_start[1]], [x_start[2]], "o", color=COLORS["start"], markersize=12, label="Start", zorder=10)
        ax.plot([x_goal[0]], [x_goal[1]], [x_goal[2]], "*", color=COLORS["goal"], markersize=15, label="Goal", zorder=10)
    else:
        ax.plot(x_start[0], x_start[1], "o", color=COLORS["start"], markersize=12, label="Start", zorder=10)
        ax.plot(x_goal[0], x_goal[1], "*", color=COLORS["goal"], markersize=15, label="Goal", zorder=10)


def plot_ellipsoid_2d(
    P: np.ndarray,
    center: np.ndarray,
    ax: plt.Axes,
    n_std: float = 1.0,
    **kwargs,
) -> Ellipse:
    """Plot 2D ellipse from P matrix (convenience wrapper for plot_ellipse_from_P).

    Args:
        P: Shape matrix (2x2 or larger, uses first 2x2)
        center: Center point (2D or larger, uses first 2 elements)
        ax: Matplotlib axes
        n_std: Scaling factor
        **kwargs: Additional arguments for Ellipse patch

    Returns:
        ellipse: Matplotlib Ellipse patch
    """
    return plot_ellipse_from_P(P, center, ax, n_std=n_std, **kwargs)


def plot_tracking_error(
    x_nominal: np.ndarray,
    x_actual: np.ndarray,
    time: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot tracking error over time.

    Args:
        x_nominal: Nominal state trajectory (N, n)
        x_actual: Actual state trajectory (N, n)
        time: Time vector (if None, uses indices)
        ax: Matplotlib axes (creates new if None)

    Returns:
        ax: Matplotlib axes with tracking error plot
    """
    if ax is None:
        _, ax = setup_figure(figsize=(10, 6))

    if time is None:
        time = np.arange(len(x_nominal))

    error = x_actual - x_nominal
    error_norm = np.linalg.norm(error, axis=1)

    ax.plot(time, error_norm, color=COLORS["unsafe"], linewidth=2, label="Tracking Error")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Error ||x_actual - x_nominal||", fontsize=11)
    ax.set_title("Tracking Error vs Time", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def save_figure(fig: plt.Figure, path, **kwargs) -> None:
    """Save figure to file.

    Args:
        fig: Matplotlib figure
        path: Path to save file
        **kwargs: Additional arguments for savefig
    """
    default_kwargs = {"dpi": 300, "bbox_inches": "tight"}
    default_kwargs.update(kwargs)
    fig.savefig(path, **default_kwargs)


def set_equal_aspect(ax: plt.Axes) -> None:
    """Set equal aspect ratio for 2D axes.

    Args:
        ax: Matplotlib axes
    """
    ax.set_aspect("equal", adjustable="box")


# Export key functions
__all__ = [
    "COLORS",
    "add_start_goal_markers",
    "plot_ellipse_from_P",
    "plot_ellipsoid_2d",
    "plot_ellipsoid_envelope_spatial",
    "plot_funnel_envelope_detailed",
    "plot_state_bounds_vs_time",
    "plot_tracking_error",
    "save_figure",
    "set_equal_aspect",
    "setup_figure",
]
