"""Feasibility verification plotting - detailed constraint visualization.

This module provides detailed plotting to verify that MVIE ellipsoids
respect all constraints (obstacles, workspace, state bounds) at every timestep.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def compute_linearized_obstacle_bounds(
    nominal,
    obstacles,
    beta: float,
    state_dim_idx: int,
) -> tuple:
    """Compute linearized obstacle avoidance bounds for a specific state dimension.

    For each timestep k, compute the tightest bound from all obstacles
    on the state dimension state_dim_idx.

    Args:
        nominal: NominalTrajectory with x_nom
        obstacles: List of Obstacle objects
        beta: Safety margin
        state_dim_idx: Which state dimension to compute bounds for (0=x, 1=y, etc.)

    Returns:
        upper_bounds: Array of upper bounds at each timestep, shape (N+1,)
        lower_bounds: Array of lower bounds at each timestep, shape (N+1,)
        constraint_active: Array of booleans indicating if constraint is active
    """
    N = nominal.N
    upper_bounds = np.full(N + 1, np.inf)
    lower_bounds = np.full(N + 1, -np.inf)
    constraint_active = np.zeros(N + 1, dtype=bool)

    for k in range(N + 1):
        x_nom_k = nominal.x_nom[k]

        for obs in obstacles:
            # Only consider obstacles that affect this dimension
            if state_dim_idx >= len(obs.center):
                continue

            # Extract position
            x_pos = x_nom_k[: len(obs.center)]
            obs_center = obs.center

            diff = x_pos - obs_center
            dist = np.linalg.norm(diff)

            obs_radius_safe = obs.effective_radius + beta

            # Skip if nominal is inside obstacle (shouldn't happen for good planning)
            if dist < obs_radius_safe:
                continue

            if dist < 1e-6:
                continue

            # Linearized constraint: h(x) = obs_radius + beta - ||x - obs_center|| ≤ 0
            # This is a hyperplane perpendicular to (x_nom - obs_center)
            # For dimension i: constraint is approximately
            #   (x[i] - x_nom[i]) * (x_nom[i] - obs_center[i]) / dist ≤ clearance

            # Direction from obstacle to nominal
            direction = diff / dist

            # The constraint affects dimension i proportionally to direction[i]
            # Maximum deviation in dimension i while staying feasible:
            #   clearance = dist - obs_radius_safe
            #   deviation_i ≈ clearance / |direction[i]|  (conservative)

            clearance = dist - obs_radius_safe

            if abs(direction[state_dim_idx]) > 1e-6:
                # Bound in dimension i
                # If direction[i] > 0: upper bound is x_nom[i] + clearance / direction[i]
                # If direction[i] < 0: lower bound is x_nom[i] + clearance / direction[i]

                deviation_bound = clearance / abs(direction[state_dim_idx])

                if direction[state_dim_idx] > 0:
                    # Obstacle is "below" in this dimension
                    bound = x_nom_k[state_dim_idx] + deviation_bound
                    upper_bounds[k] = min(upper_bounds[k], bound)
                    constraint_active[k] = True
                else:
                    # Obstacle is "above" in this dimension
                    bound = x_nom_k[state_dim_idx] - deviation_bound
                    lower_bounds[k] = max(lower_bounds[k], bound)
                    constraint_active[k] = True

    return upper_bounds, lower_bounds, constraint_active


def plot_feasibility_verification_detailed(  # noqa: C901, PLR0912, PLR0915
    nominal,
    envelope_list,
    obstacles,
    workspace,
    constraints,  # noqa: ARG001
    beta: float,
    output_path: Optional[str] = None,
    state_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot detailed feasibility verification with all constraints.

    For each state dimension, shows:
    - Nominal trajectory (solid line)
    - Workspace bounds (red dashed horizontal lines)
    - Linearized obstacle bounds (orange shaded regions)
    - Ellipsoid-derived bounds (blue shaded region)

    This lets you verify visually that ellipsoids respect all constraints.

    Args:
        nominal: NominalTrajectory object
        envelope_list: List of FeasibilityEnvelope objects
        obstacles: List of Obstacle objects
        workspace: Workspace object with bounds
        constraints: SystemConstraints object
        beta: Safety margin used in MVIE
        output_path: Path to save figure (optional)
        state_labels: List of state labels (e.g., ['x', 'y', 'θ'])

    Returns:
        fig: Matplotlib figure
    """
    n = nominal.state_dim
    time = np.arange(nominal.N + 1) * nominal.dt

    # Create figure with subplots for each state
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Extract ellipsoid bounds for each timestep
    # We need to map envelope_list back to full trajectory
    ellipsoid_upper = np.full((n, nominal.N + 1), np.nan)
    ellipsoid_lower = np.full((n, nominal.N + 1), np.nan)

    for env in envelope_list:
        k_start = env.k_start
        k_end = env.k_end

        # Use per-timestep ellipsoids
        for idx, k in enumerate(range(k_start, min(k_end + 2, nominal.N + 1))):
            if idx < len(env.P_min_timestep):
                P_ell = env.P_min_timestep[idx]
                P = P_ell.P
                center = P_ell.c

                # For each dimension, bound is ±1/sqrt(P_ii)
                for i in range(min(n, P.shape[0])):
                    bound = 1.0 / np.sqrt(max(P[i, i], 1e-10))
                    ellipsoid_upper[i, k] = center[i] + bound
                    ellipsoid_lower[i, k] = center[i] - bound

    # Plot each state dimension
    for i in range(n):
        ax = axes[i]

        # --- 1. Workspace bounds (red dashed) ---
        if hasattr(workspace, "x_min") and hasattr(workspace, "x_max"):
            if i == 0:  # x dimension
                ax.axhline(
                    workspace.x_min,
                    color="#D62828",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.8,
                    label="Workspace bounds",
                    zorder=1,
                )
                ax.axhline(
                    workspace.x_max,
                    color="#D62828",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.8,
                    zorder=1,
                )
            elif i == 1 and hasattr(workspace, "y_min"):  # y dimension
                ax.axhline(
                    workspace.y_min,
                    color="#D62828",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.8,
                    label="Workspace bounds",
                    zorder=1,
                )
                ax.axhline(
                    workspace.y_max,
                    color="#D62828",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.8,
                    zorder=1,
                )
            elif i == 2 and hasattr(workspace, "z_min"):  # z dimension
                ax.axhline(
                    workspace.z_min,
                    color="#D62828",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.8,
                    label="Workspace bounds",
                    zorder=1,
                )
                ax.axhline(
                    workspace.z_max,
                    color="#D62828",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.8,
                    zorder=1,
                )

        # --- 2. Linearized obstacle bounds (orange shaded) ---
        if i < 2:  # Only for spatial dimensions (x, y)
            upper_obs, lower_obs, active = compute_linearized_obstacle_bounds(nominal, obstacles, beta, i)

            # Plot only where constraints are active
            first_active = True
            for k in range(nominal.N + 1):
                if active[k]:
                    if not np.isinf(upper_obs[k]):
                        # Draw a small region around this timestep
                        k_window = 5  # Show constraint for +/- 5 timesteps
                        t_start = max(0, k - k_window) * nominal.dt
                        t_end = min(nominal.N, k + k_window) * nominal.dt

                        # Upper constraint
                        if k == 0 or not active[k - 1] or abs(upper_obs[k] - upper_obs[k - 1]) > 0.1:
                            ax.fill_between(
                                [t_start, t_end],
                                [upper_obs[k], upper_obs[k]],
                                [ax.get_ylim()[1] if len(ax.get_ylim()) > 0 else 1000] * 2,
                                color="#F18F01",
                                alpha=0.15,
                                label="Obstacle constraint" if first_active else "",
                                zorder=2,
                            )
                            first_active = False

                    if not np.isinf(lower_obs[k]):
                        k_window = 5
                        t_start = max(0, k - k_window) * nominal.dt
                        t_end = min(nominal.N, k + k_window) * nominal.dt

                        # Lower constraint
                        if k == 0 or not active[k - 1] or abs(lower_obs[k] - lower_obs[k - 1]) > 0.1:
                            ax.fill_between(
                                [t_start, t_end],
                                [ax.get_ylim()[0] if len(ax.get_ylim()) > 0 else -1000] * 2,
                                [lower_obs[k], lower_obs[k]],
                                color="#F18F01",
                                alpha=0.15,
                                zorder=2,
                            )

        # --- 3. Ellipsoid-derived bounds (blue shaded) ---
        valid_mask = ~np.isnan(ellipsoid_upper[i, :])
        if np.any(valid_mask):
            ax.fill_between(
                time,
                ellipsoid_lower[i, :],
                ellipsoid_upper[i, :],
                where=valid_mask,
                color="#2E86AB",
                alpha=0.25,
                label="Ellipsoid bounds (P_min)",
                zorder=3,
            )

            # Plot boundary lines
            ax.plot(
                time,
                ellipsoid_upper[i, :],
                color="#2E86AB",
                linewidth=2,
                alpha=0.7,
                zorder=4,
            )
            ax.plot(
                time,
                ellipsoid_lower[i, :],
                color="#2E86AB",
                linewidth=2,
                alpha=0.7,
                zorder=4,
            )

        # --- 4. Nominal trajectory (thick black line) ---
        ax.plot(
            time,
            nominal.x_nom[:, i],
            color="black",
            linewidth=3,
            label="Nominal trajectory",
            zorder=5,
        )

        # --- Styling ---
        label = state_labels[i] if state_labels and i < len(state_labels) else f"$x_{i}$"
        ax.set_ylabel(label, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=1)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

        if i == 0:
            ax.set_title(
                "Feasibility Verification: Constraints and Ellipsoid Bounds vs Time",
                fontsize=15,
                fontweight="bold",
                pad=15,
            )

        if i == n - 1:
            ax.set_xlabel("Time (s)", fontsize=13, fontweight="bold")

        # Set y-limits with some padding
        y_data = nominal.x_nom[:, i]
        y_min, y_max = np.nanmin(y_data), np.nanmax(y_data)
        y_range = y_max - y_min
        padding = 0.3 * y_range if y_range > 0 else 1.0
        ax.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Feasibility verification plot saved to: {output_path}")

    return fig


def plot_feasibility_summary_table(  # noqa: C901, PLR0912, PLR0915
    nominal,
    envelope_list,
    obstacles,
    workspace,
    beta: float,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Create a summary table showing constraint violations (if any).

    Args:
        nominal: NominalTrajectory object
        envelope_list: List of FeasibilityEnvelope objects
        obstacles: List of Obstacle objects
        workspace: Workspace object
        beta: Safety margin
        output_path: Path to save figure (optional)

    Returns:
        fig: Matplotlib figure with table
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    # Check for violations
    violations = []
    nx = nominal.state_dim

    # Check workspace violations
    for env in envelope_list:
        for idx, k in enumerate(range(env.k_start, min(env.k_end + 2, nominal.N + 1))):
            if idx >= len(env.P_min_timestep):
                continue

            P_ell = env.P_min_timestep[idx]
            P = P_ell.P
            center = P_ell.c

            # Check x bounds using actual ellipsoid shape (not just diagonal)
            # For ellipsoid E(P) = {η | η^T P η ≤ 1}, check if any point violates workspace
            # We check the extreme point in the direction of the workspace boundary
            if hasattr(workspace, "x_min"):
                # Check lower bound: find minimum x in ellipsoid
                # This is: min_x = center[0] - sqrt(e_x^T P^{-1} e_x) where e_x = [1, 0, ...]
                try:
                    P_inv = np.linalg.inv(P)
                    e_x = np.zeros(nx)
                    e_x[0] = 1.0
                    max_deviation_x = np.sqrt(e_x.T @ P_inv @ e_x)
                    min_x = center[0] - max_deviation_x
                    if min_x < workspace.x_min - 1e-6:  # Small tolerance for numerical errors
                        violations.append(f"k={k}: x lower bound violation (min_x={min_x:.4f} < {workspace.x_min})")

                    # Check upper bound
                    max_x = center[0] + max_deviation_x
                    if max_x > workspace.x_max + 1e-6:
                        violations.append(f"k={k}: x upper bound violation (max_x={max_x:.4f} > {workspace.x_max})")
                except np.linalg.LinAlgError:
                    # Fallback to diagonal approximation
                    bound = 1.0 / np.sqrt(max(P[0, 0], 1e-10))
                    if center[0] - bound < workspace.x_min - 1e-6:
                        violations.append(f"k={k}: x lower bound violation (diagonal approx)")
                    if center[0] + bound > workspace.x_max + 1e-6:
                        violations.append(f"k={k}: x upper bound violation (diagonal approx)")

            # Check y bounds using actual ellipsoid shape
            if len(center) > 1 and hasattr(workspace, "y_min"):
                try:
                    P_inv = np.linalg.inv(P)
                    e_y = np.zeros(nx)
                    e_y[1] = 1.0
                    max_deviation_y = np.sqrt(e_y.T @ P_inv @ e_y)
                    min_y = center[1] - max_deviation_y
                    if min_y < workspace.y_min - 1e-6:
                        violations.append(f"k={k}: y lower bound violation (min_y={min_y:.4f} < {workspace.y_min})")

                    max_y = center[1] + max_deviation_y
                    if max_y > workspace.y_max + 1e-6:
                        violations.append(f"k={k}: y upper bound violation (max_y={max_y:.4f} > {workspace.y_max})")
                except np.linalg.LinAlgError:
                    # Fallback to diagonal approximation
                    bound = 1.0 / np.sqrt(max(P[1, 1], 1e-10))
                    if center[1] - bound < workspace.y_min - 1e-6:
                        violations.append(f"k={k}: y lower bound violation (diagonal approx)")
                    if center[1] + bound > workspace.y_max + 1e-6:
                        violations.append(f"k={k}: y upper bound violation (diagonal approx)")

    # Check obstacle violations using actual ellipsoid shape
    # For ellipsoid E(P), check if any point in ellipsoid violates obstacle constraint
    for env in envelope_list:
        for idx, k in enumerate(range(env.k_start, min(env.k_end + 2, nominal.N + 1))):
            if idx >= len(env.P_min_timestep):
                continue

            P_ell = env.P_min_timestep[idx]
            P = P_ell.P
            center = P_ell.c

            for obs in obstacles:
                obs_center = obs.center
                obs_radius_safe = obs.effective_radius + beta

                # Find the point in the ellipsoid closest to the obstacle
                # This is the point that minimizes ||x - obs_center|| subject to (x - center)^T P (x - center) ≤ 1
                # Direction from center to obstacle
                diff = center[: len(obs_center)] - obs_center
                dist_center_to_obs = np.linalg.norm(diff)

                if dist_center_to_obs < 1e-6:
                    violations.append(f"k={k}: Center at obstacle {obs.id} center")
                    continue

                # Direction vector from obstacle to center
                direction = diff / dist_center_to_obs

                # Project ellipsoid onto this direction
                # The extreme point in direction of obstacle is at: center - direction * max_deviation
                try:
                    # Extract 2D submatrix for spatial dimensions
                    P_2d = P[: len(obs_center), : len(obs_center)]
                    P_inv_2d = np.linalg.inv(P_2d)

                    # Maximum deviation in direction toward obstacle
                    max_deviation = np.sqrt(direction.T @ P_inv_2d @ direction)

                    # Closest point to obstacle (point in ellipsoid closest to obstacle center)
                    closest_point = center[: len(obs_center)] - direction * max_deviation
                    dist_closest = np.linalg.norm(closest_point - obs_center)

                    if dist_closest < obs_radius_safe - 1e-6:
                        violations.append(
                            f"k={k}: Obstacle {obs.id} violation "
                            f"(closest_dist={dist_closest:.4f} < safe_radius={obs_radius_safe:.4f})"
                        )
                except np.linalg.LinAlgError:
                    # Fallback: check if center is too close
                    if dist_center_to_obs < obs_radius_safe * 1.2:
                        violations.append(f"k={k}: Close to obstacle {obs.id} (dist={dist_center_to_obs:.2f})")

    # Create summary text
    summary_text = "FEASIBILITY VERIFICATION SUMMARY\n"
    summary_text += "=" * 60 + "\n\n"
    summary_text += f"Total timesteps: {nominal.N + 1}\n"
    summary_text += f"Total segments: {len(envelope_list)}\n"
    summary_text += f"Safety margin (β): {beta}\n"
    summary_text += f"Obstacles: {len(obstacles)}\n\n"

    if len(violations) == 0:
        summary_text += "✓ NO VIOLATIONS DETECTED\n\n"
        summary_text += "All ellipsoid bounds respect:\n"
        summary_text += "  • Workspace constraints\n"
        summary_text += "  • Obstacle avoidance (with safety margin)\n"
        summary_text += "  • State bounds\n"
        color = "lightgreen"
    else:
        summary_text += f"⚠ {len(violations)} POTENTIAL VIOLATIONS\n\n"
        summary_text += "Violations:\n"
        for v in violations[:20]:  # Show first 20
            summary_text += f"  • {v}\n"
        if len(violations) > 20:
            summary_text += f"  ... and {len(violations) - 20} more\n"
        color = "lightyellow"

    ax.text(
        0.5,
        0.5,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        horizontalalignment="center",
        bbox={"boxstyle": "round", "facecolor": color, "alpha": 0.8, "pad": 20},
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Feasibility summary saved to: {output_path}")

    return fig
