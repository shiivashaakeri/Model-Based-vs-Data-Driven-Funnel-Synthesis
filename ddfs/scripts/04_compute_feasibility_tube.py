#!/usr/bin/env python3
"""
Phase 6a: Compute Feasibility Tube

This script computes the feasibility tube (P_min(k) and R_max(k)) around
the nominal trajectory, accounting for state/input constraints and obstacles.

The computed ellipsoid envelopes define the feasible region for funnel synthesis
and ensure that trajectories remain within bounds and avoid obstacles.

Usage:
    python scripts/04_compute_feasibility_tube.py [nominal_traj.pkl] [segments.pkl]

Output:
    - feasibility_constraints.pkl: P_min(k), R_max(k), P_min,i, R_max,i
    - feasibility_tube.png/pdf: Visualization of the tube
    - feasibility_analysis.png: Eigenvalue analysis
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import cm
from matplotlib.patches import Circle

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddfs.synthesis.ellipsoid_utils import EllipsoidConstraints, EllipsoidUtility
from ddfs.utils.config_loader import ConfigLoader


def load_trajectory(filepath: Path) -> dict:
    """Load nominal trajectory from Phase 3."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def load_segments(filepath: Path) -> list:
    """Load segments from Phase 4 offline data."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    # Extract segments from dataset
    if "segmented_data" in data:
        return data["segmented_data"]
    elif "segments" in data:
        return data["segments"]
    else:
        raise ValueError("No segments found in data file. Available keys: " + ", ".join(data.keys()))


def load_obstacles_from_env_config(config_path: Path) -> list:
    """Load obstacles from environment configuration."""
    if not config_path.exists():
        print(f"   ⚠️  Environment config not found: {config_path}")
        return []

    with open(config_path, "r") as f:
        env_config = yaml.safe_load(f)

    obstacles = []

    if "obstacles" in env_config:
        for obs_dict in env_config["obstacles"]:
            obs_type = obs_dict.get("type", "circle")

            if obs_type == "circle":
                obstacles.append(
                    {"type": "circle", "center": np.array(obs_dict["center"]), "radius": obs_dict["radius"]}
                )
            elif obs_type == "ellipse":
                obstacles.append(
                    {
                        "type": "ellipse",
                        "center": np.array(obs_dict["center"]),
                        "semi_axes": np.array(obs_dict["semi_axes"]),
                        "rotation": obs_dict.get("rotation", 0.0),
                    }
                )

    return obstacles


def visualize_feasibility_analysis(constraints: EllipsoidConstraints, trajectory: dict, output_dir: Path):  # noqa: PLR0915, ARG001
    """
    Create detailed analysis plots of feasibility constraints.

    Args:
        constraints: Computed ellipsoid constraints
        trajectory: Nominal trajectory
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. P_min eigenvalues over time
    ax1 = axes[0, 0]
    P_eigs = np.array([np.linalg.eigvalsh(P) for P in constraints.P_min_k])
    for i in range(3):
        ax1.plot(P_eigs[:, i], label=f"λ_{i + 1}", linewidth=2)
    ax1.set_xlabel("Timestep k")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("P_min(k) Eigenvalues (State Feasibility)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. R_max eigenvalues over time
    ax2 = axes[0, 1]
    R_eigs = np.array([np.linalg.eigvalsh(R) for R in constraints.R_max_k])
    for i in range(2):
        ax2.plot(R_eigs[:, i], label=f"λ_{i + 1}", linewidth=2)
    ax2.set_xlabel("Timestep k")
    ax2.set_ylabel("Eigenvalue")
    ax2.set_title("R_max(k) Eigenvalues (Input Feasibility)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Ellipsoid volumes over time
    ax3 = axes[1, 0]
    P_vols = [1.0 / np.sqrt(np.linalg.det(P)) for P in constraints.P_min_k]
    R_vols = [np.sqrt(np.linalg.det(R)) for R in constraints.R_max_k]

    ax3_twin = ax3.twinx()
    ax3.plot(P_vols, "b-", label="State ellipsoid volume", linewidth=2)
    ax3_twin.plot(R_vols, "r-", label="Input ellipsoid volume", linewidth=2)
    ax3.set_xlabel("Timestep k")
    ax3.set_ylabel("State Volume", color="b")
    ax3_twin.set_ylabel("Input Volume", color="r")
    ax3.set_title("Ellipsoid Volumes Along Trajectory")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")

    # 4. Per-segment envelope comparison
    ax4 = axes[1, 1]
    n_segments = len(constraints.P_min_i)
    segment_ids = np.arange(n_segments)

    # Average eigenvalues per segment
    P_seg_eigs = [np.mean(np.linalg.eigvalsh(P)) for P in constraints.P_min_i]
    R_seg_eigs = [np.mean(np.linalg.eigvalsh(R)) for R in constraints.R_max_i]

    ax4_twin = ax4.twinx()
    ax4.bar(segment_ids - 0.2, P_seg_eigs, width=0.4, label="P_min,i (avg)", color="b", alpha=0.7)
    ax4_twin.bar(segment_ids + 0.2, R_seg_eigs, width=0.4, label="R_max,i (avg)", color="r", alpha=0.7)
    ax4.set_xlabel("Segment Index")
    ax4.set_ylabel("P_min,i Eigenvalue", color="b")
    ax4_twin.set_ylabel("R_max,i Eigenvalue", color="r")
    ax4.set_title("Per-Segment Envelope Tightness")
    ax4.legend(loc="upper left")
    ax4_twin.legend(loc="upper right")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    output_path = output_dir / "feasibility_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved analysis plot: {output_path}")

    output_path_pdf = output_dir / "feasibility_analysis.pdf"
    plt.savefig(output_path_pdf, bbox_inches="tight")

    plt.close()


def visualize_feasibility_tube_detailed(  # noqa: C901
    trajectory: dict, constraints: EllipsoidConstraints, obstacles: list, config: dict, output_dir: Path
):
    """
    Create detailed visualization of the feasibility tube.

    Args:
        trajectory: Nominal trajectory
        constraints: Computed constraints
        obstacles: List of obstacles
        config: Visualization config
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=tuple(config["figsize"]))

    X = trajectory["x_traj"]
    N = X.shape[0]

    # Plot nominal trajectory
    ax.plot(X[:, 0], X[:, 1], "k-", linewidth=3, label="Nominal trajectory", zorder=10)
    ax.plot(X[0, 0], X[0, 1], "go", markersize=12, label="Start", zorder=11)
    ax.plot(X[-1, 0], X[-1, 1], "r^", markersize=12, label="Goal", zorder=11)

    # Plot obstacles
    if obstacles:
        for i, obs in enumerate(obstacles):
            if obs["type"] == "circle":
                circle = Circle(
                    obs["center"],
                    obs["radius"],
                    fill=True,
                    facecolor="red",
                    alpha=0.4,
                    edgecolor="darkred",
                    linewidth=2,
                    label="Obstacle" if i == 0 else "",
                )
                ax.add_patch(circle)

    # Plot P_min(k) ellipsoids at selected timesteps
    n_ellipses = config["n_ellipses"]
    alpha = config["alpha"]
    step = max(1, N // n_ellipses)
    timesteps = range(0, N, step)

    util = EllipsoidUtility()

    if config["segment_colors"] and "segment_times" in constraints.metadata:
        # Color by segment

        colors = cm.rainbow(np.linspace(0, 1, len(constraints.segment_times)))

        for k in timesteps:
            # Find which segment this timestep belongs to
            segment_idx = 0
            for i, seg_times in enumerate(constraints.segment_times):
                if k in seg_times:
                    segment_idx = i
                    break

            P_min_k = constraints.P_min_k[k]
            util._plot_ellipsoid_2d(
                P_min_k,
                center=X[k],
                ax=ax,
                fill=True,
                facecolor=colors[segment_idx],
                alpha=alpha,
                edgecolor=colors[segment_idx],
                linewidth=1,
            )
    else:
        # Single color
        for k in timesteps:
            P_min_k = constraints.P_min_k[k]
            util._plot_ellipsoid_2d(
                P_min_k, center=X[k], ax=ax, fill=True, facecolor="blue", alpha=alpha, edgecolor="blue", linewidth=0.5
            )

    # Plot segment boundaries
    if config["show_segments"]:
        for seg_times in constraints.segment_times[:-1]:
            k_end = seg_times[-1]
            ax.plot([X[k_end, 0]], [X[k_end, 1]], "mo", markersize=6, zorder=9)

    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("y (m)", fontsize=14)
    ax.set_title("Feasibility Tube: P_min(k) Ellipsoids with Obstacle Avoidance", fontsize=16)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    plt.tight_layout()

    # Save
    formats = config.get("formats", ["png", "pdf"])
    for fmt in formats:
        output_path = output_dir / f"feasibility_tube.{fmt}"
        plt.savefig(output_path, dpi=config.get("dpi", 150), bbox_inches="tight")
        print(f"Saved tube visualization: {output_path}")

    plt.close()


def visualize_obstacle_clearance(  # noqa: PLR0912, C901, PLR0915
    trajectory: dict, constraints: EllipsoidConstraints, obstacles: list, output_dir: Path
):
    """
    Create detailed obstacle clearance visualization.

    Shows:
    1. Full trajectory with obstacles and sparse ellipsoids
    2. Zoomed view near obstacles with dense ellipsoids
    3. Distance measurements to verify clearance
    """

    X = trajectory["x_traj"]
    N = X.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- LEFT PLOT: Full trajectory overview ---
    ax_full = axes[0]

    # Plot trajectory
    ax_full.plot(X[:, 0], X[:, 1], "k-", linewidth=2, label="Nominal trajectory", zorder=10)
    ax_full.plot(X[0, 0], X[0, 1], "go", markersize=10, label="Start", zorder=11)
    ax_full.plot(X[-1, 0], X[-1, 1], "r^", markersize=10, label="Goal", zorder=11)

    # Plot obstacles with safety margins
    if obstacles:
        for i, obs in enumerate(obstacles):
            if obs["type"] == "circle":
                # Inner obstacle (actual)
                circle_inner = Circle(
                    obs["center"],
                    obs["radius"],
                    fill=True,
                    facecolor="red",
                    alpha=0.6,
                    edgecolor="darkred",
                    linewidth=2,
                    label="Obstacle" if i == 0 else "",
                )
                ax_full.add_patch(circle_inner)

                # Safety margin boundary
                safety = obs.get("safety_margin", 0.2)
                circle_safety = Circle(
                    obs["center"],
                    obs["radius"] + safety,
                    fill=False,
                    edgecolor="orange",
                    linewidth=2,
                    linestyle="--",
                    label="Safety margin" if i == 0 else "",
                )
                ax_full.add_patch(circle_safety)

                # Label obstacle
                ax_full.text(
                    obs["center"][0],
                    obs["center"][1],
                    f"Obs {i + 1}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                )

    # Sample ellipsoids (not too many - clearer view)
    util = EllipsoidUtility()
    step = max(1, N // 15)
    for k in range(0, N, step):
        P_min_k = constraints.P_min_k[k]
        util._plot_ellipsoid_2d(
            P_min_k,
            center=X[k],
            ax=ax_full,
            fill=True,
            facecolor="blue",
            alpha=0.15,
            edgecolor="blue",
            linewidth=1,
        )

    ax_full.set_xlabel("x (m)", fontsize=12)
    ax_full.set_ylabel("y (m)", fontsize=12)
    ax_full.set_title("Full Trajectory with Feasibility Ellipsoids", fontsize=14, fontweight="bold")
    ax_full.legend(fontsize=10, loc="best")
    ax_full.grid(True, alpha=0.3)
    ax_full.axis("equal")

    # --- RIGHT PLOT: Zoomed view near obstacles ---
    ax_zoom = axes[1]

    if obstacles:
        # Find the region containing obstacles
        obs_centers = [obs["center"] for obs in obstacles if obs["type"] == "circle"]
        if obs_centers:
            obs_x = [c[0] for c in obs_centers]
            obs_y = [c[1] for c in obs_centers]

            # Define zoom region with margin
            margin = 2.0
            x_min, x_max = min(obs_x) - margin, max(obs_x) + margin
            y_min, y_max = min(obs_y) - margin, max(obs_y) + margin

            # Find trajectory points in this region
            in_region = [k for k in range(N) if x_min <= X[k, 0] <= x_max and y_min <= X[k, 1] <= y_max]

            if in_region:
                k_min, k_max = min(in_region), max(in_region)

                # Plot trajectory segment
                ax_zoom.plot(
                    X[k_min : k_max + 1, 0],
                    X[k_min : k_max + 1, 1],
                    "k-",
                    linewidth=3,
                    label="Nominal",
                    zorder=10,
                )

                # Mark waypoints
                waypoint_step = max(1, (k_max - k_min) // 10)
                for k in range(k_min, k_max + 1, waypoint_step):
                    ax_zoom.plot(X[k, 0], X[k, 1], "ko", markersize=4, zorder=9)

                # Plot obstacles in detail
                for i, obs in enumerate(obstacles):
                    if obs["type"] == "circle":
                        # Actual obstacle
                        circle_inner = Circle(
                            obs["center"],
                            obs["radius"],
                            fill=True,
                            facecolor="red",
                            alpha=0.7,
                            edgecolor="darkred",
                            linewidth=3,
                            label=f"Obs {i + 1}",
                        )
                        ax_zoom.add_patch(circle_inner)

                        # Safety margin
                        safety = obs.get("safety_margin", 0.2)
                        circle_safety = Circle(
                            obs["center"],
                            obs["radius"] + safety,
                            fill=False,
                            edgecolor="orange",
                            linewidth=3,
                            linestyle="--",
                            label=f"Safety (Obs {i + 1})",
                        )
                        ax_zoom.add_patch(circle_safety)

                # Plot ellipsoids densely in this region
                step_zoom = max(1, (k_max - k_min) // 20)
                for k in range(k_min, k_max + 1, step_zoom):
                    P_min_k = constraints.P_min_k[k]
                    util._plot_ellipsoid_2d(
                        P_min_k,
                        center=X[k],
                        ax=ax_zoom,
                        fill=True,
                        facecolor="cyan",
                        alpha=0.3,
                        edgecolor="blue",
                        linewidth=2,
                    )

                # Compute and display minimum clearances
                for i, obs in enumerate(obstacles):
                    if obs["type"] == "circle":
                        center = obs["center"]
                        radius = obs["radius"]
                        safety = obs.get("safety_margin", 0.2)

                        distances_clearance = []
                        for k in range(k_min, k_max + 1):
                            dist = np.linalg.norm(X[k, :2] - center)
                            distances_clearance.append(dist - radius)

                        min_clearance = min(distances_clearance)
                        k_closest = k_min + int(np.argmin(distances_clearance))

                        violates_safety = min_clearance < safety

                        # Draw line from obstacle center to closest trajectory point
                        line_color = "red" if violates_safety else "green"
                        line_style = "-" if violates_safety else "--"
                        ax_zoom.plot(
                            [center[0], X[k_closest, 0]],
                            [center[1], X[k_closest, 1]],
                            color=line_color,
                            linestyle=line_style,
                            linewidth=2,
                            alpha=0.7,
                            zorder=8,
                        )

                        # Annotate clearance distance
                        mid_x = (center[0] + X[k_closest, 0]) / 2
                        mid_y = (center[1] + X[k_closest, 1]) / 2

                        if min_clearance < safety:
                            bg_color = "red"
                            text_color = "white"
                            status = "⚠ VIOLATION!"
                        elif min_clearance < safety + 0.2:
                            bg_color = "yellow"
                            text_color = "black"
                            status = f"{min_clearance:.2f}m (close)"
                        else:
                            bg_color = "lightgreen"
                            text_color = "black"
                            status = f"{min_clearance:.2f}m ✓"

                        ax_zoom.text(
                            mid_x,
                            mid_y,
                            status,
                            fontsize=10,
                            fontweight="bold",
                            color=text_color,
                            bbox={"boxstyle": "round,pad=0.4", "facecolor": bg_color, "alpha": 0.9},
                        )

                        # Mark closest point
                        ax_zoom.plot(X[k_closest, 0], X[k_closest, 1], "r*", markersize=15, zorder=12)

                ax_zoom.set_xlim(x_min, x_max)
                ax_zoom.set_ylim(y_min, y_max)
                ax_zoom.legend(fontsize=9, loc="best")
            else:
                ax_zoom.text(
                    0.5,
                    0.5,
                    "No trajectory points\nnear obstacles",
                    ha="center",
                    va="center",
                    transform=ax_zoom.transAxes,
                    fontsize=14,
                )

    ax_zoom.set_xlabel("x (m)", fontsize=12)
    ax_zoom.set_ylabel("y (m)", fontsize=12)
    ax_zoom.set_title("Zoomed: Obstacle Clearance Verification", fontsize=14, fontweight="bold")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.axis("equal")

    plt.tight_layout()

    # Save
    output_path = output_dir / "obstacle_clearance.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved obstacle clearance plot: {output_path}")

    output_path_pdf = output_dir / "obstacle_clearance.pdf"
    plt.savefig(output_path_pdf, bbox_inches="tight")

    plt.close()


def main():  # noqa: C901, PLR0912, PLR0915
    """Main function for feasibility tube computation."""
    parser = argparse.ArgumentParser(description="Compute feasibility tube for DDFS")
    parser.add_argument("nominal_traj", nargs="?", type=str, help="Path to nominal trajectory pickle file")
    parser.add_argument("segments", nargs="?", type=str, help="Path to segments/offline data pickle file")
    parser.add_argument(
        "--config", type=str, default="config/feasibility_config.yaml", help="Path to feasibility config file"
    )
    parser.add_argument("--output-dir", type=str, default="data/feasibility", help="Output directory for results")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PHASE 6a: FEASIBILITY TUBE COMPUTATION")
    print("=" * 70)

    # 1. Load configurations
    print("\n Step 1: Loading configurations...")

    config_loader = ConfigLoader()

    # Load feasibility config
    feasibility_config_path = Path(args.config)
    if feasibility_config_path.exists():
        # Extract config name (stem) without .yaml extension for config_loader
        config_name = feasibility_config_path.stem
        feasibility_config = config_loader.load(config_name)
    else:
        print(f"    Config not found at {args.config}, using defaults")
        feasibility_config = {
            "ellipsoid_params": {"x_max": 10.0, "u_max": 2.0, "solver": "ECOS", "safety_margin": 0.2},
            "visualization": {
                "n_ellipses": 30,
                "alpha": 0.2,
                "figsize": [14, 12],
                "dpi": 150,
                "show_segments": True,
                "segment_colors": True,
                "formats": ["png", "pdf"],
            },
            "output": {"save_plots": True, "save_data": True},
        }

    # 2. Load data files
    print("\n Step 2: Loading data files...")

    # Find nominal trajectory
    if args.nominal_traj:
        nominal_traj_path = Path(args.nominal_traj)
    else:
        nominal_dir = Path("data/nominal_trajectories")
        if nominal_dir.exists():
            # Search recursively in subdirectories
            nominal_files = sorted(nominal_dir.glob("**/*.pkl"))
            if nominal_files:
                nominal_traj_path = nominal_files[-1]
                print(f"   Using most recent nominal: {nominal_traj_path.name}")
            else:
                print("    No nominal trajectory files found!")
                return 1
        else:
            print("    Nominal trajectories directory not found!")
            return 1

    # Find segments
    if args.segments:
        segments_path = Path(args.segments)
    else:
        data_dir = Path("data/offline_datasets")
        if data_dir.exists():
            # Search recursively in subdirectories
            data_files = sorted(data_dir.glob("**/*.pkl"))
            if data_files:
                segments_path = data_files[-1]
                print(f"   Using most recent data: {segments_path.name}")
            else:
                print("    No segment data files found!")
                return 1
        else:
            print("    Offline data directory not found!")
            return 1

    # Load data
    print(f"   Loading nominal trajectory: {nominal_traj_path}")
    trajectory = load_trajectory(nominal_traj_path)

    print(f"   Loading segments: {segments_path}")
    segments = load_segments(segments_path)

    print(f"   Loaded trajectory: {trajectory['x_traj'].shape[0]} timesteps")
    print(f"   Loaded segments: {len(segments)} segments")

    # 3. Load constraints and obstacles
    print("\n Step 3: Loading constraints and obstacles...")

    state_constraints = feasibility_config.get("state_constraints", {})
    input_constraints = feasibility_config.get("input_constraints", {})

    print(f"   State bounds: {state_constraints.get('box', {}).get('x_min', 'N/A')}")
    print(f"                 {state_constraints.get('box', {}).get('x_max', 'N/A')}")
    print(f"   Input bounds: {input_constraints.get('box', {}).get('u_min', 'N/A')}")
    print(f"                 {input_constraints.get('box', {}).get('u_max', 'N/A')}")

    # Load obstacles from environment config
    env_config_path = Path("config/environment.yaml")
    obstacles = load_obstacles_from_env_config(env_config_path)

    # Add any extra obstacles from feasibility config
    if feasibility_config.get("obstacles"):
        obstacles.extend(feasibility_config["obstacles"])

    print(f"   Loaded {len(obstacles)} obstacles")
    for i, obs in enumerate(obstacles):
        if obs["type"] == "circle":
            print(f"      {i + 1}. Circle at {obs['center']}, radius={obs['radius']}")
        elif obs["type"] == "ellipse":
            print(f"      {i + 1}. Ellipse at {obs['center']}, axes={obs['semi_axes']}")

    # 4. Initialize ellipsoid utility
    print("\n Step 4: Initializing ellipsoid utility...")

    ellipsoid_params = feasibility_config.get("ellipsoid_params", {})
    util = EllipsoidUtility(
        n_states=3,
        n_controls=2,
        x_max=ellipsoid_params.get("x_max", 10.0),
        u_max=ellipsoid_params.get("u_max", 2.0),
        solver=ellipsoid_params.get("solver", "ECOS"),
        safety_margin=ellipsoid_params.get("safety_margin", 0.2),
    )

    print("   Initialized with:")
    print(f"      - x_max: {ellipsoid_params.get('x_max', 10.0)}")
    print(f"      - u_max: {ellipsoid_params.get('u_max', 2.0)}")
    print(f"      - safety_margin: {ellipsoid_params.get('safety_margin', 0.2)} m")
    print(f"      - solver: {ellipsoid_params.get('solver', 'ECOS')}")

    # 5. Compute feasibility envelopes
    print("\n Step 5: Computing feasibility envelopes...")
    print("   This may take 30-60 seconds...")

    constraints = util.compute_all_envelopes(
        trajectory=trajectory,
        segments=segments,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        obstacles=obstacles,
        verbose=True,
    )

    # 6. Save results
    print("\n Step 6: Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save constraints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    constraints_file = output_dir / f"feasibility_constraints_{timestamp}.pkl"
    constraints.save(constraints_file)

    # Also save as "latest"
    latest_file = output_dir / "feasibility_constraints_latest.pkl"
    constraints.save(latest_file)

    # 7. Visualizations
    if feasibility_config.get("output", {}).get("save_plots", True):
        print("\n Step 7: Creating visualizations...")

        # Analysis plots
        visualize_feasibility_analysis(constraints, trajectory, output_dir)

        # Tube visualization
        vis_config = feasibility_config.get("visualization", {}).copy()
        # Add formats from output config if not in visualization config
        if "formats" not in vis_config:
            vis_config["formats"] = feasibility_config.get("output", {}).get("formats", ["png", "pdf"])
        visualize_feasibility_tube_detailed(trajectory, constraints, obstacles, vis_config, output_dir)
        visualize_obstacle_clearance(trajectory, constraints, obstacles, output_dir)

    # 8. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Trajectory: {trajectory['x_traj'].shape[0]} timesteps")
    print(f"Segments:   {len(segments)} segments")
    print(f"Obstacles:  {len(obstacles)} obstacles")
    print()

    # Analyze P_min eigenvalues
    P_eigs_all = [np.linalg.eigvalsh(P) for P in constraints.P_min_k]
    P_eigs_min = np.min([np.min(eigs) for eigs in P_eigs_all])
    P_eigs_max = np.max([np.max(eigs) for eigs in P_eigs_all])

    print("P_min(k) eigenvalues:")
    print(f"  Range: [{P_eigs_min:.4f}, {P_eigs_max:.4f}]")
    print(f"  Mean:  {np.mean([np.mean(eigs) for eigs in P_eigs_all]):.4f}")
    print()

    # Analyze R_max eigenvalues
    R_eigs_all = [np.linalg.eigvalsh(R) for R in constraints.R_max_k]
    R_eigs_min = np.min([np.min(eigs) for eigs in R_eigs_all])
    R_eigs_max = np.max([np.max(eigs) for eigs in R_eigs_all])

    print("R_max(k) eigenvalues:")
    print(f"  Range: [{R_eigs_min:.4f}, {R_eigs_max:.4f}]")
    print(f"  Mean:  {np.mean([np.mean(eigs) for eigs in R_eigs_all]):.4f}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
