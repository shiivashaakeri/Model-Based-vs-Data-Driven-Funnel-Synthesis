"""
Offline data collection for DDFS.

This script:
1. Loads nominal trajectory from Phase 3
2. Creates plant model with mismatch
3. Collects trajectories from multiple initial states with excitation
4. Segments trajectories into time windows
5. Builds Hankel matrices per segment
6. Checks informativity (persistence of excitation)
7. Saves dataset and generates visualizations

Usage:
    python scripts/02_collect_offline_data.py [nominal_trajectory_file]

    If no file is specified, loads the most recent nominal trajectory.
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ddfs.data.collector import OfflineDataCollector  # noqa: E402
from ddfs.data.hankel import HankelMatrixBuilder  # noqa: E402
from ddfs.data.informativity import InformativityChecker  # noqa: E402
from ddfs.data.segmenter import TrajectorySegmenter  # noqa: E402
from ddfs.environment.collision import CollisionChecker  # noqa: E402
from ddfs.environment.obstacles import CircularObstacle, EllipsoidalObstacle  # noqa: E402
from ddfs.models.plant import PlantModel  # noqa: E402
from ddfs.models.unicycle import UnicycleModel  # noqa: E402
from ddfs.utils.config_loader import ConfigLoader  # noqa: E402


def load_nominal_trajectory(filepath=None):
    """Load nominal trajectory from Phase 3."""
    if filepath is None:
        # Find most recent trajectory
        traj_dir = Path("data/nominal_trajectories")
        if not traj_dir.exists():
            raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

        folders = [f for f in traj_dir.iterdir() if f.is_dir() and f.name.startswith("unicycle_nominal_")]

        if len(folders) == 0:
            raise FileNotFoundError(f"No nominal trajectories found in {traj_dir}")

        folder_path = max(folders, key=lambda p: p.stat().st_mtime)
        pkl_files = list(folder_path.glob("*.pkl"))

        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No .pkl file found in {folder_path}")

        filepath = pkl_files[0]

    else:
        filepath = Path(filepath)

    print(f"📂 Loading nominal trajectory: {filepath.name}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data


def create_obstacles_from_config(obstacle_configs):
    """Create obstacle objects from configuration."""
    obstacles = []

    for obs_config in obstacle_configs:
        obs_type = obs_config["type"]

        if obs_type == "circle":
            obs = CircularObstacle(
                center=np.array(obs_config["center"]),
                radius=obs_config["radius"],
                safety_margin=obs_config.get("safety_margin", 0.0),
            )
            obstacles.append(obs)

        elif obs_type == "ellipse":
            obs = EllipsoidalObstacle(
                center=np.array(obs_config["center"]),
                semi_axes=np.array(obs_config["semi_axes"]),
                rotation=obs_config.get("rotation", 0.0),
                safety_margin=obs_config.get("safety_margin", 0.0),
            )
            obstacles.append(obs)

    return obstacles


def validate_trajectories(dataset, nominal_data, config, collision_checker):  # noqa: PLR0912, C901, PLR0915
    """Validate collected trajectories."""
    print("\n" + "=" * 70)
    print("TRAJECTORY VALIDATION")
    print("=" * 70)

    x_nom = nominal_data["x_traj"]  # noqa: F841
    n_trajectories = len(dataset)

    issues = []

    # Check for collisions
    if config.get("validation", {}).get("check_collision", True):
        print("\n1. Checking for collisions...")
        n_collisions = 0

        for i, traj_data in enumerate(dataset):
            x_traj = traj_data["x"]
            collision, timestep, obs_idx = collision_checker.check_trajectory_collision(x_traj)

            if collision:
                n_collisions += 1
                issues.append(f"Trajectory {i}: collision at timestep {timestep} with obstacle {obs_idx}")

        if n_collisions > 0:
            print(f"  ⚠️  {n_collisions}/{n_trajectories} trajectories have collisions")
        else:
            print(f"  ✓ No collisions detected in {n_trajectories} trajectories")

    # Check maximum deviation
    max_deviation = config.get("validation", {}).get("max_deviation", 1.0)
    print(f"\n2. Checking maximum deviation (limit: {max_deviation}m)...")

    max_dev_observed = 0.0
    n_exceeded = 0

    for i, traj_data in enumerate(dataset):
        eta = traj_data["eta"]
        max_eta = np.max(np.linalg.norm(eta, axis=1))
        max_dev_observed = max(max_dev_observed, max_eta)

        if max_eta > max_deviation:
            n_exceeded += 1
            issues.append(f"Trajectory {i}: max deviation {max_eta:.4f}m exceeds limit")

    print(f"  Max deviation observed: {max_dev_observed:.4f}m")
    if n_exceeded > 0:
        print(f"  ⚠️  {n_exceeded}/{n_trajectories} trajectories exceed deviation limit")
    else:
        print("  ✓ All trajectories within deviation limit")

    # Check constraints
    if config.get("validation", {}).get("check_constraints", True):
        print("\n3. Checking state/input constraints...")

        x_min, x_max = nominal_data["x_bounds"]
        u_min, u_max = nominal_data["u_bounds"]

        n_state_violations = 0
        n_input_violations = 0

        for i, traj_data in enumerate(dataset):
            x_traj = traj_data["x"]
            u_traj = traj_data["u"]

            # Check state constraints
            if np.any(x_traj < x_min) or np.any(x_traj > x_max):
                n_state_violations += 1

            # Check input constraints
            if np.any(u_traj < u_min) or np.any(u_traj > u_max):
                n_input_violations += 1

        if n_state_violations > 0:
            print(f"  ⚠️  {n_state_violations}/{n_trajectories} trajectories violate state constraints")
        else:
            print("  ✓ All trajectories satisfy state constraints")

        if n_input_violations > 0:
            print(f"  ⚠️  {n_input_violations}/{n_trajectories} trajectories violate input constraints")
        else:
            print("  ✓ All trajectories satisfy input constraints")

    print("\n" + "=" * 70)

    if len(issues) == 0:
        print("✓ VALIDATION PASSED")
    else:
        print(f"⚠️  VALIDATION WARNINGS ({len(issues)} issues)")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")

    print("=" * 70)

    return len(issues) == 0, issues


def plot_trajectories(dataset, nominal_data, obstacles, output_dir):
    """Plot collected trajectories in workspace."""
    print("\n📊 Generating trajectory plots...")

    x_nom = nominal_data["x_traj"]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot workspace bounds
    ws_bounds = nominal_data["workspace_bounds"]
    ax.set_xlim(ws_bounds[0][0] - 0.5, ws_bounds[1][0] + 0.5)
    ax.set_ylim(ws_bounds[0][1] - 0.5, ws_bounds[1][1] + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot obstacles
    for obs in obstacles:
        if hasattr(obs, "plot"):
            obs.plot(ax, facecolor="red", alpha=0.3, edgecolor="darkred", linewidth=2)

    # Plot collected trajectories
    for i, traj_data in enumerate(dataset):
        x_traj = traj_data["x"]
        alpha = 0.3 if len(dataset) > 10 else 0.5
        ax.plot(x_traj[:, 0], x_traj[:, 1], "b-", alpha=alpha, linewidth=1, label="Collected" if i == 0 else "")
        ax.plot(x_traj[0, 0], x_traj[0, 1], "bo", markersize=4, alpha=0.5)

    # Plot nominal trajectory (on top)
    ax.plot(x_nom[:, 0], x_nom[:, 1], "g-", linewidth=3, label="Nominal", zorder=10)
    ax.plot(x_nom[0, 0], x_nom[0, 1], "go", markersize=12, label="Start", zorder=11)
    ax.plot(x_nom[-1, 0], x_nom[-1, 1], "r*", markersize=20, label="Goal", zorder=11)

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title(f"Offline Data Collection: {len(dataset)} Trajectories", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)

    # Save
    output_path = output_dir / "collected_trajectories.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved: {output_path}")

    plt.close()


def plot_deviation_statistics(dataset, output_dir):
    """Plot deviation statistics over time."""
    print("\n📊 Generating deviation statistics plots...")

    N = dataset[0]["eta"].shape[0] - 1
    n = dataset[0]["eta"].shape[1]  # noqa: F841

    # Compute statistics
    all_eta_norms = np.zeros((len(dataset), N + 1))
    for i, traj_data in enumerate(dataset):
        all_eta_norms[i, :] = np.linalg.norm(traj_data["eta"], axis=1)

    mean_eta = np.mean(all_eta_norms, axis=0)
    std_eta = np.std(all_eta_norms, axis=0)
    max_eta = np.max(all_eta_norms, axis=0)
    min_eta = np.min(all_eta_norms, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    timesteps = np.arange(N + 1)

    ax.fill_between(timesteps, mean_eta - std_eta, mean_eta + std_eta, alpha=0.3, label="±1 std")
    ax.fill_between(timesteps, min_eta, max_eta, alpha=0.15, label="Min-Max")
    ax.plot(timesteps, mean_eta, "b-", linewidth=2, label="Mean")

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("||η(k)|| [m]", fontsize=12)
    ax.set_title("State Deviation Magnitude Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save
    output_path = output_dir / "deviation_statistics.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved: {output_path}")

    plt.close()


def plot_segment_informativity(results, output_dir):
    """Plot informativity results per segment."""
    print("\n📊 Generating informativity plots...")

    n_segments = results["n_segments"]  # noqa: F841
    segment_results = results["segment_results"]

    # Extract data
    segment_indices = [r["segment_idx"] for r in segment_results]
    ranks = [r["info"]["rank"] for r in segment_results]
    required_rank = segment_results[0]["info"]["required_rank"]
    condition_numbers = [r["info"]["condition_number"] for r in segment_results]
    min_singular_values = [r["info"]["min_singular_value"] for r in segment_results]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Rank plot
    ax = axes[0]
    colors = ["green" if r["informative"] else "red" for r in segment_results]
    ax.bar(segment_indices, ranks, color=colors, alpha=0.6, edgecolor="black")
    ax.axhline(required_rank, color="blue", linestyle="--", linewidth=2, label=f"Required: {required_rank}")
    ax.set_xlabel("Segment Index", fontsize=11)
    ax.set_ylabel("Rank", fontsize=11)
    ax.set_title("Data Matrix Rank per Segment", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Condition number plot
    ax = axes[1]
    ax.semilogy(segment_indices, condition_numbers, "o-", color="purple", linewidth=2, markersize=6)
    ax.set_xlabel("Segment Index", fontsize=11)
    ax.set_ylabel("Condition Number", fontsize=11)
    ax.set_title("Data Matrix Conditioning", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Minimum singular value plot
    ax = axes[2]
    ax.semilogy(segment_indices, min_singular_values, "s-", color="orange", linewidth=2, markersize=6)
    ax.set_xlabel("Segment Index", fontsize=11)
    ax.set_ylabel("Minimum Singular Value", fontsize=11)
    ax.set_title("Data Matrix Minimum Singular Value", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = output_dir / "informativity_analysis.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved: {output_path}")

    plt.close()


def save_dataset(
    dataset, segmented_data, all_matrices, informativity_results, nominal_data, config, output_dir
):
    """Save complete dataset."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"offline_data_{timestamp}"
    folder_path = output_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Save main dataset
    filename = f"{folder_name}.pkl"
    filepath = folder_path / filename

    data = {
        "trajectories": dataset,
        "segmented_data": segmented_data,
        "hankel_matrices": all_matrices,
        "informativity_results": informativity_results,
        "nominal_x": nominal_data["x_traj"],
        "nominal_u": nominal_data["u_traj"],
        "dt": nominal_data["dt"],
        "N": nominal_data["N"],
        "config": config,
        "timestamp": timestamp,
        "folder_name": folder_name,
    }

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"\n💾 Dataset saved to: {filepath}")
    print(f"   Folder: {folder_path}")
    print(f"   File size: {filepath.stat().st_size / 1024:.2f} KB")

    return filepath, folder_path


def main():  # noqa: PLR0915
    """Main execution function."""
    print("\n" + "=" * 70)
    print("OFFLINE DATA COLLECTION - PHASE 4")
    print("=" * 70)

    # 1. Load configuration
    print("\n📋 Loading configuration...")
    config_loader = ConfigLoader()

    try:
        data_config = config_loader.load("data_collection_params")
        print("  ✓ Loaded data_collection_params.yaml")
    except FileNotFoundError:
        print("  ⚠️  data_collection_params.yaml not found, using defaults")
        data_config = {
            "sampling": {"n_samples": 20, "semi_axes": [0.2, 0.2, 0.1], "seed": 42},
            "excitation": {"magnitude": 0.1, "type": "uniform"},
            "plant_mismatch": {
                "velocity_scale": 0.95,
                "angular_rate_scale": 1.03,
                "slip_coefficient": 0.02,
            },
            "segmentation": {"segment_length": 20, "overlap": 0},
            "informativity": {"rank_threshold": 1e-10},
            "output": {"output_dir": "data/offline_datasets"},
            "validation": {"check_collision": True, "max_deviation": 1.0},
        }

    # 2. Load nominal trajectory
    print("\n📂 Loading nominal trajectory...")
    nominal_filepath = sys.argv[1] if len(sys.argv) > 1 else None
    nominal_data = load_nominal_trajectory(nominal_filepath)

    x_nom = nominal_data["x_traj"]
    u_nom = nominal_data["u_traj"]
    dt = nominal_data["dt"]
    N = nominal_data["N"]

    print(f"  ✓ Loaded: {x_nom.shape[0]} states, {u_nom.shape[0]} inputs")

    # 3. Create models
    print("\n🤖 Creating models...")
    x0 = x_nom[0, :]
    xf = x_nom[-1, :]

    twin = UnicycleModel(x0=x0, xf=xf)
    plant = PlantModel(
        twin=twin,
        parameter_mismatch=data_config["plant_mismatch"],
        x0=x0,
        xf=xf,
    )

    print(f"  Digital Twin: {twin}")
    print(f"  Plant:        {plant}")
    print(
        f"  Mismatch:     velocity={plant.param_mismatch['velocity_scale']:.2f}, "
        f"angular={plant.param_mismatch['angular_rate_scale']:.2f}, "
        f"slip={plant.param_mismatch['slip_coefficient']:.3f}"
    )

    # 4. Create environment
    print("\n🌍 Setting up environment...")
    obstacles = create_obstacles_from_config(nominal_data["obstacles"])
    collision_checker = CollisionChecker(obstacles)
    print(f"  ✓ Created {len(obstacles)} obstacles")

    # 5. Collect data
    print("\n📊 Collecting offline data...")
    collector = OfflineDataCollector(
        plant=plant,
        nominal_x=x_nom,
        nominal_u=u_nom,
        dt=dt,
        excitation_magnitude=data_config["excitation"]["magnitude"],
        seed=data_config["sampling"]["seed"],
    )

    dataset = collector.collect_dataset(
        n_samples=data_config["sampling"]["n_samples"],
        semi_axes=np.array(data_config["sampling"]["semi_axes"]),
        verbose=True,
    )

    collector.print_statistics()

    # 6. Validate trajectories
    validation_passed, issues = validate_trajectories(dataset, nominal_data, data_config, collision_checker)

    # 7. Segment trajectories
    print("\n✂️  Segmenting trajectories...")
    segmenter = TrajectorySegmenter(
        N=N,
        segment_length=data_config["segmentation"]["segment_length"],
        overlap=data_config["segmentation"]["overlap"],
    )

    segmenter.print_segment_summary()

    segmented_data = segmenter.segment_dataset(dataset, verbose=True)

    # 8. Build Hankel matrices
    print("\n🔢 Building Hankel matrices...")
    builder = HankelMatrixBuilder(n=3, m=2)
    all_matrices = builder.build_all_segments(segmented_data, segmenter, verbose=True)

    # Print sample segment
    builder.print_matrix_summary(all_matrices[0], seg_idx=0)

    # 9. Check informativity
    print("\n✅ Checking informativity...")
    checker = InformativityChecker(n=3, m=2, rank_threshold=data_config["informativity"]["rank_threshold"])
    informativity_results = checker.check_all_segments(all_matrices, verbose=True)

    # 10. Save results
    output_dir = Path(data_config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath, folder_path = save_dataset(
        dataset, segmented_data, all_matrices, informativity_results, nominal_data, data_config, output_dir
    )

    # 11. Generate plots
    if data_config["output"].get("save_trajectory_plots", True):
        plot_trajectories(dataset, nominal_data, obstacles, folder_path)
        plot_deviation_statistics(dataset, folder_path)

    if data_config["output"].get("save_informativity_report", True):
        plot_segment_informativity(informativity_results, folder_path)

    # 12. Save informativity report
    report = checker.generate_report(all_matrices)
    report_path = folder_path / "informativity_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n📄 Informativity report saved to: {report_path}")

    # 13. Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("📊 Data Collection:")
    print(f"   Trajectories:  {len(dataset)}")
    print(f"   Segments:      {segmenter.n_segments}")
    print(f"   Samples/seg:   {all_matrices[0]['L']}")

    print("\n✅ Validation:")
    print(f"   Status:        {'PASSED' if validation_passed else 'WARNINGS'}")

    print("\n🔍 Informativity:")
    print(f"   All segments:  {'✓ INFORMATIVE' if informativity_results['all_informative'] else '✗ NOT INFORMATIVE'}")
    print(f"   Passed:        {informativity_results['n_passed']}/{informativity_results['n_segments']}")

    print("\n💾 Output:")
    print(f"   Location:      {folder_path.name}/")
    print(f"   Dataset:       {filepath.name}")

    print("\n" + "=" * 70)

    if informativity_results["all_informative"]:
        print("✓ DATA COLLECTION COMPLETE - READY FOR PHASE 5")
        print("\nNext steps:")
        print("  1. Review plots in output folder")
        print("  2. Proceed to Phase 5: Uncertainty Quantification")
        print("     (compute gamma, L_r, L_J, C, β_i)")
    else:
        print("⚠️  DATA COLLECTION COMPLETE - INFORMATIVITY ISSUES")
        print("\nRecommendations:")
        print("  1. Review informativity report")
        print("  2. Increase n_samples or excitation_magnitude")
        print("  3. Re-run data collection")

    print("=" * 70 + "\n")

    return 0 if informativity_results["all_informative"] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
