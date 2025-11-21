#!/usr/bin/env python3
"""
Phase 5: Compute Uncertainty Constants

This script computes all uncertainty constants required for DDFS:
1. gamma: Plant-twin mismatch along nominal trajectory
2. L_r: Linearization error Lipschitz constant
3. L_J: Jacobian Lipschitz constant
4. C: Increment bound (C = L_J * v_max)
5. β_i: Per-segment uncertainty bounds from data

Usage:
    python scripts/03_compute_uncertainty.py [nominal_trajectory.pkl] [collected_data.pkl]

Output:
    - uncertainty_constants.pkl: All computed constants
    - uncertainty_analysis.png: Visualization of uncertainty distributions
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddfs.models.plant import PlantModel
from ddfs.models.unicycle import UnicycleModel
from ddfs.uncertainty.uncertainty_constants import UncertaintyConstants, UncertaintyQuantifier
from ddfs.utils.config_loader import ConfigLoader


def load_nominal_trajectory(filepath: Path) -> dict:
    """Load nominal trajectory from Phase 3."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def load_collected_data(filepath: Path) -> dict:
    """Load collected offline data from Phase 4."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def visualize_uncertainty(constants: UncertaintyConstants, output_dir: Path):  # noqa: PLR0915
    """
    Create visualization of uncertainty constants.

    Args:
        constants: Computed uncertainty constants
        output_dir: Directory to save plots
    """
    fig = plt.figure(figsize=(14, 10))  # noqa: F841

    # 1. gamma per timestep
    if constants.gamma_per_timestep is not None:
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(constants.gamma_per_timestep, "b-", linewidth=1.5)
        ax1.axhline(constants.gamma, color="r", linestyle="--", label=f"max gamma = {constants.gamma:.6f}")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Mismatch ||f_plant - f_twin||")
        ax1.set_title("Plant-Twin Mismatch Along Nominal")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. L_r per state
    if constants.L_r_per_state is not None:
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(constants.L_r_per_state, "g-", linewidth=1.5)
        ax2.axhline(constants.L_r, color="r", linestyle="--", label=f"max L_r = {constants.L_r:.6f}")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Linearization Error Lipschitz L_r")
        ax2.set_title("Linearization Error Along Nominal")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. L_J sample distribution
    if constants.L_J_samples is not None:
        ax3 = plt.subplot(3, 2, 3)
        ax3.hist(constants.L_J_samples, bins=50, alpha=0.7, edgecolor="black")
        ax3.axvline(constants.L_J, color="r", linestyle="--", linewidth=2, label=f"max L_J = {constants.L_J:.6f}")
        ax3.axvline(
            np.mean(constants.L_J_samples),
            color="b",
            linestyle="--",
            linewidth=2,
            label=f"mean = {np.mean(constants.L_J_samples):.6f}",
        )
        ax3.set_xlabel("L_J Sample Value")
        ax3.set_ylabel("Frequency")
        ax3.set_title(f"Jacobian Lipschitz Distribution ({constants.n_samples_L_J} samples)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. β_i per segment
    if constants.beta_i:
        ax4 = plt.subplot(3, 2, 4)
        segment_indices = np.arange(len(constants.beta_i))
        ax4.bar(segment_indices, constants.beta_i, alpha=0.7, edgecolor="black")
        ax4.axhline(
            np.mean(constants.beta_i), color="r", linestyle="--", label=f"mean β = {np.mean(constants.beta_i):.6f}"
        )
        ax4.set_xlabel("Segment Index")
        ax4.set_ylabel("β_i")
        ax4.set_title(f"Per-Segment Uncertainty Bounds ({len(constants.beta_i)} segments)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Summary table
    ax5 = plt.subplot(3, 2, 5)
    ax5.axis("off")

    summary_data = [
        ["Constant", "Value", "Description"],
        ["gamma", f"{constants.gamma:.6f}", "Plant-twin mismatch"],
        ["L_r", f"{constants.L_r:.6f}", "Linearization error"],
        ["L_J", f"{constants.L_J:.6f}", "Jacobian Lipschitz"],
        ["C", f"{constants.C:.6f}", "Increment bound"],
        ["v_max", f"{constants.v_max:.6f}", "Velocity bound"],
        ["", "", ""],
        ["# Segments", f"{len(constants.beta_i)}", ""],
        ["β_i min", f"{min(constants.beta_i):.6f}" if constants.beta_i else "N/A", ""],
        ["β_i max", f"{max(constants.beta_i):.6f}" if constants.beta_i else "N/A", ""],
        ["β_i mean", f"{np.mean(constants.beta_i):.6f}" if constants.beta_i else "N/A", ""],
    ]

    table = ax5.table(cellText=summary_data, cellLoc="left", loc="center", colWidths=[0.3, 0.3, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax5.set_title("Uncertainty Constants Summary", fontsize=12, weight="bold", pad=20)

    # 6. Ratio visualization
    ax6 = plt.subplot(3, 2, 6)
    constants_list = [constants.gamma, constants.L_r, constants.L_J, constants.C]
    constant_names = ["gamma", "L_r", "L_J", "C"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bars = ax6.bar(constant_names, constants_list, color=colors, alpha=0.7, edgecolor="black")
    ax6.set_ylabel("Constant Value")
    ax6.set_title("Relative Magnitude of Constants")
    ax6.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, constants_list):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "uncertainty_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved uncertainty analysis plot: {output_path}")

    # Also save as PDF
    output_path_pdf = output_dir / "uncertainty_analysis.pdf"
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved uncertainty analysis plot: {output_path_pdf}")

    plt.close()


def main():  # noqa: C901, PLR0912, PLR0915
    """Main function for uncertainty computation."""
    parser = argparse.ArgumentParser(description="Compute uncertainty constants for DDFS")
    parser.add_argument("nominal_traj", nargs="?", type=str, help="Path to nominal trajectory pickle file")
    parser.add_argument("collected_data", nargs="?", type=str, help="Path to collected offline data pickle file")
    parser.add_argument("--output-dir", type=str, default="data/uncertainty", help="Output directory for results")
    parser.add_argument(
        "--config", type=str, default="config/uncertainty_config.yaml", help="Path to uncertainty config file"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PHASE 5: UNCERTAINTY QUANTIFICATION")
    print("=" * 70)

    # 1. Load configurations
    print("\n📁 Step 1: Loading configurations...")
    config_loader = ConfigLoader()

    # Load base configs
    unicycle_config = config_loader.load("unicycle_params")  # noqa: F841

    # Load uncertainty config
    uncertainty_config_path = Path(args.config)
    if uncertainty_config_path.exists():
        # Extract config name (without path and extension)
        config_name = uncertainty_config_path.stem
        uncertainty_config = config_loader.load(config_name)
    else:
        # Try loading by name if it's a config name
        try:
            config_name = args.config.replace("config/", "").replace(".yaml", "")
            uncertainty_config = config_loader.load(config_name)
        except Exception:
            print(f"   ⚠️  Config not found at {args.config}, using defaults")
            uncertainty_config = {
                "epsilon_fd": 1e-6,
                "n_samples_L_J": 10000,
                "v_max": 1.0,
                "sampling_box": {
                    "x_min": [-10, -10, -2 * np.pi],
                    "x_max": [10, 10, 2 * np.pi],
                    "u_min": [-2.0, -2.0],
                    "u_max": [2.0, 2.0],
                },
            }

    # 2. Find or use provided data files
    print("\n📁 Step 2: Loading data files...")

    # Find nominal trajectory
    if args.nominal_traj:
        nominal_traj_path = Path(args.nominal_traj)
    else:
        # Look for most recent nominal trajectory
        nominal_dir = Path("data/nominal_trajectories")
        if nominal_dir.exists():
            # Look for subdirectories with pkl files
            subdirs = sorted([d for d in nominal_dir.iterdir() if d.is_dir()])
            nominal_files = []
            for subdir in subdirs:
                pkl_files = list(subdir.glob("*.pkl"))
                nominal_files.extend(pkl_files)
            if nominal_files:
                nominal_traj_path = nominal_files[-1]
                print(f"   Using most recent nominal: {nominal_traj_path.name}")
            else:
                print("   ❌ No nominal trajectory files found!")
                return 1
        else:
            print("   ❌ Nominal trajectories directory not found!")
            return 1

    # Find collected data
    if args.collected_data:
        collected_data_path = Path(args.collected_data)
    else:
        # Look for most recent collected data
        data_dir = Path("data/offline_datasets")
        if data_dir.exists():
            # Look for subdirectories with pkl files
            subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
            data_files = []
            for subdir in subdirs:
                pkl_files = list(subdir.glob("*.pkl"))
                data_files.extend(pkl_files)
            if data_files:
                collected_data_path = data_files[-1]
                print(f"   Using most recent data: {collected_data_path.name}")
            else:
                print("   ❌ No collected data files found!")
                return 1
        else:
            print("   ❌ Offline datasets directory not found!")
            return 1

    # Load data
    print(f"   Loading nominal trajectory: {nominal_traj_path}")
    nominal_traj = load_nominal_trajectory(nominal_traj_path)

    print(f"   Loading collected data: {collected_data_path}")
    collected_data = load_collected_data(collected_data_path)

    # 3. Initialize models
    print("\n🤖 Step 3: Initializing models...")

    # Get initial and goal states from nominal trajectory
    x0 = nominal_traj["x_traj"][0, :]
    xf = nominal_traj["x_traj"][-1, :]

    # Twin model (nominal)
    twin = UnicycleModel(x0=x0, xf=xf)
    print("   ✓ Twin model: UnicycleModel")

    # Plant model (with mismatch) - get from collected data or data collection config
    plant_mismatch = None
    if "metadata" in collected_data and "plant_mismatch" in collected_data["metadata"]:
        plant_mismatch = collected_data["metadata"]["plant_mismatch"]
    else:
        # Try loading from data collection config
        try:
            data_collection_config = config_loader.load("data_collection_config")
            if "plant_mismatch" in data_collection_config:
                plant_mismatch = data_collection_config["plant_mismatch"]
        except Exception:
            pass

    if plant_mismatch:
        plant = PlantModel(
            twin=twin,
            parameter_mismatch=plant_mismatch,
            x0=x0,
            xf=xf,
        )
        print("   ✓ Plant model: PlantModel with mismatch")
        print(
            f"      velocity_scale={plant_mismatch.get('velocity_scale', 'N/A')}, "
            f"angular_rate_scale={plant_mismatch.get('angular_rate_scale', 'N/A')}, "
            f"slip_coefficient={plant_mismatch.get('slip_coefficient', 'N/A')}"
        )
    else:
        # Use twin as plant (no mismatch)
        plant = twin
        print(" No mismatch specified, using twin as plant")

    # 4. Initialize uncertainty quantifier
    print("\n🔧 Step 4: Initializing uncertainty quantifier...")

    quantifier = UncertaintyQuantifier(
        plant=plant,
        twin=twin,
        n_states=3,
        n_controls=2,
        epsilon_fd=uncertainty_config.get("epsilon_fd", 1e-6),
        n_samples_L_J=uncertainty_config.get("n_samples_L_J", 10000),
        sampling_box=uncertainty_config.get("sampling_box"),
    )
    print(" Quantifier initialized")
    print(f"   - FD epsilon: {uncertainty_config.get('epsilon_fd', 1e-6):.2e}")
    print(f"   - L_J samples: {uncertainty_config.get('n_samples_L_J', 10000)}")

    # 5. Compute all uncertainty constants
    print("\nStep 5: Computing uncertainty constants...")

    # Convert nominal trajectory to expected format
    nominal_traj_formatted = {
        "X": nominal_traj["x_traj"],  # (N+1, n) -> (N, n) for states
        "U": nominal_traj["u_traj"],  # (N, m)
        "T": np.arange(nominal_traj["x_traj"].shape[0]) * nominal_traj.get("dt", 0.1),  # (N+1,)
    }
    # Note: X has N+1 states, but U has N inputs. The quantifier expects matching lengths.
    # We'll use the first N states to match U
    N = nominal_traj["u_traj"].shape[0]
    nominal_traj_formatted["X"] = nominal_traj["x_traj"][:N, :]  # Use first N states
    nominal_traj_formatted["T"] = nominal_traj_formatted["T"][:N]  # Use first N timesteps

    # Convert collected data to expected format
    collected_data_formatted = collected_data.copy()
    if "hankel_matrices" in collected_data and "segments" not in collected_data:
        # Convert hankel_matrices to segments format
        hankel_matrices = collected_data["hankel_matrices"]
        segments = []
        for i, matrices in enumerate(hankel_matrices):
            segment = {
                "H_i": matrices.get("H"),  # Past data matrix
                "Xi_i": matrices.get("Xi"),  # Input deviation matrix
            }
            segments.append(segment)
        collected_data_formatted["segments"] = segments

    v_max = uncertainty_config.get("v_max", 1.0)
    constants = quantifier.compute_all(
        nominal_trajectory=nominal_traj_formatted, collected_data=collected_data_formatted, v_max=v_max, verbose=True
    )

    # 6. Save results
    print("\n Step 6: Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save constants
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    constants_file = output_dir / f"uncertainty_constants_{timestamp}.pkl"
    constants.save(constants_file)

    # Also save the most recent as "latest"
    latest_file = output_dir / "uncertainty_constants_latest.pkl"
    constants.save(latest_file)

    # 7. Visualize
    print("\nStep 7: Creating visualizations...")
    visualize_uncertainty(constants, output_dir)

    # 8. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(constants.summary())


    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
