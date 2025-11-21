"""
Visualize offline collected data.

This script loads saved offline data and creates comprehensive visualizations:
- Trajectories in workspace
- Deviation statistics
- Hankel matrix properties
- Informativity analysis

Usage:
    python scripts/02b_visualize_offline_data.py [dataset_file]

    If no file is specified, loads the most recent dataset.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_dataset(filepath=None):
    """Load dataset from file."""
    if filepath is None:
        # Find most recent dataset
        data_dir = Path("data/offline_datasets")
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name.startswith("offline_data_")]

        if len(folders) == 0:
            raise FileNotFoundError(f"No offline datasets found in {data_dir}")

        folder_path = max(folders, key=lambda p: p.stat().st_mtime)
        pkl_files = list(folder_path.glob("*.pkl"))

        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No .pkl file found in {folder_path}")

        filepath = pkl_files[0]
        print(f"📂 Loading most recent dataset: {folder_path.name}/")

    else:
        filepath = Path(filepath)

    print(f"   File: {filepath.name}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data, filepath.parent


def plot_detailed_trajectories(data, output_dir):  # noqa: PLR0915
    """Create detailed trajectory visualization."""
    print("\n📊 Creating detailed trajectory plots...")

    trajectories = data["trajectories"]
    x_nom = data["nominal_x"]

    # Create obstacles
    obstacles = []  # noqa: F841
    for obs_config in data["config"]["plant_mismatch"]:  # This will fail, need to fix
        pass  # Will implement properly

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. All trajectories
    ax = axes[0, 0]
    for i, traj_data in enumerate(trajectories):
        x_traj = traj_data["x"]
        ax.plot(x_traj[:, 0], x_traj[:, 1], "b-", alpha=0.3, linewidth=0.8)
        if i == 0:
            ax.plot([], [], "b-", alpha=0.5, linewidth=2, label="Collected")

    ax.plot(x_nom[:, 0], x_nom[:, 1], "g-", linewidth=3, label="Nominal", zorder=10)
    ax.plot(x_nom[0, 0], x_nom[0, 1], "go", markersize=12, label="Start", zorder=11)
    ax.plot(x_nom[-1, 0], x_nom[-1, 1], "r*", markersize=20, label="Goal", zorder=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"All {len(trajectories)} Collected Trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    # 2. Initial state distribution
    ax = axes[0, 1]
    x0_samples = np.array([traj["x0"] for traj in trajectories])

    # Plot initial state ellipsoid
    semi_axes = data["config"]["sampling"]["semi_axes"]
    ellipse = Ellipse(
        xy=x_nom[0, :2],
        width=2 * semi_axes[0],
        height=2 * semi_axes[1],
        facecolor="green",
        alpha=0.2,
        edgecolor="green",
        linewidth=2,
        label="Sampling ellipsoid",
    )
    ax.add_patch(ellipse)

    ax.scatter(x0_samples[:, 0], x0_samples[:, 1], c="blue", s=50, alpha=0.6, label="Sampled x0")
    ax.plot(x_nom[0, 0], x_nom[0, 1], "go", markersize=12, label="Nominal x0", zorder=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Initial State Sampling")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    # 3. State deviation over time
    ax = axes[1, 0]
    N = trajectories[0]["eta"].shape[0] - 1
    all_eta_norms = np.zeros((len(trajectories), N + 1))

    for i, traj_data in enumerate(trajectories):
        eta_norms = np.linalg.norm(traj_data["eta"], axis=1)
        all_eta_norms[i, :] = eta_norms
        ax.plot(eta_norms, "b-", alpha=0.2, linewidth=0.8)

    mean_eta = np.mean(all_eta_norms, axis=0)
    ax.plot(mean_eta, "r-", linewidth=3, label="Mean")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||η(k)|| [m]")
    ax.set_title("State Deviation Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Input deviation over time
    ax = axes[1, 1]
    all_xi_norms = np.zeros((len(trajectories), N))

    for i, traj_data in enumerate(trajectories):
        xi_norms = np.linalg.norm(traj_data["xi"], axis=1)
        all_xi_norms[i, :] = xi_norms
        ax.plot(xi_norms, "purple", alpha=0.2, linewidth=0.8)

    mean_xi = np.mean(all_xi_norms, axis=0)
    ax.plot(mean_xi, "orange", linewidth=3, label="Mean")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||ξ(k)||")
    ax.set_title("Input Deviation Magnitude (Excitation)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    output_path = output_dir / "detailed_trajectories.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved: {output_path}")

    plt.close()


def plot_hankel_analysis(data, output_dir):
    """Plot Hankel matrix analysis."""
    print("\n📊 Creating Hankel matrix analysis...")

    all_matrices = data["hankel_matrices"]
    n_segments = len(all_matrices)  # noqa: F841

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extract statistics
    segment_indices = [m["segment_idx"] for m in all_matrices]
    L_values = [m["L"] for m in all_matrices]

    # Compute norms
    H_norms = [np.linalg.norm(m["H"], ord="fro") for m in all_matrices]
    Hplus_norms = [np.linalg.norm(m["H_plus"], ord="fro") for m in all_matrices]
    Xi_norms = [np.linalg.norm(m["Xi"], ord="fro") for m in all_matrices]

    # 1. Number of samples per segment
    ax = axes[0, 0]
    ax.bar(segment_indices, L_values, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Segment Index")
    ax.set_ylabel("Number of Samples (L)")
    ax.set_title("Data Samples per Segment")
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Frobenius norms
    ax = axes[0, 1]
    ax.plot(segment_indices, H_norms, "o-", label="||H||_F", linewidth=2, markersize=6)
    ax.plot(segment_indices, Hplus_norms, "s-", label="||H+||_F", linewidth=2, markersize=6)
    ax.plot(segment_indices, Xi_norms, "^-", label="||Ξ||_F", linewidth=2, markersize=6)
    ax.set_xlabel("Segment Index")
    ax.set_ylabel("Frobenius Norm")
    ax.set_title("Data Matrix Norms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Sample Hankel matrix visualization (segment 0)
    ax = axes[1, 0]
    H_sample = all_matrices[0]["H"]
    im = ax.imshow(H_sample, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Data Sample Index")
    ax.set_ylabel("State-Input Dimension")
    ax.set_title(f"Hankel Matrix H (Segment 0): {H_sample.shape}")
    plt.colorbar(im, ax=ax)

    # 4. Sample H+ matrix visualization (segment 0)
    ax = axes[1, 1]
    Hplus_sample = all_matrices[0]["H_plus"]
    im = ax.imshow(Hplus_sample, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Data Sample Index")
    ax.set_ylabel("State Dimension")
    ax.set_title(f"Future State Matrix H+ (Segment 0): {Hplus_sample.shape}")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    output_path = output_dir / "hankel_analysis.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved: {output_path}")

    plt.close()


def print_comprehensive_summary(data):
    """Print comprehensive dataset summary."""
    print("\n" + "=" * 70)
    print("OFFLINE DATASET SUMMARY")
    print("=" * 70)

    # Data collection info
    print("\n📊 Data Collection:")
    print(f"  Trajectories:      {len(data['trajectories'])}")
    print(f"  Timesteps:         {data['N']}")
    print(f"  Time step (dt):    {data['dt']:.4f}s")
    print(f"  State dimension:   {data['nominal_x'].shape[1]}")
    print(f"  Input dimension:   {data['nominal_u'].shape[1]}")

    # Sampling info
    sampling = data["config"]["sampling"]
    print("\n🎲 Sampling:")
    print(f"  n_samples:         {sampling['n_samples']}")
    print(f"  Semi-axes:         {sampling['semi_axes']}")
    print(f"  Seed:              {sampling['seed']}")

    # Excitation info
    excitation = data["config"]["excitation"]
    print("\n⚡ Excitation:")
    print(f"  Magnitude:         ±{excitation['magnitude']}")
    print(f"  Type:              {excitation['type']}")

    # Segmentation info
    print("\n✂️  Segmentation:")
    print(f"  Number of segments: {len(data['hankel_matrices'])}")
    print(f"  Segment length:     {data['config']['segmentation']['segment_length']}")
    print(f"  Overlap:            {data['config']['segmentation']['overlap']}")

    # Hankel matrices
    print("\n🔢 Hankel Matrices:")
    for i, matrices in enumerate(data["hankel_matrices"][:3]):  # Show first 3
        print(f"  Segment {i}:")
        print(f"    H shape:   {matrices['H'].shape}")
        print(f"    H+ shape:  {matrices['H_plus'].shape}")
        print(f"    Ξ shape:   {matrices['Xi'].shape}")
        print(f"    Samples:   {matrices['L']}")
    if len(data["hankel_matrices"]) > 3:
        print(f"  ... and {len(data['hankel_matrices']) - 3} more segments")

    # Informativity
    results = data["informativity_results"]
    print("\n✅ Informativity:")
    print(f"  All informative:    {results['all_informative']}")
    print(f"  Passed:             {results['n_passed']}/{results['n_segments']}")
    print(f"  Failed:             {results['n_failed']}/{results['n_segments']}")

    # Timestamp
    print(f"\n📅 Generated:         {data['timestamp']}")

    print("=" * 70)


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("OFFLINE DATA VISUALIZATION")
    print("=" * 70)

    # Load dataset
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    data, output_dir = load_dataset(filepath)

    # Print summary
    print_comprehensive_summary(data)

    # Generate plots
    plot_detailed_trajectories(data, output_dir)
    plot_hankel_analysis(data, output_dir)

    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE")
    print(f"  Output directory: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
