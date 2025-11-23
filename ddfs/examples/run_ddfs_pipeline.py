#!/usr/bin/env python3
"""
DDFS Pipeline - Main Integration Script

This script runs the complete Data-Driven Funnel Synthesis pipeline:
    Step 1: Setup (models, workspace, obstacles, constraints)
    Step 2: Phase 1 - Nominal Planning (SCvx)
    Step 3: Phase 2 - Data Collection
    Step 4: Phase 3 - Uncertainty Quantification
    Step 5: Phase 4 - Feasibility Envelopes (MVIE)
    Step 6: Phase 5 - Funnel Synthesis (SDP)
    Step 7: Phase 6 - Deployment (Tracking Controller)

Run: python examples/run_ddfs_pipeline.py
"""

import numpy as np
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

print("=" * 80)
print("DDFS PIPELINE - STEP BY STEP INTEGRATION")
print("=" * 80)
print()

# Load configuration
from ddfs.utils import load_config

config_path = Path("config/ddfs_config.yaml")
print(f"[CONFIG] Loading configuration from: {config_path}")

if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}\nPlease create config/ddfs_config.yaml")

config = load_config(config_path)
print(f"✓ Configuration loaded")
print(f"  System: {config.system_type}")
print(f"  State dim: {config.get_system_config()['state_dim']}")
print(f"  Input dim: {config.get_system_config()['input_dim']}")
print()

# ==============================================================================
# STEP 1: SETUP (Models, Workspace, Obstacles, Constraints)
# ==============================================================================

print("=" * 80)
print("STEP 1: SETUP")
print("=" * 80)
print()

# --- 1.1: Create Digital Twin (Nominal Model) ---
print("[1.1] Creating digital twin model...")

from ddfs.models import UnicycleTwin, QuadrotorTwin

system_config = config.get_system_config()
dt = system_config["dt"]

if config.system_type == "unicycle":
    twin = UnicycleTwin(dt=dt)
elif config.system_type == "quadrotor":
    mass = system_config["mass"]
    inertia_diag = system_config["inertia"]
    inertia = np.diag(inertia_diag)
    gravity = system_config["gravity"]
    twin = QuadrotorTwin(mass=mass, inertia=inertia, gravity=gravity, dt=dt)
else:
    raise ValueError(f"Unknown system type: {config.system_type}")

print(f"✓ Digital twin created: {twin}")
print(f"  State dimension (n): {twin.state_dim}")
print(f"  Input dimension (m): {twin.input_dim}")
print(f"  Timestep (dt): {twin.dt:.6f} seconds")
print()

# --- 1.2: Create Plant Model (Real System with Mismatch) ---
print("[1.2] Creating plant model with mismatch...")

from ddfs.models import create_plant_from_config

mismatch_params = config.get_plant_mismatch_params()
plant = create_plant_from_config(twin, mismatch_params)

print(f"✓ Plant model created: {plant}")
print(f"  Mismatch parameters:")
for key, value in mismatch_params.items():
    print(f"    {key}: {value}")
print()

# --- 1.3: Create Workspace ---
print("[1.3] Creating workspace...")

workspace = config.get_workspace()

print(f"✓ Workspace created: {workspace}")
print(f"  Bounds: {workspace.bounds}")
if hasattr(workspace, 'area'):
    print(f"  Area: {workspace.area:.2f}")
elif hasattr(workspace, 'volume'):
    print(f"  Volume: {workspace.volume:.2f}")
print()

# --- 1.4: Create Obstacles ---
print("[1.4] Creating obstacles...")

obstacles = config.get_obstacles()

print(f"✓ Obstacles created: {len(obstacles)} obstacles")
for i, obs in enumerate(obstacles):
    print(
        f"  [{i}] {obs.id}: center={obs.center}, radius={obs.radius:.2f}, effective_radius={obs.effective_radius:.2f}"
    )
print()

# --- 1.5: Create System Constraints ---
print("[1.5] Creating system constraints...")

constraints = config.get_constraints()

print(f"✓ Constraints created: {constraints}")
print(f"  State bounds available: {hasattr(constraints, 'x_min')}")
print(f"  Input bounds available: {hasattr(constraints, 'u_min')}")
if hasattr(constraints, "u_min") and hasattr(constraints, "u_max"):
    print(f"  Input bounds: u_min={constraints.u_min}, u_max={constraints.u_max}")
print()

# --- 1.6: Verify Setup ---
print("[1.6] Verifying setup...")

# Test that twin can step
x_test = np.zeros(twin.state_dim)
if config.system_type == "quadrotor":
    x_test[6] = 1.0  # Set quaternion to identity [qw=1, qx=0, qy=0, qz=0]

u_test = np.zeros(twin.input_dim)
if config.system_type == "unicycle":
    u_test[0] = 1.0  # Small forward velocity
elif config.system_type == "quadrotor":
    u_test[0] = plant.m_actual * 9.81  # Hover thrust

x_next_twin = twin.step(x_test, u_test)
x_next_plant = plant.step(x_test, u_test)

print(f"✓ Twin step test passed: x_next shape = {x_next_twin.shape}")
print(f"✓ Plant step test passed: x_next shape = {x_next_plant.shape}")

# Compute mismatch
mismatch = plant.compute_mismatch(x_test, u_test)
print(f"✓ Plant-twin mismatch: {mismatch:.6e}")
print()

# --- 1.7: Setup Output Directory ---
print("[1.7] Setting up output directory...")

output_dir = config.get_output_dir()
output_dir.mkdir(parents=True, exist_ok=True)

print(f"✓ Output directory: {output_dir}")
print()

# ==============================================================================
# STEP 1 COMPLETE
# ==============================================================================

print("=" * 80)
print("✓ STEP 1 COMPLETE: Setup")
print("=" * 80)
print()
print("Summary:")
print(f"  - Digital twin: {twin.__class__.__name__}")
print(f"  - Plant model: {plant.__class__.__name__}")
print(f"  - Workspace: {workspace.__class__.__name__}")
print(f"  - Obstacles: {len(obstacles)}")
print(f"  - Constraints: {constraints.__class__.__name__}")
print(f"  - Output directory: {output_dir}")
print()
print("Next step: Phase 1 - Nominal Planning")
print("  (To be implemented)")
print()

# ==============================================================================
# SAVE SETUP SUMMARY
# ==============================================================================

print("[SAVE] Saving setup summary...")

summary_path = output_dir / "step1_setup_summary.txt"
with open(summary_path, "w") as f:
    f.write("DDFS PIPELINE - STEP 1 SETUP SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"System Type: {config.system_type}\n")
    f.write(f"State Dimension: {twin.state_dim}\n")
    f.write(f"Input Dimension: {twin.input_dim}\n")
    f.write(f"Timestep: {twin.dt:.6f} seconds\n\n")

    f.write("Digital Twin:\n")
    f.write(f"  {twin}\n\n")

    f.write("Plant Model:\n")
    f.write(f"  {plant}\n")
    f.write(f"  Mismatch parameters:\n")
    for key, value in mismatch_params.items():
        f.write(f"    {key}: {value}\n")
    f.write("\n")

    f.write(f"Workspace:\n")
    f.write(f"  {workspace}\n")
    f.write(f"  Bounds: {workspace.bounds}\n")
    if hasattr(workspace, 'area'):
        f.write(f"  Area: {workspace.area:.2f}\n")
    elif hasattr(workspace, 'volume'):
        f.write(f"  Volume: {workspace.volume:.2f}\n")
    f.write("\n")

    f.write(f"Obstacles ({len(obstacles)}):\n")
    for i, obs in enumerate(obstacles):
        f.write(f"  [{i}] {obs}\n")
    f.write("\n")

    f.write(f"Constraints:\n")
    f.write(f"  {constraints}\n\n")

    f.write(f"Verification:\n")
    f.write(f"  Twin step: OK\n")
    f.write(f"  Plant step: OK\n")
    f.write(f"  Plant-twin mismatch: {mismatch:.6e}\n")

print(f"✓ Summary saved to: {summary_path}")
print()

print("=" * 80)
print("STEP 1 COMPLETE - Ready for Phase 1 (Nominal Planning)")
print("=" * 80)
print()

# ==============================================================================
# STEP 2: PHASE 1 - NOMINAL PLANNING (SCvx)
# ==============================================================================

print("=" * 80)
print("STEP 2: PHASE 1 - NOMINAL PLANNING")
print("=" * 80)
print()

# --- 2.1: Setup Planner ---
print("[2.1] Setting up SCvx planner...")

from ddfs.planning import SCvxPlanner  # noqa: E402

planning_params = config.get_planning_params()

# Create planner configuration
planner_config = {
    "max_iterations": planning_params.get("max_iterations", 20),
    "convergence_tol": planning_params.get("convergence_tol", 0.001),
    "trust_region": planning_params.get("trust_region", 1.0),
    "verbose": planning_params.get("verbose", True),
    "weight_state": planning_params.get("weight_state", 1.0),
    "weight_input": planning_params.get("weight_input", 0.1),
    "weight_virtual": planning_params.get("weight_virtual", 1000.0),
}

planner = SCvxPlanner(twin, constraints, obstacles, config=planner_config)

print(f"✓ Planner created: {planner}")
print(f"  Max iterations: {planner.max_iterations}")
print(f"  Convergence tolerance: {planner.convergence_tol}")
print(f"  Trust region: {planner.trust_region}")
print()

# --- 2.2: Plan Nominal Trajectory ---
print("[2.2] Planning nominal trajectory...")

from ddfs.planning import NominalTrajectory  # noqa: E402

x0 = np.array(planning_params["x0"])
xf = np.array(planning_params["xf"])
N = planning_params["N"]

print(f"  Initial state: {x0}")
print(f"  Goal state: {xf}")
print(f"  Horizon: N = {N}")
print(f"  Final time: tf = {N * dt:.2f} seconds")
print()

try:
    nominal = planner.plan(x0=x0, xf=xf, N=N)

    print(f"✓ Nominal trajectory planned: {nominal}")
    print(f"  Duration: {nominal.tf:.2f} seconds")
    print(f"  Timesteps: {nominal.N + 1} states, {nominal.N} controls")
    print()

except Exception as e:
    print(f"✗ Planning failed: {e}")
    print("  Using straight-line trajectory as fallback...")

    # Fallback: straight-line trajectory
    x_nom = np.zeros((N + 1, twin.state_dim))
    for i in range(N + 1):
        alpha = i / N
        x_nom[i] = (1 - alpha) * x0 + alpha * xf

    u_nom = np.zeros((N, twin.input_dim))
    if config.system_type == "unicycle":
        u_nom[:, 0] = 1.0  # Constant forward velocity
    elif config.system_type == "quadrotor":
        u_nom[:, 0] = twin.m * 9.81  # Hover thrust

    nominal = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=dt)
    print(f"✓ Fallback trajectory created: {nominal}")
    print()

# --- 2.3: Save Nominal Trajectory ---
print("[2.3] Saving nominal trajectory...")

nominal_path = output_dir / "nominal_trajectory.pkl"
nominal.save(nominal_path)

print(f"✓ Nominal trajectory saved to: {nominal_path}")
print()

# --- 2.4: Visualize Nominal Trajectory ---
print("[2.4] Visualizing nominal trajectory...")

try:
    import matplotlib.pyplot as plt

    if config.system_type == "unicycle":
        from ddfs.visualization.unicycle_viz import plot_nominal_trajectory  # noqa: E402

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_nominal_trajectory(nominal, workspace, obstacles, ax=ax, show_heading=True, heading_interval=10)

        # Save figure
        fig_path = output_dir / "step2_nominal_trajectory.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"✓ Visualization saved to: {fig_path}")

    elif config.system_type == "quadrotor":
        from ddfs.visualization.quadrotor_viz import plot_nominal_trajectory_3d  # noqa: E402

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        plot_nominal_trajectory_3d(nominal, workspace, obstacles, ax=ax, show_velocity=False)

        # Save figure
        fig_path = output_dir / "step2_nominal_trajectory_3d.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"✓ Visualization saved to: {fig_path}")

    print()

except ImportError as e:
    print(f"⚠ Visualization skipped (missing dependencies): {e}")
    print()

# --- 2.5: Trajectory Statistics ---
print("[2.5] Computing trajectory statistics...")

# Extract position (first 2 or 3 dimensions)
if config.system_type == "unicycle":
    pos_traj = nominal.x_nom[:, :2]  # [x, y]
elif config.system_type == "quadrotor":
    pos_traj = nominal.x_nom[:, :3]  # [x, y, z]

# Compute path length
path_segments = np.diff(pos_traj, axis=0)
segment_lengths = np.linalg.norm(path_segments, axis=1)
total_path_length = np.sum(segment_lengths)

# Compute straight-line distance
straight_line_dist = np.linalg.norm(pos_traj[-1] - pos_traj[0])

# Compute control effort
control_effort = np.sum(np.linalg.norm(nominal.u_nom, axis=1) * dt)

print(f"✓ Trajectory statistics:")
print(f"  Path length: {total_path_length:.2f} m")
print(f"  Straight-line distance: {straight_line_dist:.2f} m")
print(f"  Path efficiency: {straight_line_dist / total_path_length * 100:.1f}%")
print(f"  Control effort: {control_effort:.2f}")
print(f"  Average speed: {total_path_length / nominal.tf:.2f} m/s")
print()

# --- 2.6: Verify Obstacle Avoidance ---
print("[2.6] Verifying obstacle avoidance...")

violations = 0
for k in range(nominal.N + 1):
    x_k = nominal.x_nom[k]
    for obs in obstacles:
        if obs.contains(x_k, include_margin=True):
            violations += 1
            if violations <= 3:  # Only print first 3
                print(f"  ⚠ Violation at k={k}: inside {obs.id}")

if violations == 0:
    print(f"✓ No obstacle violations detected")
else:
    print(f"✗ {violations} obstacle violations detected")
print()

# ==============================================================================
# STEP 2 COMPLETE
# ==============================================================================

print("=" * 80)
print("✓ STEP 2 COMPLETE: Phase 1 - Nominal Planning")
print("=" * 80)
print()
print("Summary:")
print(f"  - Planner: {planner.__class__.__name__}")
print(f"  - Trajectory: {nominal.N} steps, {nominal.tf:.2f} seconds")
print(f"  - Path length: {total_path_length:.2f} m")
print(f"  - Obstacle violations: {violations}")
print(f"  - Output: {nominal_path}")
print()
print("Next step: Phase 2 - Data Collection")
print("  (To be implemented)")
print()

# ==============================================================================
# SAVE STEP 2 SUMMARY
# ==============================================================================

print("[SAVE] Saving Phase 1 summary...")

phase1_summary_path = output_dir / "step2_phase1_summary.txt"
with open(phase1_summary_path, "w") as f:
    f.write("DDFS PIPELINE - STEP 2: PHASE 1 NOMINAL PLANNING\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Planner: {planner.__class__.__name__}\n")
    f.write(f"Max iterations: {planner.max_iterations}\n")
    f.write(f"Convergence tolerance: {planner.convergence_tol}\n\n")

    f.write(f"Initial state: {x0}\n")
    f.write(f"Goal state: {xf}\n")
    f.write(f"Horizon: N = {N}\n")
    f.write(f"Timestep: dt = {dt:.6f} seconds\n")
    f.write(f"Final time: tf = {nominal.tf:.2f} seconds\n\n")

    f.write("Trajectory Statistics:\n")
    f.write(f"  Path length: {total_path_length:.2f} m\n")
    f.write(f"  Straight-line distance: {straight_line_dist:.2f} m\n")
    f.write(f"  Path efficiency: {straight_line_dist / total_path_length * 100:.1f}%\n")
    f.write(f"  Control effort: {control_effort:.2f}\n")
    f.write(f"  Average speed: {total_path_length / nominal.tf:.2f} m/s\n\n")

    f.write(f"Obstacle Avoidance:\n")
    f.write(f"  Violations: {violations}\n\n")

    f.write(f"Saved Files:\n")
    f.write(f"  - Trajectory: {nominal_path.name}\n")
    if "fig_path" in locals():
        f.write(f"  - Visualization: {fig_path.name}\n")

print(f"✓ Summary saved to: {phase1_summary_path}")
print()

print("=" * 80)
print("STEP 2 COMPLETE - Ready for Phase 2 (Data Collection)")
print("=" * 80)
