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

from pathlib import Path

import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================

print("=" * 80)
print("DDFS PIPELINE - STEP BY STEP INTEGRATION")
print("=" * 80)
print()

# Load configuration
from ddfs.utils import load_config  # noqa: E402

config_path = Path("config/ddfs_config.yaml")
print(f"[CONFIG] Loading configuration from: {config_path}")

if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}\nPlease create config/ddfs_config.yaml")

config = load_config(config_path)
print("✓ Configuration loaded")
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

from ddfs.models import QuadrotorTwin, UnicycleTwin  # noqa: E402

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

from ddfs.models import create_plant_from_config  # noqa: E402

mismatch_params = config.get_plant_mismatch_params()
plant = create_plant_from_config(twin, mismatch_params)

print(f"✓ Plant model created: {plant}")
print("  Mismatch parameters:")
for key, value in mismatch_params.items():
    print(f"    {key}: {value}")
print()

# --- 1.3: Create Workspace ---
print("[1.3] Creating workspace...")

workspace = config.get_workspace()

print(f"✓ Workspace created: {workspace}")
print(f"  Bounds: {workspace.bounds}")
print(f"  Volume: {workspace.volume():.2f}")
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
    f.write("  Mismatch parameters:\n")
    for key, value in mismatch_params.items():
        f.write(f"    {key}: {value}\n")
    f.write("\n")

    f.write("Workspace:\n")
    f.write(f"  {workspace}\n")
    f.write(f"  Bounds: {workspace.bounds}\n")
    f.write(f"  Volume: {workspace.volume():.2f}\n\n")

    f.write(f"Obstacles ({len(obstacles)}):\n")
    for i, obs in enumerate(obstacles):
        f.write(f"  [{i}] {obs}\n")
    f.write("\n")

    f.write("Constraints:\n")
    f.write(f"  {constraints}\n\n")

    f.write("Verification:\n")
    f.write("  Twin step: OK\n")
    f.write("  Plant step: OK\n")
    f.write(f"  Plant-twin mismatch: {mismatch:.6e}\n")

print(f"✓ Summary saved to: {summary_path}")
print()

print("=" * 80)
print("STEP 1 COMPLETE - Ready for Phase 1 (Nominal Planning)")
print("=" * 80)
