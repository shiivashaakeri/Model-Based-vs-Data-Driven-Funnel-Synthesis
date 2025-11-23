# DDFS Examples

This directory contains example scripts demonstrating the DDFS pipeline.

## 📋 Main Pipeline Script

### `run_ddfs_pipeline.py` - Step-by-Step DDFS Pipeline

Complete pipeline implementation built incrementally:

**Current Status:** ✅ Step 1 Complete

- ✅ **Step 1: Setup** (Models, Workspace, Obstacles, Constraints)
- ⏳ **Step 2: Phase 1** - Nominal Planning (SCvx)
- ⏳ **Step 3: Phase 2** - Data Collection
- ⏳ **Step 4: Phase 3** - Uncertainty Quantification
- ⏳ **Step 5: Phase 4** - Feasibility Envelopes (MVIE)
- ⏳ **Step 6: Phase 5** - Funnel Synthesis (SDP)
- ⏳ **Step 7: Phase 6** - Deployment (Tracking Controller)

### Running Step 1

```bash
# From project root
python examples/run_ddfs_pipeline.py
```

### Expected Output

```
================================================================================
DDFS PIPELINE - STEP BY STEP INTEGRATION
================================================================================

[CONFIG] Loading configuration from: config/ddfs_config.yaml
✓ Configuration loaded
  System: unicycle
  State dim: 3
  Input dim: 2

================================================================================
STEP 1: SETUP
================================================================================

[1.1] Creating digital twin model...
✓ Digital twin created: UnicycleTwin(state_dim=3, input_dim=2, dt=0.131148)
  State dimension (n): 3
  Input dimension (m): 2
  Timestep (dt): 0.131148 seconds

[1.2] Creating plant model with mismatch...
✓ Plant model created: UnicyclePlant(velocity_scale=0.950, angular_scale=1.030, slip_coefficient=0.0200)
  Mismatch parameters:
    velocity_scale: 0.95
    angular_scale: 1.03
    slip_coefficient: 0.02

[1.3] Creating workspace...
✓ Workspace created: Workspace2D(bounds=(0.0, 12.0, 0.0, 8.0))
  Bounds: (0.0, 12.0, 0.0, 8.0)
  Volume: 96.00

[1.4] Creating obstacles...
✓ Obstacles created: 2 obstacles
  [0] obs_1: center=[4. 3.], radius=1.00, effective_radius=1.25
  [1] obs_2: center=[8. 3.], radius=1.00, effective_radius=1.25

[1.5] Creating system constraints...
✓ Constraints created: UnicycleConstraints(state_dim=3, input_dim=2)
  State bounds available: True
  Input bounds available: True
  Input bounds: u_min=[0. -2.], u_max=[2. 2.]

[1.6] Verifying setup...
✓ Twin step test passed: x_next shape = (3,)
✓ Plant step test passed: x_next shape = (3,)
✓ Plant-twin mismatch: 5.000000e-02

[1.7] Setting up output directory...
✓ Output directory: results/unicycle

================================================================================
✓ STEP 1 COMPLETE: Setup
================================================================================

Summary:
  - Digital twin: UnicycleTwin
  - Plant model: UnicyclePlant
  - Workspace: Workspace2D
  - Obstacles: 2
  - Constraints: UnicycleConstraints
  - Output directory: results/unicycle

Next step: Phase 1 - Nominal Planning
  (To be implemented)

[SAVE] Saving setup summary...
✓ Summary saved to: results/unicycle/step1_setup_summary.txt

================================================================================
STEP 1 COMPLETE - Ready for Phase 1 (Nominal Planning)
================================================================================
```

### Generated Files

After running, the following files are created:

```
results/
└── unicycle/
    └── step1_setup_summary.txt   # Setup verification report
```

## 🔧 Configuration

The pipeline uses `config/ddfs_config.yaml` for all parameters.

To switch systems, edit the config:
```yaml
system:
  active: "unicycle"  # or "quadrotor"
```

## 📝 What Step 1 Does

### 1.1 - Digital Twin
- Creates nominal model for planning
- Unicycle: kinematic model (3 states, 2 inputs)
- Quadrotor: full dynamics with quaternions (13 states, 4 inputs)

### 1.2 - Plant Model
- Creates real system with mismatch
- Unicycle: velocity scaling, angular scaling, slip
- Quadrotor: mass, inertia, drag, thrust efficiency

### 1.3 - Workspace
- Defines valid operating region
- Unicycle: 2D rectangle
- Quadrotor: 3D cuboid

### 1.4 - Obstacles
- Creates collision objects to avoid
- Unicycle: circles
- Quadrotor: spheres

### 1.5 - Constraints
- State and input bounds
- Ensures physical feasibility

### 1.6 - Verification
- Tests that models can step forward
- Computes plant-twin mismatch

### 1.7 - Output Setup
- Creates results directory
- Saves summary report

## 🚀 Next Steps

### Adding Step 2 (Phase 1: Nominal Planning)
```python
# To be added next:
# - Create SCvx planner
# - Generate nominal trajectory from x0 to xf
# - Avoid obstacles
# - Satisfy constraints
```

### Adding Step 3 (Phase 2: Data Collection)
```python
# To be added:
# - Collect M trajectories using plant
# - Add excitation signals
# - Segment trajectories
# - Build Hankel matrices
```

And so on for remaining phases...

## 🐛 Troubleshooting

### Config not found
```
FileNotFoundError: Configuration file not found: config/ddfs_config.yaml
```
**Solution:** Create config file or run from project root

### Import errors
```
ModuleNotFoundError: No module named 'ddfs'
```
**Solution:** Install package: `pip install -e .`

### Wrong system type
```
ValueError: Unknown system type: invalid
```
**Solution:** Set `system.active` to "unicycle" or "quadrotor" in config

## 📚 Related Documentation

- `CONFIG_REFERENCE.md` - Configuration guide
- `INTEGRATION_TESTS.md` - Testing documentation
- `VISUALIZATION_GUIDE.md` - Plotting utilities