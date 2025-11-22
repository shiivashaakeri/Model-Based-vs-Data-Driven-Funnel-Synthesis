# DDFS Project Outline
## Data-Driven Funnel Synthesis Pipeline

---

## 📁 Project Structure

```
ddfs/
├── config/                          # Configuration files
│   └── ddfs_config.yaml             # Main configuration (system, planning, environment)
│
├── ddfs/                            # Main package
│   ├── __init__.py
│   │
│   ├── models/                      # System dynamics models
│   │   ├── __init__.py
│   │   ├── base.py                  # Base classes: DynamicsModel, PlantModel
│   │   ├── unicycle.py              # Unicycle model (2D)
│   │   ├── quadrotor.py              # Quadrotor model (3D)
│   │   └── plant.py                 # Physical plant models
│   │
│   ├── planning/                    # Trajectory planning
│   │   ├── __init__.py
│   │   ├── nominal_trajectory.py   # NominalTrajectory class
│   │   └── scvx.py                  # SCvx planner with obstacle avoidance
│   │
│   ├── data_collection/             # Phase 2: Data collection
│   │   ├── __init__.py
│   │   ├── collector.py             # DataCollector, Trajectory, ExcitationSignalGenerator
│   │   ├── segmenter.py             # TrajectorySegmenter, SegmentedData
│   │   └── hankel.py                # HankelMatrixBuilder, SegmentHankelMatrices
│   │
│   ├── uncertainty/                 # Phase 3: Uncertainty quantification
│   │   ├── __init__.py
│   │   ├── quantifier.py            # Uncertainty quantifier
│   │   └── constants.py             # Uncertainty constants and bounds
│   │
│   ├── feasibility/                 # Phase 5: Feasibility checking
│   │   ├── __init__.py
│   │   └── ellipsoid_solver.py      # Ellipsoid feasibility solver
│   │
│   ├── core/                        # Core components
│   │   ├── __init__.py
│   │   ├── obstacles.py             # Obstacle definitions (Obstacle, CircleObstacle, SphereObstacle)
│   │   ├── workspace.py             # Workspace boundaries
│   │   ├── constraints.py           # System constraints (SystemConstraints, UnicycleConstraints, QuadrotorConstraints)
│   │   └── config.py                # DDFSConfig for configuration management
│   │
│   ├── visualization/               # Visualization tools
│   │   ├── __init__.py
│   │   ├── plotters.py              # TrajectoryPlotter base class
│   │   ├── unicycle_viz.py          # UnicyclePlotter for 2D visualization
│   │   └── quadrotor_viz.py         # QuadrotorPlotter for 3D visualization
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── config_loader.py          # Configuration loading (load_config)
│       └── factory.py                # System factory (SystemBundle, create_system_from_config)
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_models.py               # Model tests
│   ├── test_planning.py             # Planning tests
│   ├── test_data_collection.py      # Data collection tests
│   ├── test_uncertainty.py          # Uncertainty tests
│   ├── test_feasibility.py         # Feasibility tests
│   └── test_core.py                 # Core components tests
│
├── results/                         # Output directory (system-specific)
│   ├── unicycle/
│   │   └── phase1_nominal/          # Phase 1 results for unicycle
│   └── quadrotor/
│       └── phase1_nominal/           # Phase 1 results for quadrotor
│
├── run_ddfs.py                      # Main pipeline runner
├── setup.py                         # Package setup
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation
```

---

## 🔄 Pipeline Phases

### **Phase 1: Nominal Trajectory Planning** ✅ (Implemented)
- **Module**: `planning/`
- **Purpose**: Generate feasible nominal trajectory from x₀ to xf with obstacle avoidance
- **Components**:
  - `scvx.py`: Successive Convexification planner
  - `nominal_trajectory.py`: Trajectory data structure
- **Output**: `results/{system}/phase1_nominal/nominal_trajectory.pkl`

### **Phase 2: Offline Data Collection** 📝 (To be implemented)
- **Module**: `data_collection/`
- **Purpose**: Collect M trajectories from plant with excitation signals
- **Components**:
  - `collector.py`: Collect trajectories with excitation
  - `segmenter.py`: Segment trajectories into time windows
  - `hankel.py`: Build Hankel matrices from segments
- **Output**: Segmented data and Hankel matrices

### **Phase 3: Uncertainty Quantification** 📝 (To be implemented)
- **Module**: `uncertainty/`
- **Purpose**: Quantify model uncertainty from collected data
- **Components**:
  - `quantifier.py`: Uncertainty quantification algorithms
  - `constants.py`: Uncertainty bounds and constants
- **Output**: Uncertainty bounds for each segment

### **Phase 4: Funnel Synthesis** 📝 (To be implemented)
- **Module**: `synthesis/` (not yet created)
- **Purpose**: Synthesize robust control funnels using SDP
- **Components**:
  - `controller.py`: Controller synthesis
  - `lmi_matrices.py`: LMI matrix construction
  - `sdp_solver.py`: SDP optimization solver
- **Output**: Funnel parameters and controllers

### **Phase 5: Feasibility Checking** 📝 (To be implemented)
- **Module**: `feasibility/`
- **Purpose**: Verify funnel feasibility and constraint satisfaction
- **Components**:
  - `ellipsoid_solver.py`: Ellipsoid-based feasibility solver
- **Output**: Feasibility verification results

### **Phase 6: Deployment** 📝 (To be implemented)
- **Module**: `deployment/` (not yet created)
- **Purpose**: Deploy controller with safety monitoring
- **Components**:
  - `simulator.py`: Simulation environment
  - `safety_monitor.py`: Real-time safety monitoring
- **Output**: Deployment results and safety logs

---

## 🎯 Key Modules

### **Models** (`ddfs/models/`)
- **Base Classes**:
  - `DynamicsModel`: Abstract base for system dynamics
  - `TwinModel`: Base class for digital twin models
  - `PlantModel`: Physical plant model interface
- **Implementations**:
  - `UnicycleTwin`: 2D unicycle digital twin (x, y, θ)
  - `QuadrotorTwin`: 3D quadrotor digital twin (13 states)
  - `UnicyclePlant`: Physical unicycle plant
  - `QuadrotorPlant`: Physical quadrotor plant

### **Planning** (`ddfs/planning/`)
- **SCvxPlanner**: Successive Convexification with obstacle avoidance
- **NominalTrajectory**: Trajectory data structure (states, inputs, timesteps)

### **Core** (`ddfs/core/`)
- **Obstacles**: `Obstacle`, `CircleObstacle`, `SphereObstacle` classes
- **Workspace**: `Workspace` class for environment boundaries
- **Constraints**: `SystemConstraints`, `UnicycleConstraints`, `QuadrotorConstraints`
- **Config**: `DDFSConfig` class for configuration management

### **Visualization** (`ddfs/visualization/`)
- **TrajectoryPlotter**: Base class for trajectory visualization
- **UnicyclePlotter**: 2D trajectory plotting for unicycle systems
- **QuadrotorPlotter**: 3D trajectory plotting for quadrotor systems

### **Utils** (`ddfs/utils/`)
- **Config Loading**: `load_config()` function for YAML configuration parsing
- **Factory**: `SystemBundle` and `create_system_from_config()` for system creation

---

## 📊 Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ **Complete** | Nominal planning with SCvx + obstacles |
| Phase 2 | 📝 **Planned** | Data collection infrastructure ready |
| Phase 3 | 📝 **Planned** | Uncertainty quantification ready |
| Phase 4 | 📝 **Planned** | Funnel synthesis ready |
| Phase 5 | 📝 **Planned** | Feasibility checking ready |
| Phase 6 | 📝 **Planned** | Deployment ready |

---

## 🚀 Usage

### Run Pipeline
```bash
# Run Phase 1 (nominal planning)
python run_ddfs.py

# Run with custom config
python run_ddfs.py --config my_config.yaml

# Run specific phase (when implemented)
python run_ddfs.py --phase 2
```

### Configuration
- **System Selection**: Set `system.active` in `config/ddfs_config.yaml`
  - Options: `"unicycle"` or `"quadrotor"`
- **Output Organization**: Results saved to `results/{system}/phase{N}_{name}/`

---

## 📦 Dependencies

See `requirements.txt` for full list. Key dependencies:
- `numpy`: Numerical computations
- `cvxpy`: Convex optimization
- `jax`: JAX for plant models (optional)
- `matplotlib`: Visualization
- `pyyaml`: Configuration parsing

---

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Test coverage:
- ✅ Models (unicycle, quadrotor)
- ✅ Planning (SCvx)
- ✅ Core (obstacles, workspace, constraints, config)
- 📝 Data collection
- 📝 Uncertainty quantification
- 📝 Feasibility

---

## 📝 Notes

- **System-Specific Results**: Results are organized by system type (`unicycle/`, `quadrotor/`)
- **Modular Design**: Each phase is self-contained and can be run independently
- **Extensible**: Easy to add new systems or modify existing phases

