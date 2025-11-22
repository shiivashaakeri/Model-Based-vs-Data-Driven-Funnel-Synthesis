#!/usr/bin/env python3
"""
DDFS Pipeline Runner - Phase 1: Nominal Planning

This script runs the Data-Driven Funnel Synthesis pipeline starting with
Phase 1: Nominal trajectory planning using SCvx with obstacle avoidance.

Usage:
    python run_ddfs.py [--config CONFIG_PATH] [--phase PHASE]

Currently implemented:
    Phase 1: Nominal Planning (SCvx with obstacles)

To be implemented:
    Phase 2-6: Data collection, uncertainty quantification, funnel synthesis, etc.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ddfs.models.quadrotor import QuadrotorConstraints, QuadrotorTwin
from ddfs.models.unicycle import UnicycleConstraints, UnicycleTwin
from ddfs.planning import Obstacle, SCvxParameters, SCvxPlanner
from ddfs.visualization import DDFSPlotter


class DDFSRunner:
    """
    Main runner class for DDFS pipeline.

    Orchestrates all phases of the pipeline from configuration.
    """

    def __init__(self, config_path: Path):
        """
        Initialize runner with configuration.

        Parameters
        ----------
        config_path : Path
            Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

        # Setup output directory
        self.output_dir = Path(self.config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine active system
        self.system_type = self.config["system"]["active"]

        # Create system-specific subdirectories
        self.system_dir = self.output_dir / self.system_type
        self.system_dir.mkdir(exist_ok=True)

        # Create subdirectories for each phase
        self.phase1_dir = self.system_dir / "phase1_nominal"
        self.phase1_dir.mkdir(exist_ok=True)

        # Initialize plotter
        self.plotter = DDFSPlotter(figsize=(12, 8), dpi=150)
        print(f"\n{'=' * 70}")
        print("DDFS Pipeline Runner")
        print(f"{'=' * 70}")
        print(f"Experiment: {self.config['experiment']['name']}")
        print(f"System:     {self.system_type}")
        print(f"Output:     {self.system_dir}")
        print(f"{'=' * 70}\n")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _get_system_config(self) -> Dict[str, Any]:
        """Get configuration for active system."""
        return self.config["system"][self.system_type]

    def _get_planning_config(self) -> Dict[str, Any]:
        """Get planning configuration for active system."""
        return self.config["planning"][self.system_type]

    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration for active system."""
        return self.config["environment"][self.system_type]

    def _create_twin_model(self):
        """Create digital twin model based on configuration."""
        sys_config = self._get_system_config()

        if self.system_type == "unicycle":
            twin = UnicycleTwin(dt=sys_config["dt"])
            constraints = UnicycleConstraints.from_config(sys_config)
            return twin, constraints

        elif self.system_type == "quadrotor":
            twin_params = sys_config["twin_params"]
            twin = QuadrotorTwin(mass=twin_params["mass"], inertia=np.diag(twin_params["inertia"]), dt=sys_config["dt"])
            constraints = QuadrotorConstraints.from_config(sys_config)
            return twin, constraints

        else:
            raise ValueError(f"Unknown system type: {self.system_type}")

    def _create_obstacles(self) -> Tuple[list, list]:
        """
        Create obstacle objects from configuration.

        Returns
        -------
        obstacles : list
            List of Obstacle objects for planner
        obstacle_dicts : list
            List of obstacle dictionaries for plotting
        """
        env_config = self._get_environment_config()
        obstacles = []
        obstacle_dicts = []

        for obs_config in env_config["obstacles"]:
            center = np.array(obs_config["center"])
            radius = obs_config["radius"]

            # Create Obstacle object for planner
            obs = Obstacle(center=center, radius=radius)
            obstacles.append(obs)

            # Create dict for plotting
            obs_dict = {
                "id": obs_config["id"],
                "type": obs_config["type"],
                "center": center,
                "radius": radius,
                "safety_margin": obs_config.get("safety_margin", 0.0),
            }
            obstacle_dicts.append(obs_dict)

        return obstacles, obstacle_dicts

    def _setup_scvx_planner(self, twin, constraints, obstacles):
        """Setup SCvx planner with constraints and obstacles."""
        sys_config = self._get_system_config()

        # Get state and input bounds
        if self.system_type == "unicycle":
            state_bounds = (
                np.array(
                    [
                        sys_config["state_bounds"]["x_min"],
                        sys_config["state_bounds"]["y_min"],
                        sys_config["state_bounds"]["theta_min"],
                    ]
                ),
                np.array(
                    [
                        sys_config["state_bounds"]["x_max"],
                        sys_config["state_bounds"]["y_max"],
                        sys_config["state_bounds"]["theta_max"],
                    ]
                ),
            )
            position_indices = [0, 1]  # x, y

        elif self.system_type == "quadrotor":
            state_bounds = (
                np.array(
                    [
                        sys_config["state_bounds"]["x_min"],
                        sys_config["state_bounds"]["y_min"],
                        sys_config["state_bounds"]["z_min"],
                        -sys_config["state_bounds"]["v_max"],
                        -sys_config["state_bounds"]["v_max"],
                        -sys_config["state_bounds"]["v_max"],
                        -1,
                        -1,
                        -1,
                        -1,  # Quaternion bounds (will be normalized)
                        -sys_config["state_bounds"]["omega_max"],
                        -sys_config["state_bounds"]["omega_max"],
                        -sys_config["state_bounds"]["omega_max"],
                    ]
                ),
                np.array(
                    [
                        sys_config["state_bounds"]["x_max"],
                        sys_config["state_bounds"]["y_max"],
                        sys_config["state_bounds"]["z_max"],
                        sys_config["state_bounds"]["v_max"],
                        sys_config["state_bounds"]["v_max"],
                        sys_config["state_bounds"]["v_max"],
                        1,
                        1,
                        1,
                        1,  # Quaternion bounds
                        sys_config["state_bounds"]["omega_max"],
                        sys_config["state_bounds"]["omega_max"],
                        sys_config["state_bounds"]["omega_max"],
                    ]
                ),
            )
            position_indices = [0, 1, 2]  # x, y, z

        input_bounds = (constraints.u_min, constraints.u_max)

        # SCvx parameters
        scvx_params = SCvxParameters(
            max_iterations=30, convergence_tol=1e-3, trust_region_init=1.0, weight_nu=1e5, verbose=True, solver="ECOS"
        )

        # Create planner
        planner = SCvxPlanner(
            model=twin,
            params=scvx_params,
            state_bounds=state_bounds,
            input_bounds=input_bounds,
            obstacles=obstacles,
            position_indices=position_indices,
        )

        return planner

    def run_phase1_nominal_planning(self):  # noqa: PLR0915
        """
        Phase 1: Nominal trajectory planning using SCvx.

        Plans a feasible trajectory on the digital twin from x0 to xf
        while avoiding obstacles and satisfying constraints.
        """
        print(f"\n{'=' * 70}")
        print("PHASE 1: NOMINAL TRAJECTORY PLANNING")
        print(f"{'=' * 70}\n")

        # Create twin model and constraints
        print("Setting up system...")
        twin, constraints = self._create_twin_model()
        print(f"  ✓ Twin model: {twin}")
        print(f"  ✓ Constraints: {constraints}")

        # Create obstacles
        print("\nSetting up obstacles...")
        obstacles, obstacle_dicts = self._create_obstacles()
        print(f"  ✓ Loaded {len(obstacles)} obstacles")
        for obs_dict in obstacle_dicts:
            print(f"    - {obs_dict['id']}: center={obs_dict['center']}, r={obs_dict['radius']}")

        # Setup planner
        print("\nSetting up SCvx planner...")
        planner = self._setup_scvx_planner(twin, constraints, obstacles)
        print("  ✓ Planner configured")

        # Get planning parameters
        plan_config = self._get_planning_config()
        x0 = np.array(plan_config["x0"])
        xf = np.array(plan_config["xf"])
        N = plan_config["N"]
        dt = self._get_system_config()["dt"]

        print("\nPlanning parameters:")
        print(f"  x0: {x0}")
        print(f"  xf: {xf}")
        print(f"  N:  {N}")
        print(f"  dt: {dt:.4f}s")
        print(f"  tf: {N * dt:.2f}s")

        # Run planner
        print(f"\n{'-' * 70}")
        print("Running SCvx planner...")
        print(f"{'-' * 70}")

        trajectory = planner.plan(x0, xf, N, dt)

        print(f"\n{'-' * 70}")
        print("✓ Planning complete!")
        print(f"{'-' * 70}")
        print(f"  Trajectory: {trajectory}")

        # Save trajectory
        traj_path = self.phase1_dir / "nominal_trajectory.pkl"
        trajectory.save(traj_path)
        print(f"\n✓ Saved trajectory: {traj_path}")

        # Verify no collisions
        print(f"\n{'-' * 70}")
        print("Verifying obstacle avoidance...")
        print(f"{'-' * 70}")

        collisions = 0
        min_clearance = float("inf")

        pos_indices = [0, 1] if self.system_type == "unicycle" else [0, 1, 2]

        for k in range(N + 1):
            pos = trajectory.x_nom[k, pos_indices]
            for obs, obs_dict in zip(obstacles, obstacle_dicts):
                dist = np.linalg.norm(pos - obs.center) - obs.radius
                min_clearance = min(min_clearance, dist)
                if dist < 0:
                    collisions += 1
                    print(f"  ⚠ COLLISION at k={k}: distance to {obs_dict['id']} = {dist:.3f}m")

        if collisions == 0:
            print("  ✓ No collisions detected!")
            print(f"  ✓ Minimum clearance: {min_clearance:.3f}m")
        else:
            print(f"  ✗ WARNING: {collisions} collision(s) detected!")

        # Generate visualizations
        print(f"\n{'-' * 70}")
        print("Generating visualizations...")
        print(f"{'-' * 70}")

        env_config = self._get_environment_config()
        sys_config = self._get_system_config()

        # Spatial trajectory plot
        if self.system_type == "unicycle":
            # 2D plot
            spatial_path = self.phase1_dir / "nominal_trajectory_2d.png"
            self.plotter.plot_nominal_2d(
                trajectory=trajectory,
                obstacles=obstacle_dicts,
                workspace=env_config["workspace"],
                save_path=spatial_path,
                title="Unicycle Nominal Trajectory",
            )
        else:
            # 3D plot
            spatial_path = self.phase1_dir / "nominal_trajectory_3d.png"
            self.plotter.plot_nominal_3d(
                trajectory=trajectory,
                obstacles=obstacle_dicts,
                workspace=env_config["workspace"],
                save_path=spatial_path,
                title="Quadrotor Nominal Trajectory",
            )

        # State trajectories
        if self.system_type == "unicycle":
            state_labels = ["x", "y", "θ"]
        else:
            state_labels = ["x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "ωx", "ωy", "ωz"]

        states_path = self.phase1_dir / "nominal_states.png"
        self.plotter.plot_states_vs_time(
            trajectory=trajectory,
            state_bounds=sys_config["state_bounds"],
            state_labels=state_labels,
            save_path=states_path,
            title=f"{self.system_type.capitalize()} State Trajectories",
        )

        # Input trajectories
        input_labels = ["v", "ω"] if self.system_type == "unicycle" else ["T", "τx", "τy", "τz"]

        inputs_path = self.phase1_dir / "nominal_inputs.png"
        self.plotter.plot_inputs_vs_time(
            trajectory=trajectory,
            input_bounds=sys_config["input_bounds"],
            input_labels=input_labels,
            save_path=inputs_path,
            title=f"{self.system_type.capitalize()} Input Trajectories",
        )

        # Summary plot
        summary_path = self.phase1_dir / "nominal_summary.png"
        self.plotter.plot_nominal_summary(
            trajectory=trajectory,
            obstacles=obstacle_dicts,
            workspace=env_config["workspace"],
            state_bounds=sys_config["state_bounds"],
            input_bounds=sys_config["input_bounds"],
            state_labels=state_labels,
            input_labels=input_labels,
            save_path=summary_path,
            system_name=self.system_type.capitalize(),
        )

        print(f"\n✓ Phase 1 complete! Results saved to: {self.phase1_dir}")

        return trajectory

    def run(self, phase: int = 1):
        """
        Run DDFS pipeline up to specified phase.

        Parameters
        ----------
        phase : int
            Phase to run (1-6)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nStarting DDFS Pipeline at {timestamp}")

        if phase >= 1:
            self.run_phase1_nominal_planning()

        if phase >= 2:
            print("\n⚠ Phase 2 (Offline Data Collection) not yet implemented")

        if phase >= 3:
            print("\n⚠ Phase 3+ not yet implemented")

        print(f"\n{'=' * 70}")
        print("DDFS Pipeline Complete!")
        print(f"{'=' * 70}")
        print(f"Results: {self.system_dir}")
        print(f"{'=' * 70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DDFS Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1 (nominal planning) with default config
  python run_ddfs.py

  # Run with custom config
  python run_ddfs.py --config my_config.yaml

  # Run multiple phases (when implemented)
  python run_ddfs.py --phase 3
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/ddfs_config.yaml"),
        help="Path to configuration YAML file (default: config/ddfs_config.yaml)",
    )

    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="Run pipeline up to this phase (default: 1)"
    )

    args = parser.parse_args()

    # Check config file exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        print("Please create a config file or specify path with --config")
        sys.exit(1)

    # Create runner and execute
    try:
        runner = DDFSRunner(config_path=args.config)
        runner.run(phase=args.phase)
    except Exception as e:
        print(f"\n{'=' * 70}")
        print("ERROR: Pipeline failed!")
        print(f"{'=' * 70}")
        print(f"{type(e).__name__}: {e}")
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
