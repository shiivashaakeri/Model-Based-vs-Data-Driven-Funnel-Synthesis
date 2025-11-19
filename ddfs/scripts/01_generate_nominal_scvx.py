"""
Generate nominal trajectory using SCvx planner.

This script:
1. Loads configuration files
2. Creates unicycle model and workspace with obstacles
3. Runs SCvx trajectory optimization
4. Validates the resulting trajectory
5. Saves trajectory to disk

Usage:
    python scripts/01_generate_nominal_scvx.py
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ddfs.environment.collision import CollisionChecker  # noqa: E402
from ddfs.environment.obstacles import CircularObstacle, EllipsoidalObstacle  # noqa: E402
from ddfs.environment.workspace import Workspace  # noqa: E402
from ddfs.models.unicycle import UnicycleModel  # noqa: E402
from ddfs.planning.constraints import BoxConstraints, ConstraintValidator  # noqa: E402
from ddfs.planning.convexification import DynamicsLinearizer  # noqa: E402
from ddfs.planning.scvx_planner import SCvxPlanner  # noqa: E402
from ddfs.utils.config_loader import ExperimentConfig  # noqa: E402


def create_obstacles_from_config(obstacle_configs):
    """
    Create obstacle objects from configuration.

    Args:
        obstacle_configs: List of obstacle configuration dicts

    Returns:
        obstacles: List of Obstacle objects
    """
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

        else:
            print(f"Warning: Unknown obstacle type '{obs_type}', skipping")

    return obstacles


def validate_trajectory(x_traj, u_traj, config, collision_checker, model):  # noqa: PLR0912, PLR0915
    """
    Validate the planned trajectory.

    Args:
        x_traj: State trajectory (N+1, 3)
        u_traj: Input trajectory (N, 2)
        config: ExperimentConfig object
        collision_checker: CollisionChecker object
        model: UnicycleModel object

    Returns:
        valid: True if all checks pass
        report: Validation report dict
    """
    print("\n" + "=" * 70)
    print("TRAJECTORY VALIDATION")
    print("=" * 70)

    report = {}
    all_valid = True

    # 1. Check for collisions
    print("\n1. Collision Check:")
    collision, timestep, obs_idx = collision_checker.check_trajectory_collision(x_traj)

    if collision:
        print(f"  COLLISION detected at timestep {timestep} with obstacle {obs_idx}")
        all_valid = False
        report["collision"] = {"detected": True, "timestep": timestep, "obstacle": obs_idx}
    else:
        print("  No collisions detected")
        report["collision"] = {"detected": False}

    # 2. Compute minimum clearance
    min_clearance, clearance_timestep, clearance_obs = collision_checker.get_trajectory_clearance(x_traj)
    print("\n2. Obstacle Clearance:")
    print(f"   Minimum clearance: {min_clearance:.4f} m at timestep {clearance_timestep}")

    min_acceptable = config.environment.get("safety", {}).get("minimum_clearance", 0.1)
    if min_clearance < min_acceptable:
        print(f"  Warning: Clearance below minimum acceptable ({min_acceptable} m)")
    else:
        print(f"  Clearance above minimum ({min_acceptable} m)")

    report["clearance"] = {
        "minimum": float(min_clearance),
        "timestep": int(clearance_timestep),
        "obstacle": int(clearance_obs),
    }

    # 3. Check state constraints
    print("\n3. State Constraints:")
    x_min, x_max = config.get_state_bounds()
    state_constraints = BoxConstraints(x_min, x_max, name="state")

    valid_state, state_report = ConstraintValidator.validate_state_trajectory(x_traj, state_constraints, tolerance=1e-6)

    if valid_state:
        print("  All state constraints satisfied")
    else:
        print(f"  State constraints violated at {len(state_report['violation_timesteps'])} timesteps")
        print(f"      Max violation: {state_report['max_violation']:.6f}")
        all_valid = False

    report["state_constraints"] = state_report

    # 4. Check input constraints
    print("\n4. Input Constraints:")
    u_min, u_max = config.get_input_bounds()
    input_constraints = BoxConstraints(u_min, u_max, name="input")

    valid_input, input_report = ConstraintValidator.validate_input_trajectory(u_traj, input_constraints, tolerance=1e-6)

    if valid_input:
        print("  All input constraints satisfied")
    else:
        print(f"  Input constraints violated at {len(input_report['violation_timesteps'])} timesteps")
        print(f"      Max violation: {input_report['max_violation']:.6f}")
        all_valid = False

    report["input_constraints"] = input_report

    # 5. Check goal reaching
    print("\n5. Goal Reaching:")
    xf = config.get_goal_state()
    goal_error = np.linalg.norm(x_traj[-1] - xf)

    max_goal_error = config.scvx.get("validation", {}).get("max_goal_error", 0.1)

    if goal_error < max_goal_error:
        print(f"  Goal reached (error: {goal_error:.6f} m)")
    else:
        print(f"  Goal error: {goal_error:.6f} m (max acceptable: {max_goal_error} m)")

    report["goal_error"] = float(goal_error)

    # 6. Check dynamics consistency (linearization error)
    print("\n6. Dynamics Consistency:")
    linearizer = DynamicsLinearizer(model, dt=config.get_dt(), method="rk4")
    errors = linearizer.compute_linearization_error(x_traj, u_traj)
    max_error = np.max(errors)
    mean_error = np.mean(errors)

    max_acceptable = config.scvx.get("validation", {}).get("max_linearization_error", 0.01)

    print(f"   Max linearization error:  {max_error:.6f}")
    print(f"   Mean linearization error: {mean_error:.6f}")

    if max_error < max_acceptable:
        print("  Linearization errors acceptable")
    else:
        print(f"  Max error above threshold ({max_acceptable})")

    report["linearization"] = {"max_error": float(max_error), "mean_error": float(mean_error)}

    # Overall result
    print("\n" + "=" * 70)
    if all_valid:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED - Issues detected")
    print("=" * 70 + "\n")

    report["overall_valid"] = all_valid

    return all_valid, report


def save_trajectory(x_traj, u_traj, config, metadata, output_dir="data/nominal_trajectories"):
    """
    Save trajectory to disk in organized folder.

    Args:
        x_traj: State trajectory
        u_traj: Input trajectory
        config: ExperimentConfig object
        metadata: Additional metadata dict
        output_dir: Output directory

    Returns:
        filepath: Path to saved file
        folder_path: Path to trajectory folder
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate folder name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"unicycle_nominal_{timestamp}"
    folder_path = output_path / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = f"{folder_name}.pkl"
    filepath = folder_path / filename

    # Prepare data
    data = {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "dt": config.get_dt(),
        "N": config.get_horizon(),
        "x0": config.get_initial_state(),
        "xf": config.get_goal_state(),
        "x_bounds": config.get_state_bounds(),
        "u_bounds": config.get_input_bounds(),
        "obstacles": config.get_obstacles(),
        "workspace_bounds": config.get_workspace_bounds(),
        "timestamp": timestamp,
        "folder_name": folder_name,
        "metadata": metadata,
    }

    # Save
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"\n💾 Trajectory saved to: {filepath}")
    print(f"   Folder: {folder_path}")
    print(f"   File size: {filepath.stat().st_size / 1024:.2f} KB")

    return filepath, folder_path


def print_statistics(x_traj, u_traj, config):
    """Print trajectory statistics."""
    print("\n" + "=" * 70)
    print("TRAJECTORY STATISTICS")
    print("=" * 70)

    N = len(u_traj)
    dt = config.get_dt()

    # Distance traveled
    distances = np.linalg.norm(np.diff(x_traj[:, :2], axis=0), axis=1)
    total_distance = np.sum(distances)

    # Velocity statistics
    v_mean = np.mean(u_traj[:, 0])
    v_max = np.max(u_traj[:, 0])
    v_min = np.min(u_traj[:, 0])

    # Angular velocity statistics
    omega_mean = np.mean(np.abs(u_traj[:, 1]))
    omega_max = np.max(np.abs(u_traj[:, 1]))

    # Heading changes
    theta_changes = np.abs(np.diff(x_traj[:, 2]))
    total_heading_change = np.sum(theta_changes)

    print("\nTrajectory:")
    print(f"  Duration:        {N * dt:.2f} seconds ({N} steps)")
    print(f"  Distance:        {total_distance:.3f} m")
    print(f"  Avg speed:       {total_distance / (N * dt):.3f} m/s")

    print("\nLinear Velocity:")
    print(f"  Mean:            {v_mean:.3f} m/s")
    print(f"  Range:           [{v_min:.3f}, {v_max:.3f}] m/s")

    print("\nAngular Velocity:")
    print(f"  Mean (abs):      {omega_mean:.3f} rad/s")
    print(f"  Max (abs):       {omega_max:.3f} rad/s")

    print("\nHeading:")
    print(f"  Total change:    {total_heading_change:.3f} rad ({np.degrees(total_heading_change):.1f}°)")

    print("=" * 70 + "\n")


def main():  # noqa: PLR0915
    """Main execution function."""
    print("NOMINAL TRAJECTORY GENERATION - SCvx PLANNER")


    # 1. Load configuration
    print(" Loading configuration...")
    try:
        config = ExperimentConfig()
        config.print_summary()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # 2. Create unicycle model
    print("\n Creating unicycle model...")
    x0 = config.get_initial_state()
    xf = config.get_goal_state()
    model = UnicycleModel(x0=x0, xf=xf)
    print(f"   Model: {model}")

    # 3. Create obstacles and workspace
    print("\n Setting up environment...")
    obstacles = create_obstacles_from_config(config.get_obstacles())
    print(f"   Created {len(obstacles)} obstacles:")
    for i, obs in enumerate(obstacles):
        print(f"     {i + 1}. {obs}")

    collision_checker = CollisionChecker(obstacles)

    workspace_bounds = config.get_workspace_bounds()
    workspace = Workspace(bounds=workspace_bounds, obstacles=obstacles)
    print(f"   Workspace: {workspace}")

    # 4. Create SCvx planner
    print("\n  Initializing SCvx planner...")
    scvx_params = config.get_scvx_params()

    # Add additional parameters from config
    scvx_params["initialization_method"] = config.scvx.get("initialization", {}).get("method", "straight_line")
    scvx_params["detour_margin"] = config.scvx.get("initialization", {}).get("detour_margin", 1.5)
    scvx_params["use_slack"] = config.scvx.get("obstacles", {}).get("use_slack", True)
    scvx_params["slack_penalty"] = config.scvx.get("obstacles", {}).get("slack_penalty", 1e5)
    scvx_params["max_solver_iters"] = config.scvx.get("solver", {}).get("max_iters", 1000)
    scvx_params["solver_abstol"] = config.scvx.get("solver", {}).get("abstol", 1e-6)
    scvx_params["solver_reltol"] = config.scvx.get("solver", {}).get("reltol", 1e-5)
    scvx_params["solver_feastol"] = config.scvx.get("solver", {}).get("feastol", 1e-6)

    planner = SCvxPlanner(
        model=model, dt=config.get_dt(), N=config.get_horizon(), collision_checker=collision_checker, params=scvx_params
    )
    print(f"   Planner initialized with {config.get_horizon()} timesteps")
    print(f"   Initialization method: {scvx_params['initialization_method']}")
    print(f"   Using slack variables: {scvx_params['use_slack']}")

    # 5. Run planning
    print("\n Running SCvx optimization...\n")

    x_bounds = config.get_state_bounds()
    u_bounds = config.get_input_bounds()

    try:
        x_traj, u_traj, converged = planner.plan(x0=x0, xf=xf, x_bounds=x_bounds, u_bounds=u_bounds)
    except Exception as e:
        print(f"\n Planning failed with error: {e}")
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        return 1

    # 6. Check convergence
    if converged:
        print("\n SCvx converged successfully!")
    else:
        print("\n SCvx did not converge (max iterations reached)")
        print("    Trajectory may not be optimal, but will continue with validation...")

    # 7. Get planning history
    history = planner.get_history()
    print("\n Planning Statistics:")
    print(f"   Iterations:      {len(history['cost'])}")
    if len(history["cost"]) > 0:
        print(f"   Final cost:      {history['cost'][-1]:.4f}")
        print(f"   Initial cost:    {history['cost'][0]:.4f}")
        print(f"   Cost reduction:  {history['cost'][0] - history['cost'][-1]:.4f}")

    # 8. Validate trajectory
    valid, validation_report = validate_trajectory(x_traj, u_traj, config, collision_checker, model)

    # 9. Print statistics
    print_statistics(x_traj, u_traj, config)

    # 10. Save trajectory
    metadata = {
        "converged": converged,
        "iterations": len(history["cost"]),
        "validation": validation_report,
        "scvx_history": {
            "cost": [float(c) for c in history["cost"]],
            "trust_region_rho": [float(r) for r in history["trust_region_rho"]],
        },
    }

    output_dir = config.scvx.get("output", {}).get("output_dir", "data/nominal_trajectories")
    filepath, folder_path = save_trajectory(x_traj, u_traj, config, metadata, output_dir)

    # 11. Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f" Planning:      {'Converged' if converged else 'Max iterations reached'}")
    print(f" Validation:    {'Passed' if valid else 'Failed (see warnings above)'}")
    print(f" Trajectory:    Saved to {folder_path.name}/")
    print("=" * 70)

    print("\n Next steps:")
    print(f"   1. Run: python scripts/01b_visualize_nominal.py {filepath}")
    print("   2. Review trajectory visualization (plots will be saved in the same folder)")
    print("   3. If satisfied, proceed to data collection (Phase 4)")

    print("\nNOMINAL TRAJECTORY GENERATION COMPLETE")

    return 0 if (converged and valid) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
