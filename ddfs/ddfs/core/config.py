"""
Configuration Management System for DDFS.

This module provides a centralized configuration system for managing
hyperparameters, system parameters, solver settings, and experiment
configurations using YAML files with validation.

Features:
- YAML-based configuration files
- Hierarchical configuration merging (default -> system-specific)
- Validation of required parameters
- Type checking and bounds validation
- Easy access via dot notation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

# =============================================================================
# Data Classes for Structured Configuration
# =============================================================================


@dataclass
class SimulationConfig:
    """Simulation parameters."""

    dt: float = 0.02  # Timestep [s]
    N: int = 1000  # Horizon length [steps]
    t_final: float = field(init=False)  # Total time [s], computed

    def __post_init__(self):
        self.t_final = self.dt * self.N


@dataclass
class DDFSConfig:
    """Data-Driven Funnel Synthesis algorithm parameters."""

    T: int = 100  # Segment length [steps]
    L: int = 60  # Data window length [steps]
    alpha: float = 0.98  # Lyapunov decay rate
    mu: float = 1.02  # Cross-segment growth factor
    epsilon_bar: float = 0.15  # Excitation bound


@dataclass
class SolverConfig:
    """Optimization solver settings."""

    name: str = "MOSEK"  # Solver name
    verbose: bool = False  # Solver verbosity
    max_iters: int = 10000  # Maximum iterations
    eps_abs: float = 1e-8  # Absolute tolerance
    eps_rel: float = 1e-8  # Relative tolerance

    # SCvx-specific parameters
    scvx_max_iters: int = 50  # Maximum SCvx iterations
    scvx_tol: float = 1e-6  # SCvx convergence tolerance
    trust_region_init: float = 1.0  # Initial trust region radius
    trust_region_min: float = 1e-4  # Minimum trust region radius
    trust_region_max: float = 10.0  # Maximum trust region radius


@dataclass
class ObstacleConfig:
    """Obstacle specification."""

    center: List[float]  # Center coordinates
    radius: float  # Radius


@dataclass
class BoundsConfig:
    """State and input bounds."""

    x_min: List[float]  # State lower bounds
    x_max: List[float]  # State upper bounds
    u_min: List[float]  # Input lower bounds
    u_max: List[float]  # Input upper bounds


@dataclass
class SystemConfig:
    """System-specific configuration."""

    name: str  # System name
    n_states: int  # State dimension
    n_inputs: int  # Input dimension
    state_labels: List[str]  # State variable names
    input_labels: List[str]  # Input variable names

    # Initial and final conditions
    x_init: List[float]  # Initial state
    x_final: List[float]  # Final/goal state

    # Bounds
    bounds: BoundsConfig = None

    # Physical parameters (system-specific)
    params: Dict[str, Any] = field(default_factory=dict)

    # Obstacles
    obstacles: List[ObstacleConfig] = field(default_factory=list)

    # Mismatch parameters for plant vs twin
    mismatch: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LipschitzConfig:
    """Lipschitz constant estimation parameters."""

    n_samples: int = 1000  # Number of samples for estimation
    perturbation_scale: float = 1e-4  # Finite difference step size
    use_analytical: bool = False  # Use analytical bounds if available


@dataclass
class VisualizationConfig:
    """Visualization settings."""

    save_figures: bool = True
    figure_format: str = "png"
    dpi: int = 150
    show_plots: bool = True
    plot_funnels: bool = True
    plot_constraints: bool = True
    plot_obstacles: bool = True


@dataclass
class Config:
    """
    Master configuration container.

    This class holds all configuration parameters for the DDFS framework.
    """

    simulation: SimulationConfig
    ddfs: DDFSConfig
    solver: SolverConfig
    system: SystemConfig
    lipschitz: LipschitzConfig
    visualization: VisualizationConfig

    # Metadata
    config_name: str = "default"
    output_dir: str = "results"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        # Check DDFS parameters
        if self.ddfs.L > self.ddfs.T:
            raise ValueError(
                f"Data window L ({self.ddfs.L}) cannot exceed "
                f"segment length T ({self.ddfs.T})"
            )

        if not 0 < self.ddfs.alpha < 1:
            raise ValueError(
                f"Lyapunov decay rate alpha ({self.ddfs.alpha}) "
                f"must be in (0, 1)"
            )

        if self.ddfs.mu < 1:
            raise ValueError(
                f"Cross-segment growth factor mu ({self.ddfs.mu}) "
                f"must be >= 1"
            )

        # Check dwell time condition from Theorem 2
        dwell_time_min = -np.log(self.ddfs.mu) / np.log(self.ddfs.alpha)
        if dwell_time_min >= self.ddfs.T:
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"Segment length T ({self.ddfs.T}) should be > "
                f"{dwell_time_min:.2f} for stability (Theorem 2)"
            )

        # Check bounds dimensions match state/input dimensions
        if self.system.bounds is not None and len(self.system.bounds.x_min) != self.system.n_states:
            raise ValueError(
                f"State bounds dimension mismatch: "
                f"x_min has {len(self.system.bounds.x_min)} elements, "
                f"expected {self.system.n_states}"
            )

    def get_output_path(self, filename: str) -> Path:
        """Get full path for output file."""
        output_dir = Path(self.output_dir) / self.system.name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return _dataclass_to_dict(self)

    def save(self, filepath: Union[str, Path]):
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Configuration Loading Functions
# =============================================================================


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass to dictionary."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Values in `override` take precedence over `base`.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_bounds_config(d: Dict) -> BoundsConfig:
    """Convert dictionary to BoundsConfig."""
    return BoundsConfig(
        x_min=d.get("x_min", []),
        x_max=d.get("x_max", []),
        u_min=d.get("u_min", []),
        u_max=d.get("u_max", []),
    )


def _dict_to_obstacle_config(d: Dict) -> ObstacleConfig:
    """Convert dictionary to ObstacleConfig."""
    return ObstacleConfig(
        center=d["center"],
        radius=d["radius"],
    )


def _dict_to_system_config(d: Dict) -> SystemConfig:
    """Convert dictionary to SystemConfig."""
    bounds = None
    if "bounds" in d and d["bounds"] is not None:
        bounds = _dict_to_bounds_config(d["bounds"])

    obstacles = []
    if "obstacles" in d and d["obstacles"] is not None:
        obstacles = [_dict_to_obstacle_config(obs) for obs in d["obstacles"]]

    return SystemConfig(
        name=d["name"],
        n_states=d["n_states"],
        n_inputs=d["n_inputs"],
        state_labels=d.get("state_labels", []),
        input_labels=d.get("input_labels", []),
        x_init=d["x_init"],
        x_final=d["x_final"],
        bounds=bounds,
        params=d.get("params", {}),
        obstacles=obstacles,
        mismatch=d.get("mismatch", {}),
    )


def _dict_to_config(d: Dict) -> Config:
    """Convert dictionary to Config dataclass."""
    return Config(
        simulation=SimulationConfig(**d.get("simulation", {})),
        ddfs=DDFSConfig(**d.get("ddfs", {})),
        solver=SolverConfig(**d.get("solver", {})),
        system=_dict_to_system_config(d["system"]),
        lipschitz=LipschitzConfig(**d.get("lipschitz", {})),
        visualization=VisualizationConfig(**d.get("visualization", {})),
        config_name=d.get("config_name", "default"),
        output_dir=d.get("output_dir", "results"),
    )


def load_yaml(filepath: Union[str, Path]) -> Dict:
    """Load YAML file and return dictionary."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def load_config(
    config_path: Union[str, Path],
    default_path: Optional[Union[str, Path]] = None,
) -> Config:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to system-specific configuration file.
    default_path : str or Path, optional
        Path to default configuration file. If provided, the system-specific
        config is merged on top of defaults.

    Returns
    -------
    Config
        Validated configuration object.

    Example
    -------
    >>> config = load_config("config/unicycle.yaml", "config/default.yaml")
    >>> print(config.ddfs.alpha)
    0.98
    """
    # Load system-specific config
    system_config = load_yaml(config_path)

    # If default path provided, merge configs
    if default_path is not None:
        default_config = load_yaml(default_path)
        merged_config = _deep_merge(default_config, system_config)
    else:
        merged_config = system_config

    # Convert to Config object (includes validation)
    return _dict_to_config(merged_config)


def get_config_dir() -> Path:
    """Get the default configuration directory."""
    # Look for config directory relative to package root
    package_root = Path(__file__).parent.parent.parent
    config_dir = package_root / "config"

    if config_dir.exists():
        return config_dir

    # Fallback to current working directory
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists():
        return cwd_config

    raise FileNotFoundError(
        "Could not find config directory. "
        "Expected at package root or current working directory."
    )


def load_system_config(system_name: str) -> Config:
    """
    Convenience function to load configuration for a named system.

    Parameters
    ----------
    system_name : str
        Name of the system ('unicycle' or 'quadrotor').

    Returns
    -------
    Config
        Validated configuration object.

    Example
    -------
    >>> config = load_system_config("unicycle")
    """
    config_dir = get_config_dir()
    default_path = config_dir / "default.yaml"
    system_path = config_dir / f"{system_name}.yaml"

    if not system_path.exists():
        raise FileNotFoundError(
            f"Configuration file for system '{system_name}' not found at {system_path}"
        )

    return load_config(system_path, default_path)
