"""
Configuration loader utility for DDFS project.

Provides utilities to:
- Load YAML configuration files
- Merge multiple configs
- Validate configuration parameters
- Provide easy access to nested parameters
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml


class ConfigLoader:
    """
    Configuration loader for YAML files.

    Loads, validates, and provides access to configuration parameters.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files (default: ddfs/config/)
        """
        if config_dir is None:
            # Default to ddfs/config/ directory
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.config_dir = project_root / "config"
        else:
            self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

        self.configs = {}

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            config: Loaded configuration as dictionary
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Store in cache
        self.configs[config_name] = config

        return config

    def load_all(self, config_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Load multiple configuration files.

        Args:
            config_names: List of config names to load (default: all .yaml files)

        Returns:
            configs: Dictionary mapping config_name to config dict
        """
        if config_names is None:
            # Load all YAML files in config directory
            config_names = [f.stem for f in self.config_dir.glob("*.yaml")]

        configs = {}
        for name in config_names:
            configs[name] = self.load(name)

        return configs

    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation.

        Args:
            config_name: Name of config file
            key_path: Dot-separated path to value (e.g., "algorithm.max_iterations")
            default: Default value if key not found

        Returns:
            value: Configuration value

        Example:
            loader.get("scvx_params", "algorithm.max_iterations")  # Returns 30
        """
        # Load config if not already loaded
        if config_name not in self.configs:
            self.load(config_name)

        config = self.configs[config_name]

        # Navigate nested dictionary
        keys = key_path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def merge(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configuration files.

        Later configs override earlier ones.

        Args:
            *config_names: Names of configs to merge

        Returns:
            merged: Merged configuration
        """
        merged = {}

        for name in config_names:
            config = self.configs.get(name) or self.load(name)
            merged = self._deep_merge(merged, config)

        return merged

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            update: Dictionary with updates

        Returns:
            merged: Merged dictionary
        """
        merged = deepcopy(base)

        for key, value in update.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)

        return merged

    def validate_scvx_config(self, config: Optional[Dict] = None) -> bool:  # noqa: C901, PLR0912
        """
        Validate SCvx configuration parameters.

        Args:
            config: Config to validate (default: loaded scvx_params)

        Returns:
            valid: True if valid

        Raises:
            ValueError: If validation fails
        """
        if config is None:
            config = self.configs.get("scvx_params") or self.load("scvx_params")

        # Check required fields
        required_fields = ["dt", "N", "initial_state", "goal_state", "state_bounds", "input_bounds", "algorithm"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        # Validate dt
        if config["dt"] <= 0:
            raise ValueError(f"dt must be positive, got {config['dt']}")

        # Validate N
        if config["N"] <= 0:
            raise ValueError(f"N must be positive, got {config['N']}")

        # Validate state dimensions
        if len(config["initial_state"]) != 3:
            raise ValueError(f"initial_state must be 3D, got {len(config['initial_state'])}")

        if len(config["goal_state"]) != 3:
            raise ValueError(f"goal_state must be 3D, got {len(config['goal_state'])}")

        # Validate bounds
        x_min = config["state_bounds"]["x_min"]
        x_max = config["state_bounds"]["x_max"]

        if not all(x_min[i] < x_max[i] for i in range(len(x_min))):
            raise ValueError("state_bounds: x_min must be < x_max")

        u_min = config["input_bounds"]["u_min"]
        u_max = config["input_bounds"]["u_max"]

        if not all(u_min[i] < u_max[i] for i in range(len(u_min))):
            raise ValueError("input_bounds: u_min must be < u_max")

        # Validate algorithm parameters
        algo = config["algorithm"]

        if algo["max_iterations"] <= 0:
            raise ValueError("max_iterations must be positive")

        if algo["tol_x"] <= 0 or algo["tol_u"] <= 0:
            raise ValueError("Tolerances must be positive")

        tr = algo["trust_region"]
        if tr["rho_min"] >= tr["rho_max"]:
            raise ValueError("rho_min must be < rho_max")

        if tr["beta_expand"] <= 1.0:
            raise ValueError("beta_expand must be > 1.0")

        if tr["gamma_contract"] >= 1.0:
            raise ValueError("gamma_contract must be < 1.0")

        return True

    def validate_environment_config(self, config: Optional[Dict] = None) -> bool:  # noqa: C901
        """
        Validate environment configuration.

        Args:
            config: Config to validate (default: loaded environment)

        Returns:
            valid: True if valid

        Raises:
            ValueError: If validation fails
        """
        if config is None:
            config = self.configs.get("environment") or self.load("environment")

        # Check workspace bounds
        ws = config["workspace"]
        if ws["x_min"] >= ws["x_max"]:
            raise ValueError("workspace: x_min must be < x_max")
        if ws["y_min"] >= ws["y_max"]:
            raise ValueError("workspace: y_min must be < y_max")

        # Validate obstacles
        if "obstacles" not in config or len(config["obstacles"]) == 0:
            raise ValueError("At least one obstacle must be defined")

        for i, obs in enumerate(config["obstacles"]):
            if "type" not in obs:
                raise ValueError(f"Obstacle {i}: missing 'type'")

            if obs["type"] == "circle":
                if "center" not in obs or "radius" not in obs:
                    raise ValueError(f"Obstacle {i}: missing 'center' or 'radius'")

                if len(obs["center"]) != 2:
                    raise ValueError(f"Obstacle {i}: center must be 2D")

                if obs["radius"] <= 0:
                    raise ValueError(f"Obstacle {i}: radius must be positive")

                # Check obstacle is within workspace
                center = obs["center"]
                radius = obs["radius"] + obs.get("safety_margin", 0)

                if (
                    center[0] - radius < ws["x_min"]
                    or center[0] + radius > ws["x_max"]
                    or center[1] - radius < ws["y_min"]
                    or center[1] + radius > ws["y_max"]
                ):
                    print(f"Warning: Obstacle {i} may extend outside workspace")

        return True

    def to_numpy(self, config_name: str, key_path: str) -> np.ndarray:
        """
        Get configuration value as numpy array.

        Args:
            config_name: Config file name
            key_path: Dot-separated key path

        Returns:
            array: Numpy array
        """
        value = self.get(config_name, key_path)
        return np.array(value)

    def print_summary(self, config_name: str):
        """
        Print summary of configuration.

        Args:
            config_name: Config file name
        """
        if config_name not in self.configs:
            self.load(config_name)

        config = self.configs[config_name]

        print("=" * 60)
        print(f"Configuration: {config_name}")
        print("=" * 60)

        self._print_dict(config, indent=0)

        print("=" * 60)

    def _print_dict(self, d: Dict, indent: int = 0):
        """Helper to pretty-print nested dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            elif isinstance(value, list):
                if len(value) <= 5:
                    print("  " * indent + f"{key}: {value}")
                else:
                    print("  " * indent + f"{key}: [... {len(value)} items ...]")
            elif isinstance(value, str) and len(value) > 60:
                print("  " * indent + f"{key}: {value[:60]}...")
            else:
                print("  " * indent + f"{key}: {value}")

    def save(self, config_name: str, config: Dict, overwrite: bool = False):
        """
        Save configuration to YAML file.

        Args:
            config_name: Config file name (without .yaml)
            config: Configuration dictionary
            overwrite: Allow overwriting existing file
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if config_path.exists() and not overwrite:
            raise FileExistsError(f"Config file already exists: {config_path}. Use overwrite=True to replace.")

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Saved configuration to: {config_path}")

    def __repr__(self) -> str:
        return f"ConfigLoader(config_dir={self.config_dir}, loaded={list(self.configs.keys())})"


class ExperimentConfig:
    """
    Wrapper for experiment configuration combining multiple config files.
    """

    def __init__(
        self,
        scvx_config: Optional[str] = None,
        environment_config: Optional[str] = None,
        unicycle_config: Optional[str] = None,
        config_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize experiment configuration.

        Args:
            scvx_config: Name of SCvx config file (default: "scvx_params")
            environment_config: Name of environment config (default: "environment")
            unicycle_config: Name of unicycle config (default: "unicycle_params")
            config_dir: Config directory path
        """
        self.loader = ConfigLoader(config_dir)

        # Load configs
        self.scvx_config_name = scvx_config or "scvx_params"
        self.environment_config_name = environment_config or "environment"
        self.unicycle_config_name = unicycle_config or "unicycle_params"

        self.scvx = self.loader.load(self.scvx_config_name)
        self.environment = self.loader.load(self.environment_config_name)
        self.unicycle = self.loader.load(self.unicycle_config_name)

        # Validate
        self.validate()

    def validate(self):
        """Validate all configurations."""
        self.loader.validate_scvx_config(self.scvx)
        self.loader.validate_environment_config(self.environment)
        print("✓ All configurations validated successfully")

    def get_scvx_params(self) -> Dict[str, Any]:
        """Get SCvx planner parameters."""
        return {
            "max_iterations": self.scvx["algorithm"]["max_iterations"],
            "tol_x": self.scvx["algorithm"]["tol_x"],
            "tol_u": self.scvx["algorithm"]["tol_u"],
            "trust_region_rho": self.scvx["algorithm"]["trust_region"]["rho_init"],
            "trust_region_rho_max": self.scvx["algorithm"]["trust_region"]["rho_max"],
            "trust_region_rho_min": self.scvx["algorithm"]["trust_region"]["rho_min"],
            "trust_region_beta": self.scvx["algorithm"]["trust_region"]["beta_expand"],
            "trust_region_gamma": self.scvx["algorithm"]["trust_region"]["gamma_contract"],
            "weight_trust_region": self.scvx["algorithm"]["weights"]["trust_region"],
            "weight_control": self.scvx["algorithm"]["weights"]["control"],
            "weight_terminal": self.scvx["algorithm"]["weights"]["terminal"],
            "verbose": self.scvx["algorithm"]["verbose"],
        }

    def get_initial_state(self) -> np.ndarray:
        """Get initial state."""
        return np.array(self.scvx["initial_state"])

    def get_goal_state(self) -> np.ndarray:
        """Get goal state."""
        return np.array(self.scvx["goal_state"])

    def get_state_bounds(self) -> tuple:
        """Get state bounds (x_min, x_max)."""
        x_min = np.array(self.scvx["state_bounds"]["x_min"])
        x_max = np.array(self.scvx["state_bounds"]["x_max"])
        return x_min, x_max

    def get_input_bounds(self) -> tuple:
        """Get input bounds (u_min, u_max)."""
        u_min = np.array(self.scvx["input_bounds"]["u_min"])
        u_max = np.array(self.scvx["input_bounds"]["u_max"])
        return u_min, u_max

    def get_workspace_bounds(self) -> tuple:
        """Get workspace bounds."""
        ws = self.environment["workspace"]
        lower = np.array([ws["x_min"], ws["y_min"]])
        upper = np.array([ws["x_max"], ws["y_max"]])
        return lower, upper

    def get_obstacles(self) -> List[Dict]:
        """Get obstacle definitions."""
        return self.environment["obstacles"]

    def get_dt(self) -> float:
        """Get time step."""
        return self.scvx["dt"]

    def get_horizon(self) -> int:
        """Get planning horizon."""
        return self.scvx["N"]

    def print_summary(self):
        """Print experiment configuration summary."""
        print("\n" + "=" * 70)
        print("EXPERIMENT CONFIGURATION SUMMARY")
        print("=" * 70)

        print("\n📍 Planning Problem:")
        print(f"  Initial state: {self.get_initial_state()}")
        print(f"  Goal state:    {self.get_goal_state()}")
        print(f"  Horizon:       {self.get_horizon()} steps ({self.get_dt() * self.get_horizon():.1f}s)")
        print(f"  Time step:     {self.get_dt()}s")

        print("\n🌍 Environment:")
        ws_lower, ws_upper = self.get_workspace_bounds()
        print(f"  Workspace:     [{ws_lower[0]}, {ws_upper[0]}] x [{ws_lower[1]}, {ws_upper[1]}]")
        print(f"  Obstacles:     {len(self.get_obstacles())} obstacles")
        for i, obs in enumerate(self.get_obstacles()):
            if obs["type"] == "circle":
                print(f"    {i + 1}. Circle at {obs['center']}, r={obs['radius']}+{obs.get('safety_margin', 0)}")

        print("\n SCvx Algorithm:")
        print(f"  Max iterations: {self.scvx['algorithm']['max_iterations']}")
        print(f"  Tolerances:     tol_x={self.scvx['algorithm']['tol_x']}, tol_u={self.scvx['algorithm']['tol_u']}")
        print(
            f"  Trust region:   ρ∈[{self.scvx['algorithm']['trust_region']['rho_min']}, "  # noqa: RUF001
            f"{self.scvx['algorithm']['trust_region']['rho_max']}], ρ_init={self.scvx['algorithm']['trust_region']['rho_init']}"  # noqa: RUF001, E501
        )

        print("\n" + "=" * 70 + "\n")

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(scvx={self.scvx_config_name}, "
            f"env={self.environment_config_name}, "
            f"model={self.unicycle_config_name})"
        )
