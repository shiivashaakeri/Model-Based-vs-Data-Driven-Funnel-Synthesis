# ddfs/ddfs/core/config.py

"""
Configuration management for DDFS.

This module provides a smart configuration loader that:
    - Loads configuration from YAML files
    - Validates configuration structure
    - Creates system objects (constraints, workspace, obstacles)
    - Provides system-specific settings
    - Acts as single source of truth for all parameters

Key Classes
-----------
DDFSConfig : Main configuration manager
    Loads YAML, validates, and provides access to all settings

Usage
-----
>>> from ddfs.core.config import DDFSConfig
>>>
>>> config = DDFSConfig('config/ddfs_config.yaml')
>>> print(config.system_type)  # 'unicycle' or 'quadrotor'
>>>
>>> # Get system-specific objects
>>> constraints = config.get_constraints()
>>> workspace = config.get_workspace()
>>> obstacles = config.get_obstacles()
>>>
>>> # Get planning parameters
>>> planning_params = config.get_planning_params()
>>> print(planning_params['N'], planning_params['dt'])
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ddfs.core.constraints import QuadrotorConstraints, SystemConstraints, UnicycleConstraints
from ddfs.core.obstacles import Obstacle, create_obstacles_from_config
from ddfs.core.workspace import Workspace, Workspace2D, Workspace3D


class DDFSConfig:
    """
    Unified configuration manager for DDFS.

    This class loads configuration from YAML and provides convenient
    access to all system parameters and objects.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file

    Attributes
    ----------
    raw : dict
        Raw configuration dictionary from YAML
    system_type : str
        Active system type ('unicycle' or 'quadrotor')

    Examples
    --------
    >>> from ddfs.core.config import DDFSConfig
    >>>
    >>> config = DDFSConfig('config/ddfs_config.yaml')
    >>>
    >>> # Get system type
    >>> print(config.system_type)
    'unicycle'
    >>>
    >>> # Get system-specific config
    >>> system_config = config.get_system_config()
    >>> print(system_config['dt'])
    0.131
    >>>
    >>> # Create objects
    >>> constraints = config.get_constraints()
    >>> workspace = config.get_workspace()
    >>> obstacles = config.get_obstacles()
    """

    def __init__(self, config_path: str | Path):
        """
        Initialize configuration manager.

        Parameters
        ----------
        config_path : str or Path
            Path to YAML configuration file

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist
        ValueError
            If configuration is invalid
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Load YAML
        with open(self.config_path, "r") as f:
            self.raw = yaml.safe_load(f)

        # Validate basic structure
        self._validate_config()

        # Extract system type
        self.system_type = self.raw["system"]["active"]

        # Cache for created objects (lazy loading)
        self._constraints_cache: Optional[SystemConstraints] = None
        self._workspace_cache: Optional[Workspace] = None
        self._obstacles_cache: Optional[List[Obstacle]] = None

    def _validate_config(self):
        """
        Validate configuration structure.

        Raises
        ------
        ValueError
            If required fields are missing or invalid
        """
        # Check required top-level keys
        required_keys = ["experiment", "system", "planning", "environment"]
        for key in required_keys:
            if key not in self.raw:
                raise ValueError(f"Missing required config section: '{key}'")

        # Check system config
        if "active" not in self.raw["system"]:
            raise ValueError("Missing 'system.active' field")

        active_system = self.raw["system"]["active"]
        if active_system not in ["unicycle", "quadrotor"]:
            raise ValueError(f"Invalid system type: '{active_system}'. Must be 'unicycle' or 'quadrotor'")

        if active_system not in self.raw["system"]:
            raise ValueError(f"Missing system config for active system: '{active_system}'")

        # Check planning config
        if active_system not in self.raw["planning"]:
            raise ValueError(f"Missing planning config for system: '{active_system}'")

        # Check environment config
        if active_system not in self.raw["environment"]:
            raise ValueError(f"Missing environment config for system: '{active_system}'")

    def get_system_config(self) -> Dict[str, Any]:
        """
        Get configuration for active system.

        Returns
        -------
        system_config : dict
            System-specific configuration (state_dim, input_dim, dt, etc.)

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> system_config = config.get_system_config()
        >>> print(system_config['state_dim'], system_config['input_dim'])
        3 2
        """
        return self.raw["system"][self.system_type]

    def get_planning_params(self) -> Dict[str, Any]:
        """
        Get planning parameters for active system.

        Returns
        -------
        planning_params : dict
            Planning parameters (tf, N, x0, xf, etc.)

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> params = config.get_planning_params()
        >>> print(params['N'], params['tf'])
        61 8.0
        """
        return self.raw["planning"][self.system_type]

    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment configuration for active system.

        Returns
        -------
        env_config : dict
            Environment configuration (workspace, obstacles)

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> env = config.get_environment_config()
        >>> print(env['workspace'])
        """
        return self.raw["environment"][self.system_type]

    def get_constraints(self) -> SystemConstraints:
        """
        Get constraint object for active system.

        Creates and caches constraint object on first call.

        Returns
        -------
        constraints : SystemConstraints
            UnicycleConstraints or QuadrotorConstraints

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> constraints = config.get_constraints()
        >>> print(type(constraints).__name__)
        'UnicycleConstraints'
        """
        if self._constraints_cache is not None:
            return self._constraints_cache

        # Get system config
        system_config = self.get_system_config()

        # Create appropriate constraint object
        if self.system_type == "unicycle":
            constraints = UnicycleConstraints.from_config(system_config)
        elif self.system_type == "quadrotor":
            constraints = QuadrotorConstraints.from_config(system_config)
        else:
            raise ValueError(f"Unknown system type: {self.system_type}")

        # Cache and return
        self._constraints_cache = constraints
        return constraints

    def get_workspace(self) -> Workspace:
        """
        Get workspace object for active system.

        Creates and caches workspace object on first call.

        Returns
        -------
        workspace : Workspace
            Workspace2D or Workspace3D

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> workspace = config.get_workspace()
        >>> print(workspace.bounds)
        (0.0, 12.0, 0.0, 8.0)
        """
        if self._workspace_cache is not None:
            return self._workspace_cache

        # Get environment config
        env_config = self.get_environment_config()
        workspace_config = env_config["workspace"]

        # Create appropriate workspace object
        if self.system_type == "unicycle":
            workspace = Workspace2D.from_config(workspace_config)
        elif self.system_type == "quadrotor":
            workspace = Workspace3D.from_config(workspace_config)
        else:
            raise ValueError(f"Unknown system type: {self.system_type}")

        # Cache and return
        self._workspace_cache = workspace
        return workspace

    def get_obstacles(self) -> List[Obstacle]:
        """
        Get obstacle objects for active system.

        Creates and caches obstacle objects on first call.

        Returns
        -------
        obstacles : list of Obstacle
            List of CircleObstacle or SphereObstacle

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> obstacles = config.get_obstacles()
        >>> print(len(obstacles))
        2
        >>> print(obstacles[0].center)
        [4. 3.]
        """
        if self._obstacles_cache is not None:
            return self._obstacles_cache

        # Get environment config
        env_config = self.get_environment_config()
        obstacles_config = env_config.get("obstacles", [])

        # Create obstacles using factory function
        obstacles = create_obstacles_from_config(obstacles_config, self.system_type)

        # Cache and return
        self._obstacles_cache = obstacles
        return obstacles

    def get_plant_mismatch_params(self) -> Dict[str, Any]:
        """
        Get plant mismatch parameters for active system.

        Returns
        -------
        mismatch_params : dict
            Plant mismatch parameters (velocity_scale, mass_scale, etc.)

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> mismatch = config.get_plant_mismatch_params()
        >>> print(mismatch['velocity_scale'])
        0.95
        """
        system_config = self.get_system_config()
        return system_config.get("plant_mismatch", {})

    def get_data_collection_params(self) -> Dict[str, Any]:
        """
        Get data collection parameters (Phase 2).

        Returns
        -------
        data_params : dict
            Data collection parameters (M, excitation, segmentation, etc.)

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> data_params = config.get_data_collection_params()
        >>> print(data_params.get('M', 50))
        50
        """
        return self.raw.get("data_collection", {})

    def get_uncertainty_params(self) -> Dict[str, Any]:
        """
        Get uncertainty quantification parameters (Phase 3).

        Returns
        -------
        uncertainty_params : dict
            Uncertainty quantification parameters

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> uncertainty_params = config.get_uncertainty_params()
        """
        return self.raw.get("uncertainty", {})

    def get_synthesis_params(self) -> Dict[str, Any]:
        """
        Get funnel synthesis parameters (Phase 4).

        Returns
        -------
        synthesis_params : dict
            Funnel synthesis parameters

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> synthesis_params = config.get_synthesis_params()
        """
        return self.raw.get("synthesis", {})

    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get experiment metadata.

        Returns
        -------
        experiment_info : dict
            Experiment name, description, output_dir, seed

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> info = config.get_experiment_info()
        >>> print(info['name'])
        'ddfs_offline'
        """
        return self.raw["experiment"]

    def get_output_dir(self) -> Path:
        """
        Get output directory for results.

        Returns
        -------
        output_dir : Path
            Path to output directory

        Examples
        --------
        >>> config = DDFSConfig('config/ddfs_config.yaml')
        >>> output_dir = config.get_output_dir()
        >>> print(output_dir)
        results/unicycle
        """
        base_dir = self.raw["experiment"].get("output_dir", "results")
        return Path(base_dir) / self.system_type

    def summary(self) -> str:
        """
        Generate human-readable summary of configuration.

        Returns
        -------
        summary : str
            Configuration summary
        """
        lines = [
            "=" * 70,
            "DDFS CONFIGURATION",
            "=" * 70,
            "",
            f"Experiment: {self.raw['experiment']['name']}",
            f"Description: {self.raw['experiment']['description']}",
            f"Output Directory: {self.get_output_dir()}",
            "",
            f"Active System: {self.system_type.upper()}",
            "",
            "System Configuration:",
        ]

        # System config
        system_config = self.get_system_config()
        lines.append(f"  State dimension: {system_config['state_dim']}")
        lines.append(f"  Input dimension: {system_config['input_dim']}")
        lines.append(f"  Timestep: {system_config['dt']:.6f} s")

        # Planning config
        lines.append("")
        lines.append("Planning Configuration:")
        planning_params = self.get_planning_params()
        lines.append(f"  Horizon: N = {planning_params['N']}")
        lines.append(f"  Final time: tf = {planning_params['tf']} s")
        lines.append(f"  Initial state: {planning_params['x0']}")
        lines.append(f"  Goal state: {planning_params['xf']}")

        # Environment
        lines.append("")
        lines.append("Environment:")
        workspace = self.get_workspace()
        lines.append(f"  Workspace: {workspace}")
        obstacles = self.get_obstacles()
        lines.append(f"  Obstacles: {len(obstacles)}")
        for obs in obstacles:
            lines.append(f"    - {obs}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"DDFSConfig(system='{self.system_type}', config_path='{self.config_path}')"


def load_config(config_path: str | Path) -> DDFSConfig:
    """
    Convenience function to load configuration.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file

    Returns
    -------
    config : DDFSConfig
        Configuration object

    Examples
    --------
    >>> from ddfs.core.config import load_config
    >>> config = load_config('config/ddfs_config.yaml')
    """
    return DDFSConfig(config_path)
