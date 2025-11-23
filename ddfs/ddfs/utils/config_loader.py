"""Configuration loading utilities.

This module provides a simple wrapper around DDFSConfig for convenience.
The main implementation is in ddfs.core.config.
"""

from pathlib import Path
from typing import Union

from ddfs.core.config import DDFSConfig


def load_config(config_path: Union[str, Path]) -> DDFSConfig:
    """Load configuration from a YAML file.

    This is a convenience wrapper around DDFSConfig for easier imports.

    Args:
        config_path: Path to the configuration file (YAML format).

    Returns:
        DDFSConfig object containing parsed configuration.

    Examples:
        >>> from ddfs.utils import load_config
        >>> config = load_config('config/ddfs_config.yaml')
        >>> print(config.system_type)
        'unicycle'

        >>> # Or use the core module directly
        >>> from ddfs.core.config import DDFSConfig
        >>> config = DDFSConfig('config/ddfs_config.yaml')

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    return DDFSConfig(config_path)


def load_default_config() -> DDFSConfig:
    """Load the default configuration file.

    Looks for config/ddfs_config.yaml in the project root.

    Returns:
        DDFSConfig object with default configuration.

    Examples:
        >>> from ddfs.utils import load_default_config
        >>> config = load_default_config()
    """
    config_path = Path("config/ddfs_config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Default config not found at {config_path}. "
            "Please create config/ddfs_config.yaml or specify path explicitly."
        )

    return DDFSConfig(config_path)
