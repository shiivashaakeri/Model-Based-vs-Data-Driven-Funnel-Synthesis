"""Utilities package for DDFS.

This package provides utility functions for configuration loading,
logging, file I/O, and other helper functions.

Key Components:
    - load_config: Load configuration from YAML file
    - load_default_config: Load default config/ddfs_config.yaml

Usage:
    >>> from ddfs.utils import load_config
    >>> config = load_config('config/ddfs_config.yaml')

    >>> # Or load default config
    >>> from ddfs.utils import load_default_config
    >>> config = load_default_config()
"""

from ddfs.utils.config_loader import load_config, load_default_config

__all__ = [
    "load_config",
    "load_default_config",
]

__version__ = "0.1.0"
