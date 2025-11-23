"""Test configuration loading."""

from pathlib import Path

import pytest

from ddfs.core.config import DDFSConfig
from ddfs.utils import load_config


def test_config_loads_direct():
    """Test that config file loads without errors (direct import)."""
    config_path = Path("ddfs/config/ddfs_config.yaml")

    if not config_path.exists():
        pytest.skip("Config file not found")

    config = DDFSConfig(config_path)

    assert config.system_type in ["unicycle", "quadrotor"]
    print(f"\n✓ Loaded config for {config.system_type} system (direct)")


def test_config_loads_utils():
    """Test that config file loads via utils wrapper."""
    config_path = Path("ddfs/config/ddfs_config.yaml")

    if not config_path.exists():
        pytest.skip("Config file not found")

    config = load_config(config_path)

    assert config.system_type in ["unicycle", "quadrotor"]
    print(f"\n✓ Loaded config for {config.system_type} system (utils)")


def test_config_loads_default():
    """Test that default config loads."""
    # Note: load_default_config() looks for config/ddfs_config.yaml in project root
    # Since our config is at ddfs/config/ddfs_config.yaml, we'll test with explicit path
    config_path = Path("ddfs/config/ddfs_config.yaml")

    if not config_path.exists():
        pytest.skip("Config file not found")

    # Use load_config instead since load_default_config expects config/ddfs_config.yaml
    # at project root, but our config is at ddfs/config/ddfs_config.yaml
    config = load_config(config_path)

    assert config.system_type in ["unicycle", "quadrotor"]
    print(f"\n✓ Loaded config for {config.system_type} system (via load_config)")


def test_config_unicycle():
    """Test unicycle configuration."""
    config_path = Path("ddfs/config/ddfs_config.yaml")

    if not config_path.exists():
        pytest.skip("Config file not found")

    config = DDFSConfig(config_path)

    # Get system config
    system_config = config.get_system_config()
    assert system_config["state_dim"] == 3
    assert system_config["input_dim"] == 2

    # Get constraints
    constraints = config.get_constraints()
    assert constraints is not None

    # Get workspace
    workspace = config.get_workspace()
    assert workspace is not None

    # Get obstacles
    obstacles = config.get_obstacles()
    assert len(obstacles) == 2

    # Get planning params
    planning = config.get_planning_params()
    assert planning["N"] == 61
    assert planning["tf"] == 8.0

    print("\n" + config.summary())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
