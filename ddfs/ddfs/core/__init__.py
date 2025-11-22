# ddfs/ddfs/core/__init__.py

"""
Core definitions package for DDFS.

This package provides the central definitions and configuration management
for the entire DDFS pipeline. All other modules depend on these core classes.

Key Components
--------------
Configuration:
    DDFSConfig : Smart configuration manager
    load_config : Convenience function to load config

Constraints:
    SystemConstraints : Abstract base class
    UnicycleConstraints : 2D state/input constraints
    QuadrotorConstraints : 3D state/input constraints

Obstacles:
    Obstacle : Abstract base class
    CircleObstacle : 2D circular obstacles
    SphereObstacle : 3D spherical obstacles
    create_obstacles_from_config : Factory function
    check_collision_free : Check if point avoids all obstacles
    minimum_distance_to_obstacles : Closest obstacle distance

Workspace:
    Workspace : Abstract base class
    Workspace2D : Rectangular 2D workspace
    Workspace3D : Rectangular cuboid 3D workspace

Usage Examples
--------------
Loading configuration:
    >>> from ddfs.core import DDFSConfig
    >>> config = DDFSConfig('config/ddfs_config.yaml')
    >>> print(config.system_type)
    'unicycle'

Creating system objects:
    >>> constraints = config.get_constraints()
    >>> workspace = config.get_workspace()
    >>> obstacles = config.get_obstacles()

Direct object creation:
    >>> from ddfs.core import UnicycleConstraints, Workspace2D, CircleObstacle
    >>>
    >>> constraints = UnicycleConstraints(
    ...     x_min=0.0, x_max=12.0,
    ...     y_min=0.0, y_max=8.0,
    ...     v_min=0.0, v_max=2.0,
    ...     omega_max=2.0
    ... )
    >>>
    >>> workspace = Workspace2D(
    ...     x_min=0.0, x_max=12.0,
    ...     y_min=0.0, y_max=8.0
    ... )
    >>>
    >>> obs = CircleObstacle(
    ...     obstacle_id="obs_1",
    ...     center=[4.0, 3.0],
    ...     radius=1.0,
    ...     safety_margin=0.25
    ... )

Philosophy
----------
The core package is the "single source of truth" for:
    - System parameters (from config)
    - Constraints (state/input bounds)
    - Obstacles (collision avoidance)
    - Workspace (environment bounds)

All other modules (models, planning, synthesis, etc.) import from core
to ensure consistency across the entire pipeline.
"""

# Configuration
from ddfs.core.config import DDFSConfig, load_config

# Constraints
from ddfs.core.constraints import (
    QuadrotorConstraints,
    SystemConstraints,
    UnicycleConstraints,
)

# Obstacles
from ddfs.core.obstacles import (
    CircleObstacle,
    Obstacle,
    SphereObstacle,
    check_collision_free,
    create_obstacles_from_config,
    minimum_distance_to_obstacles,
)

# Workspace
from ddfs.core.workspace import Workspace, Workspace2D, Workspace3D

__all__ = [
    "CircleObstacle",
    # Configuration
    "DDFSConfig",
    # Obstacles
    "Obstacle",
    "QuadrotorConstraints",
    "SphereObstacle",
    # Constraints
    "SystemConstraints",
    "UnicycleConstraints",
    # Workspace
    "Workspace",
    "Workspace2D",
    "Workspace3D",
    "check_collision_free",
    "create_obstacles_from_config",
    "load_config",
    "minimum_distance_to_obstacles",
]

__version__ = "0.1.0"
