# ddfs/ddfs/models/__init__.py

"""
Dynamics models package.

This package provides dynamics models for the DDFS pipeline:
- Base classes for all dynamics models
- Twin models (digital twins for planning)
- Plant models (real systems with mismatch)
- System-specific implementations (unicycle, quadrotor)

Available Models
----------------
UnicycleTwin : Kinematic unicycle model (digital twin)
    State: [x, y, θ] (3D)
    Input: [v, ω] (2D)

UnicyclePlant : Unicycle with model mismatch
    Mismatch: velocity scaling, angular scaling, slip

QuadrotorTwin : Full 3D quadrotor model (digital twin)
    State: [p, v, q, ω] (13D)
    Input: [T, τ] (4D)

QuadrotorPlant : Quadrotor with model mismatch
    Mismatch: mass, inertia, drag, thrust efficiency

Usage Examples
--------------
Creating a unicycle twin:
    >>> from ddfs.models import UnicycleTwin
    >>> twin = UnicycleTwin(dt=0.1)
    >>> print(twin.state_dim, twin.input_dim)
    3 2

Creating a unicycle plant with mismatch:
    >>> from ddfs.models import UnicycleTwin, UnicyclePlant
    >>> twin = UnicycleTwin(dt=0.1)
    >>> plant = UnicyclePlant(twin, velocity_scale=0.95, slip_coefficient=0.02)

Using the factory function:
    >>> from ddfs.models import UnicycleTwin, create_plant_from_config
    >>> twin = UnicycleTwin(dt=0.1)
    >>> config = {'velocity_scale': 0.95, 'angular_scale': 1.03}
    >>> plant = create_plant_from_config(twin, config)

Creating example configurations:
    >>> from ddfs.models import create_unicycle_example, create_quadrotor_example
    >>> unicycle_config = create_unicycle_example()
    >>> quadrotor_config = create_quadrotor_example()

Notes
-----
- Constraints are NOT in this package (see ddfs.core.constraints)
- Obstacles are NOT in this package (see ddfs.core.obstacles)
- This package focuses solely on system dynamics
"""

# Base classes
from ddfs.models.base import (
    DynamicsModel,
    PlantModel,
    TwinModel,
    validate_state_input_dims,
)

# Plant models (with mismatch)
from ddfs.models.plant import (
    QuadrotorPlant,
    UnicyclePlant,
    create_plant_from_config,
)

# Quadrotor models
from ddfs.models.quadrotor import QuadrotorTwin, create_quadrotor_example

# Unicycle models
from ddfs.models.unicycle import UnicycleTwin, create_unicycle_example

__all__ = [
    # Base classes
    "DynamicsModel",
    "PlantModel",
    "QuadrotorPlant",
    # Quadrotor
    "QuadrotorTwin",
    "TwinModel",
    "UnicyclePlant",
    # Unicycle
    "UnicycleTwin",
    # Factory
    "create_plant_from_config",
    "create_quadrotor_example",
    "create_unicycle_example",
    "validate_state_input_dims",
]

__version__ = "0.1.0"
