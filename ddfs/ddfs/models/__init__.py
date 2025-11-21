# ddfs/ddfs/models/__init__.py

"""
Dynamics models package.

This package provides dynamics models for the DDFS pipeline:
- Base classes for all dynamics models
- Twin models (digital twins for planning)
- Plant models (real systems with mismatch)
- System-specific implementations (unicycle, quadrotor)

Available models:
    - UnicycleTwin: Kinematic unicycle (3D state, 2D input)
    - UnicyclePlant: Unicycle with model mismatch
    - QuadrotorTwin: Full 3D quadrotor (13D state, 4D input)
    - QuadrotorPlant: Quadrotor with model mismatch

Available constraint classes:
    - UnicycleConstraints: State and input constraints for unicycle
    - QuadrotorConstraints: State and input constraints for quadrotor

Usage:
    >>> from ddfs.models import UnicycleTwin, UnicyclePlant
    >>> twin = UnicycleTwin(dt=0.1)
    >>> plant = UnicyclePlant(twin, velocity_scale=0.95)
    >>> x = jnp.array([0.0, 0.0, 0.0])
    >>> u = jnp.array([1.0, 0.5])
    >>> x_next = plant.step(x, u)
"""

from .base import DynamicsModel, PlantModel, TwinModel, validate_state_input_dims
from .plant import QuadrotorPlant, UnicyclePlant, create_plant_from_config
from .quadrotor import QuadrotorConstraints, QuadrotorTwin, create_quadrotor_example
from .unicycle import UnicycleConstraints, UnicycleTwin, create_unicycle_example

__all__ = [
    # Base classes
    "DynamicsModel",
    "PlantModel",
    "QuadrotorConstraints",
    "QuadrotorPlant",
    # Quadrotor
    "QuadrotorTwin",
    "TwinModel",
    "UnicycleConstraints",
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
