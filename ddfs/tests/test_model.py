"""Minimal tests for ddfs.models module."""

import jax.numpy as jnp
import numpy as np
import pytest

from ddfs.models import (
    QuadrotorPlant,
    QuadrotorTwin,
    UnicyclePlant,
    UnicycleTwin,
    create_plant_from_config,
)


def test_unicycle_twin():
    """Test UnicycleTwin creation and basic operations."""
    twin = UnicycleTwin(dt=0.1)

    assert twin.state_dim == 3
    assert twin.input_dim == 2

    # Test step
    x = jnp.array([0.0, 0.0, 0.0])
    u = jnp.array([1.0, 0.5])
    x_next = twin.step(x, u)

    assert x_next.shape == (3,)
    assert not jnp.allclose(x_next, x)  # Should have moved


def test_unicycle_plant():
    """Test UnicyclePlant with mismatch."""
    twin = UnicycleTwin(dt=0.1)
    plant = UnicyclePlant(twin, velocity_scale=0.95, slip_coefficient=0.02)

    assert plant.state_dim == 3
    assert plant.input_dim == 2

    # Test mismatch computation
    x = jnp.array([1.0, 1.0, 0.0])
    u = jnp.array([1.0, 0.5])

    mismatch = plant.compute_mismatch(x, u)
    assert mismatch >= 0  # Mismatch is non-negative


def test_quadrotor_twin():
    """Test QuadrotorTwin creation and basic operations."""
    twin = QuadrotorTwin(mass=0.0293, dt=0.078)

    assert twin.state_dim == 13
    assert twin.input_dim == 4

    # Test hover equilibrium
    x = jnp.zeros(13)
    x = x.at[6].set(1.0)  # Identity quaternion
    u = jnp.array([0.0293 * 9.81, 0.0, 0.0, 0.0])  # Hover thrust

    x_next = twin.step(x, u)
    assert x_next.shape == (13,)


def test_quadrotor_plant():
    """Test QuadrotorPlant with mismatch."""
    twin = QuadrotorTwin(mass=0.0293, dt=0.078)
    plant = QuadrotorPlant(twin, mass_scale=0.98, drag_coefficient=0.01)

    assert plant.state_dim == 13
    assert plant.input_dim == 4
    assert plant.m_actual == pytest.approx(0.0293 * 0.98)


def test_jacobians():
    """Test Jacobian computation."""
    twin = UnicycleTwin(dt=0.1)

    x = jnp.array([1.0, 1.0, 0.5])
    u = jnp.array([1.0, 0.5])

    A, B = twin.jacobians(x, u)

    assert A.shape == (3, 3)
    assert B.shape == (3, 2)


def test_create_plant_from_config():
    """Test factory function for plant creation."""
    # Unicycle
    twin_uni = UnicycleTwin(dt=0.1)
    config_uni = {"velocity_scale": 0.95, "angular_scale": 1.03}
    plant_uni = create_plant_from_config(twin_uni, config_uni)

    assert isinstance(plant_uni, UnicyclePlant)
    assert plant_uni.velocity_scale == 0.95

    # Quadrotor
    twin_quad = QuadrotorTwin(dt=0.078)
    config_quad = {"mass_scale": 0.98, "thrust_efficiency": 0.95}
    plant_quad = create_plant_from_config(twin_quad, config_quad)

    assert isinstance(plant_quad, QuadrotorPlant)
    assert plant_quad.mass_scale == 0.98


def test_state_normalization():
    """Test state normalization."""
    # Unicycle angle wrapping
    twin_uni = UnicycleTwin(dt=0.1)
    x_uni = jnp.array([1.0, 2.0, 4.0])  # θ > π
    x_norm = twin_uni.normalize_state(x_uni)
    assert -np.pi <= x_norm[2] <= np.pi

    # Quadrotor quaternion normalization
    twin_quad = QuadrotorTwin(dt=0.078)
    x_quad = jnp.zeros(13)
    x_quad = x_quad.at[6:10].set(jnp.array([0.7, 0.7, 0.0, 0.0]))
    x_norm = twin_quad.normalize_state(x_quad)
    q_norm = x_norm[6:10]
    assert jnp.allclose(jnp.linalg.norm(q_norm), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
