"""Minimal tests for ddfs.uncertainty module."""

import numpy as np
import pytest

from ddfs.uncertainty import UncertaintyBounds, UncertaintyConstants


def test_uncertainty_constants_creation():
    """Test UncertaintyConstants creation."""
    constants = UncertaintyConstants(
        gamma=0.01,
        L_r=0.5,
        L_J=1.0,
        v=0.1,
        C=0.1,
        beta_i=[0.02, 0.02, 0.02],
        segment_indices=[0, 25, 50],
    )

    assert constants.gamma == 0.01
    assert constants.L_r == 0.5
    assert constants.L_J == 1.0
    assert len(constants.beta_i) == 3
    assert len(constants.segment_indices) == 3


def test_uncertainty_constants_validation():
    """Test that invalid constants raise errors."""
    # Mismatched lengths
    with pytest.raises(ValueError):
        UncertaintyConstants(
            gamma=0.01,
            L_r=0.5,
            L_J=1.0,
            v=0.1,
            C=0.1,
            beta_i=[0.02, 0.02],  # Length 2
            segment_indices=[0, 25, 50],  # Length 3
        )

    # Negative values
    with pytest.raises(ValueError):
        UncertaintyConstants(
            gamma=-0.01,  # Should be non-negative
            L_r=0.5,
            L_J=1.0,
            v=0.1,
            C=0.1,
            beta_i=[0.02],
            segment_indices=[0],
        )


def test_uncertainty_constants_properties():
    """Test UncertaintyConstants properties."""
    constants = UncertaintyConstants(
        gamma=0.01,
        L_r=0.5,
        L_J=1.0,
        v=0.1,
        C=0.1,
        beta_i=[0.02, 0.03, 0.04],
        segment_indices=[0, 25, 50],
    )

    assert constants.num_segments == 3

    # Check beta statistics
    assert constants.beta_max == pytest.approx(0.04)
    assert constants.beta_min == pytest.approx(0.02)
    assert constants.beta_mean == pytest.approx(0.03)


def test_uncertainty_constants_save_load(tmp_path):
    """Test saving and loading constants."""
    constants = UncertaintyConstants(
        gamma=0.01,
        L_r=0.5,
        L_J=1.0,
        v=0.1,
        C=0.1,
        beta_i=[0.02, 0.02, 0.02],
        segment_indices=[0, 25, 50],
    )

    # Save
    path = tmp_path / "constants.pkl"
    constants.save(path)

    # Load
    loaded = UncertaintyConstants.load(path)

    assert loaded.gamma == constants.gamma
    assert loaded.L_r == constants.L_r
    assert loaded.L_J == constants.L_J
    assert np.allclose(loaded.beta_i, constants.beta_i)
    assert loaded.segment_indices == constants.segment_indices


def test_uncertainty_bounds():
    """Test legacy UncertaintyBounds container."""
    bounds = UncertaintyBounds(
        gamma=0.01,
        L_r=0.5,
        L_J=1.0,
        beta_i=np.array([0.02, 0.03, 0.04]),
        n_samples=100,
    )

    assert bounds.gamma == 0.01
    assert bounds.L_r == 0.5
    assert bounds.L_J == 1.0
    assert len(bounds.beta_i) == 3
    assert bounds.n_samples == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
