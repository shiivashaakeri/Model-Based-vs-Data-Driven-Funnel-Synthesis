"""Minimal tests for ddfs.feasibility module."""

import numpy as np
import pytest

from ddfs.feasibility import EllipsoidParams, FeasibilityEnvelope


def test_ellipsoid_params_creation():
    """Test EllipsoidParams creation and validation."""
    P = np.eye(3)
    c = np.array([1.0, 2.0, 0.5])

    ellipsoid = EllipsoidParams(P=P, c=c, segment_index=0)

    assert ellipsoid.P.shape == (3, 3)
    assert len(ellipsoid.c) == 3
    assert ellipsoid.segment_index == 0


def test_ellipsoid_volume():
    """Test ellipsoid volume computation."""
    P = 2.0 * np.eye(3)  # Scaled identity
    c = np.zeros(3)

    ellipsoid = EllipsoidParams(P=P, c=c, segment_index=0)
    volume = ellipsoid.volume()

    assert volume > 0


def test_ellipsoid_contains():
    """Test point containment check."""
    P = np.eye(3)
    c = np.zeros(3)

    ellipsoid = EllipsoidParams(P=P, c=c, segment_index=0)

    # Origin should be inside
    assert ellipsoid.contains(c)

    # Far point should be outside
    far_point = np.array([10.0, 10.0, 10.0])
    assert not ellipsoid.contains(far_point)


def test_ellipsoid_invalid():
    """Test that invalid ellipsoids raise errors."""
    # Non-square P
    with pytest.raises(ValueError):
        EllipsoidParams(P=np.ones((3, 2)), c=np.zeros(3), segment_index=0)

    # Dimension mismatch
    with pytest.raises(ValueError):
        EllipsoidParams(P=np.eye(3), c=np.zeros(2), segment_index=0)

    # Non-positive definite
    with pytest.raises(ValueError):
        P_bad = np.array([[1, 2], [2, 1]])  # Not PD
        EllipsoidParams(P=P_bad, c=np.zeros(2), segment_index=0)


def test_feasibility_envelope():
    """Test FeasibilityEnvelope container."""
    P_0 = EllipsoidParams(P=np.eye(3), c=np.zeros(3), segment_index=0)
    P_min_0 = EllipsoidParams(P=1.5 * np.eye(3), c=np.zeros(3), segment_index=0)
    P_min_0_init = EllipsoidParams(P=2 * np.eye(3), c=np.zeros(3), segment_index=0)

    envelope = FeasibilityEnvelope(
        P_0=P_0,
        P_min_0=P_min_0,
        P_min_0_init=P_min_0_init,
        segment_indices=[0, 1, 2],
        bootstrap_consistent=True,
    )

    assert len(envelope.segment_indices) == 3
    assert envelope.bootstrap_consistent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
