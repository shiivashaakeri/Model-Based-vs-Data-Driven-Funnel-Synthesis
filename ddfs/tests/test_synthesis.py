"""Minimal tests for ddfs.synthesis module."""

import numpy as np
import pytest

from ddfs.synthesis import (
    FunnelLibrary,
    FunnelSegment,
    LMIBuilder,
    SDPSolution,
)


def test_funnel_segment_creation():
    """Test FunnelSegment creation."""
    n, m = 3, 2

    P = np.eye(n)
    K = np.random.randn(m, n)
    c = np.zeros(n)

    segment = FunnelSegment(
        segment_idx=0,
        P=P,
        K=K,
        c=c,
        volume=1.0,
        nu=0.01,
        lambda1=0.1,
        lambda2=0.1,
    )

    assert segment.segment_idx == 0
    assert segment.P.shape == (n, n)
    assert segment.K.shape == (m, n)


def test_funnel_segment_contains():
    """Test containment check."""
    n, m = 3, 2

    P = np.eye(n)
    K = np.zeros((m, n))
    c = np.zeros(n)

    segment = FunnelSegment(
        segment_idx=0,
        P=P,
        K=K,
        c=c,
        volume=1.0,
        nu=0.01,
        lambda1=0.1,
        lambda2=0.1,
    )

    # Center should be inside
    assert segment.contains(c)

    # Far point should be outside
    far_point = np.array([10.0, 10.0, 10.0])
    assert not segment.contains(far_point)


def test_funnel_library():
    """Test FunnelLibrary container."""
    n, m = 3, 2

    segments = []
    for i in range(3):
        seg = FunnelSegment(
            segment_idx=i,
            P=np.eye(n),
            K=np.random.randn(m, n),
            c=np.zeros(n),
            volume=1.0,
            nu=0.01,
            lambda1=0.1,
            lambda2=0.1,
        )
        segments.append(seg)

    library = FunnelLibrary(
        segments=segments,
        segment_indices=[0, 1, 2],
        n=n,
        m=m,
        alpha=0.95,
        mu=1.1,
    )

    assert library.num_segments == 3
    assert library.n == n
    assert library.m == m


def test_funnel_library_get_gain():
    """Test getting feedback gain from library."""
    n, m = 3, 2

    K_test = np.random.randn(m, n)

    segment = FunnelSegment(
        segment_idx=0,
        P=np.eye(n),
        K=K_test,
        c=np.zeros(n),
        volume=1.0,
        nu=0.01,
        lambda1=0.1,
        lambda2=0.1,
    )

    library = FunnelLibrary(
        segments=[segment],
        segment_indices=[0],
        n=n,
        m=m,
        alpha=0.95,
        mu=1.1,
    )

    K_retrieved = library.get_gain(segment_idx=0)
    assert np.allclose(K_retrieved, K_test)


def test_lmi_builder():
    """Test LMIBuilder creation."""
    n, m = 3, 2
    alpha = 0.95

    builder = LMIBuilder(n=n, m=m, alpha=alpha)

    assert builder.n == n
    assert builder.m == m
    assert builder.alpha == alpha


def test_sdp_solution():
    """Test SDPSolution container."""
    n, m = 3, 2

    solution = SDPSolution(
        segment_idx=0,
        P_i=np.eye(n),
        L_i=np.random.randn(m, n),
        K_i=np.random.randn(m, n),
        volume=1.0,
        nu=0.01,
        lambda1=0.1,
        lambda2=0.1,
        objective_value=0.5,
        status="optimal",
    )

    assert solution.segment_idx == 0
    assert solution.status == "optimal"
    assert solution.P_i.shape == (n, n)
    assert solution.K_i.shape == (m, n)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
