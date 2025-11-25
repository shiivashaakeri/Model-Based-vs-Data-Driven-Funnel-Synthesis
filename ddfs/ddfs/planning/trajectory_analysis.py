"""
Trajectory Analysis and Verification for DDFS.

This module provides utilities to:
- Verify Assumption 3 (bounded increments along nominal trajectory)
- Compute the increment bound constant v
- Analyze trajectory smoothness and quality
- Smooth trajectories to satisfy increment bounds
- Compute Lipschitz constants for Jacobian variation bounds

These utilities support the theoretical requirements of the DDFS algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from ddfs.models.base_model import BaseModel
from ddfs.planning.trajectory import Trajectory
from ddfs.utils.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Analysis Results
# =============================================================================


@dataclass
class IncrementAnalysis:
    """
    Results of trajectory increment analysis (Assumption 3).

    Parameters
    ----------
    v : float
        Maximum increment bound: max_k ||(x(k+1), u(k+1)) - (x(k), u(k))||
    v_state : float
        Maximum state increment: max_k ||x(k+1) - x(k)||
    v_input : float
        Maximum input increment: max_k ||u(k+1) - u(k)||
    increment_norms : np.ndarray
        Combined increment norms at each timestep.
    state_increment_norms : np.ndarray
        State increment norms at each timestep.
    input_increment_norms : np.ndarray
        Input increment norms at each timestep.
    assumption_satisfied : bool
        Whether Assumption 3 is satisfied (v is finite and reasonable).
    """

    v: float
    v_state: float
    v_input: float
    increment_norms: np.ndarray
    state_increment_norms: np.ndarray
    input_increment_norms: np.ndarray
    assumption_satisfied: bool
    max_increment_index: int = 0

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Increment Analysis (Assumption 3)",
            "-" * 40,
            f"Combined increment bound v:  {self.v:.6f}",
            f"State increment bound:       {self.v_state:.6f}",
            f"Input increment bound:       {self.v_input:.6f}",
            f"Max increment at index:      {self.max_increment_index}",
            f"Assumption 3 satisfied:      {self.assumption_satisfied}",
            "",
            "Statistics:",
            f"  Mean combined increment:   {np.mean(self.increment_norms):.6f}",
            f"  Std combined increment:    {np.std(self.increment_norms):.6f}",
            f"  Mean state increment:      {np.mean(self.state_increment_norms):.6f}",
            f"  Mean input increment:      {np.mean(self.input_increment_norms):.6f}",
        ]
        return "\n".join(lines)


@dataclass
class SmoothnessAnalysis:
    """
    Results of trajectory smoothness analysis.

    Parameters
    ----------
    max_velocity : float
        Maximum state velocity (finite difference).
    max_acceleration : float
        Maximum state acceleration (second difference).
    max_jerk : float
        Maximum state jerk (third difference).
    max_input_rate : float
        Maximum input rate of change.
    total_variation_state : float
        Total variation of state trajectory.
    total_variation_input : float
        Total variation of input trajectory.
    path_length : float
        Total path length in position space.
    """

    max_velocity: float
    max_acceleration: float
    max_jerk: float
    max_input_rate: float
    total_variation_state: float
    total_variation_input: float
    path_length: float
    velocity_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    acceleration_profile: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Smoothness Analysis",
            "-" * 40,
            f"Max velocity:          {self.max_velocity:.6f}",
            f"Max acceleration:      {self.max_acceleration:.6f}",
            f"Max jerk:              {self.max_jerk:.6f}",
            f"Max input rate:        {self.max_input_rate:.6f}",
            f"State total variation: {self.total_variation_state:.6f}",
            f"Input total variation: {self.total_variation_input:.6f}",
            f"Path length:           {self.path_length:.6f}",
        ]
        return "\n".join(lines)


@dataclass
class JacobianAnalysis:
    """
    Results of Jacobian variation analysis.

    Parameters
    ----------
    L_J : float
        Estimated Lipschitz constant for Jacobian variation.
    C : float
        Combined constant C = L_J * v for Lemma 2.
    jacobian_variation_norms : np.ndarray
        Jacobian variation norms along trajectory.
    max_A_norm : float
        Maximum norm of state Jacobian A.
    max_B_norm : float
        Maximum norm of input Jacobian B.
    """

    L_J: float
    C: float
    jacobian_variation_norms: np.ndarray
    max_A_norm: float  # noqa: N815
    max_B_norm: float  # noqa: N815

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Jacobian Analysis",
            "-" * 40,
            f"Jacobian Lipschitz L_J:      {self.L_J:.6f}",
            f"Combined constant C=L_J*v:   {self.C:.6f}",
            f"Max ||A|| along trajectory:  {self.max_A_norm:.6f}",
            f"Max ||B|| along trajectory:  {self.max_B_norm:.6f}",
            f"Mean Jacobian variation:     {np.mean(self.jacobian_variation_norms):.6f}",
        ]
        return "\n".join(lines)


# =============================================================================
# Increment Analysis (Assumption 3)
# =============================================================================


def compute_increment_bound(trajectory: Trajectory) -> IncrementAnalysis:
    """
    Compute the increment bound v for Assumption 3.

    Assumption 3 requires:
        ||(x_nom(k+1), u_nom(k+1)) - (x_nom(k), u_nom(k))|| <= v

    for all k = 0, ..., N-2.

    Parameters
    ----------
    trajectory : Trajectory
        Nominal trajectory to analyze.

    Returns
    -------
    IncrementAnalysis
        Analysis results including bound v.
    """
    N = trajectory.N

    # Compute state increments
    state_diffs = np.diff(trajectory.x, axis=0)  # (N, n_states)
    state_increment_norms = np.linalg.norm(state_diffs, axis=1)  # (N,)

    # Compute input increments (N-1 differences for N inputs)
    input_diffs = np.diff(trajectory.u, axis=0)  # (N-1, n_inputs)
    input_increment_norms = np.linalg.norm(input_diffs, axis=1)  # (N-1,)

    # Combined increments: ||(dx, du)|| for k = 0, ..., N-2
    # At step k, we compare (x(k+1), u(k+1)) with (x(k), u(k))
    # So we need state_diff[k] and input_diff[k] for k = 0, ..., N-2
    combined_increment_norms = np.zeros(N - 1)
    for k in range(N - 1):
        dx = state_diffs[k]
        du = input_diffs[k]
        combined = np.concatenate([dx, du])
        combined_increment_norms[k] = np.linalg.norm(combined)

    # Find maximum
    v = np.max(combined_increment_norms)
    v_state = np.max(state_increment_norms)
    v_input = np.max(input_increment_norms) if len(input_increment_norms) > 0 else 0.0
    max_idx = np.argmax(combined_increment_norms)

    # Check if assumption is satisfied (v should be finite and positive)
    assumption_satisfied = np.isfinite(v) and v > 0

    return IncrementAnalysis(
        v=v,
        v_state=v_state,
        v_input=v_input,
        increment_norms=combined_increment_norms,
        state_increment_norms=state_increment_norms,
        input_increment_norms=input_increment_norms,
        assumption_satisfied=assumption_satisfied,
        max_increment_index=max_idx,
    )


def verify_assumption_3(
    trajectory: Trajectory,
    v_max: Optional[float] = None,
) -> Tuple[bool, IncrementAnalysis]:
    """
    Verify that trajectory satisfies Assumption 3.

    Parameters
    ----------
    trajectory : Trajectory
        Nominal trajectory to verify.
    v_max : float, optional
        Maximum allowed increment bound. If None, only checks finiteness.

    Returns
    -------
    satisfied : bool
        Whether Assumption 3 is satisfied.
    analysis : IncrementAnalysis
        Detailed analysis results.
    """
    analysis = compute_increment_bound(trajectory)

    if v_max is not None:
        satisfied = analysis.v <= v_max and analysis.assumption_satisfied
    else:
        satisfied = analysis.assumption_satisfied

    return satisfied, analysis


# =============================================================================
# Smoothness Analysis
# =============================================================================


def analyze_smoothness(
    trajectory: Trajectory,
    position_indices: Optional[List[int]] = None,
) -> SmoothnessAnalysis:
    """
    Analyze trajectory smoothness.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to analyze.
    position_indices : list, optional
        Indices of position states for path length computation.

    Returns
    -------
    SmoothnessAnalysis
        Smoothness metrics.
    """
    dt = trajectory.dt
    x = trajectory.x
    u = trajectory.u

    # State derivatives via finite differences
    x_dot = np.diff(x, axis=0) / dt  # Velocity
    x_ddot = np.diff(x_dot, axis=0) / dt  # Acceleration
    x_dddot = np.diff(x_ddot, axis=0) / dt if len(x_ddot) > 1 else np.array([[0]])  # Jerk

    # Input derivatives
    u_dot = np.diff(u, axis=0) / dt if len(u) > 1 else np.array([[0]])

    # Compute norms
    velocity_norms = np.linalg.norm(x_dot, axis=1)
    acceleration_norms = np.linalg.norm(x_ddot, axis=1) if len(x_ddot) > 0 else np.array([0])
    jerk_norms = np.linalg.norm(x_dddot, axis=1) if len(x_dddot) > 0 else np.array([0])
    input_rate_norms = np.linalg.norm(u_dot, axis=1) if len(u_dot) > 0 else np.array([0])

    # Total variation
    total_variation_state = np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1))
    total_variation_input = np.sum(np.linalg.norm(np.diff(u, axis=0), axis=1)) if len(u) > 1 else 0.0

    # Path length
    if position_indices is None:
        position_indices = list(range(min(3, trajectory.n_states)))
    positions = x[:, position_indices]
    path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

    return SmoothnessAnalysis(
        max_velocity=np.max(velocity_norms),
        max_acceleration=np.max(acceleration_norms),
        max_jerk=np.max(jerk_norms),
        max_input_rate=np.max(input_rate_norms),
        total_variation_state=total_variation_state,
        total_variation_input=total_variation_input,
        path_length=path_length,
        velocity_profile=velocity_norms,
        acceleration_profile=acceleration_norms,
    )


# =============================================================================
# Jacobian Analysis
# =============================================================================


def analyze_jacobian_variation(
    trajectory: Trajectory,
    model: BaseModel,
) -> JacobianAnalysis:
    """
    Analyze Jacobian variation along trajectory for Lemma 2.

    Lemma 2 states:
        ||[A(k) - A(s), B(k) - B(s)]|| <= L_J * v * |k - s|

    This function estimates L_J by computing Jacobian variations.

    Parameters
    ----------
    trajectory : Trajectory
        Nominal trajectory.
    model : BaseModel
        System model for Jacobian computation.

    Returns
    -------
    JacobianAnalysis
        Jacobian analysis results.
    """
    N = trajectory.N

    # Compute Jacobians at each point
    A_list = []
    B_list = []
    for k in range(N):
        A_k, B_k = model.discrete_jacobians(trajectory.x[k], trajectory.u[k])
        A_list.append(A_k)
        B_list.append(B_k)

    # Add final point (using last input)
    A_N, B_N = model.discrete_jacobians(trajectory.x[N], trajectory.u[-1])
    A_list.append(A_N)
    B_list.append(B_N)

    # Compute Jacobian norms
    A_norms = [np.linalg.norm(A, 2) for A in A_list]
    B_norms = [np.linalg.norm(B, 2) for B in B_list]

    # Compute Jacobian variations between consecutive points
    jacobian_variations = []
    for k in range(N):
        A_diff = A_list[k + 1] - A_list[k]
        B_diff = B_list[k + 1] - B_list[k]
        # Combined norm: ||[A_diff, B_diff]||
        combined = np.hstack([A_diff, B_diff])
        jacobian_variations.append(np.linalg.norm(combined, 2))

    jacobian_variation_norms = np.array(jacobian_variations)

    # Compute increment bound v
    increment_analysis = compute_increment_bound(trajectory)
    v = increment_analysis.v

    # Estimate L_J: ||J(k+1) - J(k)|| / (v * 1) for consecutive steps
    # L_J ≈ max_k ||J(k+1) - J(k)|| / v
    if v > 1e-10:
        L_J_estimates = jacobian_variation_norms / v
        L_J = np.max(L_J_estimates)
    else:
        L_J = np.max(jacobian_variation_norms) if len(jacobian_variation_norms) > 0 else 0.0

    # Combined constant C = L_J * v
    C = L_J * v

    return JacobianAnalysis(
        L_J=L_J,
        C=C,
        jacobian_variation_norms=jacobian_variation_norms,
        max_A_norm=np.max(A_norms),
        max_B_norm=np.max(B_norms),
    )


def estimate_lipschitz_constant(
    model: BaseModel,
    x_samples: np.ndarray,
    u_samples: np.ndarray,
    eps: float = 1e-4,
) -> float:
    """
    Estimate Lipschitz constant L_J for Jacobian via sampling.

    Parameters
    ----------
    model : BaseModel
        System model.
    x_samples : np.ndarray
        State samples (n_samples, n_states).
    u_samples : np.ndarray
        Input samples (n_samples, n_inputs).
    eps : float
        Perturbation size for finite differences.

    Returns
    -------
    float
        Estimated Lipschitz constant.
    """
    n_samples = len(x_samples)
    max_variation = 0.0

    for i in range(n_samples):
        x_i = x_samples[i]
        u_i = u_samples[i]

        # Get Jacobian at nominal point
        A_i, B_i = model.discrete_jacobians(x_i, u_i)

        # Perturb state and compute Jacobian variation
        for j in range(model.n_states):
            x_pert = x_i.copy()
            x_pert[j] += eps

            A_pert, B_pert = model.discrete_jacobians(x_pert, u_i)

            A_diff = A_pert - A_i
            B_diff = B_pert - B_i
            combined_norm = np.linalg.norm(np.hstack([A_diff, B_diff]), 2)

            # Lipschitz estimate: ||J_pert - J|| / ||x_pert - x||
            variation = combined_norm / eps
            max_variation = max(max_variation, variation)

        # Perturb input
        for j in range(model.n_inputs):
            u_pert = u_i.copy()
            u_pert[j] += eps

            A_pert, B_pert = model.discrete_jacobians(x_i, u_pert)

            A_diff = A_pert - A_i
            B_diff = B_pert - B_i
            combined_norm = np.linalg.norm(np.hstack([A_diff, B_diff]), 2)

            variation = combined_norm / eps
            max_variation = max(max_variation, variation)

    return max_variation


# =============================================================================
# Trajectory Smoothing
# =============================================================================


def smooth_trajectory_gaussian(
    trajectory: Trajectory,
    sigma: float = 2.0,
    preserve_endpoints: bool = True,
) -> Trajectory:
    """
    Smooth trajectory using Gaussian filter.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to smooth.
    sigma : float
        Gaussian kernel standard deviation (in timesteps).
    preserve_endpoints : bool
        Preserve initial and final states/inputs.

    Returns
    -------
    Trajectory
        Smoothed trajectory.
    """
    x_smooth = gaussian_filter1d(trajectory.x, sigma=sigma, axis=0, mode="nearest")
    u_smooth = gaussian_filter1d(trajectory.u, sigma=sigma, axis=0, mode="nearest")

    if preserve_endpoints:
        # Restore endpoints
        x_smooth[0] = trajectory.x[0]
        x_smooth[-1] = trajectory.x[-1]
        u_smooth[0] = trajectory.u[0]
        u_smooth[-1] = trajectory.u[-1]

    return Trajectory(
        x=x_smooth,
        u=u_smooth,
        dt=trajectory.dt,
        t0=trajectory.t0,
        metadata={**trajectory.metadata, "smoothed": "gaussian", "sigma": sigma},
    )


def smooth_trajectory_cubic(
    trajectory: Trajectory,
    n_points: Optional[int] = None,
) -> Trajectory:
    """
    Smooth trajectory using cubic spline interpolation.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to smooth.
    n_points : int, optional
        Number of output points. Defaults to same as input.

    Returns
    -------
    Trajectory
        Smoothed trajectory.
    """
    if n_points is None:
        n_points = trajectory.N

    t_orig = trajectory.t
    t_new = np.linspace(t_orig[0], t_orig[-1], n_points + 1)

    # Spline for states
    spline_x = CubicSpline(t_orig, trajectory.x, axis=0)
    x_smooth = spline_x(t_new)

    # Spline for inputs
    t_u_orig = trajectory.t_inputs
    t_u_new = np.linspace(t_u_orig[0], t_u_orig[-1], n_points)
    spline_u = CubicSpline(t_u_orig, trajectory.u, axis=0)
    u_smooth = spline_u(t_u_new)

    dt_new = (t_new[-1] - t_new[0]) / n_points

    return Trajectory(
        x=x_smooth,
        u=u_smooth,
        dt=dt_new,
        t0=trajectory.t0,
        metadata={**trajectory.metadata, "smoothed": "cubic"},
    )


def smooth_trajectory_moving_average(
    trajectory: Trajectory,
    window_size: int = 5,
    preserve_endpoints: bool = True,
) -> Trajectory:
    """
    Smooth trajectory using moving average filter.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to smooth.
    window_size : int
        Moving average window size.
    preserve_endpoints : bool
        Preserve initial and final states/inputs.

    Returns
    -------
    Trajectory
        Smoothed trajectory.
    """

    def moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average along axis 0."""
        kernel = np.ones(window) / window
        result = np.zeros_like(data)
        for j in range(data.shape[1]):
            result[:, j] = np.convolve(data[:, j], kernel, mode="same")
        return result

    x_smooth = moving_average(trajectory.x, window_size)
    u_smooth = moving_average(trajectory.u, window_size)

    if preserve_endpoints:
        x_smooth[0] = trajectory.x[0]
        x_smooth[-1] = trajectory.x[-1]
        u_smooth[0] = trajectory.u[0]
        u_smooth[-1] = trajectory.u[-1]

    return Trajectory(
        x=x_smooth,
        u=u_smooth,
        dt=trajectory.dt,
        t0=trajectory.t0,
        metadata={**trajectory.metadata, "smoothed": "moving_average", "window": window_size},
    )


def enforce_increment_bound(
    trajectory: Trajectory,
    v_max: float,
    max_iterations: int = 100,
    relaxation: float = 0.5,
) -> Tuple[Trajectory, bool]:
    """
    Modify trajectory to enforce increment bound.

    Uses iterative projection to reduce large increments.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to modify.
    v_max : float
        Maximum allowed increment bound.
    max_iterations : int
        Maximum iterations for projection.
    relaxation : float
        Relaxation factor for projection (0 < relaxation <= 1).

    Returns
    -------
    trajectory_modified : Trajectory
        Modified trajectory.
    success : bool
        Whether bound was achieved.
    """
    x = trajectory.x.copy()
    u = trajectory.u.copy()
    N = trajectory.N

    for iteration in range(max_iterations):
        # Check current bound
        max_increment = 0.0
        max_idx = 0

        for k in range(N - 1):
            dx = x[k + 1] - x[k]
            du = u[k + 1] - u[k]
            combined = np.concatenate([dx, du])
            norm = np.linalg.norm(combined)

            if norm > max_increment:
                max_increment = norm
                max_idx = k

        if max_increment <= v_max:
            # Success
            return Trajectory(
                x=x,
                u=u,
                dt=trajectory.dt,
                t0=trajectory.t0,
                metadata={
                    **trajectory.metadata,
                    "increment_enforced": True,
                    "iterations": iteration,
                },
            ), True

        # Project the largest increment
        k = max_idx
        dx = x[k + 1] - x[k]
        du = u[k + 1] - u[k]
        combined = np.concatenate([dx, du])
        norm = np.linalg.norm(combined)

        if norm > v_max:
            # Scale down the increment
            scale = v_max / norm
            combined_scaled = combined * scale

            # Distribute the change
            dx_new = combined_scaled[: len(dx)]
            du_new = combined_scaled[len(dx) :]

            # Update with relaxation
            x[k + 1] = x[k] + relaxation * dx_new + (1 - relaxation) * dx
            u[k + 1] = u[k] + relaxation * du_new + (1 - relaxation) * du

    # Did not converge
    return Trajectory(
        x=x,
        u=u,
        dt=trajectory.dt,
        t0=trajectory.t0,
        metadata={**trajectory.metadata, "increment_enforced": False},
    ), False


# =============================================================================
# Comprehensive Trajectory Analysis
# =============================================================================


@dataclass
class TrajectoryAnalysisResult:
    """Comprehensive trajectory analysis results."""

    increment: IncrementAnalysis
    smoothness: SmoothnessAnalysis
    jacobian: Optional[JacobianAnalysis] = None
    feasibility: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate comprehensive summary."""
        sections = [
            self.increment.summary(),
            "",
            self.smoothness.summary(),
        ]
        if self.jacobian is not None:
            sections.extend(["", self.jacobian.summary()])

        return "\n".join(sections)


def analyze_trajectory(
    trajectory: Trajectory,
    model: Optional[BaseModel] = None,
    position_indices: Optional[List[int]] = None,
    compute_jacobians: bool = True,
) -> TrajectoryAnalysisResult:
    """
    Perform comprehensive trajectory analysis.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to analyze.
    model : BaseModel, optional
        Model for Jacobian analysis.
    position_indices : list, optional
        Position indices for path length.
    compute_jacobians : bool
        Whether to compute Jacobian analysis (requires model).

    Returns
    -------
    TrajectoryAnalysisResult
        Comprehensive analysis results.
    """
    # Increment analysis
    increment = compute_increment_bound(trajectory)

    # Smoothness analysis
    smoothness = analyze_smoothness(trajectory, position_indices)

    # Jacobian analysis (if model provided)
    jacobian = None
    if compute_jacobians and model is not None:
        jacobian = analyze_jacobian_variation(trajectory, model)

    return TrajectoryAnalysisResult(
        increment=increment,
        smoothness=smoothness,
        jacobian=jacobian,
    )


def compute_ddfs_constants(
    trajectory: Trajectory,
    model: BaseModel,
    gamma: float,
) -> Dict[str, float]:
    """
    Compute all constants needed for DDFS algorithm.

    Parameters
    ----------
    trajectory : Trajectory
        Nominal trajectory.
    model : BaseModel
        System model.
    gamma : float
        Mismatch bound from Assumption 1.

    Returns
    -------
    dict
        Dictionary with constants: v, L_J, C, gamma, L_r (placeholder).
    """
    # Increment bound v
    increment = compute_increment_bound(trajectory)
    v = increment.v

    # Jacobian analysis
    jacobian = analyze_jacobian_variation(trajectory, model)
    L_J = jacobian.L_J
    C = jacobian.C

    # L_r (Lipschitz constant for linearization error) - estimated
    # This is typically computed from second-order bounds
    # For now, use a heuristic based on Jacobian variation
    L_r = L_J * 0.1  # Placeholder - should be computed properly

    return {
        "v": v,
        "L_J": L_J,
        "C": C,
        "gamma": gamma,
        "L_r": L_r,
        "Te_max": 2 * trajectory.N,  # Maximum segment span
    }
