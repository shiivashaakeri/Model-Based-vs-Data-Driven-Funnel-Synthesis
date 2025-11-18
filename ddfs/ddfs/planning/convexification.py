"""
Convexification utilities for SCvx.

This module provides:
- Dynamics linearization around reference trajectories
- Trust region management
- Affine approximations for convex subproblems
"""

from typing import Dict, Optional, Tuple

import numpy as np

from ddfs.models.base import DynamicalSystem


class DynamicsLinearizer:
    """
    Linearize nonlinear dynamics around reference trajectories.

    Computes discrete-time affine approximation:
        x(k+1) ≈ A_d(k) x(k) + B_d(k) u(k) + c_d(k)

    where:
        A_d(k) = ∂f_d/∂x evaluated at (x_ref(k), u_ref(k))
        B_d(k) = ∂f_d/∂u evaluated at (x_ref(k), u_ref(k))
        c_d(k) = f_d(x_ref(k), u_ref(k)) - A_d(k) x_ref(k) - B_d(k) u_ref(k)
    """

    def __init__(self, model: DynamicalSystem, dt: float, method: str = "rk4"):
        """
        Initialize dynamics linearizer.

        Args:
            model: Dynamical system model to linearize
            dt: Time step
            method: Integration method ('euler' or 'rk4')
        """
        self.model = model
        self.dt = dt
        self.method = method
        self.n = model.state_dim
        self.m = model.input_dim

    def linearize_at_point(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize dynamics at a single point (x, u).

        Args:
            x: State (n,)
            u: Input (m,)

        Returns:
            A_d: Discrete-time state Jacobian (n, n)
            B_d: Discrete-time input Jacobian (n, m)
            c_d: Discrete-time affine term (n,)
        """
        # Get Jacobians via finite difference approximation
        A_d, B_d = self.model.discrete_linearization(x, u, self.dt, self.method)

        # Compute affine term
        x_next = self.model.discrete_dynamics(x, u, self.dt, self.method)
        c_d = x_next - A_d @ x - B_d @ u

        return A_d, B_d, c_d

    def linearize_trajectory(self, x_traj: np.ndarray, u_traj: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Linearize dynamics along a trajectory.

        Args:
            x_traj: State trajectory (N+1, n)
            u_traj: Input trajectory (N, m)

        Returns:
            linearization: Dict with keys:
                 'A': (N, n, n) array of state Jacobians
                 'B': (N, n, m) array of input Jacobians
                 'c': (N, n) array of affine terms
        """
        N = u_traj.shape[0]

        A_seq = np.zeros((N, self.n, self.n))
        B_seq = np.zeros((N, self.n, self.m))
        c_seq = np.zeros((N, self.n))

        for k in range(N):
            A_seq[k], B_seq[k], c_seq[k] = self.linearize_at_point(x_traj[k], u_traj[k])

        return {"A": A_seq, "B": B_seq, "c": c_seq}

    def compute_linearization_error(self, x_traj: np.ndarray, u_traj: np.ndarray) -> np.ndarray:
        """
        Compute linearization error along trajectory.

        Error: ||f(x, u) - (A x + B u + c)||

        Args:
            x_traj: State trajectory (N+1, n)
            u_traj: Input trajectory (N, m)

        Returns:
            error: (N,) array of linearization errors
        """
        N = u_traj.shape[0]
        errors = np.zeros(N)

        for k in range(N):
            A_d, B_d, c_d = self.linearize_at_point(x_traj[k], u_traj[k])

            # True next state
            x_next_true = self.model.discrete_dynamics(x_traj[k], u_traj[k], self.dt, self.method)

            # Linearized prediction
            x_next_linear = A_d @ x_traj[k] + B_d @ u_traj[k] + c_d

            # Error
            errors[k] = np.linalg.norm(x_next_true - x_next_linear)

        return errors

    def validate_linearization(
        self, x_traj: np.ndarray, u_traj: np.ndarray, tolerance: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Check if linearization is accurate along trajectory.

        Args:
            x_traj: State trajectory (N+1, n)
            u_traj: Input trajectory (N, m)
            tolerance: Maximum allowed error

        Returns:
            valid: True if linearization is accurate
            max_error: Maximum error along trajectory
        """
        errors = self.compute_linearization_error(x_traj, u_traj)
        max_error = np.max(errors)
        valid = max_error < tolerance
        return valid, max_error


class TrustRegionManager:
    """
    Manage trust region radius for SCvx iterations.

    Trust region ensures new solution stays close to reference:
        ||x - x_ref|| ≤ ρ (hard constraint)

    or
        penalty: lambda ||x - x_ref||² (soft constraint)

    """  # noqa: RUF002

    def __init__(
        self,
        rho_init: float = 1.0,
        rho_min: float = 1e-4,
        rho_max: float = 10.0,
        beta_expand: float = 1.5,
        gamma_contract: float = 0.7,
    ):
        """
        Initialize trust region manager.

        Args:
            rho_init: Initial trust region radius
            rho_min: Minimum trust region radius
            rho_max: Maximum trust region radius
            beta_expand: Trust region expansion factor
            gamma_contract: Trust region contraction factor
        """
        self.rho_init = rho_init
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.beta_expand = beta_expand
        self.gamma_contract = gamma_contract

        self.history = []

    def get_radius(self) -> float:
        """Get current trust region radius."""
        return self.rho

    def expand(self) -> float:
        """Expand trust region radius (solution accepted)."""
        self.rho = min(self.rho * self.beta_expand, self.rho_max)
        self.history.append(("expand", self.rho))

    def contract(self) -> float:
        """Contract trust region radius (solution rejected or infeasible)."""
        self.rho = max(self.rho * self.gamma_contract, self.rho_min)
        self.history.append(("contract", self.rho))

    def is_too_small(self) -> bool:
        """Check if trust region is too small to continue."""
        return self.rho <= self.rho_min

    def reset(self, rho_new: Optional[float] = None):
        """
        Reset trust region radius.

        Args:
            rho_new: New trust region radius (use initial if None)
        """
        if rho_new is None:
            self.rho = 1.0
        else:
            self.rho = np.clip(rho_new, self.rho_min, self.rho_max)
        self.history.append(("reset", self.rho))

    def get_history(self) -> list:
        """Get trust region update history."""
        return self.history

    def __repr__(self) -> str:
        return f"TrustRegionManager(rho={self.rho:.4f}, rho_min={self.rho_min:.4f}, rho_max={self.rho_max:.4f})"


class ConvexificationHelper:
    """
    Helper utilities for convexification process.
    """

    @staticmethod
    def compute_trajectory_deviation(x_new: np.ndarray, x_ref: np.ndarray) -> Dict[str, float]:
        """
        Compute deviation matrices between trajectories.

        Args:
            x_new: New state trajectory (N+1, n)
            x_ref: Reference state trajectory (N+1, n)

        Returns:
            metrics: Dict with:
                - 'l2': L2 norm ||x_new - x_ref||_2
                - 'linf': L-infinity norm ||x_new - x_ref||_∞
                - 'mean': Mean deviation
                - 'max': Maximum deviation
        """
        diff = x_new - x_ref

        return {
            "l2": np.linalg.norm(diff, ord=2),
            "linf": np.linalg.norm(diff, ord=np.inf),
            "mean": np.mean(np.linalg.norm(diff, axis=1)),
            "max": np.max(np.linalg.norm(diff, axis=1)),
        }

    @staticmethod
    def check_convergence(
        x_new: np.ndarray, x_ref: np.ndarray, u_new: np.ndarray, u_ref: np.ndarray, tol_x: float, tol_u: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check SCvx convergence criteria.

        Args:
            x_new: New state trajectory (N+1, n)
            x_ref: Reference state trajectory (N+1, n)
            u_new: New input trajectory (N, m)
            u_ref: Reference input trajectory (N, m)
            tol_x: State convergence tolerance
            tol_u: Input convergence tolerance

        Returns:
            converged: True if trajectories have converged
            metrics: Convergence metrics
        """
        dx = np.linalg.norm(x_new - x_ref, ord=np.inf)
        du = np.linalg.norm(u_new - u_ref, ord=np.inf)

        metrics = {
            "dx_linf": dx,
            "du_linf": du,
            "dx_l2": np.linalg.norm(x_new - x_ref),
            "du_l2": np.linalg.norm(u_new - u_ref),
        }

        converged = (dx < tol_x) and (du < tol_u)

        return converged, metrics

    @staticmethod
    def compute_control_cost(u_traj: np.ndarray, weight: float = 1.0) -> float:
        """
        Compute control effort cost: sum ||u||²

        Args:
            u_traj: Input trajectory (N, m)
            weight: Weight for control effort

        Returns:
            cost: Control effort cost
        """
        return weight * np.sum(np.linalg.norm(u_traj, axis=1) ** 2)

    @staticmethod
    def compute_terminal_cost(x_final: np.ndarray, x_goal: np.ndarray, weight: float = 1.0) -> float:
        """
        Compute terminal cost: ||x_final - x_goal||²

        Args:
            x_final: Final state (n,)
            x_goal: Goal state (n,)
            weight: Weight for terminal cost

        Returns:
            cost: Terminal cost
        """
        return weight * np.linalg.norm(x_final - x_goal) ** 2

    @staticmethod
    def compute_trust_region_violation(x_new: np.ndarray, x_ref: np.ndarray, rho: float) -> float:
        """
        Compute maximum trust region violation.

        Args:
            x_new: New state trajectory (N+1, n)
            x_ref: Reference state trajectory (N+1, n)
            rho: Trust region radius

        Returns:
            violation: max(||x_new - x_ref||_2 - rho, 0)
        """
        deviations = np.linalg.norm(x_new - x_ref, axis=1)
        violations = np.maximum(deviations - rho, 0)
        return np.max(violations)
