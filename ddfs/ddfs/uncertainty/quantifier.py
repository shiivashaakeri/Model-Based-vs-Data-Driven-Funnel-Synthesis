"""Uncertainty quantification from collected data.

This module provides the UncertaintyQuantifier class for quantifying model
uncertainty from collected trajectory data, enabling robust control design
based on data-driven uncertainty bounds.
"""

import logging
from dataclasses import dataclass

import numpy as np
from core.config import DDFSConfig
from models.base import VehicleModel
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyBounds:
    """Container for uncertainty quantification bounds.

    Attributes:
        gamma: Bound on additive uncertainty ||w||
        L_J: Lipschitz constant for Jacobian uncertainty
        L_r: Lipschitz constant for remainder term
        beta_i: Per-timestep uncertainty bounds (list of N values)
        n_samples: Number of samples used for estimation
    """

    gamma: float
    L_J: float
    L_r: float
    beta_i: NDArray[np.float64]  # Shape: (N,)
    n_samples: int

    def __repr__(self) -> str:
        return (
            f"UncertaintyBounds(gamma={self.gamma:.6f}, "
            f"L_J={self.L_J:.6f}, L_r={self.L_r:.6f}, "
            f"n_samples={self.n_samples})"
        )


class UncertaintyQuantifier:
    """UncertaintyQuantifier class for quantifying model uncertainty.

    This class analyzes collected trajectory data and quantifies the uncertainty
    between the plant model and the digital twin, providing uncertainty bounds
    that can be used for robust funnel synthesis.

    The quantification follows the framework:
    - gamma: bounds additive uncertainty ||w||
    - L_J: Lipschitz constant for Jacobian uncertainty
    - L_r: Lipschitz constant for remainder term
    - beta_i: per-timestep uncertainty bounds for funnel synthesis

    Example:
        >>> config = DDFSConfig(...)
        >>> plant = UnicyclePlant(config)
        >>> twin = UnicycleTwin(config)
        >>> quantifier = UncertaintyQuantifier(config, plant, twin)
        >>>
        >>> # Quantify uncertainty from collected data
        >>> bounds = quantifier.quantify(states, controls)
        >>> print(f"gamma = {bounds.gamma}")
        >>> print(f"L_J = {bounds.L_J}")
        >>> print(f"beta_i range: [{bounds.beta_i.min()}, {bounds.beta_i.max()}]")
    """

    def __init__(
        self,
        config: DDFSConfig,
        plant: VehicleModel,
        twin: VehicleModel,
    ):
        """Initialize the uncertainty quantifier.

        Args:
            config: DDFS configuration
            plant: Plant model (ground truth)
            twin: Digital twin model (nominal)
        """
        self.config = config
        self.plant = plant
        self.twin = twin

        logger.info("Initialized UncertaintyQuantifier")
        logger.info(f"  State dimension: {self.config.nx}")
        logger.info(f"  Control dimension: {self.config.nu}")
        logger.info(f"  Time step: {self.config.dt}")

    def quantify(
        self,
        states: NDArray[np.float64],
        controls: NDArray[np.float64],
        n_lipschitz_samples: int = 1000,
    ) -> UncertaintyBounds:
        """Quantify uncertainty from collected trajectory data.

        This method computes all uncertainty bounds needed for robust funnel
        synthesis:
        1. gamma: additive uncertainty bound
        2. L_J: Jacobian Lipschitz constant
        3. L_r: remainder Lipschitz constant
        4. beta_i: per-timestep uncertainty bounds

        Args:
            states: Collected state trajectory, shape (N, nx)
            controls: Collected control trajectory, shape (N-1, nu)
            n_lipschitz_samples: Number of samples for Lipschitz estimation

        Returns:
            UncertaintyBounds object containing all computed bounds

        Example:
            >>> states = np.random.randn(100, 3)
            >>> controls = np.random.randn(99, 2)
            >>> bounds = quantifier.quantify(states, controls)
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: UNCERTAINTY QUANTIFICATION")
        logger.info("=" * 60)

        N = len(states) - 1
        logger.info(f"Processing trajectory with {N} timesteps")
        logger.info(f"State shape: {states.shape}")
        logger.info(f"Control shape: {controls.shape}")

        # Validate inputs
        self._validate_inputs(states, controls)

        # Compute gamma (additive uncertainty bound)
        logger.info("\n[1/4] Computing gamma (additive uncertainty)...")
        gamma = self._compute_gamma(states, controls)
        logger.info(f"  gamma = {gamma:.6f}")

        # Compute L_J (Jacobian Lipschitz constant)
        logger.info("\n[2/4] Computing L_J (Jacobian Lipschitz)...")
        L_J = self._compute_lipschitz_jacobian(n_lipschitz_samples)
        logger.info(f"  L_J = {L_J:.6f}")

        # Compute L_r (remainder Lipschitz constant)
        logger.info("\n[3/4] Computing L_r (remainder Lipschitz)...")
        L_r = self._compute_lipschitz_remainder(n_lipschitz_samples)
        logger.info(f"  L_r = {L_r:.6f}")

        # Compute beta_i (per-timestep bounds)
        logger.info("\n[4/4] Computing beta_i (per-timestep bounds)...")
        beta_i = self._compute_beta_i(states, controls, gamma, L_J, L_r)
        logger.info(f"  beta_i: min={beta_i.min():.6f}, max={beta_i.max():.6f}, mean={beta_i.mean():.6f}")

        bounds = UncertaintyBounds(
            gamma=gamma,
            L_J=L_J,
            L_r=L_r,
            beta_i=beta_i,
            n_samples=N,
        )

        logger.info("\n" + "=" * 60)
        logger.info("UNCERTAINTY QUANTIFICATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"{bounds}")

        return bounds

    def _validate_inputs(
        self,
        states: NDArray[np.float64],
        controls: NDArray[np.float64],
    ) -> None:
        """Validate input trajectory data.

        Args:
            states: State trajectory, shape (N, nx)
            controls: Control trajectory, shape (N-1, nu)

        Raises:
            ValueError: If inputs have incorrect shapes or dimensions
        """
        N_states = len(states)
        N_controls = len(controls)

        if N_states != N_controls + 1:
            raise ValueError(
                f"State trajectory length ({N_states}) must be control trajectory length + 1 ({N_controls + 1})"
            )

        if states.shape[1] != self.config.nx:
            raise ValueError(f"State dimension {states.shape[1]} != config.nx {self.config.nx}")

        if controls.shape[1] != self.config.nu:
            raise ValueError(f"Control dimension {controls.shape[1]} != config.nu {self.config.nu}")

        logger.debug("Input validation passed")

    def _compute_gamma(
        self,
        states: NDArray[np.float64],
        controls: NDArray[np.float64],
    ) -> float:
        """Compute additive uncertainty bound gamma.

        gamma = max_i ||x_{i+1} - f_twin(x_i, u_i)||

        This quantifies the maximum additive disturbance between the plant
        and twin models.

        Args:
            states: State trajectory, shape (N, nx)
            controls: Control trajectory, shape (N-1, nu)

        Returns:
            gamma: Additive uncertainty bound
        """
        N = len(controls)
        max_error = 0.0

        for i in range(N):
            x_current = states[i]
            u_current = controls[i]
            x_next_actual = states[i + 1]

            # Propagate twin model
            x_next_twin = self.twin.discrete_dynamics(x_current, u_current)

            # Compute error
            error = np.linalg.norm(x_next_actual - x_next_twin)
            max_error = max(max_error, error)

        gamma = max_error
        logger.debug(f"Computed gamma from {N} timesteps")

        return gamma

    def _compute_lipschitz_jacobian(
        self,
        n_samples: int,
    ) -> float:
        """Compute Lipschitz constant for Jacobian using random sampling.

        L_J = max ||A(x1,u1) - A(x2,u2)|| / ||(x1,u1) - (x2,u2)||

        where A(x,u) is the Jacobian of the twin dynamics.

        Args:
            n_samples: Number of random sample pairs to test

        Returns:
            L_J: Lipschitz constant for Jacobian
        """
        max_lipschitz = 0.0

        # Sample from workspace
        x_min = np.array([self.config.workspace.x_min, self.config.workspace.y_min, -np.pi])
        x_max = np.array([self.config.workspace.x_max, self.config.workspace.y_max, np.pi])
        u_min = np.array([self.config.constraints.v_min, self.config.constraints.omega_min])
        u_max = np.array([self.config.constraints.v_max, self.config.constraints.omega_max])

        for _ in range(n_samples):
            # Sample two random points
            x1 = np.random.uniform(x_min, x_max)
            u1 = np.random.uniform(u_min, u_max)
            x2 = np.random.uniform(x_min, x_max)
            u2 = np.random.uniform(u_min, u_max)

            # Compute Jacobians
            A1, B1 = self.twin.compute_linearization(x1, u1)
            A2, B2 = self.twin.compute_linearization(x2, u2)

            # Stack into full Jacobian [A, B]
            J1 = np.hstack([A1, B1])
            J2 = np.hstack([A2, B2])

            # Compute Lipschitz ratio
            numerator = np.linalg.norm(J1 - J2, ord="fro")

            # Stack state and control for denominator
            z1 = np.concatenate([x1, u1])
            z2 = np.concatenate([x2, u2])
            denominator = np.linalg.norm(z1 - z2)

            if denominator > 1e-10:  # Avoid division by zero
                lipschitz = numerator / denominator
                max_lipschitz = max(max_lipschitz, lipschitz)

        L_J = max_lipschitz
        logger.debug(f"Computed L_J from {n_samples} samples")

        return L_J

    def _compute_lipschitz_remainder(
        self,
        n_samples: int,
    ) -> float:
        """Compute Lipschitz constant for remainder term using random sampling.

        L_r = max ||r(x1,u1,dx1,du1) - r(x2,u2,dx2,du2)|| / ||(x1,u1,dx1,du1) - (x2,u2,dx2,du2)||

        where r(x,u,dx,du) = f(x+dx,u+du) - f(x,u) - A(x,u)dx - B(x,u)du
        is the Taylor series remainder term.

        Args:
            n_samples: Number of random sample pairs to test

        Returns:
            L_r: Lipschitz constant for remainder
        """
        max_lipschitz = 0.0

        # Sample from workspace
        x_min = np.array([self.config.workspace.x_min, self.config.workspace.y_min, -np.pi])
        x_max = np.array([self.config.workspace.x_max, self.config.workspace.y_max, np.pi])
        u_min = np.array([self.config.constraints.v_min, self.config.constraints.omega_min])
        u_max = np.array([self.config.constraints.v_max, self.config.constraints.omega_max])

        # Sample perturbations
        dx_scale = 0.1 * (x_max - x_min)
        du_scale = 0.1 * (u_max - u_min)

        for _ in range(n_samples):
            # Sample two random base points and perturbations
            x1 = np.random.uniform(x_min, x_max)
            u1 = np.random.uniform(u_min, u_max)
            dx1 = np.random.uniform(-dx_scale, dx_scale)
            du1 = np.random.uniform(-du_scale, du_scale)

            x2 = np.random.uniform(x_min, x_max)
            u2 = np.random.uniform(u_min, u_max)
            dx2 = np.random.uniform(-dx_scale, dx_scale)
            du2 = np.random.uniform(-du_scale, du_scale)

            # Compute remainder at point 1
            r1 = self._compute_remainder(x1, u1, dx1, du1)

            # Compute remainder at point 2
            r2 = self._compute_remainder(x2, u2, dx2, du2)

            # Compute Lipschitz ratio
            numerator = np.linalg.norm(r1 - r2)

            # Stack all variables for denominator
            z1 = np.concatenate([x1, u1, dx1, du1])
            z2 = np.concatenate([x2, u2, dx2, du2])
            denominator = np.linalg.norm(z1 - z2)

            if denominator > 1e-10:  # Avoid division by zero
                lipschitz = numerator / denominator
                max_lipschitz = max(max_lipschitz, lipschitz)

        L_r = max_lipschitz
        logger.debug(f"Computed L_r from {n_samples} samples")

        return L_r

    def _compute_remainder(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        dx: NDArray[np.float64],
        du: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute Taylor series remainder term.

        r(x,u,dx,du) = f(x+dx,u+du) - f(x,u) - A(x,u)dx - B(x,u)du

        Args:
            x: State
            u: Control
            dx: State perturbation
            du: Control perturbation

        Returns:
            Remainder vector
        """
        # Evaluate dynamics at base point
        f_base = self.twin.discrete_dynamics(x, u)

        # Evaluate dynamics at perturbed point
        f_pert = self.twin.discrete_dynamics(x + dx, u + du)

        # Compute linearization
        A, B = self.twin.compute_linearization(x, u)

        # Compute remainder
        remainder = f_pert - f_base - A @ dx - B @ du

        return remainder

    def _compute_beta_i(
        self,
        states: NDArray[np.float64],  # noqa: ARG002
        controls: NDArray[np.float64],
        gamma: float,
        L_J: float,
        L_r: float,
    ) -> NDArray[np.float64]:
        """Compute per-timestep uncertainty bounds beta_i.

        For each timestep i, beta_i provides a bound on the uncertainty
        propagation used in funnel synthesis. The exact formula depends on
        the specific uncertainty model being used.

        A simple approach: beta_i = gamma + safety_margin

        Args:
            states: State trajectory, shape (N, nx)
            controls: Control trajectory, shape (N-1, nu)
            gamma: Additive uncertainty bound
            L_J: Jacobian Lipschitz constant
            L_r: Remainder Lipschitz constant

        Returns:
            beta_i: Per-timestep bounds, shape (N,)
        """
        N = len(controls)
        beta_i = np.zeros(N)

        # Simple conservative approach: use gamma plus a margin based on Lipschitz constants
        safety_margin = 0.1 * (L_J + L_r)

        for i in range(N):
            # Base uncertainty
            beta_i[i] = gamma + safety_margin

            # Could add more sophisticated per-timestep analysis here
            # For example, consider local trajectory curvature, velocity, etc.

        logger.debug(f"Computed beta_i for {N} timesteps")

        return beta_i

    def save_bounds(
        self,
        bounds: UncertaintyBounds,
        filepath: str,
    ) -> None:
        """Save uncertainty bounds to file.

        Args:
            bounds: Uncertainty bounds to save
            filepath: Path to save file
        """
        data = {
            "gamma": bounds.gamma,
            "L_J": bounds.L_J,
            "L_r": bounds.L_r,
            "beta_i": bounds.beta_i,
            "n_samples": bounds.n_samples,
        }
        np.savez(filepath, **data)
        logger.info(f"Saved uncertainty bounds to {filepath}")

    def load_bounds(
        self,
        filepath: str,
    ) -> UncertaintyBounds:
        """Load uncertainty bounds from file.

        Args:
            filepath: Path to load file

        Returns:
            Loaded uncertainty bounds
        """
        data = np.load(filepath)
        bounds = UncertaintyBounds(
            gamma=float(data["gamma"]),
            L_J=float(data["L_J"]),
            L_r=float(data["L_r"]),
            beta_i=data["beta_i"],
            n_samples=int(data["n_samples"]),
        )
        logger.info(f"Loaded uncertainty bounds from {filepath}")
        return bounds
