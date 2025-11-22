# ddfs/ddfs/data_collection/collector.py
"""
Data Collection Module for Phase 2: Offline Data Collection

This module collects M trajectories from the plant (with obstacles removed)
using open-loop control with excitation signals.

Key components:
- DataCollector: Collects M trials from plant
- Trajectory: Single trajectory container
- ExcitationSignalGenerator: Generates excitation signals
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np

from ddfs.models.base import PlantModel
from ddfs.planning.nominal_trajectory import NominalTrajectory


@dataclass
class Trajectory:
    """
    Single trajectory data container.

    Attributes
    ----------
    x : np.ndarray, shape (N+1, n)
        State trajectory
    u : np.ndarray, shape (N, m)
        Input trajectory
    eta : np.ndarray, shape (N+1, n)
        State deviations from nominal: η(k) = x(k) - x_nom(k)
    xi : np.ndarray, shape (N, m)
        Input deviations from nominal: ξ(k) = u(k) - u_nom(k)
    trial_id : int
        Trial identifier (1 to M)
    x0 : np.ndarray, shape (n,)
        Initial state for this trial
    """

    x: np.ndarray
    u: np.ndarray
    eta: np.ndarray
    xi: np.ndarray
    trial_id: int
    x0: np.ndarray

    def __post_init__(self):
        """Validate dimensions."""
        N_plus_1 = self.x.shape[0]
        N = self.u.shape[0]
        assert N_plus_1 == N + 1, f"x has {N_plus_1} rows, u has {N} rows"
        assert self.eta.shape == self.x.shape, "eta and x must have same shape"
        assert self.xi.shape == self.u.shape, "xi and u must have same shape"

    @property
    def N(self) -> int:
        """Horizon length."""
        return self.u.shape[0]

    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.x.shape[1]

    @property
    def input_dim(self) -> int:
        """Input dimension."""
        return self.u.shape[1]

    def save(self, path: Path):
        """Save trajectory to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Trajectory":
        """Load trajectory from file."""
        with open(path, "rb") as f:
            return pickle.load(f)


class ExcitationSignalGenerator:
    """
    Generates excitation signals for data collection.

    Supports multiple excitation types:
    - 'gaussian': White Gaussian noise
    - 'chirp': Frequency-swept sinusoid
    - 'multisine': Sum of multiple sinusoids
    - 'prbs': Pseudo-random binary sequence
    """

    def __init__(self, signal_type: str = "gaussian", amplitude: float = 0.1, seed: Optional[int] = None):
        """
        Initialize excitation signal generator.

        Parameters
        ----------
        signal_type : str
            Type of excitation ('gaussian', 'chirp', 'multisine', 'prbs')
        amplitude : float
            Signal amplitude
        seed : Optional[int]
            Random seed for reproducibility
        """
        self.signal_type = signal_type
        self.amplitude = amplitude
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def generate(self, N: int, m: int) -> np.ndarray:
        """
        Generate excitation signal.

        Parameters
        ----------
        N : int
            Number of timesteps
        m : int
            Input dimension

        Returns
        -------
        epsilon : np.ndarray, shape (N, m)
            Excitation signal
        """
        if self.signal_type == "gaussian":
            return self._gaussian(N, m)
        elif self.signal_type == "chirp":
            return self._chirp(N, m)
        elif self.signal_type == "multisine":
            return self._multisine(N, m)
        elif self.signal_type == "prbs":
            return self._prbs(N, m)
        else:
            raise ValueError(f"Unknown signal type: {self.signal_type}")

    def _gaussian(self, N: int, m: int) -> np.ndarray:
        """White Gaussian noise."""
        return self.amplitude * np.random.randn(N, m)

    def _chirp(self, N: int, m: int) -> np.ndarray:
        """Frequency-swept sinusoid."""
        t = np.linspace(0, 1, N)
        epsilon = np.zeros((N, m))

        for i in range(m):
            # Chirp from low to high frequency
            f0 = 0.1  # Start frequency
            f1 = 2.0  # End frequency
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2)
            epsilon[:, i] = self.amplitude * np.sin(phase)

        return epsilon

    def _multisine(self, N: int, m: int) -> np.ndarray:
        """Sum of multiple sinusoids at different frequencies."""
        t = np.linspace(0, 1, N)
        epsilon = np.zeros((N, m))

        # Use 5 frequencies per input
        frequencies = [0.5, 1.0, 2.0, 3.0, 5.0]

        for i in range(m):
            signal = np.zeros(N)
            for freq in frequencies:
                phase = 2 * np.pi * freq * t + np.random.rand() * 2 * np.pi
                signal += np.sin(phase)
            epsilon[:, i] = self.amplitude * signal / len(frequencies)

        return epsilon

    def _prbs(self, N: int, m: int) -> np.ndarray:
        """Pseudo-random binary sequence."""
        epsilon = np.zeros((N, m))

        for i in range(m):
            # Generate PRBS
            prbs = np.random.choice([-1, 1], size=N)
            epsilon[:, i] = self.amplitude * prbs

        return epsilon


class DataCollector:
    """
    Collects trajectory data from plant.

    Applies open-loop control with excitation:
        u(k) = u_nom(k) + ε(k)

    NO feedback is used during data collection!
    """

    def __init__(self, plant: PlantModel, nominal: NominalTrajectory, config: Dict[str, Any]):
        """
        Initialize data collector.

        Parameters
        ----------
        plant : PlantModel
            Plant model (with mismatch)
        nominal : NominalTrajectory
            Nominal trajectory
        config : Dict[str, Any]
            Configuration dictionary with:
            - M: Number of trials
            - initial_sampling: Initial state sampling config
            - excitation: Excitation signal config
        """
        self.plant = plant
        self.nominal = nominal
        self.config = config

        # Extract config
        self.M = config.get("M", 50)

        # Initial sampling config
        initial_config = config.get("initial_sampling", {})
        self.P_min_0 = initial_config.get("P_min_0", None)
        self.initial_std_scale = initial_config.get("std_scale", 0.1)

        # Excitation config
        excitation_config = config.get("excitation", {})
        self.excitation_type = excitation_config.get("type", "gaussian")
        self.excitation_amplitude = excitation_config.get("amplitude", 0.1)
        self.excitation_seed = excitation_config.get("seed", None)

        # Create excitation generator
        self.excitation_gen = ExcitationSignalGenerator(
            signal_type=self.excitation_type, amplitude=self.excitation_amplitude, seed=self.excitation_seed
        )

    def _sample_initial_state(self, trial_id: int) -> np.ndarray:  # noqa: ARG002
        """
        Sample initial state from ellipsoid around x_nom(0).

        If P_min_0 is provided, samples from E(P_min_0).
        Otherwise, samples from Gaussian around x_nom(0).

        Parameters
        ----------
        trial_id : int
            Trial identifier (for seeding)

        Returns
        -------
        x0 : np.ndarray, shape (n,)
            Sampled initial state
        """
        x_nom_0 = self.nominal.x_nom[0]
        n = self.nominal.state_dim

        if self.P_min_0 is not None:
            # Sample from ellipsoid E(P_min_0)
            # x = x_nom_0 + P_min_0^(-1/2) * w, where w ~ Uniform(||w|| ≤ 1)

            # Compute P_min_0^(-1/2)
            eigvals, eigvecs = np.linalg.eigh(self.P_min_0)
            P_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Sample uniformly from unit ball
            # Use accept-reject or normalized Gaussian
            u = np.random.randn(n)
            u = u / np.linalg.norm(u)  # On unit sphere
            r = np.random.rand() ** (1.0 / n)  # Radius
            w = r * u

            # Transform to ellipsoid
            delta_x = P_inv_sqrt @ w
            x0 = x_nom_0 + delta_x
        else:
            # Sample from Gaussian
            std = self.initial_std_scale
            x0 = x_nom_0 + std * np.random.randn(n)

        return x0

    def collect_single_trial(self, trial_id: int, verbose: bool = False) -> Trajectory:
        """
        Collect single trajectory.

        Applies open-loop control:
            u(k) = u_nom(k) + ε(k)

        Parameters
        ----------
        trial_id : int
            Trial identifier
        verbose : bool
            Print progress

        Returns
        -------
        trajectory : Trajectory
            Collected trajectory data
        """
        N = self.nominal.N
        n = self.nominal.state_dim
        m = self.nominal.input_dim
        dt = self.nominal.dt

        # Sample initial state
        x0 = self._sample_initial_state(trial_id)

        if verbose:
            print(f"  Trial {trial_id}/{self.M}: x0 = {x0}")

        # Generate excitation signal
        epsilon = self.excitation_gen.generate(N, m)

        # Pre-allocate arrays
        x_traj = np.zeros((N + 1, n))
        u_traj = np.zeros((N, m))

        x_traj[0] = x0

        # Open-loop rollout
        for k in range(N):
            # Control: u(k) = u_nom(k) + ε(k)
            u_nom_k = self.nominal.u_nom[k]
            u_k = u_nom_k + epsilon[k]
            u_traj[k] = u_k

            # Step plant (convert to JAX arrays for plant)
            x_next = self.plant.step(jnp.array(x_traj[k]), jnp.array(u_k), dt)
            x_traj[k + 1] = np.array(x_next)

        # Compute deviations
        eta = x_traj - self.nominal.x_nom  # (N+1, n)
        xi = u_traj - self.nominal.u_nom  # (N, m)

        # Create trajectory object
        trajectory = Trajectory(x=x_traj, u=u_traj, eta=eta, xi=xi, trial_id=trial_id, x0=x0)

        return trajectory

    def collect_trials(self, verbose: bool = True) -> List[Trajectory]:
        """
        Collect M trajectories.

        For each trial m = 1, ..., M:
        1. Sample x0^(m) from E(P_min_0) around x_nom(0)
        2. Run open-loop for N steps: u(k) = u_nom(k) + ε(k)
        3. Record full trajectory

        Parameters
        ----------
        verbose : bool
            Print progress

        Returns
        -------
        trajectories : List[Trajectory]
            List of M collected trajectories
        """
        if verbose:
            print(f"\nCollecting {self.M} trajectories...")
            print(f"  Excitation: {self.excitation_type}, amplitude={self.excitation_amplitude}")
            print(f"  Horizon: N={self.nominal.N}, dt={self.nominal.dt}")

        trajectories = []

        for m in range(1, self.M + 1):
            traj = self.collect_single_trial(m, verbose=verbose and m <= 5)
            trajectories.append(traj)

            if verbose and m % 10 == 0:
                print(f"  Collected {m}/{self.M} trajectories")

        if verbose:
            print(f"✓ Collected all {self.M} trajectories")

        return trajectories
