# ddfs/ddfs/data_collection/collector.py

"""
Data collection for trajectory and excitation signal generation.

This module provides classes for collecting trajectory data from the plant,
including DataCollector for managing data collection, Trajectory for
representing collected trajectories, and ExcitationSignalGenerator for
generating excitation signals to improve data quality.

Phase 2: Offline Data Collection
---------------------------------
Collects M trajectories from the plant (with obstacles removed for safety)
using open-loop control with excitation signals:
    u(k) = u_nom(k) + ε(k)

Key components:
- DataCollector: Collects M trials from plant
- Trajectory: Single trajectory container
- ExcitationSignalGenerator: Generates excitation signals
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np

from ddfs.models.base import PlantModel
from ddfs.planning.nominal_trajectory import NominalTrajectory


@dataclass
class Trajectory:
    """
    Single trajectory data container.

    Stores a complete trajectory collected from the plant, including
    states, inputs, and deviations from the nominal trajectory.

    Attributes
    ----------
    x : np.ndarray
        State trajectory, shape (N+1, n)
    u : np.ndarray
        Input trajectory, shape (N, m)
    eta : np.ndarray
        State deviations from nominal: η(k) = x(k) - x_nom(k), shape (N+1, n)
    xi : np.ndarray
        Input deviations from nominal: ξ(k) = u(k) - u_nom(k), shape (N, m)
    trial_id : int
        Trial identifier (1 to M)
    x0 : np.ndarray
        Initial state for this trial, shape (n,)

    Examples
    --------
    >>> import numpy as np
    >>> from ddfs.data_collection import Trajectory
    >>>
    >>> N, n, m = 10, 3, 2
    >>> x = np.random.randn(N+1, n)
    >>> u = np.random.randn(N, m)
    >>> eta = np.random.randn(N+1, n)
    >>> xi = np.random.randn(N, m)
    >>>
    >>> traj = Trajectory(x=x, u=u, eta=eta, xi=xi, trial_id=1, x0=x[0])
    >>> print(traj.N, traj.state_dim, traj.input_dim)
    10 3 2
    """

    x: np.ndarray  # (N+1, n) - State trajectory
    u: np.ndarray  # (N, m) - Input trajectory
    eta: np.ndarray  # (N+1, n) - State deviations
    xi: np.ndarray  # (N, m) - Input deviations
    trial_id: int  # Trial identifier
    x0: np.ndarray  # (n,) - Initial state

    def __post_init__(self):
        """Validate dimensions."""
        N_plus_1 = self.x.shape[0]
        N = self.u.shape[0]

        if N_plus_1 != N + 1:
            raise ValueError(f"x has {N_plus_1} rows, u has {N} rows (expected N+1 and N)")
        if self.eta.shape != self.x.shape:
            raise ValueError("eta and x must have same shape")
        if self.xi.shape != self.u.shape:
            raise ValueError("xi and u must have same shape")

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

    def save(self, path: Union[Path, str]):
        """
        Save trajectory to file.

        Parameters
        ----------
        path : Path or str
            Path to save location
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[Path, str]) -> "Trajectory":
        """
        Load trajectory from file.

        Parameters
        ----------
        path : Path or str
            Path to saved trajectory

        Returns
        -------
        trajectory : Trajectory
            Loaded trajectory object
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        """String representation."""
        return f"Trajectory(trial={self.trial_id}, N={self.N}, state_dim={self.state_dim}, input_dim={self.input_dim})"


class ExcitationSignalGenerator:
    """
    Generates excitation signals for data collection.

    Excitation signals ε(k) are added to nominal inputs to create
    rich data for uncertainty quantification:
        u(k) = u_nom(k) + ε(k)

    Supported signal types:
        - 'gaussian': White Gaussian noise
        - 'chirp': Frequency-swept sinusoid
        - 'multisine': Sum of multiple sinusoids
        - 'prbs': Pseudo-random binary sequence

    Parameters
    ----------
    signal_type : str, optional
        Type of excitation ('gaussian', 'chirp', 'multisine', 'prbs'), by default 'gaussian'
    amplitude : float, optional
        Signal amplitude, by default 0.1
    seed : int, optional
        Random seed for reproducibility, by default None

    Examples
    --------
    >>> from ddfs.data_collection import ExcitationSignalGenerator
    >>>
    >>> gen = ExcitationSignalGenerator(signal_type='gaussian', amplitude=0.1, seed=42)
    >>> epsilon = gen.generate(N=100, m=2)
    >>> print(epsilon.shape)
    (100, 2)
    """

    def __init__(
        self,
        signal_type: str = "gaussian",
        amplitude: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize excitation signal generator.

        Parameters
        ----------
        signal_type : str, optional
            Type of excitation
        amplitude : float, optional
            Signal amplitude
        seed : int, optional
            Random seed
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
        epsilon : np.ndarray
            Excitation signal, shape (N, m)
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

    def __repr__(self) -> str:
        """String representation."""
        return f"ExcitationSignalGenerator(type='{self.signal_type}', amplitude={self.amplitude}, seed={self.seed})"


class DataCollector:
    """
    Collects trajectory data from plant.

    Applies open-loop control with excitation:
        u(k) = u_nom(k) + ε(k)

    NO feedback is used during data collection!

    Obstacles should be REMOVED during data collection for safety.

    Parameters
    ----------
    plant : PlantModel
        Plant model (with mismatch)
    nominal : NominalTrajectory
        Nominal trajectory from Phase 1
    config : Dict[str, Any]
        Configuration with:
        - M: Number of trials
        - initial_sampling: Initial state sampling config
        - excitation: Excitation signal config

    Examples
    --------
    >>> from ddfs.core import DDFSConfig
    >>> from ddfs.models import UnicycleTwin, UnicyclePlant
    >>> from ddfs.planning import NominalTrajectory
    >>> from ddfs.data_collection import DataCollector
    >>>
    >>> # Setup
    >>> config = DDFSConfig('config/ddfs_config.yaml')
    >>> twin = UnicycleTwin(dt=0.131)
    >>> plant = UnicyclePlant(twin, velocity_scale=0.95)
    >>>
    >>> # Assume we have nominal trajectory
    >>> # nominal = ...
    >>>
    >>> # Collect data
    >>> collector = DataCollector(plant, nominal, config={'M': 50})
    >>> trajectories = collector.collect_trials()
    """

    def __init__(
        self,
        plant: PlantModel,
        nominal: NominalTrajectory,
        config: Dict[str, Any],
    ):
        """
        Initialize data collector.

        Parameters
        ----------
        plant : PlantModel
            Plant model (with mismatch)
        nominal : NominalTrajectory
            Nominal trajectory
        config : dict
            Configuration dictionary
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
            signal_type=self.excitation_type,
            amplitude=self.excitation_amplitude,
            seed=self.excitation_seed,
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
        x0 : np.ndarray
            Sampled initial state, shape (n,)
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
        verbose : bool, optional
            Print progress, by default False

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
        trajectory = Trajectory(
            x=x_traj,
            u=u_traj,
            eta=eta,
            xi=xi,
            trial_id=trial_id,
            x0=x0,
        )

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
        verbose : bool, optional
            Print progress, by default True

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

    def __repr__(self) -> str:
        """String representation."""
        return f"DataCollector(M={self.M}, excitation='{self.excitation_type}', amplitude={self.excitation_amplitude})"
