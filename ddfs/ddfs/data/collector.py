"""
Offline data collection for DDFS.

This module collects trajectory data from the physical plant by:
1. Sampling initial states from an ellipsoid around the nominal initial state
2. Running the plant with nominal inputs + excitation signals
3. Recording state deviations, input deviations, and next state deviations
4. Storing data for later segmentation and funnel synthesis

The collected data is used to build Hankel matrices for data-driven control.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ddfs.models.plant import PlantModel


class OfflineDataCollector:
    """
    Collect offline trajectory data from the physical plant.

    Samples initial states, runs trajectries with excitation,
    and stores deviation data for funnel synthesis.
    """

    def __init__(
        self,
        plant: PlantModel,
        nominal_x: np.ndarray,
        nominal_u: np.ndarray,
        dt: float,
        excitation_magnitude: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize offline data collector.

        Args:
            plant: Physcial plant model
            nominal_x: Nominal state trajectory (N+1, n)
            nominal_u: Nominal input trajectory (N, m)
            dt: Time step
            excitation_magnitude: Magnitude of excitation signal
            seed: Random seed for reproducibility
        """
        self.plant = plant
        self.nominal_x = nominal_x
        self.nominal_u = nominal_u
        self.dt = dt
        self.excitation_magnitude = excitation_magnitude

        self.N = nominal_u.shape[0]  # number of time steps
        self.n = nominal_x.shape[1]  # state dimension
        self.m = nominal_u.shape[1]  # input dimension

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Storage for collected trajectories
        self.trajectories = []

    def sample_initial_states(
        self, n_samples: int, semi_axes: np.ndarray, center: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Sample initial states from an ellipsoid.

        Sample uniformly from ellipsoid: (x-c)^T diag(1/a^2) (x-c) <= 1

        Args:
            n_samples: Number of samples to draw
            semi_axes: Semi-axes of ellipsoid (n,)
            center: Center of ellipsoid (n,)  (default: nominal initial state)

        Returns:
            x0_samples: Sampled initial states (n_samples, n)
        """
        if center is None:
            center = self.nominal_x[0, :]

        # Sample from unit sphere
        samples = self.rng.normal(size=(n_samples, self.n))
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples = samples / norms

        # Sample radii uniformly from [0, 1] with appropriate distribution for uniform volume
        radii = self.rng.uniform(0, 1, size=(n_samples, 1)) ** (1 / self.n)

        # Scale by radii and semi-axes
        samples = samples * radii * semi_axes[np.newaxis, :]

        # Translate to center
        x0_samples = samples + center[np.newaxis, :]

        return x0_samples

    def generate_excitation_signal(self, length: int) -> np.ndarray:
        """
        Generate excitation signal for persistence of excitation.

        Uses uniform random excitation bounded by excitation_magnitude.

        Args:
            length: Length of excitation signal (number of time steps)

        Returns:
            excitation: Excitation signal (length, m)
        """
        excitation = self.rng.uniform(-self.excitation_magnitude, self.excitation_magnitude, size=(length, self.m))
        return excitation

    def collect_single_trajectory(
        self, x0: np.ndarray, excitation: Optional[np.ndarray] = None, verbose: bool = False
    ) -> Dict:
        """
        Collect a single trajectory from initial state x0.

        Args:
            x0: Initial state (n,)
            excitation: Excitation signal (N, m) - generated if None
            verbose: Print progress

        Returns:
            trajectory_data: Dictionary containing:
                - 'x': State trajectory (N+1, n)
                - 'u': Input trajectory (N, m)
                - 'eta': State deviation trajectory (N+1, n)
                - 'xi': Input deviation trajectory (N, m)
                - 'eta_next': Next state deviation trajectory (N, n)
                - 'x0': Initial state (n,)
                - 'excitation': Excitation signal (N, m)
        """
        if excitation is None:
            excitation = self.generate_excitation_signal(self.N)

        # Storage
        x_traj = np.zeros((self.N + 1, self.n))
        u_traj = np.zeros((self.N, self.m))
        eta_traj = np.zeros((self.N + 1, self.n))
        xi_traj = np.zeros((self.N, self.m))
        eta_next_traj = np.zeros((self.N, self.n))

        # Initial state
        x_traj[0, :] = x0
        eta_traj[0, :] = x0 - self.nominal_x[0, :]

        # Run trajectory
        for k in range(self.N):
            # Current state deviation
            eta_k = x_traj[k, :] - self.nominal_x[k, :]

            # Apply nominal input + excitation
            u_k = self.nominal_u[k, :] + excitation[k, :]
            u_traj[k, :] = u_k

            # Input deviation
            xi_k = u_k - self.nominal_u[k, :]
            xi_traj[k, :] = xi_k

            # Step forward
            x_next = self.plant.discrete_dynamics(x_traj[k, :], u_traj[k, :], self.dt, method="rk4")
            x_traj[k + 1, :] = x_next

            # Next state deviation
            eta_next = x_next - self.nominal_x[k + 1, :]
            eta_traj[k + 1, :] = eta_next
            eta_next_traj[k, :] = eta_next

            if verbose and k % 20 == 0:
                print(f"   Step {k}/{self.N} : ||eta||: {np.linalg.norm(eta_k):.4f}")

        # Store trajectory data
        trajectory_data = {
            "x": x_traj,
            "u": u_traj,
            "eta": eta_traj,
            "xi": xi_traj,
            "eta_next": eta_next_traj,
            "x0": x0,
            "excitation": excitation,
        }
        return trajectory_data

    def collect_dataset(
        self,
        n_samples: int,
        semi_axes: np.ndarray,
        center: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Collect full dataset from multiple initial states.

        Args:
            n_samples: Number of trajectories to collect
            semi_axes: Semi-axes of ellipsoid (n,)
            center: Center of ellipsoid (n,)  (default: nominal initial state)
            verbose: Print progress

        Returns:
            dataset: List of trajectory dictionaries
        """
        if verbose:
            print("=" * 70)
            print("OFFLINE DATA COLLECTION")
            print("=" * 70)
            print(f"Samples:      {n_samples}")
            print(f"Horizon:      {self.N} steps ({self.N * self.dt:.2f}s)")
            print(f"Excitation:   ±{self.excitation_magnitude}")
            print(f"Semi-axes:    {semi_axes}")
            print("=" * 70)

        # Sample initial states
        x0_samples = self.sample_initial_states(n_samples, semi_axes, center)

        # Collect trajectories
        dataset = []
        for i in range(n_samples):
            if verbose:
                print(f"\n[{i + 1}/{n_samples}] Collecting trajectory from x0: {x0_samples[i]}")

            # Collect trajectory
            traj_data = self.collect_single_trajectory(x0_samples[i], verbose=verbose)
            dataset.append(traj_data)

            if verbose:
                final_eta = traj_data["eta"][-1, :]
                print(f"  Final ||η||: {np.linalg.norm(final_eta):.4f}")

        self.trajectories = dataset

        if verbose:
            print("\n" + "=" * 70)
            print(f" Collected {len(dataset)} trajectories")
            print("=" * 70)

        return dataset

    def get_statistics(self) -> Dict:
        """
        Computes statistics from collected trajectories.

        Returns:
            stats: Dictionary containing:
        """
        if len(self.trajectories) == 0:
            raise ValueError("No trajectories collected yet")

        # Aggregate data
        all_eta = np.concatenate([traj["eta"] for traj in self.trajectories], axis=0)
        all_xi = np.concatenate([traj["xi"] for traj in self.trajectories], axis=0)

        stats = {
            "n_trajectories": len(self.trajectories),
            "n_timesteps": self.N,
            "n_datapoints": len(self.trajectories) * (self.N),
            "eta_mean": np.mean(all_eta, axis=0),
            "eta_std": np.std(all_eta, axis=0),
            "eta_max": np.max(np.linalg.norm(all_eta, axis=1)),
            "xi_mean": np.mean(all_xi, axis=0),
            "xi_std": np.std(all_xi, axis=0),
            "xi_max": np.max(np.linalg.norm(all_xi, axis=1)),
        }

        return stats

    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        print(f"Trajectories:    {stats['n_trajectories']}")
        print(f"Timesteps:       {stats['n_timesteps']}")
        print(f"Total datapoints: {stats['n_datapoints']}")
        print("\nState Deviations (η):")
        print(f"  Mean:  {stats['eta_mean']}")
        print(f"  Std:   {stats['eta_std']}")
        print(f"  Max:   {stats['eta_max']:.6f}")
        print("\nInput Deviations (ξ):")
        print(f"  Mean:  {stats['xi_mean']}")
        print(f"  Std:   {stats['xi_std']}")
        print(f"  Max:   {stats['xi_max']:.6f}")
        print("=" * 70)

    def save_dataset(self, filepath: str):
        """
        Save collected dataset to disk.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "trajectories": self.trajectories,
            "nominal_x": self.nominal_x,
            "nominal_u": self.nominal_u,
            "dt": self.dt,
            "N": self.N,
            "n": self.n,
            "m": self.m,
            "excitation_magnitude": self.excitation_magnitude,
            "statistics": self.get_statistics() if len(self.trajectories) > 0 else None,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"\n💾 Dataset saved to: {filepath}")
        print(f"   File size: {filepath.stat().st_size / 1024:.2f} KB")

    @staticmethod
    def load_dataset(filepath: str) -> Dict:
        """
        Load dataset from disk.

        Args:
            filepath: Path to saved dataset

        Returns:
            data: Loaded dataset dictionary
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        print(f"📂 Loaded dataset from: {filepath}")
        if data.get("statistics") is not None:
            print(f"   Trajectories: {data['statistics']['n_trajectories']}")
            print(f"   Datapoints:   {data['statistics']['n_datapoints']}")

        return data

    def __repr__(self) -> str:
        return (
            f"OfflineDataCollector(N={self.N}, n={self.n}, m={self.m}, "
            f"trajectories={len(self.trajectories)}, "
            f"excitation=±{self.excitation_magnitude})"
        )
