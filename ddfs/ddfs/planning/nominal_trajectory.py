# ddfs/ddfs/planning/nominal_trajectory.py

"""
NominalTrajectory Container for Phase 1: Nominal Planning

This module provides the container for nominal trajectory data computed
from the digital twin during Phase 1 planning.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp


@dataclass
class NominalTrajectory:
    """
    Container for nominal trajectory data from Phase 1 planning.

    A nominal trajectory represents a feasible solution on the digital twin
    from x0 to xf while avoiding obstacles.

    Attributes
    ----------
    x_nom : np.ndarray, shape (N+1, n)
        Nominal state trajectory, where:
        - N+1 is the number of timesteps (including t=0)
        - n is the state dimension
    u_nom : np.ndarray, shape (N, m)
        Nominal input trajectory, where:
        - N is the number of control intervals
        - m is the input dimension
    N : int
        Planning horizon (number of control intervals)
    dt : float
        Timestep duration (seconds)

    Notes
    -----
    - The trajectory is indexed from k=0 to k=N
    - State x_nom[k] is the state at time t=k*dt
    - Input u_nom[k] is the control applied from t=k*dt to t=(k+1)*dt
    - Final state x_nom[N] has no associated control
    """

    x_nom: jnp.ndarray  # (N+1, n) - State trajectory
    u_nom: jnp.ndarray  # (N, m) - Input trajectory
    N: int  # Planning horizon
    dt: float  # Timestep duration

    def __post_init__(self):
        """Validate dimensions after initialization."""
        assert self.x_nom.shape[0] == self.N + 1, f"x_nom must have N+1={self.N + 1} rows, got {self.x_nom.shape[0]}"
        assert self.u_nom.shape[0] == self.N, f"u_nom must have N={self.N} rows, got {self.u_nom.shape[0]}"

    @property
    def state_dim(self) -> int:
        """State dimension (n)."""
        return self.x_nom.shape[1]

    @property
    def input_dim(self) -> int:
        """Input dimension (m)."""
        return self.u_nom.shape[1]

    @property
    def tf(self) -> float:
        """Final time."""
        return self.N * self.dt

    def evaluate_at(self, k: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get (x_nom(k), u_nom(k)) at timestep k.

        Parameters
        ----------
        k : int
            Timestep index (0 <= k < N)

        Returns
        -------
        x_k : np.ndarray, shape (n,)
            State at timestep k
        u_k : np.ndarray, shape (m,)
            Control at timestep k

        Notes
        -----
        For k=N (final timestep), u_nom(N) is not defined.
        Use this method for k < N only.
        """
        assert 0 <= k < self.N, f"k={k} must be in [0, {self.N - 1}]"
        return self.x_nom[k], self.u_nom[k]

    def save(self, path: Path):
        """
        Save nominal trajectory to pickle file.

        Parameters
        ----------
        path : Path
            Path to save location (should end in .pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "NominalTrajectory":
        """
        Load nominal trajectory from pickle file.

        Parameters
        ----------
        path : Path
            Path to saved nominal trajectory

        Returns
        -------
        nominal : NominalTrajectory
            Loaded nominal trajectory object
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NominalTrajectory(N={self.N}, dt={self.dt:.4f}, state_dim={self.state_dim}, input_dim={self.input_dim})"
        )
