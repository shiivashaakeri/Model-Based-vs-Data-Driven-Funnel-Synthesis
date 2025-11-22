# ddfs/ddfs/planning/nominal_trajectory.py

"""
Nominal trajectory data structure.

This module provides the NominalTrajectory dataclass for representing
planned trajectories from Phase 1 planning.

A nominal trajectory is a feasible solution computed by the digital twin
that goes from initial state x0 to goal state xf while avoiding obstacles.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np


@dataclass
class NominalTrajectory:
    """
    Container for nominal trajectory data from Phase 1 planning.

    A nominal trajectory represents a feasible solution on the digital twin
    from x0 to xf while avoiding obstacles.

    Attributes
    ----------
    x_nom : np.ndarray
        Nominal state trajectory, shape (N+1, n)
        - N+1 is the number of timesteps (including t=0)
        - n is the state dimension
    u_nom : np.ndarray
        Nominal input trajectory, shape (N, m)
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

    Examples
    --------
    >>> import numpy as np
    >>> from ddfs.planning import NominalTrajectory
    >>>
    >>> # Create a simple trajectory
    >>> N = 10
    >>> n, m = 3, 2
    >>> x_nom = np.random.randn(N+1, n)
    >>> u_nom = np.random.randn(N, m)
    >>>
    >>> traj = NominalTrajectory(x_nom=x_nom, u_nom=u_nom, N=N, dt=0.1)
    >>> print(traj.state_dim, traj.input_dim)
    3 2
    >>> print(traj.tf)
    1.0
    """

    x_nom: np.ndarray  # (N+1, n) - State trajectory
    u_nom: np.ndarray  # (N, m) - Input trajectory
    N: int  # Planning horizon
    dt: float  # Timestep duration

    def __post_init__(self):
        """Validate dimensions after initialization."""
        if self.x_nom.shape[0] != self.N + 1:
            raise ValueError(f"x_nom must have N+1={self.N + 1} rows, got {self.x_nom.shape[0]}")
        if self.u_nom.shape[0] != self.N:
            raise ValueError(f"u_nom must have N={self.N} rows, got {self.u_nom.shape[0]}")

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

    def evaluate_at(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get (x_nom(k), u_nom(k)) at timestep k.

        Parameters
        ----------
        k : int
            Timestep index (0 <= k < N)

        Returns
        -------
        x_k : np.ndarray
            State at timestep k, shape (n,)
        u_k : np.ndarray
            Control at timestep k, shape (m,)

        Raises
        ------
        ValueError
            If k is out of range

        Notes
        -----
        For k=N (final timestep), u_nom(N) is not defined.
        Use this method for k < N only.

        Examples
        --------
        >>> traj = NominalTrajectory(x_nom, u_nom, N=10, dt=0.1)
        >>> x_5, u_5 = traj.evaluate_at(5)
        """
        if not 0 <= k < self.N:
            raise ValueError(f"k={k} must be in [0, {self.N - 1}]")
        return self.x_nom[k], self.u_nom[k]

    def get_time_vector(self) -> np.ndarray:
        """
        Get time vector for the trajectory.

        Returns
        -------
        t : np.ndarray
            Time vector [0, dt, 2*dt, ..., N*dt], shape (N+1,)

        Examples
        --------
        >>> traj = NominalTrajectory(x_nom, u_nom, N=10, dt=0.1)
        >>> t = traj.get_time_vector()
        >>> print(t)
        [0.  0.1 0.2 ... 1.0]
        """
        return np.linspace(0, self.tf, self.N + 1)

    def save(self, path: Union[Path, str]):
        """
        Save nominal trajectory to pickle file.

        Parameters
        ----------
        path : Path or str
            Path to save location (should end in .pkl)

        Examples
        --------
        >>> traj.save('results/unicycle/nominal_trajectory.pkl')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[Path, str]) -> "NominalTrajectory":
        """
        Load nominal trajectory from pickle file.

        Parameters
        ----------
        path : Path or str
            Path to saved nominal trajectory

        Returns
        -------
        nominal : NominalTrajectory
            Loaded nominal trajectory object

        Examples
        --------
        >>> traj = NominalTrajectory.load('results/unicycle/nominal_trajectory.pkl')
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NominalTrajectory(N={self.N}, dt={self.dt:.4f}, state_dim={self.state_dim}, input_dim={self.input_dim})"
        )
