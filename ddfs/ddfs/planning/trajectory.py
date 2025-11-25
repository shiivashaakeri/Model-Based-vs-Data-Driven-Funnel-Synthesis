"""
Trajectory Representation and Storage for DDFS.

This module provides data structures for storing and manipulating
nominal trajectories, including:
- State and input sequences
- Time indexing
- Interpolation (linear, cubic)
- Slicing and segmentation
- Serialization (save/load)
- Validation and analysis utilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ddfs.utils.io_utils import load_npz, save_npz
from ddfs.utils.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Trajectory Data Class
# =============================================================================


@dataclass
class Trajectory:
    """
    Container for state-input trajectories.

    Stores a discrete-time trajectory with states x(k), inputs u(k),
    and associated time information.

    Parameters
    ----------
    x : np.ndarray
        State trajectory of shape (N+1, n_states).
    u : np.ndarray
        Input trajectory of shape (N, n_inputs).
    dt : float
        Timestep between samples [s].
    t0 : float
        Initial time [s].
    metadata : dict, optional
        Additional metadata (e.g., solver info, system name).

    Properties
    ----------
    N : int
        Number of timesteps (length of input trajectory).
    n_states : int
        State dimension.
    n_inputs : int
        Input dimension.
    t : np.ndarray
        Time vector of shape (N+1,).
    duration : float
        Total trajectory duration [s].
    """

    x: np.ndarray
    u: np.ndarray
    dt: float
    t0: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trajectory data after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate trajectory dimensions and consistency."""
        if self.x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {self.x.shape}")
        if self.u.ndim != 2:
            raise ValueError(f"u must be 2D, got shape {self.u.shape}")

        N_x = self.x.shape[0] - 1  # Number of steps from state trajectory
        N_u = self.u.shape[0]  # Number of steps from input trajectory

        if N_x != N_u:
            raise ValueError(
                f"Inconsistent trajectory lengths: x has {self.x.shape[0]} points (N={N_x}), u has {N_u} points"
            )

        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def N(self) -> int:
        """Number of timesteps."""
        return self.u.shape[0]

    @property
    def n_states(self) -> int:
        """State dimension."""
        return self.x.shape[1]

    @property
    def n_inputs(self) -> int:
        """Input dimension."""
        return self.u.shape[1]

    @property
    def t(self) -> np.ndarray:
        """Time vector of shape (N+1,)."""
        return self.t0 + np.arange(self.N + 1) * self.dt

    @property
    def t_inputs(self) -> np.ndarray:
        """Time vector for inputs of shape (N,)."""
        return self.t0 + np.arange(self.N) * self.dt

    @property
    def duration(self) -> float:
        """Total trajectory duration [s]."""
        return self.N * self.dt

    @property
    def t_final(self) -> float:
        """Final time [s]."""
        return self.t0 + self.duration

    @property
    def x_init(self) -> np.ndarray:
        """Initial state."""
        return self.x[0].copy()

    @property
    def x_final(self) -> np.ndarray:
        """Final state."""
        return self.x[-1].copy()

    @property
    def u_init(self) -> np.ndarray:
        """Initial input."""
        return self.u[0].copy()

    @property
    def u_final(self) -> np.ndarray:
        """Final input."""
        return self.u[-1].copy()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (N, n_states, n_inputs)."""
        return (self.N, self.n_states, self.n_inputs)

    # =========================================================================
    # Indexing and Slicing
    # =========================================================================

    def get_state(self, k: int) -> np.ndarray:
        """
        Get state at timestep k.

        Parameters
        ----------
        k : int
            Timestep index (0 to N inclusive).

        Returns
        -------
        np.ndarray
            State x(k).
        """
        if k < 0 or k > self.N:
            raise IndexError(f"State index {k} out of range [0, {self.N}]")
        return self.x[k].copy()

    def get_input(self, k: int) -> np.ndarray:
        """
        Get input at timestep k.

        Parameters
        ----------
        k : int
            Timestep index (0 to N-1 inclusive).

        Returns
        -------
        np.ndarray
            Input u(k).
        """
        if k < 0 or k >= self.N:
            raise IndexError(f"Input index {k} out of range [0, {self.N - 1}]")
        return self.u[k].copy()

    def get_state_input(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state and input at timestep k.

        Parameters
        ----------
        k : int
            Timestep index (0 to N-1 inclusive).

        Returns
        -------
        x_k : np.ndarray
            State x(k).
        u_k : np.ndarray
            Input u(k).
        """
        return self.get_state(k), self.get_input(k)

    def get_time(self, k: int) -> float:
        """
        Get time at timestep k.

        Parameters
        ----------
        k : int
            Timestep index.

        Returns
        -------
        float
            Time t(k).
        """
        return self.t0 + k * self.dt

    def slice(self, k_start: int, k_end: int) -> "Trajectory":
        """
        Extract a sub-trajectory from k_start to k_end.

        Parameters
        ----------
        k_start : int
            Starting timestep (inclusive).
        k_end : int
            Ending timestep (exclusive for inputs, inclusive for states).

        Returns
        -------
        Trajectory
            Sub-trajectory.
        """
        if k_start < 0 or k_end > self.N or k_start >= k_end:
            raise ValueError(f"Invalid slice [{k_start}, {k_end}) for trajectory with N={self.N}")

        return Trajectory(
            x=self.x[k_start : k_end + 1].copy(),
            u=self.u[k_start:k_end].copy(),
            dt=self.dt,
            t0=self.get_time(k_start),
            metadata={**self.metadata, "sliced_from": (k_start, k_end)},
        )

    def __getitem__(self, key: Union[int, slice]) -> Union[Tuple[np.ndarray, np.ndarray], "Trajectory"]:
        """
        Index or slice the trajectory.

        If key is int: returns (x[k], u[k]) tuple
        If key is slice: returns sub-Trajectory
        """
        if isinstance(key, int):
            if key < 0:
                key = self.N + key
            return self.get_state_input(key)
        elif isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.N
            if start < 0:
                start = self.N + start
            if stop < 0:
                stop = self.N + stop
            return self.slice(start, stop)
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def __len__(self) -> int:
        """Return number of timesteps N."""
        return self.N

    def __iter__(self):
        """Iterate over (x_k, u_k) pairs."""
        for k in range(self.N):
            yield self.x[k], self.u[k]

    # =========================================================================
    # Interpolation
    # =========================================================================

    def interpolate_state(
        self,
        t_query: float,
        method: str = "linear",
    ) -> np.ndarray:
        """
        Interpolate state at arbitrary time.

        Parameters
        ----------
        t_query : float
            Query time [s].
        method : str
            Interpolation method: 'linear', 'cubic', or 'nearest'.

        Returns
        -------
        np.ndarray
            Interpolated state.
        """
        if t_query < self.t0 or t_query > self.t_final:
            raise ValueError(f"Query time {t_query} outside trajectory range [{self.t0}, {self.t_final}]")

        # Find bracketing indices
        k = int((t_query - self.t0) / self.dt)
        k = min(k, self.N - 1)  # Clamp to valid range

        if method == "nearest":
            # Nearest neighbor
            t_k = self.get_time(k)
            t_k1 = self.get_time(k + 1)
            if abs(t_query - t_k) <= abs(t_query - t_k1):
                return self.x[k].copy()
            else:
                return self.x[k + 1].copy()

        elif method == "linear":
            # Linear interpolation
            t_k = self.get_time(k)
            alpha = (t_query - t_k) / self.dt
            alpha = np.clip(alpha, 0.0, 1.0)
            return (1 - alpha) * self.x[k] + alpha * self.x[k + 1]

        elif method == "cubic":
            # Cubic spline interpolation (local)
            return self._cubic_interpolate_state(t_query)

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def interpolate_input(
        self,
        t_query: float,
        method: str = "zoh",
    ) -> np.ndarray:
        """
        Interpolate input at arbitrary time.

        Parameters
        ----------
        t_query : float
            Query time [s].
        method : str
            Interpolation method: 'zoh' (zero-order hold), 'linear', or 'nearest'.

        Returns
        -------
        np.ndarray
            Interpolated input.
        """
        if t_query < self.t0 or t_query > self.t_final:
            raise ValueError(f"Query time {t_query} outside trajectory range [{self.t0}, {self.t_final}]")

        # Find index
        k = int((t_query - self.t0) / self.dt)
        k = min(k, self.N - 1)  # Clamp to valid range

        if method in {"zoh", "nearest"}:
            # Zero-order hold: use input at start of interval
            return self.u[k].copy()

        elif method == "linear":
            # Linear interpolation between inputs
            if k >= self.N - 1:
                return self.u[-1].copy()

            t_k = self.get_time(k)
            alpha = (t_query - t_k) / self.dt
            alpha = np.clip(alpha, 0.0, 1.0)
            return (1 - alpha) * self.u[k] + alpha * self.u[k + 1]

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def _cubic_interpolate_state(self, t_query: float) -> np.ndarray:
        """Cubic spline interpolation for state."""
        from scipy.interpolate import CubicSpline  # noqa: PLC0415

        # Build spline (cached would be more efficient for repeated queries)
        t_vec = self.t
        spline = CubicSpline(t_vec, self.x, axis=0)
        return spline(t_query)

    def resample(self, dt_new: float) -> "Trajectory":
        """
        Resample trajectory to new timestep.

        Parameters
        ----------
        dt_new : float
            New timestep [s].

        Returns
        -------
        Trajectory
            Resampled trajectory.
        """
        if dt_new <= 0:
            raise ValueError(f"dt_new must be positive, got {dt_new}")

        # New time vector
        N_new = int(self.duration / dt_new)
        t_new = self.t0 + np.arange(N_new + 1) * dt_new

        # Interpolate states
        x_new = np.zeros((N_new + 1, self.n_states))
        for i, t_i in enumerate(t_new):
            t_i_clamped = min(t_i, self.t_final)
            x_new[i] = self.interpolate_state(t_i_clamped, method="linear")

        # Interpolate inputs
        u_new = np.zeros((N_new, self.n_inputs))
        t_u_new = self.t0 + np.arange(N_new) * dt_new
        for i, t_i in enumerate(t_u_new):
            u_new[i] = self.interpolate_input(t_i, method="zoh")

        return Trajectory(
            x=x_new,
            u=u_new,
            dt=dt_new,
            t0=self.t0,
            metadata={**self.metadata, "resampled_from_dt": self.dt},
        )

    # =========================================================================
    # Segmentation (for DDFS algorithm)
    # =========================================================================

    def get_segment(self, segment_idx: int, segment_length: int) -> "Trajectory":
        """
        Extract segment for DDFS algorithm.

        Parameters
        ----------
        segment_idx : int
            Segment index i.
        segment_length : int
            Segment length T.

        Returns
        -------
        Trajectory
            Segment trajectory.
        """
        k_start = segment_idx * segment_length
        k_end = min((segment_idx + 1) * segment_length, self.N)
        return self.slice(k_start, k_end)

    def get_segments(self, segment_length: int) -> List["Trajectory"]:
        """
        Split trajectory into segments.

        Parameters
        ----------
        segment_length : int
            Segment length T.

        Returns
        -------
        list
            List of segment trajectories.
        """
        n_segments = (self.N + segment_length - 1) // segment_length
        segments = []
        for i in range(n_segments):
            segments.append(self.get_segment(i, segment_length))
        return segments

    def get_segment_indices(self, segment_length: int) -> List[Tuple[int, int]]:
        """
        Get segment boundary indices.

        Parameters
        ----------
        segment_length : int
            Segment length T.

        Returns
        -------
        list
            List of (k_start, k_end) tuples.
        """
        indices = []
        k = 0
        while k < self.N:
            k_end = min(k + segment_length, self.N)
            indices.append((k, k_end))
            k = k_end
        return indices

    # =========================================================================
    # Analysis and Metrics
    # =========================================================================

    def compute_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute min/max bounds on states.

        Returns
        -------
        x_min : np.ndarray
            Minimum state values.
        x_max : np.ndarray
            Maximum state values.
        """
        return self.x.min(axis=0), self.x.max(axis=0)

    def compute_input_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute min/max bounds on inputs.

        Returns
        -------
        u_min : np.ndarray
            Minimum input values.
        u_max : np.ndarray
            Maximum input values.
        """
        return self.u.min(axis=0), self.u.max(axis=0)

    def compute_path_length(self, position_indices: Optional[List[int]] = None) -> float:
        """
        Compute total path length in position space.

        Parameters
        ----------
        position_indices : list, optional
            Indices of position states. Default: [0, 1] for 2D or [0, 1, 2] for 3D.

        Returns
        -------
        float
            Total path length.
        """
        if position_indices is None:
            position_indices = [0, 1] if self.n_states >= 2 else [0]
            if self.n_states >= 3:
                position_indices = [0, 1, 2]

        positions = self.x[:, position_indices]
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    def compute_total_variation(self, component: str = "input") -> float:
        """
        Compute total variation of trajectory.

        Parameters
        ----------
        component : str
            'state' or 'input'.

        Returns
        -------
        float
            Total variation.
        """
        if component == "state":
            diffs = np.diff(self.x, axis=0)
        elif component == "input":
            diffs = np.diff(self.u, axis=0)
        else:
            raise ValueError(f"component must be 'state' or 'input', got {component}")

        return np.sum(np.linalg.norm(diffs, axis=1))

    def compute_smoothness(self) -> Dict[str, float]:
        """
        Compute smoothness metrics.

        Returns
        -------
        dict
            Dictionary with smoothness metrics.
        """
        # State derivatives (finite difference)
        x_dot = np.diff(self.x, axis=0) / self.dt
        x_ddot = np.diff(x_dot, axis=0) / self.dt

        # Input derivatives
        u_dot = np.diff(self.u, axis=0) / self.dt

        return {
            "max_state_velocity": np.max(np.linalg.norm(x_dot, axis=1)),
            "max_state_acceleration": np.max(np.linalg.norm(x_ddot, axis=1)) if len(x_ddot) > 0 else 0.0,
            "max_input_rate": np.max(np.linalg.norm(u_dot, axis=1)) if len(u_dot) > 0 else 0.0,
            "state_variation": self.compute_total_variation("state"),
            "input_variation": self.compute_total_variation("input"),
        }

    def compute_nominal_increment_bound(self) -> float:
        """
        Compute bound v on nominal trajectory increments (Assumption 3).

        v = max_k ||(x_nom(k+1), u_nom(k+1)) - (x_nom(k), u_nom(k))||

        Returns
        -------
        float
            Maximum increment norm v.
        """
        max_increment = 0.0

        for k in range(self.N - 1):
            # State increment
            dx = self.x[k + 1] - self.x[k]
            # Input increment
            du = self.u[k + 1] - self.u[k]
            # Combined increment
            increment = np.concatenate([dx, du])
            increment_norm = np.linalg.norm(increment)
            max_increment = max(max_increment, increment_norm)

        return max_increment

    # =========================================================================
    # Serialization
    # =========================================================================

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save trajectory to file.

        Parameters
        ----------
        filepath : str or Path
            Output filepath. Extension determines format:
            - .npz: NumPy compressed archive
            - .npy: NumPy array (states only, not recommended)
        """
        filepath = Path(filepath)

        if filepath.suffix == ".npz":
            save_npz(
                filepath,
                x=self.x,
                u=self.u,
                dt=np.array([self.dt]),
                t0=np.array([self.t0]),
                metadata=self.metadata,
            )
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        logger.debug(f"Saved trajectory to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Trajectory":
        """
        Load trajectory from file.

        Parameters
        ----------
        filepath : str or Path
            Input filepath.

        Returns
        -------
        Trajectory
            Loaded trajectory.
        """
        filepath = Path(filepath)

        if filepath.suffix == ".npz":
            data = load_npz(filepath)
            return cls(
                x=data["x"],
                u=data["u"],
                dt=float(data["dt"][0]),
                t0=float(data.get("t0", [0.0])[0]),
                metadata=data.get("metadata", {}),
            )
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary."""
        return {
            "x": self.x.tolist(),
            "u": self.u.tolist(),
            "dt": self.dt,
            "t0": self.t0,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create trajectory from dictionary."""
        return cls(
            x=np.array(data["x"]),
            u=np.array(data["u"]),
            dt=data["dt"],
            t0=data.get("t0", 0.0),
            metadata=data.get("metadata", {}),
        )

    # =========================================================================
    # Modification
    # =========================================================================

    def append(self, other: "Trajectory") -> "Trajectory":
        """
        Append another trajectory to this one.

        Parameters
        ----------
        other : Trajectory
            Trajectory to append.

        Returns
        -------
        Trajectory
            Combined trajectory.
        """
        if self.n_states != other.n_states or self.n_inputs != other.n_inputs:
            raise ValueError("Trajectory dimensions must match")

        if abs(self.dt - other.dt) > 1e-10:
            raise ValueError(f"Timesteps must match: {self.dt} vs {other.dt}")

        # Combine (skip duplicate state at junction)
        x_combined = np.vstack([self.x, other.x[1:]])
        u_combined = np.vstack([self.u, other.u])

        return Trajectory(
            x=x_combined,
            u=u_combined,
            dt=self.dt,
            t0=self.t0,
            metadata={**self.metadata, "appended": True},
        )

    def with_offset(
        self,
        state_offset: np.ndarray = None,
        input_offset: np.ndarray = None,
    ) -> "Trajectory":
        """
        Create trajectory with constant offset added.

        Parameters
        ----------
        state_offset : np.ndarray, optional
            Offset to add to all states.
        input_offset : np.ndarray, optional
            Offset to add to all inputs.

        Returns
        -------
        Trajectory
            Offset trajectory.
        """
        x_new = self.x.copy()
        u_new = self.u.copy()

        if state_offset is not None:
            x_new += state_offset

        if input_offset is not None:
            u_new += input_offset

        return Trajectory(
            x=x_new,
            u=u_new,
            dt=self.dt,
            t0=self.t0,
            metadata={**self.metadata, "offset": True},
        )

    def copy(self) -> "Trajectory":
        """Create a deep copy of the trajectory."""
        return Trajectory(
            x=self.x.copy(),
            u=self.u.copy(),
            dt=self.dt,
            t0=self.t0,
            metadata=self.metadata.copy(),
        )

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"Trajectory(\n"
            f"  N={self.N},\n"
            f"  n_states={self.n_states},\n"
            f"  n_inputs={self.n_inputs},\n"
            f"  dt={self.dt},\n"
            f"  duration={self.duration:.2f}s,\n"
            f"  t=[{self.t0:.2f}, {self.t_final:.2f}]\n"
            f")"
        )

    def summary(self) -> str:
        """Generate detailed summary string."""
        x_min, x_max = self.compute_state_bounds()
        u_min, u_max = self.compute_input_bounds()

        lines = [
            "Trajectory Summary",
            "-" * 40,
            f"Steps:     N = {self.N}",
            f"States:    n = {self.n_states}",
            f"Inputs:    m = {self.n_inputs}",
            f"Timestep:  dt = {self.dt} s",
            f"Duration:  T = {self.duration:.2f} s",
            f"Time:      [{self.t0:.2f}, {self.t_final:.2f}] s",
            "",
            "State bounds:",
        ]

        for i in range(self.n_states):
            lines.append(f"  x[{i}]: [{x_min[i]:.4f}, {x_max[i]:.4f}]")

        lines.append("")
        lines.append("Input bounds:")

        for i in range(self.n_inputs):
            lines.append(f"  u[{i}]: [{u_min[i]:.4f}, {u_max[i]:.4f}]")

        return "\n".join(lines)


# =============================================================================
# Deviation Trajectory
# =============================================================================


@dataclass
class DeviationTrajectory:
    """
    Container for deviation trajectories (η, ξ) from nominal.

    Used in DDFS for tracking errors:
    - η(k) = x(k) - x_nom(k)  (state deviation)
    - ξ(k) = u(k) - u_nom(k)  (input deviation)

    Parameters
    ----------
    eta : np.ndarray
        State deviation trajectory of shape (N+1, n_states).
    xi : np.ndarray
        Input deviation trajectory of shape (N, n_inputs).
    nominal : Trajectory
        Reference nominal trajectory.
    """

    eta: np.ndarray
    xi: np.ndarray
    nominal: Trajectory

    def __post_init__(self):
        """Validate dimensions."""
        if self.eta.shape[0] != self.nominal.N + 1:
            raise ValueError(f"eta length {self.eta.shape[0]} doesn't match nominal N+1={self.nominal.N + 1}")
        if self.xi.shape[0] != self.nominal.N:
            raise ValueError(f"xi length {self.xi.shape[0]} doesn't match nominal N={self.nominal.N}")

    @property
    def N(self) -> int:
        return self.nominal.N

    @property
    def dt(self) -> float:
        return self.nominal.dt

    def get_absolute_trajectory(self) -> Trajectory:
        """
        Convert deviation to absolute trajectory.

        Returns
        -------
        Trajectory
            Absolute trajectory x = x_nom + η, u = u_nom + ξ.
        """
        return Trajectory(
            x=self.nominal.x + self.eta,
            u=self.nominal.u + self.xi,
            dt=self.nominal.dt,
            t0=self.nominal.t0,
            metadata={"from_deviation": True},
        )

    @classmethod
    def from_trajectories(
        cls,
        actual: Trajectory,
        nominal: Trajectory,
    ) -> "DeviationTrajectory":
        """
        Create deviation trajectory from actual and nominal.

        Parameters
        ----------
        actual : Trajectory
            Actual trajectory.
        nominal : Trajectory
            Nominal trajectory.

        Returns
        -------
        DeviationTrajectory
            Deviation trajectory.
        """
        if actual.N != nominal.N:
            raise ValueError(f"Trajectory lengths must match: {actual.N} vs {nominal.N}")

        return cls(
            eta=actual.x - nominal.x,
            xi=actual.u - nominal.u,
            nominal=nominal,
        )

    def max_state_deviation(self) -> float:
        """Maximum state deviation norm."""
        return np.max(np.linalg.norm(self.eta, axis=1))

    def max_input_deviation(self) -> float:
        """Maximum input deviation norm."""
        return np.max(np.linalg.norm(self.xi, axis=1))

    def rms_state_deviation(self) -> float:
        """RMS state deviation."""
        return np.sqrt(np.mean(np.sum(self.eta**2, axis=1)))

    def rms_input_deviation(self) -> float:
        """RMS input deviation."""
        return np.sqrt(np.mean(np.sum(self.xi**2, axis=1)))


# =============================================================================
# Factory Functions
# =============================================================================


def create_trajectory(
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    t0: float = 0.0,
    **metadata,
) -> Trajectory:
    """
    Create trajectory with metadata.

    Parameters
    ----------
    x : np.ndarray
        State trajectory.
    u : np.ndarray
        Input trajectory.
    dt : float
        Timestep.
    t0 : float
        Initial time.
    **metadata
        Additional metadata.

    Returns
    -------
    Trajectory
        Created trajectory.
    """
    return Trajectory(x=x, u=u, dt=dt, t0=t0, metadata=metadata)


def create_constant_trajectory(
    x_const: np.ndarray,
    u_const: np.ndarray,
    N: int,
    dt: float,
    t0: float = 0.0,
) -> Trajectory:
    """
    Create trajectory with constant state and input.

    Parameters
    ----------
    x_const : np.ndarray
        Constant state.
    u_const : np.ndarray
        Constant input.
    N : int
        Number of timesteps.
    dt : float
        Timestep.
    t0 : float
        Initial time.

    Returns
    -------
    Trajectory
        Constant trajectory.
    """
    x = np.tile(x_const, (N + 1, 1))
    u = np.tile(u_const, (N, 1))
    return Trajectory(x=x, u=u, dt=dt, t0=t0, metadata={"type": "constant"})


def create_linear_trajectory(
    x_init: np.ndarray,
    x_final: np.ndarray,
    u_const: np.ndarray,
    N: int,
    dt: float,
    t0: float = 0.0,
) -> Trajectory:
    """
    Create trajectory with linear state interpolation.

    Parameters
    ----------
    x_init : np.ndarray
        Initial state.
    x_final : np.ndarray
        Final state.
    u_const : np.ndarray
        Constant input.
    N : int
        Number of timesteps.
    dt : float
        Timestep.
    t0 : float
        Initial time.

    Returns
    -------
    Trajectory
        Linear trajectory.
    """
    alpha = np.linspace(0, 1, N + 1).reshape(-1, 1)
    x = (1 - alpha) * x_init + alpha * x_final
    u = np.tile(u_const, (N, 1))
    return Trajectory(x=x, u=u, dt=dt, t0=t0, metadata={"type": "linear"})


def concatenate_trajectories(trajectories: List[Trajectory]) -> Trajectory:
    """
    Concatenate multiple trajectories.

    Parameters
    ----------
    trajectories : list
        List of trajectories to concatenate.

    Returns
    -------
    Trajectory
        Concatenated trajectory.
    """
    if not trajectories:
        raise ValueError("Cannot concatenate empty list")

    result = trajectories[0].copy()
    for traj in trajectories[1:]:
        result = result.append(traj)

    return result
