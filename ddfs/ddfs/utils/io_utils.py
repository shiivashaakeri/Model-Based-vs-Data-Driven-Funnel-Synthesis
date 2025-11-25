"""
File I/O Utilities for DDFS.

This module provides utilities for:
- Saving and loading data in various formats (NPZ, JSON)
- Directory management
- Experiment output organization
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# =============================================================================
# Directory Management
# =============================================================================


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    Path
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns
    -------
    Path
        Path to project root.
    """
    # Navigate up from this file to find project root
    current = Path(__file__).resolve()

    # Look for markers of project root
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback to current working directory
    return Path.cwd()


def get_results_dir(system_name: Optional[str] = None) -> Path:
    """
    Get results directory, optionally for a specific system.

    Parameters
    ----------
    system_name : str, optional
        Name of system subdirectory.

    Returns
    -------
    Path
        Path to results directory.
    """
    results = get_project_root() / "results"
    if system_name:
        results = results / system_name
    return ensure_dir(results)


def create_experiment_dir(
    base_dir: Union[str, Path],
    experiment_name: str,
    timestamp: bool = True,
) -> Path:
    """
    Create a directory for experiment outputs.

    Parameters
    ----------
    base_dir : str or Path
        Base directory for experiments.
    experiment_name : str
        Name of the experiment.
    timestamp : bool, optional
        Whether to append timestamp to directory name.

    Returns
    -------
    Path
        Path to created experiment directory.
    """
    base_dir = Path(base_dir)

    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{experiment_name}_{ts}"
    else:
        dir_name = experiment_name

    experiment_dir = base_dir / dir_name
    return ensure_dir(experiment_dir)


# =============================================================================
# NPZ File Operations
# =============================================================================


def save_npz(
    filepath: Union[str, Path],
    compressed: bool = True,
    **arrays: np.ndarray,
) -> None:
    """
    Save numpy arrays to NPZ file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    compressed : bool, optional
        Whether to use compression.
    **arrays : np.ndarray
        Named arrays to save.

    Example
    -------
    >>> save_npz("data.npz", x=x_array, u=u_array, t=t_array)
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if compressed:
        np.savez_compressed(filepath, **arrays)
    else:
        np.savez(filepath, **arrays)


def load_npz(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load numpy arrays from NPZ file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict
        Dictionary of named arrays.

    Example
    -------
    >>> data = load_npz("data.npz")
    >>> x = data["x"]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with np.load(filepath, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_trajectory(
    filepath: Union[str, Path],
    x: np.ndarray,
    u: np.ndarray,
    t: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save trajectory data to NPZ file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    x : np.ndarray
        State trajectory, shape (N+1, n_states).
    u : np.ndarray
        Input trajectory, shape (N, n_inputs).
    t : np.ndarray, optional
        Time array, shape (N+1,).
    metadata : dict, optional
        Additional metadata to save.
    """
    data = {"x": x, "u": u}

    if t is not None:
        data["t"] = t

    if metadata is not None:
        # Convert metadata to array for NPZ storage
        data["metadata"] = np.array([metadata], dtype=object)

    save_npz(filepath, **data)


def load_trajectory(
    filepath: Union[str, Path],
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Load trajectory data from NPZ file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict
        Dictionary with keys 'x', 'u', optionally 't' and 'metadata'.
    """
    data = load_npz(filepath)

    result = {
        "x": data["x"],
        "u": data["u"],
    }

    if "t" in data:
        result["t"] = data["t"]

    if "metadata" in data:
        result["metadata"] = data["metadata"][0]

    return result


# =============================================================================
# JSON File Operations
# =============================================================================


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(
    filepath: Union[str, Path],
    data: Dict[str, Any],
    indent: int = 2,
) -> None:
    """
    Save dictionary to JSON file.

    Handles numpy types automatically.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    data : dict
        Data to save.
    indent : int, optional
        JSON indentation level.
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=indent)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict
        Loaded data.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)


def save_results_summary(
    filepath: Union[str, Path],
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    notes: Optional[str] = None,
) -> None:
    """
    Save experiment results summary to JSON.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    config : dict
        Configuration used for experiment.
    metrics : dict
        Computed metrics and results.
    notes : str, optional
        Additional notes about the experiment.
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
    }

    if notes:
        summary["notes"] = notes

    save_json(filepath, summary)


# =============================================================================
# Funnel Data I/O
# =============================================================================


def save_funnel_data(
    filepath: Union[str, Path],
    P_matrices: List[np.ndarray],
    K_matrices: List[np.ndarray],
    segment_indices: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save funnel synthesis results.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    P_matrices : list of np.ndarray
        List of Lyapunov matrices P_i.
    K_matrices : list of np.ndarray
        List of feedback gain matrices K_i.
    segment_indices : np.ndarray
        Segment start indices.
    metadata : dict, optional
        Additional metadata.
    """
    # Stack matrices for storage
    P_stack = np.stack(P_matrices, axis=0)
    K_stack = np.stack(K_matrices, axis=0)

    data = {
        "P_matrices": P_stack,
        "K_matrices": K_stack,
        "segment_indices": segment_indices,
    }

    if metadata is not None:
        data["metadata"] = np.array([metadata], dtype=object)

    save_npz(filepath, **data)


def load_funnel_data(
    filepath: Union[str, Path],
) -> Dict[str, Union[List[np.ndarray], np.ndarray, Dict]]:
    """
    Load funnel synthesis results.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict
        Dictionary with P_matrices, K_matrices, segment_indices, metadata.
    """
    data = load_npz(filepath)

    # Unstack matrices
    P_matrices = [data["P_matrices"][i] for i in range(data["P_matrices"].shape[0])]
    K_matrices = [data["K_matrices"][i] for i in range(data["K_matrices"].shape[0])]

    result = {
        "P_matrices": P_matrices,
        "K_matrices": K_matrices,
        "segment_indices": data["segment_indices"],
    }

    if "metadata" in data:
        result["metadata"] = data["metadata"][0]

    return result


# =============================================================================
# Data Collection I/O
# =============================================================================


def save_data_matrices(
    filepath: Union[str, Path],
    H: np.ndarray,
    H_plus: np.ndarray,
    Xi: np.ndarray,
    segment_index: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save data matrices from a segment.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    H : np.ndarray
        State deviation matrix H_i.
    H_plus : np.ndarray
        Next state deviation matrix H_i^+.
    Xi : np.ndarray
        Input deviation matrix Ξ_i.
    segment_index : int
        Index of the segment.
    metadata : dict, optional
        Additional metadata.
    """
    data = {
        "H": H,
        "H_plus": H_plus,
        "Xi": Xi,
        "segment_index": np.array([segment_index]),
    }

    if metadata is not None:
        data["metadata"] = np.array([metadata], dtype=object)

    save_npz(filepath, **data)


def load_data_matrices(
    filepath: Union[str, Path],
) -> Dict[str, Union[np.ndarray, int, Dict]]:
    """
    Load data matrices from file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict
        Dictionary with H, H_plus, Xi, segment_index, metadata.
    """
    data = load_npz(filepath)

    result = {
        "H": data["H"],
        "H_plus": data["H_plus"],
        "Xi": data["Xi"],
        "segment_index": int(data["segment_index"][0]),
    }

    if "metadata" in data:
        result["metadata"] = data["metadata"][0]

    return result


# =============================================================================
# File Listing and Discovery
# =============================================================================


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """
    List files in directory matching pattern.

    Parameters
    ----------
    directory : str or Path
        Directory to search.
    pattern : str, optional
        Glob pattern (e.g., "*.npz").
    recursive : bool, optional
        Whether to search recursively.

    Returns
    -------
    list of Path
        List of matching file paths.
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    if recursive:
        return sorted(directory.rglob(pattern))
    else:
        return sorted(directory.glob(pattern))


def get_latest_file(
    directory: Union[str, Path],
    pattern: str = "*",
) -> Optional[Path]:
    """
    Get the most recently modified file matching pattern.

    Parameters
    ----------
    directory : str or Path
        Directory to search.
    pattern : str, optional
        Glob pattern.

    Returns
    -------
    Path or None
        Path to most recent file, or None if no files found.
    """
    files = list_files(directory, pattern)

    if not files:
        return None

    return max(files, key=lambda f: f.stat().st_mtime)
