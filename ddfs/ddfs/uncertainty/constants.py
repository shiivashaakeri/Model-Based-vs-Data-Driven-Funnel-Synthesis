# ddfs/ddfs/uncertainty/constants.py

"""
Uncertainty Constants for Phase 3: Uncertainty Quantification

This module defines the container for all uncertainty constants used in
the data-driven funnel synthesis approach.

Based on:
- Assumption 1: Uniform mismatch bound gamma
- Lemma 1: Disturbance bound with L_r
- Lemma 2: Variation in linearization with L_J
- Assumption 3: Nominal increment bound v
- Equation 21: Per-segment uncertainty bounds β_i

Key constants:
- gamma: Plant-twin mismatch bound
- L_r: Linearization error Lipschitz constant (L_J/2)
- L_J: Jacobian Lipschitz constant
- v: Nominal increment bound
- C: Increment bound (C = L_J x v)
- β_i: Per-segment uncertainty bounds
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass
class UncertaintyConstants:
    """
    Container for all uncertainty constants.

    These constants characterize the uncertainty in the plant-twin system
    and are used for robust funnel synthesis.

    Attributes
    ----------
    gamma : float
        Plant-twin mismatch bound (Assumption 1).
        Satisfies: ||Δ(x, u)|| ≤ gamma for all (x, u) ∈ X x U
        where Δ(x, u) = f_plant(x, u) - f_twin(x, u)

    L_r : float
        Linearization error Lipschitz constant (Lemma 1).
        Used in disturbance bound: ||d(k)|| ≤ gamma + L_r ||[η; ξ]||²
        Typically L_r = L_J / 2

    L_J : float
        Jacobian Lipschitz constant (Lemma 2).
        Bounds variation in linearization:
        ||[A(k) - A(s), B(k) - B(s)]|| ≤ L_J x v x |k - s|

    v : float
        Nominal increment bound (Assumption 3).
        Satisfies: ||[x_nom(k+1); u_nom(k+1)] - [x_nom(k); u_nom(k)]|| ≤ v
        for all k = 0, ..., N-2

    C : float
        Increment bound (Lemma 2).
        Computed as C = L_J x v
        Used in β_i calculation

    beta_i : List[float]
        Per-segment uncertainty bounds (Equation 21).
        For segment i:
        β_i = Σ_{k ∈ T_i} (C|k-k_i| ||[η(k); ξ(k)]|| + gamma + L_r ||[η(k); ξ(k)]||²)²
        where T_i is the data collection window for segment i

    segment_indices : List[int]
        Segment indices corresponding to beta_i values

    gamma_per_timestep : Optional[np.ndarray]
        Plant-twin mismatch at each timestep along nominal (for diagnostics).
        Shape: (N,)

    L_r_samples : Optional[np.ndarray]
        Samples used to estimate L_r (for diagnostics).
        Shape: (n_samples,)

    L_J_samples : Optional[np.ndarray]
        Samples used to estimate L_J (for diagnostics).
        Shape: (n_samples,)

    computation_details : Dict
        Additional details about how constants were computed
    """

    # Core constants (required)
    gamma: float
    L_r: float
    L_J: float
    v: float
    C: float
    beta_i: List[float]
    segment_indices: List[int]

    # Optional diagnostic data
    gamma_per_timestep: Optional[np.ndarray] = None
    L_r_samples: Optional[np.ndarray] = None
    L_J_samples: Optional[np.ndarray] = None
    computation_details: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate constants."""
        # Check non-negativity
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")
        if self.L_r < 0:
            raise ValueError(f"L_r must be non-negative, got {self.L_r}")
        if self.L_J < 0:
            raise ValueError(f"L_J must be non-negative, got {self.L_J}")
        if self.v < 0:
            raise ValueError(f"v must be non-negative, got {self.v}")
        if self.C < 0:
            raise ValueError(f"C must be non-negative, got {self.C}")

        # Check consistency: C = L_J x v
        expected_C = self.L_J * self.v
        if abs(self.C - expected_C) > 1e-6:
            raise ValueError(f"C should equal L_J x v: C={self.C}, L_J x v={expected_C}")

        # Check beta_i dimensions
        if len(self.beta_i) != len(self.segment_indices):
            raise ValueError(
                f"beta_i ({len(self.beta_i)}) and segment_indices ({len(self.segment_indices)}) must have same length"
            )

        # Check beta_i non-negativity
        for i, beta in enumerate(self.beta_i):
            if beta < 0:
                raise ValueError(f"beta_i[{i}] must be non-negative, got {beta}")

    @property
    def num_segments(self) -> int:
        """Number of segments."""
        return len(self.beta_i)

    @property
    def beta_max(self) -> float:
        """Maximum beta value across all segments."""
        return max(self.beta_i) if self.beta_i else 0.0

    @property
    def beta_min(self) -> float:
        """Minimum beta value across all segments."""
        return min(self.beta_i) if self.beta_i else 0.0

    @property
    def beta_mean(self) -> float:
        """Mean beta value across all segments."""
        return float(np.mean(self.beta_i)) if self.beta_i else 0.0

    def get_beta(self, segment_idx: int) -> float:
        """
        Get β_i for a specific segment.

        Parameters
        ----------
        segment_idx : int
            Segment index

        Returns
        -------
        beta : float
            Uncertainty bound for this segment
        """
        try:
            idx = self.segment_indices.index(segment_idx)
            return self.beta_i[idx]
        except ValueError:
            raise ValueError(f"Segment {segment_idx} not found in segment_indices")

    def summary(self) -> str:
        """
        Generate human-readable summary of constants.

        Returns
        -------
        summary : str
            Summary string
        """
        lines = [
            "=" * 70,
            "UNCERTAINTY CONSTANTS",
            "=" * 70,
            "",
            "Core Constants:",
            f"  gamma:             {self.gamma:.6e}  [Plant-twin mismatch]",
            f"  L_r:               {self.L_r:.6e}  [Linearization error Lipschitz]",
            f"  L_J:               {self.L_J:.6e}  [Jacobian Lipschitz]",
            f"  v:                 {self.v:.6e}  [Nominal increment bound]",
            f"  C = L_J x v:       {self.C:.6e}  [Increment bound]",
            "",
            f"Per-Segment Bounds (β_i): {self.num_segments} segments",
        ]

        if self.num_segments > 0:
            beta_min = min(self.beta_i)
            beta_max = max(self.beta_i)
            beta_mean = np.mean(self.beta_i)

            lines.extend(
                [
                    f"  Min:     {beta_min:.6e}",
                    f"  Max:     {beta_max:.6e}",
                    f"  Mean:    {beta_mean:.6e}",
                ]
            )

            # Show first few
            n_show = min(5, self.num_segments)
            lines.append(f"  First {n_show}:")
            for i in range(n_show):
                seg_idx = self.segment_indices[i]
                beta = self.beta_i[i]
                lines.append(f"    Segment {seg_idx}: β_{seg_idx} = {beta:.6e}")

            if self.num_segments > n_show:
                lines.append(f"    ... ({self.num_segments - n_show} more)")

        # Add diagnostic info if available
        if self.gamma_per_timestep is not None:
            lines.extend(
                [
                    "",
                    "Diagnostics:",
                    f"  gamma_per_timestep: {self.gamma_per_timestep.shape}",
                    f"    Min: {np.min(self.gamma_per_timestep):.6e}",
                    f"    Max: {np.max(self.gamma_per_timestep):.6e} (used as gamma)",
                    f"    Mean: {np.mean(self.gamma_per_timestep):.6e}",
                ]
            )

        if self.L_r_samples is not None:
            lines.extend(
                [
                    f"  L_r_samples: {len(self.L_r_samples)} samples",
                    f"    Max: {np.max(self.L_r_samples):.6e} (used as L_r)",
                ]
            )

        if self.L_J_samples is not None:
            lines.extend(
                [
                    f"  L_J_samples: {len(self.L_J_samples)} samples",
                    f"    Max: {np.max(self.L_J_samples):.6e} (used as L_J)",
                ]
            )

        if self.computation_details:
            lines.extend(
                [
                    "",
                    "Computation Details:",
                ]
            )
            for key, value in self.computation_details.items():
                lines.append(f"  {key}: {value}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def validate_assumptions(self) -> bool:  # noqa: PLR0912
        """
        Validate that constants satisfy theoretical assumptions.

        Returns
        -------
        valid : bool
            True if all assumptions are satisfied
        """
        valid = True

        # Check Assumption 1: gamma ≥ 0
        if self.gamma < 0:
            print("✗ Assumption 1 violated: gamma must be non-negative")
            valid = False
        else:
            print("✓ Assumption 1: gamma ≥ 0")

        # Check Lemma 1: L_r ≥ 0
        if self.L_r < 0:
            print("✗ Lemma 1 violated: L_r must be non-negative")
            valid = False
        else:
            print("✓ Lemma 1: L_r ≥ 0")

        # Check Lemma 2: L_J ≥ 0
        if self.L_J < 0:
            print("✗ Lemma 2 violated: L_J must be non-negative")
            valid = False
        else:
            print("✓ Lemma 2: L_J ≥ 0")

        # Check Assumption 3: v ≥ 0
        if self.v < 0:
            print("✗ Assumption 3 violated: v must be non-negative")
            valid = False
        else:
            print("✓ Assumption 3: v ≥ 0")

        # Check C = L_J x v
        expected_C = self.L_J * self.v
        if abs(self.C - expected_C) > 1e-6:
            print(f"✗ C = L_J x v violated: C={self.C}, L_J x v={expected_C}")
            valid = False
        else:
            print("✓ C = L_J x v")

        # Check β_i ≥ 0
        all_beta_positive = all(beta >= 0 for beta in self.beta_i)
        if not all_beta_positive:
            print("✗ Some β_i are negative")
            valid = False
        else:
            print(f"✓ All β_i ≥ 0 ({len(self.beta_i)} segments)")

        # Check typical relation: L_r ≈ L_J / 2
        if abs(self.L_r - self.L_J / 2) > 1e-6:
            print(f"⚠ Note: L_r = {self.L_r:.6e}, L_J/2 = {self.L_J / 2:.6e}")
            print("  (L_r is typically set to L_J/2, but this is not required)")
        else:
            print("✓ L_r = L_J / 2 (typical choice)")

        return valid

    def save(self, path: Union[Path, str]):
        """
        Save constants to pickle file.

        Parameters
        ----------
        path : Path or str
            Path to save location (should end in .pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[Path, str]) -> "UncertaintyConstants":
        """
        Load constants from pickle file.

        Parameters
        ----------
        path : Path or str
            Path to saved constants

        Returns
        -------
        constants : UncertaintyConstants
            Loaded constants object
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UncertaintyConstants("
            f"gamma={self.gamma:.3e}, "
            f"L_r={self.L_r:.3e}, "
            f"L_J={self.L_J:.3e}, "
            f"v={self.v:.3e}, "
            f"C={self.C:.3e}, "
            f"segments={self.num_segments})"
        )
