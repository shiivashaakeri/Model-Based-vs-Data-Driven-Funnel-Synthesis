"""
Informativity checking for DDFS.

This module verifies that collected data satisfies persistence of excitation conditions
required for data-driven control synthesis (Theorem 1 from the paper).

Key condition: rank([H_i; Ξ_i]) = n + m (data is informative)
"""

from typing import Dict, List, Tuple

import numpy as np


class InformativityChecker:
    """
    Check informativity (persistence of excitation) of collected data.

    For data-driven synthesis to work, the data must be sufficiently rich
    to capture the system dynamics.
    """

    def __init__(self, n: int, m: int, rank_threshold: float = 1e-10):
        """
        Initialize informativity checker.

        Args:
            n: State dimension
            m: Input dimension
            rank_threshold: Singular value threshold for rank computation
        """
        self.n = n
        self.m = m
        self.rank_threshold = rank_threshold

    def check_rank_condition(self, H: np.ndarray, Xi: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check rank condition: rank([H; Ξ]) = n + m.

        This is the fundamental informativity condition from Theorem 1.

        Args:
            H: Past state-input matrix (n+m, L)
            Xi: Past input deviation matrix (m, L)

        Returns:
            informative: True if data is informative
            info: Dictionary with rank information
        """
        # Stack H and Xi
        stacked = np.vstack([H, Xi])

        # Compute singular values
        singular_values = np.linalg.svd(stacked, compute_uv=False)

        # Compute rank
        rank = np.sum(singular_values > self.rank_threshold)

        # Required rank
        required_rank = self.n + self.m

        # Check condition
        informative = rank >= required_rank

        info = {
            "rank": int(rank),
            "required_rank": required_rank,
            "informative": informative,
            "singular_values": singular_values,
            "condition_number": float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > self.rank_threshold
            else np.inf,
            "min_singular_value": float(singular_values[-1]),
            "max_singular_value": float(singular_values[0]),
            "rank_deficiency": max(0, required_rank - rank),
        }

        return informative, info

    def check_segment_informativity(self, matrices: Dict[str, np.ndarray]) -> Tuple[bool, Dict]:
        """
        Check informativity for a single segment.

        Args:
            matrices: Dictionary with 'H' and 'Xi'

        Returns:
            informative: True if segment data is informative
            info: Dictionary with detailed information
        """
        H = matrices["H"]
        Xi = matrices["Xi"]

        informative, rank_info = self.check_rank_condition(H, Xi)

        # Additional checks
        n_samples = H.shape[1]
        min_samples = self.n + self.m

        info = {
            **rank_info,
            "n_samples": n_samples,
            "min_required_samples": min_samples,
            "sufficient_samples": n_samples >= min_samples,
            "sample_margin": n_samples - min_samples,
        }

        return informative, info

    def check_all_segments(self, all_matrices: List[Dict[str, np.ndarray]], verbose: bool = True) -> Dict:
        """
        Check informativity for all segments.

        Args:
            all_matrices: List of matrix dictionaries from HankelMatrixBuilder
            verbose: Print detailed results

        Returns:
            results: Dictionary with informativity results for all segments
        """
        n_segments = len(all_matrices)

        if verbose:
            print("=" * 70)
            print("INFORMATIVITY CHECKING")
            print("=" * 70)
            print(f"Number of segments: {n_segments}")
            print(f"Required rank:      {self.n + self.m}")
            print("=" * 70)

        all_informative = True
        segment_results = []

        for seg_idx, matrices in enumerate(all_matrices):
            informative, info = self.check_segment_informativity(matrices)

            segment_results.append({"segment_idx": seg_idx, "informative": informative, "info": info})

            if not informative:
                all_informative = False

            if verbose:
                status = "✓ PASS" if informative else "✗ FAIL"
                print(f"\nSegment {seg_idx}: {status}")
                print(f"  Rank:          {info['rank']} / {info['required_rank']}")
                print(f"  Samples:       {info['n_samples']} (min: {info['min_required_samples']})")
                print(f"  Condition:     {info['condition_number']:.2e}")
                print(f"  σ_min:         {info['min_singular_value']:.2e}")  # noqa: RUF001

                if not informative:
                    print(f"  ⚠️  Rank deficiency: {info['rank_deficiency']}")

        if verbose:
            print("\n" + "=" * 70)
            if all_informative:
                print("✓ ALL SEGMENTS ARE INFORMATIVE")
            else:
                failed = sum(1 for r in segment_results if not r["informative"])
                print(f"✗ {failed}/{n_segments} SEGMENTS FAILED INFORMATIVITY CHECK")
            print("=" * 70)

        results = {
            "all_informative": all_informative,
            "n_segments": n_segments,
            "n_passed": sum(1 for r in segment_results if r["informative"]),
            "n_failed": sum(1 for r in segment_results if not r["informative"]),
            "segment_results": segment_results,
        }

        return results

    def suggest_improvements(self, info: Dict) -> List[str]:
        """
        Suggest improvements if data is not informative.

        Args:
            info: Informativity info dictionary

        Returns:
            suggestions: List of improvement suggestions
        """
        suggestions = []

        if not info["informative"]:
            # Check rank deficiency
            if info["rank_deficiency"] > 0:
                suggestions.append(
                    f"Rank is deficient by {info['rank_deficiency']}. "
                    "Increase excitation magnitude or collect more trajectories."
                )

            # Check sample count
            if info["sample_margin"] < 10:
                suggestions.append(
                    f"Only {info['sample_margin']} samples above minimum. "
                    "Consider collecting more trajectories for robustness."
                )

            # Check conditioning
            if info["condition_number"] > 1e10:
                suggestions.append(
                    "Data matrix is poorly conditioned. Ensure excitation signal has sufficient diversity."
                )

            # Check minimum singular value
            if info["min_singular_value"] < 1e-8:
                suggestions.append(
                    "Minimum singular value is very small. Data may have redundant or nearly redundant samples."
                )

        return suggestions

    def compute_excitation_energy(self, Xi: np.ndarray) -> Dict:
        """
        Compute energy metrics of excitation signal.

        Args:
            Xi: Input deviation matrix (m, L)

        Returns:
            energy_metrics: Dictionary with excitation energy metrics
        """
        # Total energy
        total_energy = np.sum(Xi**2)

        # Per-input energy
        per_input_energy = np.sum(Xi**2, axis=1)

        # Temporal statistics
        col_norms = np.linalg.norm(Xi, axis=0)

        energy_metrics = {
            "total_energy": float(total_energy),
            "per_input_energy": per_input_energy,
            "mean_col_norm": float(np.mean(col_norms)),
            "std_col_norm": float(np.std(col_norms)),
            "max_col_norm": float(np.max(col_norms)),
            "min_col_norm": float(np.min(col_norms)),
        }

        return energy_metrics

    def analyze_data_diversity(self, H: np.ndarray) -> Dict:
        """
        Analyze diversity of collected data.

        Args:
            H: Past state-input matrix (n+m, L)

        Returns:
            diversity_metrics: Dictionary with diversity metrics
        """
        # Compute covariance
        H_centered = H - np.mean(H, axis=1, keepdims=True)
        cov = (H_centered @ H_centered.T) / (H.shape[1] - 1)

        # Eigenvalues of covariance (measure of data spread)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        diversity_metrics = {
            "covariance_eigenvalues": eigenvalues,
            "trace": float(np.trace(cov)),
            "determinant": float(np.linalg.det(cov)),
            "condition_number": float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 1e-15 else np.inf,
        }

        return diversity_metrics

    def generate_report(self, all_matrices: List[Dict[str, np.ndarray]]) -> str:
        """
        Generate comprehensive informativity report.

        Args:
            all_matrices: List of matrix dictionaries

        Returns:
            report: String report
        """
        results = self.check_all_segments(all_matrices, verbose=False)

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("INFORMATIVITY REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Total segments: {results['n_segments']}")
        report_lines.append(f"Passed:         {results['n_passed']}")
        report_lines.append(f"Failed:         {results['n_failed']}")
        report_lines.append("")

        if results["all_informative"]:
            report_lines.append("✓ ALL SEGMENTS ARE INFORMATIVE")
            report_lines.append("")
            report_lines.append("The collected data satisfies persistence of excitation.")
            report_lines.append("Data-driven funnel synthesis can proceed.")
        else:
            report_lines.append("✗ SOME SEGMENTS ARE NOT INFORMATIVE")
            report_lines.append("")
            report_lines.append("Failed segments:")
            for seg_result in results["segment_results"]:
                if not seg_result["informative"]:
                    seg_idx = seg_result["segment_idx"]
                    info = seg_result["info"]
                    report_lines.append(f"  Segment {seg_idx}:")
                    report_lines.append(f"    Rank: {info['rank']} / {info['required_rank']}")
                    report_lines.append(f"    Deficiency: {info['rank_deficiency']}")

                    suggestions = self.suggest_improvements(info)
                    if suggestions:
                        report_lines.append("    Suggestions:")
                        for suggestion in suggestions:
                            report_lines.append(f"      - {suggestion}")

        report_lines.append("=" * 70)

        return "\n".join(report_lines)

    def __repr__(self) -> str:
        return f"InformativityChecker(n={self.n}, m={self.m}, threshold={self.rank_threshold:.2e})"
