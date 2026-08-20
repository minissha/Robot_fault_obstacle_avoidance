"""
statistics_utils.py

Blueprint v2 Section 5.3 (Strengthen Statistical Rigor):
  - Mann-Whitney U instead of / alongside independent t-tests on binary outcomes.
  - Effect size (rank-biserial correlation) alongside every p-value.
  - Bootstrap confidence intervals for bar charts.

All functions are NaN-safe for degenerate inputs (all-zero or all-one
groups), returning (nan, nan) with a printed note rather than crashing --
which is itself part of the Section 2.3/2.4 statistical-hygiene fix (the
original 15-seed runs produced NaN ANOVA/t-test cells silently).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def mannwhitney_with_effect_size(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Returns (U statistic, p-value, rank-biserial correlation effect size)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0 or (np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]):
        return float("nan"), float("nan"), float("nan")
    try:
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    n1, n2 = len(a), len(b)
    rank_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2)
    return float(u_stat), float(p_val), float(rank_biserial)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95,
                  seed: int = 0) -> Tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) via percentile bootstrap. NaN-safe."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1 or np.all(values == values[0]):
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(values.mean()), float(lo), float(hi)
