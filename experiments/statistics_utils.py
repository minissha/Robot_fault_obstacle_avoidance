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

import os
from typing import Optional, Tuple

import numpy as np
from scipy import stats

# These come back from the CSV as the strings "True"/"False", so they get
# converted explicitly. Letting pandas guess breaks in an annoying way if a
# single cell is ever blank.
BOOL_FIELDS: Tuple[str, ...] = ("success", "collision", "timeout", "stuck_oscillation")

TRIAL_KEY_FIELDS: Tuple[str, ...] = ("controller", "tier", "condition", "seed_idx")


def load_canonical_trials(csv_path: str, expected_rows: Optional[int] = None):
    """
    Read results/full_trial_results.csv, or complain loudly enough that you
    can't accidentally analyse a mess.

    The results file used to end up with more than one run's worth of rows
    in it, each scored against different models. The figures averaged them
    together, the stats script read only one, and the same number came out
    three different ways depending on where you looked.

    So if there's more than one run in the file this raises instead of
    quietly picking one. Picking one silently is how the problem happened in
    the first place, and the right answer is to re-run cleanly rather than
    guess which half to trust. Same for duplicated trials or a boolean
    column with junk in it.

    Pass expected_rows if you want it to also check the run finished.
    """
    import pandas as pd     # imported here so the rest of the module still
                            # works if pandas isn't installed

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found -- run experiments/run_full_experiment.py first."
        )

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"{csv_path} contains a header but no trials.")

    if "artifact_fingerprint" not in df.columns:
        raise ValueError(
            f"{csv_path} predates artifact fingerprinting and cannot be verified "
            f"as internally consistent. Delete it and re-run "
            f"experiments/run_full_experiment.py."
        )

    fingerprints = df["artifact_fingerprint"].value_counts()
    if len(fingerprints) > 1:
        detail = "\n".join(f"    {fp}: {n} rows" for fp, n in fingerprints.items())
        raise ValueError(
            f"{csv_path} mixes {len(fingerprints)} experiment runs:\n{detail}\n"
            f"Rows written under different fingerprints were scored against "
            f"different trained artifacts (fault detector / NSGA-II result / ANN) "
            f"and are NOT comparable. Delete the file and re-run "
            f"experiments/run_full_experiment.py for one clean result set."
        )

    dupes = df.duplicated(subset=list(TRIAL_KEY_FIELDS), keep=False)
    if bool(dupes.any()):
        sample = (df.loc[dupes, list(TRIAL_KEY_FIELDS)]
                    .drop_duplicates().head(3).to_dict("records"))
        raise ValueError(
            f"{csv_path} contains {int(dupes.sum())} duplicate trial rows "
            f"(same controller/tier/condition/seed_idx), e.g. {sample}. "
            f"Delete the file and re-run experiments/run_full_experiment.py."
        )

    for col in BOOL_FIELDS:
        if col in df.columns:
            df[col] = _to_bool(df[col])

    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"{csv_path} holds {len(df)} trials but {expected_rows} were expected "
            f"-- the run is incomplete or was interrupted. Re-run "
            f"experiments/run_full_experiment.py (it resumes)."
        )

    df.attrs["artifact_fingerprint"] = str(fingerprints.index[0])
    df.attrs["source_csv"] = csv_path
    return df


def _to_bool(series):
    """
    Turn a column of True/False (or 1/0) into actual booleans.

    Anything else raises. A blank cell would otherwise quietly become True
    and inflate a success rate, which is the sort of thing you'd never spot.
    """
    if series.dtype == bool:
        return series
    mapped = (series.astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": True, "false": False, "1": True, "0": False}))
    if bool(mapped.isna().any()):
        bad = series[mapped.isna()].unique()[:5]
        raise ValueError(
            f"Column '{series.name}' has {int(mapped.isna().sum())} value(s) that "
            f"are neither true nor false, e.g. {list(bad)!r}. The CSV is corrupt; "
            f"delete it and re-run experiments/run_full_experiment.py."
        )
    return mapped.astype(bool)


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
