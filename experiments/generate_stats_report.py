"""
generate_stats_report.py

Produces results/stats_report.md + results/stats_report.json from
results/full_trial_results.csv:
  - per (controller, tier, condition_group) success rate + bootstrap 95% CI
  - pairwise Mann-Whitney U (vs A_baseline) + rank-biserial effect size,
    per tier, on clean-condition success (the audit's "FLC worse than
    baseline" claim, made checkable/reproducible rather than asserted)
  - NaN-safe throughout (Blueprint Sec 2.3/2.4 statistical-hygiene fix)

Run from the project root (after experiments/run_full_experiment.py):
    python experiments/generate_stats_report.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import FULL_TRIAL_CSV, RESULTS_DIR
from experiments.statistics_utils import bootstrap_ci, mannwhitney_with_effect_size


def condition_group(cond: str) -> str:
    return "clean" if cond == "clean" else "faulty"


def main() -> None:
    if not os.path.exists(FULL_TRIAL_CSV):
        raise FileNotFoundError(f"{FULL_TRIAL_CSV} not found -- run experiments/run_full_experiment.py first.")
    df = pd.read_csv(FULL_TRIAL_CSV)
    df["condition_group"] = df["condition"].apply(condition_group)

    controllers = sorted(df["controller"].unique())
    tiers = sorted(df["tier"].unique())

    summary_rows = []
    for c in controllers:
        for tier in tiers:
            for grp in ("clean", "faulty"):
                vals = df[(df.controller == c) & (df.tier == tier) &
                          (df.condition_group == grp)]["success"].astype(float).values
                mean, lo, hi = bootstrap_ci(vals)
                summary_rows.append({
                    "controller": c, "tier": tier, "condition_group": grp,
                    "n": len(vals), "success_rate": mean, "ci95_low": lo, "ci95_high": hi,
                })
    summary_df = pd.DataFrame(summary_rows)

    pairwise_rows = []
    baseline_name = "A_baseline"
    for c in controllers:
        if c == baseline_name:
            continue
        for tier in tiers:
            a = df[(df.controller == baseline_name) & (df.tier == tier) &
                   (df.condition_group == "clean")]["success"].astype(float).values
            b = df[(df.controller == c) & (df.tier == tier) &
                   (df.condition_group == "clean")]["success"].astype(float).values
            u, p, eff = mannwhitney_with_effect_size(a, b)
            pairwise_rows.append({
                "comparison": f"{baseline_name} vs {c}", "tier": tier, "condition": "clean",
                "n_baseline": len(a), "n_other": len(b),
                "U": u, "p_value": p, "rank_biserial_effect_size": eff,
                "nan_free": bool(not (np.isnan(u) or np.isnan(p))),
            })
    pairwise_df = pd.DataFrame(pairwise_rows)

    # Inverse-success-anomaly check: within each controller/tier, is faulty
    # success ever >= clean success? Mann-Whitney U + effect size, per
    # controller/tier, clean vs faulty -- this is the check that makes
    # "faulty sometimes outperforms clean" a reproducible, testable claim
    # instead of an eyeballed one.
    clean_vs_faulty_rows = []
    for c in controllers:
        for tier in tiers:
            clean = df[(df.controller == c) & (df.tier == tier) &
                       (df.condition_group == "clean")]["success"].astype(float).values
            faulty = df[(df.controller == c) & (df.tier == tier) &
                        (df.condition_group == "faulty")]["success"].astype(float).values
            u, p, eff = mannwhitney_with_effect_size(clean, faulty)
            clean_mean = clean.mean() if len(clean) else float("nan")
            faulty_mean = faulty.mean() if len(faulty) else float("nan")
            clean_vs_faulty_rows.append({
                "controller": c, "tier": tier,
                "n_clean": len(clean), "n_faulty": len(faulty),
                "clean_success": clean_mean, "faulty_success": faulty_mean,
                "faulty_outperforms_clean": bool(faulty_mean > clean_mean),
                "U": u, "p_value": p, "rank_biserial_effect_size": eff,
                "nan_free": bool(not (np.isnan(u) or np.isnan(p))),
            })
    clean_vs_faulty_df = pd.DataFrame(clean_vs_faulty_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "stats_summary.csv"), index=False)
    pairwise_df.to_csv(os.path.join(RESULTS_DIR, "stats_pairwise.csv"), index=False)
    clean_vs_faulty_df.to_csv(os.path.join(RESULTS_DIR, "stats_clean_vs_faulty.csv"), index=False)

    lines = ["# Statistics Report", "", "## Success rate (mean, 95% bootstrap CI)", "",
             summary_df.round(3).to_markdown(index=False), "",
             "## Pairwise Mann-Whitney U vs A_baseline (clean condition, per tier)", "",
             pairwise_df.round(4).to_markdown(index=False), "",
             "## Clean vs faulty, per controller/tier (inverse-success-anomaly check)", "",
             clean_vs_faulty_df.round(4).to_markdown(index=False), ""]
    report_path = os.path.join(RESULTS_DIR, "stats_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    n_nan = int((~pairwise_df["nan_free"]).sum()) + int((~clean_vs_faulty_df["nan_free"]).sum())
    n_anomalies = int(clean_vs_faulty_df["faulty_outperforms_clean"].sum())
    print(f"Wrote {report_path}")
    print(f"NaN cells across all pairwise stats: {n_nan}/{len(pairwise_df) + len(clean_vs_faulty_df)} "
          f"({'PASS -- statistical hygiene OK' if n_nan == 0 else 'FAIL -- investigate degenerate cells'})")
    print(f"Inverse-success anomalies (faulty success > clean success): {n_anomalies}/{len(clean_vs_faulty_df)} "
          f"controller/tier cells -- see 'p_value' column for whether each is statistically distinguishable.")
    print(summary_df.round(3).to_string(index=False))
    print()
    print(pairwise_df.round(4).to_string(index=False))
    print()
    print(clean_vs_faulty_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
