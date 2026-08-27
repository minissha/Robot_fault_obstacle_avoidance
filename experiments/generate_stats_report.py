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
from experiments.statistics_utils import (
    bootstrap_ci, load_canonical_trials, mannwhitney_with_effect_size,
)


def condition_group(cond: str) -> str:
    return "clean" if cond == "clean" else "faulty"


def main() -> None:
    # Go through the shared loader rather than reading the CSV directly, so
    # this can't end up reporting a different run than the figures do.
    df = load_canonical_trials(FULL_TRIAL_CSV)
    fingerprint = df.attrs["artifact_fingerprint"]
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

    # Say which run these numbers came from. Otherwise a stats table and a
    # figure from two different runs look equally trustworthy.
    lines = ["# Statistics Report", "",
             f"Generated from `{os.path.relpath(FULL_TRIAL_CSV, RESULTS_DIR)}` "
             f"at artifact fingerprint `{fingerprint}` "
             f"({len(df)} trials, {df['controller'].nunique()} controllers, "
             f"{df['seed_idx'].nunique()} seeds/cell).", "",
             "Every number below traces to that single result set; the loader "
             "refuses to run if the CSV mixes experiment runs.", "",
             "## Success rate (mean, 95% bootstrap CI)", "",
             summary_df.round(3).to_markdown(index=False), "",
             "## Pairwise Mann-Whitney U vs A_baseline (clean condition, per tier)", "",
             pairwise_df.round(4).to_markdown(index=False), "",
             "## Clean vs faulty, per controller/tier (inverse-success-anomaly check)", "",
             clean_vs_faulty_df.round(4).to_markdown(index=False), ""]
    report_path = os.path.join(RESULTS_DIR, "stats_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # Also dump everything as JSON so the write-up can pull numbers from one
    # place instead of anyone re-typing them out of the markdown tables.
    json_path = os.path.join(RESULTS_DIR, "stats_report.json")
    with open(json_path, "w") as f:
        json.dump({
            "artifact_fingerprint": fingerprint,
            "source_csv": os.path.basename(FULL_TRIAL_CSV),
            "n_trials": int(len(df)),
            "n_seeds_per_cell": int(df["seed_idx"].nunique()),
            "summary": summary_df.to_dict("records"),
            "pairwise_vs_baseline": pairwise_df.to_dict("records"),
            "clean_vs_faulty": clean_vs_faulty_df.to_dict("records"),
        }, f, indent=2, default=str)

    n_nan = int((~pairwise_df["nan_free"]).sum()) + int((~clean_vs_faulty_df["nan_free"]).sum())
    n_anomalies = int(clean_vs_faulty_df["faulty_outperforms_clean"].sum())
    print(f"Wrote {report_path}")
    print(f"Wrote {json_path}")
    print(f"Source: {FULL_TRIAL_CSV} @ fingerprint {fingerprint} ({len(df)} trials)")
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
