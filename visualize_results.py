"""
visualize_results.py

Blueprint v2 Section 3 (Visualization Requirements) + Section 6.5 (new
figures). Generates every figure that current data supports into
results/figures/*.png. Each figure is skipped (with a printed reason,
not a fabricated placeholder) if its required input file doesn't exist
yet -- run experiments/run_full_experiment.py and/or
experiments/train_fault_detector.py first for full coverage.

Run from the project root:
    python visualize_results.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    FAULT_AWARE_SUFFIX, FIGURES_DIR, FULL_TRIAL_CSV, NSGA2_PATH, RESULTS_DIR,
)
from experiments.statistics_utils import bootstrap_ci

try:
    import pandas as pd
except ImportError:
    pd = None


def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def load_trials():
    if pd is None:
        print("[skip all trial-based figures] pandas not installed.")
        return None
    if not os.path.exists(FULL_TRIAL_CSV):
        print(f"[skip all trial-based figures] {FULL_TRIAL_CSV} not found -- "
              f"run experiments/run_full_experiment.py first.")
        return None
    return pd.read_csv(FULL_TRIAL_CSV)


def fig1_2_clean_vs_faulty(df) -> None:
    print("Figure 1/2: clean vs faulty success + performance drop")
    sub = df[df["condition"].isin(["clean"]) | df["condition"].str.startswith("fault_")]
    sub = sub.copy()
    sub["faulty"] = sub["condition"] != "clean"

    controllers = sorted(sub["controller"].unique())
    tiers = sorted(sub["tier"].unique())

    fig, axes = plt.subplots(1, len(tiers), figsize=(6 * len(tiers), 4), squeeze=False)
    drop_records = []
    for ti, tier in enumerate(tiers):
        ax = axes[0][ti]
        x = np.arange(len(controllers))
        width = 0.35
        for fi, faulty in enumerate([False, True]):
            means, los, his = [], [], []
            for c in controllers:
                vals = sub[(sub.controller == c) & (sub.tier == tier) & (sub.faulty == faulty)]["success"].values
                m, lo, hi = bootstrap_ci(vals.astype(float))
                means.append(m); los.append(m - lo); his.append(hi - m)
            ax.bar(x + (fi - 0.5) * width, means, width,
                   yerr=[los, his], capsize=3, label="faulty" if faulty else "clean")
        ax.set_xticks(x); ax.set_xticklabels(controllers, rotation=30, ha="right")
        ax.set_ylabel("Success rate")
        ax.set_title(f"Fig 1: Success rate, clean vs faulty ({tier})")
        ax.legend()
    _save(fig, "fig1_success_clean_vs_faulty.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for tier in tiers:
        drops = []
        for c in controllers:
            clean = sub[(sub.controller == c) & (sub.tier == tier) & (~sub.faulty)]["success"].mean()
            faulty = sub[(sub.controller == c) & (sub.tier == tier) & (sub.faulty)]["success"].mean()
            drops.append((clean or 0) - (faulty or 0))
        ax.plot(controllers, drops, marker="o", label=tier)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Performance drop (clean success - faulty success)")
    ax.set_title("Fig 2: Performance drop per controller/tier")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    _save(fig, "fig2_performance_drop.png")


def fig3_severity_curve(df) -> None:
    print("Figure 3: noise-spike severity degradation curve")
    sev_rows = df[df["condition"].str.startswith("fault_noise_spike_sev")]
    if sev_rows.empty:
        print("  [skip] no severity-sweep rows found.")
        return
    sev_rows = sev_rows.copy()
    sev_rows["severity"] = sev_rows["condition"].str.extract(r"sev(\d+)").astype(int)

    fig, ax = plt.subplots(figsize=(7, 4))
    for c in sorted(sev_rows["controller"].unique()):
        means = []
        sevs = sorted(sev_rows["severity"].unique())
        for s in sevs:
            vals = sev_rows[(sev_rows.controller == c) & (sev_rows.severity == s)]["success"]
            means.append(vals.mean())
        ax.plot(sevs, means, marker="o", label=c)
    ax.set_xlabel("Noise-spike severity level"); ax.set_ylabel("Success rate")
    ax.set_title("Fig 3: Degradation curve vs fault severity")
    ax.legend()
    _save(fig, "fig3_severity_degradation_curve.png")


def fig4_per_fault_breakdown(df) -> None:
    print("Figure 4: per-fault-type breakdown")
    sub = df[df["fault_type"].notna() & (df["fault_type"] != "")]
    if sub.empty:
        print("  [skip] no fault-typed rows found.")
        return
    fault_types = sorted(sub["fault_type"].unique())
    controllers = sorted(sub["controller"].unique())
    x = np.arange(len(fault_types))
    width = 0.8 / max(1, len(controllers))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, c in enumerate(controllers):
        means = [sub[(sub.controller == c) & (sub.fault_type == ft)]["success"].mean() for ft in fault_types]
        ax.bar(x + i * width, means, width, label=c)
    ax.set_xticks(x + width * (len(controllers) - 1) / 2)
    ax.set_xticklabels(fault_types)
    ax.set_ylabel("Success rate"); ax.set_title("Fig 4: Success rate by fault type")
    ax.legend(fontsize=8)
    _save(fig, "fig4_per_fault_type_breakdown.png")


def fig8_failure_mode_taxonomy(df) -> None:
    print("Figure 8: failure-mode taxonomy")
    controllers = sorted(df["controller"].unique())
    modes = ["success", "collision", "timeout", "stuck_oscillation"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(controllers))
    for mode in modes:
        col = "success" if mode == "success" else mode
        counts = [df[(df.controller == c)][col].sum() for c in controllers]
        totals = [len(df[df.controller == c]) for c in controllers]
        fracs = np.array(counts) / np.maximum(1, np.array(totals))
        ax.bar(controllers, fracs, bottom=bottom, label=mode)
        bottom += fracs
    ax.set_ylabel("Fraction of trials"); ax.set_title("Fig 8: Failure-mode taxonomy")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    _save(fig, "fig8_failure_mode_taxonomy.png")


def fig9_distributions(df) -> None:
    print("Figure 9: path length / clearance distributions")
    controllers = sorted(df["controller"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in ((axes[0], "path_length", "Path length"),
                            (axes[1], "min_clearance", "Min clearance")):
        data_clean = [df[(df.controller == c) & (df.condition == "clean")][col].dropna().values
                      for c in controllers]
        ax.boxplot(data_clean, tick_labels=controllers)
        ax.set_title(f"Fig 9: {title} (clean)"); ax.tick_params(axis="x", rotation=30)
    _save(fig, "fig9_distributions_boxplot.png")


def fig10_compute_interpretability() -> None:
    print("Figure 10: compute/interpretability table-as-figure")
    from controllers.fuzzy_handtuned import RULES, CRITICAL_FRONT
    import textwrap
    rows_raw = [
        ("A_baseline", "if/else thresholds", "0 rules", "yes"),
        ("B_handtuned_flc", f"Mamdani FLC + crisp safety override (<{CRITICAL_FRONT:.0f} units)",
         f"{len(RULES)} rules + 1 override", "yes"),
        ("D_nsga2_flc (knee/safety/eff.)", "Mamdani FLC (NSGA-II-evolved boundaries)",
         f"{len(RULES)} rules", "yes"),
        ("E_ann_imitator", "MLP (20 hidden units)", "~140 weights", "no"),
    ]
    # Wrap long cell text onto multiple lines instead of letting matplotlib
    # overflow it into neighboring cells (the Fig 10 overlap bug: long
    # "Mechanism" strings for B_handtuned_flc/D_nsga2_flc were wide enough
    # to visually collide with the next column at the previous fixed
    # figure width/font size).
    rows = [[ "\n".join(textwrap.wrap(str(cell), width=26)) for cell in row] for row in rows_raw]

    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.axis("off")
    table = ax.table(cellText=rows,
                      colLabels=["Controller", "Mechanism", "Rules/Params", "Interpretable?"],
                      loc="center", cellLoc="center",
                      colWidths=[0.24, 0.38, 0.20, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    # Extra vertical padding per row (proportional to wrapped line count)
    # and a touch of horizontal padding so wrapped multi-line cells never
    # visually touch the row above/below or the column to the right.
    # Height must be applied UNIFORMLY across a row (matplotlib's Table
    # grid renders misaligned borders if only some cells in a row are
    # resized), so compute one height per row from its most-wrapped cell.
    row_heights = []
    for r in range(len(rows) + 1):  # +1 for header row
        if r == 0:
            row_heights.append(0.16)
            continue
        max_lines = max(rows[r - 1][c].count("\n") + 1 for c in range(len(rows[r - 1])))
        row_heights.append(0.16 + 0.055 * (max_lines - 1))
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.PAD = 0.04
        cell.set_height(row_heights[row_idx])
    ax.set_title("Fig 10: Compute/interpretability comparison", pad=14)
    _save(fig, "fig10_compute_interpretability_table.png")


def fig11_confusion_matrix() -> None:
    print("Figure 11: fault-detection confusion matrix (row-normalized = recall)")
    path = os.path.join(RESULTS_DIR, "fault_detector_report.json")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found -- run experiments/train_fault_detector.py first.")
        return
    with open(path) as f:
        report = json.load(f)
    cm_norm = np.array(report["confusion_matrix_row_normalized"])
    cm_raw = np.array(report["confusion_matrix"])
    labels = report["labels"]
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n(n={int(cm_raw[i, j])})",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_title(f"Fig 11: Fault detection, per-class recall\n"
                 f"(balanced_acc={report.get('balanced_accuracy', float('nan')):.2f}, "
                 f"macro_F1={report.get('macro_f1', float('nan')):.2f}, "
                 f"acc={report['accuracy']:.2f})")
    fig.colorbar(im)
    _save(fig, "fig11_fault_detection_confusion_matrix.png")


def fig12_fault_blind_vs_aware(df) -> None:
    print("Figure 12: fault-blind vs fault-aware performance drop")
    aware_names = [c for c in df["controller"].unique() if c.endswith(FAULT_AWARE_SUFFIX)]
    if not aware_names:
        print("  [skip] no fault-aware controller rows found -- run "
              "experiments/train_fault_detector.py then run_full_experiment.py.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for aware_name in aware_names:
        blind_name = aware_name[: -len(FAULT_AWARE_SUFFIX)]
        if blind_name not in df["controller"].unique():
            continue
        for tier in sorted(df["tier"].unique()):
            clean_blind = df[(df.controller == blind_name) & (df.tier == tier) & (df.condition == "clean")]["success"].mean()
            faulty_blind = df[(df.controller == blind_name) & (df.tier == tier) & (df.condition != "clean")]["success"].mean()
            clean_aware = df[(df.controller == aware_name) & (df.tier == tier) & (df.condition == "clean")]["success"].mean()
            faulty_aware = df[(df.controller == aware_name) & (df.tier == tier) & (df.condition != "clean")]["success"].mean()
            drop_blind = (clean_blind or 0) - (faulty_blind or 0)
            drop_aware = (clean_aware or 0) - (faulty_aware or 0)
            ax.bar([f"{blind_name}\n({tier})", f"{aware_name}\n({tier})"], [drop_blind, drop_aware])
    ax.set_ylabel("Performance drop (clean - faulty success)")
    ax.set_title("Fig 12: Fault-blind vs fault-aware performance drop")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    _save(fig, "fig12_fault_blind_vs_aware.png")


def fig5_6_pareto() -> None:
    print("Figure 5/6: NSGA-II Pareto front + convergence")
    if not os.path.exists(NSGA2_PATH):
        print(f"  [skip] {NSGA2_PATH} not found -- run controllers/fuzzy_nsga2.optimize() "
              f"(requires pymoo) and save its result first.")
        return
    data = np.load(NSGA2_PATH)
    F = data["pareto_F"]
    selected = [(k, lbl, mk) for k, lbl, mk in
                (("knee_index", "knee", "*"), ("safety_index", "safety-weighted", "^"),
                 ("efficiency_index", "efficiency-weighted", "s")) if k in data]

    # 3D scatter + 2D projections in one figure (Fig 5): the 3D view alone
    # is ambiguous about depth/ordering: the 2D projections let a reader
    # verify exact trade-off positions without relying on perspective.
    fig = plt.figure(figsize=(14, 4.2))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.scatter(F[:, 0], F[:, 1], F[:, 2], c="tab:blue", alpha=0.4, s=25)
    ax_se = fig.add_subplot(1, 3, 2)
    ax_se.scatter(F[:, 0], F[:, 1], c="tab:blue", alpha=0.4, s=25)
    ax_ss = fig.add_subplot(1, 3, 3)
    ax_ss.scatter(F[:, 0], F[:, 2], c="tab:blue", alpha=0.4, s=25)

    for key, label, marker in selected:
        idx = int(data[key])
        ax3d.scatter([F[idx, 0]], [F[idx, 1]], [F[idx, 2]], c="red", marker=marker, s=110, label=label)
        ax_se.scatter([F[idx, 0]], [F[idx, 1]], c="red", marker=marker, s=90, label=label)
        ax_ss.scatter([F[idx, 0]], [F[idx, 2]], c="red", marker=marker, s=90, label=label)

    ax3d.set_xlabel("Safety"); ax3d.set_ylabel("Efficiency"); ax3d.set_zlabel("Smoothness")
    ax3d.set_title("3D Pareto front")
    ax_se.set_xlabel("Safety (obj)"); ax_se.set_ylabel("Efficiency (obj)")
    ax_se.set_title("Safety vs Efficiency"); ax_se.legend(fontsize=7)
    ax_ss.set_xlabel("Safety (obj)"); ax_ss.set_ylabel("Smoothness (obj)")
    ax_ss.set_title("Safety vs Smoothness")
    fig.suptitle("Fig 5: NSGA-II Pareto front (3D + 2D projections, all 3 objectives minimized)")
    _save(fig, "fig5_pareto_front_3d.png")

    if "convergence_hypervolume" in data:
        hv = data["convergence_hypervolume"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(hv) + 1), hv, marker=".", markersize=3)
        ax.set_xlabel("Generation"); ax.set_ylabel("Hypervolume (ref=[10,10,10])")
        final_gen = len(hv)
        final_hv = float(hv[-1])
        ax.annotate(f"gen {final_gen}: HV={final_hv:.2f}", xy=(final_gen, final_hv),
                    xytext=(0.6, 0.15), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->"), fontsize=8)
        ax.set_title("Fig 6: NSGA-II convergence (hypervolume vs generation)")
        _save(fig, "fig6_nsga2_convergence.png")
        print(f"  Final hypervolume from THIS run: {final_hv:.4f} at generation {final_gen} "
              f"(ref_point=[10,10,10]) -- read directly from {NSGA2_PATH}, not hardcoded.")
    else:
        print("  [skip fig6] no convergence_hypervolume in nsga2_result.npz "
              "(re-run fuzzy_nsga2.optimize() with the updated code).")


def analyze_safety_vs_efficiency_hypothesis(df) -> None:
    """
    Blueprint hypothesis check: does the safety-weighted Pareto solution
    show a smaller performance drop under sensor faults than the
    efficiency-weighted solution? Computed ONLY from real trial rows for
    D_nsga2_flc_safety / D_nsga2_flc_efficiency -- prints a clear message
    (not a fabricated answer) if that data isn't present.
    """
    print("\nHypothesis check: safety-weighted vs efficiency-weighted performance drop")
    names = df["controller"].unique()
    if "D_nsga2_flc_safety" not in names or "D_nsga2_flc_efficiency" not in names:
        print("  [cannot verify] D_nsga2_flc_safety / D_nsga2_flc_efficiency rows not found in "
              "full_trial_results.csv -- NSGA-II has not been run in this environment. "
              "No claim is made either way.")
        return
    rows = []
    for c in ("D_nsga2_flc_safety", "D_nsga2_flc_efficiency"):
        for tier in sorted(df["tier"].unique()):
            clean = df[(df.controller == c) & (df.tier == tier) & (df.condition == "clean")]["success"].mean()
            faulty = df[(df.controller == c) & (df.tier == tier) & (df.condition != "clean")]["success"].mean()
            rows.append((c, tier, (clean or 0) - (faulty or 0)))
    for c, tier, drop in rows:
        print(f"  {c} ({tier}): performance drop = {drop:.3f}")
    safety_drop = np.mean([d for c, t, d in rows if c == "D_nsga2_flc_safety"])
    eff_drop = np.mean([d for c, t, d in rows if c == "D_nsga2_flc_efficiency"])
    print(f"  Mean performance drop -- safety-weighted: {safety_drop:.3f}, "
          f"efficiency-weighted: {eff_drop:.3f}")
    print(f"  Hypothesis (safety-weighted drop < efficiency-weighted drop): "
          f"{'SUPPORTED' if safety_drop < eff_drop else 'NOT SUPPORTED'} by this data.")


def main() -> None:
    df = load_trials()
    if df is not None and len(df) > 0:
        fig1_2_clean_vs_faulty(df)
        fig3_severity_curve(df)
        fig4_per_fault_breakdown(df)
        fig8_failure_mode_taxonomy(df)
        fig9_distributions(df)
        fig12_fault_blind_vs_aware(df)
        analyze_safety_vs_efficiency_hypothesis(df)
    fig10_compute_interpretability()
    fig11_confusion_matrix()
    fig5_6_pareto()
    print(f"\nAll available figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
