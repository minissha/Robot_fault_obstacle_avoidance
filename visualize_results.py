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
    ANN_MODEL_PATH, FAULT_AWARE_SUFFIX, FAULT_DETECTOR_PATH, FIGURES_DIR,
    FULL_TRIAL_CSV, NSGA2_PATH, RESULTS_DIR, get_eval_trial_seeds,
)
from experiments.statistics_utils import bootstrap_ci, load_canonical_trials

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
    """
    Uses the same loader as the stats script, so a figure and a number in
    the report can't end up describing different runs. That used to happen:
    Figure 1 and stats_report.md disagreed about the same controller because
    one averaged both runs in the file and the other read only one.
    """
    if pd is None:
        print("[skip all trial-based figures] pandas not installed.")
        return None
    try:
        df = load_canonical_trials(FULL_TRIAL_CSV)
    except FileNotFoundError as e:
        # No file just means nothing's been run yet, so skip quietly. A file
        # that exists but is a mess should stop everything, so that error is
        # deliberately left uncaught.
        print(f"[skip all trial-based figures] {e}")
        return None
    print(f"Trials: {len(df)} rows @ artifact fingerprint "
          f"{df.attrs['artifact_fingerprint']}")
    return df


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
    """Path length and minimum clearance, clean against faulty.

    The blueprint asks for both conditions side by side, not just clean:
    the point is to show the whole distribution rather than a mean, and how
    a fault stretches it.
    """
    print("Figure 9: path length / clearance distributions, clean vs faulty")
    controllers = sorted(df["controller"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(1.6 * len(controllers) + 4, 8.5))
    for ax, col, title in ((axes[0], "path_length", "Path length"),
                            (axes[1], "min_clearance", "Minimum clearance")):
        positions, data, colours = [], [], []
        for i, c in enumerate(controllers):
            for j, (label, mask) in enumerate((
                    ("clean", df.condition == "clean"),
                    ("faulty", df.condition != "clean"))):
                vals = df[(df.controller == c) & mask][col].dropna().values
                if not len(vals):
                    continue
                positions.append(i * 2.6 + j * 0.9)
                data.append(vals)
                colours.append("tab:blue" if j == 0 else "tab:red")
        bp = ax.boxplot(data, positions=positions, widths=0.75,
                        patch_artist=True, showfliers=False)
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour); patch.set_alpha(0.45)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_xticks([i * 2.6 + 0.45 for i in range(len(controllers))])
        ax.set_xticklabels(controllers, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f"Fig 9: {title} distribution, clean (blue) vs faulty (red)")
    fig.tight_layout()
    _save(fig, "fig9_distributions_boxplot.png")

def fig10_compute_interpretability() -> None:
    """Cost against interpretability, with inference time actually measured.

    The blueprint asks for inference time per control step alongside the
    rule/parameter counts. It used to be a hand-written table with no timing
    column at all, which is the one number in it that cannot be read off the
    source.
    """
    print("Figure 10: compute/interpretability table-as-figure")
    import textwrap, time
    from controllers.fuzzy_handtuned import RULES, CRITICAL_AHEAD, HandTunedFLC
    from controllers.baseline import BaselineController
    from simulation_core import N_SENSORS, OBSERVATION_DIM

    def time_per_step(controller, n: int = 2000) -> float:
        """Microseconds per predict() call, on observations drawn from a real
        episode so the timing reflects the branches actually taken."""
        from simulation_core import RobotSimulator
        probe = RobotSimulator("dense", seed=get_eval_trial_seeds()[0],
                                faulty=False).run(controller)
        obs = np.array(probe.full_observation_log, dtype=np.float64)
        if not len(obs):
            return float("nan")
        controller.reset()
        picks = obs[np.arange(n) % len(obs)]
        t0 = time.perf_counter()
        for row in picks:
            controller.predict(row)
        return (time.perf_counter() - t0) / n * 1e6

    def safe_time(build) -> float:
        """A stale trained artifact (wrong input width, say) should leave one
        cell blank, not take the whole figure down."""
        try:
            return time_per_step(build())
        except Exception as exc:
            print(f"  [warn] could not time a controller: "
                  f"{type(exc).__name__}: {exc}")
            return float("nan")

    timings = {"A_baseline": safe_time(BaselineController),
               "B_handtuned_flc": safe_time(HandTunedFLC)}
    if os.path.exists(NSGA2_PATH):
        from controllers.fuzzy_nsga2 import load_optimized_controller
        timings["D_nsga2_flc"] = safe_time(
            lambda: load_optimized_controller(NSGA2_PATH, "knee"))
    if os.path.exists(ANN_MODEL_PATH):
        from controllers.ann_imitator import ANNController
        timings["E_ann_imitator"] = safe_time(
            lambda: ANNController.load(ANN_MODEL_PATH))

    def us(name):
        v = timings.get(name)
        return "n/a" if v is None or np.isnan(v) else f"{v:.1f} us"

    n_weights = "n/a"
    if os.path.exists(ANN_MODEL_PATH):
        try:
            import joblib
            mlp = joblib.load(ANN_MODEL_PATH)
            mlp = mlp.named_steps["mlp"] if hasattr(mlp, "named_steps") else mlp
            total = (sum(w.size for w in mlp.coefs_)
                      + sum(b.size for b in mlp.intercepts_))
            n_weights = f"~{total:,} weights"
        except Exception:
            pass

    rows_raw = [
        ("A_baseline", "crisp thresholds on the clearance profile",
         "3 thresholds, 4 speeds", us("A_baseline"), "yes"),
        ("B_handtuned_flc", f"Mamdani FLC + crisp safety reflex (<{CRITICAL_AHEAD:.0f} units)",
         f"{len(RULES)} rules + 1 reflex", us("B_handtuned_flc"), "yes"),
        ("D_nsga2_flc", "Mamdani FLC, boundaries evolved by NSGA-II",
         f"{len(RULES)} rules, 8 genes", us("D_nsga2_flc"), "yes"),
        ("E_ann_imitator", "MLP 64-64, DAgger-trained", n_weights,
         us("E_ann_imitator"), "no"),
    ]
    rows = [["\n".join(textwrap.wrap(str(cell), width=24)) for cell in row]
            for row in rows_raw]

    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.axis("off")
    table = ax.table(cellText=rows,
                      colLabels=["Controller", "Mechanism", "Rules / parameters",
                                  "Inference time\nper step", "Interpretable?"],
                      loc="center", cellLoc="center",
                      colWidths=[0.20, 0.30, 0.20, 0.16, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.4)
    for (r, _), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(weight="bold")
    ax.set_title(f"Fig 10: Compute and interpretability "
                  f"({N_SENSORS} rays, {OBSERVATION_DIM}-dim observation)", pad=16)
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


def _best_controller_name(df) -> str:
    """Whichever controller did best on the clean runs. Same rule as the
    experiment runner uses, so the figures show the same one."""
    clean = df[df["condition"] == "clean"]
    clean = clean[~clean["controller"].str.endswith(FAULT_AWARE_SUFFIX)]
    if len(clean) == 0:
        return "B_handtuned_flc"
    rates = clean.groupby("controller")["success"].mean()
    return sorted(rates.index, key=lambda n: (-rates[n], n))[0]


def fig7_trajectory_overlay(df) -> None:
    """
    Two runs of the same controller on the same map, one clean and one with
    a broken sensor, drawn on top of each other.

    Everything else in here is bar charts showing that faults cost you
    success. This is the one that shows what that actually looks like - same
    seed, same obstacles, the paths splitting at the point the fault starts.
    """
    print("Figure 7: trajectory overlay, clean vs faulty (single seed)")
    from simulation_core import (
        GOAL_POS, GOAL_RADIUS, START_POS, WORLD_SIZE, RobotSimulator,
        sample_fault_spec_typed,
    )
    from controllers.baseline import BaselineController
    from controllers.fuzzy_handtuned import HandTunedFLC

    name = _best_controller_name(df)
    # Only the two dependency-free controllers can be rebuilt here without
    # loading a trained artifact; fall back to the FLC for any other winner.
    factory = {"A_baseline": BaselineController}.get(name, HandTunedFLC)
    if factory is HandTunedFLC and name != "B_handtuned_flc":
        name = "B_handtuned_flc"

    tier = "sparse"
    fault_type = "bias"

    # Pick the most informative seed: clean reaches the goal, faulty does not.
    # Deterministic (first match over the held-out seeds), and falls back to
    # the first seed if no such contrast exists rather than inventing one.
    chosen, clean_res, faulty_res = None, None, None
    for seed in get_eval_trial_seeds():
        c = RobotSimulator(tier, seed=seed, faulty=False).run(factory())
        spec = sample_fault_spec_typed(seed, fault_type=fault_type)
        f = RobotSimulator(tier, seed=seed, faulty=True, spec_override=spec).run(factory())
        if chosen is None:
            chosen, clean_res, faulty_res = seed, c, f      # fallback
        if c.metrics.success and not f.metrics.success:
            chosen, clean_res, faulty_res = seed, c, f
            break

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for obs in RobotSimulator(tier, seed=chosen, faulty=False).obstacles:
        ax.add_patch(plt.Rectangle((obs.x_min, obs.y_min),
                                    obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                                    facecolor="0.75", edgecolor="0.45", zorder=1))

    for res, colour, style, label in (
            (clean_res, "tab:blue", "-", "clean"),
            (faulty_res, "tab:red", "--", f"faulty ({fault_type})")):
        traj = np.array(res.trajectory)
        outcome = ("success" if res.metrics.success
                   else "collision" if res.metrics.collision else "timeout")
        ax.plot(traj[:, 0], traj[:, 1], style, color=colour, lw=1.8, zorder=3,
                label=f"{label} -> {outcome} ({res.metrics.steps} steps)")
        ax.plot(traj[-1, 0], traj[-1, 1], "x", color=colour, ms=9, mew=2, zorder=4)

    # Mark where the fault actually switches on -- without it a reader cannot
    # tell which part of the divergence the fault is responsible for.
    spec = faulty_res.fault_spec
    if spec is not None:
        traj = np.array(faulty_res.trajectory)
        i = min(spec.start_step, len(traj) - 1)
        ax.plot(traj[i, 0], traj[i, 1], "o", color="tab:red", ms=8,
                markerfacecolor="none", mew=2, zorder=5,
                label=f"fault onset (step {spec.start_step}, sensor {spec.sensor_index})")

    ax.add_patch(plt.Circle(GOAL_POS, GOAL_RADIUS, facecolor="tab:green",
                             alpha=0.35, edgecolor="tab:green", zorder=2))
    ax.plot(*START_POS, "o", color="k", ms=7, zorder=5)
    ax.annotate("start", START_POS, textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.annotate("goal", GOAL_POS, textcoords="offset points", xytext=(-30, 8), fontsize=8)

    ax.set_xlim(0, WORLD_SIZE); ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Fig 7: {name} on {tier} seed {chosen}\nclean vs {fault_type}-faulty trajectory")
    _save(fig, "fig7_trajectory_overlay.png")


def fig13_sensor_trust_trace(df) -> None:
    """
    How much the fault-aware controller trusts each ray, step by step, over
    one faulty episode.

    The wrapper does not switch a ray off; it slides its weight between 1
    and 0 with the detector's confidence, and mixes the reading towards
    what the neighbouring rays imply. That is the adaptive sensor weighting
    the blueprint asks for in Section 6.3, and this is the trace of it
    working. The bias fault is used because it is the one fault a wider
    sensor fan does not absorb on its own.
    """
    print("Figure 13: per-ray trust weights over one faulty episode")
    if not os.path.exists(FAULT_DETECTOR_PATH):
        print(f"  [skip] {FAULT_DETECTOR_PATH} not found -- run "
              f"experiments/train_fault_detector.py first.")
        return

    from simulation_core import (
        RobotSimulator, SENSOR_NAMES, sample_fault_spec_typed,
    )
    from controllers.fault_detector import FaultDetector
    from controllers.fault_aware import FaultAwareController
    from controllers.fuzzy_handtuned import HandTunedFLC

    detector = FaultDetector.load(FAULT_DETECTOR_PATH)
    fault_type = "bias"

    # Pick the first held-out seed where the fault actually gets detected, so
    # the figure shows the mechanism rather than a quiet episode. Falls back
    # to the first seed rather than inventing one.
    chosen = None
    for seed in get_eval_trial_seeds()[:40]:
        spec = sample_fault_spec_typed(seed, fault_type=fault_type)
        ctrl = FaultAwareController(HandTunedFLC(), detector)
        RobotSimulator("sparse", seed=seed, faulty=True, spec_override=spec).run(ctrl)
        trust = np.array(ctrl.trust_log)
        if chosen is None:
            chosen = (seed, spec, trust)
        if len(trust) and trust[:, spec.sensor_index].min() < 0.5:
            chosen = (seed, spec, trust)
            break
    seed, spec, trust = chosen
    if not len(trust):
        print("  [skip] episode produced no trust trace.")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = np.arange(len(trust))
    for i in range(trust.shape[1]):
        faulted = i == spec.sensor_index
        ax.plot(steps, trust[:, i],
                color="tab:red" if faulted else "0.75",
                linewidth=2.0 if faulted else 0.9,
                zorder=3 if faulted else 1,
                label=f"{SENSOR_NAMES[i]} (faulted)" if faulted else None)

    ax.axvspan(spec.start_step, min(spec.start_step + spec.duration, len(trust)),
               color="tab:orange", alpha=0.12, zorder=0,
               label=f"{fault_type} fault active")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Trust weight (1 = reading used as-is)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Fig 13: Adaptive sensor trust, {fault_type} fault on "
                  f"{SENSOR_NAMES[spec.sensor_index]} (seed {seed})")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, "fig13_sensor_trust_trace.png")

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
        fig7_trajectory_overlay(df)
        fig8_failure_mode_taxonomy(df)
        fig9_distributions(df)
        fig12_fault_blind_vs_aware(df)
        fig13_sensor_trust_trace(df)
        analyze_safety_vs_efficiency_hypothesis(df)
    fig10_compute_interpretability()
    fig11_confusion_matrix()
    fig5_6_pareto()
    print(f"\nAll available figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
