"""
config.py

Single source of truth for experiment-wide constants: seed counts, fault
severity levels, controller registry, file paths. Blueprint v2, Section
2.3/5.3 requires >=30 seeds and per-fault-type / severity-level reporting;
this module is the one place those numbers live so every script (debug,
experiment runner, fault-detector trainer, visualizer) stays consistent.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
NSGA2_PATH = os.path.join(RESULTS_DIR, "nsga2_result.npz")
DATASET_PATH = os.path.join(RESULTS_DIR, "dataset.npz")
ANN_MODEL_PATH = os.path.join(RESULTS_DIR, "ann_model.joblib")
FAULT_DETECTOR_PATH = os.path.join(RESULTS_DIR, "fault_detector.joblib")
FULL_TRIAL_CSV = os.path.join(RESULTS_DIR, "full_trial_results.csv")
DEMO_DIR = os.path.join(RESULTS_DIR, "demo")

for _d in (RESULTS_DIR, FIGURES_DIR, DEMO_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Experiment grid (Blueprint v2 Sec 2.3 / 5.3: >=30 seeds)
# --------------------------------------------------------------------------
N_SEEDS: int = 50
EVAL_SEED_BASE: int = 555
TIERS = ("sparse", "dense")

# Fault conditions evaluated per (controller, tier, seed):
#   "clean"                    -- no fault
#   "fault_dropout"             -- dropout fault, fixed magnitude
#   "fault_bias"                 -- bias fault, fixed magnitude
#   "fault_stale"                 -- stale-reading fault, fixed magnitude
#   "fault_noise_spike_sev{1,2,3}" -- noise-spike fault at 3 severity levels
#                                      (Blueprint v2 Sec 5.1: severity sweep)
FAULT_TYPES_FIXED = ("dropout", "bias", "stale")
NOISE_SPIKE_SEVERITIES = {
    1: (6.0, 12.0),     # low
    2: (15.0, 30.0),    # medium (matches original FaultSpec range)
    3: (40.0, 60.0),    # high
}

# --------------------------------------------------------------------------
# Fault-detection classifier (Blueprint v2 Sec 6.2)
# --------------------------------------------------------------------------
FAULT_DETECTOR_WINDOW: int = 12          # rolling window length in timesteps
FAULT_DETECTOR_CLASSES = ("none", "dropout", "bias", "noise_spike", "stale")
FAULT_DETECTOR_N_EPISODES: int = 160     # increased from 90 for better minority-class recall
FAULT_DETECTOR_CONF_THRESHOLD: float = 0.6  # for confidence-gated control

# --------------------------------------------------------------------------
# Held-out evaluation seeds: SINGLE SOURCE OF TRUTH so every script (full
# experiment grid, NSGA-II training-seed sampling, imitation-data
# generation) can guarantee its own seeds are disjoint from the seeds used
# for final A/B/D/E(+fault-aware) evaluation. This directly addresses the
# "no leakage" requirement: NSGA-II gene fitness and the ANN's imitation
# targets must never be computed on a seed that is later used to report
# that same controller's held-out success rate.
# --------------------------------------------------------------------------
def get_eval_trial_seeds(n_seeds: int = N_SEEDS, base: int = EVAL_SEED_BASE):
    import numpy as np
    rng = np.random.default_rng(base)
    return [int(s) for s in rng.integers(0, 1_000_000, size=n_seeds)]


def sample_disjoint_seeds(rng, n: int, exclude: set, hi: int = 1_000_000):
    """Draw n unique int seeds from rng, guaranteed disjoint from `exclude`."""
    out = []
    seen = set()
    while len(out) < n:
        s = int(rng.integers(0, hi))
        if s in exclude or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# --------------------------------------------------------------------------
# Controller registry (names must match across experiment + viz scripts)
# --------------------------------------------------------------------------
CONTROLLER_NAMES = (
    "A_baseline",
    "B_handtuned_flc",
    "D_nsga2_flc",
    "E_ann_imitator",
)
# 5th condition added by Blueprint v2 Sec 6.2: best controller run fault-blind
# vs fault-aware. Populated at experiment time once the best controller from
# CONTROLLER_NAMES is identified from clean-condition results.
FAULT_AWARE_SUFFIX = "_fault_aware"
