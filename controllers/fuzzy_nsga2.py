"""
fuzzy_nsga2.py

Controller D: NSGA-II-tuned Mamdani fuzzy logic controller.

Reuses the exact rule base and inference machinery of HandTunedFLC
(controllers/fuzzy_handtuned.py), but evolves the eight membership-function
boundary genes
[a_near, a_far, l_near, l_far, r_near, r_far, v_near, v_far]
via NSGA-II (implemented with pymoo) over three objectives:

    O1 Safety      = collision_rate + penalty for low minimum clearance
    O2 Efficiency  = mean(path_length / straight_line_distance) over
                      successful episodes, with a fixed penalty for
                      failures that is strictly worse than any real
                      success (see BUGFIX note in _evaluate_genes)
    O3 Smoothness  = mean steering-angle change between consecutive steps

All three objectives are minimized. The final controller is selected from
the Pareto front by knee-point selection (minimum Euclidean distance, in
normalized objective space, to the ideal point).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from controllers.fuzzy_handtuned import HandTunedFLC
from simulation_core import RobotSimulator, MAX_STEPS, START_POS, GOAL_POS, ROBOT_RADIUS
from config import get_eval_trial_seeds, sample_disjoint_seeds
import math

STRAIGHT_LINE_DISTANCE: float = math.dist(START_POS, GOAL_POS)
# Failure penalty for the efficiency objective: must exceed the WORST
# possible ratio a genuine success could ever produce (path_length is
# capped at MAX_STEPS*ROBOT_SPEED=800, so path_length/STRAIGHT_LINE_DISTANCE
# <= 800/118.79 ~= 6.7 for any real success), so a failed episode is
# always scored strictly worse than any successful one, however indirect.
EFFICIENCY_FAILURE_PENALTY: float = 8.0

# Clearance the safety objective aims for: one body diameter of daylight
# between the robot and whatever it is passing.
CLEARANCE_TARGET: float = 2.0 * ROBOT_RADIUS

# Gene bounds: [a_near, a_far, l_near, l_far, r_near, r_far, v_near, v_far].
# The first six set where each input stops counting as Near and starts
# counting as Far; the last two do the same for the speed side, so the
# optimiser can decide independently how early to start turning and how
# early to lift off.
GENE_LOWER: Tuple[float, ...] = (6.0, 22.0, 6.0, 22.0, 6.0, 22.0, 5.0, 20.0)
GENE_UPPER: Tuple[float, ...] = (26.0, 60.0, 26.0, 60.0, 26.0, 60.0, 26.0, 60.0)
N_GENES: int = len(GENE_LOWER)

# 48 rather than 16. At 16 the front was noisy enough that the
# safety-weighted and efficiency-weighted picks kept landing on the very
# same solution, which makes comparing them meaningless.
N_TRAINING_EPISODES: int = 48
POP_SIZE: int = 32              # within 30-40
N_GENERATIONS: int = 40


class NSGA2FLC(HandTunedFLC):
    """Identical to HandTunedFLC but constructed from evolved gene parameters."""
    pass


def _evaluate_genes(genes: np.ndarray, training_seeds: List[int]) -> Tuple[float, float, float]:
    """
    Run the FLC defined by `genes` over `training_seeds` (mixed sparse/dense,
    clean condition, used only for optimization) and return (O1, O2, O3).

    BUGFIX (this pass -- CONFIRMED SEVERE): the efficiency objective was
    `path_length / time_to_goal`. Since the robot moves at a FIXED speed
    every step (ROBOT_SPEED=2.0 units/step, verified empirically: every
    successful episode gives EXACTLY path_length/time_to_goal == 2.0,
    regardless of path directness), this objective carried ZERO
    information for distinguishing a good successful controller from a
    bad one -- it was a constant. Worse: the failure penalty was 1.0,
    which is LOWER (better, since NSGA-II MINIMIZES) than the constant
    2.0 every success produced. NSGA-II was therefore optimizing this
    axis in the WRONG DIRECTION: minimizing it rewarded MORE failures,
    not fewer -- directly contradicting the safety objective and the
    actual success-rate metric the final experiment reports. This is very
    likely the primary cause of "D performs substantially worse than
    A/B": the optimizer was not optimizing the same quantity the final
    experiment measures, exactly the mismatch this audit asked to check.

    Fixed to `path_length / straight_line_distance` for successful
    episodes (a genuine directness ratio, minimum 1.0 for a perfectly
    straight path, always > 1.0 for any real trajectory) with a fixed
    failure penalty (EFFICIENCY_FAILURE_PENALTY=8.0) set strictly above
    the worst-case ratio any real success could produce (max path_length
    800 / 118.79 ~= 6.7), so failing is now unambiguously worse than
    succeeding on this axis, for every gene set. NOT re-run this pass
    (requires pymoo, unavailable in this sandbox) -- see FINAL_REPORT.md
    for the exact command; the existing nsga2_result.npz was generated
    under the OLD, broken objective and left in place rather than
    silently replaced with an unverified guess.
    """
    lower_bounds = genes[0::2]
    upper_bounds = genes[1::2]
    if np.any(lower_bounds >= upper_bounds):
        # Invalid ordering: penalize heavily instead of crashing the search.
        return 10.0, 10.0, 10.0

    controller = HandTunedFLC(params=tuple(genes.tolist()))

    collisions = 0
    clearances: List[float] = []
    successful_perf: List[float] = []
    smoothness_vals: List[float] = []

    for i, seed in enumerate(training_seeds):
        tier = "sparse" if i % 2 == 0 else "dense"
        sim = RobotSimulator(tier=tier, seed=seed, faulty=False)
        result = sim.run(controller)
        m = result.metrics

        if m.collision:
            collisions += 1
        clearances.append(m.min_clearance)
        smoothness_vals.append(m.steering_smoothness)

        if m.success:
            successful_perf.append(m.path_length / STRAIGHT_LINE_DISTANCE)
        else:
            successful_perf.append(EFFICIENCY_FAILURE_PENALTY)

    n = len(training_seeds)
    collision_rate = collisions / n
    mean_clearance = float(np.mean(clearances))
    # min_clearance is now the real distance from the robot's edge to the
    # nearest thing, not the smallest range reading, so the scale it is
    # compared against had to come down with it. CLEARANCE_TARGET is one
    # body diameter of breathing room.
    clearance_penalty = max(0.0, (CLEARANCE_TARGET - mean_clearance) / CLEARANCE_TARGET)
    safety = collision_rate + 0.5 * clearance_penalty

    efficiency = float(np.mean(successful_perf))
    smoothness = float(np.mean(smoothness_vals)) / 45.0  # normalize by max steering range

    return safety, efficiency, smoothness


def optimize(seed: int = 0, n_generations: int = N_GENERATIONS,
             pop_size: int = POP_SIZE, verbose: bool = True):
    """
    Run NSGA-II over the eight FLC boundary genes. Returns a dict with the
    Pareto front (X, F), convergence history, and THREE selected solutions
    (knee point + safety-weighted extreme + efficiency-weighted extreme),
    not just the knee point -- for the "evaluate multiple Pareto solutions"
    requirement.

    training_seeds are drawn disjoint from the held-out evaluation seeds
    (config.get_eval_trial_seeds) so the genes are never fitted on a seed
    later used to report D_nsga2_flc's held-out success rate -- no leakage.
    """
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.callback import Callback
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.indicators.hv import HV

    rng = np.random.default_rng(seed)
    eval_seeds = set(get_eval_trial_seeds())
    training_seeds = sample_disjoint_seeds(rng, N_TRAINING_EPISODES, exclude=eval_seeds)

    class FLCProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=N_GENES,
                n_obj=3,
                n_constr=0,
                xl=np.array(GENE_LOWER),
                xu=np.array(GENE_UPPER),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            o1, o2, o3 = _evaluate_genes(x, training_seeds)
            out["F"] = [o1, o2, o3]

    # Reference point for hypervolume: worst-case per-objective values seen
    # in _evaluate_genes' invalid-ordering penalty (10,10,10) provide a
    # safe, evaluation-independent upper bound (not fitted to results).
    hv_indicator = HV(ref_point=np.array([10.0, 10.0, 10.0]))

    class ConvergenceCallback(Callback):
        """Records per-generation hypervolume and per-objective min/mean,
        so NSGA-II convergence (Blueprint Fig 6) is reproducible from a
        saved .npz rather than re-run from scratch."""

        def __init__(self):
            super().__init__()
            self.data["hypervolume"] = []
            self.data["gen_F_min"] = []
            self.data["gen_F_mean"] = []

        def notify(self, algorithm):
            F = algorithm.pop.get("F")
            self.data["hypervolume"].append(float(hv_indicator(F)))
            self.data["gen_F_min"].append(F.min(axis=0).tolist())
            self.data["gen_F_mean"].append(F.mean(axis=0).tolist())

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    callback = ConvergenceCallback()
    result = pymoo_minimize(
        FLCProblem(),
        algorithm,
        ("n_gen", n_generations),
        seed=seed,
        verbose=verbose,
        callback=callback,
    )

    pareto_X = result.X
    pareto_F = result.F
    if pareto_X.ndim == 1:
        pareto_X = pareto_X.reshape(1, -1)
        pareto_F = pareto_F.reshape(1, -1)

    knee_idx = select_knee_point(pareto_F)
    safety_idx = int(np.argmin(pareto_F[:, 0]))       # safety-weighted extreme
    efficiency_idx = int(np.argmin(pareto_F[:, 1]))   # efficiency-weighted extreme

    return {
        "pareto_X": pareto_X,
        "pareto_F": pareto_F,
        "knee_index": knee_idx,
        "knee_genes": pareto_X[knee_idx],
        "safety_index": safety_idx,
        "safety_genes": pareto_X[safety_idx],
        "efficiency_index": efficiency_idx,
        "efficiency_genes": pareto_X[efficiency_idx],
        "training_seeds": np.array(training_seeds),
        "convergence_hypervolume": np.array(callback.data["hypervolume"]),
        "convergence_gen_F_min": np.array(callback.data["gen_F_min"]),
        "convergence_gen_F_mean": np.array(callback.data["gen_F_mean"]),
    }


def select_knee_point(pareto_F: np.ndarray) -> int:
    """
    Select the knee-point solution from a Pareto front: the point with the
    minimum Euclidean distance (in min-max normalized objective space) to
    the ideal point (0, 0, 0).
    """
    f_min = pareto_F.min(axis=0)
    f_max = pareto_F.max(axis=0)
    span = np.where(f_max - f_min > 1e-9, f_max - f_min, 1.0)
    normalized = (pareto_F - f_min) / span
    distances = np.linalg.norm(normalized, axis=1)
    return int(np.argmin(distances))


def load_optimized_controller(npz_path: str, which: str = "knee") -> NSGA2FLC:
    """Load a previously saved NSGA-II result and build the controller.
    which: "knee" (default, used as D_nsga2_flc), "safety", or "efficiency"
    -- lets run_full_experiment.py evaluate all three Pareto-front
    alternatives, not only the knee point."""
    data = np.load(npz_path)
    key = {"knee": "knee_genes", "safety": "safety_genes", "efficiency": "efficiency_genes"}[which]
    genes = tuple(data[key].tolist())
    return NSGA2FLC(params=genes)
