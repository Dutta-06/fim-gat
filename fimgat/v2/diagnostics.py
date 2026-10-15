"""Cost model for move proposers (docs/theory_v2.md, W3) and the price loop (W4).

W3. Per accepted move, exhaustive search costs T_ex. A top-C proposer costs c_round = (scoring) + C * c_s
(exact checks); it contains an improving swap with probability pi and, when it does, its best swap gives
q (<= 1) of the exhaustive best swap's improvement. With an exhaustive fallback on a miss (the "hybrid",
which still ends at a single-swap local optimum):

    speed-up over exhaustive = (pi q + 1 - pi) T_ex / (c_round + (1 - pi) T_ex) <= 1 / (1 - pi)
                                                                           (Proposition W3.1)

and a learned proposer whose round costs at least cost_ratio * c_round can beat the simple hybrid by at most

    H = 1 / ((1 - phi) (pi q + 1 - pi) cost_ratio),   phi = (1 - pi) T_ex / (c_round + (1 - pi) T_ex),

phi being the share of the simple hybrid's time spent in fallbacks (Proposition W3.2). Both are a
stationary model (pi, q constant along the descent), not a theorem about every run; measure_proposer
measures their inputs at a state.

W4. The spread price mu_t = kp e_t + ki I_t, I_t = I_(t-1) + e_t, on the relative error e, with a
linearised plant e_(t+1) = e_t - gamma (mu_t - mu_bar), is stable iff 0 < gamma kp < 2, ki > 0 and
gamma (2 kp + ki) < 4 (Jury's criterion; Proposition W4.4).
"""
import math
import time

import numpy as np

from .cover import CoverState
from .local_search import TOL, best_swap
from .proposer import _top


# ------------------------------------------------------------------ W3: proposer cost model

def hybrid_speedup(pi, q, c_round, T_ex):
    """Proposition W3.1: expected speed-up of a top-C proposer with exhaustive fallback over exhaustive
    search alone (time per unit of disparity progress)."""
    pi, q = float(pi), float(q)
    return (pi * q + 1.0 - pi) * float(T_ex) / (float(c_round) + (1.0 - pi) * float(T_ex))


def fallback_share(pi, c_round, T_ex):
    """phi: the share of the hybrid's expected time per round spent in exhaustive fallbacks."""
    miss = (1.0 - float(pi)) * float(T_ex)
    return miss / (float(c_round) + miss)


def learning_headroom(pi, q, c_round, T_ex, cost_ratio=1.0):
    """Proposition W3.2: the largest speed-up over the simple hybrid of a proposer that always proposes the
    exhaustive best swap but whose round costs cost_ratio * c_round."""
    phi = fallback_share(pi, c_round, T_ex)
    return 1.0 / ((1.0 - phi) * (float(pi) * float(q) + 1.0 - float(pi)) * float(cost_ratio))


def measure_proposer(est, seeds, floor, grids=((4, 2), (8, 4), (16, 8), (32, 16)), scores=None, n_time=64,
                     device=None, seed=0):
    """Inputs of the W3 model at one state (labelled disparity, RR floor, no community family).

    A proposer checks every swap in the grid of its n_add best adds x n_drop best drops, as improve() does
    for the simple and the learned proposer. scores(state, floor) -> (add (n,), drop (n,)), higher = better
    (drop on seeds = better to remove); default imitation.simple_scores. Returns {"improvable",
    "best_all", "T_ex", "c_scores", "c_s", "pi": {(a, b): bool}, "q": {(a, b): float}}: whether some swap
    improves, the exhaustive best improvement, the time of local_search.best_swap, of the scoring and of
    one exact swap check, and for each grid whether it contains an improving swap and the ratio of its best
    improvement to the exhaustive best (C = a * b checks per round)."""
    from .imitation import improvement_matrix, simple_scores
    from .verify import evaluate_swap
    scores = simple_scores if scores is None else scores
    state = CoverState(est, [int(s) for s in seeds])
    t0 = time.perf_counter()
    best_swap(state, floor, device=device)
    T_ex = time.perf_counter() - t0
    imp, cols = improvement_matrix(state, floor, device=device)
    best_all = float(imp.max()) if imp.size else -math.inf
    t0 = time.perf_counter()
    add_sc, drop_sc = scores(state, floor)
    c_scores = time.perf_counter() - t0
    seeds_arr = np.asarray(state.seeds, dtype=np.int64)
    col_of = np.full(int(state.index.n), -1, dtype=np.int64)
    col_of[cols] = np.arange(len(cols), dtype=np.int64)
    improvable = best_all > TOL
    pi, q = {}, {}
    for a, b in grids:
        rows = _top(drop_sc[seeds_arr], np.arange(len(seeds_arr), dtype=np.int64), int(b))   # positions
        adds = _top(add_sc, cols, int(a))
        block = imp[np.ix_(rows, col_of[adds])] if len(rows) and len(adds) else np.zeros(0)
        top = float(block.max()) if block.size else -math.inf
        hit = bool(improvable and top > TOL)
        pi[(int(a), int(b))] = hit
        q[(int(a), int(b))] = top / best_all if hit else 0.0
    rng = np.random.default_rng(seed)
    non = np.flatnonzero(~state.is_seed)
    sample = [(int(rng.choice(seeds_arr)), int(rng.choice(non))) for _ in range(int(n_time))]
    t0 = time.perf_counter()
    for u, v in sample:
        evaluate_swap(state, u, v)
    c_s = (time.perf_counter() - t0) / max(len(sample), 1)
    return {"improvable": improvable, "best_all": best_all, "T_ex": T_ex, "c_scores": c_scores, "c_s": c_s,
            "pi": pi, "q": q}


def learning_decision(measurements, grid=(16, 8), cost_ratio=1.0, improve_share=None, headroom_min=None,
                      share_min=None):
    """The W3 decision rule of docs/design_v2.md section 18: should a learned proposer be trained at all?

    measurements: measure_proposer outputs on large-tier validation states, scored with improve()'s default
    simple proposer (imitation.simple_scores). The improvable states are pooled at `grid`: pi is the share
    whose grid holds an improving swap, q the mean progress ratio over those hits, c_round = mean scoring time
    + C * mean check time (C = a * b) and T_ex the mean exhaustive time. H = learning_headroom(pi, q, c_round,
    T_ex) bounds how much faster than the simple hybrid any single-swap proposer can be (Proposition W3.2);
    cost_ratio is the learned proposer's measured per-round cost relative to the simple one's (1 before a
    prototype exists). Train only if H / cost_ratio >= headroom_min (config.LEARN_HEADROOM_MIN) and, when
    improve_share (the improvement phase's share of per-query time) is given, improve_share >= share_min
    (config.LEARN_SHARE_MIN).

    Returns {"train", "H", "H_over_cost", "pi", "q", "c_round", "T_ex", "n_states", "reason"}."""
    from .config import LEARN_HEADROOM_MIN, LEARN_SHARE_MIN
    headroom_min = LEARN_HEADROOM_MIN if headroom_min is None else float(headroom_min)
    share_min = LEARN_SHARE_MIN if share_min is None else float(share_min)
    grid, cost_ratio = (int(grid[0]), int(grid[1])), float(cost_ratio)
    imp = [m for m in measurements if m["improvable"]]
    out = {"train": False, "H": math.nan, "H_over_cost": math.nan, "pi": math.nan, "q": math.nan,
           "c_round": math.nan, "T_ex": math.nan, "n_states": len(imp)}
    if not imp:
        out["reason"] = "no improvable state: the search is already at a local optimum, nothing to learn"
        return out
    hits = [m for m in imp if m["pi"][grid]]
    pi = len(hits) / len(imp)
    q = float(np.mean([m["q"][grid] for m in hits])) if hits else 0.0
    T_ex = float(np.mean([m["T_ex"] for m in imp]))
    c_round = (float(np.mean([m["c_scores"] for m in imp]))
               + grid[0] * grid[1] * float(np.mean([m["c_s"] for m in imp])))
    H = learning_headroom(pi, q, c_round, T_ex)
    out.update(H=H, H_over_cost=H / cost_ratio, pi=pi, q=q, c_round=c_round, T_ex=T_ex)
    if H / cost_ratio < headroom_min:
        out["reason"] = f"headroom H / cost_ratio = {H / cost_ratio:.2f} < {headroom_min:g}"
    elif improve_share is not None and float(improve_share) < share_min:
        out["reason"] = f"improvement is {float(improve_share):.0%} of per-query time < {share_min:.0%} (Amdahl)"
    else:
        out["train"] = True
        out["reason"] = f"headroom H / cost_ratio = {H / cost_ratio:.2f} >= {headroom_min:g}"
    return out


# ------------------------------------------------------------------ W4: the price loop

def pi_loop_stable(gamma, kp=1.0, ki=0.5):
    """Jury's criterion for the linearised spread-price loop (Proposition W4.4)."""
    g, kp, ki = float(gamma), float(kp), float(ki)
    return bool(g > 0 and ki > 0 and 0.0 < g * kp < 2.0 and g * (2.0 * kp + ki) < 4.0)


def pi_loop_radius(gamma, kp=1.0, ki=0.5):
    """Spectral radius of the linearised loop's state matrix on (e_t, I_(t-1))."""
    g, kp, ki = float(gamma), float(kp), float(ki)
    A = np.array([[1.0 - g * (kp + ki), -g * ki], [1.0, 1.0]])
    return float(np.abs(np.linalg.eigvals(A)).max())
