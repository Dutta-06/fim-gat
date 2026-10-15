"""Reference spread and the equal-spread test (docs/design_v2.md section 2).

reference_spread    sigma_ref(G, k): the best of `runs` RR-Greedy runs, each on its own RR sample,
                    scored by forward Monte Carlo. It depends only on (G, k), so it is computed once
                    per graph and budget and reused by every fairness query.
equal_spread_test   one-sided test of sigma(S) >= rho * sigma_ref, on paired simulations with a
                    delta-method standard error for the ratio of means.

Every spread comes from forward Monte Carlo (metrics_v2.simulate, on the sorted seed set), never
from RR samples, so the test and evaluate_v2 see the same draws: mc_spreads(g, S, n, s) equals
evaluate_v2(g, S, n, s, return_sims=True)["sims"].

Pairing. diffusion._mc_forward reseeds its RNG at the start of each of its min(n_sims, 64) chunks
and consumes it edge by edge in BFS order. The same seed with the same set therefore gives
bit-identical simulations (ratio exactly 1, se exactly 0). The stream is NOT common across
different sets: they visit different edges, so the draws fall out of step after the first edge
where their cascades differ, and simulations of different sets are effectively independent. The
delta-method se stays valid for such pairs; it just gains nothing from the pairing.
"""
import numpy as np
from scipy.stats import norm

from ..baselines import RRGreedy
from ..diffusion import ExposureEstimator
from .config import REFERENCE_RUNS, RHO_DEFAULT
from .metrics_v2 import simulate


def mc_spreads(graph, seeds, n_sims=2000, seed=12345):
    """Spread (number of active nodes) of each forward-MC simulation: float64 (n_sims,)."""
    # UNVERIFIED: bit-identical repeats rely on numba keeping one RNG state per thread, so the
    # np.random.seed at the top of each prange chunk of _mc_forward fixes that chunk's stream.
    _, grp = simulate(graph, seeds, n_sims, seed)
    return grp.sum(1, dtype=np.float64)


def reference_spread(graph, k, runs=REFERENCE_RUNS, R=16, seed=0, n_sims=2000, device="cpu",
                     max_roots=None, mc_seed=12345):
    """Best-of-runs RR-Greedy reference for budget k.

    Run i builds ExposureEstimator(graph, R, seed=seed*1000 + i, device, max_roots) and takes
    RRGreedy(est).greedy(k). Every run is scored on the same MC stream (mc_seed) and the highest
    mean spread wins (the first on ties). Picking the maximum on the stream that scores it biases
    "spread" slightly upward (winner's curse). An equal-spread test run with seed == mc_seed
    re-simulates that same winning draw, which makes the test slightly harder to pass for every
    method alike.

    Returns {"seeds", "spread", "run_spreads", "run_seeds", "best_run", "spread_se"}.
    """
    k, runs = int(k), int(runs)
    if not 0 <= k <= graph.n:
        raise ValueError(f"budget k={k} outside [0, n={graph.n}]")
    if runs < 1:
        raise ValueError("runs must be >= 1")
    run_seeds, run_spreads, run_se, scored = [], [], [], {}
    for i in range(runs):
        est = ExposureEstimator(graph, R=int(R), seed=int(seed) * 1000 + i, device=device, max_roots=max_roots)
        s = [int(v) for v in RRGreedy(est).greedy(k)]
        del est
        key = tuple(sorted(s))
        if key not in scored:                       # identical sets give identical draws: simulate once
            sims = mc_spreads(graph, s, n_sims, mc_seed)
            se = float(sims.std(ddof=1) / np.sqrt(len(sims))) if len(sims) > 1 else float("nan")
            scored[key] = (float(sims.mean()), se)
        run_seeds.append(s)
        run_spreads.append(scored[key][0])
        run_se.append(scored[key][1])
    best = int(np.argmax(run_spreads))
    return {"seeds": list(run_seeds[best]), "spread": run_spreads[best], "run_spreads": run_spreads,
            "run_seeds": run_seeds, "best_run": best, "spread_se": run_se[best]}


def _sorted_ids(seeds):
    return np.sort(np.asarray(seeds, dtype=np.int64).ravel())


def _given_sims(sims, n_sims, what):
    x = np.asarray(sims, dtype=np.float64).ravel()
    if len(x) != int(n_sims):
        raise ValueError(f"{what} has {len(x)} simulations, expected n_sims={int(n_sims)}")
    return x


def equal_spread_test(graph, seeds, ref_seeds, rho=RHO_DEFAULT, n_sims=2000, seed=12345, alpha=0.05,
                      sims=None, ref_sims=None):
    """One-sided test of spread(seeds) >= rho * spread(ref_seeds) at level alpha.

        X = mc_spreads(seeds), Y = mc_spreads(ref_seeds)   (same n_sims and seed; paired by index)
        ratio = mean(X) / mean(Y)
        se    = std(X - ratio * Y, ddof=1) / (sqrt(n) * mean(Y))       (delta method)
        lower = ratio - z_{1-alpha} * se,   passed = lower >= rho

    Identical sets (in any order) share one simulation, so ratio == 1.0 and se == 0.0 exactly and
    the reference always passes for rho <= 1. sims / ref_sims optionally supply precomputed X / Y,
    for example the reference's draws reused across methods, or evaluate_v2(..., return_sims=True)
    ["sims"]. They must come from mc_spreads with this same n_sims and seed, or the pairing is lost;
    only their length is checked.

    Returns {"passed", "ratio", "lower", "se", "spread", "ref_spread"}.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if int(n_sims) < 2:
        raise ValueError("n_sims must be >= 2 for a standard error")
    a, b = _sorted_ids(seeds), _sorted_ids(ref_seeds)
    if len(b) == 0:
        raise ValueError("empty reference set")
    Y = mc_spreads(graph, b, n_sims, seed) if ref_sims is None else _given_sims(ref_sims, n_sims, "ref_sims")
    if sims is not None:
        X = _given_sims(sims, n_sims, "sims")
    elif np.array_equal(a, b):
        X = Y
    else:
        X = mc_spreads(graph, a, n_sims, seed)
    mx, my = float(X.mean()), float(Y.mean())
    if not my > 0.0:                        # cannot happen for simulated draws: seeds count themselves
        raise ValueError("reference spread is not positive")
    ratio = mx / my
    se = float(np.std(X - ratio * Y, ddof=1) / (np.sqrt(len(X)) * my))
    lower = ratio - float(norm.ppf(1.0 - float(alpha))) * se
    return {"passed": bool(lower >= rho), "ratio": ratio, "lower": lower, "se": se,
            "spread": mx, "ref_spread": my}
