"""Lower bounds on the best achievable disparity (docs/theory_v2.md, W2).

Lemma W2.2 (floor-forced mean). Labelled groups partition the nodes, so for every seed set
max_g r_g(S) >= sigma(S) / n, and every set meeting the floor sigma(S) >= F has D(S) >= F / n - min_g r_g(S).

Proposition W2.3 (granularity). With |S| = k at most k groups contain a seed. If U_g bounds group g's best
rate and U0_g its best rate when none of its members is a seed, then for every |S| = k

    min_g r_g(S) <= M := min(u_(k+1), min_g U_g),      u_(k+1) = the (k+1)-th smallest U0_g (G > k),

so every feasible set has D(S) >= F / n - M. This is the "granularity floor": with more hard-to-reach groups
than seeds, some group stays near its unseeded maximum while the floor keeps the mean rate up.

The U's come from the RR sample: greedy on the group's RR sets, the online bound for monotone submodular
coverage (OPT <= f(A) + the k largest marginal gains at A, minimised over the greedy prefixes A), and the
one-sided KL bound of Lemma 2 (certify.kl_bounds) applied to the (fixed, unknown) optimal set, whose sample
mean is at most B / N; the KL upper bound increases in the sample mean. Each U then holds for the true rate
with probability >= 1 - delta / (2 G), and the whole bound with probability >= 1 - delta.
"""
import math

import numpy as np

from .certify import kl_bounds
from .cover import cover_index


def _group_sets(ix, g):
    """(n_sets, nodes, set_of): the RR sets rooted in group g as a flat node list and each entry's set."""
    sets = np.flatnonzero(ix.set_group == g)
    starts = ix.ptr[sets]
    lens = ix.ptr[sets + 1] - starts
    total = int(lens.sum())
    off = np.cumsum(lens) - lens
    entries = np.repeat(starts - off, lens) + np.arange(total, dtype=np.int64)
    return len(sets), ix.nodes[entries], np.repeat(np.arange(len(sets), dtype=np.int64), lens)


def coverage_upper(nodes, set_of, n_sets, n, k, allowed):
    """Upper bound on max |{covered sets}| over k allowed seeds: min over greedy prefixes A of
    f(A) + sum of the k largest marginal gains at A (monotone submodular coverage)."""
    covered = np.zeros(n_sets, dtype=bool)
    chosen = np.zeros(n, dtype=bool)
    blocked = ~np.asarray(allowed, dtype=bool)
    kk = min(int(k), n)
    cov, best = 0.0, math.inf
    for t in range(int(k) + 1):
        gains = np.bincount(nodes[~covered[set_of]], minlength=n).astype(np.float64)
        gains[blocked | chosen] = 0.0
        top = float(np.partition(gains, n - kk)[n - kk:].sum()) if kk > 0 else 0.0
        best = min(best, cov + top)
        if t == int(k):
            break
        v = int(np.argmax(gains))
        if gains[v] <= 0.0:
            break
        chosen[v] = True
        covered[set_of[nodes == v]] = True
        cov += gains[v]
    return min(best, float(n_sets))


def group_rate_upper(est, k, exclude_own=False, delta=0.05):
    """U (G,): with probability >= 1 - delta, U_g >= max over |S| = k of the true rate r_g(S) for every
    present group (exclude_own=True: over S with no member of g). NaN for absent groups."""
    ix = cover_index(est)
    groups = np.asarray(est.graph.groups, dtype=np.int64)
    present = np.flatnonzero(ix.present)
    U = np.full(ix.G, np.nan)
    if present.size == 0:
        return U
    level = float(delta) / len(present)
    for g in present:
        n_sets, nodes, set_of = _group_sets(ix, int(g))
        allowed = np.ones(ix.n, dtype=bool)
        if exclude_own:
            allowed[groups == g] = False
        B = coverage_upper(nodes, set_of, n_sets, ix.n, k, allowed)
        # units: RR sets when every node is a root, else per-root averages in [0, 1]; same mean B / n_sets
        units = float(n_sets) if est.all_roots else n_sets / float(ix.R)
        U[g] = float(kl_bounds(B / max(n_sets, 1), units, level)[1])
    return U


def lower_bound_from(U, U0, k, mean_floor):
    """Deterministic part of Proposition W2.3 for given bounds (present groups only, no NaN)."""
    U, U0 = np.asarray(U, dtype=np.float64), np.asarray(U0, dtype=np.float64)
    u_k1 = float(np.sort(U0)[int(k)]) if len(U0) > int(k) else math.inf
    M = min(u_k1, float(U.min()))
    return {"lb": max(0.0, float(mean_floor) - M), "maximin_ub": M, "u_k1": u_k1, "min_U": float(U.min())}


def disparity_lower_bound(est, k, floor_total, delta=0.05):
    """Lower bound on min D(S) over |S| = k with true spread >= floor_total, valid with probability
    >= 1 - delta (Lemma W2.2 + Proposition W2.3). Returns {"lb", "mean_floor", "maximin_ub", "u_k1",
    "min_U", "U", "U0", "n_groups", "k"}; lb is NaN when a non-empty group has no RR roots."""
    ix = cover_index(est)
    sizes = np.asarray(est.graph.group_sizes)
    out = {"k": int(k), "n_groups": int(ix.present.sum())}
    if np.any((sizes > 0) & ~ix.present):
        out.update(lb=math.nan, mean_floor=math.nan, maximin_ub=math.nan, u_k1=math.nan, min_U=math.nan,
                   U=None, U0=None)
        return out
    mean_floor = float(floor_total) / float(sizes.sum())
    U = group_rate_upper(est, k, exclude_own=False, delta=delta / 2.0)
    U0 = group_rate_upper(est, k, exclude_own=True, delta=delta / 2.0)
    P = np.flatnonzero(ix.present)
    out.update(lower_bound_from(U[P], U0[P], k, mean_floor), mean_floor=mean_floor, U=U, U0=U0)
    return out
