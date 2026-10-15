"""v2 evaluation metrics, from forward Monte Carlo only (never from RR samples).

evaluate_v2 scores a seed set, always with group *rates*, on
  * the labelled attribute: spread, group rates, disparity (max - min over present groups);
  * held-out attributes: every other attribute in graph.meta["attrs"], which no method sees;
  * communities: ground-truth communities when given (overlap allowed, nodes may belong to
    none), otherwise v1's Louvain partition (metrics.communities) restricted to groups of
    at least min_comm members. Community metrics are NaN when nothing qualifies.

Differences from v1 metrics.evaluate: disparity is the headline number, held-out attributes
also report disparity, communities may overlap, and the seeds are sorted before simulating.
Sorting makes every number a function of the seed *set* rather than of its listing order,
and makes evaluate_v2(..., return_sims=True)["sims"] identical, draw for draw, to
reference.mc_spreads with the same n_sims and seed.
"""
import numpy as np

from .. import metrics as M
from ..diffusion import monte_carlo

NAN = float("nan")


def simulate(graph, seeds, n_sims=2000, seed=12345):
    """diffusion.monte_carlo on the sorted seed set: (node_prob (n,), grp (n_sims, G) int32).

    The same set and seed give bit-identical simulations, whatever order the set is listed in.
    Inputs are checked here because the numba kernel does no bounds checking."""
    s = np.sort(np.asarray(seeds, dtype=np.int64).ravel())
    if len(s) and (s[0] < 0 or s[-1] >= graph.n):
        raise ValueError("seed id out of range")
    if np.any(s[1:] == s[:-1]):
        raise ValueError("duplicate seeds")
    if int(n_sims) < 1:
        raise ValueError("n_sims must be >= 1")
    if graph.n and (graph.groups.min() < 0 or graph.groups.max() >= graph.n_groups):
        raise ValueError("group ids must lie in [0, n_groups)")
    return monte_carlo(graph, s, n_sims=int(n_sims), seed=int(seed))


def _gap(r):
    """max - min; 0.0 with fewer than two entries."""
    return float(r.max() - r.min()) if len(r) >= 2 else 0.0


def _attr_rates(node_prob, groups):
    """Rate of every non-empty group of an attribute. Negative (or NaN) codes mean 'missing'."""
    groups = np.asarray(groups).ravel()
    if groups.dtype.kind == "f":
        groups = np.where(np.isfinite(groups), groups, -1)
    groups = groups.astype(np.int64)
    if len(groups) != len(node_prob):
        raise ValueError(f"attribute has {len(groups)} entries for {len(node_prob)} nodes")
    ok = groups >= 0
    sizes = np.bincount(groups[ok])
    hit = np.bincount(groups[ok], weights=node_prob[ok], minlength=len(sizes))
    keep = sizes > 0
    return hit[keep] / sizes[keep]


def _community_rates(node_prob, communities):
    """Mean node_prob over each community with at least one member in the graph.

    Communities may overlap. A negative id marks a member outside this graph (for example
    outside a sampled subgraph) and is ignored; repeated ids within a community count once."""
    n = len(node_prob)
    if isinstance(communities, dict):
        communities = communities.values()
    elif isinstance(communities, np.ndarray) and communities.ndim == 1 and communities.dtype != object:
        raise ValueError("communities must be a list of node-id arrays, not one array or a label vector")
    parts = [np.asarray(list(c) if isinstance(c, (set, frozenset)) else c, dtype=np.int64).ravel()
             for c in communities]
    if not parts:
        return np.zeros(0)
    C = len(parts)
    cid = np.repeat(np.arange(C, dtype=np.int64), [len(p) for p in parts])
    mem = np.concatenate(parts)
    if mem.size and mem.max() >= n:
        raise ValueError("community member id >= graph.n: communities must use this graph's ids")
    keep = mem >= 0
    code = np.unique(cid[keep] * np.int64(n) + mem[keep])        # one entry per (community, member)
    cid, mem = code // n, code % n
    size = np.bincount(cid, minlength=C)
    hit = np.bincount(cid, weights=node_prob[mem], minlength=C)
    ok = size > 0
    return hit[ok] / size[ok]


def _louvain_rates(graph, node_prob, min_comm):
    """Rates of v1 Louvain communities with >= min_comm members; empty if none (or graph too big)."""
    lab = M.communities(graph)
    if lab is None:
        return np.zeros(0)
    lab = np.asarray(lab, dtype=np.int64)
    if len(lab) != graph.n:
        # metrics.communities caches by graph name, so two graphs sharing a name collide
        raise ValueError(f"cached Louvain partition for {graph.name!r} has {len(lab)} labels for "
                         f"{graph.n} nodes: graph names must be unique")
    sizes = np.bincount(lab)
    hit = np.bincount(lab, weights=node_prob, minlength=len(sizes))
    ok = sizes >= max(int(min_comm), 1)
    return hit[ok] / sizes[ok]


def evaluate_v2(graph, seeds, n_sims=2000, seed=12345, communities=None, min_comm=10, return_sims=False):
    """Forward-MC metrics of one seed set.

    Keys: influence, influence_se, rates (present labelled groups), disparity, min_rate, max_rate,
    heldout_disparity, heldout_worst_rel (means over held-out attributes; NaN if there are none),
    comm_disparity, comm_worst_rel, comm_q10_rel (NaN if no community qualifies), n_comm_eval,
    and sims (per-simulation spread, float64) when return_sims. "_rel" values are divided by the
    population mean activation probability, as in v1.

    communities: list of node-id arrays (ground truth; overlap allowed; every non-empty one is
    used, so filter by size beforehand if wanted). None falls back to Louvain with >= min_comm."""
    node_prob, grp = simulate(graph, seeds, n_sims, seed)
    totals = grp.sum(1, dtype=np.float64)
    present = graph.present_groups
    rates = grp[:, present].mean(0) / graph.group_sizes[present]
    mean_exp = max(float(node_prob.mean()), 1e-12)
    ns = len(totals)
    out = {
        "influence": float(totals.mean()),
        "influence_se": float(totals.std(ddof=1) / np.sqrt(ns)) if ns > 1 else NAN,
        "rates": rates.tolist(),
        "disparity": _gap(rates),
        "min_rate": float(rates.min()) if len(rates) else NAN,
        "max_rate": float(rates.max()) if len(rates) else NAN,
    }

    # held-out attributes: everything stored on the graph except the one optimised for
    label = graph.meta.get("attribute")
    disp_h, worst_h = [], []
    for a, groups in (graph.meta.get("attrs") or {}).items():
        groups = np.asarray(groups)
        # an array identical to graph.groups is the labelled attribute under a stale meta name
        if a == label or np.array_equal(groups, graph.groups):
            continue
        r = _attr_rates(node_prob, groups)
        if len(r):
            disp_h.append(_gap(r))
            worst_h.append(float(r.min()) / mean_exp)
    out["heldout_disparity"] = float(np.mean(disp_h)) if disp_h else NAN
    out["heldout_worst_rel"] = float(np.mean(worst_h)) if worst_h else NAN

    # communities nobody labelled: ground truth if given, else Louvain (v1 empty-community case -> NaN)
    r = _louvain_rates(graph, node_prob, min_comm) if communities is None else _community_rates(node_prob, communities)
    if len(r):
        out["comm_disparity"] = _gap(r)
        out["comm_worst_rel"] = float(r.min()) / mean_exp
        out["comm_q10_rel"] = float(np.quantile(r, 0.1)) / mean_exp
    else:
        out["comm_disparity"] = out["comm_worst_rel"] = out["comm_q10_rel"] = NAN
    out["n_comm_eval"] = int(len(r))
    if return_sims:
        out["sims"] = totals
    return out
