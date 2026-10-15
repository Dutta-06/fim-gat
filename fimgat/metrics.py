"""Evaluation metrics, computed from forward Monte Carlo simulation only.

All group quantities are *rates* (fraction of the group reached), so group
sizes don't masquerade as unfairness. Besides the attribute the method was
optimised for, every seed set is also scored on
  * every other sensitive attribute stored on the graph (held-out attributes), and
  * Louvain communities (unlabelled groups; "information deserts"),
which the methods never see.
"""
import os

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance

from .diffusion import monte_carlo

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
_COMM = {}


def communities(graph, max_nodes=300_000):
    """Louvain communities of the undirected graph (seed 0), cached on disk. None if too big."""
    if graph.n > max_nodes:
        return None
    key = graph.name.split(":")[0]
    if key in _COMM:
        return _COMM[key]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{key}_louvain.npy")
    if os.path.exists(path):
        lab = np.load(path)
    else:
        und = nx.Graph()
        und.add_nodes_from(range(graph.n))
        und.add_edges_from(zip(graph.src.tolist(), graph.dst.tolist()))
        lab = np.zeros(graph.n, dtype=np.int64)
        for c, members in enumerate(nx.community.louvain_communities(und, seed=0)):
            lab[list(members)] = c
        np.save(path, lab)
    _COMM[key] = lab
    return lab


def _rates(node_prob, groups, min_size=1):
    sizes = np.bincount(groups)
    ok = sizes >= min_size
    return (np.bincount(groups, weights=node_prob, minlength=len(sizes))[ok] / sizes[ok]), sizes[ok]


def evaluate(graph, seeds, n_sims=5000, seed=12345, desert_frac=0.25):
    seeds = np.asarray(seeds, dtype=np.int64)
    assert len(np.unique(seeds)) == len(seeds), "duplicate seeds"
    node_prob, grp = monte_carlo(graph, seeds, n_sims=n_sims, seed=seed)
    present = graph.present_groups
    sizes = graph.group_sizes[present]
    per_sim_rates = grp[:, present] / sizes                      # (sims, G')
    rates = per_sim_rates.mean(0)
    totals = grp.sum(1)
    mean_exp = node_prob.mean()
    thresh = desert_frac * mean_exp

    # exposure-distribution fairness: per-node activation probability by group
    dists = [node_prob[graph.groups == g] for g in np.flatnonzero(present)]
    w1 = max((wasserstein_distance(a, b) for i, a in enumerate(dists) for b in dists[i + 1:]),
             default=0.0)
    desert = np.array([np.mean(d < thresh) for d in dists])

    out = {
        "influence": float(totals.mean()),
        "influence_se": float(totals.std(ddof=1) / np.sqrt(len(totals))),
        "min_rate": float(rates.min()),
        "max_rate": float(rates.max()),
        "disparity": float(rates.max() - rates.min()),
        "fair_ratio": float(rates.min() / max(rates.max(), 1e-12)),
        "min_rate_se": float(per_sim_rates[:, rates.argmin()].std(ddof=1) / np.sqrt(len(totals))),
        "exposure_w1": float(w1),
        "exposure_w1_rel": float(w1 / max(mean_exp, 1e-12)),
        "worst_desert": float(desert.max()),
        "desert_gap": float(desert.max() - desert.min()),
        "rates": rates.tolist(),
    }
    # held-out attributes: worst-group rate relative to the population mean exposure
    attrs = graph.meta.get("attrs", {})
    rel = []
    for a, groups in attrs.items():
        r, _ = _rates(node_prob, groups)
        out[f"worst_{a}"] = float(r.min())
        if a != graph.meta.get("attribute"):
            rel.append(r.min() / max(mean_exp, 1e-12))
    out["heldout_worst_rel"] = float(np.mean(rel)) if rel else np.nan
    # unlabelled communities (Louvain, >= 10 members)
    comm = communities(graph)
    if comm is not None:
        r, s = _rates(node_prob, comm, min_size=10)
        if len(r) == 0:                  # no community with >= 10 members (fragmented graphs)
            out["worst_community_rel"] = out["community_q10_rel"] = out["desert_communities"] = np.nan
        else:
            out["worst_community_rel"] = float(r.min() / max(mean_exp, 1e-12))
            out["community_q10_rel"] = float(np.quantile(r, 0.1) / max(mean_exp, 1e-12))
            out["desert_communities"] = float(np.mean(r < thresh))
    return out
