"""Dataset loaders used by v2 (fimgat/v2/datasets.py builds on them). v1's SPA-500 loaders were removed on
2026-10-10: v2 dropped SPA (design_v2.md section 8).

Pokec      Slovak social network with real region / gender / age (Takac & Zabovsky
           2012). Subgraphs are built once by experiments/build_pokec.py.
DC-SBM     Synthetic degree-corrected stochastic block model with a controllable
           homophily level, for controlled studies.
"""
import os

import numpy as np

from .graph import from_edges

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_pokec(name, attribute="region"):
    """Load a Pokec subgraph produced by experiments/build_pokec.py (weighted cascade)."""
    z = np.load(os.path.join(ROOT, "pokec", f"{name}.npz"), allow_pickle=True)
    groups = z[attribute]
    names = list(z[f"{attribute}_names"])
    attrs = {a: z[a] for a in ("region", "gender", "age")}
    return from_edges(name, int(z["n"]), z["edges"], groups, names, undirected=False,
                      weighted_cascade=True,
                      meta={"dataset": "pokec", "attribute": attribute, "attrs": attrs})


def make_dcsbm(n=1000, minority=0.3, homophily=0.8, avg_degree=8, gamma=2.5,
               prob=0.08, seed=0):
    """Degree-corrected SBM with two groups.

    Each node gets a power-law propensity theta (exponent gamma), so there are
    hubs. Each edge picks its first endpoint in proportion to theta; with
    probability `homophily` the second endpoint comes from the same group,
    otherwise from the other group, again in proportion to theta.
    """
    rng = np.random.default_rng(seed)
    groups = (rng.random(n) < minority).astype(np.int64)      # 1 = minority
    theta = (1 - rng.random(n)) ** (-1 / (gamma - 1))
    theta = np.minimum(theta, np.sqrt(n))
    members = [np.flatnonzero(groups == g) for g in (0, 1)]
    member_p = [theta[m] / theta[m].sum() for m in members]
    m_edges = n * avg_degree // 2
    u = rng.choice(n, size=m_edges, p=theta / theta.sum())
    same = rng.random(m_edges) < homophily
    target_group = np.where(same, groups[u], 1 - groups[u])
    v = np.empty(m_edges, dtype=np.int64)
    for g in (0, 1):
        sel = target_group == g
        v[sel] = rng.choice(members[g], size=sel.sum(), p=member_p[g])
    return from_edges(f"sbm_h{homophily:.2f}_s{seed}", n, np.stack([u, v], 1), groups,
                      ["majority", "minority"], prob=prob, undirected=True,
                      meta={"dataset": "sbm", "homophily": homophily, "attribute": "group"})

