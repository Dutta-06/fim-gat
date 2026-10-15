"""Shared fixtures for the FIM-GAT test suites (see tests/README.md)."""
import itertools
import json
import os
import sys
import time
from collections import deque

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
OLD_PROJECT = r"C:\Users\Odwitiyo\Desktop\xAI Project\Baseline_Test"      # read-only, never modified
REPORTS = os.path.join(ROOT, "tests", "_reports")

from fimgat.graph import from_edges  # noqa: E402

BUDGETS = (25, 50, 100, 150, 200)          # agreed 2026-09-27: every dataset, every method


# ----------------------------------------------------------------- isolation

@pytest.fixture(autouse=True, scope="session")
def _isolate_community_cache(tmp_path_factory):
    """metrics.evaluate caches Louvain partitions under data/cache; tests must not write there."""
    import shutil
    import fimgat.metrics as M
    old = M.CACHE
    tmp = str(tmp_path_factory.mktemp("louvain_cache"))
    if os.path.isdir(old):                                   # reuse existing partitions read-only
        for f in os.listdir(old):
            shutil.copy2(os.path.join(old, f), tmp)
    M.CACHE = tmp
    M._COMM.clear()
    yield
    M.CACHE = old
    M._COMM.clear()


# ----------------------------------------------------------------- devices

def cuda_ok():
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device():
    return "cuda" if cuda_ok() else "cpu"


requires_cuda = pytest.mark.skipif(not cuda_ok(), reason="needs CUDA")


# ----------------------------------------------------------------- exact Independent Cascade

def exact_ic(graph, seeds):
    """Exact activation probability of every node, by enumerating all 2^m live-edge worlds.
    Only for tiny graphs (m <= 16)."""
    m = graph.m
    assert m <= 16, "exact enumeration is exponential in the number of edges"
    seeds = list(seeds)
    prob = np.zeros(graph.n)
    for live in itertools.product((0, 1), repeat=m):
        w = 1.0
        adj = [[] for _ in range(graph.n)]
        for e, on in enumerate(live):
            pe = graph.prob[e]
            w *= pe if on else (1.0 - pe)
            if on:
                adj[graph.src[e]].append(graph.dst[e])
        if w == 0.0:
            continue
        seen = np.zeros(graph.n, bool)
        seen[seeds] = True
        q = deque(seeds)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        prob += w * seen
    return prob


def group_rates(graph, node_prob):
    sizes = np.maximum(graph.group_sizes, 1)
    return np.bincount(graph.groups, weights=node_prob, minlength=graph.n_groups) / sizes


# ----------------------------------------------------------------- tiny and small graphs

@pytest.fixture
def tiny_graph():
    """8 nodes, 12 directed edges with mixed probabilities, 3 groups (one of size 1)."""
    edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5], [5, 3], [1, 6],
                      [6, 7], [7, 6], [0, 5], [4, 1]])
    p = dict(zip(map(tuple, edges.tolist()), [0.5, 0.4, 0.3, 0.6, 0.2, 0.7, 0.5, 0.3, 0.8, 0.4, 0.25, 0.35]))
    g = from_edges("tiny", 8, edges, groups=[0, 0, 0, 1, 1, 1, 2, 1], group_names=["a", "b", "c"],
                   undirected=False, prob=0.3)
    # from_edges sorts/dedups edges, so assign probabilities by (src, dst), not by input order
    g.prob[:] = [p[(int(s), int(d))] for s, d in zip(g.src, g.dst)]
    return g


@pytest.fixture
def tiny_wc_graph():
    """Tiny undirected graph under weighted cascade (the v2 diffusion model)."""
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [3, 4], [4, 5], [5, 6]])
    return from_edges("tiny_wc", 7, edges, groups=[0, 0, 1, 1, 1, 0, 1], group_names=["a", "b"],
                      undirected=True, weighted_cascade=True)


@pytest.fixture(scope="session")
def sbm_graph():
    """2,000-node DC-SBM with weighted cascade and 3 synthetic groups (fast, many cascades)."""
    from fimgat.data import make_dcsbm
    g = make_dcsbm(n=2000, homophily=0.8, seed=11)
    wc = from_edges("sbm_wc", g.n, np.stack([g.src, g.dst], 1), g.groups, g.group_names,
                    undirected=False, weighted_cascade=True)
    rng = np.random.default_rng(0)
    three = np.where(wc.groups == 1, 2, rng.integers(0, 2, wc.n))       # minority -> group 2
    return wc.with_groups(three, ["x", "y", "z"])


def degenerate_graphs():
    """Adversarial topologies for stress tests: (name, graph)."""
    out = []
    n = 60
    star = np.array([[0, i] for i in range(1, n)])
    out.append(("star", from_edges("star", n, star, np.r_[0, np.ones(n - 1, int)], ["hub", "leaf"],
                                   undirected=True, weighted_cascade=True)))
    path = np.array([[i, i + 1] for i in range(n - 1)])
    out.append(("path", from_edges("path", n, path, np.arange(n) % 2, ["even", "odd"],
                                   undirected=True, weighted_cascade=True)))
    two = np.r_[[[i, i + 1] for i in range(29)], [[i, i + 1] for i in range(30, 59)]]
    out.append(("disconnected", from_edges("disc", n, two, (np.arange(n) >= 30).astype(int), ["L", "R"],
                                           undirected=True, weighted_cascade=True)))
    iso = np.array([[i, i + 1] for i in range(39)])                  # nodes 40..59 isolated
    out.append(("isolated_nodes", from_edges("iso", n, iso, (np.arange(n) >= 40).astype(int), ["conn", "iso"],
                                             undirected=True, weighted_cascade=True)))
    rng = np.random.default_rng(3)
    er = rng.integers(0, n, (240, 2))
    out.append(("single_group", from_edges("one", n, er, np.zeros(n, int), ["all"],
                                           undirected=True, weighted_cascade=True)))
    out.append(("singleton_groups", from_edges("sing", n, er, np.arange(n), [str(i) for i in range(n)],
                                               undirected=True, weighted_cascade=True)))
    grp = np.r_[np.zeros(55, int), np.ones(5, int)]
    dirE = np.array([[55 + i, rng.integers(0, 55)] for i in range(5)] + er[er[:, 1] < 55].tolist())
    out.append(("unreachable_group", from_edges("unreach", n, dirE, grp, ["main", "sources"],
                                                undirected=False, weighted_cascade=True)))
    return out


# ----------------------------------------------------------------- data availability

def data_path(*parts):
    return os.path.join(DATA, *parts)


def skip_unless_exists(path, why):
    if not os.path.exists(path):
        pytest.skip(f"{why}: {path} not present yet")


# ----------------------------------------------------------------- stress report

@pytest.fixture(scope="session")
def stress_report():
    """Append timing / memory records to tests/_reports/stress_results.jsonl (one run per line)."""
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, "stress_results.jsonl")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    gpu = torch.cuda.get_device_name(0) if cuda_ok() else "cpu"

    def record(**row):
        row = {"when": stamp, "gpu": gpu, **row}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    return record


class GpuPeak:
    """Context manager measuring peak allocated/reserved GPU memory (GB)."""

    def __enter__(self):
        self.alloc = self.reserved = 0.0
        if cuda_ok():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if cuda_ok():
            torch.cuda.synchronize()
            self.alloc = torch.cuda.max_memory_allocated() / 2**30
            self.reserved = torch.cuda.max_memory_reserved() / 2**30
        self.secs = time.time() - self.t0
        return False


def gpu_total_gb():
    return torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda_ok() else 0.0
