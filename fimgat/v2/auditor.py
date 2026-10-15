"""Hive-mind auditor: the most under- and over-exposed cohesive communities around a seed set
(docs/design_v2.md section 5).

The auditor is the adversary of the improvement phase. It proposes communities whose exposure
deviates most from the graph mean, and the verifier then refuses moves that widen the gap across
them. Communities come only from local clustering of the undirected graph (v1
`features.undirected_csr`) on the current exposure field; the auditor never touches the evaluation
partition (`metrics.communities`) or any ground truth (leakage rule).

For each start node, an Andersen-Chung-Lang push computes the lazy-walk personalised PageRank
approximation p with residual r (teleport alpha, eps = 1/(20 max_size)). Every node the push
touched is swept in decreasing (p + alpha r)/deg. Since ppr = p + pr(r) >= p + alpha r, this is
still a lower bound on the PageRank, but unlike p alone it also ranks nodes the push reached
without pushing: with eps tied to max_size, a block whose volume is near 1/eps (any block with
mean degree above about 1) leaves many of its own members with p = 0. A prefix C qualifies if
min_size <= |C| <= max_size and conductance(C) <= max_conductance (both tracked incrementally),
and qualifying prefixes are scored by the scan statistic sqrt(|C|) (mean_C(s) - mean_V(s)) with
s = 1 - exposure ("under") or s = exposure ("over").

Cost per start: every push at u adds at least alpha*eps*deg(u) to |p|_1 <= 1, so the push does
O(1/(eps alpha)) edge work and touches O(1/(eps alpha)) nodes, independent of n; the sweep is
linear in the volume of the swept prefixes. The O(n) workspaces are allocated once per call.
"""
import math

import numpy as np
from numba import njit

from ..features import undirected_csr
from .config import MAX_CONDUCTANCE, MIN_COMMUNITY
from .objective import to_numpy


# ------------------------------------------------------------ undirected structure (cached)

def _structure(graph):
    """(ptr, nbr, src, deg) of the undirected, de-duplicated graph, cached on the graph object."""
    cached = getattr(graph, "_v2_ucsr", None)
    if cached is None or cached[0].shape[0] != graph.n + 1:
        ptr, nbr, src = undirected_csr(graph)
        ptr = np.ascontiguousarray(ptr, dtype=np.int64)
        cached = (ptr, np.ascontiguousarray(nbr, dtype=np.int64), np.ascontiguousarray(src, dtype=np.int64),
                  np.diff(ptr))
        graph._v2_ucsr = cached
    return cached


def conductance(graph, nodes):
    """cut(S, V\\S) / min(vol(S), vol(V\\S)) on the undirected, unweighted, de-duplicated graph;
    1.0 if either volume is 0. Self-loops count toward volume, never toward the cut."""
    ptr, nbr, _, deg = _structure(graph)
    S = np.unique(np.asarray(nodes, dtype=np.int64).ravel())
    if len(S) == 0:
        return 1.0
    cnt = deg[S]
    vol = int(cnt.sum())
    den = min(vol, int(ptr[-1]) - vol)
    if den <= 0:
        return 1.0
    in_s = np.zeros(graph.n, dtype=bool)
    in_s[S] = True
    starts = ptr[S]
    arcs = np.repeat(starts - (np.cumsum(cnt) - cnt), cnt) + np.arange(vol)     # arcs leaving S's nodes
    cut = int(np.count_nonzero(~in_s[nbr[arcs]]))
    return cut / den


def node_exposure(est, root_exposure):
    """Exposure of every node (n,): roots keep their own value, a non-root gets the mean over its
    root neighbours (undirected), or the global root mean if it has none."""
    a = to_numpy(root_exposure).ravel()
    if a.shape[0] != est.n_roots:
        raise ValueError(f"root_exposure has {a.shape[0]} entries, the estimator has {est.n_roots} roots")
    if est.all_roots:
        return a.copy()
    n = est.graph.n
    cached = getattr(est, "_v2_nodexp", None)
    if cached is None:
        _, nbr, src, _ = _structure(est.graph)
        roots = np.asarray(est.roots, dtype=np.int64)
        is_root = np.zeros(n, dtype=bool)
        is_root[roots] = True
        keep = is_root[nbr]
        cached = (src[keep], np.searchsorted(roots, nbr[keep]), np.bincount(src[keep], minlength=n), roots)
        est._v2_nodexp = cached
    src_k, root_pos, cnt, roots = cached
    sums = np.bincount(src_k, weights=a[root_pos], minlength=n)
    fill = float(a.mean()) if len(a) else 0.0
    out = np.where(cnt > 0, sums / np.maximum(cnt, 1), fill)
    out[roots] = a
    return out


# ------------------------------------------------------------ numba kernels

@njit(cache=True)
def _acl_push(ptr, nbr, deg, start, alpha, eps, p, r, seen, inq, queue, touched):
    """Andersen-Chung-Lang push for the lazy-walk PPR from `start` (deg[start] > 0).

    p, r (float64) and seen, inq (bool) are size-n workspaces that must be zero / False on entry;
    inq is False again on exit. queue is a size-n ring buffer. The ids of every node that received
    residual are written to touched[:nt]; nt is returned.
    """
    n = deg.shape[0]
    half = 0.5 * (1.0 - alpha)
    r[start] = 1.0
    seen[start] = True
    touched[0] = start
    nt = 1
    queue[0] = start
    inq[start] = True
    head = 0
    tail = 1
    if tail == n:
        tail = 0
    size = 1
    while size > 0:
        u = queue[head]
        head += 1
        if head == n:
            head = 0
        size -= 1
        inq[u] = False
        du = deg[u]
        ru = r[u]
        if du == 0 or ru < eps * du:
            continue
        p[u] += alpha * ru
        r[u] = half * ru
        share = half * ru / du
        for e in range(ptr[u], ptr[u + 1]):
            w = nbr[e]
            if not seen[w]:
                seen[w] = True
                touched[nt] = w
                nt += 1
            r[w] += share
            if not inq[w] and r[w] >= eps * deg[w]:
                inq[w] = True
                queue[tail] = w
                tail += 1
                if tail == n:
                    tail = 0
                size += 1
        if not inq[u] and r[u] >= eps * du:
            inq[u] = True
            queue[tail] = u
            tail += 1
            if tail == n:
                tail = 0
            size += 1
    return nt


@njit(cache=True)
def _sweep(order, ptr, nbr, deg, score, total_vol, min_size, max_size, max_cond, mean_v, in_s):
    """Best prefix of `order` under the scan statistic, among prefixes with min_size <= |C| <= max_size
    and conductance <= max_cond. When v joins S: cut += deg(v) - 2 * (#neighbours already in S)
    (self-loops never cut), vol += deg(v). in_s is a False size-n workspace, restored on exit.
    Returns (length, statistic); length 0 if no prefix qualifies."""
    m = min(order.shape[0], max_size)
    cut = 0
    vol = 0
    ssum = 0.0
    best = -np.inf
    best_len = 0
    for i in range(m):
        v = order[i]
        in_s[v] = True
        c_in = 0
        c_self = 0
        for e in range(ptr[v], ptr[v + 1]):
            w = nbr[e]
            if w == v:
                c_self += 1
            elif in_s[w]:
                c_in += 1
        cut += deg[v] - 2 * c_in - c_self
        vol += deg[v]
        ssum += score[v]
        size = i + 1
        if size >= min_size:
            den = min(vol, total_vol - vol)
            phi = cut / den if den > 0 else 1.0
            if phi <= max_cond:
                stat = np.sqrt(size) * (ssum / size - mean_v)
                if stat > best:
                    best = stat
                    best_len = size
    for i in range(m):
        in_s[order[i]] = False
    return best_len, best


# ------------------------------------------------------------ search

def _starts(score, deg, n_starts, seed):
    """Top ceil(n_starts/2) nodes by score, plus n_starts//2 seeded draws from the rest of the top 5%
    (a pool of at least n_starts nodes). Isolated nodes cannot start a community and are skipped."""
    cand = np.flatnonzero(deg > 0)
    if len(cand) == 0 or n_starts <= 0:
        return np.zeros(0, dtype=np.int64)
    cand = cand[np.argsort(-score[cand], kind="stable")]
    n_rand = int(n_starts) // 2
    n_top = min(int(n_starts) - n_rand, len(cand))
    n_pool = min(len(cand), max((len(cand) + 19) // 20, int(n_starts)))
    pool = cand[n_top:max(n_top, n_pool)]
    k = min(n_rand, len(pool))
    rnd = np.random.default_rng(seed).choice(pool, k, replace=False) if k > 0 else pool[:0]
    return np.concatenate([cand[:n_top], rnd]).astype(np.int64)


def find_community(graph, exposure, side="under", min_size=MIN_COMMUNITY, max_conductance=MAX_CONDUCTANCE,
                   seed=0, alpha=0.15, n_starts=8, max_size=None):
    """Most under- ("under") or over-exposed ("over") cohesive community: sorted node ids, or None.

    exposure: (n,) node exposure in [0, 1] (see node_exposure). Deterministic for a given seed.
    """
    n = graph.n
    e = np.ascontiguousarray(to_numpy(exposure).ravel())
    if e.shape[0] != n:
        raise ValueError(f"exposure has {e.shape[0]} entries, the graph has {n} nodes")
    if side == "under":
        score = 1.0 - e
    elif side == "over":
        score = e.copy()
    else:
        raise ValueError(f"side must be 'under' or 'over', not {side!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    max_size = n // 2 if max_size is None else int(max_size)
    min_size = max(int(min_size), 1)
    if max_size < 1 or min_size > max_size or min_size > n:
        return None
    ptr, nbr, _, deg = _structure(graph)
    starts = _starts(score, deg, n_starts, seed)
    if len(starts) == 0:
        return None

    eps = 1.0 / (20.0 * max_size)
    total_vol = int(ptr[-1])
    mean_v = float(score.mean())
    p, r = np.zeros(n), np.zeros(n)
    seen, inq, in_s = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    queue, touched = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    best, best_c = -np.inf, None
    # UNVERIFIED: not run; the planted-block spec tests were traced by hand
    for s0 in starts:
        nt = _acl_push(ptr, nbr, deg, int(s0), float(alpha), eps, p, r, seen, inq, queue, touched)
        t = touched[:nt].copy()
        order = t[np.argsort(-(p[t] + alpha * r[t]) / deg[t], kind="stable")]
        p[t] = 0.0
        r[t] = 0.0
        seen[t] = False
        size, stat = _sweep(order, ptr, nbr, deg, score, total_vol, min_size, max_size,
                            float(max_conductance), mean_v, in_s)
        if size > 0 and stat > best:
            best, best_c = stat, np.sort(order[:size])
    return best_c


def audit(graph, exposure, min_size=MIN_COMMUNITY, max_conductance=MAX_CONDUCTANCE, seed=0,
          alpha=0.15, n_starts=8, max_size=None):
    """(under, over): the most under- and over-exposed cohesive communities (each an id array or None)."""
    kw = dict(min_size=min_size, max_conductance=max_conductance, seed=seed, alpha=alpha,
              n_starts=n_starts, max_size=max_size)
    return (find_community(graph, exposure, side="under", **kw),
            find_community(graph, exposure, side="over", **kw))


# ------------------------------------------------------------ multi-scale audit (docs/theory_v2.md, W1)

def size_bands(min_size, max_size, ratio=2.0):
    """Geometric size bands [(lo, hi), ...] covering [min_size, max_size], each with hi <= ratio * lo.

    The scan statistic sqrt(|C|) * deviation favours large communities: over sizes [m, M] the winner's
    deviation can be as small as sqrt(m / M) of the most extreme community's. Within one band it is at
    least 1 / sqrt(ratio) of the band's most extreme (Proposition W1.2)."""
    if ratio <= 1.0:
        raise ValueError("ratio must exceed 1")
    lo, top, out = max(int(min_size), 1), int(max_size), []
    while lo <= top:
        hi = max(lo, min(top, int(math.floor(lo * float(ratio)))))
        out.append((lo, hi))
        lo = hi + 1
    return out


def audit_bands(graph, exposure, bands, max_conductance=MAX_CONDUCTANCE, seed=0, alpha=0.15, n_starts=8):
    """Multi-scale audit: the most under- and the most over-exposed cohesive community within every size
    band (lo, hi). Returns the communities found (id arrays), band by band, under before over."""
    out = []
    for i, (lo, hi) in enumerate(bands):
        for side in ("under", "over"):
            c = find_community(graph, exposure, side=side, min_size=lo, max_conductance=max_conductance,
                               seed=int(seed) + i, alpha=alpha, n_starts=n_starts, max_size=hi)
            if c is not None and len(c):
                out.append(c)
    return out
