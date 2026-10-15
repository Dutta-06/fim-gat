"""Exact incremental verifier state on the estimator's RR sets (docs/v2_interfaces.md section 5).

For a binary seed set, CoverState keeps how many seeds each RR set contains (counts), the covered
sets per root (root_cov) and the covered sets per group (hits, integers). Rates follow the
ExposureEstimator definition, rate_g = hits_g / (R * root_counts[g]) and total = rates @ group_sizes,
in float64. They are recomputed from the integer hits after every update, so they never drift.
swap_delta forms the post-swap values with the same expressions and returns their differences, so
total + d_total is the post-swap total exactly whenever the two totals are within a factor of 2
(Sterbenz), and to one rounding otherwise. A swap u -> v touches only the sets that contain u or v:
    lost    sets containing u with count 1 that do not contain v
    gained  sets containing v with count 0
    (sets containing both keep their coverage)
The static structures (node -> sets inverse, per-set group, weight and root, node -> root position)
are built once per estimator and cached on it as est._v2_cover. The loops over RR sets use numba.

est.evaluate_set scores a set covered once as 1 - 1.01e-6 rather than 1, because it clamps x to
float32(1 - 1e-6) before taking the log. Its rates and total therefore sit up to about 1.01e-6
(relative) below the exact values here, plus float32 rounding.
"""
# UNVERIFIED: nothing here has been executed. The numba typing was checked by hand only, and the
# spec tests compare totals with evaluate_set at rel=1e-6, which is close to the clamp bias above.
import numpy as np
from numba import njit

from .objective import disparity


# ------------------------------------------------------------------ numba kernels

@njit(cache=True)
def _invert(ptr, nodes, n):
    """node -> RR sets containing it (CSR). Sets are visited in order, so each row is sorted."""
    node_ptr = np.zeros(n + 1, dtype=np.int64)
    for t in range(nodes.shape[0]):
        node_ptr[nodes[t] + 1] += 1
    for v in range(n):
        node_ptr[v + 1] += node_ptr[v]
    cur = node_ptr[:n].copy()
    node_sets = np.empty(nodes.shape[0], dtype=np.int64)
    for j in range(ptr.shape[0] - 1):
        for t in range(ptr[j], ptr[j + 1]):
            v = nodes[t]
            node_sets[cur[v]] = j
            cur[v] += 1
    return node_ptr, node_sets


@njit(cache=True)
def _build_counts(node_ptr, node_sets, seeds, n_sets):
    counts = np.zeros(n_sets, dtype=np.int32)
    for i in range(seeds.shape[0]):
        u = seeds[i]
        for t in range(node_ptr[u], node_ptr[u + 1]):
            counts[node_sets[t]] += 1
    return counts


@njit(cache=True)
def _add_node(node_ptr, node_sets, set_group, R, v, counts, hits, root_cov):
    for t in range(node_ptr[v], node_ptr[v + 1]):
        j = node_sets[t]
        if counts[j] == 0:
            hits[set_group[j]] += 1
            root_cov[j // R] += 1
        counts[j] += 1


@njit(cache=True)
def _remove_node(node_ptr, node_sets, set_group, R, u, counts, hits, root_cov):
    for t in range(node_ptr[u], node_ptr[u + 1]):
        j = node_sets[t]
        counts[j] -= 1
        if counts[j] == 0:
            hits[set_group[j]] -= 1
            root_cov[j // R] -= 1


@njit(cache=True)
def _loss_counts(seeds, counts, node_ptr, node_sets, set_group, G):
    """(k, G) number of sets, per group, that only seed i covers."""
    out = np.zeros((seeds.shape[0], G))
    for i in range(seeds.shape[0]):
        u = seeds[i]
        for t in range(node_ptr[u], node_ptr[u + 1]):
            j = node_sets[t]
            if counts[j] == 1:
                out[i, set_group[j]] += 1.0
    return out


@njit(cache=True)
def _gain_counts(ptr, nodes, set_group, counts, n, G):
    """(n, G) number of uncovered sets, per group, containing each node (one pass over all nnz)."""
    out = np.zeros((n, G))
    for j in range(ptr.shape[0] - 1):
        if counts[j] != 0:
            continue
        g = set_group[j]
        for t in range(ptr[j], ptr[j + 1]):
            out[nodes[t], g] += 1.0
    return out


@njit(cache=True)
def _overlap_counts(seeds, counts, node_ptr, node_sets, ptr, nodes, set_group, n, G):
    """Sets only seed i covers that also contain v, per group, aggregated per (i, v).

    Returns (oi, ov, oc (E, G)), grouped by i in increasing order. The other members of a set that
    only u covers are never seeds, so every v here is a non-seed.
    """
    k = seeds.shape[0]
    stamp = np.empty(n, dtype=np.int64)
    stamp[:] = -1
    E = 0
    for i in range(k):
        u = seeds[i]
        for t in range(node_ptr[u], node_ptr[u + 1]):
            j = node_sets[t]
            if counts[j] != 1:
                continue
            for q in range(ptr[j], ptr[j + 1]):
                v = nodes[q]
                if v != u and stamp[v] != i:
                    stamp[v] = i
                    E += 1
    oi = np.empty(E, dtype=np.int64)
    ov = np.empty(E, dtype=np.int64)
    oc = np.zeros((E, G))
    slot = np.empty(n, dtype=np.int64)
    stamp[:] = -1
    e = 0
    for i in range(k):
        u = seeds[i]
        for t in range(node_ptr[u], node_ptr[u + 1]):
            j = node_sets[t]
            if counts[j] != 1:
                continue
            g = set_group[j]
            for q in range(ptr[j], ptr[j + 1]):
                v = nodes[q]
                if v == u:
                    continue
                if stamp[v] != i:
                    stamp[v] = i
                    slot[v] = e
                    oi[e] = i
                    ov[e] = v
                    e += 1
                oc[slot[v], g] += 1.0
    return oi, ov, oc


# ------------------------------------------------------------------ static structures

class CoverIndex:
    """Static RR-set structures shared by every CoverState on one estimator (est._v2_cover)."""

    def __init__(self, est):
        g = est.graph
        self.n, self.G, self.R = int(g.n), int(g.n_groups), int(est.R)
        self.n_roots = int(est.n_roots)
        self.ptr = np.ascontiguousarray(est.ptr, dtype=np.int64)
        self.nodes = np.ascontiguousarray(est.nodes, dtype=np.int64)
        self.n_sets = len(self.ptr) - 1
        self.nnz = len(self.nodes)
        if self.n_sets != self.n_roots * self.R:
            raise ValueError(f"expected n_roots * R = {self.n_roots * self.R} RR sets, got {self.n_sets}")
        self.node_ptr, self.node_sets = _invert(self.ptr, self.nodes, self.n)
        rc = np.asarray(est.root_counts, dtype=np.float64)
        self.gw = 1.0 / (self.R * np.maximum(rc, 1.0))                  # rate of one covered set, per group
        self.root_groups = np.asarray(est.root_groups_np, dtype=np.int64)
        self.set_group = np.repeat(self.root_groups, self.R)
        self.set_w = self.gw[self.set_group]
        self.set_root = np.arange(self.n_sets, dtype=np.int64) // self.R
        self.root_index = np.full(self.n, -1, dtype=np.int64)
        self.root_index[np.asarray(est.roots, dtype=np.int64)] = np.arange(self.n_roots, dtype=np.int64)
        self.sizes = np.asarray(g.group_sizes, dtype=np.float64)
        self.present = (np.asarray(g.group_sizes) > 0) & (rc > 0)


def cover_index(est):
    """The cached CoverIndex of est, built on first use."""
    ix = getattr(est, "_v2_cover", None)
    if ix is None or ix.n_sets != len(est.ptr) - 1 or ix.nnz != len(est.nodes):
        ix = CoverIndex(est)
        est._v2_cover = ix
    return ix


# ------------------------------------------------------------------ mutable state

class CoverState:
    """Exact RR cover state of a seed set (distinct node ids; order kept in .seeds).

    Fields: est, index (CoverIndex), seeds (list[int]), is_seed (bool (n,)), counts (int32 per RR set),
    root_cov (int64 per root), hits (int64 (G,)), rates (float64 (G,)), total (float), present, sizes.
    """

    def __init__(self, est, seeds):
        ix = cover_index(est)
        arr = np.asarray(seeds, dtype=np.int64).ravel()
        if arr.size and (arr.min() < 0 or arr.max() >= ix.n):
            raise ValueError(f"seed ids must lie in [0, {ix.n})")
        if len(np.unique(arr)) != arr.size:
            raise ValueError("seeds must be distinct")
        self.est, self.index = est, ix
        self.present, self.sizes = ix.present, ix.sizes
        self.seeds = [int(s) for s in arr]
        self.is_seed = np.zeros(ix.n, dtype=np.bool_)
        self.is_seed[arr] = True
        self.counts = _build_counts(ix.node_ptr, ix.node_sets, arr, ix.n_sets)
        covered = self.counts > 0
        self.root_cov = covered.reshape(ix.n_roots, ix.R).sum(axis=1, dtype=np.int64)
        self.hits = np.rint(np.bincount(ix.root_groups, weights=self.root_cov.astype(np.float64),
                                        minlength=ix.G)).astype(np.int64)
        self._refresh()

    def _refresh(self):
        # the same two expressions as swap_delta, so predicted and applied values agree bit for bit
        self.rates = self.hits * self.index.gw
        self.total = float(self.rates @ self.sizes)

    @property
    def k(self):
        return len(self.seeds)

    @property
    def disp(self):
        return disparity(self.rates, self.present)

    # ---------------------------------------------------------- validation

    def _node(self, x):
        x = int(x)
        if not 0 <= x < self.index.n:
            raise ValueError(f"node {x} is outside [0, {self.index.n})")
        return x

    def _check_swap(self, u, v):
        u, v = self._node(u), self._node(v)
        if not self.is_seed[u]:
            raise ValueError(f"node {u} is not a seed")
        if self.is_seed[v]:
            raise ValueError(f"node {v} is already a seed")
        return u, v

    def _sets(self, x):
        ix = self.index
        return ix.node_sets[ix.node_ptr[x]:ix.node_ptr[x + 1]]

    # ---------------------------------------------------------- swaps (read-only)

    def swap_detail(self, u, v):
        """(loss_sets, gain_sets): RR sets that lose / gain coverage if u is swapped out and v in."""
        u, v = self._check_swap(u, v)
        su, sv = self._sets(u), self._sets(v)
        one = su[self.counts[su] == 1]
        loss = one[~np.isin(one, sv, assume_unique=True)]           # node rows are sorted and unique
        gain = sv[self.counts[sv] == 0]
        return loss, gain

    def swap_delta(self, u, v):
        """Exact (d_rates (G,), d_total) of swapping seed u out and non-seed v in."""
        d_rates, d_total, _, _ = self.swap_delta_detail(u, v)
        return d_rates, d_total

    def swap_delta_detail(self, u, v):
        """(d_rates, d_total, loss_sets, gain_sets) of the swap from one swap_detail pass; d_rates and
        d_total are computed exactly as in swap_delta."""
        loss, gain = self.swap_detail(u, v)
        ix = self.index
        d_hits = np.bincount(ix.set_group[gain], minlength=ix.G) - np.bincount(ix.set_group[loss], minlength=ix.G)
        new_rates = (self.hits + d_hits) * ix.gw
        new_total = float(new_rates @ self.sizes)
        return new_rates - self.rates, new_total - self.total, loss, gain

    def root_delta_from(self, loss, gain):
        """(root positions, net change in root_cov) for given lost / gained RR sets; zero-net roots left out."""
        ix = self.index
        r = np.concatenate([ix.set_root[gain], ix.set_root[loss]])
        if r.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        d = np.concatenate([np.ones(len(gain), dtype=np.int64), -np.ones(len(loss), dtype=np.int64)])
        roots, inv = np.unique(r, return_inverse=True)
        dc = np.zeros(len(roots), dtype=np.int64)
        np.add.at(dc, inv.ravel(), d)
        keep = dc != 0
        return roots[keep].astype(np.int64), dc[keep]

    def root_touch_from(self, sets):
        """(root positions, number of the given RR sets rooted there): the sets whose coverage changes."""
        r = self.index.set_root[np.asarray(sets, dtype=np.int64)]
        if r.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        roots, cnt = np.unique(r, return_counts=True)
        return roots.astype(np.int64), cnt.astype(np.int64)

    def swap_root_delta(self, u, v):
        """(root positions, change in root_cov) of the swap; roots with no net change are left out."""
        loss, gain = self.swap_detail(u, v)
        return self.root_delta_from(loss, gain)

    # ---------------------------------------------------------- updates

    def _kernel_args(self):
        ix = self.index
        return ix.node_ptr, ix.node_sets, ix.set_group, ix.R

    def apply_swap(self, u, v):
        """Swap seed u out and v in; v takes u's position in .seeds."""
        u, v = self._check_swap(u, v)
        i = self.seeds.index(u)
        a = self._kernel_args()
        _add_node(*a, v, self.counts, self.hits, self.root_cov)
        _remove_node(*a, u, self.counts, self.hits, self.root_cov)
        self.seeds[i] = v
        self.is_seed[u], self.is_seed[v] = False, True
        self._refresh()

    def add(self, v):
        """Add a non-seed v (appended to .seeds)."""
        v = self._node(v)
        if self.is_seed[v]:
            raise ValueError(f"node {v} is already a seed")
        _add_node(*self._kernel_args(), v, self.counts, self.hits, self.root_cov)
        self.seeds.append(v)
        self.is_seed[v] = True
        self._refresh()

    def remove(self, u):
        """Remove seed u."""
        u = self._node(u)
        if not self.is_seed[u]:
            raise ValueError(f"node {u} is not a seed")
        _remove_node(*self._kernel_args(), u, self.counts, self.hits, self.root_cov)
        self.seeds.remove(u)
        self.is_seed[u] = False
        self._refresh()

    # ---------------------------------------------------------- scores for proposers and search

    def _seed_array(self):
        return np.asarray(self.seeds, dtype=np.int64)

    def loss_counts(self):
        """(k, G) float64 number of sets per group that only each seed covers (rows follow .seeds)."""
        ix = self.index
        return _loss_counts(self._seed_array(), self.counts, ix.node_ptr, ix.node_sets, ix.set_group, ix.G)

    def gain_counts(self):
        """(n, G) float64 number of uncovered sets per group containing each node (seed rows are 0)."""
        ix = self.index
        return _gain_counts(ix.ptr, ix.nodes, ix.set_group, self.counts, ix.n, ix.G)

    def overlap_counts(self):
        """(oi, ov, oc): for seed index oi[e] and non-seed ov[e], the sets per group (oc[e], (G,))
        that only that seed covers and that also contain ov[e]. Entries are grouped by oi, increasing."""
        ix = self.index
        return _overlap_counts(self._seed_array(), self.counts, ix.node_ptr, ix.node_sets, ix.ptr, ix.nodes,
                               ix.set_group, ix.n, ix.G)

    def removal_loss(self):
        """(k, G) exact rate lost per group by removing each seed (rows follow .seeds)."""
        return self.loss_counts() * self.index.gw

    def gains(self):
        """(n, G) exact rate gained per group by adding each node; rows of seeds are 0."""
        return self.gain_counts() * self.index.gw

    def exposure(self):
        """Per-root exposure a_r = root_cov / R, float64 (n_roots,)."""
        return self.root_cov / float(self.index.R)

    def copy(self):
        """Deep copy of the mutable arrays; the static structures are shared."""
        c = object.__new__(CoverState)
        c.est, c.index, c.present, c.sizes = self.est, self.index, self.present, self.sizes
        c.seeds = list(self.seeds)
        c.is_seed = self.is_seed.copy()
        c.counts = self.counts.copy()
        c.root_cov = self.root_cov.copy()
        c.hits = self.hits.copy()
        c.rates = self.rates.copy()
        c.total = self.total
        return c
