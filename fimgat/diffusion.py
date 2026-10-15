"""Independent Cascade machinery.

* sample_rr_sets     reverse-reachable (RR) sets (Borgs et al. 2014), numba, multithreaded.
* ExposureEstimator  differentiable, unbiased estimate of the activation probability of
                     every root node under a *fractional* seed vector x, i.e. the
                     multilinear extension of IC influence:
                         P(v active | x) = E_R[ 1 - prod_{u in RR(v)} (1 - x_u) ]
                     Evaluated as a gather + index_add over RR-set memberships.
                     On large graphs a group-stratified subset of nodes is used as roots
                     (memory ~ n_roots * R * mean RR size).
* monte_carlo        forward IC simulation with its own random stream, used only for
                     final evaluation, never for training or model selection.
"""
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
import torch
from numba import njit, prange


# ---------------------------------------------------------------- RR sampling

@njit(cache=True, nogil=True)
def _rr_chunk(in_ptr, in_src, in_p, roots, R, seed):
    np.random.seed(seed)
    n = in_ptr.shape[0] - 1
    total = roots.shape[0] * R
    ptr = np.zeros(total + 1, dtype=np.int64)
    cap = max(total * 4, 1024)
    buf = np.empty(cap, dtype=np.int32)
    stamp_of = np.zeros(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int32)
    pos = 0
    s = 0
    for ri in range(roots.shape[0]):
        root = roots[ri]
        for _ in range(R):
            s += 1
            head = 0
            tail = 1
            queue[0] = root
            stamp_of[root] = s
            while head < tail:
                v = queue[head]
                head += 1
                for e in range(in_ptr[v], in_ptr[v + 1]):
                    u = in_src[e]
                    if stamp_of[u] != s and np.random.random() < in_p[e]:
                        stamp_of[u] = s
                        queue[tail] = u
                        tail += 1
            if pos + tail > cap:
                cap = max(cap * 2, pos + tail)
                nb = np.empty(cap, dtype=np.int32)
                nb[:pos] = buf[:pos]
                buf = nb
            buf[pos:pos + tail] = queue[:tail]
            pos += tail
            ptr[s] = pos
    return ptr, buf[:pos].copy()


RR_CHUNKS = 16     # root chunks, each with its own seed; fixed, so a sample is the same on every machine


def sample_rr_sets(graph, roots, R, seed, n_jobs=None):
    """R RR sets for every root, ordered root-major. Returns (ptr, nodes).

    The roots are split into n_jobs (default RR_CHUNKS) chunks, chunk i seeded with seed * 1000 + i, so the sample
    depends only on (graph, roots, R, seed, n_jobs), not on the machine. Before 2026-10-15 the default was
    min(16, NUMBA_NUM_THREADS), which is 16 on any machine with at least 16 numba threads (the samples are unchanged
    there) but fewer chunks, and so other samples, on smaller machines such as a 4-vCPU cloud VM."""
    in_ptr, in_src, in_p = graph.in_csr
    roots = np.asarray(roots, dtype=np.int64)
    n_jobs = n_jobs or RR_CHUNKS
    chunks = [c for c in np.array_split(roots, n_jobs) if len(c)]
    workers = max(1, min(len(chunks), numba.config.NUMBA_NUM_THREADS))
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda ic: _rr_chunk(in_ptr, in_src, in_p, ic[1], R, seed * 1000 + ic[0]),
                            enumerate(chunks)))
    ptrs, nodes, offset = [np.zeros(1, dtype=np.int64)], [], 0
    for p, b in parts:
        ptrs.append(p[1:] + offset)
        nodes.append(b)
        offset += len(b)
    return np.concatenate(ptrs), np.concatenate(nodes).astype(np.int64)


def stratified_roots(graph, max_roots, seed, min_per_group=2000):
    """All nodes if they fit, else a group-stratified sample that keeps small groups whole."""
    if max_roots is None or max_roots >= graph.n:
        return np.arange(graph.n)
    rng = np.random.default_rng(seed)
    out = []
    for g in range(graph.n_groups):
        members = np.flatnonzero(graph.groups == g)
        if len(members) == 0:
            continue
        m = min(len(members), max(min_per_group, int(round(max_roots * len(members) / graph.n))))
        out.append(rng.choice(members, m, replace=False))
    return np.sort(np.concatenate(out))


# ------------------------------------------------------- differentiable estimator

class ExposureEstimator:
    """Differentiable exposure of root nodes under fractional seeds x in [0,1]^n.

        miss_j = prod_{u in RR_j} (1 - x_u)       a_r = 1 - mean_{j rooted at r} miss_j
        rate_g = mean_{roots r in g} a_r          total = sum_g n_g * rate_g
    """

    def __init__(self, graph, R=32, seed=0, device="cpu", max_roots=None):
        self.graph, self.R, self.device = graph, R, torch.device(device)
        n, G = graph.n, graph.n_groups
        self.roots = stratified_roots(graph, max_roots, seed)
        self.n_roots = len(self.roots)
        self.all_roots = self.n_roots == n
        ptr, nodes = sample_rr_sets(graph, self.roots, R, seed)
        self.n_sets = len(ptr) - 1
        sizes = np.diff(ptr)
        self.mean_rr_size = float(sizes.mean())
        self.nnz = len(nodes)
        rows = np.repeat(np.arange(self.n_sets), sizes)
        dev = self.device
        self.rows = torch.as_tensor(rows, device=dev)
        self.cols = torch.as_tensor(nodes, device=dev)
        self.ptr, self.nodes = ptr, nodes                              # numpy, for greedy
        root_groups = graph.groups[self.roots]
        self.root_groups_np = root_groups
        self.root_groups = torch.as_tensor(root_groups, device=dev)
        self.roots_t = torch.as_tensor(self.roots, device=dev)
        rcount = np.bincount(root_groups, minlength=G).astype(np.float64)
        self.root_counts = rcount
        self.sizes = torch.as_tensor(graph.group_sizes, dtype=torch.float32, device=dev)
        self.present = torch.as_tensor(graph.present_groups & (rcount > 0), device=dev)
        self.present_idx = torch.nonzero(self.present).flatten()
        onehot = torch.zeros(self.n_roots, G, device=dev)
        onehot[torch.arange(self.n_roots, device=dev), self.root_groups] = 1.0
        self.group_mean = onehot / torch.as_tensor(np.maximum(rcount, 1), dtype=torch.float32, device=dev)
        # per-set weight turning a hit into d(expected #active in group g): n_g / (R * roots_g)
        set_group = np.repeat(root_groups, R)
        w = graph.group_sizes[set_group] / (R * np.maximum(rcount[set_group], 1))
        self.set_group = torch.as_tensor(set_group, device=dev)
        self.set_weight = torch.as_tensor(w, dtype=torch.float32, device=dev)

    def exposure(self, x):
        logmiss = torch.log1p(-x.clamp(0.0, 1.0 - 1e-6))
        z = torch.zeros(self.n_sets, device=x.device, dtype=x.dtype).index_add_(0, self.rows, logmiss[self.cols])
        miss = torch.exp(z)
        return 1.0 - miss.view(self.n_roots, self.R).mean(1), miss

    def __call__(self, x):
        a, miss = self.exposure(x)
        rates = a @ self.group_mean
        return a, rates, rates @ self.sizes, miss

    @torch.no_grad()
    def group_gains(self, x, miss):
        """d(expected #active in group g) / d x_u for every node u and group g: (n, G)."""
        n, G = self.graph.n, self.graph.n_groups
        contrib = (miss * self.set_weight)[self.rows] / (1.0 - x.clamp(max=1 - 1e-6))[self.cols]
        flat = torch.zeros(n * G, device=x.device).index_add_(0, self.cols * G + self.set_group[self.rows], contrib)
        return flat.view(n, G)

    @torch.no_grad()
    def batch_exposure_rates(self, X, max_elems=int(6e7)):
        """rates (C, G) and totals (C,) for C seed vectors X (C, n), chunked to bound memory."""
        C = X.shape[0]
        per = max(1, max_elems // max(self.nnz, 1))
        rates = []
        for c0 in range(0, C, per):
            logmiss = torch.log1p(-X[c0:c0 + per].clamp(0, 1 - 1e-6)).T          # (n, c)
            z = torch.zeros(self.n_sets, logmiss.shape[1], device=X.device).index_add_(0, self.rows, logmiss[self.cols])
            a = 1.0 - torch.exp(z).view(self.n_roots, self.R, -1).mean(1)      # (roots, c)
            rates.append(a.T @ self.group_mean)
        rates = torch.cat(rates)
        return rates, rates @ self.sizes

    def set_to_x(self, seeds):
        x = torch.zeros(self.graph.n, device=self.device)
        x[torch.as_tensor(np.asarray(seeds, dtype=np.int64), device=self.device)] = 1.0
        return x

    @torch.no_grad()
    def evaluate_set(self, seeds):
        _, rates, total, _ = self(self.set_to_x(seeds))
        return rates.cpu().numpy(), float(total)


# ------------------------------------------------------------- Monte Carlo IC

@njit(parallel=True, cache=True)
def _mc_forward(out_ptr, out_dst, out_p, seeds, groups, n_groups, n_sims, seed):
    n = out_ptr.shape[0] - 1
    nchunk = min(n_sims, 64)
    node_counts = np.zeros((nchunk, n), dtype=np.int32)
    grp = np.zeros((n_sims, n_groups), dtype=np.int32)
    for c in prange(nchunk):
        np.random.seed(seed + 7919 * c)
        stamp_of = np.zeros(n, dtype=np.int64)
        queue = np.empty(n, dtype=np.int64)
        for s in range(c * n_sims // nchunk, (c + 1) * n_sims // nchunk):
            st = s + 1
            tail = 0
            for i in range(seeds.shape[0]):
                v = seeds[i]
                if stamp_of[v] != st:
                    stamp_of[v] = st
                    queue[tail] = v
                    tail += 1
            head = 0
            while head < tail:
                u = queue[head]
                head += 1
                for e in range(out_ptr[u], out_ptr[u + 1]):
                    w = out_dst[e]
                    if stamp_of[w] != st and np.random.random() < out_p[e]:
                        stamp_of[w] = st
                        queue[tail] = w
                        tail += 1
            for i in range(tail):
                v = queue[i]
                node_counts[c, v] += 1
                grp[s, groups[v]] += 1
    return node_counts.sum(axis=0), grp


def monte_carlo(graph, seeds, n_sims=5000, seed=12345):
    """Forward IC simulation. Returns per-node activation probability and
    per-simulation group activation counts (n_sims, G)."""
    out_ptr, out_dst, out_p = graph.out_csr
    counts, grp = _mc_forward(out_ptr, out_dst, out_p, np.asarray(seeds, dtype=np.int64),
                              graph.groups, graph.n_groups, n_sims, seed)
    return counts / n_sims, grp
