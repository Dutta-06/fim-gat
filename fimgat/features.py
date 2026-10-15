"""Graph context and node features for the seed policy.

Every group-related feature is *relative* (own group vs. currently under-served
groups), never a one-hot of the group's identity. The policy is therefore
equivariant to relabelling groups and transfers across graphs whose sensitive
attribute has different values or a different number of groups.

Static (once per graph):
    log out/in degree, log expected out-degree, log reverse PageRank, core number,
    log singleton spread, log singleton reach into own group (rate),
    log own-group share, fraction of neighbours in own group
Dynamic (every Frank-Wolfe round, from the current fractional seeds x):
    x_v, exposure a_v, own-group normalised rate, own-group deficit weight,
    log exact gradient of J_beta (the classical FW direction),
    log marginal gain in total spread, log deficit-weighted marginal rate gain,
    log own-group marginal rate gain, log deficit-weighted singleton reach,
    fraction of neighbours in under-served groups, beta, round index

Everything here is O(n + m) (numba / scipy / torch index ops) so it runs on
graphs with millions of nodes.
"""
import numpy as np
import scipy.sparse as sp
import torch
from numba import njit

N_STATIC = 9
N_DYNAMIC = 12
EDGE_DIM = 3          # (p, is_reverse, is_self_loop)


def _z(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-8)


def _tz(t):
    return (t - t.mean(0, keepdim=True)) / (t.std(0, keepdim=True) + 1e-6)


@njit(cache=True)
def _core_numbers(ptr, nbr):
    """Batagelj-Zaversnik O(m) k-core decomposition of an undirected CSR graph."""
    n = ptr.shape[0] - 1
    deg = np.empty(n, dtype=np.int64)
    for v in range(n):
        deg[v] = ptr[v + 1] - ptr[v]
    md = deg.max() if n > 0 else 0
    bins = np.zeros(md + 1, dtype=np.int64)
    for v in range(n):
        bins[deg[v]] += 1
    start = 0
    for d in range(md + 1):
        c = bins[d]
        bins[d] = start
        start += c
    pos = np.empty(n, dtype=np.int64)
    vert = np.empty(n, dtype=np.int64)
    for v in range(n):
        pos[v] = bins[deg[v]]
        vert[pos[v]] = v
        bins[deg[v]] += 1
    for d in range(md, 0, -1):
        bins[d] = bins[d - 1]
    bins[0] = 0
    for i in range(n):
        v = vert[i]
        for e in range(ptr[v], ptr[v + 1]):
            u = nbr[e]
            if deg[u] > deg[v]:
                du = deg[u]
                pu = pos[u]
                pw = bins[du]
                w = vert[pw]
                if u != w:
                    pos[u] = pw
                    vert[pu] = w
                    pos[w] = pu
                    vert[pw] = u
                bins[du] += 1
                deg[u] -= 1
    return deg


def undirected_csr(graph):
    n = graph.n
    code = np.unique(np.concatenate([graph.src * n + graph.dst, graph.dst * n + graph.src]))
    u, v = code // n, code % n
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=n), out=ptr[1:])
    return ptr, v, u


def reverse_pagerank(graph, alpha=0.85, iters=40):
    """PageRank on the reversed, probability-weighted graph: high for nodes that reach many."""
    n = graph.n
    A = sp.csr_matrix((graph.prob, (graph.dst, graph.src)), shape=(n, n))
    out = np.asarray(A.sum(1)).ravel()
    P = (sp.diags(1.0 / np.maximum(out, 1e-12)) @ A).T.tocsr()
    dangling = out == 0
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = alpha * (P @ r) + (1 - alpha) / n + alpha * r[dangling].sum() / n
    return r


class GraphContext:
    """Everything the policy and critic need about one graph.

    device       where features / estimator live (GPU)
    edge_device  where the message-passing edge list lives. Put it on the CPU for
                 graphs whose edges don't fit next to the activations; the policy
                 then runs chunk by chunk (inference only).
    """

    def __init__(self, graph, est, device, edge_device=None):
        self.graph, self.est = graph, est
        self.device = torch.device(device)
        self.edge_device = torch.device(edge_device or device)
        n, G = graph.n, graph.n_groups
        self.n, self.G = n, G
        dev = self.device
        own = graph.groups
        sizes = np.maximum(graph.group_sizes, 1).astype(np.float64)

        # singleton reach per group from RR sets (root-sampled, reweighted)
        nnz_group = np.repeat(np.repeat(est.root_groups_np, est.R), np.diff(est.ptr))
        w = 1.0 / (est.R * np.maximum(est.root_counts, 1))                    # per group
        cnt = np.bincount(est.nodes * G + nnz_group, weights=w[nnz_group], minlength=n * G).reshape(n, G)
        reach = cnt * graph.group_sizes[None, :]                             # expected # of group-g nodes reached
        self.reach_rate = torch.as_tensor(cnt * n, dtype=torch.float32, device=dev)

        uptr, unbr, usrc = undirected_csr(graph)
        core = _core_numbers(uptr, unbr).astype(np.float64)
        udeg = np.diff(uptr)
        nbr_cnt = np.bincount(usrc * G + own[unbr], minlength=n * G).reshape(n, G)
        nbr_frac = nbr_cnt / np.maximum(udeg, 1)[:, None]
        self.nbr_frac = torch.as_tensor(nbr_frac, dtype=torch.float32, device=dev)
        exp_out = np.bincount(graph.src, weights=graph.prob, minlength=n)

        static = np.stack([
            _z(np.log1p(graph.out_degree)),
            _z(np.log1p(graph.in_degree)),
            _z(np.log1p(exp_out)),
            _z(np.log(reverse_pagerank(graph) * n)),
            _z(core),
            _z(np.log1p(reach.sum(1))),
            _z(np.log1p(cnt[np.arange(n), own] * n)),
            _z(np.log(sizes[own] / n)),
            nbr_frac[np.arange(n), own],
        ], 1)
        assert static.shape[1] == N_STATIC
        self.static = torch.as_tensor(static, dtype=torch.float32, device=dev)
        self.own = torch.as_tensor(own, device=dev)
        self.sizes = torch.as_tensor(sizes, dtype=torch.float32, device=dev)
        self.present_idx = est.present_idx

        # message-passing edges: both directions + explicit self-loops, sorted by target
        src = np.concatenate([graph.src, graph.dst, np.arange(n)])
        dst = np.concatenate([graph.dst, graph.src, np.arange(n)])
        m = graph.m
        attr = np.zeros((2 * m + n, EDGE_DIM), dtype=np.float32)
        attr[:2 * m, 0] = np.concatenate([graph.prob, graph.prob])
        attr[m:2 * m, 1] = 1.0
        attr[2 * m:, 2] = 1.0
        order = np.argsort(dst, kind="stable")
        self.dst_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(dst, minlength=n), out=self.dst_ptr[1:])
        edev = self.edge_device
        self.edge_index = torch.as_tensor(np.stack([src[order], dst[order]]), device=edev)
        self.edge_attr = torch.as_tensor(attr[order], device=edev)

        # imputing exposure of non-root nodes from root neighbours (only if roots are sampled)
        if not est.all_roots:
            is_root = np.zeros(n, dtype=bool)
            is_root[est.roots] = True
            keep = is_root[unbr]
            self._imp_src = torch.as_tensor(usrc[keep], device=dev)
            self._imp_dst_root = torch.as_tensor(np.searchsorted(est.roots, unbr[keep]), device=dev)
            self._imp_deg = torch.as_tensor(np.bincount(usrc[keep], minlength=n), dtype=torch.float32, device=dev)
        self.total_ref = {}

    def neighbourhood_mean(self, v, hops=2):
        """Average a node signal over closed neighbourhoods (both directions), `hops` times."""
        if not hasattr(self, "_nb_deg"):
            self._nb_deg = torch.bincount(self.edge_index[1], minlength=self.n).float().clamp(min=1)
        src, dst = self.edge_index[0], self.edge_index[1]
        for _ in range(hops):
            v = torch.zeros_like(v).index_add_(0, dst, v[src]) / self._nb_deg
        return v

    def full_exposure(self, a_roots):
        """Exposure for every node: exact at roots, neighbour-mean of roots elsewhere."""
        est = self.est
        if est.all_roots:
            return a_roots
        nb = torch.zeros(self.n, device=a_roots.device).index_add_(0, self._imp_src, a_roots[self._imp_dst_root])
        a = torch.where(self._imp_deg > 0, nb / self._imp_deg.clamp(min=1), a_roots.mean().expand(self.n))
        return a.index_copy(0, est.roots_t, a_roots)

    def deficit(self, rates, total, tau=0.1):
        """Soft indicator of under-served groups: softmax(-normalised rate / tau) over present groups."""
        norm = rates / (total / self.n + 1e-9)
        logits = torch.full_like(norm, -1e9).index_copy(0, self.present_idx, -norm[self.present_idx] / tau)
        return torch.softmax(logits, 0), norm

    @torch.no_grad()
    def dynamic(self, x, a_roots, rates, total, gains, grad_j, beta, t, rounds, k):
        n = self.n
        a = self.full_exposure(a_roots)
        d, norm = self.deficit(rates, total)
        own = self.own
        rate_gain = gains / self.sizes * n                        # (n, G) marginal gain in normalised rate
        dyn = torch.stack([
            x * n / (10.0 * k),
            a,
            norm[own],
            d[own] * self.G,
            _tz(torch.log1p(grad_j * self.total_ref[k])),
            _tz(torch.log1p(gains.sum(1))),
            _tz(torch.log1p(rate_gain @ d)),
            _tz(torch.log1p(rate_gain.gather(1, own[:, None]).squeeze(1))),
            _tz(torch.log1p(self.reach_rate @ d)),
            self.nbr_frac @ d,
            torch.full_like(a, float(beta)),
            torch.full_like(a, t / max(rounds - 1, 1)),
        ], 1)
        return dyn

    def memory_mb(self):
        e = self.edge_index.numel() * 8 + self.edge_attr.numel() * 4
        est = self.est.rows.numel() * 16 + self.est.n_sets * 12
        return {"edges_mb": e / 2**20, "estimator_mb": est / 2**20}
