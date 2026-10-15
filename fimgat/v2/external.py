"""External baselines ported into the v2 framework (docs/design_v2.md sections 9 and 19). Each is written from its
paper and runs on our RR samples, so every method sees the same RR sets and gets the same certificate.

BSM       Wang, Li, Bonchi, Wang, "Bicriteria Submodular Maximization" (EDBT 2024), Algorithm 1 (BSM-TSGreedy). With
          f the RR spread and g_i the RR rate of group i, it maximises f subject to min_i g_i >= tau * OPT_g, where
          OPT_g = max over |S| = k of min_i g_i(S), estimated by Saturate (Krause et al. 2008). Stage 1 greedily
          covers (1/c) sum_i min(1, g_i(S) / (tau OPT_g)); if that cannot finish within k seeds, the Saturate set is
          returned; otherwise stage 2 fills the budget with Greedy's picks for f, in Greedy's order. The adapter
          (bsm) takes the largest tau in TAUS whose set meets the RR floor on the search sample, else Greedy's set
          (else the reference), so the method chooses its own trade-off from the sample it searches on. The released
          code (github.com/yhwang1990/code-bsm-release, Apache-2.0) was read, not copied.
MultiFW   Tsang, Wilder, Rice, Tambe, Zick, "Group-Fairness in Influence Maximization" (IJCAI 2019), maximin. The
          multiobjective Frank-Wolfe on the multilinear extension, x <- x + v / T, where v solves
          max over v in P_k of min_i <grad r_i(x), v> by multiplicative weights over the groups (best response: the
          top k of the weighted gradient). Then randomised dependent rounding (systematic sampling, which keeps every
          marginal), repeated, keeping the set with the highest minimum group rate on the sample. Written from the
          paper: the released code (github.com/bwilder0/fair_influmax_code_release) has no licence, and its group
          objective counts only one hop of spread.

RFIM      Gao & Ito, "Robust Fair Influence Maximization under Multiple Community Partitions" (SIGMOD 2026): Saturate
          over Q weighted-coverage objectives F_q (alpha-weighted community spreads), maximising min_q F_q / OPT_q
          with exactly k seeds. Partitions: our labelled groups and a label-propagation partition, with the paper's
          three-level alpha, and the inverse of each (rfim_partitions). Written from the paper; the MIT-licensed C++
          needs GCC, which this machine lacks.

All three are deterministic given their seed. Group rates are those of RRGreedy: a covered RR set rooted in group g adds
1 / (R * #roots in g) to rate_g; only present groups (members and roots) count.
"""
import math

import numpy as np
import torch
from numba import njit, prange

from ..baselines import RRGreedy

TAUS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
EPS = 1e-12


# ------------------------------------------------------------------ incremental per-group gains

@njit(cache=True)
def _init_gains(ptr, nodes, set_group, set_w, n, G):
    gain = np.zeros((n, G), dtype=np.float32)
    for j in range(ptr.shape[0] - 1):
        g = set_group[j]
        w = set_w[j]
        for t in range(ptr[j], ptr[j + 1]):
            gain[nodes[t], g] += w
    return gain


@njit(cache=True)
def _add(u, node_ptr, node_sets, ptr, nodes, set_group, set_w, covered, gain, rates):
    """Seed u: cover its uncovered RR sets, add their weight to the group rates and take it off every member's gain."""
    for t in range(node_ptr[u], node_ptr[u + 1]):
        j = node_sets[t]
        if covered[j]:
            continue
        covered[j] = True
        g = set_group[j]
        w = set_w[j]
        rates[g] += w
        for s in range(ptr[j], ptr[j + 1]):
            gain[nodes[s], g] -= w


@njit(cache=True, parallel=True)
def _cover_scores(gain, need, chosen):
    """sum_i min(gain[v, i], need_i) for every node v (-1 for chosen nodes): the marginal gain of the truncated cover."""
    n, G = gain.shape
    out = np.empty(n)
    for v in prange(n):
        if chosen[v]:
            out[v] = -1.0
            continue
        s = 0.0
        for i in range(G):
            if need[i] > 0.0:
                gv = gain[v, i]
                s += gv if gv < need[i] else need[i]
        out[v] = s
    return out


class _Cover:
    """Greedy state on one RR sample: seeds, covered sets, group rates and the per-node per-group gains."""

    def __init__(self, rg):
        self.rg = rg
        self.gain = _init_gains(rg.ptr, rg.nodes, rg.set_group, rg.set_w, rg.n, rg.G)
        self.covered = np.zeros(len(rg.ptr) - 1, dtype=np.bool_)
        self.rates = np.zeros(rg.G)
        self.chosen = np.zeros(rg.n, dtype=np.bool_)
        self.seeds = []

    def add(self, u):
        u = int(u)
        if self.chosen[u]:
            return
        rg = self.rg
        _add(u, rg.node_ptr, rg.node_sets, rg.ptr, rg.nodes, rg.set_group, rg.set_w, self.covered, self.gain,
             self.rates)
        self.chosen[u] = True
        self.seeds.append(u)

    def cover_to(self, target, k, present):
        """Greedy partial cover: add seeds until every present group's rate reaches target (or k seeds, or no gain).
        Returns True if the target was reached."""
        need = np.zeros(self.rg.G)
        while True:
            need[:] = 0.0
            need[present] = np.maximum(target - self.rates[present], 0.0)
            if need.max() <= 1e-9:
                return True
            if len(self.seeds) >= k:
                return False
            sc = _cover_scores(self.gain, need, self.chosen)
            u = int(np.argmax(sc))
            if sc[u] <= 0.0:
                return False
            self.add(u)


def eval_rates(rg, seeds):
    """Exact (rates (G,), RR total) of a seed set on rg's sample."""
    covered = np.zeros(len(rg.ptr) - 1, dtype=np.bool_)
    for u in seeds:
        covered[rg.node_sets[rg.node_ptr[int(u)]:rg.node_ptr[int(u) + 1]]] = True
    rates = np.bincount(rg.set_group[covered], weights=rg.set_w[covered], minlength=rg.G)
    return rates, float(rates @ rg.sizes)


# ------------------------------------------------------------------ BSM

def saturate(rg, k, iters=12):
    """Saturate (Krause et al. 2008) for max over |S| = k of min_i r_i(S), with the budget kept at k: bisection on c
    for the largest target that the greedy partial cover reaches within k seeds. Returns (seeds, OPT_g estimate =
    the min present-group rate of those seeds)."""
    present = np.flatnonzero(rg.present)
    lo, hi, best = 0.0, 1.0, None
    for _ in range(int(iters)):
        c = 0.5 * (lo + hi)
        st = _Cover(rg)
        if st.cover_to(c, k, present):
            lo, best = c, list(st.seeds)
        else:
            hi = c
    if best is None:                                # even a tiny target fails: the maximin greedy set
        best = rg.maximin(k)
    best = _pad(rg, best, k)
    rates, _ = eval_rates(rg, best)
    return best, float(rates[present].min()) if len(present) else 0.0


def _pad(rg, seeds, k):
    """Fill a set to k seeds with Greedy's next picks (spread), as BSM's stage 2 does."""
    seeds = list(dict.fromkeys(int(v) for v in seeds))
    if len(seeds) >= k:
        return seeds[:k]
    st = _Cover(rg)
    for u in seeds:
        st.add(u)
    while len(st.seeds) < k:
        sc = st.gain.astype(np.float64) @ rg.sizes
        sc[st.chosen] = -np.inf
        st.add(int(np.argmax(sc)))
    return st.seeds


def bsm_tsgreedy(rg, k, tau, greedy_seeds, sat_seeds, opt_g):
    """BSM-TSGreedy (Algorithm 1) for one balance factor tau."""
    present = np.flatnonzero(rg.present)
    st = _Cover(rg)
    if not st.cover_to(float(tau) * float(opt_g), k, present):
        return list(sat_seeds)                      # the constraint cannot be met within k: the Saturate set
    for v in greedy_seeds:                          # stage 2: Greedy's picks, in Greedy's order
        if len(st.seeds) >= k:
            break
        st.add(int(v))
    return _pad(rg, st.seeds, k)


def _bsm_parts(rg, k, cache):
    """(Greedy's seeds, Saturate's seeds, OPT_g) for budget k: the rho-free parts of BSM, cached per k."""
    key = ("bsm", int(k))
    if key not in cache:
        g_seeds = rg.greedy(int(k))
        s_seeds, opt = saturate(rg, int(k))
        cache[key] = (g_seeds, s_seeds, opt)
    return cache[key]


def bsm_sweep(est, k, rg=None, taus=TAUS, cache=None):
    """BSM's whole utility-fairness trade-off for budget k: [(tau, seeds)] for every tau in taus (largest first),
    each BSM-TSGreedy's set with no floor applied. It traces the spread-disparity curve that the bsm adapter samples
    one point of (docs/design_v2.md section 21); rho-free, so one sweep serves every floor."""
    rg = RRGreedy(est) if rg is None else rg
    cache = {} if cache is None else cache
    g_seeds, s_seeds, opt = _bsm_parts(rg, k, cache)
    return [(float(tau), [int(v) for v in bsm_tsgreedy(rg, int(k), tau, g_seeds, s_seeds, opt)])
            for tau in sorted(taus, reverse=True)]


def bsm(est, k, floor, reference=None, rg=None, taus=TAUS, cache=None, return_info=False):
    """The BSM baseline: the largest tau whose BSM-TSGreedy set meets the RR floor on est, else Greedy's set, else the
    reference. rg: an RRGreedy index of est (built if None); cache: a dict kept per estimator, which stores the
    rho-free parts (Greedy, Saturate) per k."""
    rg = RRGreedy(est) if rg is None else rg
    k = int(k)
    cache = {} if cache is None else cache
    g_seeds, s_seeds, opt = _bsm_parts(rg, k, cache)
    info = {"opt_g": opt, "tau": None}
    for tau in sorted(taus, reverse=True):
        seeds = bsm_tsgreedy(rg, k, tau, g_seeds, s_seeds, opt)
        if eval_rates(rg, seeds)[1] >= float(floor):
            info["tau"] = float(tau)
            return (seeds, info) if return_info else seeds
    if eval_rates(rg, g_seeds)[1] >= float(floor) or reference is None:
        info["fallback"] = "greedy"
        out = list(g_seeds)
    else:
        info["fallback"] = "reference"
        out = [int(v) for v in reference]
    return (out, info) if return_info else out


# ------------------------------------------------------------------ MultiFW (Tsang et al. 2019, maximin)

def _systematic(x, k, rng):
    """k distinct nodes with P(v chosen) = x_v (sum x = k, 0 <= x <= 1): systematic sampling over a random order."""
    n = len(x)
    perm = rng.permutation(n)
    c = np.cumsum(x[perm])
    pts = rng.random() + np.arange(int(k))
    idx = np.minimum(np.searchsorted(c, pts, side="right"), n - 1)
    out = list(dict.fromkeys(int(perm[i]) for i in idx))
    if len(out) < k:                                # float round-off: fill with the largest remaining x
        rest = [int(v) for v in np.argsort(-x) if int(v) not in set(out)]
        out += rest[:k - len(out)]
    return out[:k]


@torch.no_grad()
def multifw_maxmin(est, k, iters=20, mw_steps=100, rounds=20, seed=0, rg=None, return_info=False):
    """Tsang et al.'s multiobjective Frank-Wolfe for max min_i r_i (see the module docstring), on est's multilinear
    extension. Returns the best of `rounds` dependent roundings by the minimum present-group rate on est."""
    k, n = int(k), int(est.graph.n)
    dev = est.device
    pres = est.present.clone()
    P = int(pres.sum())
    if P == 0:
        raise ValueError("no present group")
    sizes = est.sizes[pres]
    x = torch.zeros(n, device=dev)
    eta = math.sqrt(8.0 * math.log(max(P, 2)) / max(int(mw_steps), 1))
    for _ in range(int(iters)):
        _, miss = est.exposure(x)
        grad = est.group_gains(x, miss)[:, pres] / sizes            # (n, P): d rate_i / d x_v
        lam = torch.full((P,), 1.0 / P, device=dev)
        v_acc = torch.zeros(n, device=dev)
        for _ in range(int(mw_steps)):
            idx = torch.topk(grad @ lam, k).indices                 # best response to the group weights
            v_acc[idx] += 1.0
            pay = grad[idx].sum(0)                                  # <grad r_i, v> per group
            pay = pay / pay.max().clamp(min=EPS)
            lam = lam * torch.exp(-eta * pay)
            lam = lam / lam.sum()
        x = (x + v_acc / (int(mw_steps) * int(iters))).clamp(0.0, 1.0)
    xs = x.double().cpu().numpy()
    xs *= k / max(xs.sum(), EPS)
    xs = np.clip(xs, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    rg = RRGreedy(est) if rg is None else rg
    present = np.flatnonzero(rg.present)
    best, best_key = None, None
    for _ in range(int(rounds)):
        s = _systematic(xs, k, rng)
        rates, total = eval_rates(rg, s)
        key = (float(rates[present].min()), total)
        if best_key is None or key > best_key:
            best, best_key = s, key
    info = {"min_rate": best_key[0], "rr_total": best_key[1], "iters": int(iters), "rounds": int(rounds)}
    return (best, info) if return_info else best


# ------------------------------------------------------------------ RFIM (Gao & Ito, SIGMOD 2026)

@njit(cache=True)
def _label_propagation(ptr_a, nbr_a, ptr_b, nbr_b, order, iters, seed):
    """Asynchronous label propagation (Raghavan et al. 2007) over the union of two CSR adjacencies: every node, in a
    random order per sweep, takes the most frequent label among its neighbours (ties: a random one of them)."""
    np.random.seed(seed)
    n = ptr_a.shape[0] - 1
    lab = np.arange(n)
    buf = np.empty(n, dtype=np.int64)
    for it in range(iters):
        changed = 0
        perm = order[it]
        for i in range(n):
            v = perm[i]
            m = 0
            for t in range(ptr_a[v], ptr_a[v + 1]):
                buf[m] = lab[nbr_a[t]]
                m += 1
            for t in range(ptr_b[v], ptr_b[v + 1]):
                buf[m] = lab[nbr_b[t]]
                m += 1
            if m == 0:
                continue
            s = np.sort(buf[:m])
            best_c, n_best, cur, c, pick = 0, 0, s[0], 0, lab[v]
            for t in range(m):                      # mode, with reservoir tie-breaking
                if s[t] == cur:
                    c += 1
                else:
                    cur, c = s[t], 1
                if c > best_c:
                    best_c, n_best, pick = c, 1, cur
                elif c == best_c:
                    n_best += 1
                    if np.random.random() * n_best < 1.0:
                        pick = cur
            if pick != lab[v]:
                lab[v] = pick
                changed += 1
        if changed == 0:
            break
    return lab


def label_propagation(graph, iters=20, seed=0):
    """Community labels 0..C-1 of a graph by label propagation (edges as undirected)."""
    ptr_a, nbr_a, _ = graph.out_csr
    ptr_b, nbr_b, _ = graph.in_csr
    rng = np.random.default_rng(seed)
    order = np.stack([rng.permutation(graph.n) for _ in range(int(iters))]).astype(np.int64)
    lab = _label_propagation(ptr_a, nbr_a, ptr_b, nbr_b, order, int(iters), int(seed))
    return np.unique(lab, return_inverse=True)[1].astype(np.int64)


def rfim_alpha(labels, n, rng):
    """RFIM's three-level fairness parameters (paper section 5.1.3): a community of size < n/5 gets U(0, 1), of size in
    [n/5, n/2) U(0, 0.1), of size >= n/2 U(0, 0.01)."""
    size = np.bincount(labels).astype(np.float64)
    hi = np.where(size < n / 5, 1.0, np.where(size < n / 2, 0.1, 0.01))
    return rng.random(len(size)) * hi


def rfim_inverse(alpha, rng):
    """The paper's inverse partition: each community's alpha becomes 1 - alpha with probability 0.75."""
    flip = rng.random(len(alpha)) < 0.75
    return np.where(flip, 1.0 - alpha, alpha)


def rfim_partitions(graph, seed=0, lp_iters=20):
    """RFIM's partitions in our setting (design section 20): the labelled groups and a label-propagation partition, each
    with the three-level alpha, and the inverse of each (the paper's Q = 4 construction, with our labels in place of
    ground truth). No Louvain/Leiden partition: community metrics are evaluated on Louvain, so giving it to a method
    would leak. Returns [(labels (n,), alpha (C,))]."""
    rng = np.random.default_rng(seed)
    out = []
    for labels in (np.asarray(graph.groups, dtype=np.int64), label_propagation(graph, lp_iters, seed)):
        a = rfim_alpha(labels, graph.n, rng)
        out += [(labels, a), (labels, rfim_inverse(a, rng))]
    return out


@njit(cache=True)
def _init_wgains(ptr, nodes, W, n):
    Q = W.shape[1]
    gain = np.zeros((n, Q))
    for j in range(ptr.shape[0] - 1):
        for t in range(ptr[j], ptr[j + 1]):
            for q in range(Q):
                gain[nodes[t], q] += W[j, q]
    return gain


@njit(cache=True)
def _add_w(u, node_ptr, node_sets, ptr, nodes, W, covered, gain, F):
    Q = W.shape[1]
    for t in range(node_ptr[u], node_ptr[u + 1]):
        j = node_sets[t]
        if covered[j]:
            continue
        covered[j] = True
        for q in range(Q):
            F[q] += W[j, q]
        for s in range(ptr[j], ptr[j + 1]):
            for q in range(Q):
                gain[nodes[s], q] -= W[j, q]


@njit(cache=True, parallel=True)
def _sat_scores(gain, inv_opt, need, chosen):
    """sum_q min(gain[v, q] / OPT_q, need_q): the marginal gain of the truncated robust objective."""
    n, Q = gain.shape
    out = np.empty(n)
    for v in prange(n):
        if chosen[v]:
            out[v] = -1.0
            continue
        s = 0.0
        for q in range(Q):
            if need[q] > 0.0:
                g = gain[v, q] * inv_opt[q]
                s += g if g < need[q] else need[q]
        out[v] = s
    return out


class _WCover:
    """Greedy state for Q weighted coverage functions F_q(S) = sum of W[j, q] over covered RR sets j."""

    def __init__(self, rg, W):
        self.rg, self.W = rg, W
        self.gain = _init_wgains(rg.ptr, rg.nodes, W, rg.n)
        self.covered = np.zeros(len(rg.ptr) - 1, dtype=np.bool_)
        self.F = np.zeros(W.shape[1])
        self.chosen = np.zeros(rg.n, dtype=np.bool_)
        self.seeds = []

    def add(self, u):
        u = int(u)
        if self.chosen[u]:
            return
        _add_w(u, self.rg.node_ptr, self.rg.node_sets, self.rg.ptr, self.rg.nodes, self.W, self.covered, self.gain,
               self.F)
        self.chosen[u] = True
        self.seeds.append(u)


def rfim_weights(est, partitions):
    """W (n_sets, Q): RR set j rooted at r adds alpha_q[c] * |C_c| / (#RR sets rooted in C_c) to F_q, where c is r's
    community under partition q, so F_q estimates sum_c alpha_qc * E[#activated in C_c]."""
    R = int(est.R)
    roots = np.asarray(est.roots, dtype=np.int64)
    W = np.zeros((len(roots) * R, len(partitions)))
    for q, (labels, alpha) in enumerate(partitions):
        c_root = labels[roots]
        size = np.bincount(labels, minlength=len(alpha)).astype(np.float64)
        n_sets = R * np.bincount(c_root, minlength=len(alpha)).astype(np.float64)
        w_c = alpha * size / np.maximum(n_sets, 1.0)
        W[:, q] = np.repeat(w_c[c_root], R)
    return W


def rfim(est, k, partitions=None, rg=None, iters=12, seed=0, return_info=False):
    """RFIM by Saturate (the paper's S-HIST, Algorithm 1, on our RR sample): OPT_q is greedy's value on F_q divided by
    (1 - 1/e); the bisection finds the largest c such that greedy on (1/Q) sum_q min(c, F_q(S) / OPT_q) reaches c for
    every partition within k seeds (the paper's -sratio=1 setting: exactly k seeds, no bicriteria relaxation). A set
    that reaches its target early is filled with Greedy's picks for the spread. Written from the paper; the released
    C++ (github.com/TianyouG/RFIM, MIT) was read, not run."""
    k = int(k)
    rg = RRGreedy(est) if rg is None else rg
    partitions = rfim_partitions(est.graph, seed) if partitions is None else partitions
    W = rfim_weights(est, partitions)
    Q = W.shape[1]
    opt = np.zeros(Q)
    for q in range(Q):                              # greedy per partition for OPT_q
        st = _WCover(rg, W[:, q:q + 1].copy())
        for _ in range(k):
            sc = st.gain[:, 0].copy()
            sc[st.chosen] = -np.inf
            st.add(int(np.argmax(sc)))
        opt[q] = st.F[0] / (1.0 - 1.0 / math.e)
    inv_opt = 1.0 / np.maximum(opt, EPS)
    lo, hi, best = 0.0, 1.0, None
    for _ in range(int(iters)):
        c = 0.5 * (lo + hi)
        st = _WCover(rg, W)
        ok = False
        while len(st.seeds) <= k:
            need = np.maximum(c - st.F * inv_opt, 0.0)
            if need.max() <= 1e-12:
                ok = True
                break
            if len(st.seeds) == k:
                break
            sc = _sat_scores(st.gain, inv_opt, need, st.chosen)
            u = int(np.argmax(sc))
            if sc[u] <= 0.0:
                break
            st.add(u)
        if ok:
            lo, best = c, list(st.seeds)
        else:
            hi = c
    if best is None:
        best = []
    seeds = _pad(rg, best, k)
    st = _WCover(rg, W)
    for u in seeds:
        st.add(u)
    ratios = st.F * inv_opt
    info = {"min_ratio": float(ratios.min()), "c": float(lo), "Q": int(Q), "opt": opt.tolist()}
    return (seeds, info) if return_info else seeds
