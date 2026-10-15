"""Baselines. All combinatorial methods use the same RR-set estimator as FIM-GAT.

Degree            top-k by expected out-degree
Greedy            RIS greedy for total influence (Borgs et al. 2014; Tang et al. 2015)
Parity-Greedy     seed quota proportional to group size, greedy within quota
                  (budget parity, e.g. Stoica et al. 2020)
Welfare-Greedy    greedy on the alpha-fair welfare sum_g n_g r_g^alpha
                  (Rahmattalabi et al. 2021; the FIMM objective of Rui et al. 2023)
Maximin-Greedy    greedy on min_g r_g with total-influence tie-break (Fish et al. 2019)
FW-Fair           continuous greedy / Frank-Wolfe on the multilinear extension of the
                  exact J_beta objective FIM-GAT optimises (Tsang et al. 2019 style).
                  This is FIM-GAT's objective with no GNN and no critic.
AGE               adversarial graph embeddings + fair re-clustering
                  (Khajehnejad et al. 2020), implemented as in the paper.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit

from .decode import decode
from .objective import fairness_objective


# ------------------------------------------------------------ RR greedy engine

@njit(cache=True)
def _gains(ptr, nodes, set_group, set_w, covered, n, G):
    gain = np.zeros((n, G))
    for j in range(ptr.shape[0] - 1):
        if covered[j]:
            continue
        g = set_group[j]
        w = set_w[j]
        for t in range(ptr[j], ptr[j + 1]):
            gain[nodes[t], g] += w
    return gain


@njit(cache=True)
def _cover(node_ptr, node_sets, u, covered):
    for t in range(node_ptr[u], node_ptr[u + 1]):
        covered[node_sets[t]] = True


class RRGreedy:
    """Exact (non-lazy) greedy over RR sets for any objective of the group-rate vector."""

    def __init__(self, est):
        g = est.graph
        self.n, self.G = g.n, g.n_groups
        self.ptr, self.nodes = est.ptr, est.nodes
        self.set_group = np.repeat(est.root_groups_np, est.R)
        sizes = g.group_sizes.astype(np.float64)
        # a hit on a set rooted in group g adds 1 / (R * #roots in g) to rate_g
        self.set_w = 1.0 / (est.R * np.maximum(est.root_counts[self.set_group], 1))
        rows = np.repeat(np.arange(len(self.ptr) - 1), np.diff(self.ptr))
        order = np.argsort(self.nodes, kind="stable")
        self.node_sets = rows[order]
        self.node_ptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.nodes, minlength=self.n), out=self.node_ptr[1:])
        self.sizes = sizes
        self.present = (g.group_sizes > 0) & (est.root_counts > 0)
        self.groups = g.groups

    def run(self, k, score_fn, quota=None):
        covered = np.zeros(len(self.ptr) - 1, dtype=np.bool_)
        rates = np.zeros(self.G)
        seeds, left = [], None if quota is None else quota.copy()
        for _ in range(k):
            gain = _gains(self.ptr, self.nodes, self.set_group, self.set_w, covered, self.n, self.G)
            cand = rates[None, :] + gain
            val = score_fn(cand)
            val[seeds] = -np.inf
            if left is not None:
                val[left[self.groups] <= 0] = -np.inf
            u = int(np.argmax(val))
            seeds.append(u)
            rates = cand[u]
            if left is not None:
                left[self.groups[u]] -= 1
            _cover(self.node_ptr, self.node_sets, u, covered)
        return seeds

    def total(self, r):
        return r @ self.sizes

    def greedy(self, k):
        return self.run(k, self.total)

    def welfare(self, k, alpha=0.5):
        s, p = self.sizes[self.present], self.present
        return self.run(k, lambda r: (np.power(np.maximum(r[:, p], 0), alpha) * s).sum(1))

    def maximin(self, k):
        p = self.present
        return self.run(k, lambda r: r[:, p].min(1) + 1e-4 * self.total(r) / self.n)

    def parity(self, k):
        s = self.sizes * self.present
        raw = k * s / s.sum()
        quota = np.floor(raw).astype(int)
        for g in np.argsort(-(raw - quota))[: k - quota.sum()]:
            quota[g] += 1
        # a group can't receive more seeds than it has members
        return self.run(k, self.total, quota=quota)


def degree(graph, k):
    exp_out = np.bincount(graph.src, weights=graph.prob, minlength=graph.n)
    return np.argsort(-exp_out, kind="stable")[:k].tolist()


# ------------------------------------------------------------- FW-Fair

def fw_fair(est, k, beta, total_ref, steps=100):
    """Continuous greedy (Frank-Wolfe with step 1/T from 0) on J_beta, then top-k rounding."""
    n = est.graph.n
    x = torch.zeros(n, device=est.device)
    grad = None
    for _ in range(steps):
        xv = x.clone().requires_grad_(True)
        _, rates, total, _ = est(xv)
        J, _, _ = fairness_objective(rates, total, n, total_ref, beta, est.present_idx)
        grad, = torch.autograd.grad(J, xv)
        v = torch.zeros_like(x)
        v[torch.topk(grad, k).indices] = 1.0
        x = x + v / steps
    # same sample-and-select rounding as FIM-GAT; ties in x broken by the last gradient
    key = x + 1e-3 * (grad / (grad.abs().max() + 1e-12)) / steps
    return decode(est, key, k, beta, total_ref, extra=[grad])[0]


# ------------------------------------------------------------- AGE

class _AE(nn.Module):
    def __init__(self, n_in, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 4 * d), nn.ReLU(), nn.Linear(4 * d, 2 * d),
                                 nn.ReLU(), nn.Linear(2 * d, d), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 4 * d),
                                 nn.ReLU(), nn.Linear(4 * d, n_in), nn.Sigmoid())


def age(graph, k, device="cpu", dim=30, n_clusters=4, pre_epochs=300, adv_epochs=500,
        batch=50, seed=0):
    """Khajehnejad et al. (2020): autoencode adjacency rows, adversarially remove the
    sensitive attribute from the embedding, then 're-cluster' seeding."""
    from sklearn.cluster import KMeans
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, G = graph.n, graph.n_groups
    A = torch.zeros(n, n)
    A[torch.as_tensor(graph.src), torch.as_tensor(graph.dst)] = 1.0
    A = A.to(device)
    y = torch.as_tensor(graph.groups, device=device)
    ae = _AE(n, dim).to(device)
    disc = nn.Sequential(nn.Linear(dim, 25), nn.ReLU(), nn.Dropout(0.25), nn.Linear(25, 15),
                         nn.ReLU(), nn.Dropout(0.2), nn.Linear(15, 6), nn.ReLU(),
                         nn.Dropout(0.2), nn.Linear(6, G)).to(device)
    opt_ae, opt_d = torch.optim.Adam(ae.parameters()), torch.optim.Adam(disc.parameters())
    # class-balanced batches so the discriminator can't win by predicting the majority
    by_group = [np.flatnonzero(graph.groups == g) for g in range(G) if graph.group_sizes[g] > 0]

    def balanced_batch():
        per = max(1, batch // len(by_group))
        return np.concatenate([rng.choice(m, per) for m in by_group])

    for _ in range(pre_epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            xb = A[perm[i:i + batch]]
            loss = F.binary_cross_entropy(ae.dec(ae.enc(xb)), xb, reduction="sum")
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    for _ in range(40):
        idx = balanced_batch()
        with torch.no_grad():
            z = ae.enc(A[idx])
        loss = F.cross_entropy(disc(z), y[idx])
        opt_d.zero_grad(); loss.backward(); opt_d.step()
    for _ in range(adv_epochs):
        for _ in range(max(1, n // (batch * 2))):
            idx = balanced_batch()
            xb = A[idx]
            with torch.no_grad():
                z = ae.enc(xb)
            d_loss = F.cross_entropy(disc(z), y[idx])
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            z = ae.enc(xb)
            # generator: reconstruct while making the discriminator's output uniform
            conf = -F.log_softmax(disc(z), 1).mean()
            loss = F.binary_cross_entropy(ae.dec(z), xb, reduction="sum") + 10 * conf * len(idx)
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    with torch.no_grad():
        emb = ae.enc(A).cpu().numpy()

    # re-cluster seeding
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(emb)
    quota = [k // n_clusters + (1 if c < k % n_clusters else 0) for c in range(n_clusters)]
    seeds = []
    for c in range(n_clusters):
        members = np.flatnonzero(km.labels_ == c)
        if len(members) == 0:
            continue
        near = members[np.argsort(np.linalg.norm(emb[members] - km.cluster_centers_[c], axis=1))]
        portion = np.bincount(graph.groups[near[:quota[c]]], minlength=G)
        for g in range(G):
            mg = members[graph.groups[members] == g]
            if portion[g] == 0 or len(mg) == 0:
                continue
            center = emb[mg].mean(0)
            seeds.extend(mg[np.argsort(np.linalg.norm(emb[mg] - center, axis=1))][:portion[g]].tolist())
    # empty clusters can leave the budget short: top up with nearest-to-centre nodes
    if len(set(seeds)) < k:
        d = np.min(np.linalg.norm(emb[:, None, :] - km.cluster_centers_[None], axis=2), axis=1)
        for v in np.argsort(d):
            if len(set(seeds)) >= k:
                break
            if v not in seeds:
                seeds.append(int(v))
    return list(dict.fromkeys(int(s) for s in seeds))[:k]


# ------------------------------------------------------------- previous version

def prior_repadv_gat(graph, k, device="cpu", epochs=1000, seed=0):
    """Faithful port of the project's previous model (SOTA-FairXAI): 2-layer GATv2 on
    degree + noise features, one-hop coverage surrogate with a soft size penalty, a
    gradient-reversal adversary on the embeddings, a group-variance penalty, and top-k
    scores. Kept as-is (including its known issues) to measure the improvement."""
    import copy as _copy
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.utils import dropout_edge

    class _GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, a):
            ctx.a = a
            return x.clone()

        @staticmethod
        def backward(ctx, g):
            return -ctx.a * g, None

    class _Net(nn.Module):
        def __init__(self, G):
            super().__init__()
            self.gat1 = GATv2Conv(64, 64, heads=4, dropout=0.2, concat=True)
            self.gat2 = GATv2Conv(256, 32, heads=4, dropout=0.2, concat=False)
            self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                      nn.Linear(32, 1), nn.Sigmoid())
            self.adv = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                                     nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, G))

        def forward(self, x, ei, a):
            h = F.elu(self.gat1(F.elu(x), ei))
            z = self.gat2(h, ei)
            return self.head(z), self.adv(_GRL.apply(z, a))

    torch.manual_seed(seed)
    n = graph.n
    ei = torch.as_tensor(np.stack([graph.src, graph.dst]), device=device)
    outd, ind = graph.out_degree.astype(np.float32), graph.in_degree.astype(np.float32)
    gen = torch.Generator().manual_seed(42)
    x = torch.cat([torch.as_tensor(outd / (outd.max() + 1e-8))[:, None],
                   torch.as_tensor(ind / (ind.max() + 1e-8))[:, None],
                   torch.as_tensor(np.log1p(outd + ind) / np.log1p((outd + ind).max()))[:, None],
                   torch.randn(n, 61, generator=gen) * 0.01], 1).to(device)
    adj = torch.sparse_coo_tensor(ei, torch.ones(ei.shape[1], device=device), (n, n)).coalesce()
    y = torch.as_tensor(graph.groups, device=device)
    G = graph.n_groups
    net = _Net(G).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50)
    best, best_state = np.inf, None
    for ep in range(epochs):
        net.train()
        a = min(ep / 300.0, 1.0)
        lam = max(0.0, min((ep - 200) / 200.0, 1.0)) * 100.0
        ei_tr, _ = dropout_edge(ei, p=0.2, force_undirected=True)
        p, sens = net(x, ei_tr, a)
        p = p.squeeze()
        nb = torch.sparse.mm(adj, p[:, None]).squeeze(1)
        spread = -torch.sum(1 - torch.exp(-(p + nb))) + 1.5 * torch.abs(p.sum() - k)
        expected = p + nb
        rates = torch.stack([expected[y == g].mean() for g in range(G) if (y == g).any()])
        fair = torch.var(rates) if len(rates) > 1 else torch.zeros((), device=device)
        loss = spread + F.cross_entropy(sens, y) + lam * fair
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        comb = spread.item() + lam * fair.item()
        if comb < best:
            best, best_state = comb, _copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p, _ = net(x, ei, 1.0)
    return torch.topk(p.squeeze(), k).indices.cpu().tolist()
