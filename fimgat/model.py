"""FIM-GAT networks.

SeedPolicy      GATv2 generator: scores nodes from static + dynamic features; applied
                for several refinement rounds with shared weights, each round seeing
                the exposure field and marginal gains of the previous round's
                fractional seed set.
soft_topk       Exact-budget differentiable relaxation of top-k.

v1's adversaries (CommunityAdversary, ExposureCritic) and its trainer were removed on 2026-10-10; v2's
adversary is fimgat/v2/model.CommunityAdversaryV2.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch_geometric.nn import GATv2Conv

from .features import EDGE_DIM, N_DYNAMIC, N_STATIC


def _solve_tau(s, k, T, grid=32, passes=3, max_elems=int(5e7)):
    """tau with sum sigmoid((s - tau)/T) = k: a few vectorised grid refinements
    (grid^passes bracket shrink) plus linear interpolation. No host syncs."""
    lo, hi = s.min() - 20 * T, s.max() + 20 * T
    steps = torch.linspace(0, 1, grid + 1, device=s.device)
    chunk = max(1, max_elems // s.numel())
    for _ in range(passes):
        taus = lo + (hi - lo) * steps                                   # (grid+1,)
        f = torch.cat([torch.sigmoid((s[None, :] - taus[i:i + chunk, None]) / T).sum(1)
                       for i in range(0, grid + 1, chunk)])             # decreasing in tau
        j = (f > k).sum().clamp(1, grid)                                # f[j-1] > k >= f[j]
        lo, hi = taus[j - 1], taus[j]
        f_lo, f_hi = f[j - 1], f[j]
    return lo + (hi - lo) * ((f_lo - k) / (f_lo - f_hi).clamp(min=1e-12))


def soft_topk(scores, k, T):
    """x = sigmoid((s - tau)/T) with tau chosen so that sum(x) = k.

    Scores are standardised first so T has a fixed meaning. tau's implicit
    gradient is restored with a first-order term, so gradients move mass
    between nodes but never change the budget.
    """
    s = (scores - scores.mean()) / (scores.std() + 1e-6)
    with torch.no_grad():
        tau = _solve_tau(s, k, T)
        x0 = torch.sigmoid((s - tau) / T)
        w = x0 * (1 - x0)
    tau_corr = (w * (s - s.detach())).sum() / (w.sum() + 1e-12)
    return torch.sigmoid((s - tau - tau_corr) / T)


class SeedPolicy(nn.Module):
    def __init__(self, hidden=64, layers=3, heads=4, dropout=0.1, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint          # recompute per-edge attention in backward (large graphs)
        self.inp = nn.Sequential(nn.Linear(N_STATIC + N_DYNAMIC, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))
        # self-loops are part of the edge list (edge feature is_self), so full-graph and
        # chunked evaluation are exactly the same function
        self.convs = nn.ModuleList(GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=EDGE_DIM,
                                             dropout=dropout, add_self_loops=False)
                                   for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.dropout = dropout

    def _readout(self, h):
        ctx = h.mean(0, keepdim=True).expand_as(h)
        return self.head(torch.cat([h, ctx], 1)).squeeze(-1)

    def _layer(self, i, h, edge_index, edge_attr):
        return F.dropout(F.gelu(self.convs[i](self.norms[i](h), edge_index, edge_attr)),
                         self.dropout, self.training)

    def forward(self, feats, edge_index, edge_attr):
        h = self.inp(feats)
        for i in range(len(self.convs)):
            if self.checkpoint and torch.is_grad_enabled():
                h = h + torch.utils.checkpoint.checkpoint(self._layer, i, h, edge_index, edge_attr,
                                                          use_reentrant=False)
            else:
                h = h + self._layer(i, h, edge_index, edge_attr)
        return self._readout(h)

    @torch.no_grad()
    def forward_chunked(self, feats, edge_index, edge_attr, dst_ptr, max_edges=20_000_000):
        """Layer-wise inference for graphs whose edge list lives on the CPU.
        Edges must be sorted by target; each chunk holds every in-edge of a
        contiguous block of targets, so attention softmaxes are exact."""
        dev = feats.device
        n = feats.shape[0]
        bounds, lo = [], 0
        while lo < n:
            hi = int(np.searchsorted(dst_ptr, dst_ptr[lo] + max_edges, side="right")) - 1
            hi = min(max(hi, lo + 1), n)
            bounds.append((lo, hi))
            lo = hi
        h = self.inp(feats)
        for conv, norm in zip(self.convs, self.norms):
            hn = norm(h)
            out = torch.empty_like(h)
            for lo, hi in bounds:
                e0, e1 = int(dst_ptr[lo]), int(dst_ptr[hi])
                ei = edge_index[:, e0:e1].to(dev, non_blocking=True)
                ea = edge_attr[e0:e1].to(dev, non_blocking=True)
                ei = torch.stack([ei[0], ei[1] - lo])
                out[lo:hi] = conv((hn, hn[lo:hi]), ei, ea)
            h = h + F.gelu(out)
        return self._readout(h)
