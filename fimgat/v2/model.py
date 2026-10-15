"""v2 networks (docs/v2_interfaces.md section 13).

PolicyV2              v1 SeedPolicy backbone (GATv2, residual pre-norm, explicit self-loops in the
                      edge list, mean-context readout) over static + v2 dynamic features, with two
                      heads: add scores (read for non-seeds) and drop scores (read for seeds).
CommunityAdversaryV2  GraphSAGE over the static features giving soft community memberships S (n, K).
                      It replaces v1's per-node CommunityAdversary, whose weights collapsed onto a
                      few nodes on Pokec: this adversary works at community level. It is trained to
                      maximise the policy's smooth community disparity, kept cohesive by the DMoN
                      modularity loss and kept away from tiny communities by size_penalty (the
                      minimum-size rule of the auditor, design_v2 section 5).
dmon_loss             DMoN (Tsitsulin et al. 2023) on an edge list: -modularity + collapse regulariser.
size_penalty          sum_k relu(m - |C_k|) / m on soft community sizes.
soft_community_rates  (S^T a) / (S^T 1): mean exposure of each soft community over RR roots.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch_geometric.nn import GATv2Conv, SAGEConv

from .features import EDGE_DIM, IN_DIM, N_STATIC


def _head(hidden):
    return nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))


class PolicyV2(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=64, layers=3, heads=4, dropout=0.1, checkpoint=False):
        super().__init__()
        self.in_dim = in_dim
        self.checkpoint = checkpoint          # recompute per-edge attention in backward (large graphs)
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        # self-loops are part of the edge list (edge feature is_self), so full-graph and chunked
        # evaluation are exactly the same function
        self.convs = nn.ModuleList(GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=EDGE_DIM,
                                             dropout=dropout, add_self_loops=False)
                                   for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.add_head, self.drop_head = _head(hidden), _head(hidden)
        self.dropout = dropout

    def _readout(self, h):
        z = torch.cat([h, h.mean(0, keepdim=True).expand_as(h)], 1)
        return self.add_head(z).squeeze(-1), self.drop_head(z).squeeze(-1)

    def _layer(self, i, h, edge_index, edge_attr):
        return F.dropout(F.gelu(self.convs[i](self.norms[i](h), edge_index, edge_attr)),
                         self.dropout, self.training)

    def forward(self, feats, edge_index, edge_attr):
        """(add_scores (n,), drop_scores (n,)): higher add = better node to add (read on non-seeds),
        higher drop = better seed to remove (read on seeds)."""
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
        """Layer-wise inference for graphs whose edge list lives on the CPU (as v1 SeedPolicy).
        Edges must be sorted by target; each chunk holds every in-edge of a contiguous block of
        targets, so attention softmaxes are exact."""
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

    def score(self, ctx, feats):
        """(add, drop) on ctx's graph: full-graph when the edges sit on ctx.device, else chunked
        (inference only), as v1 train.policy_scores."""
        if ctx.edge_device == ctx.device:
            return self(feats, ctx.edge_index, ctx.edge_attr)
        if torch.is_grad_enabled():
            raise RuntimeError("chunked (CPU-edge) mode is inference-only; call under torch.no_grad()")
        return self.forward_chunked(feats, ctx.edge_index, ctx.edge_attr, ctx.dst_ptr,
                                    max_edges=getattr(ctx, "chunk_edges", 20_000_000))


class CommunityAdversaryV2(nn.Module):
    """Soft memberships S (n, K), a softmax over K communities, from structural features only
    (never the labelled groups and never the evaluation partition)."""

    def __init__(self, in_dim=N_STATIC, K=16, hidden=32, layers=2):
        super().__init__()
        self.K = K
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(SAGEConv(hidden, hidden, aggr="mean") for _ in range(layers))
        self.out = nn.Linear(hidden, K)

    def forward(self, static, edge_index):
        h = F.gelu(self.inp(static))
        for conv in self.convs:
            h = h + F.gelu(conv(h, edge_index))
        return torch.softmax(self.out(h), dim=-1)


def dmon_loss(S, edge_index, n):
    """-Q(S) + sqrt(K)/n * ||sum_i S_i|| - 1 (Tsitsulin et al. 2023), from an edge list.

    Q = (1/2m) [sum_(i,j) A_ij <S_i, S_j> - ||S^T d||^2 / 2m], with A the multigraph of the
    edges given (self-loops dropped), d its degree vector (bincount of sources) and 2m = sum d.
    edge_index should be symmetric (as GraphContext.edge_index); duplicated edges act as weights.
    No dense n x n matrix is formed: memory is O(E K) for the gather (training graphs only)."""
    n, K = int(n), S.shape[1]
    collapse = math.sqrt(K) / n * S.sum(0).norm() - 1.0
    src, dst = edge_index[0].to(S.device), edge_index[1].to(S.device)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return collapse
    deg = torch.bincount(src, minlength=n).to(S.dtype)
    two_m = deg.sum()
    AS = torch.zeros_like(S).index_add(0, dst, S.index_select(0, src))      # (A S)_i = sum_{j->i} S_j
    within = (S * AS).sum()
    null = (deg @ S).pow(2).sum() / two_m
    return -(within - null) / two_m + collapse


def size_penalty(S, min_size):
    """sum_k relu(min_size - S[:, k].sum()) / min_size."""
    m = float(min_size)
    if m <= 0:
        return S.sum() * 0.0
    return torch.relu(m - S.sum(0)).sum() / m


def soft_community_rates(S_roots, a_roots):
    """(S^T a) / (S^T 1), (K,): S_roots (n_roots, K) memberships of the RR roots, a_roots (n_roots,)."""
    return (S_roots.t() @ a_roots) / S_roots.sum(0).clamp(min=1e-12)
