"""Sample-and-select decoding from node scores to a seed set.

Candidates are the deterministic top-k plus Gumbel-top-k samples at several
temperatures. Each is scored by J_beta on the graph's own RR sample (never the
evaluation Monte Carlo) and the best is returned. Used identically for FIM-GAT
and for the FW-Fair baseline.
"""
import torch

from .objective import fairness_objective


@torch.no_grad()
def batch_J(est, X, total_ref, beta):
    """J_beta for C seed vectors at once. X: (C, n)."""
    rates, total = est.batch_exposure_rates(X)
    J, _, _ = fairness_objective(rates, total, est.graph.n, total_ref, beta, est.present_idx)
    return J


@torch.no_grad()
def decode(est, scores, k, beta, total_ref, n_samples=64, temps=(0.05, 0.1, 0.2, 0.4), seed=0,
           extra=()):
    """scores: final node scores (n,). extra: additional score vectors whose top-k are also candidates."""
    n = est.graph.n
    s = (scores - scores.mean()) / (scores.std() + 1e-6)
    cands = [torch.topk(s, k).indices] + [torch.topk(e, k).indices for e in extra]
    gen = torch.Generator(device=s.device).manual_seed(seed)
    for i in range(n_samples):
        u = torch.rand(n, generator=gen, device=s.device).clamp(1e-9, 1 - 1e-9)
        gumbel = -torch.log(-torch.log(u))
        cands.append(torch.topk(s / temps[i % len(temps)] + gumbel, k).indices)
    X = torch.zeros(len(cands), n, device=s.device)
    for c, idx in enumerate(cands):
        X[c, idx] = 1.0
    J = batch_J(est, X, total_ref, beta)
    best = int(torch.argmax(J))
    return cands[best].cpu().tolist(), float(J[best])
