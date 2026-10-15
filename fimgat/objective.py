"""The fairness-efficiency objective shared by FIM-GAT and the FW-Fair baseline.

    J_beta(x) = (1 - beta) * total(x) / total_ref  +  beta * softmin_g rate_g(x) / (total_ref / n)

total_ref is the greedy (unconstrained) spread on the same RR sample, so J is
scale-free and comparable across graphs. beta = 0 is plain influence
maximisation; beta = 1 is maximin group fairness (Tsang et al. 2019).
"""
import math

import torch


def softmin(values, idx, tau):
    """Smooth lower bound of min over entries idx (LongTensor); exact when all are equal."""
    v = values.index_select(-1, idx)
    return -tau * (torch.logsumexp(-v / tau, dim=-1) - math.log(v.shape[-1]))


@torch.no_grad()
def objective_group_weights(rates, n, total_ref, beta, present_idx, sizes, tau=0.05):
    """w (G,) with dJ/dx_u = sum_g gains[u, g] * w_g, where gains[u, g] = d(#active in g)/dx_u.

    dJ/d rate_g = (1-beta) n_g / ref + beta * softmin_weight_g * n / ref, and
    d rate_g / dx_u = gains[u, g] / n_g.
    """
    v = rates.index_select(0, present_idx) * (n / total_ref)
    sm = torch.zeros_like(rates).index_copy(0, present_idx, torch.softmax(-v / tau, 0))
    return ((1.0 - beta) + beta * sm * n / sizes.clamp(min=1)) / total_ref


def fairness_objective(rates, total, n, total_ref, beta, present_idx, tau=0.05):
    eff = total / total_ref
    fair = softmin(rates * (n / total_ref), present_idx, tau)
    return (1.0 - beta) * eff + beta * fair, eff, fair
