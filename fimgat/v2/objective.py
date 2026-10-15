"""v2 objective: group disparity under a spread floor (docs/design_v2.md section 3).

    D(S) = max_g r_g(S) - min_g r_g(S)      over present groups (or over communities)
    minimise D(S)  subject to  |S| = k,  total(S) >= rho * total_ref

The verifier scores the exact D. Training and Frank-Wolfe use the smooth upper bound
tau*logsumexp(r/tau) + tau*logsumexp(-r/tau), which lies in [D, D + 2 tau log G], inside an
augmented Lagrangian for the floor. The signed prices w_g = mu n_g / n + beta_g - alpha_g
steer search in rate space: beta lifts the least-served groups, alpha pushes the best-served
down, and mu buys spread. With alpha = 0 this reduces to v1's maximin weights, which let tiny
groups be over-served.
"""
import numpy as np
import torch

from .config import PRICE_TAU


def to_numpy(a, dtype=np.float64):
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=dtype)


def present_mask(present, G):
    """Bool numpy mask (G,); None means every group."""
    return np.ones(G, dtype=bool) if present is None else to_numpy(present, dtype=bool)


def disparity(rates, present=None):
    """max - min of rates over present entries; 0.0 when fewer than 2 are present."""
    r = to_numpy(rates)
    r = r[present_mask(present, r.shape[-1])]
    return float(r.max() - r.min()) if r.size >= 2 else 0.0


# ------------------------------------------------------------ smooth versions (torch)

def _present_idx(present, ref):
    if present is None:
        return torch.arange(ref.shape[-1], device=ref.device)
    if torch.is_tensor(present):
        m = present.to(ref.device)
    else:
        m = torch.as_tensor(np.ascontiguousarray(present, dtype=bool), device=ref.device)
    return torch.nonzero(m.bool()).flatten()


def smooth_max(r, present, tau):
    """tau * logsumexp(r / tau) over present entries (last dim); in [max, max + tau log G]."""
    idx = _present_idx(present, r)
    if idx.numel() == 0:
        return r.sum(-1) * 0.0
    return tau * torch.logsumexp(r.index_select(-1, idx) / tau, dim=-1)


def smooth_min(r, present, tau):
    """-tau * logsumexp(-r / tau) over present entries (last dim); in [min - tau log G, min]."""
    idx = _present_idx(present, r)
    if idx.numel() == 0:
        return r.sum(-1) * 0.0
    return -tau * torch.logsumexp(-r.index_select(-1, idx) / tau, dim=-1)


def smooth_disparity(rates_t, present, tau=PRICE_TAU):
    """smooth_max - smooth_min, differentiable, in [D, D + 2 tau log G_present]; 0 below 2 groups."""
    idx = _present_idx(present, rates_t)
    if idx.numel() < 2:
        return rates_t.sum(-1) * 0.0
    v = rates_t.index_select(-1, idx) / tau
    return tau * (torch.logsumexp(v, dim=-1) + torch.logsumexp(-v, dim=-1))


# ------------------------------------------------------------ signed prices (numpy)

def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def price_weights(alpha, beta, mu, sizes, present):
    """w = mu * sizes / sizes[present].sum() + beta - alpha on present groups, 0 elsewhere."""
    s = to_numpy(sizes)
    p = present_mask(present, len(s))
    w = np.zeros(len(s))
    tot = s[p].sum()
    spread = float(mu) * s[p] / tot if tot > 0 else 0.0
    w[p] = spread + to_numpy(beta)[p] - to_numpy(alpha)[p]
    return w


def signed_prices(rates, present, sizes, mu, tau=PRICE_TAU):
    """(w, alpha, beta), numpy (G,): alpha = softmax(r/tau) (over-served), beta = softmax(-r/tau)
    (under-served), both over present groups and 0 elsewhere."""
    r = to_numpy(rates)
    p = present_mask(present, len(r))
    alpha, beta = np.zeros(len(r)), np.zeros(len(r))
    if p.any():
        z = r[p] / tau
        alpha[p], beta[p] = _softmax(z), _softmax(-z)
    return price_weights(alpha, beta, mu, sizes, p), alpha, beta


# ------------------------------------------------------------ training loss (torch)

def disparity_lagrangian(rates, total, total_ref, rho, present, tau, mu, penalty,
                         comm_rates=None, comm_present=None, gamma=0.5):
    """Loss to minimise: smooth D (+ gamma * smooth D_C) + mu * short + penalty/2 * short^2,
    with short = relu(rho - total / total_ref). Works on (G,) or batched (C, G) rates."""
    loss = smooth_disparity(rates, present, tau)
    if comm_rates is not None:
        loss = loss + float(gamma) * smooth_disparity(comm_rates, comm_present, tau)
    ref = total_ref if torch.is_tensor(total_ref) else float(total_ref)
    gap = float(rho) - total / ref
    if not torch.is_tensor(gap):
        gap = rates.new_tensor(gap)
    short = torch.relu(gap)
    return loss + float(mu) * short + 0.5 * float(penalty) * short ** 2
