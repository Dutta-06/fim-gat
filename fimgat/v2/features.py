"""v2 node features (docs/v2_interfaces.md section 12).

Static features and the per-graph tensors (reach_rate, nbr_frac, full_exposure) are v1's
GraphContext. The dynamic block has 14 columns instead of v1's 12: deficit weights become the
signed prices w_g = mu n_g / N + beta_g - alpha_g of objective.signed_prices, beta (J_beta) gives
way to rho and the floor slack, and every seed carries its removal loss, so one network scores
adds (non-seeds) and drops (seeds).

Units. `gains` and `removal` are expected numbers of group-g nodes, (n, G), in est.group_gains
units. At fractional seeds x (training, multilinear extension) F is linear in each x_u, so with
d_u = dF/dx_u (est.group_gains):
    add gain of u      (1 - x_u) * d_u     value of raising x_u to 1
    removal loss of u  x_u * d_u           value lost by setting x_u to 0 (default of `removal`)
For a binary seed set these are exactly CoverState.gains() and CoverState.removal_loss() times the
group sizes, which is what dynamic_from_state passes, so training (multilinear, fractional x) and
inference (binary, CoverState) see the same quantities. The prices w act on *rates*, so every
w-weighted column contracts w with rate changes (counts / n_g): the add column is the simple
proposer's add score CoverState.gains() @ w, scaled by total_ref[k] before the signed log.

Invariance to group relabelling. No column depends on a group's label: group quantities enter only
as the node's own group (rate, price, rate gain) or contracted with w (gains, reach, neighbours,
removal), and w is computed entry-wise from (rates, sizes, present), so relabelling permutes w with
the gain columns and leaves every contraction unchanged; G enters only as a count. As in v1 this is
exact when every node is a root. With group-stratified root sampling the RR sample itself depends on
the label order (stratified_roots draws group by group), so invariance then holds in distribution.
"""
import numpy as np
import torch

from ..features import EDGE_DIM, N_STATIC, GraphContext, _tz  # noqa: F401  (EDGE_DIM re-exported)

N_DYNAMIC_V2 = 14
IN_DIM = N_STATIC + N_DYNAMIC_V2


def build_context(graph, est, device, edge_device=None):
    """v1 GraphContext (static features, message-passing edges, exposure imputation) plus
    present_np, the bool (G,) mask of interfaces section 1. total_ref is filled by the caller."""
    ctx = GraphContext(graph, est, device, edge_device=edge_device)
    ctx.total_ref = {}
    ctx.present_np = (graph.group_sizes > 0) & (est.root_counts > 0)
    return ctx


def signlog(z):
    return torch.sign(z) * torch.log1p(z.abs())


def _t(a, dev):
    """float32 tensor on dev (from numpy, a python scalar or a tensor on any device)."""
    if torch.is_tensor(a):
        return a.detach().to(device=dev, dtype=torch.float32)
    return torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float32, device=dev)


def _dynamic(ctx, x, a_roots, rates, total, gains, w, rho, t, rounds, k, total_ref, removal):
    n, G, dev = ctx.n, ctx.G, ctx.device
    x, a_roots, rates, gains, w = (_t(v, dev) for v in (x, a_roots, rates, gains, w))
    total, total_ref = float(total), max(float(total_ref), 1e-12)
    own = ctx.own
    a = ctx.full_exposure(a_roots)
    norm = rates / (total / n + 1e-9)
    add = (1.0 - x)[:, None] * gains                                   # (n, G) counts
    rem = x[:, None] * gains if removal is None else _t(removal, dev)
    rate_add = add / ctx.sizes                                         # (n, G) rate gained per group
    seed = x > 0.5
    drop = torch.zeros(n, device=dev)
    if int(seed.sum()) >= 2:                                           # z-scored among seeds only
        drop[seed] = _tz(signlog((rem[seed] / ctx.sizes) @ w * total_ref))
    dyn = torch.stack([
        x * n / (10.0 * max(int(k), 1)),
        a,
        norm[own],
        w[own] * G,
        _tz(signlog(rate_add @ w * total_ref)),
        _tz(torch.log1p(add.sum(1))),
        _tz(torch.log1p(rate_add.gather(1, own[:, None]).squeeze(1) * n)),
        _tz(signlog(ctx.reach_rate @ w)),
        ctx.nbr_frac @ w,
        torch.full_like(a, float(rho)),
        torch.full_like(a, total / total_ref - float(rho)),
        torch.full_like(a, t / max(rounds - 1, 1)),
        drop,
        seed.float(),
    ], 1)
    assert dyn.shape[1] == N_DYNAMIC_V2
    return dyn


@torch.no_grad()
def dynamic_v2(ctx, x, a_roots, rates, total, gains, w, rho, t, rounds, k, removal=None):
    """(n, 14) float32 features on ctx.device.

    x (n,) fractional seeds; a_roots (n_roots,) exposure; rates (G,); total (float);
    gains (n, G) = est.group_gains(x, miss), i.e. dF_g/dx_u in expected numbers of nodes;
    w (G,) signed prices; removal (n, G) removal loss in the same units (default x * gains)."""
    k = int(k)
    return _dynamic(ctx, x, a_roots, rates, total, gains, w, rho, t, rounds, k, ctx.total_ref[k], removal)


@torch.no_grad()
def dynamic_from_state(ctx, state, w, rho, total_ref, t, rounds):
    """dynamic_v2 for a binary seed set held in a CoverState (float64 numpy, exact on the RR sample).

    x is the seed indicator, a_roots = state.exposure(), gains = state.gains() * sizes and
    removal = state.removal_loss() * sizes on the seed rows (0 elsewhere): rates converted to
    expected numbers of nodes, the units of est.group_gains."""
    n, dev = ctx.n, ctx.device
    expo = np.asarray(state.exposure(), dtype=np.float64)
    if expo.shape[0] != ctx.est.n_roots:
        raise ValueError("the CoverState was built on an estimator with other RR roots than ctx.est")
    seeds = np.asarray(state.seeds, dtype=np.int64)
    sizes = ctx.graph.group_sizes.astype(np.float64)
    seeds_t = torch.as_tensor(seeds, device=dev)
    x = torch.zeros(n, device=dev)
    x[seeds_t] = 1.0
    # UNVERIFIED: CoverState.gains() is (n, G) with zero seed rows and removal_loss() is (k, G) in
    # the order of state.seeds (contract section 5); cover.py did not exist when this was written.
    # The seed rows of gains are zeroed again by the (1 - x) factor in _dynamic.
    gains = _t(np.asarray(state.gains(), dtype=np.float64) * sizes[None, :], dev)
    removal = torch.zeros(n, ctx.G, device=dev)
    if len(seeds):
        removal[seeds_t] = _t(np.asarray(state.removal_loss(), dtype=np.float64) * sizes[None, :], dev)
    return _dynamic(ctx, x, expo, np.asarray(state.rates, dtype=np.float64), float(state.total), gains, w,
                    rho, t, rounds, len(seeds), total_ref, removal)
