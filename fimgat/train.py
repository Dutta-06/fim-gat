"""The v1 policy's context (prepare), rollout and decoding (select_seeds). The foundation tests exercise them; v2
has its own versions in fimgat/v2/train.py. The v1 trainer was removed on 2026-10-10 (design_v2.md section 19)."""
import numpy as np
import torch

from .baselines import RRGreedy
from .decode import decode
from .diffusion import ExposureEstimator
from .features import GraphContext
from .model import soft_topk
from .objective import objective_group_weights


def prepare(graph, k, device, R=32, rr_seed=0, max_roots=None, edge_device=None):
    """Estimator + features + greedy reference spread for one graph and budget(s)."""
    est = ExposureEstimator(graph, R=R, seed=rr_seed, device=device, max_roots=max_roots)
    ctx = GraphContext(graph, est, device, edge_device=edge_device)
    ctx.greedy_seeds = {}
    for kk in np.atleast_1d(k):
        kk = int(kk)
        seeds = RRGreedy(est).greedy(kk)
        ctx.greedy_seeds[kk] = seeds
        ctx.total_ref[kk] = est.evaluate_set(seeds)[1]
    return ctx


def policy_scores(policy, ctx, feats):
    if ctx.edge_device == ctx.device:
        return policy(feats, ctx.edge_index, ctx.edge_attr)
    assert not torch.is_grad_enabled(), "chunked (CPU-edge) mode is inference-only"
    return policy.forward_chunked(feats, ctx.edge_index, ctx.edge_attr, ctx.dst_ptr,
                                  max_edges=getattr(ctx, "chunk_edges", 20_000_000))


def rollout(policy, ctx, k, beta, T, rounds, grad_round=None, update="replace"):
    """Gradient-informed refinement rounds on the multilinear extension.

    Every round the policy sees the exposure field, per-group marginal gains and the
    exact gradient of J_beta at the current fractional seeds x_t and outputs scores s_t.

    update="replace" (default): x_{t+1} = softtopk(s_t, k). Each round proposes a whole,
        near-binary seed set (T is annealed), so the policy optimises actual seed sets.
        This matters for maximin objectives, whose fractional relaxation overstates
        what integral sets achieve (softmin is concave).
    update="batch": learned batched greedy, x_{t+1} = x_t + softtopk(s_t - mask, k/rounds):
        each round adds k/rounds near-hard seeds chosen given the current set (integral
        like "replace", overlap-aware like greedy).
    update="cg": learned continuous greedy, x_{t+1} = x_t + softtopk(s_t, k) / rounds
        (classical continuous greedy is s_t = grad J). Kept as an ablation.

    grad_round: if set, only that round's policy call is differentiated (memory of a
    single round). In "replace" mode the rollout stops there and the loss is J at that
    round's output; in "cg" mode it runs to the end and the loss is J at the final
    iterate. Returns [(scores, x_{t+1})] per round."""
    est = ctx.est
    if update == "replace":
        x = torch.full((ctx.n,), k / ctx.n, device=ctx.device)
        # each round overwrites x, so a truncated rollout ends at the differentiated round
        last = rounds if grad_round is None else grad_round + 1
    else:
        x = torch.zeros(ctx.n, device=ctx.device)
        last = rounds
    per = k / rounds
    out = []
    for t in range(last):
        with torch.no_grad():
            a, rates, total, miss = est(x)
            gains = est.group_gains(x, miss)
            w = objective_group_weights(rates, ctx.n, ctx.total_ref[k], beta, ctx.present_idx, est.sizes)
            dyn = ctx.dynamic(x, a, rates, total, gains, gains @ w, beta, t, rounds, k)
        with torch.set_grad_enabled(torch.is_grad_enabled() and (grad_round is None or t == grad_round)):
            s = policy_scores(policy, ctx, torch.cat([ctx.static, dyn], 1))
            if update == "batch":
                # next batch of k/rounds seeds; already-chosen mass is masked out
                zs = (s - s.mean()) / (s.std() + 1e-6) - 20.0 * x.detach()
                v = soft_topk(zs, per, T)
            else:
                v = soft_topk(s, k, T)
        if update == "replace":
            x = v
        elif update == "batch":
            x = torch.clamp(x + v, max=1.0)
        else:
            x = x + v / rounds
        out.append((s, x))
    return out


@torch.no_grad()
def select_seeds(policy, ctx, k, beta, rounds, T=0.05, n_samples=64, greedy_only=False, update="replace"):
    """Seeds for one graph: decode the final fractional iterate. greedy_only=True is
    plain top-k; otherwise sample-and-select on the graph's own RR sample."""
    policy.eval()
    outs = rollout(policy, ctx, k, beta, T, rounds, update=update)
    s, x = outs[-1]
    zs = (s - s.mean()) / (s.std() + 1e-6)
    key = x + 1e-3 * zs / rounds                    # final mass, ties broken by the last scores
    if greedy_only:
        return torch.topk(key, k).indices.cpu().tolist()
    return decode(ctx.est, key, k, beta, ctx.total_ref[k], n_samples=n_samples,
                  extra=[o[0] for o in outs])[0]
