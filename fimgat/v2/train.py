"""Training, model selection and seed selection for the v2 constructor (docs/v2_interfaces.md section 14).

prepare_v2       estimator, v1 GraphContext features and the RR-greedy reference set and total per budget.
rollout_v2       v1's "replace" dynamics: each round proposes a whole soft top-k set from the add scores,
                 given dynamic_v2 features of the previous round's fractional seeds and the signed prices
                 objective.signed_prices(rates, present, sizes, mu).
V2Trainer        per step, the community adversary (soft memberships from structure only) takes two ascent
                 steps on the policy's smooth community disparity, regularised by DMoN modularity and a
                 minimum community size; then the policy takes one descent step on
                 objective.disparity_lagrangian including that community term, and the spread price mu,
                 a dual variable per (graph, budget), moves by dual ascent. One random round is
                 backpropagated (truncate), as in v1, so memory is that of a single round.
select_seeds_v2  sample-and-select decoding (v1 decode.py): candidates are ranked on the graph's own RR
                 sample and the lowest exact (float64) disparity meeting the floor wins, else RR-greedy.

Training targets the acceptance floor config.rr_floor(rho, total_ref) = min(rho + DELTA_MARGIN, 1) * total_ref,
while the features show rho itself, as the proposer does at inference. Only the add head is trained here;
the drop head gets no gradient from the constructor objective (it is meant for the learned proposer,
design_v2 section 7).
Checkpoints are chosen on validation graphs only, from RR samples, never from Monte Carlo.
"""
import copy
import math
import time

import numpy as np
import torch
from numba import njit

from ..baselines import RRGreedy
from ..diffusion import ExposureEstimator
from ..model import soft_topk
from .config import MIN_COMMUNITY, PRICE_TAU, rr_floor
from .features import IN_DIM, build_context, dynamic_v2
from .model import CommunityAdversaryV2, PolicyV2, dmon_loss, size_penalty, soft_community_rates
from .objective import disparity, disparity_lagrangian, signed_prices, smooth_disparity

TEMPS = (0.05, 0.1, 0.2, 0.4)
LOOSE = 1e-4          # float32 batch totals this far (relative) below the floor still get the exact check


# ------------------------------------------------------------------ exact RR evaluation (float64)

@njit(cache=True)
def _root_cover(ptr, nodes, is_seed, R):
    """Covered RR sets per root (sets are root-major: set j belongs to root j // R)."""
    n_sets = ptr.shape[0] - 1
    cov = np.zeros(n_sets // R, dtype=np.int64)
    for j in range(n_sets):
        for t in range(ptr[j], ptr[j + 1]):
            if is_seed[nodes[t]]:
                cov[j // R] += 1
                break
    return cov


def exact_rates(est, seeds):
    """Float64 group rates (G,) and RR total of a binary seed set (interfaces section 1)."""
    is_seed = np.zeros(est.graph.n, dtype=np.bool_)
    is_seed[np.asarray(seeds, dtype=np.int64)] = True
    cov = _root_cover(est.ptr, est.nodes, is_seed, int(est.R))
    hits = np.bincount(est.root_groups_np, weights=cov.astype(np.float64), minlength=est.graph.n_groups)
    rates = hits / (est.R * np.maximum(est.root_counts, 1.0))
    return rates, float(rates @ est.graph.group_sizes)


def _present(ctx):
    p = getattr(ctx, "present_np", None)
    return (ctx.graph.group_sizes > 0) & (ctx.est.root_counts > 0) if p is None else p


# ------------------------------------------------------------------ preparation and rollout

def prepare_v2(graph, ks, device, R=16, rr_seed=0, max_roots=None, edge_device=None):
    """Estimator + v2 context + RR-greedy reference per budget: ctx.greedy_seeds[k] and
    ctx.total_ref[k] (its exact float64 RR total, the base of the floor)."""
    est = ExposureEstimator(graph, R=R, seed=rr_seed, device=device, max_roots=max_roots)
    ctx = build_context(graph, est, device, edge_device=edge_device)
    ks = sorted({int(k) for k in np.atleast_1d(ks)})
    if ks and not 1 <= ks[0] <= ks[-1] <= graph.n:
        raise ValueError(f"budgets {ks} outside [1, n={graph.n}]")
    ctx.greedy_seeds = {}
    if ks:
        # greedy is deterministic, so greedy(k) is the k-prefix of greedy(max k): one run serves all budgets
        full = RRGreedy(est).greedy(ks[-1])
        for k in ks:
            ctx.greedy_seeds[k] = [int(v) for v in full[:k]]
            ctx.total_ref[k] = exact_rates(est, ctx.greedy_seeds[k])[1]
    return ctx


def rollout_v2(policy, ctx, k, rho, T, rounds, mu, grad_round=None):
    """Replace dynamics: x_0 = k/n, x_{t+1} = soft_topk(add_t, k). grad_round: only that round's policy
    call is differentiated and the rollout stops there. Returns [(add_scores, x_{t+1})] per round."""
    est, k = ctx.est, int(k)
    present, sizes = _present(ctx), ctx.graph.group_sizes
    x = torch.full((ctx.n,), k / ctx.n, device=ctx.device)
    last = rounds if grad_round is None else min(int(grad_round), rounds - 1) + 1
    out = []
    for t in range(last):
        with torch.no_grad():
            a, rates, total, miss = est(x)
            gains = est.group_gains(x, miss)
            w, _, _ = signed_prices(rates, present, sizes, mu)
            dyn = dynamic_v2(ctx, x, a, rates, total, gains, w, rho, t, rounds, k)
        with torch.set_grad_enabled(torch.is_grad_enabled() and (grad_round is None or t == grad_round)):
            add, _ = policy.score(ctx, torch.cat([ctx.static, dyn], 1))
            x = soft_topk(add, k, T)
        out.append((add, x))
    return out


# ------------------------------------------------------------------ decoding

@torch.no_grad()
def _candidates(key, k, n_samples, temps, seed, extra=()):
    """Distinct candidate sets (int64 numpy): top-k of the z-scored key, top-k of each extra score
    vector, and Gumbel-top-k samples of the key at several temperatures (v1 decode)."""
    s = (key - key.mean()) / (key.std() + 1e-6)
    cands = [torch.topk(s, k).indices] + [torch.topk(e, k).indices for e in extra]
    gen = torch.Generator(device=s.device).manual_seed(seed)
    for i in range(n_samples):
        u = torch.rand(s.shape[0], generator=gen, device=s.device).clamp(1e-9, 1 - 1e-9)
        cands.append(torch.topk(s / temps[i % len(temps)] - torch.log(-torch.log(u)), k).indices)
    out, seen = [], set()
    for c in cands:
        idx = c.cpu().numpy().astype(np.int64)
        tag = np.sort(idx).tobytes()
        if tag not in seen:
            seen.add(tag)
            out.append(idx)
    return out


@torch.no_grad()
def _rank(est, cands, chunk=16):
    """Float32 disparity and RR total of every candidate, for ranking only."""
    n, dev, pidx = est.graph.n, est.device, est.present_idx
    disp, tot = [], []
    for c0 in range(0, len(cands), chunk):
        part = cands[c0:c0 + chunk]
        X = torch.zeros(len(part), n, device=dev)
        for i, idx in enumerate(part):
            X[i, torch.as_tensor(idx, device=dev)] = 1.0
        rates, total = est.batch_exposure_rates(X)
        r = rates.index_select(1, pidx)
        d = r.max(1).values - r.min(1).values if r.shape[1] >= 2 else torch.zeros(len(part), device=dev)
        disp.append(d.double().cpu())
        tot.append(total.double().cpu())
    return torch.cat(disp).numpy(), torch.cat(tot).numpy()


def _select(est, cands, floor, present, n_exact):
    """(disp, total, index) of the lowest exact disparity among candidates whose exact RR total meets the
    floor (ties: larger total), checking candidates in float32 order until n_exact pass; None if none."""
    disp32, tot32 = _rank(est, cands)
    best, n_ok = None, 0
    for i in np.lexsort((-tot32, disp32)):
        # UNVERIFIED: LOOSE assumes float32 batch totals are within 1e-4 (relative) of the float64 ones
        if tot32[i] < floor * (1.0 - LOOSE):
            continue
        rates, total = exact_rates(est, cands[i])
        if total < floor:
            continue
        d = disparity(rates, present)
        if best is None or (d, -total) < (best[0], -best[1]):
            best = (d, total, int(i))
        n_ok += 1
        if n_ok >= n_exact:
            break
    return best


@torch.no_grad()
def select_seeds_v2(policy, ctx, k, rho, floor, rounds, n_samples=64, mu=0.0, T=0.05, temps=TEMPS, seed=0,
                    n_exact=8, return_info=False):
    """Seeds for one query: decode the rollout on the graph's own RR sample. Returns ctx.greedy_seeds[k]
    when no candidate meets floor (an exact RR total). mu: spread price inside the features (the trainer
    passes its learned per-budget price, V2Trainer.mu_for)."""
    policy.eval()
    k = int(k)
    outs = rollout_v2(policy, ctx, k, rho, T, rounds, mu)
    s, x = outs[-1]
    zs = (s - s.mean()) / (s.std() + 1e-6)
    key = x + 1e-3 * zs / rounds                    # final mass, ties broken by the last scores
    cands = _candidates(key, k, n_samples, temps, seed, extra=[o[0] for o in outs])
    best = _select(ctx.est, cands, float(floor), _present(ctx), n_exact)
    info = {"n_candidates": len(cands), "fallback": best is None}
    if best is None:
        ref = getattr(ctx, "greedy_seeds", {}).get(k)
        seeds = [int(v) for v in (RRGreedy(ctx.est).greedy(k) if ref is None else ref)]
    else:
        seeds = cands[best[2]].tolist()
        info.update(disp=best[0], total=best[1])
    return (seeds, info) if return_info else seeds


# ------------------------------------------------------------------ trainer

def _nanmean(vals):
    """Mean over the non-NaN values; NaN (without a warning) when there are none."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def _plain(obj):
    """Python primitives only, so checkpoints load with torch.load(weights_only=True)."""
    if isinstance(obj, dict):
        return {(_plain(k) if isinstance(k, (np.generic, int, float, str, bool)) else str(k)): _plain(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


class V2Trainer:
    def __init__(self, device, seed=0, rounds=3, lr=3e-3, adv_lr=3e-3, K_comm=16, gamma_comm=0.5,
                 min_size=MIN_COMMUNITY, penalty=10.0, checkpoint=False, adv_steps=2, tau=PRICE_TAU,
                 dual_lr=1.0, rho_range=(0.95, 1.0), hidden=64, layers=3, heads=4, truncate=True):
        """gamma_comm: weight of the adversary's community disparity in the policy loss. min_size: soft
        community size below which a community is penalised and left out of the disparity. penalty:
        quadratic floor penalty of the augmented Lagrangian. dual_lr: step of the dual ascent on mu."""
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.rounds, self.gamma_comm, self.min_size = int(rounds), float(gamma_comm), float(min_size)
        self.penalty, self.adv_steps, self.tau = float(penalty), int(adv_steps), float(tau)
        self.dual_lr, self.rho_range, self.truncate = float(dual_lr), tuple(rho_range), truncate
        self.policy = PolicyV2(IN_DIM, hidden, layers, heads, checkpoint=checkpoint).to(self.device)
        self.adversary = CommunityAdversaryV2(K=K_comm).to(self.device)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.opt_adv = torch.optim.Adam(self.adversary.parameters(), lr=adv_lr)
        self.hparams = dict(seed=seed, rounds=self.rounds, lr=lr, adv_lr=adv_lr, K_comm=K_comm,
                            gamma_comm=self.gamma_comm, min_size=self.min_size, penalty=self.penalty,
                            adv_steps=self.adv_steps, tau=self.tau, dual_lr=self.dual_lr,
                            rho_range=list(self.rho_range), hidden=hidden, layers=layers, heads=heads,
                            in_dim=IN_DIM, truncate=truncate)
        self.mu = {}                 # (id(ctx), k) -> spread price, by dual ascent
        self.mu_ref = {}             # k -> price loaded from a checkpoint
        self.log = []
        self.train_time = 0.0
        self.val_fallback = float("nan")

    # ---------------------------------------------------------- adversary

    def _memberships(self, ctx):
        """S (n, K) and its rows at the RR roots (the only nodes with an exposure)."""
        S = self.adversary(ctx.static, ctx.edge_index)
        return S, (S if ctx.est.all_roots else S.index_select(0, ctx.est.roots_t))

    @torch.no_grad()
    def _comm_present(self, S, S_roots):
        """Communities counted in the disparity: at least min_size nodes and one root of soft mass."""
        return (S.sum(0) >= self.min_size) & (S_roots.sum(0) >= 1.0)

    def mu_for(self, k):
        """Spread price for budget k: mean over training graphs, else the checkpoint's, else 0."""
        k = int(k)
        vals = [v for (_, kk), v in self.mu.items() if kk == k]
        return float(np.mean(vals)) if vals else float(self.mu_ref.get(k, 0.0))

    def mu_table(self):
        ks = {kk for _, kk in self.mu} | set(self.mu_ref)
        return {int(k): self.mu_for(k) for k in sorted(ks)}

    # ---------------------------------------------------------- one step

    def step(self, ctx, k, rho, T):
        if ctx.edge_device != ctx.device:
            raise RuntimeError("training needs ctx's edge list on ctx.device (chunked mode is inference-only)")
        self.policy.train()
        self.adversary.train()
        est, k = ctx.est, int(k)
        ref = max(float(ctx.total_ref[k]), 1e-12)
        rho_eff = rr_floor(rho, 1.0)       # min(rho + DELTA_MARGIN, 1): improve()'s capped floor (design section 18)
        key = (id(ctx), k)
        mu = self.mu.get(key, 0.0)
        grad_round = int(self.rng.integers(self.rounds)) if self.truncate else None
        x = rollout_v2(self.policy, ctx, k, rho, T, self.rounds, mu, grad_round=grad_round)[-1][1]
        a, rates, total, _ = est(x)

        nan = float("nan")
        comm = cp = None
        dmon = comm_disp = n_comm = eff_comm = nan
        if self.gamma_comm > 0:                  # gamma_comm = 0 is the no-adversary ablation
            # adversary: cohesive, not-tiny soft communities where the (detached) exposure is most unequal
            a_det = a.detach()
            for _ in range(self.adv_steps):
                S, S_r = self._memberships(ctx)
                dm = dmon_loss(S, ctx.edge_index, ctx.n)
                adv_obj = (smooth_disparity(soft_community_rates(S_r, a_det), self._comm_present(S, S_r), self.tau)
                           - dm - size_penalty(S, self.min_size))
                self.opt_adv.zero_grad()
                (-adv_obj).backward()
                self.opt_adv.step()
                dmon = float(dm.detach())
            with torch.no_grad():
                S, S_r = self._memberships(ctx)
                cp = self._comm_present(S, S_r)
                p = S.sum(0) / S.sum()
                eff_comm = float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))   # in [1, K], scale-free
                n_comm = float(cp.sum())
            comm = soft_community_rates(S_r, a)          # memberships fixed, differentiable in the exposure

        # policy: smooth disparity (+ gamma * smooth community disparity) under the floor
        loss = disparity_lagrangian(rates, total, ref, rho_eff, est.present, self.tau, mu, self.penalty,
                                    comm_rates=comm, comm_present=cp, gamma=self.gamma_comm)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        if comm is not None:
            comm_disp = disparity(comm.detach(), cp)
        ratio = float(total.detach()) / ref
        self.mu[key] = max(0.0, mu + self.dual_lr * (rho_eff - ratio))
        return {"loss": float(loss.detach()), "disp": disparity(rates.detach(), est.present),
                "comm_disp": comm_disp, "ratio": ratio, "mu": mu, "dmon": dmon, "n_comm": n_comm,
                "eff_comm": eff_comm}

    # ---------------------------------------------------------- validation and fitting

    @torch.no_grad()
    def validate(self, val_ctxs, ks, rhos=(0.99,)):
        """Mean over (graph, k, rho) of -disparity of select_seeds_v2 (exact, on each graph's own RR
        sample), minus 1 for each case whose seeds miss the floor config.rr_floor(rho, total_ref).
        The share of cases that fell back to the RR-greedy set is kept in self.val_fallback."""
        scores, fallbacks = [], []
        for ctx in val_ctxs:
            present = _present(ctx)
            for k in ks:
                k = int(k)
                for rho in rhos:
                    floor = rr_floor(rho, ctx.total_ref[k])
                    seeds, info = select_seeds_v2(self.policy, ctx, k, rho, floor, self.rounds, n_samples=16,
                                                  mu=self.mu_for(k), return_info=True)
                    rates, total = exact_rates(ctx.est, seeds)
                    scores.append(-disparity(rates, present) - (1.0 if total < floor else 0.0))
                    fallbacks.append(float(info["fallback"]))
        self.val_fallback = float(np.mean(fallbacks)) if fallbacks else float("nan")
        return float(np.mean(scores)) if scores else float("nan")

    def _snapshot(self):
        return {"policy": copy.deepcopy(self.policy.state_dict()),
                "adversary": copy.deepcopy(self.adversary.state_dict()), "mu": dict(self.mu)}

    def _restore(self, snap):
        self.policy.load_state_dict(snap["policy"])
        self.adversary.load_state_dict(snap["adversary"])
        self.mu = dict(snap["mu"])

    def fit(self, train_ctxs, val_ctxs, ks, epochs, steps_per_epoch=None, eval_every=5, verbose=True,
            T0=1.0, T1=0.1):
        """Per step: a training graph, k from ks, rho ~ U(rho_range), T annealed T0 -> T1 (cosine).
        Validates every eval_every epochs and at the end, and restores the best validated state
        (the last one when there are no validation graphs). Returns the best validation score."""
        if not train_ctxs:
            raise ValueError("no training graphs")
        ks = [int(v) for v in np.atleast_1d(ks)]
        best, snap, t0 = -math.inf, None, time.time()
        per_epoch = int(steps_per_epoch or len(train_ctxs))
        steps, it = epochs * per_epoch, 0
        for ep in range(epochs):
            order = np.resize(self.rng.permutation(len(train_ctxs)), per_epoch)
            stats = []
            for i in order:
                T = T1 + 0.5 * (T0 - T1) * (1 + math.cos(math.pi * it / max(steps, 1)))
                k = int(self.rng.choice(ks))
                rho = float(self.rng.uniform(*self.rho_range))
                stats.append(self.step(train_ctxs[int(i)], k, rho, T))
                it += 1
            if (ep + 1) % eval_every == 0 or ep == epochs - 1:
                v = self.validate(val_ctxs, ks) if val_ctxs else float("nan")
                if snap is None or math.isnan(v) or math.isnan(best) or v > best:
                    best, snap = v, self._snapshot()
                rec = {"epoch": ep + 1, "val": v, "val_fallback": self.val_fallback if val_ctxs else float("nan"),
                       "T": T, **{key: _nanmean([s[key] for s in stats]) for key in stats[0]}}
                self.log.append(rec)
                if verbose:
                    print(f"  ep {ep + 1:4d}  val={v:.4f} (fallback {rec['val_fallback']:.2f})"
                          f"  L={rec['loss']:.4f}  disp={rec['disp']:.4f}"
                          f"  comm={rec['comm_disp']:.4f}  spread={rec['ratio']:.3f}  mu={rec['mu']:.3f}"
                          f"  comms={rec['n_comm']:.1f} eff={rec['eff_comm']:.1f}/{self.adversary.K}"
                          f"  T={T:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if snap is not None:
            self._restore(snap)
        self.train_time = time.time() - t0
        return best

    # ---------------------------------------------------------- checkpoints

    def save(self, path, **meta):
        torch.save({"policy": self.policy.state_dict(), "adversary": self.adversary.state_dict(),
                    "rounds": self.rounds, "hparams": _plain(self.hparams), "mu_ref": self.mu_table(),
                    "log": _plain(self.log), "train_time": float(self.train_time), "meta": _plain(meta)}, path)

    def load(self, path):
        """Load a checkpoint written by save() into this trainer (same architecture); returns its meta."""
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ck["policy"])
        if "adversary" in ck:
            self.adversary.load_state_dict(ck["adversary"])
        self.rounds = int(ck.get("rounds", self.rounds))
        self.mu, self.mu_ref = {}, {int(k): float(v) for k, v in ck.get("mu_ref", {}).items()}
        return ck.get("meta", {})


def load_policy(path, device="cpu"):
    """(PolicyV2 in eval mode, checkpoint dict) from a V2Trainer.save() file, without a trainer.
    The checkpoint's "rounds" and "mu_ref" (k -> spread price) are what select_seeds_v2 expects."""
    ck = torch.load(path, map_location=device, weights_only=True)
    hp = ck.get("hparams", {})
    pol = PolicyV2(hp.get("in_dim", IN_DIM), hp.get("hidden", 64), hp.get("layers", 3), hp.get("heads", 4))
    pol.load_state_dict(ck["policy"])
    return pol.to(device).eval(), ck
