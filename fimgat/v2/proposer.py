"""Move proposers for the improvement phase (docs/v2_interfaces.md section 11).

simple_moves     the no-learning baseline of test T3. Add scores are the exact per-group rate gains weighted
                 by signed prices, gains() @ w, over non-seeds; drop scores are the exact removal losses,
                 removal_loss() @ w, and the seeds that cost least are dropped first. The top adds x the
                 cheapest drops become swaps, ordered by the estimated price-weighted change.
LearnedProposer  the Stage 2 proposer. A PolicyV2 scores every node from the v2 dynamic features of the
                 current binary set; its top adds x top drops become swaps, plus a few 2-3 node moves that
                 single swaps cannot make.

Proposals only steer search: improve() scores every move exactly and accepts it only through verify.accept.
A move is (u, v) (swap seed u out, v in) or (removes, adds), two equal-length tuples of node ids.
"""
import numpy as np
import torch

from .config import DELTA_MARGIN, rr_floor
from .objective import signed_prices
from .prices import PIController


def _top(scores, pool, m):
    """The (at most) m entries of pool with the largest scores[pool], best first (ties: smaller id).
    NaN scores rank last."""
    pool = np.asarray(pool, dtype=np.int64)
    m = min(int(m), len(pool))
    if m <= 0:
        return pool[:0]
    s = np.asarray(scores, dtype=np.float64)[pool]
    s = np.where(np.isnan(s), -np.inf, s)
    part = np.argpartition(-s, m - 1)[:m] if m < len(pool) else np.arange(len(pool))
    return pool[part[np.lexsort((pool[part], -s[part]))]]


def _pairs(drops, adds, drop_score, add_score):
    """Every swap (u, v) in drops x adds, ordered by drop_score[u] + add_score[v], best first."""
    if len(drops) == 0 or len(adds) == 0:
        return []
    tot = np.asarray(drop_score, dtype=np.float64)[:, None] + np.asarray(add_score, dtype=np.float64)[None, :]
    tot = np.where(np.isnan(tot), -np.inf, tot)
    i, j = np.unravel_index(np.argsort(-tot, axis=None, kind="stable"), tot.shape)
    return [(int(u), int(v)) for u, v in zip(np.asarray(drops)[i], np.asarray(adds)[j])]


def simple_moves(state, w, n_add, n_drop):
    """n_add * n_drop swaps (u, v): the n_add non-seeds with the largest gains() @ w times the n_drop seeds
    with the smallest removal_loss() @ w. Ordered by the price-weighted change add[v] - drop[u], which
    ignores the overlap of u and v, so that a time-limited search scores the most promising first."""
    seeds = np.asarray(state.seeds, dtype=np.int64)
    n, k = int(state.est.graph.n), len(seeds)
    if k == 0 or k >= n:
        return []
    w = np.asarray(w, dtype=np.float64).ravel()
    is_seed = np.zeros(n, dtype=bool)
    is_seed[seeds] = True
    add = np.asarray(state.gains(), dtype=np.float64) @ w                 # (n,)
    drop = np.asarray(state.removal_loss(), dtype=np.float64) @ w         # (k,), row i is seeds[i]
    adds = _top(add, np.flatnonzero(~is_seed), n_add)
    pos = _top(-drop, np.arange(k), n_drop)                              # cheapest removals first
    return _pairs(seeds[pos], adds, -drop[pos], add[adds])


def community_moves(state, family, n_add=16, n_drop=8, tau=None):
    """Swaps steered by damped prices over a community family (docs/theory_v2.md, W1): the epsilon-constraint
    phase that lowers D_C. Community prices alpha / beta (prices.damped_prices on the family's rates) become
    per-root weights (mixture.root_weights); adds rank by the weighted gain, drops by the smallest weighted
    unique loss, and the n_add x n_drop grid is returned best combined score first. [] without 2 communities.
    Use as improve(..., proposer=lambda s, f, rng: community_moves(s, f), disp_cap=...)."""
    if family is None or len(family) < 2:
        return []
    from .mixture import _node_of, root_weights
    from .prices import damped_prices
    r = family.state_rates(state)
    alpha, beta = damped_prices(r, np.ones(len(r), dtype=bool), *(() if tau is None else (tau,)))
    W = root_weights(state, family, beta - alpha, use_groups=False)
    ix = state.index
    om_e = W[ix.set_root][ix.node_sets]
    node_of = _node_of(ix)
    cnt = state.counts[ix.node_sets]
    gain_w = np.bincount(node_of, weights=om_e * (cnt == 0), minlength=ix.n)
    loss_w = np.bincount(node_of, weights=om_e * (cnt == 1), minlength=ix.n)
    seeds = np.asarray(state.seeds, dtype=np.int64)
    adds = _top(gain_w, np.flatnonzero(~state.is_seed), n_add)
    drops = seeds[_top(-loss_w[seeds], np.arange(len(seeds), dtype=np.int64), n_drop)]
    return _pairs(drops, adds, -loss_w[drops], gain_w[adds])


def _dynamic_from_state():
    try:
        from .features import dynamic_from_state
    except ImportError as e:
        # UNVERIFIED: the contract names the module fimgat/v2/features.py (section 12) but section 11 says
        # features_v2; accept either
        try:
            from .features_v2 import dynamic_from_state
        except ImportError:
            raise e
    return dynamic_from_state


def _np(t):
    if torch.is_tensor(t):
        t = t.detach().float().reshape(-1).cpu().numpy()
    return np.asarray(t, dtype=np.float64).ravel()


class LearnedProposer:
    """Callable proposer (state, family, rng) -> list of moves, from a trained PolicyV2.

    Each call builds the 14 v2 dynamic features of the current binary seed set
    (features.dynamic_from_state) and runs policy.score(ctx, [static | dynamic]) for (add, drop) scores.
    Returns the n_drop top drops x n_add top adds as swaps (best combined score first), then up to n_multi
    (removes, adds) moves whose sizes cycle through multi_size: the top-s drops with the top-s adds first,
    then random s-subsets of the two top lists drawn from rng.

    Prices follow training (train.rollout_v2): w = objective.signed_prices(rates, present, sizes, mu), with
    mu from a PI controller on the floor config.rr_floor(rho, total_ref) = min(rho + DELTA_MARGIN, 1) * total_ref,
    the floor improve() applies (docs/design_v2.md section 18; before 2026-10-10 it was uncapped, so at rho = 1
    the controller aimed above improve's floor and its price wound up). total_ref is the RR total of
    the reference set for this query's budget; None reads ctx.total_ref[k]. improve() calls reset() at
    the start of every query. The family argument is accepted for the proposer interface and not used.
    """

    ROUNDS = 3        # constructor rounds in training; an improvement state is fed as its last round

    def __init__(self, policy, ctx, rho, total_ref, n_add=16, n_drop=8, n_multi=4, multi_size=(2, 3),
                 delta=DELTA_MARGIN):
        self.policy, self.ctx, self.rho, self.delta = policy, ctx, float(rho), float(delta)
        self.total_ref = None if total_ref is None else float(total_ref)
        self.n_add, self.n_drop, self.n_multi = int(n_add), int(n_drop), int(n_multi)
        self.multi_size = tuple(int(s) for s in np.atleast_1d(multi_size))
        self.controller = PIController()

    def reset(self):
        """Forget the spread-price state (a new query)."""
        self.controller.reset()

    def _total_ref(self, state):
        return self.total_ref if self.total_ref is not None else float(self.ctx.total_ref[len(state.seeds)])

    def floor(self, state):
        """The controller's floor: config.rr_floor(rho, total_ref, delta), the floor improve() is given."""
        return rr_floor(self.rho, self._total_ref(state), self.delta)

    def prices(self, state):
        """Signed prices w (G,) for the current state; advances the PI controller by one step."""
        floor = self.floor(state)
        _, _, mu = self.controller.update(state.rates, state.present, state.total, floor, sizes=state.sizes)
        w, _, _ = signed_prices(state.rates, state.present, state.sizes, mu)
        return w

    @torch.no_grad()
    def scores(self, state):
        """(add, drop) scores, float64 numpy (n,) each. Higher drop = better to swap out."""
        ctx = self.ctx
        w = self.prices(state)
        # UNVERIFIED: dynamic_from_state takes total_ref as a float and returns only the 14 dynamic columns;
        # t = rounds - 1 presents the refined binary set as the constructor's last round
        dyn = _dynamic_from_state()(ctx, state, w, self.rho, self._total_ref(state), self.ROUNDS - 1, self.ROUNDS)
        feats = torch.cat([ctx.static, dyn.to(device=ctx.static.device, dtype=ctx.static.dtype)], 1)
        was = self.policy.training
        self.policy.eval()
        try:
            add, drop = self.policy.score(ctx, feats)
        finally:
            self.policy.train(was)
        return _np(add), _np(drop)

    def _multi(self, drops, adds, rng):
        sizes = [s for s in self.multi_size if 2 <= s <= min(len(drops), len(adds))]
        out, seen = [], set()
        if not sizes or self.n_multi <= 0:
            return out
        for i in range(8 * self.n_multi):                 # a few draws may repeat; bounded retries
            if len(out) >= self.n_multi:
                break
            s = sizes[i % len(sizes)]
            if i < len(sizes):
                rem, add = drops[:s], adds[:s]
            else:
                rem, add = rng.choice(drops, s, replace=False), rng.choice(adds, s, replace=False)
            move = (tuple(sorted(int(u) for u in rem)), tuple(sorted(int(v) for v in add)))
            if move not in seen:
                seen.add(move)
                out.append(move)
        return out

    def __call__(self, state, family=None, rng=None):
        rng = np.random.default_rng(0 if rng is None else rng)
        add, drop = self.scores(state)
        n = int(state.est.graph.n)
        if len(add) != n or len(drop) != n:
            raise ValueError(f"policy returned {len(add)} add / {len(drop)} drop scores for {n} nodes")
        seeds = np.asarray(state.seeds, dtype=np.int64)
        if len(seeds) == 0 or len(seeds) >= n:
            return []
        is_seed = np.zeros(n, dtype=bool)
        is_seed[seeds] = True
        adds = _top(add, np.flatnonzero(~is_seed), self.n_add)
        drops = _top(drop, seeds, self.n_drop)
        # UNVERIFIED: assumes the drop head ranks "better to remove" higher (model / train_proposer convention)
        return _pairs(drops, adds, drop[drops], add[adds]) + self._multi(drops, adds, rng)
