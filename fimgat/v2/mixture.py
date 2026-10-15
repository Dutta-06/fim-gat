"""Mixture of campaigns (docs/theory_v2.md, Proposition W1.3 and Theorem W1.4).

A single seed set has no game-theoretic fairness guarantee: in the learner-versus-auditor game the balanced
solution is in general a *mixture* of seed sets (Proposition W1.3 gives an instance where every single set
has disparity 1 and a 50/50 mixture has 0). This module computes such a mixture over a finite audited
family A (the present labelled groups and/or a CommunityFamily):

* the auditor runs two Hedge instances over A, one looking for the highest rate, one for the lowest;
* the learner answers each auditor mixture (q_max, q_min) with an approximate best response: swap local
  search that raises sum_a (q_min - q_max)_a r_a(S) on the RR sample while the RR total stays at or above
  the floor, warm-started from its previous answer;
* the deployed lottery is the mixture of the distinct answers that minimises the disparity of the
  expected rates (a small linear programme); every campaign in it meets the floor.

Theorem W1.4: if every answer is an eps-best response, the uniform mixture of T answers has
D_A(expected rates) <= V_A + eps + sqrt(2 ln |A| / T), where V_A is the best value any mixture of
floor-feasible sets can reach; the LP mixture is at least as good. The guarantee is ex ante (on expected
rates over the lottery). Individual campaigns can be unfair, and are reported.
"""
import math

import numpy as np
from scipy.optimize import linprog

from .cover import CoverState
from .proposer import _top
from .verify import CommunityFamily

TOL = 1e-12


def action_rates(state, family=None, use_groups=True):
    """Rates of the audited actions on the RR sample: present labelled groups, then family communities."""
    parts = []
    if use_groups:
        parts.append(np.asarray(state.rates, dtype=np.float64)[np.asarray(state.present, dtype=bool)])
    if family is not None and len(family):
        parts.append(family.state_rates(state))
    return np.concatenate(parts) if parts else np.zeros(0)


def root_weights(state, family, w, use_groups=True):
    """Per-root weight W (n_roots,) with sum_a w_a r_a(S) = sum_j W[root(j)] X_j(S) exactly on the sample:
    a covered RR set rooted in group g adds gw[g] to r_g, and one rooted in community c adds 1 / (R |c|)."""
    ix = state.index
    w = np.asarray(w, dtype=np.float64)
    W = np.zeros(ix.n_roots)
    pos = 0
    if use_groups:
        P = np.flatnonzero(state.present)
        wg = np.zeros(ix.G)
        wg[P] = w[:len(P)]
        W += (wg * ix.gw)[ix.root_groups]
        pos = len(P)
    if family is not None and len(family):
        family._build()
        coef = w[pos:pos + len(family)] / (ix.R * family._size)
        W += np.bincount(family._idx, weights=coef[family._comm_of], minlength=ix.n_roots)
    return W


def _node_of(ix):
    cached = getattr(ix, "_w1_node_of", None)
    if cached is None:
        cached = np.repeat(np.arange(ix.n, dtype=np.int64), np.diff(ix.node_ptr))
        ix._w1_node_of = cached
    return cached


def best_response(state, root_w, floor, n_add=16, n_drop=8, max_moves=20):
    """Approximate best response (mutates state): swaps that raise sum_j root_w[root(j)] X_j while the RR
    total stays >= floor. Candidates are the n_add non-seeds with the largest weighted gain times the
    n_drop seeds with the smallest weighted unique loss; each swap is scored exactly. Returns #moves."""
    ix = state.index
    omega = np.asarray(root_w, dtype=np.float64)[ix.set_root]           # weight of each RR set
    node_of, om_e = _node_of(ix), omega[ix.node_sets]
    moves = 0
    for _ in range(int(max_moves)):
        cnt = state.counts[ix.node_sets]
        gain_w = np.bincount(node_of, weights=om_e * (cnt == 0), minlength=ix.n)
        loss_w = np.bincount(node_of, weights=om_e * (cnt == 1), minlength=ix.n)
        seeds = np.asarray(state.seeds, dtype=np.int64)
        adds = _top(gain_w, np.flatnonzero(~state.is_seed), n_add)
        drops = seeds[_top(-loss_w[seeds], np.arange(len(seeds), dtype=np.int64), n_drop)]
        best, best_val = None, TOL
        for u in drops:
            for v in adds:
                _, d_total, loss, gain = state.swap_delta_detail(int(u), int(v))
                val = float(omega[gain].sum() - omega[loss].sum())
                if val > best_val and state.total + d_total >= floor:
                    best, best_val = (int(u), int(v)), val
        if best is None:
            break
        state.apply_swap(*best)
        moves += 1
    return moves


def lp_mixture(rates):
    """Weights (m,) on the simplex minimising max_a - min_a of the mixed rates (rates (m, N)): an LP."""
    rates = np.asarray(rates, dtype=np.float64)
    m, N = rates.shape
    if m == 1 or N < 2:
        return np.full(m, 1.0 / m)
    c = np.zeros(m + 2)
    c[m], c[m + 1] = 1.0, -1.0                                           # minimise U - L
    upper = np.hstack([rates.T, -np.ones((N, 1)), np.zeros((N, 1))])     # sum_t lam_t r_a(S_t) <= U
    lower = np.hstack([-rates.T, np.zeros((N, 1)), np.ones((N, 1))])     # L <= sum_t lam_t r_a(S_t)
    res = linprog(c, A_ub=np.vstack([upper, lower]), b_ub=np.zeros(2 * N),
                  A_eq=np.hstack([np.ones((1, m)), np.zeros((1, 2))]), b_eq=[1.0],
                  bounds=[(0, None)] * m + [(None, None), (None, None)], method="highs")
    if not res.success:
        return np.full(m, 1.0 / m)
    lam = np.clip(res.x[:m], 0.0, None)
    return lam / lam.sum()


def _disp(r):
    return float(r.max() - r.min()) if len(r) >= 2 else 0.0


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def mixture_campaigns(est, start, floor, communities=(), T=30, use_groups=True, n_add=16, n_drop=8,
                      max_moves=20, eta=None):
    """Mixture of floor-feasible campaigns against a Hedge auditor over the labelled groups (use_groups)
    and the given communities (node-id arrays or a CommunityFamily). start must meet floor.

    Returns {"sets", "weights", "rates", "mix_rates", "mix_disp", "uniform_disp", "campaign_disp",
    "best_single_disp", "regret_bound", "n_actions", "T", "moves"}: the distinct campaigns, their LP
    weights (never worse than the uniform lottery over the T answers), their action rates (m, N) on the sample, the mixed rates and their disparity, the disparity of
    the uniform mixture of all T answers (the quantity Theorem W1.4 bounds), each campaign's own
    disparity, the best single campaign's, and sqrt(2 ln N / T)."""
    family = communities if isinstance(communities, CommunityFamily) else CommunityFamily(est, communities)
    state = CoverState(est, [int(s) for s in start])
    if state.total < floor:
        raise ValueError("start must meet the floor")
    N = len(action_rates(state, family, use_groups))
    T = max(int(T), 1)
    if N < 2:
        r = action_rates(state, family, use_groups)[None, :]
        return {"sets": [list(state.seeds)], "weights": np.ones(1), "rates": r, "mix_rates": r[0],
                "mix_disp": 0.0, "uniform_disp": 0.0, "campaign_disp": np.zeros(1), "best_single_disp": 0.0,
                "regret_bound": 0.0, "n_actions": N, "T": 0, "moves": []}
    eta = math.sqrt(8.0 * math.log(N) / T) if eta is None else float(eta)
    lw_max, lw_min = np.zeros(N), np.zeros(N)
    keys, rates, moves = [], [], []
    for _ in range(T):
        w = _softmax(lw_min) - _softmax(lw_max)                          # learner raises q_min, lowers q_max
        moves.append(best_response(state, root_weights(state, family, w, use_groups), floor,
                                   n_add, n_drop, max_moves))
        r = action_rates(state, family, use_groups)
        keys.append(tuple(sorted(int(s) for s in state.seeds)))
        rates.append(r)
        lw_max += eta * r                                                # Hedge, payoffs in [0, 1]
        lw_min += eta * (1.0 - r)
    rates = np.asarray(rates)
    first = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    idx = list(first.values())
    Rm = rates[idx]
    pos = {key: j for j, key in enumerate(first)}
    emp = np.zeros(len(idx))
    for key in keys:
        emp[pos[key]] += 1.0 / T                                         # the uniform lottery over the T answers
    lam = lp_mixture(Rm)
    if _disp(emp @ Rm) < _disp(lam @ Rm):                                # LP failed: keep Theorem W1.4's lottery
        lam = emp
    mix = lam @ Rm
    camp = np.array([_disp(row) for row in Rm])
    return {"sets": [list(keys[i]) for i in idx], "weights": lam, "rates": Rm, "mix_rates": mix,
            "mix_disp": _disp(mix), "uniform_disp": _disp(rates.mean(axis=0)), "campaign_disp": camp,
            "best_single_disp": float(camp.min()), "regret_bound": math.sqrt(2.0 * math.log(N) / T),
            "n_actions": N, "T": T, "moves": moves}
