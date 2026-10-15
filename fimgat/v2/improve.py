"""Improvement phase of FIM-GAT v2 (docs/design_v2.md section 7; docs/v2_interfaces.md section 10).

From a seed set meeting the RR floor (otherwise the reference set), every round proposes swaps or
multi-node moves, scores each candidate exactly on the RR sample (verify.evaluate_swap / evaluate_move)
and applies the best one that verify.accept allows: the total stays at or above the floor, neither the
labelled disparity D nor the community disparity D_C gets worse, and at least one of them strictly
improves. D therefore never increases, and the current set is valid whenever the search stops.

D_C is taken over a CommunityFamily that only grows: the given communities, plus the most under- and
over-exposed cohesive communities the hive-mind auditor finds at round 0, every audit_every rounds and
once more before stopping. Prices only steer the proposals; the verifier decides.

Termination and cycling (docs/theory_v2.md section 2). Within one family no seed set can recur, but when
the family grows D_C is re-scored, and a set left earlier can look better again (an explicit example is in
tests/test_theory_v2.py). With tabu=True (default) no seed set is ever visited twice; sets are hashed
incrementally (Zobrist), so the check costs O(1) per candidate. min_gain / min_gain_comm and z make every
accepted move improve by a stated amount (Theorem 1 bounds the number of moves); max_audits caps the
family-growth events.
"""
import math
import time

import numpy as np

from ..baselines import RRGreedy
from .config import (AUDIT_BAND_RATIO, AUDIT_MAX_SIZE, KAPPA_SE, MAX_AUDITS, MAX_CONDUCTANCE, MIN_COMMUNITY,
                     PRICE_KI, PRICE_KP, PRICE_SETPOINT, PROPOSER_FALLBACK, TABU, Z_ACCEPT)
from .cover import CoverState
from .objective import price_weights
from .prices import PIController
from .proposer import simple_moves
from .verify import (CommunityFamily, accept, disparity_se, evaluate_move, evaluate_state, evaluate_swap,
                     noise_model)

_HASH_SEED = 0x5EED_F1A7          # fixed: the Zobrist keys only need to be distinct, not secret


class _SetHash:
    """Zobrist hash of a seed set: XOR of one random 63-bit key per node (positive int64). A swap updates
    it in O(1)."""

    def __init__(self, n):
        self.keys = np.random.default_rng(_HASH_SEED).integers(1, 2**63 - 1, size=int(n), dtype=np.int64)

    def of(self, seeds):
        h = 0
        for s in seeds:
            h ^= int(self.keys[int(s)])
        return h

    def moved(self, h, rem, add):
        for x in tuple(rem) + tuple(add):
            h ^= int(self.keys[int(x)])
        return h

BATCH = 32            # candidates scored between two reads of the clock
# Moves must clear the floor by this relative margin (50x below DELTA_MARGIN). A set exactly at the float64
# floor can read below it in est.evaluate_set, which is float32, counts a covered RR set as 1 - ~1e-6 and
# rounds sums over thousands of roots. UNVERIFIED: 5e-5 is assumed to exceed that gap on test-sized graphs.
FLOOR_RTOL = 5e-5


def _as_seeds(seeds, n, what):
    """k distinct node ids in [0, n) as a list of ints; ValueError otherwise."""
    if isinstance(seeds, dict):
        seeds = seeds["seeds"]
    if hasattr(seeds, "detach"):
        seeds = seeds.detach().cpu().numpy()
    s = np.asarray(seeds).ravel()
    if s.size == 0:
        raise ValueError(f"{what}: empty seed set")
    if s.dtype.kind not in "iu" and (s.dtype.kind != "f" or not np.all(s == np.round(s))):
        raise ValueError(f"{what}: non-integer node ids")
    s = s.astype(np.int64)
    if s.min() < 0 or s.max() >= n:
        raise ValueError(f"{what}: node ids outside [0, {n})")
    if len(np.unique(s)) != len(s):
        raise ValueError(f"{what}: duplicate seeds")
    return s.tolist()


def _parse(move, is_seed):
    """(removes, adds) tuples of a swap (u, v) or a move (removes, adds); None if it is not a valid move
    (sizes differ, repeats, a remove that is not a seed, an add that is a seed or out of range)."""
    try:
        a, b = move
    except (TypeError, ValueError):
        return None
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        rem, add = (int(a),), (int(b),)
    else:
        try:
            rem = tuple(int(u) for u in np.atleast_1d(np.asarray(a, dtype=np.int64)).ravel())
            add = tuple(int(v) for v in np.atleast_1d(np.asarray(b, dtype=np.int64)).ravel())
        except (TypeError, ValueError):
            return None
    n = len(is_seed)
    if not rem or len(rem) != len(add) or len(set(rem)) != len(rem) or len(set(add)) != len(add):
        return None
    if any(x < 0 or x >= n for x in rem + add):
        return None
    if not all(is_seed[u] for u in rem) or any(is_seed[v] for v in add):
        return None
    return rem, add


def _rank(ev):
    return ev.disp, ev.comm_disp, -ev.total


def _rank_comm(ev):
    return ev.comm_disp, ev.disp, -ev.total


def _best_move(state, moves, before, floor, family, deadline, accept_kw=None, blocked=None, comm_first=False):
    """Best accepted candidate (lowest disp, then comm_disp, then higher total), scored exactly.

    accept_kw: extra keyword arguments for verify.accept (with a noise model, candidates are scored with
    MoveDetail). blocked(rem, add) -> bool skips a candidate (tabu) before it is scored.
    Returns (move or None, n_scored, n_blocked, out_of_time). The clock is read after every BATCH
    candidates and at the end; when time is up the best move found so far is still returned (it is
    verified)."""
    accept_kw = accept_kw or {}
    detail = accept_kw.get("noise") is not None and accept_kw["noise"].z > 0
    rank = _rank_comm if comm_first else _rank
    is_seed = np.zeros(int(state.est.graph.n), dtype=bool)
    is_seed[np.asarray(state.seeds, dtype=np.int64)] = True
    best = best_ev = None
    seen, n_scored, n_blocked = set(), 0, 0
    for move in moves:
        mv = _parse(move, is_seed)
        if mv is None:
            continue
        key = (tuple(sorted(mv[0])), tuple(sorted(mv[1])))
        if key in seen:
            continue
        seen.add(key)
        rem, add = mv
        if blocked is not None and blocked(rem, add):
            n_blocked += 1
            continue
        if len(rem) == 1:
            ev = evaluate_swap(state, rem[0], add[0], family, detail=detail)
        else:
            ev = evaluate_move(state, rem, add, family, detail=detail)
        n_scored += 1
        if accept(before, ev, floor, **accept_kw) and (best is None or rank(ev) < rank(best_ev)):
            best, best_ev = mv, ev
        if n_scored % BATCH == 0 and time.perf_counter() >= deadline:
            return best, n_scored, n_blocked, True
    return best, n_scored, n_blocked, time.perf_counter() >= deadline


def _apply(state, rem, add):
    if len(rem) == 1:
        state.apply_swap(rem[0], add[0])
        return
    for u in rem:
        state.remove(u)
    for v in add:
        state.add(v)


def improve(est, start, floor, communities=None, proposer="simple", max_rounds=50, max_seconds=None,
            reference=None, audit=True, audit_every=5, min_size=MIN_COMMUNITY, max_conductance=MAX_CONDUCTANCE,
            n_add=16, n_drop=8, rng_seed=0, *, z=None, min_gain=0.0, min_gain_comm=0.0, tabu=None,
            max_audits=MAX_AUDITS, audit_band_ratio=AUDIT_BAND_RATIO, fallback=PROPOSER_FALLBACK,
            price_setpoint=PRICE_SETPOINT, audit_max_size=AUDIT_MAX_SIZE, return_family=False, price_gains=None,
            disp_cap=None):
    """Verified improvement of a seed set. Returns (seeds, history).

    audit_max_size caps the auditor's community size at min(audit_max_size, n // 2) (None: n // 2).
    return_family=True attaches the family's communities (lists of node ids) to the last history entry
    as "communities" (docs/v2_interfaces.md, improve).

    Theory options (docs/theory_v2.md): z (default config.Z_ACCEPT; 0 = plain acceptance) scores
    improvements with verify.certified_gain; min_gain / min_gain_comm are the smallest accepted
    improvements of D / D_C; tabu (default config.TABU) forbids revisiting a seed set; max_audits caps
    auditor calls. History entries also carry "move" (the applied (removes, adds), else None) and
    "n_tabu" (candidates skipped because they lead to a visited set).

    Weak-spot options (docs/theory_v2.md, W1-W4). Since the decisions of 2026-10-10 (docs/design_v2.md section 18)
    the defaults come from config: multi-scale audits (AUDIT_BAND_RATIO), the exhaustive fallback
    (PROPOSER_FALLBACK) and setpoint-scaled prices (PRICE_SETPOINT, PRICE_KP, PRICE_KI). The published behaviour is
    audit_band_ratio=None, fallback=None, price_setpoint=0.0, price_gains=(1.0, 0.5).
    audit_band_ratio (W1): None audits at one scale; a ratio r > 1 audits every size band of
    auditor.size_bands(min_size, cap, r) on both sides, cap as below. fallback (W3): "exhaustive" makes a round
    whose proposals contain no accepted move fall back to local_search.best_swap (instead of the simple
    proposer's widening), so the search still ends at a single-swap local optimum of the labelled rule; such moves
    have "fallback_move": True in the history. price_setpoint (W4): relative slack above the floor that
    the simple proposer's spread price aims for (0 keeps the price at 0 while the floor holds); price_gains
    (kp, ki) overrides the PI gains (default (config.PRICE_KP, config.PRICE_KI)). disp_cap (W1): an epsilon-constraint phase; D may
    rise up to the cap when D_C strictly improves (verify.accept), and proposals are ranked by D_C first. The
    exhaustive fallback still pre-filters on D and ranks by D, so with a cap it only adds D-lowering moves.
    D then need not be monotone, but (D_C, D) decreases lexicographically, so no set recurs in an epoch.

    start: k distinct node ids. floor: exact RR floor, (rho + DELTA_MARGIN) * total_ref. If start's RR
    total is below it, the search restarts from reference (RR-greedy if None) and every history entry has
    fallback=True. communities: node-id arrays (or a CommunityFamily, whose communities are copied) that
    seed the family. proposer: "simple" (simple_moves with n_add x n_drop swaps and damped PI prices; one
    widening to 2 n_add x 2 n_drop before stopping), "exhaustive" (local_search.best_swap), or a callable
    (state, family, rng) -> list of moves, where a move is (u, v) or (removes, adds); a reset() method,
    if present, is called first. Invalid or repeated moves are skipped.

    Rounds stop after one with no accepted move (after the widening, and after one re-audit that adds a
    community, when audit is on), after max_rounds, or when max_seconds has elapsed, checked after every
    BATCH candidates, before every round and before every audit. Moves must also clear the floor by
    FLOOR_RTOL (relative). history[0] is the start (round 0, after fallback and the first audit); then
    one entry per round with keys round, total, disp, comm_disp, fallback, accepted, n_candidates,
    family_size and secs (seconds since the call). disp never increases along history (without disp_cap).
    """
    t0 = time.perf_counter()
    deadline = math.inf if max_seconds is None else t0 + float(max_seconds)
    simple = isinstance(proposer, str) and proposer == "simple"
    exhaustive = isinstance(proposer, str) and proposer == "exhaustive"
    if not (simple or exhaustive or (callable(proposer) and not isinstance(proposer, str))):
        raise ValueError(f"unknown proposer {proposer!r}: use 'simple', 'exhaustive' or a callable")
    if fallback not in (None, "exhaustive"):
        raise ValueError(f"fallback must be None or 'exhaustive', not {fallback!r}")
    use_fallback = fallback == "exhaustive" and not exhaustive
    n, floor = int(est.graph.n), float(floor)
    if not math.isfinite(floor):
        raise ValueError(f"floor must be finite, got {floor}")
    start = _as_seeds(start, n, "start")
    k = len(start)

    state = CoverState(est, start)
    restarted = bool(state.total < floor)
    if restarted:
        ref = RRGreedy(est).greedy(k) if reference is None else _as_seeds(reference, n, "reference")
        if len(ref) != k:
            raise ValueError(f"reference has {len(ref)} seeds, start has {k}")
        state = CoverState(est, ref)
    floor_acc = floor + FLOOR_RTOL * abs(floor)

    if isinstance(communities, CommunityFamily):
        communities = communities.communities
    family = CommunityFamily(est, () if communities is None else communities)
    rng = np.random.default_rng(rng_seed)
    n_audits = [0]
    z = float(Z_ACCEPT if z is None else z)
    tabu = bool(TABU if tabu is None else tabu)
    hasher = _SetHash(n) if tabu else None
    visited = set()
    cur_hash = [0]

    def accept_kw():
        kw = {"min_gain": float(min_gain), "min_gain_comm": float(min_gain_comm)}
        if disp_cap is not None:
            kw["disp_cap"] = float(disp_cap)
        if z > 0:
            kw["noise"] = noise_model(state, family, z)        # rebuilt whenever the family changes
        return kw

    def blocked(rem, add):
        return hasher.moved(cur_hash[0], rem, add) in visited

    def run_audit():
        """Add the auditor's under- and over-exposed communities; returns how many joined the family."""
        if not audit or time.perf_counter() >= deadline:
            return 0
        if max_audits is not None and n_audits[0] >= int(max_audits):
            return 0
        from .auditor import audit as audit_pair, audit_bands, node_exposure, size_bands
        expo = node_exposure(est, state.exposure())
        cap = max(n // 2, 1) if audit_max_size is None else max(min(int(audit_max_size), n // 2), 1)
        if audit_band_ratio is None:
            found = audit_pair(est.graph, expo, min_size=min_size, max_conductance=max_conductance,
                               seed=int(rng_seed) + n_audits[0], max_size=cap)
        else:
            found = audit_bands(est.graph, expo, size_bands(min_size, cap, audit_band_ratio),
                                max_conductance=max_conductance, seed=int(rng_seed) + n_audits[0])
        n_audits[0] += 1
        size = len(family)
        for c in found:
            if c is not None and len(c):
                family.add(c)
        return len(family) - size

    def propose(na, nd, kw):
        if simple:
            alpha, beta, mu = ctrl.update(state.rates, state.present, state.total, floor)
            return simple_moves(state, price_weights(alpha, beta, mu, state.sizes, state.present), na, nd)
        if exhaustive:
            from .local_search import best_swap
            excl = (lambda u, v: blocked((u,), (v,))) if tabu else None
            out = best_swap(state, floor_acc, family=family, accept_kw=kw, exclude=excl)
            return [] if out is None else [(int(out[0]), int(out[1]))]
        return proposer(state, family, rng) or []

    def entry(r, ev, accepted, n_cand, move=None, n_tabu=0, via_fallback=False):
        return {"round": r, "total": float(ev.total), "disp": float(ev.disp), "comm_disp": float(ev.comm_disp),
                "fallback": restarted, "accepted": bool(accepted), "n_candidates": int(n_cand),
                "family_size": len(family), "secs": time.perf_counter() - t0,
                "move": None if move is None else [list(move[0]), list(move[1])], "n_tabu": int(n_tabu),
                "fallback_move": bool(via_fallback)}

    def exhaustive_move(kw):
        """local_search.best_swap as a move ((u,), (v,)), or None (W3 fallback)."""
        from .local_search import best_swap
        excl = (lambda u, v: blocked((u,), (v,))) if tabu else None
        out = best_swap(state, floor_acc, family=family, accept_kw=kw, exclude=excl,
                        deadline=None if math.isinf(deadline) else deadline)
        return None if out is None else ((int(out[0]),), (int(out[1]),))

    def score_state():
        return evaluate_state(state, family, detail=z > 0)

    run_audit()
    cur = score_state()
    kw = accept_kw()
    if tabu:
        cur_hash[0] = hasher.of(state.seeds)
        visited.add(cur_hash[0])
    history = [entry(0, cur, False, 0)]
    if not (simple or exhaustive) and callable(getattr(proposer, "reset", None)):
        proposer.reset()
    kp, ki = (PRICE_KP, PRICE_KI) if price_gains is None else (float(price_gains[0]), float(price_gains[1]))
    ctrl = PIController(kp=kp, ki=ki, sizes=state.sizes, setpoint=price_setpoint) if simple else None
    na, nd = int(n_add), int(n_drop)
    every = int(audit_every) if audit_every else 0
    widened = reaudited = False

    for r in range(1, int(max_rounds) + 1):
        if time.perf_counter() >= deadline:
            break
        if every > 0 and r > 1 and (r - 1) % every == 0 and run_audit():
            cur, kw = score_state(), accept_kw()          # the family grew: re-score "before" on it
        if time.perf_counter() >= deadline:
            break
        best, n_cand, n_blk, out_of_time = _best_move(state, propose(na, nd, kw), cur, floor_acc, family,
                                                      deadline, kw, blocked if tabu else None,
                                                      comm_first=disp_cap is not None)
        via_fallback = False
        if best is None and use_fallback and not out_of_time:
            best = exhaustive_move(kw)                    # W3: a proposer miss falls back to exhaustive search
            via_fallback = best is not None
            out_of_time = time.perf_counter() >= deadline
        if best is not None:
            _apply(state, *best)
            if tabu:
                cur_hash[0] = hasher.moved(cur_hash[0], *best)
                visited.add(cur_hash[0])
            cur = score_state()
            reaudited = False
        history.append(entry(r, cur, best is not None, n_cand, best, n_blk, via_fallback))
        if out_of_time:
            break
        if best is None:
            if simple and not widened and not use_fallback:
                widened, na, nd = True, 2 * na, 2 * nd
                continue
            if audit and not reaudited:
                reaudited = True
                if run_audit():
                    cur, kw = score_state(), accept_kw()
                    continue
            break
    if return_family:
        history[-1]["communities"] = [[int(v) for v in c] for c in family.communities]
    return [int(s) for s in state.seeds], history


def hive_search(est, start, floor, proposer="exhaustive", communities=None, kappa_se=KAPPA_SE, max_rounds=50,
                max_seconds=None, rng_seed=0, **kw):
    """Option (b) of test T2 (docs/design_v2.md sections 10 and 18): the labelled-only search, then the
    epsilon-constraint phase. Returns (seeds, info).

    Phase 1: improve(..., audit=False) with `proposer` (labelled disparity only; restarts from the reference if
    start misses the floor). Phase 2, from phase 1's result: improve(..., audit=True) with community-price proposals
    (proposer.community_moves), the default multi-scale audits, and disp_cap = kappa = D_a + kappa_se * SE(D_a)
    (verify.disparity_se): labelled disparity may rise by at most kappa_se standard errors while the community
    disparity falls (theory Proposition W1.5). max_seconds covers both phases; communities seed phase 2's family;
    kw goes to both improve() calls (for example reference, audit_max_size, min_size, max_conductance).

    info: {"phase1", "phase2" (the two histories), "start1" (phase 1's actual start: the reference after a
    restart), "kappa", "disp_a", "se_a", "communities" (phase 2's final family)}."""
    for key in ("audit", "disp_cap", "return_family"):
        if key in kw:
            raise ValueError(f"hive_search sets {key} itself")
    from .proposer import community_moves
    t0 = time.perf_counter()
    s_a, h_a = improve(est, start, floor, proposer=proposer, audit=False, max_rounds=max_rounds,
                       max_seconds=max_seconds, rng_seed=rng_seed, **kw)
    start1 = list(start)
    if h_a[0]["fallback"]:
        ref = kw.get("reference")
        start1 = RRGreedy(est).greedy(len(start)) if ref is None else list(ref)
    state = CoverState(est, s_a)
    d_a, se = float(state.disp), disparity_se(state)
    kappa = d_a + float(kappa_se) * se
    left = None if max_seconds is None else max(0.0, float(max_seconds) - (time.perf_counter() - t0))
    s_b, h_b = improve(est, s_a, floor, communities=communities, audit=True, disp_cap=kappa, max_rounds=max_rounds,
                       max_seconds=left, rng_seed=rng_seed, return_family=True,
                       proposer=lambda st, fam, rng: community_moves(st, fam), **kw)
    return s_b, {"phase1": h_a, "phase2": h_b, "start1": [int(v) for v in start1], "kappa": kappa, "disp_a": d_a,
                 "se_a": se, "communities": h_b[-1].get("communities", [])}
