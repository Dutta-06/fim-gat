"""Checks of the weak-spot theory in docs/theory_v2.md (W1 hidden communities, W2 lower bounds,
W3 proposer cost model, W4 price loop).

  Lemma W1.1          the transfer bound and its gap bounds hold on random instances
  Proposition W1.2    within a size band the scan-statistic winner is within 1/sqrt(ratio) of the extreme;
                      multi-scale auditing finds a small extreme community a single scale passes over
  Proposition W1.3    two stars: every floor-feasible single campaign has disparity 1, a mixture 0
  Theorem W1.4        the Hedge-vs-best-response mixture reaches the mixed value on that instance
  Proposition W1.5    the epsilon-constraint phase keeps D under its cap and (D_C, D) decreasing
  Proposition W2.1    the hardness gadget: min disparity = 1 - max spread of the embedded IM instance
  Lemma W2.2 / W2.3   the lower bound never exceeds the brute-force optimum (exact IC); the RR-based
                      group-rate upper bounds cover the exact maxima
  Proposition W3.1    model properties; the hybrid (proposer + exhaustive fallback) ends at a local optimum
  Propositions W4.2-5 the spread price is 0 throughout improve(); Jury's criterion matches the spectral
                      radius; best-response dual ascent on two sets is bang-bang, its average is feasible
"""
import itertools
import math

import numpy as np
import pytest

from conftest import exact_ic, group_rates
from fimgat.baselines import RRGreedy
from fimgat.diffusion import ExposureEstimator
from fimgat.graph import from_edges

pytestmark = pytest.mark.spec

AUD = pytest.importorskip("fimgat.v2.auditor")
HID = pytest.importorskip("fimgat.v2.hidden")
MIX = pytest.importorskip("fimgat.v2.mixture")
BND = pytest.importorskip("fimgat.v2.bounds")
DIA = pytest.importorskip("fimgat.v2.diagnostics")
IMP = pytest.importorskip("fimgat.v2.improve")
PRC = pytest.importorskip("fimgat.v2.prices")
CFG = pytest.importorskip("fimgat.v2.config")


# ------------------------------------------------------------------ W1: transfer bound

def test_pair_gap_bound_is_valid():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(5, 40))
        a = rng.random(n) ** rng.choice([0.5, 1.0, 3.0])
        h = rng.choice(n, int(rng.integers(1, n)), replace=False)
        c = rng.choice(n, int(rng.integers(1, n)), replace=False)
        assert abs(a[h].mean() - a[c].mean()) <= HID.pair_gap_bound(a, h, c) + 1e-12


def test_transfer_bound_holds_on_random_families():
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = int(rng.integers(20, 80))
        a = rng.random(n)
        hidden = [rng.choice(n, int(rng.integers(2, n // 2)), replace=False) for _ in range(int(rng.integers(2, 6)))]
        family = [rng.choice(n, int(rng.integers(2, n // 2)), replace=False) for _ in range(int(rng.integers(1, 5)))]
        out = HID.transfer_bound(a, hidden, family)
        assert out["holds"] and out["D_H"] <= out["bound"] + 1e-12
        r = HID.community_rates(a, hidden)
        assert out["D_H"] == pytest.approx(r.max() - r.min(), abs=1e-12)


# ------------------------------------------------------------------ W1: scale of the auditor

def test_size_bands_cover_the_range_with_bounded_ratio():
    for m, M, ratio in [(10, 1000, 2.0), (1, 37, 1.5), (25, 25, 2.0), (7, 3000, 4.0)]:
        bands = AUD.size_bands(m, M, ratio)
        assert bands[0][0] == m and bands[-1][1] == M
        assert all(b[0] == a[1] + 1 for a, b in zip(bands, bands[1:]))
        assert all(lo <= hi <= max(lo, math.floor(lo * ratio)) for lo, hi in bands)


def test_scan_statistic_winner_within_band_factor():
    """Proposition W1.2: argmax of sqrt(|C|) dev(C) over a band [b, 2b] has dev >= max dev / sqrt(2);
    over [m, M] it can be as low as sqrt(m / M) of the most extreme."""
    rng = np.random.default_rng(2)
    for _ in range(2000):
        b = int(rng.integers(10, 200))
        sizes = rng.integers(b, 2 * b + 1, 30)
        dev = rng.random(30)
        win = np.argmax(np.sqrt(sizes) * dev)
        assert dev[win] >= dev.max() / math.sqrt(2) - 1e-12
    sizes, dev = np.array([10, 1000]), np.array([0.5, 0.06])          # the small one is 8x more extreme
    assert np.argmax(np.sqrt(sizes) * dev) == 1                          # but the large one wins


def _planted_small_and_large(seed=0):
    """Core (300) + large block B (200, deficit 0.3) + small clique-like block s (15, deficit 0.4)."""
    rng = np.random.default_rng(seed)
    core, B, s = np.arange(300), np.arange(300, 500), np.arange(500, 515)
    edges = []
    for block, deg in ((core, 6), (B, 8), (s, 6)):
        for v in block:
            for w in rng.choice(block, deg // 2, replace=False):
                edges.append((v, w))
    for _ in range(10):
        edges.append((rng.choice(B), rng.choice(core)))
    edges += [(s[0], core[0]), (s[1], core[1])]
    g = from_edges("planted_s_B", 515, np.array(edges), np.zeros(515, dtype=np.int64), ["all"],
                   undirected=True, weighted_cascade=True)
    e = np.full(515, 0.5)
    e[B], e[s] = 0.3, 0.1
    return g, e, B, s


def test_multiscale_audit_finds_the_small_extreme_community():
    g, e, B, s = _planted_small_and_large()
    under, _ = AUD.audit(g, e, min_size=10, max_conductance=0.3, seed=0)
    found = AUD.audit_bands(g, e, AUD.size_bands(10, g.n // 2, 2.0), max_conductance=0.3, seed=0)
    jac = [len(np.intersect1d(c, s)) / len(np.union1d(c, s)) for c in found]
    assert max(jac) >= 0.8                                               # the small extreme is found
    rates = [e[c].mean() for c in found]
    assert min(rates) <= e[under].mean() + 1e-12                         # and nothing extreme is lost


# ------------------------------------------------------------------ W1: single campaign vs mixture

def _two_stars(leaves=3):
    edges = [[c0, c0 + j] for c0 in (0, leaves + 1) for j in range(1, leaves + 1)]
    n = 2 * (leaves + 1)
    return from_edges("two_stars", n, np.array(edges), [0] * (leaves + 1) + [1] * (leaves + 1), ["A", "B"],
                      undirected=True, weighted_cascade=True)


def test_two_stars_single_campaigns_are_unfair_mixtures_are_not():
    """Proposition W1.3 with exact IC: at the floor sigma >= sigma(center) every feasible single seed is a
    centre and has disparity 1; the 50/50 mixture of the centres has disparity 0."""
    g = _two_stars()
    prob = {v: exact_ic(g, [v]) for v in range(g.n)}
    sig = {v: p.sum() for v, p in prob.items()}
    floor = max(sig.values())
    feasible = [v for v in range(g.n) if sig[v] >= floor - 1e-12]
    assert feasible == [0, 4]
    for v in feasible:
        r = group_rates(g, prob[v])
        assert r.max() - r.min() == pytest.approx(1.0, abs=1e-12)
    rates = np.array([group_rates(g, prob[v]) for v in feasible])
    lam = MIX.lp_mixture(rates)
    mixed = lam @ rates
    assert mixed.max() - mixed.min() == pytest.approx(0.0, abs=1e-9)
    assert lam == pytest.approx([0.5, 0.5], abs=1e-9)


def test_mixture_campaigns_reaches_the_mixed_value():
    """Theorem W1.4 on two stars: the Hedge auditor drives the learner between the centres; the LP mixture
    has disparity 0 while the best single campaign has 1, and the uniform mixture obeys the regret bound."""
    g = _two_stars()
    est = ExposureEstimator(g, R=64, seed=0, device="cpu")
    out = MIX.mixture_campaigns(est, [0], 4.0 - 1e-9, communities=(), T=40)
    assert sorted(map(tuple, out["sets"])) == [(0,), (4,)]
    assert out["best_single_disp"] == pytest.approx(1.0, abs=1e-12)
    assert out["mix_disp"] == pytest.approx(0.0, abs=1e-9)
    assert out["uniform_disp"] <= 0.0 + 0.0 + out["regret_bound"] + 1e-12   # V = 0, exact best responses


def test_root_weights_reproduce_the_weighted_objective(sbm_graph):
    from fimgat.v2.cover import CoverState
    from fimgat.v2.verify import CommunityFamily
    est = ExposureEstimator(sbm_graph, R=8, seed=3, device="cpu")
    state = CoverState(est, RRGreedy(est).greedy(25))
    fam = CommunityFamily(est, [np.arange(0, 300), np.arange(250, 700)])
    rng = np.random.default_rng(4)
    w = rng.normal(size=3 + len(fam))
    W = MIX.root_weights(state, fam, w)
    lhs = float(w @ MIX.action_rates(state, fam))
    rhs = float(W @ state.root_cov)
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


# ------------------------------------------------------------------ W1: epsilon-constraint phase

def test_accept_disp_cap_trades_labelled_for_community_disparity():
    from fimgat.v2.verify import Evaluation, accept
    before = Evaluation(total=100.0, disp=0.010, comm_disp=0.20)
    trade = Evaluation(total=100.0, disp=0.015, comm_disp=0.10)
    assert not accept(before, trade, 99.0)                              # the published rule: D may not rise
    assert accept(before, trade, 99.0, disp_cap=0.02)
    assert not accept(before, trade, 99.0, disp_cap=0.012)              # above the cap
    assert not accept(before, Evaluation(100.0, 0.015, 0.20), 99.0, disp_cap=0.02)   # no improvement
    assert not accept(before, Evaluation(100.0, 0.005, 0.25), 99.0, disp_cap=0.02)   # D_C may never rise


def _blocks_graph(seed=0, n_blocks=8, size=40):
    rng = np.random.default_rng(seed)
    n = n_blocks * size
    block = np.repeat(np.arange(n_blocks), size)
    edges = [(v, w) for v in range(n) for w in rng.choice(np.flatnonzero(block == block[v]), 4, replace=False)]
    edges += [(v, int(rng.integers(n))) for v in range(n) if rng.random() < 0.5]
    groups = np.where(rng.random(n) < 0.6, block % 3, rng.integers(0, 3, n))
    g = from_edges("blocks", n, np.array(edges), groups, ["a", "b", "c"], undirected=True, weighted_cascade=True)
    return g, [np.flatnonzero(block == b) for b in range(n_blocks)]


def test_epsilon_phase_respects_the_cap_and_the_lexicographic_order():
    from fimgat.v2.cover import CoverState
    from fimgat.v2.proposer import community_moves
    g, _ = _blocks_graph()
    est = ExposureEstimator(g, R=16, seed=0, device="cpu")
    ref = RRGreedy(est).greedy(10)
    floor = CFG.rr_floor(0.99, CoverState(est, ref).total)
    s_a, h_a = IMP.improve(est, ref, floor, proposer="exhaustive", audit=False, max_rounds=60)
    cap = h_a[-1]["disp"] + 0.002
    seeds, hist = IMP.improve(est, s_a, floor, proposer=lambda st, fam, rng: community_moves(st, fam),
                              audit=True, audit_band_ratio=2.0, disp_cap=cap, max_rounds=60, return_family=True)
    assert all(h["disp"] <= max(hist[0]["disp"], cap) + 1e-12 for h in hist)
    assert all(h["total"] >= floor for h in hist)
    for a, b in zip(hist, hist[1:]):
        if b["accepted"] and a["family_size"] == b["family_size"]:       # within one epoch
            assert b["comm_disp"] < a["comm_disp"] - 1e-12 or (
                b["comm_disp"] <= a["comm_disp"] + 1e-12 and b["disp"] < a["disp"] - 1e-12)
    assert hist[-1]["family_size"] >= 2 and len(hist[-1]["communities"]) == hist[-1]["family_size"]


def test_community_moves_are_valid_swaps():
    from fimgat.v2.cover import CoverState
    from fimgat.v2.proposer import community_moves
    from fimgat.v2.verify import CommunityFamily
    g, blocks = _blocks_graph(1)
    est = ExposureEstimator(g, R=8, seed=1, device="cpu")
    state = CoverState(est, RRGreedy(est).greedy(10))
    assert community_moves(state, None) == [] and community_moves(state, CommunityFamily(est, blocks[:1])) == []
    moves = community_moves(state, CommunityFamily(est, blocks), n_add=6, n_drop=3)
    assert len(moves) == 18 and len(set(moves)) == 18
    assert all(state.is_seed[u] and not state.is_seed[v] for u, v in moves)


# ------------------------------------------------------------------ W2: hardness gadget and lower bound

def test_hardness_gadget_reduces_disparity_to_influence_maximisation():
    """Proposition W2.1: H (4 nodes, uncertain edges) plus b0 -> K (|K| = n_H + k, p = 1); groups A = V_H,
    B = {b0} + K; floor |K| + 1. Then min D over feasible sets = 1 - max_v sigma_H({v}) / n_H (k = 2)."""
    nH, k = 4, 2
    L = nH + k
    b0, K = nH, list(range(nH + 1, nH + 1 + L))
    h_edges = {(0, 1): 0.5, (0, 2): 0.5, (3, 2): 0.6, (1, 3): 0.3}
    edges = list(h_edges) + [(b0, x) for x in K]
    n = nH + 1 + L
    g = from_edges("gadget", n, np.array(edges), [0] * nH + [1] * (1 + L), ["A", "B"], undirected=False)
    g.prob[:] = [h_edges.get((int(s), int(d)), 1.0) for s, d in zip(g.src, g.dst)]
    best = math.inf
    for S in itertools.combinations(range(n), k):
        p = exact_ic(g, list(S))
        if p.sum() < L + 1 - 1e-12:
            continue
        r = group_rates(g, p)
        best = min(best, float(r.max() - r.min()))
    H = from_edges("H", nH, np.array(list(h_edges)), [0] * nH, ["A"], undirected=False)
    H.prob[:] = [h_edges[(int(s), int(d))] for s, d in zip(H.src, H.dst)]
    opt = max(exact_ic(H, [v]).sum() for v in range(nH))
    assert best == pytest.approx(1.0 - opt / nH, abs=1e-12)


def _exact_table(g, k):
    """Exact rates and totals of every k-set."""
    sets = list(itertools.combinations(range(g.n), k))
    rates = np.array([group_rates(g, exact_ic(g, list(S))) for S in sets])
    totals = rates @ g.group_sizes
    return sets, rates, totals


def _random_small_groups_graph(seed, n=10, m=10, n_groups=5):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < m:
        u, v = rng.choice(n, 2, replace=False)
        pairs.add((int(u), int(v)))
    g = from_edges(f"small_groups_{seed}", n, np.array(sorted(pairs)), np.arange(n) % n_groups,
                   [str(i) for i in range(n_groups)], undirected=False)
    g.prob[:] = rng.uniform(0.1, 0.9, len(g.prob))
    return g


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lower_bound_never_exceeds_the_exact_optimum(seed):
    """Lemma W2.2 + Proposition W2.3 with exact IC: the bound computed from the exact U, U0 is at most the
    brute-force minimum disparity among sets meeting the floor."""
    k = 2
    g = _random_small_groups_graph(seed)
    sets, rates, totals = _exact_table(g, k)
    F = 0.9 * totals.max()
    feas = totals >= F - 1e-12
    D_star = float((rates.max(1) - rates.min(1))[feas].min())
    U = rates.max(0)
    U0 = np.array([max(rates[i, gg] for i, S in enumerate(sets) if not np.any(g.groups[list(S)] == gg))
                   for gg in range(g.n_groups)])
    lb = BND.lower_bound_from(U, U0, k, F / g.n)
    assert lb["lb"] <= D_star + 1e-12


def test_lower_bound_is_positive_when_groups_outnumber_seeds():
    """Isolated pairs, k = 2 < 5 groups: at least 3 groups stay at 0, the floor keeps the mean up."""
    g = from_edges("pairs", 10, np.zeros((0, 2)), np.arange(10) % 5, list("abcde"), undirected=False)
    k = 2
    sets, rates, totals = _exact_table(g, k)
    F = 0.9 * totals.max()
    D_star = float((rates.max(1) - rates.min(1))[totals >= F - 1e-12].min())
    lb = BND.lower_bound_from(rates.max(0), np.zeros(5), k, F / g.n)
    assert 0 < lb["lb"] <= D_star + 1e-12
    assert lb["lb"] == pytest.approx(0.18, abs=1e-12) and D_star == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("rr_seed", [0, 1, 2])
def test_group_rate_upper_covers_the_exact_maxima(tiny_graph, rr_seed):
    k = 2
    sets, rates, _ = _exact_table(tiny_graph, k)
    est = ExposureEstimator(tiny_graph, R=2000, seed=100 + rr_seed, device="cpu")
    U = BND.group_rate_upper(est, k, exclude_own=False, delta=1e-3)
    U0 = BND.group_rate_upper(est, k, exclude_own=True, delta=1e-3)
    for gg in range(tiny_graph.n_groups):
        assert U[gg] >= rates[:, gg].max() - 1e-12
        own = [i for i, S in enumerate(sets) if not np.any(tiny_graph.groups[list(S)] == gg)]
        if own:
            assert U0[gg] >= rates[own, gg].max() - 1e-12


# ------------------------------------------------------------------ W3: proposer cost model

def test_speedup_model_properties():
    rng = np.random.default_rng(5)
    for _ in range(2000):
        pi, q = rng.random(), rng.random()
        c, T = rng.random() * 10, rng.random() * 100 + 1e-3
        s = DIA.hybrid_speedup(pi, q, c, T)
        assert s <= 1.0 / (1.0 - pi) + 1e-9
        assert DIA.learning_headroom(pi, q, c, T) >= 1.0 - 1e-12
    assert DIA.hybrid_speedup(0.0, 0.0, 0.0, 1.0) == pytest.approx(1.0)


def test_hybrid_ends_at_a_single_swap_local_optimum(sbm_graph):
    from fimgat.v2.cover import CoverState
    from fimgat.v2.local_search import best_swap
    est = ExposureEstimator(sbm_graph, R=8, seed=9, device="cpu")
    ref = RRGreedy(est).greedy(25)
    floor = CFG.rr_floor(0.99, CoverState(est, ref).total)
    seeds, hist = IMP.improve(est, ref, floor, proposer="simple", audit=False, fallback="exhaustive",
                              max_rounds=200)
    assert best_swap(CoverState(est, seeds), floor * (1 + IMP.FLOOR_RTOL)) is None
    ds = [h["disp"] for h in hist]
    assert all(b <= a + 1e-12 for a, b in zip(ds, ds[1:]))
    assert all(h["accepted"] for h in hist[1:] if h["fallback_move"])


def test_measure_proposer_grids_are_nested(sbm_graph):
    from fimgat.v2.cover import CoverState
    est = ExposureEstimator(sbm_graph, R=8, seed=10, device="cpu")
    ref = RRGreedy(est).greedy(25)
    floor = CFG.rr_floor(0.99, CoverState(est, ref).total)
    m = DIA.measure_proposer(est, ref, floor, grids=((2, 1), (4, 2), (8, 4), (16, 8), (32, 16)), n_time=8)
    hits, qs = list(m["pi"].values()), list(m["q"].values())
    assert all(b >= a for a, b in zip(hits, hits[1:]))                  # larger grids contain smaller ones
    assert all(b >= a - 1e-12 for a, b in zip(qs, qs[1:]))
    assert all(0.0 <= x <= 1.0 + 1e-12 for x in qs)


# ------------------------------------------------------------------ W4: the price loop

def test_spread_price_is_inert_in_improve(sbm_graph, monkeypatch):
    """Proposition W4.2: with the published controller (setpoint 0) every state meets the floor, so e <= 0 and
    mu = 0 in every round. The setpoint-scaled controller, the default since 2026-10-10, does act."""
    from fimgat.v2.cover import CoverState
    seen = []
    orig = PRC.PIController.update

    def spy(self, *args, **kw):
        out = orig(self, *args, **kw)
        seen.append(out[2])
        return out

    monkeypatch.setattr(PRC.PIController, "update", spy)
    est = ExposureEstimator(sbm_graph, R=8, seed=11, device="cpu")
    ref = RRGreedy(est).greedy(25)
    floor = CFG.rr_floor(0.99, CoverState(est, ref).total)
    IMP.improve(est, ref, floor, proposer="simple", audit=False, max_rounds=30, price_setpoint=0.0,
                price_gains=(1.0, 0.5))
    assert seen and all(mu == 0.0 for mu in seen)
    seen.clear()
    IMP.improve(est, ref, floor, proposer="simple", audit=False, max_rounds=30)
    assert max(seen) > 0.0


def test_setpoint_makes_the_spread_price_act():
    rates, present = np.array([0.3, 0.4]), np.array([True, True])
    assert PRC.PIController().update(rates, present, total=100.5, floor=100.0)[2] == 0.0
    assert PRC.PIController(setpoint=0.01).update(rates, present, total=100.5, floor=100.0)[2] > 0.0


def test_jury_criterion_matches_the_spectral_radius():
    for g in np.linspace(0.02, 3.0, 60):
        for kp in (0.25, 0.5, 1.0, 1.5):
            for ki in (0.05, 0.25, 0.5, 1.0):
                rad = DIA.pi_loop_radius(g, kp, ki)
                if abs(rad - 1.0) < 1e-6:
                    continue
                assert DIA.pi_loop_stable(g, kp, ki) == (rad < 1.0)
    assert DIA.pi_loop_stable(1.5) and not DIA.pi_loop_stable(1.7)       # default gains: gamma < 1.6


def test_best_response_dual_ascent_is_bang_bang():
    """Proposition W4.5: S1 meets the floor (slack +0.05) with D = 0.10, S2 breaks it (-0.03) with D = 0.02.
    Dual ascent with a best-response player never settles: both sets keep recurring and mu circles mu* = 1,
    while the running average of the slack goes to 0 (the mixture 3/8 S1 + 5/8 S2)."""
    D, slack = np.array([0.10, 0.02]), np.array([0.05, -0.03])
    mu, eta, picks, sl = 0.0, 0.5, [], []
    for _ in range(20000):
        j = int(np.argmin(D - mu * slack))
        picks.append(j)
        sl.append(slack[j])
        mu = max(0.0, mu - eta * slack[j])
    tail = np.array(picks[10000:])
    assert 0 < tail.mean() < 1                                           # both sets recur
    assert abs(np.mean(sl)) < 1e-3
    assert np.mean(tail == 0) == pytest.approx(3 / 8, abs=0.01)
