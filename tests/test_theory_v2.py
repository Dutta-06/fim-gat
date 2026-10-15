"""Checks of the theory in docs/theory_v2.md (R1 noise, R2 termination, R3 certified deployment).

Each test is the executable form of a stated result:
  Lemma 5 (certified gain)      reduces to the plain gain; is a valid lower bound; rejects tie noise;
                                tie rules (ii) and (iii)
  Lemma 3 (change noise)        with sampled roots, per-root second moments of the change
  Lemma 2 (KL bounds)           one-sided coverage against exact binomial tails
  Theorem 2 (certificate)       coverage against exact Independent Cascade on a tiny graph
  Proposition 3 / Theorem 1     improve() can revisit a seed set when the family grows; tabu prevents it;
                                the number of accepted moves obeys the bound
"""
import math

import numpy as np
import pytest
from scipy.stats import binom

from conftest import exact_ic, group_rates
from fimgat.baselines import RRGreedy
from fimgat.diffusion import ExposureEstimator
from fimgat.graph import from_edges

pytestmark = pytest.mark.spec

V = pytest.importorskip("fimgat.v2.verify")
C = pytest.importorskip("fimgat.v2.certify")
IMP = pytest.importorskip("fimgat.v2.improve")
CFG = pytest.importorskip("fimgat.v2.config")


def plain_gain(x, x_new):
    return (x.max() - x.min()) - (x_new.max() - x_new.min())


# ------------------------------------------------------------------ Lemma 5: certified gain

def test_certified_gain_reduces_to_plain_gain():
    rng = np.random.default_rng(0)
    for _ in range(500):
        G = int(rng.integers(2, 9))
        x, xn = rng.random(G), rng.random(G)
        assert V.certified_gain(x, xn, 0.0, 0.0) == pytest.approx(plain_gain(x, xn), abs=1e-12)


def test_certified_gain_is_a_valid_lower_bound():
    """True rates r, r' anywhere consistent with the half-widths: the true gain is never below the bound."""
    rng = np.random.default_rng(1)
    for _ in range(4000):
        G = int(rng.integers(2, 8))
        r = rng.random(G) * 0.3
        rn = np.clip(r + rng.normal(0, 0.03, G), 0, 1)
        lam, eta = rng.random(G) * 0.02, rng.random(G) * 0.01
        x = r + rng.uniform(-1, 1, G) * lam                      # |x - r| <= lam
        d_true = rn - r
        d_est = d_true + rng.uniform(-1, 1, G) * eta             # |d_est - d_true| <= eta
        bound = V.certified_gain(x, x + d_est, lam, eta)
        assert plain_gain(r, rn) >= bound - 1e-12


def test_certified_gain_rejects_tie_noise():
    """Two groups truly tied at the minimum: lifting the one that looks lower gives a plain gain equal to
    the noisy gap, but no true gain, because the other group stays at the minimum."""
    r = np.array([0.10, 0.10, 0.30])
    x = np.array([0.098, 0.102, 0.30])                         # estimation noise 0.002
    xn = x + np.array([0.02, 0.0, 0.0])                       # the move lifts group 0 only
    assert plain_gain(x, xn) == pytest.approx(0.004, abs=1e-12)
    lam = np.full(3, 0.003)
    assert V.certified_gain(x, xn, lam, 0.0) <= 0.0
    assert plain_gain(r, r + np.array([0.02, 0, 0])) == pytest.approx(0.0, abs=1e-12)


def test_certified_gain_tie_rules():
    """Lemma 5 (ii): a tied minimum caps only the minimum-side term, so a move that lowers the untied
    maximum can still be certified. (iii): if one group could be both the minimum and the maximum, no
    move is certified: the bound is at most -2 eta_g."""
    x, lam = np.array([0.100, 0.101, 0.300]), np.full(3, 0.003)
    assert V.certified_gain(x, x + np.array([0.02, 0.0, -0.05]), lam, 0.0) == pytest.approx(0.05, abs=1e-12)
    rng = np.random.default_rng(5)
    for x in (np.array([0.300, 0.305, 0.310]), np.array([0.280, 0.300, 0.320])):   # all tied; middle tied to both
        lt, eta = np.full(3, 0.011), rng.random(3) * 0.002
        both = ((x - lt) <= np.min(x + lt)) & ((x + lt) >= np.max(x - lt))
        assert both.any()
        for _ in range(2000):
            d = rng.normal(0, 0.02, 3)
            assert V.certified_gain(x, x + d, lt, eta) <= -2 * eta[both].max() + 1e-12


def test_accept_thresholds_and_noise_default():
    b = V.Evaluation(100.0, 0.06, 0.12)
    assert V.accept(b, V.Evaluation(100.0, 0.055, 0.12), 99.0)                      # plain rule
    assert not V.accept(b, V.Evaluation(100.0, 0.055, 0.12), 99.0, min_gain=0.01)   # below min_gain
    assert V.accept(b, V.Evaluation(100.0, 0.06, 0.10), 99.0, min_gain=0.01, min_gain_comm=0.01)
    assert CFG.Z_ACCEPT >= 0 and CFG.TABU in (True, False)


# ------------------------------------------------------------------ Lemma 2: KL bounds

@pytest.mark.parametrize("N,p", [(20, 0.05), (50, 0.3), (200, 0.01), (200, 0.5), (1000, 0.07)])
def test_kl_bounds_coverage_is_exact_binomial_safe(N, p):
    level = 0.05
    k = np.arange(N + 1)
    lo, hi = C.kl_bounds(k / N, np.full(N + 1, N), level)
    pmf = binom.pmf(k, N, p)
    assert pmf[hi < p].sum() <= level + 1e-12                  # P(upper bound misses p)
    assert pmf[lo > p].sum() <= level + 1e-12                  # P(lower bound misses p)


def test_kl_bounds_handle_extremes():
    lo, hi = C.kl_bounds(np.array([0.0, 1.0, 0.5]), np.array([10, 10, 0]), 0.05)
    assert lo[0] == 0.0 and hi[1] == 1.0 and 0.0 < hi[0] < 1.0 and 0.0 < lo[1] < 1.0
    assert lo[2] == 0.0 and hi[2] == 1.0                      # no units: vacuous bounds


# ------------------------------------------------------------------ Theorem 2: certificate coverage

def test_certificate_coverage_against_exact_ic(tiny_graph):
    """Fixed S, S_ref and rho with a small true slack T: over many independent RR samples B, the floor
    lower bound exceeds T, or the disparity upper bound falls below D, at most alpha of the time."""
    g = tiny_graph
    S, ref = [0, 4], [2, 6]
    a_s, a_r = exact_ic(g, S), exact_ic(g, ref)
    sig_s, sig_r = a_s.sum(), a_r.sum()
    rho = min(1.0, 0.97 * sig_s / sig_r)                      # true T = 3% of sigma(S) > 0
    T = sig_s - rho * sig_r
    rates = group_rates(g, a_s)
    present = g.group_sizes > 0
    D = rates[present].max() - rates[present].min()
    alpha, reps = 0.05, 300
    fail_floor = fail_disp = certified = 0
    for rep in range(reps):
        est = ExposureEstimator(g, R=16, seed=9000 + rep, device="cpu")
        fl = C.floor_lcb(est, S, ref, rho, alpha)
        ucb, _ = C.disparity_ucb(est, S, alpha)
        fail_floor += fl["lcb"] > T + 1e-12
        fail_disp += ucb < D - 1e-12
        certified += fl["lcb"] >= 0
    # each statement holds with prob >= 1 - alpha; allow binomial slack (3 sd) on the observed rate
    slack = alpha * reps + 3 * math.sqrt(alpha * (1 - alpha) * reps)
    assert fail_floor <= slack and fail_disp <= slack
    assert 0 <= certified <= reps


def test_floor_lcb_identical_set_and_reference_eligibility(sbm_graph):
    est = ExposureEstimator(sbm_graph, R=8, seed=5, device="cpu")
    ref = RRGreedy(est).greedy(25)
    same = C.floor_lcb(est, ref, ref, 0.99, 0.05)
    assert same["identical"] and same["lcb"] == 0.0
    weak = np.argsort(sbm_graph.out_degree)[:25].tolist()
    out = C.certified_choice(est, [weak], ref, 0.99, alpha=0.05)
    weak_cert = out["certificates"][0]
    assert not weak_cert["eligible"] and weak_cert["floor_lcb"] < 0
    assert out["is_reference"] and out["index"] == -1          # only the reference qualifies
    assert all(c["eligible"] for c in out["certificates"] if c["is_reference"])


def test_rate_bounds_match_exact_rates(sbm_graph):
    from fimgat.v2.train import exact_rates
    est = ExposureEstimator(sbm_graph, R=8, seed=6, device="cpu")
    S = RRGreedy(est).greedy(25)
    rates, lo, hi = C.rate_bounds(est, S, 0.01)
    ex, _ = exact_rates(est, S)
    assert np.allclose(rates, ex, rtol=0, atol=1e-12)
    assert np.all(lo <= rates + 1e-15) and np.all(rates <= hi + 1e-15)


# ------------------------------------------------------------------ R1: detail and noise model

def test_detail_evaluation_matches_plain_evaluation(sbm_graph):
    from fimgat.v2.cover import CoverState
    est = ExposureEstimator(sbm_graph, R=8, seed=7, device="cpu")
    S = RRGreedy(est).greedy(25)
    state = CoverState(est, S)
    fam = V.CommunityFamily(est, [np.arange(0, 300), np.arange(300, 700), np.arange(900, 1400)])
    rng = np.random.default_rng(3)
    non = np.flatnonzero(~state.is_seed)
    for _ in range(30):
        u, v = int(rng.choice(S)), int(rng.choice(non))
        a = V.evaluate_swap(state, u, v, fam)
        b = V.evaluate_swap(state, u, v, fam, detail=True)
        assert (a.total, a.disp, a.comm_disp) == (b.total, b.disp, b.comm_disp)    # bit for bit
        loss, gain = state.swap_detail(u, v)
        assert b.detail.touched.sum() == len(loss) + len(gain)
        m = V.evaluate_move(state, [u], [v], fam, detail=True)
        assert m.total == pytest.approx(b.total, rel=1e-12) and m.disp == pytest.approx(b.disp, abs=1e-12)
        assert np.allclose(m.detail.touched, b.detail.touched)
        assert np.allclose(m.detail.comm_touched, b.detail.comm_touched)


def test_sampled_roots_change_noise_is_per_root(sbm_graph, monkeypatch):
    """Lemma 3 with sampled roots: the roots are the units, so a move's change noise sums the squared
    number of touched RR sets per root (between-root variance included), at least the RR-set count."""
    import functools

    from fimgat import diffusion
    from fimgat.v2.cover import CoverState
    monkeypatch.setattr(diffusion, "stratified_roots",          # keep about 30% of each group as roots
                        functools.partial(diffusion.stratified_roots, min_per_group=50))
    est = ExposureEstimator(sbm_graph, R=8, seed=9, device="cpu", max_roots=600)
    assert not est.all_roots
    S = RRGreedy(est).greedy(25)
    state = CoverState(est, S)
    fam = V.CommunityFamily(est, [np.arange(0, 600), np.arange(600, 1400)])
    rng = np.random.default_rng(4)
    non = np.flatnonzero(~state.is_seed)
    concentrated = False
    for _ in range(30):
        u, v = int(rng.choice(S)), int(rng.choice(non))
        a = V.evaluate_swap(state, u, v, fam)
        b = V.evaluate_swap(state, u, v, fam, detail=True)
        assert (a.total, a.disp, a.comm_disp) == (b.total, b.disp, b.comm_disp)    # bit for bit
        loss, gain = state.swap_detail(u, v)
        roots, cnt = state.root_touch_from(np.concatenate([loss, gain]))
        want = np.bincount(state.index.root_groups[roots], weights=cnt.astype(float) ** 2,
                           minlength=len(b.detail.touched))
        assert np.allclose(b.detail.touched, want)
        assert b.detail.touched.sum() >= len(loss) + len(gain)
        concentrated |= bool((cnt > 1).any())
        m = V.evaluate_move(state, [u], [v], fam, detail=True)
        assert np.allclose(m.detail.touched, b.detail.touched)
        assert np.allclose(m.detail.comm_touched, b.detail.comm_touched)
    assert concentrated                          # some root had several touched sets: the squares matter


def test_noise_aware_improve_keeps_the_invariants(sbm_graph):
    est = ExposureEstimator(sbm_graph, R=8, seed=8, device="cpu")
    gr = RRGreedy(est)
    floor = CFG.rr_floor(0.99, est.evaluate_set(gr.greedy(25))[1])
    start = gr.parity(25)
    if est.evaluate_set(start)[1] < floor:
        start = gr.welfare(25)
    for z in (0.0, 2.0):
        seeds, hist = IMP.improve(est, start, floor, proposer="exhaustive", audit=False, max_rounds=15, z=z)
        ds = [h["disp"] for h in hist]
        assert all(b <= a + 1e-12 for a, b in zip(ds, ds[1:]))
        assert all(h["total"] >= floor for h in hist[1:] if h["accepted"])
        assert {"move", "n_tabu"} <= set(hist[-1])
        assert len(set(seeds)) == 25


# ------------------------------------------------------------------ R2: cycling and termination

def _two_component_graph():
    """Nodes 0-4: a clique P; nodes 5-24: a path Q. One labelled group, so D is identically 0."""
    clique = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    path = [(i, i + 1) for i in range(5, 24)]
    return from_edges("cycle_demo", 25, np.array(clique + path), np.zeros(25, int), ["all"],
                      undirected=True, weighted_cascade=True)


def _run_cycle_demo(monkeypatch, tabu):
    import fimgat.v2.auditor as aud
    g = _two_component_graph()
    est = ExposureEstimator(g, R=2000, seed=0, device="cpu")
    P, Q = np.arange(0, 5), np.arange(5, 25)
    fam = V.CommunityFamily(est, [P, Q])
    from fimgat.v2.cover import CoverState
    e_P = fam.state_rates(CoverState(est, [0]))[0]
    e_Q = fam.state_rates(CoverState(est, [5]))[1]
    assert e_Q < e_P < 1.0                                       # premise of the construction
    calls = {"n": 0}

    def scripted_audit(graph, exposure, **kw):
        calls["n"] += 1
        return (np.array([5]), None) if calls["n"] == 2 else (None, None)   # 2nd audit adds C3 = {5}
    monkeypatch.setattr(aud, "audit", scripted_audit)
    # the multi-scale auditor (the default since 2026-10-10) is scripted the same way
    monkeypatch.setattr(aud, "audit_bands", lambda graph, exposure, bands, **kw: [
        c for c in scripted_audit(graph, exposure) if c is not None])

    def back_and_forth(state, family, rng):
        s = state.seeds[0]
        return [(s, 5 if s == 0 else 0)]
    # the construction scripts the proposer too, so the exhaustive fallback (the default) is off
    return IMP.improve(est, [0], 0.0, communities=[P, Q], proposer=back_and_forth, audit=True, audit_every=1,
                       max_rounds=6, z=0.0, tabu=tabu, fallback=None)


def _visited_sets(start, hist):
    s, path = set(start), [frozenset(start)]
    for h in hist[1:]:
        if h["accepted"]:
            rem, add = h["move"]
            s = (s - set(rem)) | set(add)
            path.append(frozenset(s))
    return path


def test_without_tabu_the_search_revisits_a_set(monkeypatch):
    seeds, hist = _run_cycle_demo(monkeypatch, tabu=False)
    path = _visited_sets([0], hist)
    assert len(path) == 3 and path[0] == path[2] != path[1]       # A -> B -> A
    assert hist[2]["family_size"] > hist[1]["family_size"]         # the revisit followed family growth
    assert seeds == [0]


def test_tabu_prevents_the_revisit(monkeypatch):
    seeds, hist = _run_cycle_demo(monkeypatch, tabu=True)
    path = _visited_sets([0], hist)
    assert len(path) == len(set(path)) == 2                       # A -> B, never back
    assert any(h["n_tabu"] > 0 for h in hist)
    assert seeds == [5]


def test_accepted_moves_obey_the_termination_bound(sbm_graph):
    """Theorem 1 without communities: #accepted <= D_0 / min_gain."""
    est = ExposureEstimator(sbm_graph, R=8, seed=10, device="cpu")
    gr = RRGreedy(est)
    floor = CFG.rr_floor(0.98, est.evaluate_set(gr.greedy(25))[1])
    start = gr.parity(25)
    if est.evaluate_set(start)[1] < floor:
        start = gr.welfare(25)
    eps = 0.002
    seeds, hist = IMP.improve(est, start, floor, proposer="exhaustive", audit=False, max_rounds=10**4,
                              min_gain=eps)
    moves = sum(h["accepted"] for h in hist)
    assert moves <= math.floor(hist[0]["disp"] / eps)
    gains = [a["disp"] - b["disp"] for a, b in zip(hist, hist[1:]) if b["accepted"]]
    assert all(gn > eps for gn in gains)
