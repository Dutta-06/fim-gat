"""Checks of the external baselines written from their papers (fimgat/v2/external.py; docs/design_v2.md section 19).

  incremental gains   _Cover's per-group gains and rates equal a fresh recount after every seed; eval_rates equals
                      CoverState's exact rates
  truncated cover     _cover_scores is the marginal gain of sum_i min(c, g_i) on brute force
  Saturate            its set is feasible, so OPT_g never exceeds the brute-force max-min on the sample
  BSM-TSGreedy        meets min_i g_i >= tau OPT_g when stage 1 finishes; the adapter returns the largest tau of the
                      grid whose set meets the floor
  MultiFW             systematic rounding keeps every marginal; the result is k distinct nodes, no less fair (min
                      rate) than the best of many random sets
"""
import itertools

import numpy as np
import pytest

from fimgat.baselines import RRGreedy
from fimgat.diffusion import ExposureEstimator
from fimgat.graph import from_edges

pytestmark = pytest.mark.spec

EXT = pytest.importorskip("fimgat.v2.external")


def _graph(seed=0, n=60, groups=3):
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, n, (4 * n, 2))
    return from_edges("ext", n, edges, rng.integers(0, groups, n), [str(i) for i in range(groups)],
                      undirected=True, weighted_cascade=True)


@pytest.fixture(scope="module")
def small():
    g = _graph()
    est = ExposureEstimator(g, R=32, seed=1, device="cpu")
    return g, est, RRGreedy(est)


def test_incremental_gains_match_a_recount(small):
    g, est, rg = small
    st = EXT._Cover(rg)
    for u in RRGreedy(est).greedy(6):
        st.add(u)
        fresh = EXT._init_gains(rg.ptr, rg.nodes, rg.set_group, rg.set_w * (~st.covered), rg.n, rg.G)
        assert np.allclose(st.gain, fresh, atol=1e-5)
        rates, _ = EXT.eval_rates(rg, st.seeds)
        assert np.allclose(st.rates, rates, atol=1e-12)
    from fimgat.v2.cover import CoverState
    cs = CoverState(est, st.seeds)
    assert np.allclose(EXT.eval_rates(rg, st.seeds)[0], np.asarray(cs.rates), atol=1e-9)


def test_cover_scores_are_the_truncated_marginal_gain(small):
    g, est, rg = small
    st = EXT._Cover(rg)
    for u in (3, 17):
        st.add(u)
    c = 0.5 * float(st.rates[rg.present].max() + 0.2)
    need = np.zeros(rg.G)
    need[rg.present] = np.maximum(c - st.rates[rg.present], 0.0)
    sc = EXT._cover_scores(st.gain, need, st.chosen)
    base = np.minimum(c, st.rates[rg.present]).sum()
    for v in range(g.n):
        if st.chosen[v]:
            assert sc[v] == -1.0
            continue
        r, _ = EXT.eval_rates(rg, st.seeds + [v])
        assert sc[v] == pytest.approx(np.minimum(c, r[rg.present]).sum() - base, abs=1e-5)


def test_saturate_never_beats_the_brute_force_maximin():
    g = _graph(3, n=14)
    est = ExposureEstimator(g, R=64, seed=2, device="cpu")
    rg = RRGreedy(est)
    k = 3
    best = max(EXT.eval_rates(rg, list(s))[0][rg.present].min() for s in itertools.combinations(range(g.n), k))
    seeds, opt = EXT.saturate(rg, k)
    assert len(set(seeds)) == k and opt <= best + 1e-12
    assert opt == pytest.approx(EXT.eval_rates(rg, seeds)[0][rg.present].min())


def test_tsgreedy_meets_its_constraint_and_the_adapter_takes_the_largest_feasible_tau(small):
    g, est, rg = small
    k = 6
    g_seeds = rg.greedy(k)
    s_seeds, opt = EXT.saturate(rg, k)
    for tau in (0.3, 0.6, 0.9):
        seeds = EXT.bsm_tsgreedy(rg, k, tau, g_seeds, s_seeds, opt)
        assert len(set(seeds)) == k
        rates, _ = EXT.eval_rates(rg, seeds)
        assert rates[rg.present].min() >= tau * opt - 1e-9 or seeds == list(s_seeds)
    total_g = EXT.eval_rates(rg, g_seeds)[1]
    floor = 0.97 * total_g
    seeds, info = EXT.bsm(est, k, floor, rg=rg, return_info=True)
    assert EXT.eval_rates(rg, seeds)[1] >= floor
    larger = [t for t in EXT.TAUS if info["tau"] is not None and t > info["tau"]]
    for t in larger:                                    # every larger tau of the grid misses the floor
        assert EXT.eval_rates(rg, EXT.bsm_tsgreedy(rg, k, t, g_seeds, s_seeds, opt))[1] < floor


def test_systematic_rounding_keeps_the_marginals():
    rng = np.random.default_rng(0)
    x = rng.random(30)
    x = np.clip(x * 7 / x.sum(), 0, 1)
    x *= 7 / x.sum()
    x = np.clip(x, 0, 1)
    hits = np.zeros(30)
    for _ in range(4000):
        s = EXT._systematic(x, 7, rng)
        assert len(set(s)) == 7
        hits[s] += 1
    assert np.allclose(hits / 4000, x, atol=0.05)


def test_multifw_returns_a_fair_k_set(small):
    g, est, rg = small
    k = 6
    seeds, info = EXT.multifw_maxmin(est, k, iters=10, mw_steps=50, rounds=10, rg=rg, return_info=True)
    assert len(set(seeds)) == k and all(0 <= v < g.n for v in seeds)
    rng = np.random.default_rng(1)
    rand = max(EXT.eval_rates(rg, list(rng.choice(g.n, k, replace=False)))[0][rg.present].min() for _ in range(20))
    assert info["min_rate"] >= rand - 1e-12


# ------------------------------------------------------------------ RFIM

def test_label_propagation_finds_disconnected_cliques():
    a = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    b = [(i, j) for i in range(6, 12) for j in range(i + 1, 12)]
    g = from_edges("cliques", 12, np.array(a + b), np.zeros(12, int), ["all"], undirected=True, weighted_cascade=True)
    lab = EXT.label_propagation(g, seed=3)
    assert len(set(lab[:6])) == 1 and len(set(lab[6:])) == 1 and lab[0] != lab[6]
    assert set(lab) == {0, 1}


def test_rfim_alpha_follows_the_three_levels():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(60, int), np.ones(25, int), np.full(15, 2)]       # sizes 60, 25, 15 of n = 100
    a = EXT.rfim_alpha(labels, 100, rng)
    assert 0 <= a[0] <= 0.01 and 0 <= a[1] <= 0.1 and 0 <= a[2] <= 1.0
    inv = EXT.rfim_inverse(a, rng)
    assert all(x == y or abs(x + y - 1.0) < 1e-12 for x, y in zip(a, inv))


def test_rfim_weights_give_the_weighted_community_spread(small):
    g, est, rg = small
    parts = EXT.rfim_partitions(g, seed=1)
    W = EXT.rfim_weights(est, parts)
    seeds = rg.greedy(5)
    st = EXT._WCover(rg, W)
    for u in seeds:
        st.add(u)
    from fimgat.v2.certify import set_coverage
    cov = set_coverage(est, seeds).reshape(-1, est.R).mean(1)                 # covered fraction per root
    for q, (labels, alpha) in enumerate(parts):
        c = labels[est.roots]
        want = sum(alpha[j] * (labels == j).sum() * cov[c == j].mean() for j in np.unique(c))
        assert st.F[q] == pytest.approx(want, rel=1e-9)
    fresh = EXT._init_wgains(rg.ptr, rg.nodes, W * (~st.covered)[:, None], rg.n)
    assert np.allclose(st.gain, fresh, atol=1e-9)


def test_rfim_returns_a_robust_k_set(small):
    g, est, rg = small
    k = 6
    seeds, info = EXT.rfim(est, k, rg=rg, return_info=True)
    assert len(set(seeds)) == k and info["Q"] == 4
    W = EXT.rfim_weights(est, EXT.rfim_partitions(g, seed=0))
    inv = 1.0 / np.array(info["opt"])

    def min_ratio(s):
        st = EXT._WCover(rg, W)
        for u in s:
            st.add(u)
        return float((st.F * inv).min())
    rng = np.random.default_rng(2)
    assert info["min_ratio"] == pytest.approx(min_ratio(seeds))
    assert info["min_ratio"] >= max(min_ratio(list(rng.choice(g.n, k, replace=False))) for _ in range(20)) - 1e-12
