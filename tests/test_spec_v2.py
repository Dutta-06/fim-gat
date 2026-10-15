"""v2 contract tests (docs/design_v2.md). Each test skips until its fimgat.v2 module exists.

Interfaces under test (the implementation must provide exactly these):

fimgat/v2/config.py      BUDGETS, DIFFUSION, SNAP_GT, RHO_DEFAULT
fimgat/v2/objective.py   disparity(rates, present) -> float
                         smooth_disparity(rates_t, present, tau) -> torch scalar (differentiable)
                         signed_prices(rates, present, sizes, mu, tau) -> (w, alpha, beta)
fimgat/v2/prices.py      PIController(kp, ki).update(rates, present, total, floor) -> (alpha, beta, mu)
fimgat/v2/cover.py       CoverState(est, seeds): .seeds .rates .total
                         .swap_delta(u_out, v_in) -> (d_rates, d_total)   exact
                         .apply_swap(u_out, v_in)   .removal_loss() -> (k, G)
fimgat/v2/verify.py      Evaluation(total, disp, comm_disp); accept(before, after, floor) -> bool
fimgat/v2/reference.py   reference_spread(graph, k, runs, R, seed, n_sims) -> {"seeds","spread","run_spreads"}
                         equal_spread_test(graph, seeds, ref_seeds, rho, n_sims, seed, alpha)
                             -> {"passed","ratio","lower"}
fimgat/v2/auditor.py     conductance(graph, nodes) -> float
                         find_community(graph, exposure, side, min_size, max_conductance, seed)
                             -> np.ndarray of node ids, or None
fimgat/v2/improve.py     improve(est, start, floor, communities=None, proposer="simple",
                                 max_rounds=..., max_seconds=..., reference=None)
                             -> (seeds, history: list of dicts with total, disp, comm_disp, fallback)
fimgat/v2/local_search.py local_search(est, start, floor) -> seeds
fimgat/v2/datasets.py    NAMES, SOURCES, HELD_OUT, available(name), load(name, attribute=None),
                         degree_groups(graph, n_bins=2), sample(name, size, seed),
                         splits() -> {"train","val","test"}, ground_truth_communities(name)
"""
import builtins
import importlib
import time

import numpy as np
import pytest
import torch

from conftest import BUDGETS
from fimgat.baselines import RRGreedy
from fimgat.diffusion import ExposureEstimator
from fimgat.graph import from_edges

pytestmark = pytest.mark.spec


def need(module):
    try:
        return importlib.import_module(f"fimgat.v2.{module}")
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("fimgat.v2"):
            pytest.skip(f"fimgat.v2.{module} not implemented yet")
        raise


@pytest.fixture(scope="module")
def est_sbm(sbm_graph):
    return ExposureEstimator(sbm_graph, R=8, seed=21, device="cpu")


def _disp(est, seeds):
    r, t = est.evaluate_set(seeds)
    p = (est.graph.group_sizes > 0) & (est.root_counts > 0)
    return float(r[p].max() - r[p].min()), t


# ------------------------------------------------------------------ config (decisions of 2026-09-27)

def test_config_matches_agreed_decisions():
    cfg = need("config")
    assert tuple(cfg.BUDGETS) == BUDGETS == (25, 50, 100, 150, 200)
    assert cfg.DIFFUSION == "weighted_cascade"
    assert cfg.SNAP_GT == "com-youtube"
    assert 0.9 <= cfg.RHO_DEFAULT <= 1.0


# ------------------------------------------------------------------ objective and prices

def test_disparity_ignores_absent_groups():
    obj = need("objective")
    rates = np.array([0.1, 0.5, 0.0, 0.3])
    present = np.array([True, True, False, True])
    assert obj.disparity(rates, present) == pytest.approx(0.4)


@pytest.mark.parametrize("tau", [0.01, 0.05])
def test_smooth_disparity_bounds_and_gradient(tau):
    obj = need("objective")
    r = torch.tensor([0.10, 0.12, 0.30, 0.31], requires_grad=True)
    present = torch.tensor([True, True, True, True])
    s = obj.smooth_disparity(r, present, tau)
    true = 0.31 - 0.10
    assert true - 1e-6 <= float(s) <= true + 2 * tau * np.log(4) + 1e-6
    g, = torch.autograd.grad(s, r)
    assert torch.isfinite(g).all() and g[3] > 0 and g[0] < 0      # pushes max down, min up


def test_signed_prices_are_two_sided():
    obj = need("objective")
    rates = np.array([0.05, 0.20, 0.40, 0.0])
    present = np.array([True, True, True, False])
    sizes = np.array([100, 300, 50, 0])
    w, alpha, beta = obj.signed_prices(rates, present, sizes, mu=0.0, tau=0.02)
    assert np.isclose(alpha[present].sum(), 1) and np.isclose(beta[present].sum(), 1)
    assert alpha[~present].sum() == 0 and beta[~present].sum() == 0
    assert np.argmax(alpha) == 2 and np.argmax(beta) == 0            # over-served / under-served
    assert w[2] < 0 < w[0]                                           # over-served group is pushed down
    w_mu, _, _ = obj.signed_prices(rates, present, sizes, mu=5.0, tau=0.02)
    assert np.all(w_mu[present] > w[present])                        # spread price raises every weight


def test_pi_controller_spread_price_responds_to_floor():
    pr = need("prices")
    c = pr.PIController(kp=1.0, ki=0.5)
    rates, present = np.array([0.1, 0.2]), np.array([True, True])
    mus = [c.update(rates, present, total=90.0, floor=100.0)[2] for _ in range(5)]
    assert all(b >= a for a, b in zip(mus, mus[1:])) and mus[-1] > 0   # below floor: price rises
    for _ in range(50):
        _, _, mu = c.update(rates, present, total=120.0, floor=100.0)
    assert mu == pytest.approx(0.0, abs=1e-6)                           # comfortably above: back to 0


# ------------------------------------------------------------------ exact incremental cover state

def test_cover_state_matches_full_evaluation(est_sbm):
    cov = need("cover")
    rng = np.random.default_rng(0)
    S = rng.choice(est_sbm.graph.n, 25, replace=False).tolist()
    cs = cov.CoverState(est_sbm, S)
    r, t = est_sbm.evaluate_set(S)
    # evaluate_set clamps x to float32(1 - 1e-6), so its totals sit up to ~1e-6 (relative) below the exact
    # float64 RR value that CoverState computes; compare at 1e-5
    assert np.allclose(cs.rates, r, atol=1e-6) and cs.total == pytest.approx(t, rel=1e-5)


def test_swap_delta_is_exact(est_sbm):
    cov = need("cover")
    rng = np.random.default_rng(1)
    n = est_sbm.graph.n
    S = rng.choice(n, 25, replace=False).tolist()
    cs = cov.CoverState(est_sbm, S)
    outside = np.setdiff1d(np.arange(n), S)
    for _ in range(200):
        u, v = int(rng.choice(S)), int(rng.choice(outside))
        d_rates, d_total = cs.swap_delta(u, v)
        r_new, t_new = est_sbm.evaluate_set([v if s == u else s for s in S])
        assert np.allclose(cs.rates + d_rates, r_new, atol=1e-6)
        assert cs.total + d_total == pytest.approx(t_new, rel=1e-5, abs=1e-6)


def test_apply_swap_stays_consistent_over_many_swaps(est_sbm):
    cov = need("cover")
    rng = np.random.default_rng(2)
    n = est_sbm.graph.n
    cs = cov.CoverState(est_sbm, rng.choice(n, 50, replace=False).tolist())
    for _ in range(300):
        u = int(rng.choice(cs.seeds))
        v = int(rng.choice(np.setdiff1d(np.arange(n), cs.seeds)))
        cs.apply_swap(u, v)
    assert len(set(cs.seeds)) == 50
    r, t = est_sbm.evaluate_set(cs.seeds)
    assert np.allclose(cs.rates, r, atol=1e-5) and cs.total == pytest.approx(t, rel=1e-5)


def test_removal_loss_is_exact(est_sbm):
    cov = need("cover")
    S = RRGreedy(est_sbm).greedy(20)
    cs = cov.CoverState(est_sbm, S)
    loss = cs.removal_loss()
    assert loss.shape == (20, est_sbm.graph.n_groups)
    for i, s in enumerate(S):
        r_wo, _ = est_sbm.evaluate_set([x for x in S if x != s])
        assert np.allclose(loss[i], cs.rates - r_wo, atol=1e-6)


def test_invalid_swaps_are_rejected(est_sbm):
    cov = need("cover")
    cs = cov.CoverState(est_sbm, [0, 1, 2])
    with pytest.raises(ValueError):
        cs.swap_delta(5, 6)          # 5 is not a seed
    with pytest.raises(ValueError):
        cs.swap_delta(0, 1)          # 1 is already a seed


# ------------------------------------------------------------------ acceptance rule (verifier option (b))

@pytest.mark.parametrize("after,floor,ok", [
    ((100.0, 0.05, 0.10), 99.0, True),     # both better
    ((100.0, 0.05, 0.12), 99.0, True),     # disparity better, community disparity equal
    ((100.0, 0.06, 0.11), 99.0, True),     # community disparity better, disparity equal
    ((98.0, 0.01, 0.01), 99.0, False),     # floor broken: never accepted
    ((100.0, 0.04, 0.13), 99.0, False),    # community disparity worse
    ((100.0, 0.07, 0.05), 99.0, False),    # labelled disparity worse
    ((100.0, 0.06, 0.12), 99.0, False),    # no strict improvement
])
def test_acceptance_rule(after, floor, ok):
    v = need("verify")
    before = v.Evaluation(total=100.0, disp=0.06, comm_disp=0.12)
    assert v.accept(before, v.Evaluation(*after), floor) is ok


# ------------------------------------------------------------------ reference spread and equal-spread test

def test_reference_is_best_of_runs(sbm_graph):
    ref = need("reference")
    out = ref.reference_spread(sbm_graph, 25, runs=3, R=8, seed=0, n_sims=500)
    assert len(out["seeds"]) == 25 and len(out["run_spreads"]) == 3
    assert out["spread"] == pytest.approx(max(out["run_spreads"]))


def test_equal_spread_test_accepts_reference_and_rejects_weak_sets(sbm_graph):
    ref = need("reference")
    out = ref.reference_spread(sbm_graph, 25, runs=2, R=8, seed=0, n_sims=500)
    same = ref.equal_spread_test(sbm_graph, out["seeds"], out["seeds"], rho=0.99, n_sims=500, seed=1, alpha=0.05)
    assert same["passed"] and same["ratio"] == pytest.approx(1.0)
    weak = np.argsort(sbm_graph.out_degree)[:25].tolist()              # 25 least-connected nodes
    bad = ref.equal_spread_test(sbm_graph, weak, out["seeds"], rho=0.99, n_sims=500, seed=1, alpha=0.05)
    assert not bad["passed"] and bad["lower"] <= bad["ratio"]


# ------------------------------------------------------------------ hive-mind auditor

@pytest.fixture(scope="module")
def planted():
    """Two 40-node dense blocks joined by 3 edges; exposure high in A, low in B."""
    rng = np.random.default_rng(4)
    A, B = np.arange(40), np.arange(40, 80)
    e = [(int(u), int(v)) for blk in (A, B) for u in blk for v in blk if u < v and rng.random() < 0.3]
    e += [(0, 40), (5, 45), (9, 50)]
    g = from_edges("planted", 80, np.array(e), np.zeros(80, int), ["all"], undirected=True, weighted_cascade=True)
    expo = np.r_[np.full(40, 0.9), np.full(40, 0.05)] + rng.normal(0, 0.01, 80)
    return g, A, B, np.clip(expo, 0, 1)


def _jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b)


def test_auditor_finds_planted_under_and_over_exposed_blocks(planted):
    aud = need("auditor")
    g, A, B, expo = planted
    under = aud.find_community(g, expo, side="under", min_size=10, max_conductance=0.3, seed=0)
    over = aud.find_community(g, expo, side="over", min_size=10, max_conductance=0.3, seed=0)
    assert _jaccard(under, B) >= 0.8 and _jaccard(over, A) >= 0.8


def test_auditor_respects_size_and_cohesion(planted):
    aud = need("auditor")
    g, _, _, expo = planted
    c = aud.find_community(g, expo, side="under", min_size=10, max_conductance=0.3, seed=0)
    assert len(c) >= 10 and aud.conductance(g, c) <= 0.3
    assert aud.find_community(g, expo, side="under", min_size=81, max_conductance=0.3, seed=0) is None


def test_auditor_never_uses_the_evaluation_partition(planted, monkeypatch):
    """Leakage rule: the auditor must not touch metrics.communities (the evaluation partition)."""
    aud = need("auditor")
    import fimgat.metrics as M

    def forbidden(*a, **k):
        raise AssertionError("auditor used the evaluation Louvain partition")
    monkeypatch.setattr(M, "communities", forbidden)
    g, _, B, expo = planted
    assert aud.find_community(g, expo, side="under", min_size=10, max_conductance=0.3, seed=0) is not None


def test_auditor_is_deterministic_given_seed(planted):
    aud = need("auditor")
    g, _, _, expo = planted
    a = aud.find_community(g, expo, side="under", min_size=10, max_conductance=0.3, seed=3)
    b = aud.find_community(g, expo, side="under", min_size=10, max_conductance=0.3, seed=3)
    assert np.array_equal(np.sort(a), np.sort(b))


# ------------------------------------------------------------------ improvement phase and local search

def test_improve_never_worse_than_start_and_keeps_floor(est_sbm):
    imp = need("improve")
    gr = RRGreedy(est_sbm)
    ref_total = est_sbm.evaluate_set(gr.greedy(25))[1]
    floor = 0.99 * ref_total
    start = gr.parity(25)
    if est_sbm.evaluate_set(start)[1] < floor:
        start = gr.welfare(25)
    d0, t0 = _disp(est_sbm, start)
    seeds, hist = imp.improve(est_sbm, start, floor, proposer="simple", max_rounds=20)
    d1, t1 = _disp(est_sbm, seeds)
    assert len(set(seeds)) == 25
    assert t1 >= floor and d1 <= d0 + 1e-9
    ds = [h["disp"] for h in hist]
    assert all(b <= a + 1e-9 for a, b in zip(ds, ds[1:]))            # monotone: never gets worse


def test_improve_falls_back_when_start_breaks_floor(est_sbm):
    imp = need("improve")
    gr = RRGreedy(est_sbm)
    ref = gr.greedy(25)
    floor = 0.99 * est_sbm.evaluate_set(ref)[1]
    weak = np.argsort(est_sbm.graph.out_degree)[:25].tolist()
    seeds, hist = imp.improve(est_sbm, weak, floor, proposer="simple", max_rounds=5, reference=ref)
    assert est_sbm.evaluate_set(seeds)[1] >= floor and hist[0]["fallback"] is True


def test_improve_respects_time_budget(est_sbm):
    imp = need("improve")
    gr = RRGreedy(est_sbm)
    floor = 0.95 * est_sbm.evaluate_set(gr.greedy(25))[1]
    t = time.time()
    imp.improve(est_sbm, gr.parity(25), floor, proposer="simple", max_rounds=10**6, max_seconds=2.0)
    assert time.time() - t < 2.0 + 1.5


def test_local_search_returns_single_swap_local_optimum(tiny_wc_graph):
    ls = need("local_search")
    est = ExposureEstimator(tiny_wc_graph, R=2000, seed=0, device="cpu")
    gr = RRGreedy(est)
    floor = 0.9 * est.evaluate_set(gr.greedy(2))[1]
    S = ls.local_search(est, gr.greedy(2), floor)
    d, t = _disp(est, S)
    assert t >= floor
    for u in S:                                                      # brute force: no improving swap left
        for v in set(range(tiny_wc_graph.n)) - set(S):
            d2, t2 = _disp(est, [v if s == u else s for s in S])
            # tolerances above float32 rounding of evaluate_set (the verifier itself works in float64)
            assert not (t2 >= floor + 1e-6 and d2 < d - 1e-6)


# ------------------------------------------------------------------ datasets

def test_dataset_registry_and_sources():
    ds = need("datasets")
    assert set(ds.NAMES) == {"rice_facebook", "twitter_political", "pokec_z", "pokec_n", "pokec_full", "epinions",
                             "twitter_ego", "instagram", "snap_youtube", "linkedin"}
    assert not any(str(p).endswith(".nodes") for p in ds.SOURCES["linkedin"])   # real profile handles: never read
    assert {"pokec_n", "pokec_full"} <= set(ds.HELD_OUT)


def test_linkedin_loader_never_opens_the_nodes_file(monkeypatch):
    ds = need("datasets")
    if not ds.available("linkedin"):
        pytest.skip("linkedin not built yet")
    real_open = builtins.open

    def guarded(path, *a, **k):
        assert not str(path).endswith(".nodes"), "soc-linkedin.nodes must never be read"
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded)
    ds.load("linkedin")


def test_splits_are_disjoint_and_keep_held_out_graphs_out_of_training():
    ds = need("datasets")
    sp = ds.splits()
    tr, va, te = map(set, (sp["train"], sp["val"], sp["test"]))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert not (set(ds.HELD_OUT) & (tr | va))


def test_samples_are_reproducible_and_keep_full_graph_degree_groups():
    ds = need("datasets")
    if not ds.available("epinions"):
        pytest.skip("epinions not built yet")
    a = ds.sample("epinions", 5000, seed=1)
    b = ds.sample("epinions", 5000, seed=1)
    c = ds.sample("epinions", 5000, seed=2)
    assert np.array_equal(a.meta["full_ids"], b.meta["full_ids"])
    assert not np.array_equal(a.meta["full_ids"], c.meta["full_ids"])
    full = ds.load("epinions")
    groups_full = ds.degree_groups(full)
    assert np.array_equal(a.meta["attrs"]["degree"], groups_full[a.meta["full_ids"]])   # degree on FULL graph


def test_ground_truth_communities_allow_overlap():
    ds = need("datasets")
    if not ds.available("snap_youtube"):
        pytest.skip("snap_youtube not built yet")
    comms = ds.ground_truth_communities("snap_youtube")
    assert len(comms) > 0 and all(len(c) >= 3 for c in comms)
    members = np.concatenate(comms)
    assert len(members) >= len(np.unique(members))                   # overlap is allowed, not required
