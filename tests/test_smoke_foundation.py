"""Smoke tests: the v1 components that v2 builds on behave as documented.

Each test runs in seconds. Ground truth comes from exact enumeration of live-edge
worlds on tiny graphs (conftest.exact_ic), or from autograd.
"""
import numpy as np
import pytest
import torch

from conftest import exact_ic, group_rates
from fimgat.baselines import RRGreedy
from fimgat.diffusion import ExposureEstimator, monte_carlo, sample_rr_sets
from fimgat.graph import from_edges
from fimgat.metrics import evaluate
from fimgat.model import SeedPolicy, soft_topk
from fimgat.objective import fairness_objective, objective_group_weights

pytestmark = pytest.mark.smoke


# ------------------------------------------------------------------ diffusion model

def test_weighted_cascade_probabilities_sum_to_one(tiny_wc_graph):
    g = tiny_wc_graph
    s = np.bincount(g.dst, weights=g.prob, minlength=g.n)
    has_in = g.in_degree > 0
    assert np.allclose(s[has_in], 1.0)
    assert np.all(s[~has_in] == 0)


def test_from_edges_drops_self_loops_and_duplicates():
    g = from_edges("dup", 4, [[0, 1], [0, 1], [1, 1], [2, 3]], [0, 0, 1, 1], ["a", "b"],
                   undirected=True, weighted_cascade=True)
    pairs = set(zip(g.src.tolist(), g.dst.tolist()))
    assert len(pairs) == g.m == 4                                  # (0,1),(1,0),(2,3),(3,2)
    assert all(u != v for u, v in pairs)


# ------------------------------------------------------------------ RR estimator vs exact IC

@pytest.mark.parametrize("seeds", [[0], [3], [0, 6], [2, 4, 7]])
def test_rr_estimator_matches_exact_ic(tiny_graph, seeds):
    g = tiny_graph
    exact = exact_ic(g, seeds)
    est = ExposureEstimator(g, R=20000, seed=1, device="cpu")
    rates, total = est.evaluate_set(seeds)
    assert np.allclose(rates[g.present_groups], group_rates(g, exact)[g.present_groups], atol=0.02)
    assert abs(total - exact.sum()) < 0.1


def test_rr_estimator_weighted_cascade_matches_exact(tiny_wc_graph):
    g = tiny_wc_graph
    exact = exact_ic(g, [0, 5])
    est = ExposureEstimator(g, R=20000, seed=2, device="cpu")
    rates, total = est.evaluate_set([0, 5])
    assert np.allclose(rates, group_rates(g, exact), atol=0.02)


@pytest.mark.parametrize("seeds", [[0], [2, 4, 7]])
def test_monte_carlo_matches_exact_ic(tiny_graph, seeds):
    exact = exact_ic(tiny_graph, seeds)
    node_prob, grp = monte_carlo(tiny_graph, seeds, n_sims=20000, seed=5)
    assert np.allclose(node_prob, exact, atol=0.02)
    assert grp.shape == (20000, tiny_graph.n_groups)


def test_rr_sampling_is_reproducible_per_seed(sbm_graph):
    roots = np.arange(sbm_graph.n)
    p1, n1 = sample_rr_sets(sbm_graph, roots, 4, seed=7)
    p2, n2 = sample_rr_sets(sbm_graph, roots, 4, seed=7)
    p3, n3 = sample_rr_sets(sbm_graph, roots, 4, seed=8)
    assert np.array_equal(p1, p2) and np.array_equal(n1, n2)
    assert not (len(n1) == len(n3) and np.array_equal(n1, n3))


def test_every_rr_set_contains_its_root(sbm_graph):
    ptr, nodes = sample_rr_sets(sbm_graph, np.arange(sbm_graph.n), 2, seed=3)
    roots = np.repeat(np.arange(sbm_graph.n), 2)
    assert np.all(nodes[ptr[:-1]] == roots)


def test_batch_rates_equal_single_evaluations(sbm_graph):
    est = ExposureEstimator(sbm_graph, R=8, seed=4, device="cpu")
    rng = np.random.default_rng(0)
    cands = [rng.choice(sbm_graph.n, 25, replace=False) for _ in range(6)]
    X = torch.stack([est.set_to_x(c) for c in cands])
    rates, totals = est.batch_exposure_rates(X)
    for c, r, t in zip(cands, rates, totals):
        r1, t1 = est.evaluate_set(c)
        assert np.allclose(r.numpy(), r1, atol=1e-5) and abs(float(t) - t1) < 1e-3 * max(t1, 1)


# ------------------------------------------------------------------ closed-form gradients

def test_group_gains_match_autograd(sbm_graph):
    est = ExposureEstimator(sbm_graph, R=4, seed=5, device="cpu")
    torch.manual_seed(0)
    x = (torch.rand(sbm_graph.n) * 0.05).requires_grad_(True)
    a, rates, total, miss = est(x)
    gains = est.group_gains(x.detach(), miss.detach())
    for g in np.flatnonzero(sbm_graph.present_groups):
        grad, = torch.autograd.grad(rates[g] * est.sizes[g], x, retain_graph=True)
        assert torch.allclose(gains[:, g], grad, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_objective_gradient_matches_autograd(sbm_graph, beta):
    est = ExposureEstimator(sbm_graph, R=4, seed=6, device="cpu")
    x = torch.full((sbm_graph.n,), 25 / sbm_graph.n, requires_grad=True)
    a, rates, total, miss = est(x)
    total_ref = float(total.detach()) * 1.5
    J, _, _ = fairness_objective(rates, total, sbm_graph.n, total_ref, beta, est.present_idx)
    grad, = torch.autograd.grad(J, x)
    w = objective_group_weights(rates.detach(), sbm_graph.n, total_ref, beta, est.present_idx, est.sizes)
    closed = est.group_gains(x.detach(), miss.detach()) @ w
    assert torch.allclose(closed, grad, rtol=1e-3, atol=1e-7)


# ------------------------------------------------------------------ exact-budget soft top-k

@pytest.mark.parametrize("k,T", [(1, 0.1), (20, 1.0), (20, 0.1), (199, 0.1)])
def test_soft_topk_meets_budget_and_is_budget_neutral(k, T):
    torch.manual_seed(1)
    s = torch.randn(200, requires_grad=True)
    x = soft_topk(s, k, T)
    assert abs(float(x.sum()) - k) < 1e-2 * max(k, 1)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    g, = torch.autograd.grad(x.sum(), s)
    assert float(g.abs().max()) < 1e-3                          # gradients never change the budget


def test_soft_topk_with_tied_scores():
    x = soft_topk(torch.zeros(100), 10, 0.1)
    assert torch.isfinite(x).all() and abs(float(x.sum()) - 10) < 0.1


# ------------------------------------------------------------------ features: group relabelling

def test_features_invariant_to_group_relabelling(sbm_graph):
    """Relative group features are why one model serves any group definition (v2 claim)."""
    from fimgat.train import prepare
    perm = np.array([2, 0, 1])
    g2 = sbm_graph.with_groups(perm[sbm_graph.groups], [sbm_graph.group_names[i] for i in np.argsort(perm)])
    c1 = prepare(sbm_graph, 25, "cpu", R=4, rr_seed=9)
    c2 = prepare(g2, 25, "cpu", R=4, rr_seed=9)
    assert torch.allclose(c1.static, c2.static, atol=1e-5)
    x = torch.full((sbm_graph.n,), 25 / sbm_graph.n)
    dyn = []
    for c in (c1, c2):
        a, rates, total, miss = c.est(x)
        gains = c.est.group_gains(x, miss)
        w = objective_group_weights(rates, c.n, c.total_ref[25], 1.0, c.present_idx, c.est.sizes)
        dyn.append(c.dynamic(x, a, rates, total, gains, gains @ w, 1.0, 0, 3, 25))
    assert torch.allclose(dyn[0], dyn[1], atol=1e-4)


# ------------------------------------------------------------------ policy network

def test_policy_is_node_permutation_equivariant():
    torch.manual_seed(2)
    from fimgat.features import EDGE_DIM, N_DYNAMIC, N_STATIC
    n, m = 50, 200
    feats = torch.randn(n, N_STATIC + N_DYNAMIC)
    ei = torch.randint(0, n, (2, m))
    ei = torch.cat([ei, torch.arange(n).repeat(2, 1)], 1)
    ea = torch.rand(ei.shape[1], EDGE_DIM)
    pol = SeedPolicy().eval()
    out = pol(feats, ei, ea)
    perm = torch.randperm(n)
    inv = torch.argsort(perm)
    out_p = pol(feats[perm], inv[ei], ea)
    assert torch.allclose(out_p, out[perm], atol=1e-4)


def test_chunked_inference_matches_full_graph(sbm_graph):
    from fimgat.train import prepare
    ctx = prepare(sbm_graph, 25, "cpu", R=2, rr_seed=1)
    feats = torch.cat([ctx.static, torch.randn(ctx.n, 12)], 1)
    pol = SeedPolicy().eval()
    with torch.no_grad():
        full = pol(feats, ctx.edge_index, ctx.edge_attr)
        chunked = pol.forward_chunked(feats, ctx.edge_index, ctx.edge_attr, ctx.dst_ptr, max_edges=3000)
    assert torch.allclose(full, chunked, atol=1e-5)


# ------------------------------------------------------------------ baselines and metrics

@pytest.mark.parametrize("method", ["greedy", "parity", "welfare", "maximin"])
def test_greedy_family_returns_k_unique_seeds(sbm_graph, method):
    gr = RRGreedy(ExposureEstimator(sbm_graph, R=4, seed=2, device="cpu"))
    seeds = getattr(gr, method)(25)
    assert len(seeds) == 25 == len(set(seeds))


def test_greedy_beats_random_on_rr_sample(sbm_graph):
    est = ExposureEstimator(sbm_graph, R=4, seed=2, device="cpu")
    g_total = est.evaluate_set(RRGreedy(est).greedy(25))[1]
    rng = np.random.default_rng(1)
    assert all(g_total > est.evaluate_set(rng.choice(sbm_graph.n, 25, replace=False))[1] for _ in range(5))


def test_parity_respects_group_quotas(sbm_graph):
    gr = RRGreedy(ExposureEstimator(sbm_graph, R=4, seed=2, device="cpu"))
    seeds = gr.parity(30)
    counts = np.bincount(sbm_graph.groups[seeds], minlength=sbm_graph.n_groups)
    share = sbm_graph.group_sizes / sbm_graph.group_sizes.sum() * 30
    assert np.all(np.abs(counts - share) <= 1.0)


def test_evaluate_metrics_are_consistent_and_reproducible(sbm_graph):
    seeds = RRGreedy(ExposureEstimator(sbm_graph, R=4, seed=2, device="cpu")).greedy(25)
    a = evaluate(sbm_graph, seeds, n_sims=500, seed=3)
    b = evaluate(sbm_graph, seeds, n_sims=500, seed=3)
    assert a["influence"] == b["influence"]
    assert a["influence"] >= 25
    assert abs(a["disparity"] - (max(a["rates"]) - min(a["rates"]))) < 1e-12
    assert all(0.0 <= r <= 1.0 for r in a["rates"])
