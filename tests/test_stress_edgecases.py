"""Stress tests on adversarial graphs: every method must return a valid seed set and
finite metrics on topologies and group structures that break naive code.

Opt-in (`pytest -m stress`); runs on the CPU in about a minute. A failure here marks a
v1 edge-case bug to fix (or to guard) before v2 reuses that component.
"""
import numpy as np
import pytest
import torch

from conftest import degenerate_graphs
from fimgat import baselines as B
from fimgat.metrics import evaluate
from fimgat.model import SeedPolicy
from fimgat.train import prepare, select_seeds

pytestmark = pytest.mark.stress

CASES = degenerate_graphs()
KS = (1, 5, 30, 59)                      # up to n - 1 on 60-node graphs


@pytest.mark.parametrize("name,g", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("k", KS)
def test_every_method_survives_degenerate_graphs(name, g, k):
    ctx = prepare(g, k, "cpu", R=16, rr_seed=1)
    rg = B.RRGreedy(ctx.est)
    torch.manual_seed(0)
    pol = SeedPolicy().eval()
    suite = {"degree": lambda: B.degree(g, k), "greedy": lambda: rg.greedy(k), "parity": lambda: rg.parity(k),
             "welfare": lambda: rg.welfare(k), "maximin": lambda: rg.maximin(k),
             "fw_fair": lambda: B.fw_fair(ctx.est, k, 1.0, ctx.total_ref[k]),
             "fimgat_v1": lambda: select_seeds(pol, ctx, k, 1.0, 3)}
    for method, fn in suite.items():
        seeds = [int(s) for s in fn()]
        assert len(seeds) == k == len(set(seeds)), f"{method}: invalid seed set"
        assert 0 <= min(seeds) and max(seeds) < g.n
        ev = evaluate(g, seeds, n_sims=200, seed=2)
        assert np.isfinite(ev["influence"]) and ev["influence"] >= k, method
        assert 0.0 <= ev["disparity"] <= 1.0, method


@pytest.mark.parametrize("name,g", CASES, ids=[c[0] for c in CASES])
def test_features_are_finite_on_degenerate_graphs(name, g):
    ctx = prepare(g, 5, "cpu", R=8, rr_seed=3)
    assert torch.isfinite(ctx.static).all()
    x = torch.full((g.n,), 5 / g.n)
    a, rates, total, miss = ctx.est(x)
    gains = ctx.est.group_gains(x, miss)
    dyn = ctx.dynamic(x, a, rates, total, gains, gains.sum(1), 1.0, 0, 3, 5)
    assert torch.isfinite(dyn).all()


def test_unreachable_group_can_only_be_reached_by_seeding():
    """A group with no in-edges: its rate equals the share of its members that are seeds."""
    name, g = [c for c in CASES if c[0] == "unreachable_group"][0]
    src_nodes = np.flatnonzero(g.groups == 1)
    assert g.in_degree[src_nodes].sum() == 0
    ev = evaluate(g, [int(src_nodes[0])] + list(range(4)), n_sims=500, seed=1)
    rates = dict(zip(np.flatnonzero(g.present_groups), ev["rates"]))
    assert abs(rates[1] - 1 / len(src_nodes)) < 1e-12
