"""The constructor fixes of 2026-10-17 (docs/design_v2.md section 24), marker spec.

  repair      train.repair_to_floor reaches a reachable floor by swaps that only gain spread, keeps a feasible set as
              it is, and gives up on an unreachable floor
  decoding    select_seeds_v2 repairs its best candidates before falling back to greedy, and n_repair = 0 keeps the
              old behaviour
  selection   a fallback scores as a failure in V2Trainer.validate, and an exact validation tie keeps the later
              epoch, in V2Trainer.fit and in train_constructor's EpochLog alike
"""
import math
import os
import sys

import numpy as np
import pytest

from conftest import ROOT
from fimgat.baselines import RRGreedy
from fimgat.diffusion import ExposureEstimator

pytestmark = pytest.mark.spec

T = pytest.importorskip("fimgat.v2.train")
K = 10


@pytest.fixture(scope="module")
def est(sbm_graph):
    return ExposureEstimator(sbm_graph, R=8, seed=5, device="cpu")


def _total(est, seeds):
    return T.exact_rates(est, seeds)[1]


def test_repair_reaches_a_reachable_floor_and_keeps_feasible_sets(est):
    greedy = [int(v) for v in RRGreedy(est).greedy(K)]
    top = _total(est, greedy)
    rng = np.random.default_rng(0)
    start = [int(v) for v in rng.choice(est.graph.n, K, replace=False)]
    floor = 0.97 * top
    assert _total(est, start) < floor                         # the random set misses it
    seeds, swaps = T.repair_to_floor(est, start, floor)
    assert len(seeds) == K and len(set(seeds)) == K
    assert _total(est, seeds) >= floor
    assert 1 <= swaps <= K and len(set(seeds) - set(start)) <= swaps
    assert T.repair_to_floor(est, greedy, floor) == (greedy, 0)   # already feasible: unchanged
    assert T.repair_to_floor(est, start, 2.0 * top) is None       # unreachable: given up
    assert T.repair_to_floor(est, start, floor, max_swaps=0) is None


def test_select_seeds_repairs_before_falling_back(est, sbm_graph, monkeypatch):
    ctx = T.prepare_v2(sbm_graph, [K], "cpu", R=8, rr_seed=5)
    pol = T.V2Trainer("cpu", seed=0, hidden=16, heads=2).policy
    floor = 0.98 * float(ctx.total_ref[K])
    monkeypatch.setattr(T, "_select", lambda *a, **kw: None)       # no candidate meets the floor
    seeds, info = T.select_seeds_v2(pol, ctx, K, 0.98, floor, 2, n_samples=8, return_info=True)
    assert info["repaired"] and not info["fallback"] and info["repair_swaps"] >= 0
    assert _total(ctx.est, seeds) >= floor and info["total"] >= floor
    seeds0, info0 = T.select_seeds_v2(pol, ctx, K, 0.98, floor, 2, n_samples=8, n_repair=0, return_info=True)
    assert info0["fallback"] and not info0["repaired"]
    assert seeds0 == [int(v) for v in ctx.greedy_seeds[K]]          # the old behaviour


def test_fallback_scores_as_a_failure_and_ties_keep_the_later_epoch(sbm_graph, monkeypatch, tmp_path):
    ctx = T.prepare_v2(sbm_graph, [K], "cpu", R=8, rr_seed=5)
    tr = T.V2Trainer("cpu", seed=0, hidden=16, heads=2)
    assert math.isnan(tr.val_repaired)
    greedy = [int(v) for v in ctx.greedy_seeds[K]]
    monkeypatch.setattr(T, "select_seeds_v2", lambda *a, **kw: (greedy, {"fallback": True, "repaired": False}))
    rates, _ = T.exact_rates(ctx.est, greedy)
    v = tr.validate([ctx], [K])
    assert v == pytest.approx(-T.disparity(rates, T._present(ctx)) - 1.0)
    assert tr.val_fallback == 1.0 and tr.val_repaired == 0.0

    # fit: a constant validation score restores the last validated state, not the first
    calls, restored = [], []
    monkeypatch.setattr(tr, "step", lambda c, k, rho, temp: {"loss": 0.0})
    monkeypatch.setattr(tr, "validate", lambda vc, ks: calls.append(1) or 0.5)
    monkeypatch.setattr(tr, "_snapshot", lambda: len(calls))
    monkeypatch.setattr(tr, "_restore", restored.append)
    tr.fit([object()], [object()], [K], 3, steps_per_epoch=1, eval_every=1, verbose=False)
    assert restored == [3]

    # train_constructor's EpochLog names the same epoch
    exp = os.path.join(ROOT, "experiments", "v2")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    import train_constructor as TC
    elog = TC.EpochLog(str(tmp_path / "log.csv"), 1, {"model": "m.pt"}, {})
    for _ in range(2):
        elog.on_step(None, K, 0.99, 0.5, {"loss": 0.0})
        elog.on_validate(0.5, 1.0, 0.0)
    assert elog.best_epoch == 2
