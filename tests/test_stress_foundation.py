"""Stress tests at scale: every budget in {25, 50, 100, 150, 200}, Pokec-z/n and full Pokec.

Opt-in (`pytest -m stress`). Correctness is asserted: valid, unique seeds; finite metrics;
peak GPU memory below the card. Timings and memory are recorded (not asserted) in
tests/_reports/stress_results.jsonl, which gives the Stage-1 compute baseline.
Expected wall-clock on the 8 GB laptop: Pokec-z/n about 10 min, full Pokec about 15-20 min.
"""
import os

import numpy as np
import pytest
import torch

from conftest import BUDGETS, ROOT, GpuPeak, data_path, gpu_total_gb, requires_cuda, skip_unless_exists
from fimgat import baselines as B
from fimgat.data import load_pokec
from fimgat.metrics import evaluate
from fimgat.model import SeedPolicy
from fimgat.train import prepare, select_seeds

pytestmark = [pytest.mark.stress, requires_cuda]

MODEL = os.path.join(ROOT, "results", "models", "pokec_z_FIM-GAT_s0.pt")


def _policy(dev):
    skip_unless_exists(MODEL, "trained v1 Pokec model")
    state = torch.load(MODEL, map_location=dev)
    pol = SeedPolicy().to(dev)
    pol.load_state_dict(state["policy"])
    return pol.eval(), state["rounds"]


def _check(g, seeds, k):
    assert len(seeds) == k == len(set(int(s) for s in seeds))
    assert min(seeds) >= 0 and max(seeds) < g.n


@pytest.mark.parametrize("name", ["pokec_z", "pokec_n"])
def test_all_budgets_all_methods_on_pokec(name, stress_report):
    skip_unless_exists(data_path("pokec", f"{name}.npz"), "Pokec build")
    dev = "cuda"
    g = load_pokec(name, "region")
    with GpuPeak() as prep:
        ctx = prepare(g, list(BUDGETS), dev, R=16, rr_seed=77)
    stress_report(test="pokec_prepare", graph=name, secs=prep.secs, peak_alloc_gb=prep.alloc,
                  rr_sets=ctx.est.n_sets, nnz=ctx.est.nnz)
    pol, rounds = _policy(dev)
    rg = B.RRGreedy(ctx.est)
    for k in BUDGETS:
        suite = {"Greedy": lambda: ctx.greedy_seeds[k], "Parity": lambda: rg.parity(k),
                 "Welfare": lambda: rg.welfare(k), "Maximin": lambda: rg.maximin(k),
                 "FW-Fair": lambda: B.fw_fair(ctx.est, k, 1.0, ctx.total_ref[k]),
                 "FIM-GAT v1": lambda: select_seeds(pol, ctx, k, 1.0, rounds)}
        for method, fn in suite.items():
            with GpuPeak() as run:
                seeds = fn()
            _check(g, seeds, k)
            ev = evaluate(g, seeds, n_sims=500, seed=1)
            assert np.isfinite(ev["influence"]) and 0 <= ev["disparity"] <= 1
            assert run.reserved < 0.95 * gpu_total_gb()
            stress_report(test="pokec_budget", graph=name, k=k, method=method, secs=run.secs,
                          peak_alloc_gb=run.alloc, influence=ev["influence"], disparity=ev["disparity"])
    del ctx
    torch.cuda.empty_cache()


def test_full_pokec_all_budgets_zero_shot(stress_report):
    skip_unless_exists(data_path("pokec", "pokec_full.npz"), "full Pokec build")
    import sys
    sys.path.insert(0, os.path.join(ROOT, "experiments", "v2"))
    from common_v2 import PRESETS, detect_preset                 # the presets of the v2 scripts
    pname = detect_preset()
    P = PRESETS[pname]
    dev = "cuda"
    g = load_pokec("pokec_full", "region")
    with GpuPeak() as prep:
        ctx = prepare(g, list(BUDGETS), dev, R=P["full_R"], max_roots=P["full_max_roots"], rr_seed=7,
                      edge_device="cpu")
    ctx.chunk_edges = P["chunk_edges"]
    stress_report(test="full_pokec_prepare", preset=pname, secs=prep.secs, peak_alloc_gb=prep.alloc,
                  roots=ctx.est.n_roots, rr_sets=ctx.est.n_sets, nnz=ctx.est.nnz)
    assert prep.reserved < 0.95 * gpu_total_gb()
    pol, rounds = _policy(dev)
    rg = B.RRGreedy(ctx.est)
    for k in BUDGETS:
        for method, fn in {"FIM-GAT v1": lambda: select_seeds(pol, ctx, k, 1.0, rounds),
                           "FW-Fair": lambda: B.fw_fair(ctx.est, k, 1.0, ctx.total_ref[k]),
                           "Parity": lambda: rg.parity(k)}.items():
            with GpuPeak() as run:
                seeds = fn()
            _check(g, seeds, k)
            assert run.reserved < 0.95 * gpu_total_gb()
            stress_report(test="full_pokec_budget", k=k, method=method, secs=run.secs, peak_alloc_gb=run.alloc)
        with GpuPeak() as mc:
            ev = evaluate(g, ctx.greedy_seeds[k], n_sims=1000, seed=1)
        stress_report(test="full_pokec_mc", k=k, secs=mc.secs, influence=ev["influence"])
    del ctx
    torch.cuda.empty_cache()


def test_batch_evaluation_chunks_many_candidates(stress_report):
    """The v2 verifier scores many candidate sets at once; chunking must bound memory."""
    skip_unless_exists(data_path("pokec", "pokec_n.npz"), "Pokec build")
    g = load_pokec("pokec_n", "region")
    ctx = prepare(g, [200], "cuda", R=16, rr_seed=5)
    rng = np.random.default_rng(0)
    X = torch.zeros(512, g.n, device="cuda")
    for i in range(512):
        X[i, torch.as_tensor(rng.choice(g.n, 200, replace=False), device="cuda")] = 1.0
    with GpuPeak() as run:
        rates, totals = ctx.est.batch_exposure_rates(X)
    assert rates.shape == (512, g.n_groups) and torch.isfinite(rates).all()
    r0, t0 = ctx.est.evaluate_set(torch.nonzero(X[0]).flatten().tolist())
    assert np.allclose(rates[0].cpu().numpy(), r0, atol=1e-5)
    assert run.reserved < 0.95 * gpu_total_gb()
    stress_report(test="batch_512x200", secs=run.secs, peak_alloc_gb=run.alloc)
