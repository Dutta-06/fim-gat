"""v2 end-to-end pipeline tests (marker spec): every stage, chained, on the 2,000-node DC-SBM, CPU only.

Each test skips until the fimgat.v2 modules it needs import (the need() pattern of test_spec_v2.py).
Chain under test (docs/v2_interfaces.md section 19 amends the earlier sections):

    prepare_v2 -> V2Trainer.fit (2 short epochs) -> select_seeds_v2 (meets config.rr_floor on the RR sample)
      -> improve(audit=True)        floor kept, labelled disparity never rises
      -> LearnedProposer            valid moves only
      -> evaluate_v2 / equal_spread_test   finite Monte Carlo metrics
    run_suite (k = 25)              a valid set for every method it runs; a missing method carries a reason
    RecorderV2                      a row survives the CSV round trip and done() finds it
    plus: dynamic_v2 invariance under group relabelling, dmon_loss against textbook modularity.

Expected runtime (UNVERIFIED, never run): 2-5 minutes on the CPU, most of it in the first numba compile,
the baseline suite (LocalSearch runs to a local optimum) and Louvain for the community metrics. Nothing here
writes outside pytest's tmp_path (Louvain caching is redirected by conftest).
"""
import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import ROOT
from fimgat.diffusion import ExposureEstimator

pytestmark = pytest.mark.spec

K, RHO = 25, 0.99
HISTORY_KEYS = {"round", "total", "disp", "comm_disp", "fallback", "accepted", "n_candidates", "family_size", "secs"}


def need(module):
    try:
        return importlib.import_module(f"fimgat.v2.{module}")
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("fimgat.v2"):
            pytest.skip(f"fimgat.v2.{module} not implemented yet")
        raise


def _common_v2():
    """experiments/v2/common_v2.py, loaded by file path (experiments/ is not a package)."""
    name = "_fimgat_v2_common_for_tests"
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    path = os.path.join(ROOT, "experiments", "v2", "common_v2.py")
    if not os.path.exists(path):
        pytest.skip("experiments/v2/common_v2.py not written yet")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("fimgat.v2"):
            pytest.skip(f"{e.name} not implemented yet")
        raise
    sys.modules[name] = mod
    return mod


def _supports(fn, param):
    """True if fn (or a class's __init__) takes `param`: lets tests use pending core keywords when present."""
    return param in inspect.signature(fn).parameters


def _check_set(seeds, k, n, what="seeds"):
    s = [int(v) for v in seeds]
    assert all(float(a) == b for a, b in zip(seeds, s)), f"{what}: non-integer node ids"
    assert len(s) == k == len(set(s)), f"{what}: {len(s)} seeds ({len(set(s))} distinct) for budget {k}"
    assert 0 <= min(s) and max(s) < n, f"{what}: node id outside [0, {n})"
    return s


def _rr(pipe, seeds):
    """(exact float64 disparity, RR total) of a seed set on the pipeline's RR sample."""
    rates, total = pipe.T.exact_rates(pipe.est, seeds)
    return pipe.obj.disparity(rates, pipe.ctx.present_np), total


def _monotone(hist):
    ds = [h["disp"] for h in hist]
    return all(b <= a + 1e-12 for a, b in zip(ds, ds[1:]))


def _move_parts(move):
    """(removes, adds) tuples of a swap (u, v) or a multi-node move (removes, adds)."""
    a, b = move
    if isinstance(a, (int, np.integer)):
        assert isinstance(b, (int, np.integer)), f"malformed swap {move!r}"
        return (int(a),), (int(b),)
    return tuple(int(u) for u in a), tuple(int(v) for v in b)


# ------------------------------------------------------------------ the shared pipeline

@pytest.fixture(scope="module")
def pipe(sbm_graph):
    """prepare_v2 + 2 short training epochs + select_seeds_v2, once for the module.

    The graph is sbm_graph with a meta dict (a labelled attribute plus one held-out attribute), under a
    ':'-suffixed name, so it shares sbm_graph's Louvain cache entry (same topology) but not its object."""
    cfg, T, obj = need("config"), need("train"), need("objective")
    g = sbm_graph.with_groups(sbm_graph.groups, list(sbm_graph.group_names), suffix="v2spec")
    g.meta = {"dataset": "sbm_v2spec", "attribute": "xyz",
              "attrs": {"xyz": g.groups.copy(), "parity": (np.arange(g.n) % 2).astype(np.int64)}}
    ctx = T.prepare_v2(g, [K], "cpu", R=8, rr_seed=3)
    tr = T.V2Trainer("cpu", seed=0, hidden=32, heads=4)
    best = tr.fit([ctx], [ctx], [K], epochs=2, steps_per_epoch=2, eval_every=1, verbose=False)
    floor = cfg.rr_floor(RHO, ctx.total_ref[K])
    seeds = T.select_seeds_v2(tr.policy, ctx, K, RHO, floor, tr.rounds, n_samples=16, mu=tr.mu_for(K))
    return SimpleNamespace(cfg=cfg, T=T, obj=obj, g=g, ctx=ctx, est=ctx.est, tr=tr, best=best, floor=floor,
                           seeds=[int(s) for s in seeds])


# ------------------------------------------------------------------ constructor: training and decoding

def test_constructor_trains_and_selects_a_set_meeting_the_rr_floor(pipe):
    T, ctx = pipe.T, pipe.ctx
    greedy = ctx.greedy_seeds[K]
    assert ctx.total_ref[K] == T.exact_rates(ctx.est, greedy)[1]          # total_ref: exact RR total of RR-greedy
    assert math.isfinite(pipe.best)
    assert len(pipe.tr.log) == 2                                          # eval_every=1, epochs=2
    for rec in pipe.tr.log:
        assert math.isfinite(rec["loss"]) and math.isfinite(rec["ratio"]) and math.isfinite(rec["val"])
        # eff_comm, the scale-free collapse diagnostic of the community adversary, lies in [1, K]
        assert 1.0 - 1e-6 <= rec["eff_comm"] <= pipe.tr.adversary.K + 1e-6
    _check_set(pipe.seeds, K, ctx.n, "select_seeds_v2")
    _, total = _rr(pipe, pipe.seeds)
    assert total >= pipe.floor                   # a candidate meeting the floor, else the RR-greedy set


def test_checkpoint_round_trip_reproduces_the_selection(pipe, tmp_path):
    T = pipe.T
    path = str(tmp_path / "constructor.pt")
    pipe.tr.save(path, note="spec")
    pol, ck = T.load_policy(path, "cpu")
    ref_state = pipe.tr.policy.state_dict()
    assert all(torch.equal(v, ref_state[k]) for k, v in pol.state_dict().items())
    assert ck["mu_ref"][K] == pytest.approx(pipe.tr.mu_for(K))
    seeds = T.select_seeds_v2(pol, pipe.ctx, K, RHO, pipe.floor, ck["rounds"], n_samples=16, mu=ck["mu_ref"][K])
    assert sorted(int(s) for s in seeds) == sorted(pipe.seeds)             # decoding is seeded: same set


# ------------------------------------------------------------------ improvement phase

def test_improve_with_audit_keeps_floor_and_never_raises_disparity(pipe):
    imp = need("improve")
    d0, _ = _rr(pipe, pipe.seeds)
    seeds, hist = imp.improve(pipe.est, pipe.seeds, pipe.floor, proposer="simple", audit=True, max_rounds=15,
                              max_seconds=60)
    seeds = _check_set(seeds, K, pipe.ctx.n, "improve")
    d1, t1 = _rr(pipe, seeds)
    assert t1 >= pipe.floor
    assert hist and all(HISTORY_KEYS <= set(h) for h in hist) and hist[0]["round"] == 0
    assert _monotone(hist), [h["disp"] for h in hist]
    fam = [h["family_size"] for h in hist]
    assert all(b >= a for a, b in zip(fam, fam[1:]))                      # the community family only grows
    assert d1 == pytest.approx(hist[-1]["disp"], abs=1e-12) and d1 <= hist[0]["disp"] + 1e-12
    if not hist[0]["fallback"]:                                           # the start met the floor
        assert hist[0]["disp"] == pytest.approx(d0, abs=1e-12) and d1 <= d0 + 1e-12


def test_improve_can_return_its_community_family(pipe):
    imp, aud = need("improve"), need("auditor")
    if not _supports(imp.improve, "return_family"):
        pytest.skip("pending core change: improve(..., return_family=...)")
    seeds, hist = imp.improve(pipe.est, pipe.seeds, pipe.floor, proposer="simple", audit=True, max_rounds=5,
                              return_family=True)
    _check_set(seeds, K, pipe.ctx.n, "improve")
    comms = hist[-1]["communities"]
    assert len(comms) == hist[-1]["family_size"]
    for c in comms:                              # no communities were given: every member came from the auditor
        c = np.asarray(c, dtype=np.int64)
        assert len(c) >= pipe.cfg.MIN_COMMUNITY and len(np.unique(c)) == len(c)
        assert 0 <= c.min() and c.max() < pipe.ctx.n
        assert aud.conductance(pipe.g, c) <= pipe.cfg.MAX_CONDUCTANCE + 1e-12


def test_learned_proposer_returns_valid_moves(pipe):
    P, cov, imp = need("proposer"), need("cover"), need("improve")
    kw = {"mu_ref": pipe.tr.mu_for(K)} if _supports(P.LearnedProposer, "mu_ref") else {}
    prop = P.LearnedProposer(pipe.tr.policy, pipe.ctx, RHO, pipe.ctx.total_ref[K], n_add=16, n_drop=8, n_multi=4,
                             **kw)
    state = cov.CoverState(pipe.est, pipe.seeds)
    before = (list(state.seeds), state.total, state.counts.copy())
    moves = prop(state, None, np.random.default_rng(0))
    assert state.seeds == before[0] and state.total == before[1] and np.array_equal(state.counts, before[2])
    assert moves, "no moves proposed"
    seeds, n = set(state.seeds), pipe.ctx.n
    seen, n_swap, n_multi = set(), 0, 0
    for mv in moves:
        rem, add = _move_parts(mv)
        assert 1 <= len(rem) == len(add), f"{mv!r} changes the budget"
        assert len(set(rem)) == len(rem) and len(set(add)) == len(add), f"{mv!r} repeats a node"
        assert set(rem) <= seeds, f"{mv!r} removes a non-seed"
        assert not set(add) & seeds and all(0 <= v < n for v in add), f"{mv!r} adds a seed or an invalid id"
        key = (tuple(sorted(rem)), tuple(sorted(add)))
        assert key not in seen, f"duplicate move {mv!r}"
        seen.add(key)
        n_swap, n_multi = n_swap + (len(rem) == 1), n_multi + (len(rem) > 1)
        s = state.copy()                          # applying the move keeps k distinct seeds
        for u in rem:
            s.remove(u)
        for v in add:
            s.add(v)
        assert s.k == K and len(set(s.seeds)) == K
    assert n_swap <= 16 * 8 and n_multi <= 4
    out, hist = imp.improve(pipe.est, pipe.seeds, pipe.floor, proposer=prop, audit=False, max_rounds=5,
                            max_seconds=60)
    out = _check_set(out, K, n, "improve(LearnedProposer)")
    assert _rr(pipe, out)[1] >= pipe.floor and _monotone(hist)


# ------------------------------------------------------------------ Monte Carlo evaluation

def test_evaluate_v2_returns_finite_metrics(pipe):
    mv = need("metrics_v2")
    m = mv.evaluate_v2(pipe.g, pipe.seeds, n_sims=300, seed=7)
    for key in ("influence", "influence_se", "disparity", "min_rate", "max_rate", "heldout_disparity",
                "heldout_worst_rel", "comm_disparity", "comm_worst_rel", "comm_q10_rel"):
        assert math.isfinite(m[key]), f"{key} = {m[key]}"
    assert m["n_comm_eval"] >= 1                                          # Louvain communities of >= 10 nodes
    assert len(m["rates"]) == int(pipe.g.present_groups.sum())
    assert m["influence"] >= K and 0.0 <= m["disparity"] <= 1.0
    assert m["disparity"] == pytest.approx(max(m["rates"]) - min(m["rates"]), abs=1e-12)
    gt = [np.arange(0, 200), np.arange(100, 400), np.array([5, 5, 6])]   # overlapping; a repeated id counts once
    m2 = mv.evaluate_v2(pipe.g, pipe.seeds, n_sims=300, seed=7, communities=gt, return_sims=True)
    assert m2["n_comm_eval"] == 3 and all(math.isfinite(m2[c]) for c in ("comm_disparity", "comm_worst_rel"))
    assert m2["influence"] == m["influence"]                              # same set, same stream
    assert len(m2["sims"]) == 300 and float(np.mean(m2["sims"])) == pytest.approx(m2["influence"])


def test_equal_spread_test_runs_on_the_constructor_output(pipe):
    ref = need("reference")
    r = ref.reference_spread(pipe.g, K, runs=2, R=8, seed=0, n_sims=300)
    _check_set(r["seeds"], K, pipe.ctx.n, "reference")
    es = ref.equal_spread_test(pipe.g, pipe.seeds, r["seeds"], rho=RHO, n_sims=300, seed=11, alpha=0.05)
    assert isinstance(es["passed"], bool)
    assert math.isfinite(es["ratio"]) and es["ratio"] > 0
    assert math.isfinite(es["se"]) and es["se"] >= 0 and es["lower"] <= es["ratio"] + 1e-12
    assert es["passed"] == (es["lower"] >= RHO)


# ------------------------------------------------------------------ baseline suite

def test_run_suite_returns_valid_sets_and_missing_methods_carry_a_reason(pipe):
    bl, T = need("baselines_v2"), pipe.T
    # AGE and Prior take minutes on the CPU; they are checked below through the missing-method path
    methods = [m for m in bl.SUITE if m not in ("AGE", "Prior")]
    ref = pipe.ctx.greedy_seeds[K]
    res = bl.run_suite(pipe.ctx, K, RHO, pipe.floor, ref, methods=methods, cache={})
    assert set(res) == set(methods)
    for name, (seeds, secs) in res.items():
        assert seeds is not None, f"{name} is missing: {res.reason.get(name)}"
        _check_set(seeds, K, pipe.ctx.n, name)
        assert math.isfinite(secs) and secs >= 0
    for name in ("FW-Disparity", "LocalSearch", "Simple-v2"):              # these return floor-meeting sets
        assert T.exact_rates(pipe.est, res[name][0])[1] >= pipe.floor * (1 - 1e-12), name
    # a method that cannot run within the time limit is recorded as missing, with its reason
    slow = bl.run_suite(pipe.ctx, K, RHO, pipe.floor, ref, methods=["AGE"], time_limit=1.0)
    assert slow["AGE"][0] is None and isinstance(slow.reason.get("AGE"), str) and slow.reason["AGE"]


# ------------------------------------------------------------------ features: group relabelling

def test_dynamic_v2_invariant_to_group_relabelling(sbm_graph):
    """Relative group features: one model serves any group definition (v2 contract section 12)."""
    F, obj, cov = need("features"), need("objective"), need("cover")
    n, perm = sbm_graph.n, np.array([2, 0, 1])
    g2 = sbm_graph.with_groups(perm[sbm_graph.groups], [sbm_graph.group_names[i] for i in np.argsort(perm)])
    ctxs = []
    for g in (sbm_graph, g2):
        c = F.build_context(g, ExposureEstimator(g, R=4, seed=9, device="cpu"), "cpu")
        c.total_ref[K] = 321.0
        ctxs.append(c)
    assert torch.allclose(ctxs[0].static, ctxs[1].static, atol=1e-5)
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=gen) * 0.02
    idx = torch.randperm(n, generator=gen)[:K]
    x[idx] = 0.9                                   # some x > 0.5, so the seed-only drop column is exercised
    dyn = []
    for c in ctxs:
        a, rates, total, miss = c.est(x)
        gains = c.est.group_gains(x, miss)
        w, _, _ = obj.signed_prices(rates, c.present_np, c.graph.group_sizes, 0.7)
        dyn.append(F.dynamic_v2(c, x, a, rates, total, gains, w, RHO, 1, 3, K))
    assert dyn[0].shape == (n, F.N_DYNAMIC_V2) and torch.isfinite(dyn[0]).all()
    assert torch.allclose(dyn[0], dyn[1], atol=1e-4)
    # binary sets through a CoverState: the learned proposer's inference path
    seeds = idx.tolist()
    st = []
    for c in ctxs:
        s = cov.CoverState(c.est, seeds)
        w, _, _ = obj.signed_prices(s.rates, s.present, s.sizes, 0.7)
        st.append(F.dynamic_from_state(c, s, w, RHO, 321.0, 2, 3))
    assert torch.isfinite(st[0]).all() and torch.allclose(st[0], st[1], atol=1e-4)


# ------------------------------------------------------------------ community adversary loss

def _textbook_modularity(edges, labels):
    """Newman: Q = sum_c [L_c / m - (d_c / 2m)^2] for an undirected simple graph and a hard partition."""
    m = len(edges)
    deg = np.zeros(len(labels))
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    q = 0.0
    for c in set(labels):
        inside = sum(1 for u, v in edges if labels[u] == c and labels[v] == c)
        d_c = sum(deg[i] for i in range(len(labels)) if labels[i] == c)
        q += inside / m - (d_c / (2 * m)) ** 2
    return q


@pytest.mark.parametrize("edges,labels,expected", [
    ([(0, 1), (2, 3)], [0, 0, 1, 1], 0.5),        # two separate edges, one community each
    ([(0, 1), (1, 2)], [0, 0, 1], -0.125),        # a path cut after its first edge
    ([(0, 1), (1, 2)], [0, 0, 0], 0.0),           # everything in one community (the other one empty)
])
def test_dmon_loss_matches_textbook_modularity_on_two_edge_graphs(edges, labels, expected):
    mdl = need("model")
    n, K_ = len(labels), 2
    q = _textbook_modularity(edges, labels)
    assert q == pytest.approx(expected, abs=1e-12)
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    comms = [set(i for i in range(n) if labels[i] == c) for c in sorted(set(labels))]
    assert nx.community.modularity(G, comms) == pytest.approx(q, abs=1e-12)

    S = torch.zeros(n, K_)
    S[torch.arange(n), torch.tensor(labels)] = 1.0
    both = torch.tensor([list(e) for e in edges] + [[v, u] for u, v in edges]).t().contiguous()
    loops = torch.arange(n).repeat(2, 1)
    collapse = math.sqrt(K_) / n * float(S.sum(0).norm()) - 1.0          # Tsitsulin et al. regulariser
    for ei in (both, torch.cat([both, loops], 1)):                       # self-loops (as in GraphContext) are ignored
        assert float(mdl.dmon_loss(S, ei, n)) == pytest.approx(-q + collapse, abs=1e-6)


# ------------------------------------------------------------------ results recorder

def test_recorder_round_trips_a_row_and_done_finds_it(pipe, tmp_path):
    C = _common_v2()
    path = tmp_path / "v2" / "rows.csv"
    rec = C.RecorderV2(str(path), n_sims=200, preset="8gb")
    ref = pipe.ctx.greedy_seeds[K]
    row = rec.add(pipe.g, K, RHO, "FIM-GAT", pipe.seeds, 1.25, ref_seeds=ref, variant="spec", train_seed=0,
                  prep_s=0.5, note="round trip")
    rec.add(pipe.g, K, RHO, "Greedy", ref, 0.1, ref_seeds=ref)             # a baseline: default variant, NaN seed
    rec.add_missing(pipe.g, K, RHO, "AGE", "skipped: spec test")
    assert path.exists()

    back = C.RecorderV2(str(path), n_sims=200, preset="8gb")
    assert len(back.rows) == 3
    key = dict(dataset="sbm_v2spec", attribute="xyz", k=K, rho=RHO, method="FIM-GAT", variant="spec", train_seed=0)
    assert back.done(**key)
    for change in ({"rho": 0.98}, {"variant": "default"}, {"train_seed": 1}, {"k": 50}, {"attribute": "parity"},
                   {"dataset": "other"}, {"method": "Greedy"}):
        assert not back.done(**{**key, **change}), change
    (r,) = back.find(**key)
    assert json.loads(r["seeds"]) == pipe.seeds and r["status"] == "ok" and r["note"] == "round trip"
    for c in ("influence", "disparity", "es_ratio", "es_lower"):
        assert float(r[c]) == pytest.approx(float(row[c]), rel=1e-12, abs=1e-15), c

    base = dict(dataset="sbm_v2spec", attribute="xyz", k=K, rho=RHO, variant="default", train_seed=float("nan"))
    assert back.done(method="Greedy", **base)
    assert back.done(method="AGE", **base) and not back.done(method="AGE", status="ok", **base)
    with pytest.raises(ValueError):
        back.done(method="Greedy", k=K)                                    # rho and variant are required
