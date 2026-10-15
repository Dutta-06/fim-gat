"""Checks of the pre-run fixes approved on 2026-10-14 (docs/design_v2.md section 21), marker spec.

  deployment rules   certify.choose: Theorem 2's rule ("min_ucb") and the path rule ("path"); deploy_candidates in
                     path order; certified_choice keeps its default behaviour
  constants          RHO_HOLDS leaves rho = 1.0 out; T2's R and sample count; T3's pre-registered constants
  BSM sweep          external.bsm_sweep gives every tau's set; the bsm adapter picks one of them (or Greedy's set)
  analysis           summarize_v2's dominance, T3 rule and T4 filter on synthetic rows
  T3 chain           train_proposer.py -> run_t3.py -> summarize_v2.py end to end on synthetic stand-ins (the first
                     execution of the learned-proposer scripts), with fimgat.v2.datasets patched and every output in
                     pytest's tmp_path
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from fimgat.baselines import RRGreedy
from fimgat.diffusion import ExposureEstimator
from fimgat.graph import from_edges

pytestmark = pytest.mark.spec

CFG = pytest.importorskip("fimgat.v2.config")
CER = pytest.importorskip("fimgat.v2.certify")
COV = pytest.importorskip("fimgat.v2.cover")
EXT = pytest.importorskip("fimgat.v2.external")
IMP = pytest.importorskip("fimgat.v2.improve")
EXP = os.path.join(ROOT, "experiments", "v2")


def _blocks_graph(seed=0, n_blocks=8, size=40):
    rng = np.random.default_rng(seed)
    n = n_blocks * size
    block = np.repeat(np.arange(n_blocks), size)
    edges = [(v, w) for v in range(n) for w in rng.choice(np.flatnonzero(block == block[v]), 4, replace=False)]
    edges += [(v, int(rng.integers(n))) for v in range(n) if rng.random() < 0.5]
    groups = np.where(rng.random(n) < 0.6, block % 3, rng.integers(0, 3, n))
    return from_edges("blocks", n, np.array(edges), groups, ["a", "b", "c"], undirected=True, weighted_cascade=True)


@pytest.fixture(scope="module")
def blocks():
    g = _blocks_graph()
    est = ExposureEstimator(g, R=16, seed=0, device="cpu")
    ref = [int(v) for v in RRGreedy(est).greedy(10)]
    floor = CFG.rr_floor(CFG.RHO_C2, COV.CoverState(est, ref).total)
    return g, est, ref, floor


def _summarize():
    if EXP not in sys.path:
        sys.path.insert(0, EXP)
    import summarize_v2
    return summarize_v2


# ------------------------------------------------------------------ deployment rules

def _cert(pos, ucb, se=0.01, elig=True, ref=False, est=0.0):
    return {"seeds": [pos], "position": pos, "disp_ucb": ucb, "disp_se": se, "eligible": elig,
            "is_reference": ref, "floor_estimate": est}


def test_min_ucb_rule_is_theorem_2s():
    certs = [_cert(0, 0.30, ref=True), _cert(1, 0.20), _cert(2, 0.10, elig=False), _cert(3, 0.25)]
    assert CER.choose(certs, "min_ucb") == 1                               # the ineligible 0.10 never counts
    certs[3]["disp_ucb"] = 0.20                                             # a tie: the larger spread, then the earlier
    certs[3]["floor_estimate"] = 1.0
    assert CER.choose(certs, "min_ucb") == 3


def test_path_rule_takes_the_furthest_state_within_tolerance():
    certs = [_cert(0, 0.30, ref=True), _cert(1, 0.200, se=0.02), _cert(2, 0.215), _cert(3, 0.25), _cert(4, 0.21,
                                                                                                         elig=False)]
    # tolerance = 1 standard error of the smallest-bound set (0.02): states 1 and 2 qualify, 3 does not, 4 is not
    # eligible
    assert CER.choose(certs, "path") == 2
    assert CER.choose(certs, "path", tol_se=0.0) == 1                       # no tolerance: the min-UCB set
    assert CER.choose(certs, "path", tol_se=3.0) == 3                       # 0.25 <= 0.20 + 3 * 0.02
    with pytest.raises(ValueError):
        CER.choose(certs, "unknown")


def test_path_rule_prefers_a_searched_state_over_an_appended_reference():
    certs = [_cert(0, 0.21), _cert(1, 0.22), _cert(-1, 0.20, ref=True)]     # the reference, appended at position -1
    assert CER.choose(certs, "min_ucb") == 2
    assert CER.choose(certs, "path") == 0                                   # cap 0.20 + 0.01: state 0 qualifies, 1 not
    certs[1]["disp_ucb"] = 0.205
    assert CER.choose(certs, "path") == 1                                   # the furthest qualifying state
    certs[0]["disp_ucb"] = certs[1]["disp_ucb"] = 0.5
    assert CER.choose(certs, "path") == 2                                   # nothing else within tolerance


def test_deploy_candidates_come_in_path_order():
    p1 = [[1, 2], [1, 3], [3, 4], [4, 5], [5, 6]]
    p2 = [[6, 5], [6, 7], [7, 8]]
    assert CER.deploy_candidates([p1, p2], n_mid=3) == [[1, 2], [3, 4], [5, 6], [7, 8]]
    p3 = [[5, 6], [7, 8], [2, 1]]                                           # a later phase returning to the start
    assert CER.deploy_candidates([p1, p3], n_mid=0) == [[5, 6], [1, 2]]      # it counts where it last occurs


def test_certified_choice_rules_on_a_hive_path(blocks):
    g, est, ref, floor = blocks
    seeds, info = IMP.hive_search(est, ref, floor, max_rounds=30)
    s_a = CER.path_states(info["start1"], info["phase1"])[-1]
    cands = CER.deploy_candidates([CER.path_states(info["start1"], info["phase1"]),
                                   CER.path_states(s_a, info["phase2"])])
    est_b = ExposureEstimator(g, R=16, seed=779, device="cpu")
    cert = CER.certify_sets(est_b, cands, ref, CFG.RHO_C2, alpha=CFG.CERT_ALPHA)
    certs = cert["certificates"]
    i_min, i_path = CER.choose(certs, "min_ucb"), CER.choose(certs, "path")
    assert certs[i_min]["eligible"] and certs[i_path]["eligible"]
    assert certs[i_path]["position"] >= certs[i_min]["position"]
    assert certs[i_path]["disp_ucb"] <= certs[i_min]["disp_ucb"] + CFG.DEPLOY_TOL_SE * certs[i_min]["disp_se"] + 1e-12
    old = CER.certified_choice(est_b, cands, ref, CFG.RHO_C2, alpha=CFG.CERT_ALPHA)    # the default is unchanged
    assert old["seeds"] == certs[i_min]["seeds"] and old["rule"] == "min_ucb"
    new = CER.certified_choice(est_b, cands, ref, CFG.RHO_C2, alpha=CFG.CERT_ALPHA, rule="path")
    assert new["seeds"] == certs[i_path]["seeds"]
    for c in certs:                                                        # every certificate has its noise scale
        assert c["disp_se"] >= 0.0 and math.isfinite(c["disp_se"])


# ------------------------------------------------------------------ constants

def test_prerun_constants():
    assert 1.0 not in CFG.RHO_HOLDS and CFG.RHO_C1 in CFG.RHO_HOLDS and CFG.RHO_C2 in CFG.RHO_HOLDS
    assert set(CFG.RHO_HOLDS) <= set(CFG.RHO_REPORT) and 1.0 in CFG.RHO_REPORT
    assert CFG.T2_RR_PER_NODE == 64 and CFG.T2_SAMPLES * len(CFG.BUDGETS) >= CFG.T2_MIN_PAIRS + 5
    assert CFG.DEPLOY_RULES == ("min_ucb", "path") and CFG.DEPLOY_TOL_SE == CFG.KAPPA_SE
    assert CFG.T3_PRIMARY_BUDGET in CFG.TIME_BUDGETS and CFG.T3_SPEEDUP_MIN == CFG.LEARN_HEADROOM_MIN
    assert CFG.T3_RHO == CFG.RHO_C1 and max(CFG.T3_RECALL_C) == 16 * 8


# ------------------------------------------------------------------ BSM sweep

def test_bsm_sweep_gives_every_tau_and_contains_the_adapters_choice(blocks):
    g, est, ref, floor = blocks
    cache = {}
    sweep = EXT.bsm_sweep(est, 10, cache=cache)
    assert [t for t, _ in sweep] == sorted(EXT.TAUS, reverse=True)
    for _, s in sweep:
        assert len(s) == 10 and len(set(s)) == 10 and all(0 <= v < g.n for v in s)
    seeds, info = EXT.bsm(est, 10, floor, reference=ref, cache=cache, return_info=True)
    if info["tau"] is not None:
        assert seeds == dict(sweep)[info["tau"]]
    else:
        assert info["fallback"] in ("greedy", "reference")


# ------------------------------------------------------------------ analysis

def test_dominance_on_synthetic_points():
    S = _summarize()
    base = {"dataset": "d", "attribute": "a", "k": 5}
    pts = pd.DataFrame([dict(base, source="method", method=S.FIM, rho=0.99, es_ratio=0.995, disparity=0.02),
                        dict(base, source="method", method=S.FIM, rho=0.97, es_ratio=0.975, disparity=0.01),
                        dict(base, source="baseline", method="B1", rho=np.nan, es_ratio=0.99, disparity=0.03),
                        dict(base, source="baseline", method="B2", rho=np.nan, es_ratio=0.999, disparity=0.005),
                        dict(base, source="bsm_sweep", method="BSM", rho=np.nan, es_ratio=0.90, disparity=0.001)])
    out = S.dominance(pts).set_index("method")
    assert out.loc["B1", "dominated_by_method"] and not out.loc["B1", "dominates_method"]
    assert out.loc["B2", "dominates_method"] and not out.loc["B2", "dominated_by_method"]
    assert not out.loc["BSM", "dominated_by_method"] and not out.loc["BSM", "dominates_method"]


def _t3_rows(learned_trace, graphs=3):
    rows = []
    for gi in range(graphs):
        for k in (5, 10):
            for b in (10.0, 60.0, 300.0):
                simple = [[0.0, 0.05], [10.0, 0.03], [50.0, 0.02]]
                for prop, ts, tr in (("simple", -1, simple), ("learned", 0, learned_trace)):
                    reached = [p for p in tr if p[0] <= b][-1]
                    rows.append({"dataset": f"g{gi}", "attribute": "a", "k": k, "proposer": prop, "train_seed": ts,
                                 "budget_s": b, "rr_disp": reached[1], "disparity": reached[1] + 0.001,
                                 "trace": json.dumps(tr), "status": "ok"})
    return pd.DataFrame(rows)


def test_t3_rule_on_synthetic_rows():
    S = _summarize()
    md, out = [], []
    S.t3(_t3_rows([[0.0, 0.05], [5.0, 0.02]]), None, {"w3": {"train": True}}, out, md)
    text = "\n".join(md)
    assert "Faster (median speed-up >= 1.5) on 3 of 3 test graphs: pass" in text and "T3: pass" in text
    md, out = [], []
    S.t3(_t3_rows([[0.0, 0.05], [200.0, 0.04]]), None, {"w3": {"train": True}}, out, md)
    assert "T3: fail" in "\n".join(md)
    md = []
    S.t3(None, None, {"w3": {"train": False, "reason": "headroom H / cost_ratio = 1.10 < 1.5", "H": 1.1}}, [], md)
    assert "Not run under the pre-registered rule" in "\n".join(md)


def test_t4_summary_keeps_only_deployed_fim_rows(tmp_path):
    S = _summarize()
    rows = [{"dataset": "d", "attribute": "a", "method": S.FIM, "variant": "v2-b", "deployment": dep,
             "prep_s": 10.0, "query_s": q, "disparity": 0.01, "cert_passed": True}
            for dep, q in (("certified", 5.0), ("certified-path", 5.0), ("final", 1.0))]
    rows.append({"dataset": "d", "attribute": "a", "method": "Greedy", "variant": "default", "deployment": np.nan,
                 "prep_s": 3.0, "query_s": 2.0, "disparity": 0.05, "cert_passed": True})
    md = []
    S.t4(pd.DataFrame(rows), md, str(tmp_path))
    agg = pd.read_csv(tmp_path / "summary_t4.csv").set_index("method")
    assert agg.loc[S.FIM, "query_s"] == 5.0 and agg.loc["Greedy", "query_s"] == 2.0


# ------------------------------------------------------------------ the T3 chain end to end

def _planted(name, n_blocks, size, seed, dataset):
    rng = np.random.default_rng(seed)
    n = n_blocks * size
    block = np.repeat(np.arange(n_blocks), size)
    edges = [(v, int(w)) for v in range(n) for w in rng.choice(np.flatnonzero(block == block[v]), 4, replace=False)]
    edges += [(v, int(rng.integers(n))) for v in range(n) if rng.random() < 0.4]
    region = np.where(rng.random(n) < 0.7, block % 3, rng.integers(0, 3, n)).astype(np.int64)
    gender = np.where(rng.random(n) < 0.6, (block // 2) % 2, rng.integers(0, 2, n)).astype(np.int64)
    base = from_edges(name, n, np.array(edges), region, ["r0", "r1", "r2"], undirected=True, weighted_cascade=True)
    from fimgat.v2.datasets import degree_group_names, degree_groups
    attrs = {"region": region, "gender": gender, "degree": degree_groups(base)}
    names = {"region": ["r0", "r1", "r2"], "gender": ["f", "m"], "degree": degree_group_names(2)}
    base.meta = {"dataset": dataset, "attribute": None, "attrs": attrs, "attr_names": names, "tier": "small",
                 "undirected": True}
    return base


@pytest.fixture
def t3_env(monkeypatch, tmp_path):
    if EXP not in sys.path:
        sys.path.insert(0, EXP)
    import fimgat.v2.datasets as D
    graphs = {"fake_train": _planted("fake_train", 6, 50, 11, "fake_train"),
              "fake_val": _planted("fake_val", 6, 50, 12, "fake_val"),
              "fake_large": _planted("fake_large", 6, 50, 13, "fake_large")}

    def load(name, attribute=None):
        base = graphs[name]
        attribute = attribute or "region"
        g = base.with_groups(base.meta["attrs"][attribute], list(base.meta["attr_names"][attribute]))
        g.name = name
        g.meta = dict(base.meta, attribute=attribute)
        return g
    monkeypatch.setattr(D, "available", lambda name: name in graphs)
    monkeypatch.setattr(D, "load", load)
    monkeypatch.setattr(D, "LOADABLE", D.LOADABLE + tuple(graphs))
    monkeypatch.setattr(D, "DEFAULT_ATTRIBUTE", dict(D.DEFAULT_ATTRIBUTE, **{x: "region" for x in graphs}))
    monkeypatch.setattr(D, "splits", lambda: {"train": ["fake_train"], "val": ["fake_val"], "test": ["fake_large"]})
    monkeypatch.setitem(CFG.TIER, "fake_train", "small")
    monkeypatch.setitem(CFG.TIER, "fake_val", "small")
    monkeypatch.setitem(CFG.TIER, "fake_large", "large")
    tuning = tmp_path / "tuning.json"
    tuning.write_text(json.dumps({"delta": CFG.DELTA_MARGIN, "phi0": CFG.MAX_CONDUCTANCE, "start": "greedy",
                                  "tuned": True, "w3": {"train": True, "H": 2.0, "reason": "synthetic"}}))

    def run(module, argv):
        monkeypatch.setattr(sys, "argv", [module.__name__ + ".py"] + argv)
        module.main()
    return tmp_path, tuning, run


def test_train_proposer_then_t3_then_summary(t3_env):
    tmp, tuning, run = t3_env
    import run_t3
    import summarize_v2
    import train_proposer
    from fimgat.v2.train import load_policy
    models = tmp / "models"
    run(train_proposer, ["--tuning", str(tuning), "--seeds", "0", "--ks", "5", "--attributes", "region", "gender",
                         "--val-attributes", "region", "--states-per-query", "2", "--val-states-per-query", "2",
                         "--epochs", "1", "--hidden", "16", "--layers", "1", "--heads", "2", "--R", "8",
                         "--models-dir", str(models), "--log", str(tmp / "train_proposer_log.csv"),
                         "--constructor-pattern", str(tmp / "none_s{seed}.pt")])
    path = models / "proposer_s0.pt"
    assert path.exists()
    pol, ck = load_policy(str(path), "cpu")
    assert ck["hparams"]["role"] == "proposer" and ck["meta"]["init"] == "fresh"
    assert 0.0 <= ck["meta"]["best_val_recall"] <= 1.0 and set(ck["meta"]["simple_val_recall"]) == {"8", "32", "128"}
    log = pd.read_csv(tmp / "train_proposer_log.csv")
    assert list(log.epoch) == [0, 1]

    common = ["--tuning", str(tuning), "--datasets", "fake_large", "--attributes", "region", "--ks", "5",
              "--budgets", "0.5", "3", "--train-seeds", "0", "--proposer-pattern", str(models / "proposer_s{seed}.pt"),
              "--ref-file", str(tmp / "refs.json"), "--n-sims", "100", "--preset", "8gb", "--R", "8",
              "--max-roots", "0", "--reference-runs", "2", "--out", str(tmp / "t3.csv"),
              "--recall-out", str(tmp / "t3_recall.csv"), "--model-pattern", str(tmp / "none_s{seed}.pt")]
    run(run_t3, common)
    d3 = pd.read_csv(tmp / "t3.csv")
    assert set(d3.proposer) == {"simple", "exhaustive", "learned"} and set(d3.budget_s) == {0.5, 3.0}
    assert len(d3) == 3 * 2 and (d3.deployment == "search").all() and d3.disparity.notna().all()
    for _, r in d3.iterrows():                                              # the set at a budget was reached by then
        assert r.reached_s <= r.budget_s + 1e-9 and r.moves == len([t for t, _ in json.loads(r.trace)[1:]
                                                                      if t <= r.budget_s])
    rr = pd.read_csv(tmp / "t3_recall.csv")
    assert set(rr.proposer) == {"simple", "learned"} and {"recall_8", "recall_32", "recall_128"} <= set(rr.columns)
    run(run_t3, common)                                                     # resumed: nothing new
    assert len(pd.read_csv(tmp / "t3.csv")) == len(d3)

    run(summarize_v2, ["--dir", str(tmp)])                                  # reads tmp / tuning.json
    md = (tmp / "summary_v2.md").read_text(encoding="utf-8")
    assert "## T3" in md and ("T3: pass" in md or "T3: fail" in md)


def test_train_proposer_respects_the_w3_gate(t3_env, capsys):
    tmp, tuning, run = t3_env
    import train_proposer
    tuning.write_text(json.dumps({"delta": CFG.DELTA_MARGIN, "phi0": CFG.MAX_CONDUCTANCE, "start": "greedy",
                                  "w3": {"train": False, "reason": "headroom too small"}}))
    run(train_proposer, ["--tuning", str(tuning), "--models-dir", str(tmp / "models")])
    assert "not trained" in capsys.readouterr().out and not (tmp / "models").exists()
