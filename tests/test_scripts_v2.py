"""End-to-end checks of the v2 experiment scripts on tiny synthetic graphs (marker spec; docs/design_v2.md section 19).

tune_v2 -> run_t1_t2 (T1, T2) -> summarize_v2, and run_t4, with fimgat.v2.datasets patched to serve three planted-
community graphs: a validation graph, a test graph and a com-Youtube stand-in with ground-truth communities. Every
output goes to pytest's tmp_path; nothing touches data/ or results/. Checks the files and columns the analysis
depends on, three rows per FIM-GAT query (certified, certified-path and final; design section 21), BSM's tau sweep
(t1_pareto.csv), resumption, the summary's sections and figures, and that tuning refuses a test graph.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from fimgat.graph import from_edges

pytestmark = pytest.mark.spec

EXP = os.path.join(ROOT, "experiments", "v2")


def _planted(name, n_blocks, size, seed, dataset, structural=False):
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
    if structural:
        attrs, names = {"degree": attrs["degree"]}, {"degree": names["degree"]}
    base.meta = {"dataset": dataset, "attribute": None, "attrs": attrs, "attr_names": names, "tier": "small",
                 "undirected": True}
    return base, [np.flatnonzero(block == b) for b in range(n_blocks)]


@pytest.fixture
def scripts(monkeypatch, tmp_path):
    if EXP not in sys.path:
        sys.path.insert(0, EXP)
    import fimgat.v2.datasets as D
    from fimgat.v2 import config as C
    graphs = {"fake_val": _planted("fake_val", 6, 50, 1, "fake_val"),
              "fake_test": _planted("fake_test", 6, 50, 2, "fake_test"),
              "snap_youtube": _planted("snap_youtube", 8, 50, 3, "snap_youtube", structural=True)}

    def load(name, attribute=None):
        base, _ = graphs[name]
        attribute = attribute or ("degree" if name == "snap_youtube" else "region")
        g = base.with_groups(base.meta["attrs"][attribute], list(base.meta["attr_names"][attribute]))
        g.name = name
        g.meta = dict(base.meta, attribute=attribute)
        return g
    monkeypatch.setattr(D, "available", lambda name: name in graphs)
    monkeypatch.setattr(D, "load", load)
    monkeypatch.setattr(D, "LOADABLE", D.LOADABLE + ("fake_val", "fake_test"))
    monkeypatch.setattr(D, "DEFAULT_ATTRIBUTE", dict(D.DEFAULT_ATTRIBUTE, fake_val="region", fake_test="region"))
    monkeypatch.setattr(D, "splits", lambda: {"train": ["pokec_z"], "val": ["fake_val"],
                                              "test": ["fake_test", "snap_youtube"]})
    monkeypatch.setattr(D, "ground_truth_communities", lambda name, which="top5000": graphs["snap_youtube"][1])
    monkeypatch.setitem(C.TIER, "fake_val", "small")
    monkeypatch.setitem(C.TIER, "fake_test", "small")
    import run_t1_t2
    monkeypatch.setitem(run_t1_t2.T1_ATTRS, "fake_test", ("region",))
    monkeypatch.setattr(run_t1_t2, "_GT", {})
    common = ["--ref-file", str(tmp_path / "refs.json"), "--n-sims", "100", "--preset", "8gb", "--max-rounds", "8",
              "--reference-runs", "2", "--time-limit", "120", "--R", "16", "--cert-R", "64",
              "--model-pattern", str(tmp_path / "none_s{seed}.pt")]

    def run(module, argv):
        monkeypatch.setattr(sys, "argv", [module.__name__ + ".py"] + argv)
        module.main()
    return tmp_path, common, run


def test_tune_t1_t2_summary_end_to_end(scripts):
    tmp, common, run = scripts
    import run_t1_t2
    import summarize_v2
    import tune_v2
    tuning = tmp / "tuning.json"
    run(tune_v2, ["--ks", "5", "--delta-grid", "0.0025", "0.01", "--phi0-grid", "0.3", "0.5", "--w3-states", "2",
                  "--out", str(tuning), "--runs-out", str(tmp / "tuning_runs.csv")] + common)
    t = json.loads(tuning.read_text())
    assert t["tuned"] and t["start"] == "greedy" and t["start_reason"] == "no constructor checkpoint"
    assert t["delta"] in (0.0025, 0.01) and t["phi0"] in (0.3, 0.5) and "train" in t["w3"]
    run(tune_v2, ["--ks", "5", "--steps", "w3", "--w3-states", "2", "--out", str(tuning),     # one step again:
                  "--runs-out", str(tmp / "tuning_runs.csv")] + common)                     # the rest is kept
    t2_ = json.loads(tuning.read_text())
    assert {a: t2_[a] for a in ("delta", "phi0", "start", "start_reason", "delta_scores")} == \
        {a: t[a] for a in ("delta", "phi0", "start", "start_reason", "delta_scores")}
    assert len(t2_["history"]) == 2 and t2_["steps"] == ["delta", "phi0", "start", "w3"]

    t1 = tmp / "t1.csv"
    argv = ["--test", "t1", "--datasets", "fake_test", "--ks", "5", "--rhos", "0.99", "--tuning", str(tuning),
            "--out", str(t1), "--bounds-out", str(tmp / "t1_bounds.csv"), "--pareto-out", str(tmp / "t1_pareto.csv"),
            "--methods", "Degree", "Greedy", "BSM", "MultiFW-MaxMin", "LocalSearch"] + common
    run(run_t1_t2, argv)
    d = pd.read_csv(t1)
    fim = d[d.method == "FIM-GAT v2"]
    # the method and the ablation rows that need no checkpoint: exhaustive proposer, v2-a, v2-b0 (4 configurations),
    # each with three rows: Theorem 2's deployment, the path rule's and the final search set
    assert len(fim) == 3 * 4 and set(fim.deployment) == {"certified", "certified-path", "final"}
    assert {"Degree", "Greedy", "BSM", "MultiFW-MaxMin", "LocalSearch"} <= set(d.method)
    for col in ("cert_passed", "cert_lcb_rel", "es_ratio", "disparity", "heldout_disparity", "delta", "phi0", "cert_R",
                "deploy_rule", "deploy_position", "deploy_disp_se"):
        assert col in d, col
    deployed = fim[fim.deployment != "final"]
    assert deployed.cert_passed.astype(bool).all()                               # certified deployment always passes
    assert set(deployed.deploy_rule) == {"min_ucb", "path"}
    assert (d.delta == t["delta"]).all() and (d.phi0 == t["phi0"]).all()
    assert len(pd.read_csv(tmp / "t1_bounds.csv")) == 1
    dp = pd.read_csv(tmp / "t1_pareto.csv")                                      # BSM's whole trade-off, every tau
    from fimgat.v2.external import TAUS
    assert len(dp) == len(TAUS) and set(dp.method) == {"BSM"} and dp.es_ratio.notna().all()
    assert {"tau", "b_ratio", "cert_passed_0.97", "cert_passed_0.99", "cert_passed_1"} <= set(dp.columns)
    run(run_t1_t2, argv)                                                         # resumed: nothing new
    assert len(pd.read_csv(t1)) == len(d) and len(pd.read_csv(tmp / "t1_pareto.csv")) == len(dp)

    t2 = tmp / "t2.csv"
    run(run_t1_t2, ["--test", "t2", "--ks", "5", "--t2-samples", "1", "--t2-size", "250", "--tuning", str(tuning),
                    "--out", str(t2)] + common)
    d2 = pd.read_csv(t2)
    assert set(d2.variant) == {"v2-b", "v2-a", "v2-b0"} and set(d2.rho) == {0.97, 0.99}
    assert set(d2.deployment) == {"certified", "certified-path", "final"}
    assert (d2.R == 16).all()                                                    # an explicit --R wins over T2's 64
    assert d2.comm_disparity.notna().all() and (d2.n_comm_eval >= 2).all()

    run(summarize_v2, ["--dir", str(tmp), "--figures"])
    md = (tmp / "summary_v2.md").read_text(encoding="utf-8")
    assert "## T1 (C1)" in md and "## T2 (C2)" in md and "underpowered" in md
    assert "Holds across the range" in md and "T2 pass (pre-registered rule)" in md
    assert "## Spread-disparity trade-off" in md and "## T3" in md
    pts = pd.read_csv(tmp / "summary_pareto.csv")                                # every point, with dominance flags
    assert {"method", "baseline", "bsm_sweep"} <= set(pts.source) and "dominated_by_method" in pts
    figs = summarize_v2.pareto_figures(pts, str(tmp / "figures_k5"), k=5)        # --figures draws k = 50 only
    assert [os.path.basename(p) for p in figs] == ["pareto_fake_test_region_k5.png"] and os.path.getsize(figs[0]) > 0

    import interpret_v2                                                          # the case study reruns exactly
    run(interpret_v2, ["--case", "fake_test", "region", "5", "0.99", "--t1", str(t1), "--tuning", str(tuning),
                       "--out-dir", str(tmp / "cases")] + common)
    case = tmp / "cases" / "fake_test_region_k5_rho0.99"
    groups = pd.read_csv(case / "groups.csv")
    assert {"region", "gender", "degree"} <= set(groups.attribute) and groups.reach_method.notna().all()
    assert (case / "communities.csv").exists()
    text = (case / "case.md").read_text(encoding="utf-8")
    assert "rerun reproduced the recorded set: True" in text and "not pre-registered" in text


def test_tuning_refuses_a_test_graph(scripts):
    _, common, run = scripts
    import tune_v2
    with pytest.raises(SystemExit):
        run(tune_v2, ["--dataset", "fake_test", "--dry-run"] + common)


def test_run_t4_logs_cached_and_query_stages(scripts):
    tmp, common, run = scripts
    import run_t4
    run(run_t4, ["--untuned", "--datasets", "fake_test", "--ks", "5", "--rhos", "0.99", "--methods", "Degree", "BSM",
                 "--out", str(tmp / "t4.csv"), "--runtime-out", str(tmp / "t4_runtime.csv"), "--no-warmup"] + common)
    r = pd.read_csv(tmp / "t4_runtime.csv")
    cached = set(r[r.scope == "cached"].stage)
    assert {"load", "reference", "rr_sample", "cert_sample", "cover_index"} <= cached
    q = r[r.scope == "query"]
    assert set(q[q.method == "FIM-GAT v2"].stage) == {"select", "improve", "verify"}
    d = pd.read_csv(tmp / "t4.csv")
    assert set(d.method) == {"FIM-GAT v2", "Degree", "BSM"} and (~d.tuned.astype(bool)).all()
