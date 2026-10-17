"""Test C2b's scripts (experiments/v2/run_c2b.py; docs/design_v2.md section 25), marker spec: the run on a synthetic
stand-in for its graph (resumable, the secondary community sets scored), and the pre-registered pass rules."""
import json

import numpy as np
import pandas as pd
import pytest

from test_scripts_v2 import scripts  # noqa: F401  (the fixture: patched datasets, outputs in tmp_path)

pytestmark = pytest.mark.spec


def test_c2b_runs_and_writes_the_preregistered_verdicts(scripts, monkeypatch):  # noqa: F811
    tmp, common, run = scripts
    import fimgat.v2.datasets as D
    import run_c2b as B
    from fimgat.v2 import config as CFG
    load = D.load

    def alias(name, attribute=None):                        # the synthetic com-Youtube stand-in, renamed
        if name != CFG.C2B_DATASET:
            return load(name, attribute)
        g = load("snap_youtube", attribute)
        g.meta = dict(g.meta, dataset=name)
        return g
    monkeypatch.setattr(D, "load", alias)
    tuning = tmp / "tuning.json"
    tuning.write_text(json.dumps({"delta": CFG.DELTA_MARGIN, "phi0": CFG.MAX_CONDUCTANCE, "start": "greedy",
                                  "tuned": True}))
    out = tmp / "c2b"
    argv = ["--ks", "5", "--t2-samples", "1", "--t2-size", "250", "--rhos", "0.97", "--tuning", str(tuning),
            "--out-dir", str(out), "--scan-max", "400"] + common
    run(B, argv)
    rows = pd.read_csv(out / "c2b.csv")
    assert set(rows.variant) == {"a", "b-1se", f"b-{int(round(CFG.C2B_FRACTION * 100))}pct"} == set(B.LABELS)
    assert len(rows) == 2 * len(B.LABELS) and rows.graph.str.startswith(CFG.C2B_DATASET).all()
    assert (rows.test == "c2b").all() and not rows.exploratory.astype(str).isin(["True", "1", "1.0"]).any()
    dep = rows[rows.deployment == "certified-path"]
    assert dep.comm_disparity_all.notna().all() and (dep.n_comm_all >= dep.n_comm_visible).all()
    assert rows.matches_t2.isna().all()                       # no T2 rows exist for this graph
    cov = pd.read_csv(out / "coverage.csv")
    assert {"gt_conductance", "gt_conductance_all"} <= set(cov.kind)
    run(B, argv)                                              # resumed: nothing new
    assert len(pd.read_csv(out / "c2b.csv")) == len(rows)
    run(B, ["--summarize", "--out-dir", str(out)] + common)
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "PRE-REGISTERED" in md and "**H1" in md and "**H2" in md and "## Descriptive" in md


def _rows(d_hidden, d_labelled, n_samples=6, ks=(25, 50, 100, 150, 200), seed=0):
    """Synthetic certified-path rows: per case, (a) and the two runs with the given differences from (a)."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_samples):
        for k in ks:
            base = 0.4 + 0.05 * rng.random()
            for v in ("a", "b-1se", "b-5pct"):
                dh = 0.0 if v == "a" else d_hidden[v] * (1 + rng.random())
                dl = 0.0 if v == "a" else d_labelled[v] * (1 + rng.random())
                out.append({"graph": f"g_s{s}", "k": k, "rho": 0.97, "deployment": "certified-path", "variant": v,
                            "comm_disparity": base + dh, "disparity": 0.3 + dl})
    return pd.DataFrame(out)


def test_c2b_pass_rules():
    import run_c2b as B
    v = B.verdicts(_rows({"b-1se": 0.0, "b-5pct": -0.02}, {"b-1se": 0.0, "b-5pct": 0.003}))
    assert not v["H1"]["pass"] and v["H1"]["why"] == "no significant decrease"   # identical sets: no test
    assert v["H2"]["pass"] and v["H2"]["median"] < 0 < v["H2"]["labelled_median"]   # H2 may pay its stated price
    v = B.verdicts(_rows({"b-1se": -0.01, "b-5pct": -0.02}, {"b-1se": 0.002, "b-5pct": 0.0}))
    assert not v["H1"]["pass"] and v["H1"]["why"] == "labelled disparity rose significantly"   # T2's rule
    v = B.verdicts(_rows({"b-1se": -0.01, "b-5pct": -0.02}, {"b-1se": 0.0, "b-5pct": 0.0}, n_samples=3))
    assert not v["H1"]["pass"] and v["H1"]["why"] == "underpowered"            # 15 < T2_MIN_PAIRS cases
    v = B.verdicts(_rows({"b-1se": -0.01, "b-5pct": 0.01}, {"b-1se": 0.0, "b-5pct": 0.0}))
    assert v["H1"]["pass"] and not v["H2"]["pass"]                              # a rise is never a pass
