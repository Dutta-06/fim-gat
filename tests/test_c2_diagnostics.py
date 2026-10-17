"""The C2 diagnostics script (experiments/v2/c2_diagnostics.py; docs/design_v2.md section 22) end to end on a synthetic
com-Youtube stand-in, marker spec. The key check: its control runs reproduce T2's deployed sets exactly, so the room
runs differ from T2 only in the cap kappa."""
import json

import numpy as np
import pandas as pd
import pytest

from test_scripts_v2 import scripts  # noqa: F401  (the fixture: patched datasets, outputs in tmp_path)

pytestmark = pytest.mark.spec


def test_c2_diagnostics_reproduce_t2_and_write_every_table(scripts):  # noqa: F811
    tmp, common, run = scripts
    import c2_diagnostics as C
    import run_t1_t2
    from fimgat.v2 import config as CFG
    tuning = tmp / "tuning.json"
    tuning.write_text(json.dumps({"delta": CFG.DELTA_MARGIN, "phi0": CFG.MAX_CONDUCTANCE, "start": "greedy",
                                  "tuned": True}))
    t2 = tmp / "t2.csv"
    t2_args = ["--ks", "5", "--t2-samples", "1", "--t2-size", "250", "--tuning", str(tuning)] + common
    run(run_t1_t2, ["--test", "t2", "--rhos", "0.97", "--out", str(t2)] + t2_args)
    out = tmp / "c2diag"
    run(C, ["--rhos", "0.97", "--t2-csv", str(t2), "--out-dir", str(out), "--scan-max", "400"] + t2_args)
    room = pd.read_csv(out / "room.csv")
    assert set(room.variant) == set(C.LABELS) and len(room) == 2 * len(C.LABELS)
    dep = room[room.deployment == "certified-path"].set_index("variant")
    assert bool(dep.loc["a", "matches_t2"]) and bool(dep.loc["b-1se", "matches_t2"])   # T2 reproduced exactly
    d_a = float(dep.loc["a", "disp_a"])
    for f in C.FRACTIONS:
        lab = f"b-{int(round(f * 100))}pct"
        assert dep.loc[lab, "kappa"] == pytest.approx(d_a * (1 + f))
    assert (room.cert_passed.astype(str).isin(["True", "1", "1.0"]) | (room.deployment == "final")).all()
    price = pd.read_csv(out / "price.csv")
    assert len(price) == 1 and price.n_feasible_scanned.iloc[0] > 0
    budgets = sorted(["se", "1pct", "5pct", "20pct"], key=lambda b: float(price[f"budget_{b}"].iloc[0]))
    for nm in ("F", "H"):                                 # at a labelled local optimum no swap lowers D (cost >= 0)
        v = price[f"price_{nm}"].iloc[0]
        assert np.isnan(v) or v >= -1e-9
        best = [float(price[f"best_dD{nm}_within_{b}"].iloc[0]) for b in budgets]
        assert all(x >= y - 1e-12 for x, y in zip(best, best[1:]))   # a larger budget buys at least as much
    cov = pd.read_csv(out / "coverage.csv")
    assert (cov.kind == "gt_conductance").sum() == 1
    tr = cov[cov.kind == "transfer"]
    assert len(tr) == len(C.LABELS) - 1 and tr.holds.astype(str).isin(["True", "1", "1.0"]).all()   # Lemma W1.1
    run(C, ["--rhos", "0.97", "--t2-csv", str(t2), "--out-dir", str(out)] + t2_args)             # resumed: nothing
    assert len(pd.read_csv(out / "room.csv")) == len(room)
    run(C, ["--summarize", "--out-dir", str(out)] + common)
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Room" in md and "## Coverage" in md and "## Price" in md and "EXPLORATORY" in md
