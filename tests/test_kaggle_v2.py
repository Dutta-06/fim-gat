"""The Kaggle runner (kaggle/run_job.py, kaggle/import_outputs.py, kaggle/make_bundle.py; docs/design_v2.md section 23),
marker spec. Every job command must name a real script, real options, real datasets and attributes; outputs must
round-trip through export, resume and import without losing rows or leaving results/ and data/cache/."""
import io
import json
import os
import re
import sys
import zipfile

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "v2"))
import import_outputs as IO                                               # noqa: E402
import make_bundle as MB                                                  # noqa: E402
import run_job as RJ                                                      # noqa: E402
from test_prerun_fixes_v2 import t3_env  # noqa: E402,F401  (the fixture: synthetic train/val/large graphs, tmp outputs)

pytestmark = pytest.mark.spec


def _options(script):
    with open(os.path.join(ROOT, script), encoding="utf-8") as f:
        src = f.read()
    if "P.add_pipeline_args(ap)" in src:
        with open(os.path.join(ROOT, "experiments", "v2", "pipeline_v2.py"), encoding="utf-8") as f:
            src += f.read()
    return set(re.findall(r'"(--[a-z0-9-]+)"', src))


def _values(cmd, opt):
    if opt not in cmd:
        return []
    out = []
    for tok in cmd[cmd.index(opt) + 1:]:
        if tok.startswith("--"):
            break
        out.append(tok)
    return out


def test_every_job_command_names_real_scripts_options_datasets_and_methods():
    from common_v2 import PRESETS
    from fimgat.v2 import datasets as D
    from fimgat.v2.baselines_v2 import SUITE
    from interpret_v2 import CASES
    from run_t1_t2 import T1_ATTRS
    n = 0
    for job in RJ.JOBS:
        for c in RJ.commands(job):
            if isinstance(c, dict) and "call" in c:
                assert c["call"] == "merge_refs" and c["dst"].endswith("references.json")
                continue
            cmd = c["cmd"] if isinstance(c, dict) else c
            assert os.path.exists(os.path.join(ROOT, cmd[0])), (job, cmd[0])
            opts = _options(cmd[0])
            for tok in cmd[1:]:
                if tok.startswith("--"):
                    assert tok in opts, (job, cmd[0], tok)
            assert set(_values(cmd, "--preset")) <= set(PRESETS)
            assert set(_values(cmd, "--methods")) <= set(SUITE), (job, cmd)
            datasets = _values(cmd, "--datasets")
            assert set(datasets) <= set(D.NAMES), (job, cmd)
            for a in _values(cmd, "--attributes"):
                assert all(a in T1_ATTRS[d] for d in datasets), (job, cmd)
            if cmd[0].endswith("interpret_v2.py"):
                d, a, k, rho = _values(cmd, "--case")
                assert (d, a, int(k), float(rho)) in CASES
            if cmd[0].endswith(("run_t1_t2.py", "run_t3.py", "run_t4.py", "tune_v2.py", "interpret_v2.py")):
                assert _values(cmd, "--preset") == ["8gb"], (job, cmd)       # design section 23
            n += 1
    assert n > 20
    # the CPU jobs never see the constructor; the timing jobs use one lane
    for job, spec in RJ.JOBS.items():
        if not spec["gpu"]:
            assert "G1b-train" not in spec["needs"]
        if spec.get("timing"):
            assert all(len(ph["lanes"]) == 1 for ph in spec["phases"])
    # every pre-registered case without a cloud case folder is run by some job
    run = {tuple(_values(c["cmd"], "--case")[:2]) for j in RJ.JOBS for c in RJ.commands(j)
           if isinstance(c, dict) and "cmd" in c}
    assert {(d, a) for d, a, _, _ in CASES if d == "pokec_full"} <= run


def test_dry_run_of_every_job(capsys):
    for job in RJ.JOBS:
        assert RJ.main(["--job", job, "--dry-run"]) == 0
    assert RJ.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "train_constructor.py" in out and "order:" in out


def test_merge_rules_for_row_stores_json_stores_and_other_files():
    head = "dataset,k,method,status,time_s\n"
    a = (head + "g,25,A,ok,1.0\ng,50,A,ok,2.0\n").encode()
    sup = (head + "g,25,A,ok,1.0\ng,50,A,ok,2.0\ng,100,A,ok,3.0\n").encode()
    sub = (head + "g,25,A,ok,9.9\n").encode()
    other = (head + "g,25,A,ok,7.7\ng,200,A,ok,4.0\n").encode()
    assert RJ.merge_bytes("t1.csv", None, a)[0] == "new"
    assert RJ.merge_bytes("t1.csv", a, a)[0] == "same"
    act, out, _ = RJ.merge_bytes("t1.csv", a, sup)
    assert act == "newer" and out == sup
    assert RJ.merge_bytes("t1.csv", a, sub)[:2] == ("older", None)
    act, out, note = RJ.merge_bytes("t1.csv", a, other)
    got = pd.read_csv(io.BytesIO(out))
    assert act == "united" and len(got) == 3 and got.time_s.tolist() == [1.0, 2.0, 4.0]   # local rows win
    log_a, log_b = b"epoch,loss\n1,0.5\n", b"epoch,loss\n2,0.4\n"          # not a row store: whole-row keys
    act, out, _ = RJ.merge_bytes("train_log_gpu0.csv", log_a, log_b)
    assert act == "united" and len(pd.read_csv(io.BytesIO(out))) == 2
    r1, r2 = json.dumps({"x": 1}).encode(), json.dumps({"x": 1, "y": 2}).encode()
    assert RJ.merge_bytes("references.json", r1, r2)[0] == "newer"
    assert RJ.merge_bytes("references.json", r2, r1)[:2] == ("older", None)
    act, out, _ = RJ.merge_bytes("references.json", json.dumps({"x": 1, "z": 3}).encode(), r2)
    assert act == "united" and json.loads(out) == {"x": 1, "y": 2, "z": 3}
    assert RJ.merge_bytes("constructor_s0.pt", b"old", b"new") == ("replaced", b"new", "")


def test_safe_paths_only():
    assert RJ.safe_rel("results/v2/parts/t1-small/t1.csv") and RJ.safe_rel("data/cache/x_louvain.npy")
    for bad in ("../results/x", "results/../fimgat/x.py", "fimgat/v2/config.py", "/etc/passwd", "C:/x/results/a",
                "results/v2/quick/models/a.pt", "results/v2/t1.csv.123.tmp", "data/v2/instagram.npz", ""):
        assert not RJ.safe_rel(bad), bad


def _tree(root, files):
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(text if isinstance(text, bytes) else text.encode())


def test_export_resume_and_import_round_trip(tmp_path):
    head = "dataset,k,method,status\n"
    src = str(tmp_path / "src")
    _tree(src, {"results/v2/parts/t1-x/t1.csv": head + "g,25,A,ok\n", "results/v2/tuning.json": '{"start": "greedy"}',
                "fimgat/code.py": "x = 1\n"})
    base = RJ.snapshot(src)
    _tree(src, {"results/v2/parts/t1-x/t1.csv": head + "g,25,A,ok\ng,50,A,ok\n",
                "results/v2/tuning.json": '{"start": "learned"}', "results/v2/models/constructor_s0.pt": b"\x00w",
                "results/v2/quick/models/constructor_s0.pt": b"smoke", "results/v2/x.csv.7.tmp": "partial",
                "fimgat/code.py": "x = 2\n"})
    out = str(tmp_path / "out")
    files = RJ.Exporter("J1", out, base, {"runner": RJ.RUNNER, "job": "J1", "commands": []}, root=src)()
    assert files == ["results/v2/models/constructor_s0.pt", "results/v2/parts/t1-x/t1.csv", "results/v2/tuning.json"]
    with zipfile.ZipFile(os.path.join(out, "outputs_J1.zip")) as zf:
        assert sorted(zf.namelist()) == sorted([f"J1/{f}" for f in files] + ["J1/manifest.json"])
    # a job that needs J1 resumes it; one that does not ignores it
    dst = str(tmp_path / "dst")
    _tree(dst, {"results/v2/parts/t1-x/t1.csv": head + "g,25,A,ok\n", "results/v2/tuning.json": '{"start": "greedy"}'})
    RJ.resume(out, "J2", ["J1"], root=dst)
    assert len(pd.read_csv(os.path.join(dst, "results/v2/parts/t1-x/t1.csv"))) == 2
    assert json.load(open(os.path.join(dst, "results/v2/tuning.json")))["start"] == "learned"
    assert open(os.path.join(dst, "results/v2/models/constructor_s0.pt"), "rb").read() == b"\x00w"
    other = str(tmp_path / "other")
    os.makedirs(other)
    RJ.resume(out, "J3", [], root=other)
    assert not os.path.exists(os.path.join(other, "results"))
    # an output unpacked by Kaggle (a folder with manifest.json) is found too, and counted once beside its zip
    with zipfile.ZipFile(os.path.join(out, "outputs_J1.zip")) as zf:
        zf.extractall(str(tmp_path / "unpacked"))
    assert len(RJ.find_outputs(str(tmp_path / "unpacked"))) == 1
    # the laptop importer: a report first, then --apply; files outside results/ and data/cache/ are refused
    lap = str(tmp_path / "laptop")
    _tree(lap, {"results/v2/parts/t1-x/t1.csv": head + "g,25,A,ok\ng,150,A,ok\n"})
    evil = os.path.join(out, "outputs_J9.zip")
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("J9/manifest.json", json.dumps({"runner": RJ.RUNNER, "job": "J9", "updated": "2099",
                                                    "files": ["../evil.txt", "fimgat/code.py", "results/v2/ok.txt"]}))
        zf.writestr("J9/../evil.txt", "x")
        zf.writestr("J9/fimgat/code.py", "pwned")
        zf.writestr("J9/results/v2/ok.txt", "fine")
    IO.main([out, "--root", lap])
    assert not os.path.exists(os.path.join(lap, "results/v2/tuning.json"))           # report only
    IO.main([out, "--root", lap, "--apply"])
    rows = pd.read_csv(os.path.join(lap, "results/v2/parts/t1-x/t1.csv"))
    assert sorted(rows.k.tolist()) == [25, 50, 150]                                   # united: no row lost
    assert open(os.path.join(lap, "results/v2/ok.txt")).read() == "fine"
    assert not os.path.exists(os.path.join(lap, "fimgat")) and not os.path.exists(os.path.join(str(tmp_path),
                                                                                              "evil.txt"))


def test_lane_markers_conditions_and_deadline(tmp_path):
    root = str(tmp_path)
    _tree(root, {"results/v2/tuning.json": '{"start": "greedy"}'})
    seen = []
    ctx = {"root": root, "deadline": 1e18, "logdir": root, "markers": os.path.join(root, "results/done"),
           "record": seen.append, "export": lambda: []}
    lane = {"_index": 0, "cmds": [{"cmd": ["-c", "print('one')"], "once": "a"},
                                  {"cmd": ["-c", "print('two')"], "when": "start-learned"},
                                  ["-c", "import sys; sys.exit(3)"], ["-c", "print('never')"]]}
    RJ.run_lane(lane, None, ctx)
    assert [r["status"][:8] for r in seen] == ["ok", "not need", "exit 3"]           # a failure ends the lane
    seen.clear()
    RJ.run_lane(dict(lane, keep_going=True), None, ctx)
    assert seen[0]["status"] == "done in an earlier session" and seen[-1]["status"] == "ok"
    seen.clear()
    RJ.run_lane(lane, None, dict(ctx, deadline=0))
    assert seen[0]["status"].startswith("done") and seen[2]["status"].startswith("not started")


def test_merge_refs_adds_missing_entries_only(tmp_path):
    root = str(tmp_path)
    _tree(root, {"results/v2/parts/t1-a/references.json": json.dumps({"g|x|R=8": {"refs": 1}}),
                 "results/v2/parts/t3/references.json": json.dumps({"g|x|R=8": {"refs": 2}})})
    assert RJ.merge_refs("results/v2/parts/t3/references.json", root) == 0              # the entry present wins
    _tree(root, {"results/v2/parts/t1-b/references.json": json.dumps({"h|y|R=8": {"refs": 3}})})
    assert RJ.merge_refs("results/v2/parts/t3/references.json", root) == 1
    got = json.load(open(os.path.join(root, "results/v2/parts/t3/references.json")))
    assert got == {"g|x|R=8": {"refs": 2}, "h|y|R=8": {"refs": 3}}


def test_notebook_and_bundle_rules():
    with open(os.path.join(ROOT, "kaggle", "fimgat_kaggle.ipynb"), encoding="utf-8") as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4 and all(c["cell_type"] in ("markdown", "code") for c in nb["cells"])
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        compile(src, "<cell>", "exec")                                                 # every cell parses
    settings = code[0]
    assert re.search(r'^JOB = "G1b-train"', settings, re.M)
    assert set(re.findall(r"^#\s+(\S+)\s+(?:GPU|CPU)", settings, re.M)) == set(RJ.JOBS)
    assert any("kaggle/run_job.py" in s and "/kaggle/working" in s for s in code)
    for rel in ("data/old/soc-linkedin.nodes", "data/v2/linkedin.npz", "data/v2/raw/x.gz",
                "soc-pokec-profiles.txt"):
        assert MB.forbidden(rel), rel
    for rel in MB.EXTRAS + ("data/v2/instagram.npz", "results/v2/tuning.json"):
        assert not MB.forbidden(rel), rel


def test_g1_chain_constructor_then_start_step_then_learned_t1_rows(t3_env):
    """Job G1's first execution path on synthetic stand-ins: train_constructor.py (never run before 2026-10-17)
    writes checkpoints that tune_v2.py's start step and run_t1_t2.py's learned-start rows load."""
    tmp, tuning, run = t3_env
    import run_t1_t2
    import train_constructor
    import tune_v2
    models = tmp / "models"
    run(train_constructor, ["--seeds", "0", "--noadv-seeds", "0", "--ks", "5", "--epochs", "1", "--steps-per-epoch",
                            "2", "--eval-every", "1", "--R", "8", "--attributes", "region", "gender", "degree",
                            "--val-attributes", "region", "gender", "--models-dir", str(models),
                            "--log", str(tmp / "train_log.csv"), "--preset", "8gb"])
    assert (models / "constructor_s0.pt").exists() and (models / "constructor_noadv_s0.pt").exists()
    log = pd.read_csv(tmp / "train_log.csv")
    assert len(log) >= 2 and log.loss.notna().all()
    run(train_constructor, ["--seeds", "0", "--noadv-seeds", "0", "--models-dir", str(models),
                            "--log", str(tmp / "train_log.csv")])                  # resumed: nothing to train
    assert len(pd.read_csv(tmp / "train_log.csv")) == len(log)
    common = ["--ref-file", str(tmp / "refs.json"), "--n-sims", "100", "--preset", "8gb", "--R", "8",
              "--reference-runs", "2", "--max-rounds", "8", "--time-limit", "120", "--cert-R", "32",
              "--train-seeds", "0", "--model-pattern", str(models / "constructor_s{seed}.pt")]
    run(tune_v2, ["--steps", "start", "--ks", "5", "--out", str(tuning), "--runs-out", str(tmp / "runs.csv")] + common)
    got = json.loads(tuning.read_text())
    assert got["start"] in ("greedy", "learned") and got["start_reason"].startswith("constructor mean")
    from fimgat.v2 import config as CFG
    assert got["delta"] == CFG.DELTA_MARGIN and got["w3"]["train"]                   # earlier steps kept
    run(run_t1_t2, ["--test", "t1", "--datasets", "fake_large", "--attributes", "region", "--ks", "5", "--rhos",
                    "0.99", "--configs", "method", "ablation", "--methods", "--no-bounds", "--no-pareto",
                    "--tuning", str(tuning), "--out", str(tmp / "t1.csv")] + common)
    t1 = pd.read_csv(tmp / "t1.csv")
    learned = t1[(t1.start == "learned") & (t1.deployment == "certified")]
    assert len(learned) >= 1 and (learned.status == "ok").all() and set(learned.train_seed) == {0}


def test_progress_lines_reach_the_runner(tmp_path, monkeypatch):
    """common_v2.progress in a log file writes plain lines (no carriage returns) that run_job.last_progress relays,
    reading only what the current command wrote."""
    import common_v2
    log = tmp_path / "lane0.log"
    with open(log, "w", encoding="utf-8") as f:
        f.write("an earlier command\nT1 x/y: 100%|##########| 3/3 [00:01<00:00, 2.00query/s]\n")
        offset = f.tell()
        monkeypatch.setattr(sys, "stderr", f)
        monkeypatch.setattr(common_v2, "PROGRESS_LOG_SECONDS", 0.0)
        assert RJ.last_progress(str(log), offset) == ""
        bar = common_v2.progress(4, "T1 g/a")
        for _ in range(4):
            print("a result row", file=f, flush=True)
            bar.update(1)
        bar.close()
    text = log.read_text(encoding="utf-8")
    assert "\r" not in text
    got = RJ.last_progress(str(log), offset)
    assert got.startswith("T1 g/a") and "4/4 [" in got
    assert RJ.last_progress(str(tmp_path / "missing.log")) == ""
