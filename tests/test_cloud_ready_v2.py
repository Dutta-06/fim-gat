"""Checks of the cloud-readiness changes of 2026-10-15 (docs/cloud_runbook.md), marker spec.

  RR sampling     the sample depends on the seed, not on the machine's thread count (diffusion.RR_CHUNKS)
  data sources    the old project's local copy when present, else the public download; gzipped Pokec dumps load
  parallel jobs   run_t1_t2 --job puts every output under results/v2/parts/<job>/; summarize_v2 merges the parts
                  and counts repeated queries
"""
import argparse
import gzip
import os
import sys

import numba
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

pytestmark = pytest.mark.spec

EXP = os.path.join(ROOT, "experiments")


def _exp(sub=""):
    path = os.path.join(EXP, sub) if sub else EXP
    if path not in sys.path:
        sys.path.insert(0, path)


def test_rr_sample_does_not_depend_on_the_thread_count(sbm_graph, monkeypatch):
    from fimgat import diffusion
    roots = np.arange(sbm_graph.n)
    ref = diffusion.sample_rr_sets(sbm_graph, roots, 4, seed=3)
    assert diffusion.RR_CHUNKS == 16
    for threads in (1, 4, 64):
        monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", threads)
        ptr, nodes = diffusion.sample_rr_sets(sbm_graph, roots, 4, seed=3)
        assert np.array_equal(ptr, ref[0]) and np.array_equal(nodes, ref[1])
    explicit = diffusion.sample_rr_sets(sbm_graph, roots, 4, seed=3, n_jobs=16)
    assert np.array_equal(explicit[1], ref[1])
    other = diffusion.sample_rr_sets(sbm_graph, roots, 4, seed=4)
    assert not np.array_equal(other[1], ref[1])                               # the seed still matters


def test_sources_fall_back_to_the_public_download(tmp_path):
    from fimgat.v2 import datasets as D
    here = tmp_path / "local.txt"
    here.write_text("x")
    assert D._local_or(str(here), "https://example.org/f.gz") == str(here)
    assert D._local_or(str(tmp_path / "absent.txt"), "https://example.org/f.gz") == "https://example.org/f.gz"
    for name in ("epinions", "twitter_ego", "pokec_t"):
        for src in D.SOURCES[name]:
            assert os.path.exists(src) or src.startswith("https://snap.stanford.edu/data/")
    assert not any(s.endswith(".nodes") for s in D.SOURCES["linkedin"])


def test_pokec_loader_reads_snaps_gzipped_dumps(tmp_path):
    _exp()
    import build_pokec
    profiles = "1\t1\t0\t1\tzilinsky kraj, zilina\t0\tnull\t25\n2\t1\t0\t0\tnitriansky kraj, nitra\t0\tnull\tnull\n"
    with gzip.open(tmp_path / "soc-pokec-profiles.txt.gz", "wt", encoding="utf-8") as f:
        f.write(profiles)
    with gzip.open(tmp_path / "soc-pokec-relationships.txt.gz", "wt", encoding="utf-8") as f:
        f.write("1\t2\n2\t1\n")
    assert build_pokec._raw(str(tmp_path), "soc-pokec-profiles.txt").endswith(".txt.gz")
    prof, rel = build_pokec.load_raw(str(tmp_path))
    assert list(prof.index) == [1, 2] and prof.loc[1, "region"] == "zilinsky kraj, zilina"
    assert prof.loc[1, "age"] == 25 and np.isnan(prof.loc[2, "age"]) and rel.tolist() == [[1, 2], [2, 1]]
    (tmp_path / "soc-pokec-profiles.txt").write_text(profiles)                 # a plain file wins
    assert build_pokec._raw(str(tmp_path), "soc-pokec-profiles.txt").endswith(".txt")


def test_job_outputs_go_under_parts():
    _exp("v2")
    import run_t1_t2
    from fimgat.v2.config import RESULTS_V2
    a = run_t1_t2.resolve_outputs(argparse.Namespace(job="t1-small", test="t1", out=None, bounds_out=None,
                                                     pareto_out=None, ref_file=None))
    base = os.path.join(RESULTS_V2, "parts", "t1-small")
    assert a.out == os.path.join(base, "t1.csv") and a.bounds_out == os.path.join(base, "t1_bounds.csv")
    assert a.pareto_out == os.path.join(base, "t1_pareto.csv") and a.ref_file == os.path.join(base, "references.json")
    b = run_t1_t2.resolve_outputs(argparse.Namespace(job="certsens", test="t1", out="x/t1_certsens.csv",
                                                     bounds_out=None, pareto_out=None, ref_file="r.json"))
    assert b.out == "x/t1_certsens.csv" and b.ref_file == "r.json"                # explicit paths win
    c = run_t1_t2.resolve_outputs(argparse.Namespace(job=None, test="t2", out=None, bounds_out=None,
                                                     pareto_out=None, ref_file=None))
    assert c.out == os.path.join(RESULTS_V2, "t2.csv") and c.ref_file == os.path.join(RESULTS_V2, "references.json")


def test_summary_merges_parts_and_counts_repeats(tmp_path):
    _exp("v2")
    import summarize_v2 as S
    row = {"dataset": "d", "attribute": "a", "graph": "d", "k": 5, "rho": 0.99, "method": "Greedy",
           "variant": "default", "status": "ok", "disparity": 0.1}
    pd.DataFrame([row]).to_csv(tmp_path / "t1.csv", index=False)
    for job, k in (("j1", 10), ("j2", 25)):
        os.makedirs(tmp_path / "parts" / job)
        pd.DataFrame([dict(row, k=k), dict(row, k=k, status="missing")]).to_csv(tmp_path / "parts" / job / "t1.csv",
                                                                                 index=False)
    d = S.load(str(tmp_path / "t1.csv"))
    assert sorted(d.k) == [5, 10, 25] and set(d.part) == {"", "j1", "j2"} and S.duplicates(d) == 0
    os.makedirs(tmp_path / "parts" / "j3")
    pd.DataFrame([dict(row, k=10)]).to_csv(tmp_path / "parts" / "j3" / "t1.csv", index=False)  # j1's case again
    assert S.duplicates(S.load(str(tmp_path / "t1.csv"))) == 1
    assert S.load(str(tmp_path / "t2.csv")) is None
