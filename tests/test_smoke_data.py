"""Smoke tests for data: what exists today, and the integrity rules every v2 dataset must meet.

Datasets that have not been built yet are skipped with the path they are expected at.
The expected sizes come from the dataset sources (see docs/design_v2.md section 8).
"""
import os

import numpy as np
import pytest

from conftest import DATA, OLD_PROJECT, data_path, skip_unless_exists

pytestmark = [pytest.mark.smoke, pytest.mark.data]

# name -> (expected nodes, expected edges or None, edge semantics, group attributes)
# edges are counted as the source reports them (undirected pairs, or directed arcs).
V2_DATASETS = {
    "rice_facebook":     (1_205, 42_443, "undirected", ["college", "age", "major"]),
    "twitter_political": (3_753, 6_993, "undirected", ["leaning"]),
    "pokec_z":           (67_435, None, "directed", ["region", "gender", "age"]),
    "pokec_n":           (66_082, None, "directed", ["region", "gender", "age"]),
    "pokec_full":        (1_632_803, None, "directed", ["region", "gender", "age"]),
    "epinions":          (75_879, 508_837, "directed", []),
    "twitter_ego":       (81_306, 1_768_149, "directed", []),
    "instagram":         (553_628, None, "undirected", ["gender"]),
    "snap_youtube":      (1_134_890, 2_987_624, "undirected", []),
    "snap_amazon":       (334_863, 925_872, "undirected", []),       # C2b's graph (design section 25)
    "snap_dblp":         (317_080, 1_049_866, "undirected", []),     # checked for C2b, not used (section 25)
    "linkedin":          (None, None, "undirected", []),
}


# ------------------------------------------------------------------ existing v1 data

@pytest.mark.parametrize("name,n", [("pokec_z", 67_435), ("pokec_n", 66_082), ("pokec_full", 1_632_803)])
def test_existing_pokec_graphs_load_and_are_consistent(name, n):
    path = data_path("pokec", f"{name}.npz")
    skip_unless_exists(path, "Pokec build")
    z = np.load(path, allow_pickle=True)
    assert int(z["n"]) == n
    e = z["edges"]
    assert e.ndim == 2 and e.shape[1] == 2 and e.min() >= 0 and e.max() < n
    for a in ("region", "gender", "age"):
        assert len(z[a]) == n and z[a].min() >= 0


def test_old_project_raw_files_present_and_untouched():
    """v2 reads Epinions / Twitter / LinkedIn raw files from the old project; it must never write there."""
    files = [os.path.join(OLD_PROJECT, "data", "soc-Epinions1.txt"),
             os.path.join(OLD_PROJECT, "data", "twitter_combined.txt"),
             os.path.join(OLD_PROJECT, "soc-linkedin.edges")]
    for f in files:
        skip_unless_exists(f, "old-project raw file")
    before = {f: os.stat(f).st_mtime for f in files}
    with open(files[0], encoding="utf-8", errors="ignore") as fh:
        head = [next(fh) for _ in range(4)]
    assert any("Nodes: 75879" in h for h in head)                   # SNAP header of soc-Epinions1
    assert {f: os.stat(f).st_mtime for f in files} == before


# ------------------------------------------------------------------ v2 dataset integrity rules

@pytest.mark.parametrize("name", sorted(V2_DATASETS))
def test_v2_dataset_integrity(name):
    """Runs once fimgat.v2.datasets exists and the dataset is built."""
    ds = pytest.importorskip("fimgat.v2.datasets", reason="v2 loaders not implemented yet")
    n_exp, m_exp, kind, attrs = V2_DATASETS[name]
    if not ds.available(name):
        pytest.skip(f"{name} not built yet")
    g = ds.load(name)
    if n_exp is not None:
        assert g.n == n_exp
    if m_exp is not None:
        arcs = g.m // 2 if kind == "undirected" else g.m
        assert abs(arcs - m_exp) <= 0.01 * m_exp                  # tolerance for self-loops / duplicates removed
    # weighted cascade everywhere (agreed 2026-09-27)
    s = np.bincount(g.dst, weights=g.prob, minlength=g.n)
    assert np.allclose(s[g.in_degree > 0], 1.0, atol=1e-6)
    assert np.all(g.src != g.dst)
    for a in attrs:
        assert a in g.meta["attrs"] and len(g.meta["attrs"][a]) == g.n
