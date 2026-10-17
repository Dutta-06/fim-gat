"""v2 datasets (docs/design_v2.md section 8): registry, splits, loaders, builders, BFS samples, degree groups.

Every graph loads as a v1 FIMGraph under weighted cascade, p(u->v) = 1/in_degree(v), with arcs in the direction
influence flows. meta = {"dataset", "attribute", "attrs", "attr_names", "tier", "undirected"}, and every graph
carries attrs["degree"]: degree groups of the *full* graph (samples inherit them, design section 8).

Built files (build_<name>, from the raw files in SOURCES): data/v2/<name>.npz with n; arcs, every directed
influence arc src -> dst (sorted, no self-loops or duplicates, both directions for undirected data; weighted
cascade is applied at load); undirected; orig_id, the raw id of each node; and one int64 array per attribute plus
<attr>_names. Ground-truth communities go in data/v2/<name>_cmty.npz (ptr, members = top 5,000; all_ptr,
all_members = all) for com-Youtube, and for com-Amazon and com-DBLP, the graphs of the pre-registered follow-up C2b
(design section 25), which are loadable but outside the T1 registry (NAMES). Pokec loads through fimgat.data.load_pokec from data/pokec/ (pokec_z, pokec_n and
pokec_full built by experiments/build_pokec.py, pokec_t by experiments/v2/build_datasets.py).

Orientation and node set per dataset. There is no component filtering: the reference sizes (design section 8)
are the full source counts, and the source papers use whole graphs (com-Youtube is already its largest component;
the Pokec districts keep v1's largest weakly connected component).
  rice_facebook      undirected friendships, as the fair_influen_max code reads them (nx.Graph). Nodes: users in
                     attr.txt (id, college, age, major); edges with an unattributed end are dropped.
  twitter_political  undirected, as CrossWalk reads it (both directions added). Nodes: users in the .attr file;
                     links whose ends lack an attribute are skipped, as in CrossWalk. Leaning keeps the raw codes.
  epinions           SNAP "a b" means a trusts b, so influence flows b -> a: arcs are reversed.
  twitter_ego        SNAP ego-Twitter "a b" means a follows b (readme-Ego.txt), so influence flows b -> a: reversed.
  instagram          undirected, as S3D (fairness-sim-ot) treats it. Nodes: users in graph_gender_anon.csv.
  snap_youtube       undirected friendships; communities are user-created groups (overlapping, most users in none).
  snap_amazon        undirected co-purchases; communities are product categories (SNAP's deduplicated "all" file).
  snap_dblp          undirected co-authorships; communities are publication venues.
  linkedin           undirected pairs of soc-linkedin.edges. Nodes: ids that occur in an edge. soc-linkedin.nodes
                     holds real profile handles and is never opened.
  pokec_*            SNAP u -> v as listed, as in v1.
"""
import gzip
import os

import numpy as np
import pandas as pd
from numba import njit

from ..data import ROOT as DATA_ROOT
from ..data import load_pokec
from ..graph import FIMGraph, from_edges
from .config import DATA_V2, OLD_PROJECT, RAW_V2, TIER

NAMES = ("rice_facebook", "twitter_political", "pokec_z", "pokec_n", "pokec_full", "epinions", "twitter_ego",
         "instagram", "snap_youtube", "linkedin")
POKEC = ("pokec_z", "pokec_n", "pokec_t", "pokec_full")
POKEC_ATTRS = ("region", "gender", "age")
C2B = ("snap_amazon", "snap_dblp")                  # test graphs of the follow-up C2b only (design section 25)
GT_GRAPHS = ("snap_youtube",) + C2B                   # graphs with ground-truth communities (evaluation only)
LOADABLE = NAMES + ("pokec_t",) + C2B                 # pokec_t (validation) and C2B are loadable, not in NAMES
POKEC_T_DISTRICTS = ("trenciansky kraj, trencin", "trenciansky kraj, povazska bystrica")

DEFAULT_ATTRIBUTE = {"rice_facebook": "college", "twitter_political": "leaning", "instagram": "gender",
                     **{p: "region" for p in POKEC},
                     **{s: "degree" for s in ("epinions", "twitter_ego", "snap_youtube", "linkedin") + C2B}}

_OLD = os.path.join(OLD_PROJECT, "data")
_GH = "https://raw.githubusercontent.com/"
_SNAP = "https://snap.stanford.edu/data/bigdata/communities/"
_SNAP_DATA = "https://snap.stanford.edu/data/"


def _local_or(path, url):
    """The old project's local copy when it exists (the development laptop), else the same file's public source
    (a cloud VM or any other machine; 2026-10-15). The local copies were downloaded from those sources, and
    build_datasets.py checks every build against the sizes the sources report."""
    return path if os.path.exists(path) else url


_POKEC_RAW = (_local_or(os.path.join(_OLD, "soc-pokec-profiles.txt"), _SNAP_DATA + "soc-pokec-profiles.txt.gz"),
              _local_or(os.path.join(_OLD, "soc-pokec-relationships.txt"),
                        _SNAP_DATA + "soc-pokec-relationships.txt.gz"))

# name -> raw files (URLs are downloaded to data/v2/raw/<name>/; local paths are read in place, never written)
SOURCES = {
    "rice_facebook": [_GH + "redfox-codder/fair_influen_max/master/data/edges.txt",
                      _GH + "redfox-codder/fair_influen_max/master/data/attr.txt"],
    "twitter_political": [_GH + "ahmadkhajehnejad/CrossWalk/master/data/twitter/sample_4000.links",
                          _GH + "ahmadkhajehnejad/CrossWalk/master/data/twitter/sample_4000.attr"],
    **{p: list(_POKEC_RAW) for p in POKEC},
    "epinions": [_local_or(os.path.join(_OLD, "soc-Epinions1.txt"), _SNAP_DATA + "soc-Epinions1.txt.gz")],
    "twitter_ego": [_local_or(os.path.join(_OLD, "twitter_combined.txt"), _SNAP_DATA + "twitter_combined.txt.gz")],
    "instagram": [_GH + "nicolaslanzetti/fairness-sim-ot/main/data/instagram_ds/raw/graph_edges_anon.csv",
                  _GH + "nicolaslanzetti/fairness-sim-ot/main/data/instagram_ds/raw/graph_gender_anon.csv"],
    "snap_youtube": [_SNAP + "com-youtube.ungraph.txt.gz", _SNAP + "com-youtube.top5000.cmty.txt.gz",
                     _SNAP + "com-youtube.all.cmty.txt.gz"],
    "snap_amazon": [_SNAP + "com-amazon.ungraph.txt.gz", _SNAP + "com-amazon.top5000.cmty.txt.gz",
                    _SNAP + "com-amazon.all.dedup.cmty.txt.gz"],
    "snap_dblp": [_SNAP + "com-dblp.ungraph.txt.gz", _SNAP + "com-dblp.top5000.cmty.txt.gz",
                  _SNAP + "com-dblp.all.cmty.txt.gz"],
    # edges only, never the .nodes file; local only: no public source is used, so LinkedIn builds only on the laptop
    "linkedin": [os.path.join(OLD_PROJECT, "soc-linkedin.edges")],
}
CMTY_PATH = os.path.join(DATA_V2, "snap_youtube_cmty.npz")


def cmty_path(name):
    """The ground-truth community file of a graph in GT_GRAPHS."""
    return CMTY_PATH if name == "snap_youtube" else os.path.join(DATA_V2, f"{name}_cmty.npz")


# ------------------------------------------------------------------ registry and splits

def splits():
    """Train and validate on Pokec districts only; every other graph is a zero-shot test graph (interfaces s16)."""
    return {"train": ["pokec_z"], "val": ["pokec_t"], "test": [x for x in NAMES if x != "pokec_z"]}


HELD_OUT = tuple(splits()["test"])


def _check(name):
    if name not in LOADABLE:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(LOADABLE)}")


def built_path(name):
    _check(name)
    if name in POKEC:
        return os.path.join(DATA_ROOT, "pokec", f"{name}.npz")
    return os.path.join(DATA_V2, f"{name}.npz")


def available(name):
    """True if the built file exists (for a graph with ground-truth communities, also its community file)."""
    return os.path.exists(built_path(name)) and (name not in GT_GRAPHS or os.path.exists(cmty_path(name)))


def is_url(source):
    return str(source).startswith(("http://", "https://"))


def raw_files(name, raw_dir=RAW_V2):
    """[(source, local path)] for SOURCES[name]: URLs map to raw_dir/<name>/<file name>."""
    _check(name)
    return [(s, os.path.join(raw_dir, name, s.rsplit("/", 1)[-1]) if is_url(s) else s) for s in SOURCES[name]]


# ------------------------------------------------------------------ degree groups

def undirected_degree(graph):
    """Number of distinct neighbours of each node, ignoring direction."""
    n = graph.n
    if graph.m == 0:
        return np.zeros(n, dtype=np.int64)
    s, d = np.asarray(graph.src, dtype=np.int64), np.asarray(graph.dst, dtype=np.int64)
    code = np.unique(np.minimum(s, d) * np.int64(n) + np.maximum(s, d))
    lo, hi = code // n, code % n
    keep = lo != hi
    return np.bincount(lo[keep], minlength=n) + np.bincount(hi[keep], minlength=n)


def degree_group_names(n_bins=2):
    return ["low", "high"] if n_bins == 2 else [f"q{i + 1}" for i in range(n_bins)]


def degree_groups(graph, n_bins=2):
    """Quantile bins of undirected degree, low -> high. With n_bins = 2: degree <= median is 0, the rest 1."""
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if n_bins == 1 or graph.n == 0:
        return np.zeros(graph.n, dtype=np.int64)
    deg = undirected_degree(graph)
    q = np.quantile(deg, np.arange(1, n_bins) / n_bins)
    return np.searchsorted(q, deg, side="left").astype(np.int64)     # number of cut points strictly below deg


# ------------------------------------------------------------------ loading

def _graph(name, base, attrs, names, attribute, undirected):
    attribute = DEFAULT_ATTRIBUTE[name] if attribute is None else attribute
    if attribute not in attrs:
        raise KeyError(f"{name} has no attribute {attribute!r}; available: {sorted(attrs)}")
    meta = {"dataset": name, "attribute": attribute, "attrs": attrs, "attr_names": names,
            "tier": TIER[name], "undirected": bool(undirected)}
    return FIMGraph(name, base.n, base.src, base.dst, base.prob, attrs[attribute], list(names[attribute]), meta)


def load(name, attribute=None):
    """Full graph under weighted cascade, reading only its built file (never a raw file)."""
    _check(name)
    if not available(name):
        raise FileNotFoundError(f"{name} is not built ({built_path(name)}): run experiments/v2/build_datasets.py")
    if name in POKEC:
        base = load_pokec(name, "region")
        with np.load(built_path(name), allow_pickle=True) as z:
            names = {a: [str(x) for x in z[f"{a}_names"]] for a in POKEC_ATTRS}
        attrs = {a: np.asarray(base.meta["attrs"][a], dtype=np.int64) for a in POKEC_ATTRS}
        undirected = False
    else:
        with np.load(built_path(name)) as z:
            n = int(z["n"])
            undirected = bool(z["undirected"])
            keys = [f[:-len("_names")] for f in z.files if f.endswith("_names")]
            attrs = {a: z[a].astype(np.int64) for a in keys}
            names = {a: [str(x) for x in z[f"{a}_names"]] for a in keys}
            base = from_edges(name, n, z["arcs"], np.zeros(n, dtype=np.int64), ["all"],
                              undirected=undirected, weighted_cascade=True)
    attrs["degree"] = degree_groups(base)
    names["degree"] = degree_group_names(2)
    return _graph(name, base, attrs, names, attribute, undirected)


def ground_truth_communities(name, which="top5000"):
    """SNAP ground-truth communities (GT_GRAPHS) as graph ids (sorted, size >= 3, overlapping). which: "top5000" or
    "all"."""
    if name not in GT_GRAPHS:
        raise ValueError(f"ground-truth communities exist only for {GT_GRAPHS}, not {name!r}")
    if which not in ("top5000", "all"):
        raise ValueError("which must be 'top5000' or 'all'")
    path = cmty_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run experiments/v2/build_datasets.py --only {name}")
    pre = "" if which == "top5000" else "all_"
    with np.load(path) as z:
        ptr, members = z[pre + "ptr"].astype(np.int64), z[pre + "members"].astype(np.int64)
    return [members[ptr[i]:ptr[i + 1]] for i in range(len(ptr) - 1)]


# ------------------------------------------------------------------ BFS samples

@njit(cache=True)
def _bfs_forest(ptr_a, nbr_a, ptr_b, nbr_b, starts, size):
    """First `size` nodes of a BFS over the union of two CSR adjacencies, restarting from the next unseen
    entry of `starts` whenever a component is exhausted."""
    n = ptr_a.shape[0] - 1
    seen = np.zeros(n, dtype=np.bool_)
    out = np.empty(size, dtype=np.int64)
    cnt = 0
    head = 0
    for i in range(starts.shape[0]):
        if cnt >= size:
            break
        s = starts[i]
        if seen[s]:
            continue
        seen[s] = True
        out[cnt] = s
        cnt += 1
        while head < cnt and cnt < size:
            u = out[head]
            head += 1
            for t in range(ptr_a[u], ptr_a[u + 1]):
                w = nbr_a[t]
                if cnt < size and not seen[w]:
                    seen[w] = True
                    out[cnt] = w
                    cnt += 1
            for t in range(ptr_b[u], ptr_b[u + 1]):
                w = nbr_b[t]
                if cnt < size and not seen[w]:
                    seen[w] = True
                    out[cnt] = w
                    cnt += 1
    return out[:cnt]


def sample(name, size, seed, attribute=None, full=None):
    """Induced subgraph on an undirected BFS ball of `size` nodes from a seeded random start.

    The start is a uniformly random non-isolated node (np.random.default_rng(seed)); if its component runs out,
    the BFS continues from the next node of the same random order. Weighted cascade is recomputed inside the
    sample; meta["full_ids"] holds the full-graph ids (sorted; sample node i is full node full_ids[i]) and every
    attribute, including "degree", is the full graph's. `full` may pass an already loaded load(name) graph."""
    full = load(name, attribute) if full is None else full
    if full.meta.get("dataset") != name:
        raise ValueError(f"full graph is {full.meta.get('dataset')!r}, not {name!r}")
    size = int(size)
    if not 1 <= size <= full.n:
        raise ValueError(f"sample size {size} outside [1, {full.n}]")
    undirected = bool(full.meta.get("undirected", False))
    ptr_a, nbr_a, _ = full.out_csr
    if undirected:                                    # arcs are symmetric: out-neighbours are all neighbours
        ptr_b, nbr_b = np.zeros(full.n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    else:
        ptr_b, nbr_b, _ = full.in_csr
    active = (full.in_degree + full.out_degree) > 0
    rng = np.random.default_rng(seed)
    starts = np.concatenate([rng.permutation(np.flatnonzero(active)),
                             rng.permutation(np.flatnonzero(~active))]).astype(np.int64)
    ids = np.sort(_bfs_forest(ptr_a, nbr_a, ptr_b, nbr_b, starts, size))

    local = np.full(full.n, -1, dtype=np.int64)
    local[ids] = np.arange(size, dtype=np.int64)
    s, d = local[full.src], local[full.dst]
    keep = (s >= 0) & (d >= 0)
    attribute = full.meta["attribute"] if attribute is None else attribute
    attrs = {a: np.asarray(g, dtype=np.int64)[ids] for a, g in full.meta["attrs"].items()}
    if attribute not in attrs:
        raise KeyError(f"{name} has no attribute {attribute!r}; available: {sorted(attrs)}")
    names = {a: list(v) for a, v in full.meta["attr_names"].items()}
    meta = {**full.meta, "attribute": attribute, "attrs": attrs, "attr_names": names, "full_ids": ids,
            "n_full": full.n, "sample": {"size": size, "seed": int(seed), "start": int(starts[0])}}
    # unique name per (dataset, size, seed): metrics.communities caches Louvain partitions by graph name
    return from_edges(f"{name}_bfs{size}_s{int(seed)}", size, np.stack([s[keep], d[keep]], 1), attrs[attribute],
                      names[attribute], undirected=undirected, weighted_cascade=True, meta=meta)


# ------------------------------------------------------------------ raw readers

def _not_nodes_file(path):
    if str(path).endswith(".nodes"):
        raise ValueError("soc-linkedin.nodes holds real profile handles and must never be read")
    return path


def _open_text(path):
    path = _not_nodes_file(path)                      # utf-8-sig: a byte-order mark must not look like a header
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return open(path, encoding="utf-8-sig", errors="replace")


def _header_rows(path, sep):
    """1 if the first line is a header (its first field is not an integer), else 0. Comment lines count as 0."""
    with _open_text(path) as f:
        line = next((ln for ln in f if ln.strip()), "")
    if not line or line.lstrip()[0] in "#%":
        return 0
    tok = (line.split() if sep == r"\s+" else line.split(sep))[0].strip().strip('"')
    return 0 if tok.lstrip("-").isdigit() else 1


def _read_pairs(path, sep=r"\s+", comment="#", chunksize=4_000_000):
    """First two integer columns of an edge list, (E, 2) int64, read in chunks with the pandas C engine."""
    # UNVERIFIED: C engine with sep=r"\s+", usecols, comment and chunksize together under pandas 3.0 (not run);
    # .gz files rely on pandas' compression inference
    kw = dict(sep=sep, header=None, skiprows=_header_rows(path, sep), usecols=[0, 1], dtype=np.int64,
              comment=comment, engine="c", chunksize=chunksize)
    with pd.read_csv(_not_nodes_file(path), **kw) as reader:
        parts = [c.to_numpy(dtype=np.int64) for c in reader]
    return np.concatenate(parts) if parts else np.zeros((0, 2), dtype=np.int64)


def _read_tokens(path, n_cols):
    """First n_cols whitespace-separated tokens of each line of a small file; header lines are skipped."""
    rows = []
    with _open_text(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= n_cols and t[0].lstrip("-").isdigit():
                rows.append(t[:n_cols])
    return rows


def _int_pairs(rows):
    ok = [r for r in rows if r[1].lstrip("-").isdigit()]
    return np.array([[int(a), int(b)] for a, b in ok], dtype=np.int64).reshape(-1, 2)


def _read_communities(path, orig):
    """One community per line (raw ids) -> (ptr, members) in graph ids; members sorted, size >= 3."""
    parts = []
    with _open_text(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            loc = _map_ids(orig, np.array([int(x) for x in t], dtype=np.int64))
            loc = np.unique(loc[loc >= 0])
            if len(loc) >= 3:
                parts.append(loc)
    ptr = np.zeros(len(parts) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.array([len(c) for c in parts], dtype=np.int64))
    return ptr, (np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64))


# ------------------------------------------------------------------ id handling and saving

def _relabel(pairs):
    """Raw ids of an (E, 2) array -> (sorted unique raw ids, u, v in 0..n-1)."""
    orig, inv = np.unique(np.ascontiguousarray(pairs).ravel(), return_inverse=True)
    inv = inv.reshape(-1, 2)
    return orig.astype(np.int64), inv[:, 0].astype(np.int64), inv[:, 1].astype(np.int64)


def _map_ids(orig, raw):
    """Position of each raw id in the sorted id array orig, -1 where absent."""
    raw = np.asarray(raw, dtype=np.int64)
    if len(orig) == 0:
        return np.full(len(raw), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(orig, raw), len(orig) - 1)
    return np.where(orig[pos] == raw, pos, -1).astype(np.int64)


def _arcs(u, v, n, undirected):
    """Sorted unique arcs (A, 2) without self-loops; both directions if undirected (the from_edges convention)."""
    keep = u != v
    u, v = u[keep], v[keep]
    if undirected:
        u, v = np.concatenate([u, v]), np.concatenate([v, u])
    code = np.unique(u * np.int64(n) + v)
    return np.stack([code // n, code % n], 1)


def _fmt(x):
    return str(int(x)) if isinstance(x, (float, np.floating)) and float(x).is_integer() else str(x)


def _categorical(values):
    """Sorted category codes (int64) and names; missing values become an explicit last group "unknown"."""
    codes, uniq = pd.factorize(pd.Series(values), sort=True)
    codes = np.asarray(codes, dtype=np.int64)
    names = [_fmt(x) for x in uniq]
    if (codes < 0).any():
        codes[codes < 0] = len(names)
        names.append("unknown")
    return codes, names


def _atomic_savez(target, **arrays):
    """Write to a temp file, then replace: a half-written file never looks built."""
    tmp = target[:-len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, target)


def _save(name, n, arcs, undirected, orig_id, attrs=None):
    os.makedirs(DATA_V2, exist_ok=True)
    out = {"n": np.int64(n), "arcs": arcs.astype(np.int32 if n < 2**31 - 1 else np.int64),
           "undirected": np.bool_(undirected), "orig_id": np.asarray(orig_id, dtype=np.int64)}
    for a, (codes, names) in (attrs or {}).items():
        out[a] = np.asarray(codes, dtype=np.int64)
        out[f"{a}_names"] = np.array([str(x) for x in names])
    _atomic_savez(built_path(name), **out)


def _summary(name, n, arcs, undirected, **extra):
    return {"dataset": name, "n": int(n), "arcs": int(len(arcs)),
            "edges": int(len(arcs) // 2 if undirected else len(arcs)), "undirected": bool(undirected), **extra}


def _build_attributed(name, table, attr_cols, pairs, undirected):
    """Node set = the attributed ids of `table` (column "id"); edges with an unattributed end are dropped."""
    table = table.dropna(subset=["id"]).drop_duplicates("id", keep="first")
    ids = table["id"].to_numpy(dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    n = len(ids)
    u, v = _map_ids(ids, pairs[:, 0]), _map_ids(ids, pairs[:, 1])
    ok = (u >= 0) & (v >= 0)
    arcs = _arcs(u[ok], v[ok], n, undirected)
    attrs = {a: _categorical(table[a].to_numpy()[order]) for a in attr_cols}
    _save(name, n, arcs, undirected, ids, attrs)
    return _summary(name, n, arcs, undirected, source_edges=int(len(pairs)),
                    dropped_unattributed=int((~ok).sum()), isolated=int(n - len(np.unique(arcs[:, 0]))),
                    groups={a: dict(zip(nm, np.bincount(c, minlength=len(nm)).tolist()))
                            for a, (c, nm) in attrs.items()})


def _build_structural(name, pairs, undirected):
    """Node set = ids that occur in an edge, relabelled 0..n-1 in raw-id order."""
    source_edges = int(len(pairs))
    orig, u, v = _relabel(pairs)
    del pairs
    n = len(orig)
    arcs = _arcs(u, v, n, undirected)
    del u, v
    _save(name, n, arcs, undirected, orig)
    return _summary(name, n, arcs, undirected, source_edges=source_edges)


def _local(name, raw_dir):
    paths = [p for _, p in raw_files(name, raw_dir)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{name}: raw files missing: {missing} (experiments/v2/build_datasets.py downloads them)")
    return paths


# ------------------------------------------------------------------ builders

def build_rice_facebook(raw_dir=RAW_V2):
    # UNVERIFIED: formats seen only in the first lines (tab-separated, no header; attr = id, college, age, major as
    # in the source notebook); whether attr.txt lists exactly the 1,205 users and edges.txt the 42,443 pairs is not
    # checked (build_datasets.py warns on a mismatch)
    e_path, a_path = _local("rice_facebook", raw_dir)
    att = pd.read_csv(a_path, sep="\t", header=None, skiprows=_header_rows(a_path, "\t"), engine="c")
    if att.shape[1] < 4:
        raise ValueError(f"{a_path}: expected 4 columns (id, college, age, major), found {att.shape[1]}")
    att = att.iloc[:, :4]
    att.columns = ["id", "college", "age", "major"]
    return _build_attributed("rice_facebook", att, ["college", "age", "major"], _read_pairs(e_path),
                             undirected=True)


def build_twitter_political(raw_dir=RAW_V2):
    # UNVERIFIED: 3,753 users / 6,993 pairs not checked against the files; the meaning of leaning codes 0/1/2 is
    # not documented upstream, so the raw codes are kept as group names
    l_path, a_path = _local("twitter_political", raw_dir)
    rows = _read_tokens(a_path, 2)
    att = pd.DataFrame({"id": np.array([int(r[0]) for r in rows], dtype=np.int64), "leaning": [r[1] for r in rows]})
    # links lines are "u v [p1, p2]": only the first two tokens are used
    return _build_attributed("twitter_political", att, ["leaning"], _int_pairs(_read_tokens(l_path, 2)),
                             undirected=True)


def build_instagram(raw_dir=RAW_V2):
    # UNVERIFIED: no header, integer ids and genders F/M were seen in the first lines only; whether the gender file
    # lists exactly the 553,628 users is not checked (build_datasets.py warns on a mismatch)
    e_path, g_path = _local("instagram", raw_dir)
    att = pd.read_csv(g_path, sep=",", header=None, skiprows=_header_rows(g_path, ","), usecols=[0, 1],
                      dtype={0: np.int64, 1: str}, engine="c")
    att.columns = ["id", "gender"]
    return _build_attributed("instagram", att, ["gender"], _read_pairs(e_path, sep=",", comment=None),
                             undirected=True)


def build_epinions(raw_dir=RAW_V2):
    (path,) = _local("epinions", raw_dir)
    pairs = _read_pairs(path, comment="#")
    return _build_structural("epinions", pairs[:, ::-1], undirected=False)          # a trusts b: b -> a


def build_twitter_ego(raw_dir=RAW_V2):
    (path,) = _local("twitter_ego", raw_dir)
    pairs = _read_pairs(path, comment="#")
    return _build_structural("twitter_ego", pairs[:, ::-1], undirected=False)       # a follows b: b -> a


def _build_snap_communities(name, raw_dir):
    """A SNAP "com-" graph (undirected) and its ground-truth communities (top 5,000 and all)."""
    g_path, top_path, all_path = _local(name, raw_dir)
    orig, u, v = _relabel(_read_pairs(g_path, comment="#"))
    n = len(orig)
    arcs = _arcs(u, v, n, undirected=True)
    top, every = _read_communities(top_path, orig), _read_communities(all_path, orig)
    os.makedirs(DATA_V2, exist_ok=True)
    # communities first: available(name) needs both files, so a crash in between leaves it unbuilt
    _atomic_savez(cmty_path(name), ptr=top[0], members=top[1], all_ptr=every[0], all_members=every[1])
    _save(name, n, arcs, True, orig)
    return _summary(name, n, arcs, True, communities_top5000=len(top[0]) - 1, communities_all=len(every[0]) - 1)


def build_snap_youtube(raw_dir=RAW_V2):
    return _build_snap_communities("snap_youtube", raw_dir)


def build_snap_amazon(raw_dir=RAW_V2):
    return _build_snap_communities("snap_amazon", raw_dir)


def build_snap_dblp(raw_dir=RAW_V2):
    return _build_snap_communities("snap_dblp", raw_dir)


def build_linkedin(raw_dir=RAW_V2):
    # UNVERIFIED: peak RAM estimated at about 2-3 GB (19.4M pairs; np.unique with inverse), not measured
    (path,) = _local("linkedin", raw_dir)                                            # soc-linkedin.edges only
    return _build_structural("linkedin", _read_pairs(path, comment="%"), undirected=True)


BUILDERS = {"rice_facebook": build_rice_facebook, "twitter_political": build_twitter_political,
            "epinions": build_epinions, "twitter_ego": build_twitter_ego, "instagram": build_instagram,
            "snap_youtube": build_snap_youtube, "snap_amazon": build_snap_amazon, "snap_dblp": build_snap_dblp,
            "linkedin": build_linkedin}


def build_all(raw_dir=RAW_V2, only=None, force=False):
    """Build data/v2/<name>.npz for every non-Pokec dataset (or `only`); built ones are skipped unless force.

    Raw files must already be in raw_dir (experiments/v2/build_datasets.py downloads them). Pokec graphs are
    built by experiments/build_pokec.py (pokec_t by experiments/v2/build_datasets.py). Returns name -> summary."""
    names = list(BUILDERS) if only is None else list(only)
    bad = [x for x in names if x not in BUILDERS]
    if bad:
        raise ValueError(f"no v2 builder for {bad}: Pokec graphs come from experiments/build_pokec.py "
                         f"(pokec_t from experiments/v2/build_datasets.py)")
    return {x: ({"dataset": x, "skipped": "already built"} if available(x) and not force else BUILDERS[x](raw_dir))
            for x in names}
