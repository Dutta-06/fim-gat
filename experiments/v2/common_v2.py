"""Shared v2 experiment plumbing: provenance, timing, memory presets and the v2 results recorder.

RecorderV2 scores every seed set with forward Monte Carlo only (metrics_v2.evaluate_v2) plus the
paired equal-spread test against the reference seeds (reference.equal_spread_test), and appends one
CSV row per call (docs/v2_interfaces.md section 18). Rows store the seed set, so every metric can be
recomputed, and carry their provenance (code_tag, gpu, preset). The CSV is rewritten atomically after every row (temp file, then os.replace), so an
interrupted run never leaves a truncated file. A method that fails is written as a row with
status "missing" and its reason, never dropped.

Every method on a graph is evaluated on the same Monte Carlo stream (common random numbers), seeded
from the graph name as in v1. That stream is independent of the one reference_spread uses to pick
its best run, so the reference gets no winner's-curse advantage in the equal-spread test.

There is no git, so code_tag() hashes the Python sources that produce results.
"""
import hashlib
import json
import math
import os
import sys
import time
import weakref
import zlib
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fimgat.v2.baselines_v2 import check_seeds                          # noqa: E402
from fimgat.v2.config import N_SIMS, TIER                               # noqa: E402
from fimgat.v2.metrics_v2 import evaluate_v2                            # noqa: E402

NAN = float("nan")
CODE_PATHS = ("fimgat", "experiments/build_pokec.py", "experiments/v2")   # sources hashed by code_tag()

# GPU memory presets. chunk_edges bounds the edges per chunk of CPU-edge inference; graphs above checkpoint_edges
# train with activation checkpointing; full_R and full_max_roots size the RR sample of the large tier.
PRESETS = {
    "8gb":  dict(checkpoint_edges=300_000, chunk_edges=3_000_000, full_max_roots=150_000, full_R=8),
    "16gb": dict(checkpoint_edges=800_000, chunk_edges=8_000_000, full_max_roots=400_000, full_R=12),
    "40gb": dict(checkpoint_edges=4_000_000, chunk_edges=25_000_000, full_max_roots=1_000_000, full_R=16),
}


def detect_preset():
    if not torch.cuda.is_available():
        return "8gb"
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    return "40gb" if gb > 30 else "16gb" if gb > 12 else "8gb"


# ------------------------------------------------------------------ hardware, presets, timing

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def preset_config(name=None):
    """(name, settings) of a memory preset (PRESETS); None detects it from the GPU."""
    name = detect_preset() if name is None else name
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; presets are {sorted(PRESETS)}")
    return name, dict(PRESETS[name])


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    """(fn(), seconds), with CUDA synchronised on both sides so queued GPU work is counted."""
    _sync()
    t = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - t


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_gb():
    """(peak allocated, peak reserved) GPU memory in GB since reset_peak(); NaN without CUDA.
    research_protocol section 4 asks for both, per stage."""
    if not torch.cuda.is_available():
        return NAN, NAN
    return torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30


# ------------------------------------------------------------------ progress

PROGRESS_LOG_SECONDS = 60.0         # at most one progress line per minute in a log file


class _LogLines:
    """tqdm's output for a log file (Kaggle and cloud runs): each refresh on a line of its own instead of carriage
    returns, so kaggle/run_job.py can relay the last one."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, s):
        s = s.replace("\r", "").strip()
        if s:
            self.stream.write(s + "\n")

    def flush(self):
        self.stream.flush()


class _NoBar:
    def update(self, n=1):
        pass

    def set_description_str(self, s=None, refresh=True):
        pass

    def set_postfix_str(self, s="", refresh=True):
        pass

    def close(self):
        pass


def progress(total, desc="", unit="query"):
    """A progress bar on stderr (tqdm) with an ETA: live in a terminal, one line a minute in a log file. Output
    only: it never touches a result. Without tqdm (it comes with torch_geometric) it does nothing."""
    try:
        from tqdm import tqdm
    except ImportError:
        return _NoBar()
    tqdm.monitor_interval = 0                     # no monitor thread: nothing runs beside the timed stages
    tty = sys.stderr.isatty()
    return tqdm(total=total, desc=desc, unit=unit, file=sys.stderr if tty else _LogLines(sys.stderr),
                mininterval=0.5 if tty else PROGRESS_LOG_SECONDS, ascii=None if tty else True,
                ncols=None if tty else 110, dynamic_ncols=tty, smoothing=0.05)


# ------------------------------------------------------------------ provenance

@lru_cache(maxsize=None)
def code_tag(paths=CODE_PATHS, root=ROOT, length=12):
    """Short SHA-1 over the relative path and contents of every .py file under `paths` (files or
    folders relative to root; __pycache__ skipped; CRLF normalised to LF). Any source edit changes
    it. Computed once per process, so it describes the code loaded at the start of the run."""
    files = []
    for p in paths:
        full = os.path.join(root, p)
        if os.path.isfile(full):
            files.append(full)
        elif os.path.isdir(full):
            for dirpath, dirnames, names in os.walk(full):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                files += [os.path.join(dirpath, f) for f in names if f.endswith(".py")]
    rel = sorted({os.path.relpath(f, root).replace(os.sep, "/"): f for f in files}.items())
    h = hashlib.sha1()
    for r, f in rel:
        with open(f, "rb") as fh:
            data = fh.read().replace(b"\r\n", b"\n")
        h.update(r.encode() + b"\0" + data + b"\0")
    return h.hexdigest()[:length]


def mc_seed_for(graph):
    """Common-random-numbers stream of a graph, as v1 Recorder: crc32 of its name."""
    return zlib.crc32(graph.name.encode()) % (2**31)


def n_sims_for(graph, default=2000):
    """Monte Carlo simulations for a graph's tier (config.N_SIMS); `default` if the tier is unknown."""
    tier = graph.meta.get("tier") or TIER.get(graph.meta.get("dataset"))
    return int(N_SIMS.get(tier, default))


# ------------------------------------------------------------------ CSV cells and key matching

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _cell(v):
    """Containers go into the CSV as JSON strings; scalars as they are."""
    if isinstance(v, (dict, list, tuple, set, np.ndarray)):
        return json.dumps(list(v) if isinstance(v, set) else v, default=_json_default)
    return v


def _norm(v):
    """Comparable form of a key value: None for missing (None, NaN, ''), float for numbers and bools,
    str otherwise. CSV round trips turn '' into NaN and ints into floats; this undoes both."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return float(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return None if math.isnan(f) else f
    s = str(v)
    return None if s in ("", "nan", "NaN", "None") else s


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        try:                                            # e.g. a variant "1" read back as 1.0
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return a == b


def _matches(row, key):
    return all(_same(_norm(row.get(c)), _norm(v)) for c, v in key.items())


# ------------------------------------------------------------------ recorder

class RecorderV2:
    """Forward-MC evaluation plus the equal-spread test, one CSV row per recorded seed set.

    Row columns: dataset, attribute, graph, n, m, k, rho, method, variant, train_seed, status
    ("ok" / "missing"), reason, time_s, prep_s, code_tag, gpu, preset, seeds (JSON), n_sims,
    mc_seed, every evaluate_v2 metric (rates as JSON), es_passed, es_ratio, es_lower, es_se, and
    any extra keyword columns (containers stored as JSON).

    n_sims: None takes config.N_SIMS of the graph's tier. preset: None detects it from the GPU.
    Rows are appended, never replaced: check done() first, or remove() stale rows explicitly.
    Metrics of a seed set already scored in this process (same graph object, stream, communities)
    are reused, which matters for rho-free methods recorded once per rho.
    """

    def __init__(self, path, n_sims=None, preset=None, alpha=0.05, memo_size=512):
        self.path = os.path.abspath(path)
        self.n_sims = None if n_sims is None else int(n_sims)
        self.alpha = float(alpha)
        self.preset = detect_preset() if preset is None else str(preset)
        self.gpu, self.code_tag = gpu_name(), code_tag()
        self.memo_size, self._memo, self._memo_graph = int(memo_size), {}, None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.rows = self._load()

    # ---------------------------------------------------------- persistence

    def _load(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return []
        try:
            df = pd.read_csv(self.path, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            return []
        return df.to_dict("records")

    def save(self):
        """Atomic rewrite: write a temp file next to the CSV, then os.replace it over the CSV."""
        tmp = f"{self.path}.{os.getpid()}.tmp"
        pd.DataFrame(self.rows).to_csv(tmp, index=False)
        for i in range(10):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:                     # Windows: the CSV is briefly open elsewhere
                if i == 9:
                    raise
                time.sleep(0.5)

    # ---------------------------------------------------------- queries

    def find(self, **key):
        return [r for r in self.rows if _matches(r, key)]

    def done(self, **key):
        """True if any row matches every given column. The key must include rho and variant
        (contract section 18). Missing-method rows count as done; add status="ok" to the key to
        retry them."""
        need = [c for c in ("rho", "variant") if c not in key]
        if need:
            raise ValueError(f"done() needs {need} in its key")
        return any(_matches(r, key) for r in self.rows)

    def remove(self, **key):
        """Delete every row matching the key (at least one column); returns how many went."""
        if not key:
            raise ValueError("remove() needs a key")
        keep = [r for r in self.rows if not _matches(r, key)]
        gone = len(self.rows) - len(keep)
        if gone:
            self.rows = keep
            self.save()
        return gone

    # ---------------------------------------------------------- evaluation (memoised)

    def _memo_of(self, graph):
        """The memo of the current graph. Only a weak reference to the graph is kept, so a finished
        graph (full Pokec holds GBs) is never pinned in memory; a new graph starts a new memo."""
        if self._memo_graph is None or self._memo_graph() is not graph:
            self._memo, self._memo_graph = {}, weakref.ref(graph)
        elif len(self._memo) >= self.memo_size:
            self._memo.clear()
        return self._memo

    def _evaluate(self, graph, seeds, n_sims, mc_seed, communities):
        memo = self._memo_of(graph)
        key = ("ev", id(communities), n_sims, mc_seed, tuple(sorted(seeds)))
        hit = memo.get(key)
        # the stored communities object keeps its id alive; "is" rules out a reused id
        if hit is not None and hit[0] is communities:
            return hit[1]
        m = evaluate_v2(graph, seeds, n_sims=n_sims, seed=mc_seed, communities=communities)
        memo[key] = (communities, m)
        return m

    def _equal_spread(self, graph, seeds, ref_seeds, rho, n_sims, mc_seed):
        out = {"es_passed": NAN, "es_ratio": NAN, "es_lower": NAN, "es_se": NAN}
        if ref_seeds is None or _norm(rho) is None:
            return out
        if isinstance(ref_seeds, dict):
            ref_seeds = ref_seeds["seeds"]
        s, ref = sorted(seeds), sorted(check_seeds(ref_seeds, len(seeds), graph.n))
        memo = self._memo_of(graph)
        key = ("es", n_sims, mc_seed, self.alpha, tuple(s), tuple(ref))
        hit = memo.get(key)
        if hit is None:
            from fimgat.v2.reference import equal_spread_test
            # sorted inputs make the test a function of the two sets, whatever their listing order
            es = equal_spread_test(graph, s, ref, rho=float(rho), n_sims=n_sims, seed=mc_seed, alpha=self.alpha)
            hit = {"ratio": float(es["ratio"]), "lower": float(es["lower"]), "se": float(es.get("se", NAN))}
            memo[key] = hit
        # ratio, lower and se do not depend on rho; passed = lower >= rho (contract section 7)
        out.update(es_passed=bool(hit["lower"] >= float(rho)), es_ratio=hit["ratio"],
                   es_lower=hit["lower"], es_se=hit["se"])
        return out

    # ---------------------------------------------------------- rows

    def _base(self, graph, k, rho, method, variant, train_seed, secs, prep_s, status, reason):
        r = _norm(rho)
        return {"dataset": graph.meta.get("dataset") or graph.name, "attribute": graph.meta.get("attribute"),
                "graph": graph.name, "n": int(graph.n), "m": int(graph.m), "k": int(k),
                "rho": NAN if r is None else float(rho), "method": str(method), "variant": variant,
                "train_seed": train_seed, "status": status, "reason": reason,
                "time_s": float(secs) if _norm(secs) is not None else NAN,
                "prep_s": float(prep_s) if _norm(prep_s) is not None else NAN,
                "code_tag": self.code_tag, "gpu": self.gpu, "preset": self.preset}

    def _append(self, row, extra):
        clash = sorted(set(extra) & set(row))
        if clash:
            raise ValueError(f"extra columns {clash} would overwrite recorder columns")
        row.update({c: _cell(v) for c, v in extra.items()})
        self.rows.append(row)
        self.save()
        return row

    def add(self, graph, k, rho, method, seeds, secs, ref_seeds=None, variant="default", train_seed=NAN,
            prep_s=NAN, communities=None, mc_seed=None, n_sims=None, **extra):
        """Evaluate a seed set and append its row. ref_seeds: the reference seeds for this k (or the
        reference_spread dict); None leaves the es_* fields NaN, as does rho=None. communities:
        ground-truth communities for evaluate_v2 (None: v1 Louvain). Raises ValueError unless seeds
        are exactly k distinct node ids of graph."""
        seeds = check_seeds(seeds, int(k), graph.n)
        ns = int(n_sims) if n_sims is not None else self.n_sims if self.n_sims is not None else n_sims_for(graph)
        ms = mc_seed_for(graph) if mc_seed is None else int(mc_seed)
        m = self._evaluate(graph, seeds, ns, ms, communities)
        row = self._base(graph, k, rho, method, variant, train_seed, secs, prep_s, "ok", "")
        row["seeds"] = json.dumps(seeds)
        row["n_sims"], row["mc_seed"] = ns, ms
        for c, v in m.items():
            if c == "sims":
                continue
            row[c] = json.dumps([float(x) for x in v]) if c == "rates" else v
        row.update(self._equal_spread(graph, seeds, ref_seeds, rho, ns, ms))
        return self._append(row, extra)

    def add_missing(self, graph, k, rho, method, reason, secs=NAN, variant="default", train_seed=NAN,
                    prep_s=NAN, **extra):
        """Append the row of a method that produced no seed set (OOM, time limit, not applicable)."""
        row = self._base(graph, k, rho, method, variant, train_seed, secs, prep_s, "missing", str(reason))
        row["seeds"] = ""
        return self._append(row, extra)

    def add_suite(self, graph, k, rho, suite, ref_seeds, variant="default", prep_s=NAN, communities=None,
                  skip_done=True, **extra):
        """Record a baselines_v2.run_suite result: one row per method, missing ones with their reason
        and each method's info as a JSON "info" column. skip_done leaves methods already in the CSV
        for this (graph, k, rho, variant) alone. Returns the new rows."""
        reasons, infos = getattr(suite, "reason", {}), getattr(suite, "info", {})
        ident = {"dataset": graph.meta.get("dataset") or graph.name, "attribute": graph.meta.get("attribute"),
                 "graph": graph.name, "k": int(k), "rho": rho, "variant": variant}
        rows = []
        for name, (seeds, secs) in suite.items():
            if skip_done and self.done(method=name, **ident):
                continue
            cols = dict(extra)
            if infos.get(name):
                cols["info"] = infos[name]
            if seeds is None:
                rows.append(self.add_missing(graph, k, rho, name, reasons.get(name, "no seed set"), secs,
                                             variant=variant, prep_s=prep_s, **cols))
            else:
                rows.append(self.add(graph, k, rho, name, seeds, secs, ref_seeds, variant=variant,
                                     prep_s=prep_s, communities=communities, **cols))
        return rows


def fmt_row(r):
    """One log line for a recorded row."""
    head = f"    {str(r.get('graph')):16s} k={r.get('k'):<4} rho={r.get('rho')} {str(r.get('method')):16s}"
    if r.get("status") == "missing":
        return f"{head} missing: {r.get('reason')}"
    v = _norm(r.get("es_passed"))                       # a reloaded CSV may hold "True"/"False" strings
    es = "  n/a" if v is None else " pass" if v in (1.0, "True") else " FAIL"
    return (f"{head} infl={r['influence']:9.2f} disp={r['disparity']:.4f} es{es} "
            f"(ratio {r['es_ratio']:.4f}, lower {r['es_lower']:.4f}) ({r['time_s']:.1f}s)")
