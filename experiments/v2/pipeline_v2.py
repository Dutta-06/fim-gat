"""The FIM-GAT v2 experiment pipeline shared by run_t1_t2.py, run_t4.py and tune_v2.py (docs/design_v2.md sections
10, 18 and 19).

A Case is one graph (a dataset under one group attribute, or a BFS sample of one), prepared once, as a deployment
would prepare it:
  reference    the best of REFERENCE_RUNS RR-greedy runs per budget, scored by forward Monte Carlo (the sets of
               reference.reference_spread; each run's greedy serves every budget, since greedy(k) is the k-prefix
               of greedy(max k)). Kept in a JSON store, so a resumed run does not recompute it.
  rr_sample    the search sample A: ExposureEstimator with R and max_roots of the tier (config.RR_PER_NODE, or
               the memory preset for the large tier)
  cert_sample  the certificate's sample B (CPU, its own seed): independent of everything that chose a set or
               the reference, sized by cert_config's memory rule (design section 19)
  features     the context with static features (only when a learned part runs)
  rr_greedy    RR-greedy on A and its exact totals, the rest of train.prepare_v2 (only for the learned start)
  cover_index  the verifier's static RR-set index
Every query (k, rho) then runs one of:
  FIM-GAT v2   start (the constructor's set, or the reference set) -> search (variant v2-b: improve.hive_search;
               v2-b0: improve(audit=True), the published rule; v2-a: improve(audit=False), labelled only; with the
               simple hybrid, exhaustive or learned proposer) -> certified deployment (certify.certify_sets over
               certify.deploy_candidates of the search path, on B, then certify.choose by each rule of
               config.DEPLOY_RULES). Three results: the set deployed by Theorem 2's rule ("certified", T1's
               deployment), the set deployed by the path rule ("certified-path", T2's deployment; design section
               21) and the final search set ("final", the "deployment" ablation row).
  a baseline   baselines_v2.run_suite's method.
and every resulting set gets the same judge: certify.certified_equal_spread on B at level ALPHA.

The floor is config.rr_floor(rho, total_ref, delta), total_ref being the exact RR total of the reference set on A
and delta the margin tuned on validation (tune_v2.py), the same for every method that searches against the floor.
"""
import gc
import json
import os
import time
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from common_v2 import NAN, _matches, peak_gpu_gb, reset_peak, timed
from fimgat.baselines import RRGreedy
from fimgat.diffusion import ExposureEstimator, stratified_roots
from fimgat.v2.baselines_v2 import SUITE, check_seeds, run_suite
from fimgat.v2.certify import certified_equal_spread, certify_sets, choose, deploy_candidates, path_states
from fimgat.v2.config import (AUDIT_MAX_SIZE, CERT_ALPHA, CERT_NNZ_BUDGET, CERT_R_MAX, CERT_SEED, DELTA_MARGIN,
                              MAX_CONDUCTANCE,
                              MODELS_V2, N_TRAIN_SEEDS, REFERENCE_RUNS, RESULTS_V2, RR_PER_NODE, TIER, TUNING_FILE,
                              rr_floor)
from fimgat.v2.cover import cover_index
from fimgat.v2.features import build_context
from fimgat.v2.improve import hive_search, improve
from fimgat.v2.proposer import LearnedProposer
from fimgat.v2.reference import mc_spreads, reference_spread
from fimgat.v2.train import exact_rates, load_policy, select_seeds_v2

try:                                        # optional: host memory after each stage
    import psutil
except ImportError:
    psutil = None

FIM = "FIM-GAT v2"
ALPHA = 0.05                                # level of the per-set equal-spread certificate (the judge)
REF_MC_SEED = 12345                         # reference_spread's default stream for choosing its best run
NO_TRAIN = -1                               # train_seed of a pipeline with nothing learned in it
VARIANTS = ("v2-b", "v2-b0", "v2-a")        # acceptance: hive_search; the published rule; labelled only
# deployment column of a FIM-GAT row -> certify.choose rule (design section 21); "final" is the undeployed search set
DEPLOYMENTS = {"certified": "min_ucb", "certified-path": "path"}
STARTS = ("learned", "greedy")
PROPOSERS = ("simple", "exhaustive", "learned")
GREEDY_FAMILY = ("Greedy", "Parity-Greedy", "Welfare-Greedy", "Maximin-Greedy")
SEARCH = ("LocalSearch", "Simple-v2")       # internal variants of the method (design section 18.2), run as baselines
SMALL_MAX_N, MEDIUM_MAX_N = 5_000, 200_000  # tier of a BFS sample, by size (config.TIER names full datasets)
# cached stages each baseline uses (from a loaded graph); they make up its prep_s
BASELINE_NEEDS = {"Degree": ("reference", "cert_sample"), "AGE": ("reference", "cert_sample"),
                  "Prior": ("reference", "cert_sample"), "FW-Disparity": ("reference", "rr_sample", "cert_sample"),
                  **{m: ("reference", "rr_sample", "cert_sample", "suite_index") for m in GREEDY_FAMILY},
                  **{m: ("reference", "rr_sample", "cert_sample", "suite_index", "cover_index") for m in SEARCH},
                  "BSM": ("reference", "rr_sample", "cert_sample", "suite_index"),
                  "MultiFW-MaxMin": ("reference", "rr_sample", "cert_sample", "suite_index"),
                  "RFIM": ("reference", "rr_sample", "cert_sample", "suite_index")}


# ------------------------------------------------------------------ configurations

@dataclass(frozen=True)
class FimConfig:
    """One FIM-GAT v2 pipeline: acceptance variant, start set and improvement proposer."""
    variant: str = "v2-b"
    start: str = "greedy"
    proposer: str = "simple"

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant {self.variant!r} not in {VARIANTS}")
        if self.start not in STARTS:
            raise ValueError(f"start {self.start!r} not in {STARTS}")
        if self.proposer not in PROPOSERS:
            raise ValueError(f"proposer {self.proposer!r} not in {PROPOSERS}")

    @property
    def learned(self):
        return self.start == "learned" or self.proposer == "learned"

    def cols(self):
        """The columns that identify this configuration in a results row (besides method and train_seed)."""
        return {"variant": self.variant, "start": self.start, "proposer": self.proposer}

    def name(self):
        return f"{self.variant}/{self.start}/{self.proposer}"

    def needs(self):
        """Cached stages this pipeline uses: the constructor needs features, rr_greedy and its checkpoint; the
        learned proposer needs features and its checkpoint; the greedy start needs neither."""
        need = ["reference", "rr_sample", "cert_sample", "cover_index"]
        if self.start == "learned":
            need += ["features", "rr_greedy", "model"]
        elif self.proposer == "learned":
            need += ["features", "model"]
        return tuple(need)


def needs_of(method, cfg=None):
    return cfg.needs() if method == FIM else BASELINE_NEEDS.get(method, ("reference", "rr_sample", "cert_sample"))


# ------------------------------------------------------------------ command line shared by the scripts

def add_pipeline_args(ap):
    """Options every v2 experiment script shares (sampling, reference, certificate, search and models)."""
    g = ap.add_argument_group("pipeline")
    g.add_argument("--R", type=int, default=None, help="RR sets per root of A (default: tier / preset)")
    g.add_argument("--max-roots", type=int, default=None, help="RR roots of A (default: tier / preset; 0: all nodes)")
    g.add_argument("--rr-seed", type=int, default=7, help="RR seed of the search sample A (not a reference run seed)")
    g.add_argument("--ref-seed", type=int, default=0, help="reference run i uses RR seed ref_seed*1000 + i")
    g.add_argument("--reference-runs", type=int, default=REFERENCE_RUNS)
    g.add_argument("--reference-per-k", action="store_true", help="call reference_spread once per budget")
    g.add_argument("--cert-seed", type=int, default=CERT_SEED,
                   help="RR seed of the certificate's sample B; must differ from --rr-seed and the reference seeds")
    g.add_argument("--cert-R", type=int, default=None, help="RR sets per root of B (default: the memory rule)")
    g.add_argument("--cert-max-roots", type=int, default=None, help="RR roots of B (default: the memory rule; 0: all)")
    g.add_argument("--cert-nnz", type=float, default=CERT_NNZ_BUDGET, help="RR-set entries B may hold (memory rule)")
    g.add_argument("--n-sims", type=int, default=None, help="MC simulations (default: config.N_SIMS of the tier)")
    g.add_argument("--edge-device", choices=["auto", "cpu", "gpu"], default="auto",
                   help="message-passing edges; auto: CPU (chunked) above the preset's chunk_edges")
    g.add_argument("--preset", default=None, help="8gb / 16gb / 40gb (default: detected from the GPU)")
    g.add_argument("--time-limit", type=float, default=1800.0, help="seconds per method and query; <= 0: none")
    g.add_argument("--max-rounds", type=int, default=50, help="improve rounds of FIM-GAT v2 (per phase)")
    g.add_argument("--n-samples", type=int, default=64, help="Gumbel candidates of select_seeds_v2")
    g.add_argument("--deploy-mid", type=int, default=3,
                   help="search-path states between start and end that certified deployment also considers")
    g.add_argument("--train-seeds", nargs="+", type=int, default=list(range(N_TRAIN_SEEDS)))
    g.add_argument("--model-pattern", default=os.path.join(MODELS_V2, "constructor_s{seed}.pt"),
                   help="constructor checkpoint path, {seed} = train seed")
    g.add_argument("--proposer-pattern", default=os.path.join(MODELS_V2, "proposer_s{seed}.pt"))
    g.add_argument("--ref-file", default=os.path.join(RESULTS_V2, "references.json"),
                   help="reference store shared by every script")
    g.add_argument("--seed", type=int, default=0, help="numpy / torch / random seed")
    return ap


def check_args(ap, args):
    """Refuse a certificate sample that shares an RR seed with a sample that chose a set."""
    used = {int(args.rr_seed)} | {int(args.ref_seed) * 1000 + i for i in range(int(args.reference_runs))}
    if int(args.cert_seed) in used:
        ap.error(f"--cert-seed {args.cert_seed} is an RR seed that chose a set (search {args.rr_seed}, reference "
                 f"runs {sorted(used - {int(args.rr_seed)})}); the certificate needs an independent sample")


def time_limit(args):
    return args.time_limit if args.time_limit and args.time_limit > 0 else None


# ------------------------------------------------------------------ small helpers

def measure(fn):
    """(fn(), secs, peak GPU allocated GB, peak reserved GB): CUDA synchronised, peaks reset at the start."""
    reset_peak()
    out, secs = timed(fn)
    alloc, res = peak_gpu_gb()
    return out, secs, alloc, res


def rss_gb():
    return psutil.Process().memory_info().rss / 2**30 if psutil is not None else NAN


def is_oom(e):
    return isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)) or "out of memory" in str(e)


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def over_limit(secs, limit):
    """run_suite's rule for anytime methods: over the limit plus a grace of max(1 s, 10%)."""
    return limit is not None and secs > limit + max(1.0, 0.1 * limit)


def js(v):
    return json.dumps(v, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(tmp, path)


class CsvLog:
    """Rows of a small CSV (stage timings, lower bounds), rewritten atomically after every change (temp file, then
    os.replace). const: columns added to every row."""

    def __init__(self, path, **const):
        self.path, self.const = os.path.abspath(path), const
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.rows = []
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            try:
                self.rows = pd.read_csv(self.path, float_precision="round_trip").to_dict("records")
            except pd.errors.EmptyDataError:
                self.rows = []

    def save(self):
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

    def find(self, **key):
        return [r for r in self.rows if _matches(r, key)]

    def remove(self, **key):
        keep = [r for r in self.rows if not _matches(r, key)]
        if len(keep) != len(self.rows):
            self.rows = keep
            self.save()

    def add(self, row):
        row = {**row, **self.const, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        if "info" in row:
            row["info"] = js(row.get("info") or {})
        self.rows.append(row)
        self.save()
        return row


def tier_of(graph):
    """config.TIER for a full dataset; by size for a BFS sample (its meta copies the full graph's tier)."""
    if "sample" in graph.meta:
        return "small" if graph.n <= SMALL_MAX_N else "medium" if graph.n <= MEDIUM_MAX_N else "large"
    return graph.meta.get("tier") or TIER.get(graph.meta.get("dataset"), "large")


def rr_config(graph, preset, args):
    """(R, max_roots) of the search sample A: config.RR_PER_NODE for the tier, the memory preset for the large tier;
    CLI overrides."""
    R = RR_PER_NODE.get(tier_of(graph))
    max_roots = None
    if R is None:
        R, max_roots = int(preset["full_R"]), int(preset["full_max_roots"])
    if args.R is not None:
        R = int(args.R)
    if args.max_roots is not None:
        max_roots = int(args.max_roots) if args.max_roots > 0 else None
    return int(R), max_roots


def cert_config(graph, est, R, max_roots, args):
    """(R_B, max_roots_B, rule) of the certificate's sample B (design section 19).

    --cert-R / --cert-max-roots win. Otherwise B is as large as --cert-nnz RR-set entries allow, with the mean
    RR-set size of A as the per-set cost: first more roots (with sampled roots the units of the certificate are
    roots, so more roots buy more power than more sets per root), up to every node; then, with every node a root,
    more RR sets per root, up to config.CERT_R_MAX. B is never smaller than A."""
    if args.cert_R is not None or args.cert_max_roots is not None:
        R_B = R if args.cert_R is None else int(args.cert_R)
        mr = max_roots if args.cert_max_roots is None else (int(args.cert_max_roots) if args.cert_max_roots > 0
                                                            else None)
        return R_B, mr, "explicit"
    avg = max(est.nnz / max(est.n_sets, 1), 1.0)
    budget, n = float(args.cert_nnz), int(graph.n)
    if avg * n * R <= budget:                       # every node a root at A's R fits: add sets per root
        return int(min(CERT_R_MAX, max(R, budget // (avg * n)))), None, "rule"
    roots = max(int(budget // (avg * R)), int(est.n_roots))
    return int(R), (None if roots >= n else roots), "rule"


def mu_ref(ck, k):
    """The checkpoint's spread price for budget k (keys may come back as int or str), else 0 as V2Trainer.mu_for."""
    table = (ck or {}).get("mu_ref") or {}
    for key in (int(k), str(int(k)), float(k)):
        if key in table:
            return float(table[key])
    return 0.0


# ------------------------------------------------------------------ tuning (validation results)

def tuning_path():
    return os.path.join(RESULTS_V2, TUNING_FILE)


def load_tuning(path=None, required=True):
    """The validation results of tune_v2.py: {"delta", "phi0", "start", ...}. Missing: an error when required, else
    the untuned defaults (delta = config.DELTA_MARGIN, phi0 = config.MAX_CONDUCTANCE, the greedy start), tuned=False."""
    path = tuning_path() if path is None else path
    t = read_json(path)
    if not t:
        if required:
            raise FileNotFoundError(f"{path} missing: run experiments/v2/tune_v2.py on the validation split first "
                                    f"(or pass --untuned for a development run)")
        return {"phi0": MAX_CONDUCTANCE, "delta": DELTA_MARGIN, "start": "greedy", "tuned": False}
    t.setdefault("tuned", True)
    t.setdefault("delta", DELTA_MARGIN)
    return t


# ------------------------------------------------------------------ reference

def references(graph, ks, R, max_roots, runs, seed, n_sims, per_k=False):
    """k -> reference dict with reference_spread's keys, for every k in ks at once (see the module docstring)."""
    ks = sorted({int(k) for k in ks})
    if per_k:
        return {k: reference_spread(graph, k, runs=runs, R=R, seed=seed, n_sims=n_sims, device="cpu",
                                    max_roots=max_roots, mc_seed=REF_MC_SEED) for k in ks}
    fulls = []
    for i in range(int(runs)):
        est = ExposureEstimator(graph, R=int(R), seed=int(seed) * 1000 + i, device="cpu", max_roots=max_roots)
        fulls.append([int(v) for v in RRGreedy(est).greedy(ks[-1])])
        del est
        gc.collect()
    out = {}
    for k in ks:
        scored, run_seeds, run_spreads, run_se = {}, [], [], []
        for full in fulls:
            s = full[:k]
            key = tuple(sorted(s))
            if key not in scored:                       # identical sets give identical draws: simulate once
                sims = mc_spreads(graph, s, n_sims, REF_MC_SEED)
                se = float(sims.std(ddof=1) / np.sqrt(len(sims))) if len(sims) > 1 else NAN
                scored[key] = (float(sims.mean()), se)
            run_seeds.append(s)
            run_spreads.append(scored[key][0])
            run_se.append(scored[key][1])
        best = int(np.argmax(run_spreads))              # the first best run on ties, as reference_spread
        out[k] = {"seeds": list(run_seeds[best]), "spread": run_spreads[best], "run_spreads": run_spreads,
                  "run_seeds": run_seeds, "best_run": best, "spread_se": run_se[best]}
    return out


# ------------------------------------------------------------------ models

def load_models(args, dev, starts=(), proposers=()):
    """train seed -> {policy, ck, path, secs, alloc, reserved, proposer, proposer_ck} for the learned parts that
    `starts` / `proposers` (the configurations to run) need. Seeds whose checkpoints are missing are left out,
    with a message. {} when nothing learned is needed."""
    need_c, need_p = "learned" in starts, "learned" in proposers
    if not (need_c or need_p):
        return {}
    out = {}
    for ts in args.train_seeds:
        m = {"policy": None, "ck": {}, "path": "", "secs": 0.0, "alloc": NAN, "reserved": NAN, "proposer": None,
             "proposer_ck": None}
        if need_c:
            path = args.model_pattern.format(seed=ts)
            if not os.path.exists(path):
                print(f"  no constructor checkpoint for train seed {ts}: {path}", flush=True)
                continue
            (pol, ck), secs, alloc, res = measure(lambda: load_policy(path, dev))
            hp_seed = (ck.get("hparams") or {}).get("seed")
            if hp_seed is not None and int(hp_seed) != int(ts):
                print(f"  WARNING {path}: checkpoint trained with seed {hp_seed}, recorded as train seed {ts}")
            m.update(policy=pol, ck=ck, path=path, secs=secs, alloc=alloc, reserved=res)
        if need_p:
            ppath = args.proposer_pattern.format(seed=ts)
            if not os.path.exists(ppath):
                print(f"  no proposer checkpoint for train seed {ts}: {ppath}", flush=True)
                continue
            # UNVERIFIED: assumes a proposer checkpoint loads with train.load_policy like a V2Trainer.save() file;
            # no proposer has been trained (design section 18.3 gates it)
            (prop, pck), s2, a2, r2 = measure(lambda: load_policy(ppath, dev))
            m.update(proposer=prop, proposer_ck=pck, secs=m["secs"] + s2, path=m["path"] or ppath,
                     alloc=np.nanmax([m["alloc"], a2]), reserved=np.nanmax([m["reserved"], r2]))
        out[int(ts)] = m
    return out


def seeds_for(cfg, models):
    """Train seeds a configuration runs under: every loaded model if it has a learned part, else NO_TRAIN."""
    if not cfg.learned:
        return [NO_TRAIN]
    ok = [ts for ts, m in models.items() if (cfg.start != "learned" or m["policy"] is not None)
          and (cfg.proposer != "learned" or m["proposer"] is not None)]
    return sorted(ok)


def learned_proposer(m, ctx, rho, total_ref, delta=DELTA_MARGIN):
    return LearnedProposer(m["proposer"], ctx, rho, total_ref, delta=delta)


# ------------------------------------------------------------------ one prepared graph

class Case:
    """Cached stages of one graph (see the module docstring) and its per-budget reference.

    loader: () -> FIMGraph, timed as the stage "load". hook(stage, secs, alloc, reserved, info) receives every
    cached stage (run_t4 logs them; the other scripts print). comms: ground-truth communities for evaluation only,
    in this graph's node ids; never passed to a method. phi0: the auditor's conductance cap. delta: the RR floor
    margin of every method that searches against the floor (config.rr_floor)."""

    def __init__(self, loader, ks, args, dev, preset, need_features=False, need_rr_greedy=False, comms=None,
                 phi0=MAX_CONDUCTANCE, delta=DELTA_MARGIN, hook=None, label=None):
        self.loader, self.ks = loader, sorted({int(k) for k in ks})
        self.args, self.dev, self.preset = args, dev, preset
        self.need_features, self.need_rr_greedy = bool(need_features or need_rr_greedy), bool(need_rr_greedy)
        self.comms, self.phi0, self.delta, self.hook, self.label = comms, float(phi0), float(delta), hook, label
        self.cached_s, self.cached_info = {}, {}
        self.suite_cache, self._ref = {}, {}
        self.g = self.est = self.est_b = self.ctx = None

    # ---------------------------------------------------------- stages

    def stage(self, name, fn, **info):
        out, secs, alloc, res = measure(fn)
        self.note(name, secs, alloc, res, **info)
        return out

    def note(self, name, secs, alloc=NAN, res=NAN, **info):
        self.cached_s[name] = float(secs)
        self.cached_info[name] = info
        if self.hook is not None:
            self.hook(name, float(secs), alloc, res, info)

    def prep_for(self, method, cfg=None, model=None):
        """Seconds of the cached stages `method` uses; NaN if one was not timed in this run."""
        total = 0.0
        for s in needs_of(method, cfg):
            total += (model["secs"] if model else 0.0) if s == "model" else self.cached_s.get(s, NAN)
        return total

    def prepare(self):
        args = self.args
        g = self.stage("load", self.loader)
        self.g = g
        self.ns = int(args.n_sims) if args.n_sims else _n_sims(g)
        self.R, self.max_roots = rr_config(g, self.preset, args)

        # reference first: its estimators live on the CPU and are freed before the main one is built
        rkey = (f"{g.name}|{g.meta.get('attribute')}|n={g.n}|m={g.m}|R={self.R}|max_roots={self.max_roots}"
                f"|runs={args.reference_runs}|seed={args.ref_seed}|n_sims={self.ns}|mc={REF_MC_SEED}"
                f"|per_k={bool(args.reference_per_k)}")
        store = read_json(args.ref_file)
        entry = store.get(rkey)
        refs = {int(k): v for k, v in entry["refs"].items()} if entry else {}
        if entry and all(k in refs for k in self.ks):
            self.note("reference", float(entry["secs"]), ks=sorted(refs), from_file=True)
        else:
            ks_all = sorted(set(self.ks) | set(refs))
            refs = self.stage("reference", lambda: references(g, ks_all, self.R, self.max_roots, args.reference_runs,
                                                              args.ref_seed, self.ns, args.reference_per_k),
                              ks=ks_all, runs=args.reference_runs, n_sims=self.ns)
            store = read_json(args.ref_file)
            store[rkey] = {"secs": self.cached_s["reference"], "refs": {str(k): v for k, v in refs.items()}}
            write_json(args.ref_file, store)
        self.refs = refs
        free()

        est = self.stage("rr_sample", lambda: ExposureEstimator(g, R=self.R, seed=args.rr_seed, device=self.dev,
                                                                max_roots=self.max_roots),
                         R=self.R, max_roots=self.max_roots, rr_seed=args.rr_seed)
        self.est = est
        self.R_B, self.max_roots_B, rule = cert_config(g, est, self.R, self.max_roots, args)
        n_roots_B = len(stratified_roots(g, self.max_roots_B, args.cert_seed))
        self.est_b = self.stage("cert_sample", lambda: ExposureEstimator(g, R=self.R_B, seed=args.cert_seed,
                                                                         device="cpu", max_roots=self.max_roots_B),
                                R=self.R_B, max_roots=self.max_roots_B, rule=rule, roots=n_roots_B,
                                cert_seed=args.cert_seed)
        if args.edge_device == "auto":
            edge = "cpu" if 2 * g.m + g.n > self.preset["chunk_edges"] else self.dev
        else:
            edge = "cpu" if args.edge_device == "cpu" else self.dev
        self.edge_device = str(edge)
        self.greedy_total = {}
        if self.need_features:
            ctx = self.stage("features", lambda: build_context(g, est, self.dev, edge_device=edge))
            ctx.chunk_edges = int(self.preset["chunk_edges"])
            if self.need_rr_greedy:
                ks = self.ks

                def greedy():
                    full = RRGreedy(est).greedy(max(ks))
                    seeds = {k: [int(v) for v in full[:k]] for k in ks}
                    return seeds, {k: exact_rates(est, s)[1] for k, s in seeds.items()}
                ctx.greedy_seeds, self.greedy_total = self.stage("rr_greedy", greedy, k_max=max(ks))
            free()
        else:
            ctx = types.SimpleNamespace(graph=g, est=est, device=torch.device(self.dev), total_ref={})
        self.ctx = ctx
        self.stage("cover_index", lambda: cover_index(est))
        return self

    def describe(self):
        g, a, b = self.g, self.est, self.est_b
        return (f"{g.name}: n={g.n:,} m={g.m:,} groups={int(g.present_groups.sum())}; A: R={self.R} "
                f"roots={a.n_roots:,} nnz={a.nnz:,}; B: R={self.R_B} roots={b.n_roots:,} nnz={b.nnz:,}")

    def base_cols(self):
        """Columns every row of this graph carries (sample sizes, the certificate sample, phi0)."""
        a, b = self.est, self.est_b
        return {"R": self.R, "max_roots": NAN if self.max_roots is None else self.max_roots, "n_roots": a.n_roots,
                "nnz": a.nnz, "cert_R": self.R_B, "cert_roots": b.n_roots, "cert_nnz": b.nnz, "phi0": self.phi0,
                "delta": self.delta,
                "edge_device": self.edge_device}

    def ref_for(self, k):
        """(reference seeds, total_ref): the exact RR total of the reference on A, the base of the floor (also set as
        ctx.total_ref[k], so FIM-GAT's features see the floor slack with its training meaning)."""
        k = int(k)
        if k not in self._ref:
            ref = check_seeds(self.refs[k]["seeds"], k, self.g.n)
            self._ref[k] = (ref, float(exact_rates(self.est, ref)[1]))
        ref, total = self._ref[k]
        self.ctx.total_ref[k] = total
        return ref, total

    def floor_for(self, k, rho):
        """The RR floor of query (k, rho): config.rr_floor(rho, total_ref, delta)."""
        return rr_floor(rho, self.ref_for(k)[1], self.delta)

    def judge(self, seeds, k, rho):
        """The equal-spread certificate every set gets (design section 18.2), on B."""
        ref, _ = self.ref_for(k)
        return certified_equal_spread(self.est_b, seeds, ref, float(rho), alpha=ALPHA)


def _n_sims(graph):
    from fimgat.v2.config import N_SIMS
    return int(N_SIMS.get(tier_of(graph), 2000))


def cert_cols(c):
    """Results columns of a certified_equal_spread result (relative values: divided by the reference's spread
    estimate on B)."""
    if c is None:
        return {}
    return dict(cert_passed=bool(c["passed"]), cert_identical=bool(c["identical"]), cert_lcb_rel=float(c["lcb_rel"]),
                cert_estimate_rel=float(c["estimate_rel"]), cert_halfwidth_rel=float(c["halfwidth_rel"]),
                cert_alpha=ALPHA)


# ------------------------------------------------------------------ queries

def search_paths(variant, est, start, floor, prop, kw):
    """Run one variant's search. Returns (final seeds, histories (one per phase), seed-set paths, extra info)."""
    if variant == "v2-b":
        seeds, info = hive_search(est, list(start), floor, proposer=prop, **kw)
        p1 = path_states(info["start1"], info["phase1"])
        extra = {"kappa": info["kappa"], "disp_a": info["disp_a"], "se_a": info["se_a"],
                 "phase1_rounds": len(info["phase1"]) - 1, "phase2_rounds": len(info["phase2"]) - 1,
                 "_communities": info["communities"]}    # "_" keys stay out of the results rows (interpret_v2 reads them)
        return seeds, [info["phase1"], info["phase2"]], [p1, path_states(p1[-1], info["phase2"])], extra
    seeds, hist = improve(est, list(start), floor, proposer=prop, audit=variant == "v2-b0", **kw)
    start1 = kw["reference"] if hist[0]["fallback"] else list(start)
    return seeds, [hist], [path_states(start1, hist)], {}


def fim_query(case, k, rho, cfg, model, limit=None):
    """One FIM-GAT v2 query. Returns a dict: deployed and final seed sets, their judges, the stage timings and peaks,
    and what the start, search and deployment did."""
    args, est, ctx = case.args, case.est, case.ctx
    ref, total_ref = case.ref_for(k)
    floor = case.floor_for(k, rho)
    if cfg.start == "learned":
        rounds, mu = int(model["ck"].get("rounds", 3)), mu_ref(model["ck"], k)
        (start, info), s_sel, a1, r1 = measure(lambda: select_seeds_v2(
            model["policy"], ctx, k, rho, floor, rounds, n_samples=args.n_samples, mu=mu, return_info=True))
        sel = {"n_candidates": info.get("n_candidates"), "select_fallback": bool(info.get("fallback")), "mu": mu}
    else:                                           # the greedy start: the reference set, nothing to compute
        start, s_sel, a1, r1, sel = list(ref), 0.0, NAN, NAN, {}
    left = None if limit is None else max(0.0, limit - s_sel)

    def run_search():
        prop = learned_proposer(model, ctx, rho, total_ref, case.delta) if cfg.proposer == "learned" else cfg.proposer
        kw = dict(max_rounds=args.max_rounds, max_seconds=left, reference=list(ref), max_conductance=case.phi0,
                  audit_max_size=AUDIT_MAX_SIZE)
        return search_paths(cfg.variant, est, start, floor, prop, kw)
    (seeds, hists, paths, sinfo), s_imp, a2, r2 = measure(run_search)
    final = sorted(int(v) for v in seeds)

    def deploy():                                   # Theorem 2 over the search path, on sample B: one set of
        cands = deploy_candidates(paths, n_mid=args.deploy_mid)     # certificates, one choice per rule
        cert = certify_sets(case.est_b, cands, ref, float(rho), alpha=CERT_ALPHA)
        return cands, cert, {dep: choose(cert["certificates"], rule) for dep, rule in DEPLOYMENTS.items()}
    (cands, cert, picks), s_ver, a3, r3 = measure(deploy)
    judges, deployed, deploy_info = {}, {}, {}
    for dep, i in picks.items():
        c = cert["certificates"][i]
        deployed[dep] = [int(v) for v in c["seeds"]]
        key = tuple(deployed[dep])
        # the baselines' judge; implied by the choice's union bound (identical sets share one judgement)
        judges[dep] = next((judges[d] for d in judges if tuple(deployed[d]) == key), None) or \
            case.judge(deployed[dep], k, rho)
        deploy_info[dep] = {"rule": DEPLOYMENTS[dep], "n_candidates": len(cands),
                            "index": int(i) if i < cert["n_candidates"] else -1, "position": int(c["position"]),
                            "is_reference": bool(c["is_reference"]), "disp_ucb": float(c["disp_ucb"]),
                            "disp_se": float(c["disp_se"]), "floor_lcb": float(c["floor_lcb"]),
                            "delta_each": float(cert["delta_each"]),
                            "n_eligible": int(sum(bool(x["eligible"]) for x in cert["certificates"]))}
    same = next((d for d in deployed if deployed[d] == final), None)
    judge_final = judges[same] if same is not None else case.judge(final, k, rho)
    return {"start": [int(v) for v in start], "final": final, "deployed": deployed["certified"],
            "judge": judges["certified"], "deployed_path": deployed["certified-path"],
            "judge_path": judges["certified-path"], "judge_final": judge_final, "floor": floor,
            "total_ref": total_ref, "ref": ref, "select_s": s_sel, "improve_s": s_imp, "verify_s": s_ver,
            "peak_alloc_gb": np.nanmax([a1, a2, a3]), "peak_reserved_gb": np.nanmax([r1, r2, r3]),
            "peaks": {"select": (a1, r1), "improve": (a2, r2), "verify": (a3, r3)},
            "rounds": int(sum(len(h) - 1 for h in hists)),
            "accepted": int(sum(bool(x.get("accepted")) for h in hists for x in h)),
            "restarted": bool(hists[0][0]["fallback"]), "start_disp": float(hists[0][0]["disp"]),
            "rr_disp": float(hists[-1][-1]["disp"]), "rr_total": float(hists[-1][-1]["total"]),
            "family_size": hists[-1][-1].get("family_size"), "max_seconds": left, "sel": sel, "search": sinfo,
            "deploy": deploy_info["certified"], "deploy_path": deploy_info["certified-path"]}


def baseline_query(case, k, rho, name, limit=None):
    """One baseline query: run_suite's method, then the judge. Returns a dict (seeds None when the method gave no
    valid set; reason says why)."""
    ref, total_ref = case.ref_for(k)
    floor = case.floor_for(k, rho)
    if name not in SUITE:
        raise ValueError(f"unknown baseline {name!r}; the suite is {SUITE}")
    suite, wall, a1, r1 = measure(lambda: run_suite(case.ctx, k, rho, floor, ref, methods=[name], time_limit=limit,
                                                    cache=case.suite_cache))
    seeds, secs = suite[name]
    info = dict(suite.info.get(name, {}))
    if "index_s" in info and "suite_index" not in case.cached_s:
        case.note("suite_index", float(info["index_s"]), built_in=name)
    out = {"seeds": None if seeds is None else [int(v) for v in seeds], "secs": float(secs), "wall": wall,
           "info": info, "reason": suite.reason.get(name, ""), "floor": floor, "total_ref": total_ref, "ref": ref,
           "judge": None, "verify_s": 0.0, "peak_alloc_gb": a1, "peak_reserved_gb": r1,
           "peaks": {"select": (a1, r1)}}
    if seeds is not None:
        judge, s_ver, a2, r2 = measure(lambda: case.judge(seeds, k, rho))
        out.update(judge=judge, verify_s=s_ver, peak_alloc_gb=np.nanmax([a1, a2]),
                   peak_reserved_gb=np.nanmax([r1, r2]))
        out["peaks"]["verify"] = (a2, r2)
    return out


# ------------------------------------------------------------------ planning and recording

def plan_queries(rec, ident, ks, rhos, cfgs, models, methods, tuned):
    """Pending queries (k, rho, method, cfg, train_seed) of one graph, FIM-GAT first. ident: {"dataset",
    "attribute", "graph"}; tuned: {"delta", "phi0"} in force (rows made under other values are not done)."""
    todo = []
    for k in ks:
        for rho in rhos:
            for cfg in cfgs:
                for ts in seeds_for(cfg, models):
                    if not rec.done(**fim_key(ident, k, rho, cfg, ts, tuned=tuned)):
                        todo.append((k, rho, FIM, cfg, ts))
            for m in methods:
                if not rec.done(**baseline_key(ident, k, rho, m, tuned=tuned), train_seed=NAN):
                    todo.append((k, rho, m, None, NAN))
    return todo



def fim_key(ident, k, rho, cfg, ts, deployment="certified", tuned=None):
    """Resume key of a FIM-GAT row. ident: {"dataset", "attribute", "graph"} of the graph (known before loading);
    tuned: {"delta", "phi0"} in force, so rows made under other tuned values never count as done."""
    return dict(**ident, k=int(k), rho=float(rho), method=FIM, train_seed=ts, deployment=deployment, **cfg.cols(),
                **(tuned or {}))


def baseline_key(ident, k, rho, name, tuned=None):
    """Resume key of a baseline row: every tuned value in force except phi0, which only FIM-GAT uses."""
    return dict(**ident, k=int(k), rho=float(rho), method=name, variant="default",
                **{a: v for a, v in (tuned or {}).items() if a != "phi0"})


def record_fim(rec, case, k, rho, cfg, ts, res, model=None, limit=None, extra=None):
    """Three rows per FIM-GAT query: the set deployed by Theorem 2's rule (deployment="certified", T1's), the set
    deployed by the path rule (deployment="certified-path", T2's; design section 21) and the final search set
    (deployment="final", the ablation without certified deployment; its query_s has no deploy step). Both deployed
    rows share one certification step (verify_s). Over the time limit all are recorded missing, with the late sets
    kept. Returns the rows in that order."""
    g = case.g
    secs = res["select_s"] + res["improve_s"]
    common = dict(**cfg.cols(), **case.base_cols(), select_s=res["select_s"], improve_s=res["improve_s"],
                  floor=res["floor"], total_ref_rr=res["total_ref"], rr_total=res["rr_total"], rr_disp=res["rr_disp"],
                  start_seeds=res["start"], improve_rounds=res["rounds"], improve_accepted=res["accepted"],
                  improve_fallback=res["restarted"], family_size=res["family_size"],
                  peak_alloc_gb=res["peak_alloc_gb"], peak_reserved_gb=res["peak_reserved_gb"],
                  model=os.path.basename(model["path"]) if model and model.get("path") else "",
                  time_limit=NAN if limit is None else limit, prep_stages=list(cfg.needs()),
                  **{f"search_{a}": v for a, v in res["search"].items() if not a.startswith("_")}, **res["sel"],
                  **(extra or {}))
    prep = case.prep_for(FIM, cfg, model)
    rows = []
    for dep, seeds, judge, vs in (("certified", res["deployed"], res["judge"], res["verify_s"]),
                                  ("certified-path", res["deployed_path"], res["judge_path"], res["verify_s"]),
                                  ("final", res["final"], res["judge_final"], 0.0)):
        cols = dict(common, deployment=dep, verify_s=vs, query_s=secs + vs, **cert_cols(judge))
        if dep != "final":
            info = res["deploy"] if dep == "certified" else res["deploy_path"]
            cols.update({f"deploy_{a}": v for a, v in info.items()}, deployed_is_final=seeds == res["final"])
        if over_limit(secs, limit):
            rows.append(rec.add_missing(g, k, rho, FIM, f"time_limit: took {secs:.1f}s > {limit:g}s", secs,
                                        train_seed=ts, prep_s=prep, late_seeds=seeds, **cols))
        else:
            rows.append(rec.add(g, k, rho, FIM, seeds, secs, ref_seeds=res["ref"], train_seed=ts, prep_s=prep,
                                communities=case.comms, **cols))
    return rows


def record_baseline(rec, case, k, rho, name, res, limit=None, extra=None):
    g = case.g
    cols = dict(**case.base_cols(), select_s=res["secs"], verify_s=res["verify_s"],
                query_s=res["secs"] + res["verify_s"], floor=res["floor"], total_ref_rr=res["total_ref"],
                peak_alloc_gb=res["peak_alloc_gb"], peak_reserved_gb=res["peak_reserved_gb"],
                prep_stages=list(needs_of(name)), time_limit=NAN if limit is None else limit,
                info={**res["info"], "suite_wall_s": res["wall"]}, **cert_cols(res["judge"]), **(extra or {}))
    prep = case.prep_for(name)
    if res["seeds"] is None:
        return rec.add_missing(g, k, rho, name, res["reason"] or "no seed set", res["secs"], prep_s=prep, **cols)
    return rec.add(g, k, rho, name, res["seeds"], res["secs"], ref_seeds=res["ref"], prep_s=prep,
                   communities=case.comms, **cols)


def fmt_fim(row, cfg, ts, res):
    from common_v2 import fmt_row

    def which(info, seeds):
        return "reference" if info["is_reference"] else "final" if seeds == res["final"] else "path"
    dep = which(res["deploy"], res["deployed"])
    dep_p = which(res["deploy_path"], res["deployed_path"])
    cert = "pass" if res["judge"]["passed"] else "FAIL"
    return (f"{fmt_row(row)}  [{cfg.name()} s{ts}: select {res['select_s']:.1f}s search {res['improve_s']:.1f}s "
            f"({res['rounds']} rounds) certify {res['verify_s']:.1f}s, deployed the {dep} set (path rule: the "
            f"{dep_p} set), certificate {cert}]")


def fmt_baseline(row, res):
    from common_v2 import fmt_row
    if res["judge"] is None:
        return fmt_row(row)
    return f"{fmt_row(row)}  [certificate {'pass' if res['judge']['passed'] else 'FAIL'}]"
