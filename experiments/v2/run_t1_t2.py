"""Tests T1 (claim C1) and T2 (claim C2) of docs/design_v2.md (sections 10, 18 and 19).

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore run_t1_t2.py --test t1 --dry-run     # plan and rough estimate only
    PYTHONIOENCODING=utf-8 python -u -W ignore run_t1_t2.py --test t1
    PYTHONIOENCODING=utf-8 python -u -W ignore run_t1_t2.py --test t2
    PYTHONIOENCODING=utf-8 python -u -W ignore run_t1_t2.py --test t1 --datasets rice_facebook --ks 25

Both read the validation results of tune_v2.py (results/v2/tuning.json): the RR floor margin delta, the auditor's
conductance cap phi0 and the start set of the method (the constructor's set if it beat the greedy start on
validation, else the reference set).
--untuned runs on config defaults instead, for development only; such rows carry tuned=False.

T1 (C1): every test graph (datasets.splits()["test"]) under every group attribute in T1_ATTRS, every budget, every
floor in config.RHO_REPORT (0.99 is the headline). FIM-GAT runs
  the method      variant v2-b (improve.hive_search), the tuned start, the simple hybrid, certified deployment;
  ablation rows   the other start, the exhaustive proposer, variant v2-a, variant v2-b0 (design section 18.2),
and every FIM-GAT query also records its final search set (deployment "final", the row without certified
deployment). Baselines: run_suite's methods. Every set is judged by the same certificate on the sample B
(certify.certified_equal_spread, cert_* columns); the Monte Carlo equal-spread test stays in the es_* columns.
Disparity lower bounds (bounds.disparity_lower_bound, Proposition W2.3) go to t1_bounds.csv when affordable
(config.LB_MAX_WORK); the floor they use is rho times the reference's Monte Carlo spread minus 2 standard errors.

T2 (C2): BFS samples of snap_youtube (--t2-samples of --t2-size nodes, sample seeds 1, 2, ...), every budget, rho
0.97 (primary) and 0.99; FIM-GAT's v2-b against v2-a, with v2-b0 as well, all from the method's start. Metric: the
disparity over snap_youtube's ground-truth communities (--t2-communities, default every user-created group) with at
least config.EVAL_MIN_COMMUNITY members in the sample: evaluation only, no method sees them. The held-out-attribute
secondary metric of T2 comes from T1's rows (Pokec, Rice). The samples' search sample A has config.T2_RR_PER_NODE
RR sets per node (the R of the W1 evidence; design section 21); --R overrides it, which is a reported deviation, and
R is part of T2's resume key.

Pareto (T1, design section 21): once per graph and budget, BSM's whole trade-off (external.bsm_sweep, every tau, no
floor) goes to t1_pareto.csv with its Monte Carlo spread ratio and disparity and its certificate on B at every floor
of config.RHO_REPORT; --no-pareto skips it. summarize_v2.py sets every method's sets against the method's curve.

Outputs (results/v2/): t1.csv or t2.csv (RecorderV2 rows; FIM-GAT gives three per query, deployment "certified",
"certified-path" and "final"), t1_bounds.csv, t1_pareto.csv. Resumable: a query is skipped when its row exists. One
GPU job at a time; the script prints its plan and a rough estimate before running, and --dry-run stops there.

Parallel jobs (2026-10-15, docs/cloud_runbook.md): --job NAME puts every output of this run, and its reference
store, under results/v2/parts/NAME/ (t1.csv or t2.csv, t1_bounds.csv, t1_pareto.csv, references.json) unless a path
is given explicitly, so separate sessions write separate files and their branches merge without conflicts;
summarize_v2.py reads results/v2/<file> and every results/v2/parts/*/<file>. --t2-seeds runs chosen T2 samples
(default 1 .. --t2-samples), so the samples can be split over jobs; a sample's seed fixes it, wherever it runs.
"""
import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pipeline_v2 as P                                                   # noqa: E402
from common_v2 import NAN, RecorderV2, code_tag, device, gpu_name, preset_config   # noqa: E402
from fimgat.v2 import datasets as D                                       # noqa: E402
from fimgat.baselines import RRGreedy                                     # noqa: E402
from fimgat.v2.baselines_v2 import SUITE                                  # noqa: E402
from fimgat.v2.config import (BUDGETS, EVAL_MIN_COMMUNITY, LB_MAX_WORK, RESULTS_V2, RHO_C1, RHO_C2,   # noqa: E402
                              RHO_REPORT, T2_RR_PER_NODE, T2_SAMPLE_SIZE, T2_SAMPLES, TIER)
from fimgat.v2.external import TAUS                                       # noqa: E402

# group attributes T1 runs per test dataset: every real attribute, and degree groups for the structural graphs
T1_ATTRS = {"rice_facebook": ("college", "age", "major"), "twitter_political": ("leaning",),
            "pokec_n": ("region", "gender", "age"), "pokec_full": ("region", "gender", "age"),
            "instagram": ("gender",), "epinions": ("degree",), "twitter_ego": ("degree",),
            "snap_youtube": ("degree",), "linkedin": ("degree",)}
T2_DATASET = "snap_youtube"
# UNVERIFIED rough seconds per query by tier (FIM-GAT, baseline), for the printed estimate only
GUESS = {"small": (5.0, 2.0), "medium": (60.0, 20.0), "large": (900.0, 300.0)}


@dataclass
class Job:
    ident: dict                                     # dataset, attribute, graph (known before loading)
    tier: str
    loader: Callable
    comms: Optional[Callable] = None                # graph -> evaluation communities in its node ids


# ------------------------------------------------------------------ evaluation communities

_GT = {}


def gt_communities(which):
    if which not in _GT:
        _GT[which] = D.ground_truth_communities(T2_DATASET, which)
    return _GT[which]


def restrict(comms, full_ids, min_size):
    """Communities given in full-graph ids, as node ids of a graph whose node i is full node full_ids[i] (sorted);
    only those with at least min_size members inside."""
    full_ids = np.asarray(full_ids, dtype=np.int64)
    out = []
    for c in comms:
        idx = np.searchsorted(full_ids, c)
        ok = idx < len(full_ids)
        ok[ok] = full_ids[idx[ok]] == np.asarray(c)[ok]
        if int(ok.sum()) >= int(min_size):
            out.append(np.sort(idx[ok]))
    return out


# ------------------------------------------------------------------ jobs and plans

def t1_jobs(args):
    names = args.datasets or [x for x in D.splits()["test"] if D.available(x)]
    for name in names:
        for attr in (args.attributes or T1_ATTRS.get(name, (D.DEFAULT_ATTRIBUTE[name],))):
            comms = None
            if name == T2_DATASET:
                comms = (lambda g, w=args.t2_communities: restrict(gt_communities(w), np.arange(g.n), EVAL_MIN_COMMUNITY))
            yield Job({"dataset": name, "attribute": attr, "graph": name}, TIER.get(name, "large"),
                      (lambda name=name, attr=attr: D.load(name, attr)), comms)


def t2_jobs(args):
    full = {}

    def load_full():
        if "g" not in full:
            full["g"] = D.load(T2_DATASET, "degree")
        return full["g"]

    def loader(seed):
        def f():
            g = D.sample(T2_DATASET, args.t2_size, seed, attribute="degree", full=load_full())
            g.meta["tier"] = P.tier_of(g)            # a sample's tier is its size's, for R and the MC budget
            return g
        return f
    tier = "small" if args.t2_size <= P.SMALL_MAX_N else "medium" if args.t2_size <= P.MEDIUM_MAX_N else "large"
    for seed in (args.t2_seeds or range(1, int(args.t2_samples) + 1)):
        name = f"{T2_DATASET}_bfs{int(args.t2_size)}_s{seed}"
        yield Job({"dataset": T2_DATASET, "attribute": "degree", "graph": name}, tier, loader(seed),
                  lambda g, w=args.t2_communities: restrict(gt_communities(w), g.meta["full_ids"], EVAL_MIN_COMMUNITY))


def configs(test, tuning, which):
    """FIM-GAT configurations: the method first, then the ablation rows (T1) or T2's comparison."""
    start = tuning["start"]
    method = P.FimConfig("v2-b", start, "simple")
    if test == "t2":
        return [method, P.FimConfig("v2-a", start, "simple"), P.FimConfig("v2-b0", start, "simple")]
    out = [method] if "method" in which else []
    if "ablation" in which:
        other = "greedy" if start == "learned" else "learned"
        out += [P.FimConfig("v2-b", other, "simple"), P.FimConfig("v2-b", start, "exhaustive"),
                P.FimConfig("v2-a", start, "simple"), P.FimConfig("v2-b0", start, "simple")]
    return out


# ------------------------------------------------------------------ lower bounds (T1)

def lower_bound(bounds, case, k, rho):
    """bounds.disparity_lower_bound for (graph, k, rho), once, when affordable; a row either way."""
    g = case.g
    key = {"dataset": g.meta.get("dataset"), "attribute": g.meta.get("attribute"), "graph": g.name, "k": int(k),
           "rho": float(rho)}
    if bounds.find(**key):
        return
    from fimgat.v2.bounds import disparity_lower_bound
    ref = case.refs[int(k)]
    floor_total = float(rho) * max(0.0, float(ref["spread"]) - 2.0 * float(ref.get("spread_se") or 0.0))
    work = int(g.present_groups.sum()) * (int(k) + 1) * float(case.est.nnz)
    row = dict(key, floor_total=floor_total, work=work, lb=NAN, mean_floor=NAN, maximin_ub=NAN, n_groups=NAN,
               secs=NAN, status="ok", reason="")
    if work > LB_MAX_WORK:
        row.update(status="skipped", reason=f"work {work:.2g} > LB_MAX_WORK {LB_MAX_WORK:.2g}")
    else:
        t0 = time.perf_counter()
        out = disparity_lower_bound(case.est, int(k), floor_total)
        row.update(lb=out["lb"], mean_floor=out["mean_floor"], maximin_ub=out["maximin_ub"], n_groups=out["n_groups"],
                   secs=time.perf_counter() - t0)
    bounds.add(row)


# ------------------------------------------------------------------ BSM's whole trade-off (T1, Pareto)

def pareto_ident(ident):
    return {"dataset": ident["dataset"], "attribute": ident["attribute"], "graph": ident["graph"]}


def pareto_pending(prec, ident, ks):
    """Budgets whose BSM sweep is not yet complete in t1_pareto.csv."""
    return [k for k in ks if not all(prec.done(**pareto_ident(ident), k=int(k), rho=RHO_C1, method="BSM",
                                               variant=f"tau={t:g}") for t in TAUS)]


def pareto_sweep(prec, case, ident, k, extra):
    """BSM's whole trade-off at budget k (external.bsm_sweep: every tau, no floor), once per graph and budget, into
    t1_pareto.csv (design section 21). Each set is a RecorderV2 row against the reference (rho = RHO_C1 only fixes
    es_passed; es_ratio and the disparities do not depend on it), with tau, its spread ratio on B (b_ratio) and its
    certificate on B at every floor of RHO_REPORT (cert_passed_<rho>)."""
    from fimgat.v2.certify import certified_equal_spread
    from fimgat.v2.external import bsm_sweep
    g, key = case.g, pareto_ident(ident)
    ref, _ = case.ref_for(k)
    hit = case.suite_cache.get("index")
    if hit is None or hit[0] is None:                   # run_suite's RR index, shared with BSM when it ran
        t0 = time.perf_counter()
        case.suite_cache.setdefault("_est", case.est)
        hit = case.suite_cache["index"] = (RRGreedy(case.est), time.perf_counter() - t0, None)
    t0 = time.perf_counter()
    sweep = bsm_sweep(case.est, k, rg=hit[0], cache=case.suite_cache)
    secs = time.perf_counter() - t0
    for tau, seeds in sweep:
        variant = f"tau={tau:g}"
        if prec.done(**key, k=int(k), rho=RHO_C1, method="BSM", variant=variant):
            continue
        cert = {f"cert_passed_{r:g}": bool(certified_equal_spread(case.est_b, seeds, ref, r, alpha=P.ALPHA)["passed"])
                for r in RHO_REPORT}
        b_ratio = float(certified_equal_spread(case.est_b, seeds, ref, 0.0, alpha=P.ALPHA)["estimate_rel"])
        prec.add(g, k, RHO_C1, "BSM", seeds, secs, ref_seeds=ref, variant=variant, communities=case.comms, tau=tau,
                 b_ratio=b_ratio, sweep_s=secs, **cert, **case.base_cols(), **extra)
    print(f"    BSM sweep k={k}: {len(sweep)} sets in {secs:.1f}s -> t1_pareto.csv", flush=True)


# ------------------------------------------------------------------ outputs

def resolve_outputs(args):
    """Fill --out, --bounds-out, --pareto-out and --ref-file: an explicit path wins; else, with --job NAME, the file
    under results/v2/parts/NAME/; else the shared default under results/v2/."""
    base = os.path.join(RESULTS_V2, "parts", args.job) if args.job else RESULTS_V2
    args.out = args.out or os.path.join(base, f"{args.test}.csv")
    args.bounds_out = args.bounds_out or os.path.join(base, "t1_bounds.csv")
    args.pareto_out = args.pareto_out or os.path.join(base, "t1_pareto.csv")
    args.ref_file = args.ref_file or os.path.join(base, "references.json")
    return args


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", choices=["t1", "t2"], required=True)
    ap.add_argument("--datasets", nargs="+", default=None, help="T1: default every built test dataset")
    ap.add_argument("--attributes", nargs="+", default=None, help="T1: default T1_ATTRS of each dataset")
    ap.add_argument("--ks", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--rhos", nargs="+", type=float, default=None,
                    help=f"default: T1 {RHO_REPORT}; T2 ({RHO_C2}, {RHO_C1})")
    ap.add_argument("--configs", nargs="+", choices=["method", "ablation"], default=["method", "ablation"],
                    help="T1 FIM-GAT configurations (T2 always runs v2-b, v2-a and v2-b0)")
    ap.add_argument("--methods", nargs="*", default=None, help="baselines (default: T1 the whole suite; T2 none)")
    ap.add_argument("--no-fim", action="store_true", help="baselines only")
    ap.add_argument("--no-bounds", action="store_true", help="T1: skip the disparity lower bounds")
    ap.add_argument("--t2-samples", type=int, default=T2_SAMPLES)
    ap.add_argument("--t2-seeds", nargs="+", type=int, default=None,
                    help="T2: the sample seeds to run (default 1 .. --t2-samples); lets jobs split the samples")
    ap.add_argument("--t2-size", type=int, default=T2_SAMPLE_SIZE)
    ap.add_argument("--t2-communities", choices=["all", "top5000"], default="all")
    ap.add_argument("--tuning", default=None, help="tune_v2.py's output (default results/v2/tuning.json)")
    ap.add_argument("--untuned", action="store_true", help="development only: config defaults, rows tuned=False")
    ap.add_argument("--job", default=None,
                    help="write every output (and the reference store) under results/v2/parts/JOB/")
    ap.add_argument("--out", default=None, help="default results/v2/t1.csv or t2.csv (--job: under parts/JOB)")
    ap.add_argument("--bounds-out", default=None, help="default results/v2/t1_bounds.csv (--job: under parts/JOB)")
    ap.add_argument("--no-pareto", action="store_true", help="T1: skip BSM's tau sweep (t1_pareto.csv)")
    ap.add_argument("--pareto-out", default=None, help="default results/v2/t1_pareto.csv (--job: under parts/JOB)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimate, then stop")
    P.add_pipeline_args(ap)
    ap.set_defaults(ref_file=None)                  # resolve_outputs: the job's store, else results/v2/references.json
    args = resolve_outputs(ap.parse_args())
    P.check_args(ap, args)
    if args.test == "t2" and args.R is None:
        args.R = T2_RR_PER_NODE                     # design section 21; another --R is a reported deviation

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    methods = (list(SUITE) if args.test == "t1" else []) if args.methods is None else list(args.methods)
    bad = [m for m in methods if m not in SUITE]
    if bad:
        ap.error(f"unknown methods {bad}; the suite is {SUITE}")
    rhos = sorted({float(r) for r in (args.rhos or (RHO_REPORT if args.test == "t1" else (RHO_C2, RHO_C1)))})
    ks = sorted({int(k) for k in args.ks})
    tuning = P.load_tuning(args.tuning, required=not args.untuned)
    if args.untuned:
        tuning = dict(P.load_tuning(args.tuning, required=False), tuned=False)
    cfgs = [] if args.no_fim else configs(args.test, tuning, args.configs)
    out = args.out or os.path.join(RESULTS_V2, f"{args.test}.csv")

    dev = device()
    pname, preset = preset_config(args.preset)
    print(f"{args.test.upper()}: device={dev} gpu={gpu_name()} preset={pname} code_tag={code_tag()}", flush=True)
    tuned = {"delta": float(tuning["delta"]), "phi0": float(tuning["phi0"])}
    if args.cert_R is not None:                     # the certificate-size sensitivity run: one key per size of B
        tuned["cert_R"] = int(args.cert_R)
    if args.test == "t2":                           # rows made at another R of A never count as done
        tuned["R"] = int(args.R)
        if int(args.R) != T2_RR_PER_NODE:
            print(f"DEVIATION: T2 runs with R={args.R} RR sets per node, not the pre-registered {T2_RR_PER_NODE} "
                  f"(design section 21); report it", flush=True)
    print(f"tuning: delta={tuned['delta']} phi0={tuned['phi0']} start={tuning['start']} "
          f"tuned={tuning.get('tuned', True)}", flush=True)
    print("FIM-GAT configurations: " + (", ".join(c.name() for c in cfgs) or "none"), flush=True)
    print("One GPU job at a time: nothing else may use the GPU while this runs.", flush=True)
    models = P.load_models(args, dev, starts=[c.start for c in cfgs], proposers=[c.proposer for c in cfgs])
    dropped = [c.name() for c in cfgs if c.learned and not P.seeds_for(c, models)]
    if dropped:
        print(f"note: no checkpoints for {dropped}; those configurations are not run", flush=True)
    cfgs = [c for c in cfgs if not c.learned or P.seeds_for(c, models)]

    rec = RecorderV2(out, n_sims=args.n_sims, preset=pname)
    prec = (RecorderV2(args.pareto_out, n_sims=args.n_sims, preset=pname)
            if args.test == "t1" and not args.no_pareto else None)
    jobs, total = [], 0.0
    for job in (t1_jobs(args) if args.test == "t1" else t2_jobs(args)):
        if job.ident["dataset"] not in D.LOADABLE or not D.available(job.ident["dataset"]):
            print(f"{job.ident['dataset']}: not built; skipped (experiments/v2/build_datasets.py)")
            continue
        todo = P.plan_queries(rec, job.ident, ks, rhos, cfgs, models, methods, tuned)
        sweep = pareto_pending(prec, job.ident, ks) if prec is not None else []
        if not todo and not sweep:
            print(f"{job.ident['graph']} / {job.ident['attribute']}: every query already in {out}")
            continue
        f, b = GUESS.get(job.tier, GUESS["large"])
        guess = (sum(f if q[2] == P.FIM else b for q in todo) + b * len(sweep)) / 3600.0
        total += guess
        n_fim = sum(q[2] == P.FIM for q in todo)
        print(f"{job.ident['graph']} / {job.ident['attribute']} ({job.tier}): {len(todo)} queries ({n_fim} FIM-GAT, "
              f"{len(todo) - n_fim} baseline)" + (f", BSM sweep at {len(sweep)} budgets" if sweep else "")
              + f"; rough estimate {guess:.1f} h", flush=True)
        jobs.append((job, todo, sweep))
    if not jobs:
        print("nothing to do")
        return
    print(f"ROUGH ESTIMATE (UNVERIFIED, 8 GB laptop): {total:.1f} h of queries, plus the cached stages per graph",
          flush=True)
    if args.dry_run:
        return

    bounds = P.CsvLog(args.bounds_out, code_tag=code_tag()) if args.test == "t1" and not args.no_bounds else None
    limit = P.time_limit(args)
    extra = {"test": args.test, "tuned": bool(tuning.get("tuned", True)), "tuning_tag": tuning.get("code_tag", "")}

    def hook(stage, secs, alloc, res, info):
        print(f"  {stage:12s} {secs:8.1f}s  peak GPU {alloc:.2f} / {res:.2f} GB", flush=True)

    for job, todo, sweep in jobs:
        need_f = any(q[3] is not None and q[3].learned for q in todo)
        need_g = any(q[3] is not None and q[3].start == "learned" for q in todo)
        budgets = sorted({q[0] for q in todo} | set(sweep))
        print(f"\n=== {job.ident['graph']} / {job.ident['attribute']}: {len(todo)} pending queries ===", flush=True)
        try:
            case = P.Case(job.loader, budgets, args, dev, preset, need_features=need_f,
                          need_rr_greedy=need_g, phi0=tuned["phi0"], delta=tuned["delta"], hook=hook).prepare()
        except (RuntimeError, MemoryError) as e:
            if not P.is_oom(e):
                raise
            print(f"  out of memory while preparing ({type(e).__name__}); graph skipped", flush=True)
            P.free()
            continue
        if job.comms is not None:
            case.comms = job.comms(case.g)
            print(f"  evaluation communities: {len(case.comms)} with >= {EVAL_MIN_COMMUNITY} members", flush=True)
        print("  " + case.describe(), flush=True)
        if args.test == "t2":                       # A and B are host-RAM-bound at T2's R (about 22 bytes per entry)
            print(f"  memory: A {case.est.nnz * 22 / 2**30:.1f} GB, B {case.est_b.nnz * 22 / 2**30:.1f} GB "
                  f"(rough, host RAM)", flush=True)
        for k in budgets:
            case.ref_for(k)
            for rho in sorted({q[1] for q in todo if q[0] == k}):
                if bounds is not None:
                    lower_bound(bounds, case, k, rho)
                for q in todo:
                    if q[0] != k or q[1] != rho:
                        continue
                    _, _, name, cfg, ts = q
                    try:
                        if name == P.FIM:
                            res = P.fim_query(case, k, rho, cfg, models.get(ts), limit)
                            rows = P.record_fim(rec, case, k, rho, cfg, ts, res, models.get(ts), limit, extra)
                            print(P.fmt_fim(rows[0], cfg, ts, res), flush=True)
                        else:
                            res = P.baseline_query(case, k, rho, name, limit)
                            print(P.fmt_baseline(P.record_baseline(rec, case, k, rho, name, res, limit, extra), res),
                                  flush=True)
                    except (RuntimeError, MemoryError) as e:
                        if not P.is_oom(e):
                            raise
                        P.free()
                        why = "host out of memory" if isinstance(e, MemoryError) else "CUDA out of memory"
                        print(f"    {name} {cfg.name() if cfg else ''} k={k} rho={rho}: {why}", flush=True)
                        cols = dict(extra, **(cfg.cols() if cfg else {}))
                        if cfg is not None:
                            cols["deployment"] = "certified"
                        rec.add_missing(case.g, k, rho, name, why, train_seed=ts, **cols)
            if prec is not None and k in sweep:
                try:
                    pareto_sweep(prec, case, job.ident, k, extra)
                except (RuntimeError, MemoryError) as e:
                    if not P.is_oom(e):
                        raise
                    P.free()
                    print(f"    BSM sweep k={k}: out of memory ({type(e).__name__}); not recorded", flush=True)
        del case
        P.free()


if __name__ == "__main__":
    main()
