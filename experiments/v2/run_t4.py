"""Test T4 (claim C3): end-to-end runtime on the large tier, with the per-graph cached cost and the per-query
cost timed apart (docs/design_v2.md sections 4, 10, 18 and 19). The queries are those of run_t1_t2.py (the shared
pipeline_v2), timed stage by stage.

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore run_t4.py --dry-run         # plan and rough estimate only
    PYTHONIOENCODING=utf-8 python -u -W ignore run_t4.py                   # every built large-tier graph
    PYTHONIOENCODING=utf-8 python -u -W ignore run_t4.py --datasets pokec_full --ks 50 --train-seeds 0

ONE GPU JOB AT A TIME. Run nothing else on the GPU meanwhile: two jobs ran the 8 GB card out of memory in v1,
and any sharing makes the timings and the memory peaks meaningless. The script warns when another process
already holds GPU memory at start.

Claim C3 is about many fairness queries on one graph (design section 18.3): a deployment pays the cached stages
once per graph and the query stages once per query, so the time for Q queries is prep_s + Q * query_s. Both are
recorded for every method; summarize_v2.py reports them apart, at Q = 1, 10 and 100.

FIM-GAT (--variants; the start set, delta and phi0 come from tune_v2.py's tuning.json):
  v2-b    the method: improve.hive_search, the labelled-only search, then the epsilon-constraint phase
  v2-b0   improve(audit=True): the published acceptance rule
  v2-a    improve(audit=False): labelled disparity only
--proposer simple (the hybrid, the default) or learned (only if tune_v2.py's W3 gate trained one, test T3).

Cached per (dataset, attribute), as pipeline_v2.Case prepares them: load (reported; "from a loaded graph" excludes
it), reference, rr_sample, cert_sample, features and rr_greedy (learned parts only), cover_index, model (per train
seed, loaded once), suite_index (the greedy family's RR index, built inside run_suite).
Per query (k, rho): select (the constructor, 0 s for the greedy start; a baseline's own run), improve (FIM-GAT's
search), verify (the deploy-time certificate: certified choice for FIM-GAT, certified_equal_spread for a baseline).

Outputs (results/v2/):
  t4_runtime.csv  one row per stage: scope (cached / query), stage, secs, peak GPU memory allocated and reserved
                  (GB, peaks reset at the stage start, so memory already held counts), host RSS after the stage
                  (NaN without psutil), preset, gpu, code_tag and the query key
  t4.csv          RecorderV2 rows of every query (pipeline_v2.record_fim / record_baseline)
Resumable. Timings follow a warm-up on a tiny graph, so numba compilation and CUDA start-up are not charged to anyone.
Every method gets --time-limit seconds per query; beyond the limit plus max(1 s, 10%) a row is recorded missing.
"""
import argparse
import os
import random
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pipeline_v2 as P                                                   # noqa: E402
from common_v2 import NAN, RecorderV2, code_tag, device, gpu_name, preset_config   # noqa: E402
from fimgat.baselines import RRGreedy                                     # noqa: E402
from fimgat.diffusion import ExposureEstimator                            # noqa: E402
from fimgat.graph import from_edges                                       # noqa: E402
from fimgat.v2 import datasets as D                                       # noqa: E402
from fimgat.v2.baselines_v2 import SUITE, run_suite                       # noqa: E402
from fimgat.v2.certify import certified_choice, certified_equal_spread, deploy_candidates   # noqa: E402
from fimgat.v2.config import BUDGETS, CERT_ALPHA, RESULTS_V2, RHO_DEFAULT, TIER, rr_floor   # noqa: E402
from fimgat.v2.cover import cover_index                                   # noqa: E402
from fimgat.v2.features import build_context                              # noqa: E402
from fimgat.v2.metrics_v2 import evaluate_v2                              # noqa: E402
from fimgat.v2.reference import equal_spread_test                         # noqa: E402
from fimgat.v2.train import exact_rates, select_seeds_v2                  # noqa: E402

LARGE = tuple(x for x in D.NAMES if TIER.get(x) == "large")
# UNVERIFIED: rough guesses for the 8 GB laptop, scaled by hand from the v1 full-Pokec run (research_protocol
# section 6: RR sampling of 150k roots x 8 in about 3 min, greedy family 6-54 s, FW-Fair 4.5 s, FIM-GAT
# inference about 27 s). No v2 stage has been timed yet.
SCALE = {"pokec_full": 1.0, "instagram": 0.3, "snap_youtube": 0.6, "linkedin": 2.5}


def estimate_min(name, todo, runs, limit, ref_cached):
    s = SCALE.get(name, 1.0)
    cached = s * ((0 if ref_cached else runs * 4.0) + 3.0 + 3.0 + 3.0)        # + the certificate sample
    per = {P.FIM: 3.0, "FW-Disparity": 1.0, "Degree": 0.05, "AGE": 0.0, "Prior": 0.0,
           **{m: 1.0 for m in P.GREEDY_FAMILY}}
    cap = (limit or 3600.0) / 60.0
    query = sum(min(s * per.get(q[2], 15.0), cap) + s * 0.3 for q in todo)     # + evaluation MC and certificate
    return cached, query


# ------------------------------------------------------------------ warm-up

def warm_up(dev, models, methods, cfgs):
    """Compile every numba kernel and start CUDA on a tiny graph, so no timed stage pays for it."""
    rng = np.random.default_rng(0)
    n, k = 400, 5
    g = from_edges("t4_warmup", n, rng.integers(0, n, (2400, 2)), rng.integers(0, 3, n), ["a", "b", "c"],
                   undirected=True, weighted_cascade=True,
                   meta={"dataset": "t4_warmup", "attribute": "warm", "attrs": {}, "tier": "small"})
    est = ExposureEstimator(g, R=4, seed=0, device=dev)
    est_b = ExposureEstimator(g, R=2, seed=1, device="cpu")
    learned = [m for m in models.values() if m["policy"] is not None or m["proposer"] is not None]
    ctx = build_context(g, est, dev) if learned else types.SimpleNamespace(graph=g, est=est, device=torch.device(dev))
    rg = RRGreedy(est)
    ref = [int(v) for v in rg.greedy(k)]
    total = exact_rates(est, ref)[1]
    floor = rr_floor(RHO_DEFAULT, total)
    cover_index(est)
    kw = dict(max_rounds=3, reference=ref, audit_max_size=P.AUDIT_MAX_SIZE)
    for variant in sorted({c.variant for c in cfgs}):
        seeds, _, paths, _ = P.search_paths(variant, est, rg.parity(k), floor, "simple", kw)
        certified_choice(est_b, deploy_candidates(paths), ref, RHO_DEFAULT, alpha=CERT_ALPHA)
        certified_equal_spread(est_b, seeds, ref, RHO_DEFAULT, alpha=P.ALPHA)
    fast = [m for m in methods if m not in ("AGE", "Prior")]    # both are skipped on the large tier (n limits)
    if fast:
        run_suite(ctx, k, RHO_DEFAULT, floor, ref, methods=fast, time_limit=None, cache={})
    certified_equal_spread(est_b, rg.parity(k), ref, RHO_DEFAULT, alpha=P.ALPHA)
    equal_spread_test(g, rg.parity(k), ref, rho=RHO_DEFAULT, n_sims=16, seed=0, alpha=P.ALPHA)   # the recorder's es_*
    evaluate_v2(g, ref, n_sims=16, seed=0, communities=[np.arange(20)])     # given communities: no Louvain cache
    if learned:
        m = learned[0]
        ctxs = [ctx]
        if torch.device(dev).type == "cuda":            # the chunked (CPU-edge) path of the large graphs
            c2 = build_context(g, est, dev, edge_device="cpu")
            c2.chunk_edges = 1000
            ctxs.append(c2)
        for c in ctxs:
            c.total_ref[k], c.greedy_seeds = total, {k: ref}
            s = ref
            if m["policy"] is not None:
                s = select_seeds_v2(m["policy"], c, k, RHO_DEFAULT, floor, int(m["ck"].get("rounds", 3)),
                                    n_samples=4, mu=P.mu_ref(m["ck"], k))
            if m["proposer"] is not None:
                for variant in sorted({c_.variant for c_ in cfgs}):
                    P.search_paths(variant, est, s, floor, P.learned_proposer(m, c, RHO_DEFAULT, total),
                                   dict(kw, max_rounds=2))
    P.free()


# ------------------------------------------------------------------ one graph

class T4Run:
    """Every pending query of one (dataset, attribute), with each cached and query stage logged."""

    def __init__(self, ident, todo, models, args, rec, log, dev, pname, preset, tuned, extra):
        self.ident, self.todo, self.models, self.args = ident, todo, models, args
        self.rec, self.log, self.dev, self.pname, self.preset = rec, log, dev, pname, preset
        self.tuned, self.extra, self.gpu = tuned, extra, gpu_name()
        self.limit = P.time_limit(args)
        self.pending = []                               # cached stages noted before the graph was loaded
        self.case = None

    def base(self):
        g = self.case.g if self.case is not None else None
        out = {**self.ident, "n": NAN, "m": NAN}
        if g is not None:
            out.update(graph=g.name, n=int(g.n), m=int(g.m))
            if getattr(self.case, "R", None) is not None:      # set right after loading, before any sample
                out.update(R=self.case.R, max_roots=NAN if self.case.max_roots is None else self.case.max_roots)
        return out

    def row(self, scope, stage, secs, alloc, res, k=NAN, rho=NAN, method="", variant="", train_seed=NAN,
            status="ok", reason="", info=None):
        if scope == "cached":                           # latest measurement of a cached stage wins
            self.log.remove(dataset=self.ident["dataset"], attribute=self.ident["attribute"], scope="cached",
                            stage=stage, train_seed=train_seed, preset=self.pname, gpu=self.gpu)
        return self.log.add({**self.base(), "scope": scope, "stage": stage, "k": k, "rho": rho, "method": method,
                             "variant": variant, "train_seed": train_seed, "secs": float(secs),
                             "peak_alloc_gb": alloc, "peak_reserved_gb": res, "rss_after_gb": P.rss_gb(),
                             "status": status, "reason": reason, "info": info or {}})

    def hook(self, stage, secs, alloc, res, info):
        print(f"  {stage:12s} {secs:8.1f}s  peak GPU {alloc:.2f} / {res:.2f} GB (alloc / reserved)", flush=True)
        if self.case is None or self.case.g is None:
            self.pending.append((stage, secs, alloc, res, info))
        else:
            self.row("cached", stage, secs, alloc, res, info=info)

    def run(self):
        args, todo = self.args, self.todo
        cfgs = [q[3] for q in todo if q[3] is not None]
        print(f"\n=== {self.ident['graph']} / {self.ident['attribute']}: {len(todo)} pending queries ===", flush=True)
        self.case = P.Case(lambda: D.load(self.ident["dataset"], self.ident["attribute"]), sorted({q[0] for q in todo}),
                           args, self.dev, self.preset, need_features=any(c.learned for c in cfgs),
                           need_rr_greedy=any(c.start == "learned" for c in cfgs), phi0=self.tuned["phi0"],
                           delta=self.tuned["delta"], hook=self.hook)
        case = self.case
        case.prepare()
        for stage, secs, alloc, res, info in self.pending:
            self.row("cached", stage, secs, alloc, res, info=info)
        self.pending = []
        if self.ident["dataset"] == "snap_youtube":     # evaluation only, never passed to a method
            from run_t1_t2 import EVAL_MIN_COMMUNITY, gt_communities, restrict
            case.comms = restrict(gt_communities("all"), np.arange(case.g.n), EVAL_MIN_COMMUNITY)
        print("  " + case.describe(), flush=True)
        for ts in sorted({q[4] for q in todo if q[2] == P.FIM}):
            m = self.models.get(ts)
            if m and m["path"]:
                self.row("cached", "model", m["secs"], m["alloc"], m["reserved"], train_seed=ts,
                         info={"path": m["path"], "loaded_once": True})
        for k in sorted({q[0] for q in todo}):
            _, total_ref = case.ref_for(k)
            print(f"  k={k}: reference RR total {total_ref:.1f}", flush=True)
            for rho in sorted({q[1] for q in todo if q[0] == k}):
                for q in todo:
                    if q[0] != k or q[1] != rho:
                        continue
                    try:
                        self.query(q)
                    except (RuntimeError, MemoryError) as e:
                        if not P.is_oom(e):
                            raise
                        P.free()
                        why = "host out of memory" if isinstance(e, MemoryError) else "CUDA out of memory"
                        print(f"    {q[2]} k={k} rho={rho}: {why}", flush=True)
                        cols = dict(self.extra, **(q[3].cols() if q[3] else {}))
                        if q[3] is not None:
                            cols["deployment"] = "certified"
                        self.rec.add_missing(case.g, k, rho, q[2], why, train_seed=q[4], **cols)

    def query(self, q):
        k, rho, name, cfg, ts = q
        case = self.case
        if name == P.FIM:
            key = dict(k=k, rho=rho, method=P.FIM, variant=cfg.variant, train_seed=ts)
            self.log.remove(dataset=self.ident["dataset"], attribute=self.ident["attribute"], scope="query",
                            preset=self.pname, gpu=self.gpu, **key)
            res = P.fim_query(case, k, rho, cfg, self.models.get(ts), self.limit)
            for stage, secs in (("select", res["select_s"]), ("improve", res["improve_s"]), ("verify", res["verify_s"])):
                a, r = res["peaks"][stage]
                info = {"select": {"start": cfg.start, **res["sel"]},
                        "improve": {"rounds": res["rounds"], "accepted": res["accepted"], "restarted": res["restarted"],
                                    "family_size": res["family_size"], "max_seconds": res["max_seconds"],
                                    **{x: v for x, v in res["search"].items() if not x.startswith("_")}},
                        "verify": {"rule": "certify_sets + choose", "min_ucb": res["deploy"],
                                   "path": res["deploy_path"],
                                   "certificate": bool(res["judge"]["passed"])}}[stage]
                self.row("query", stage, secs, a, r, info=info, **key)
            rows = P.record_fim(self.rec, case, k, rho, cfg, ts, res, self.models.get(ts), self.limit, self.extra)
            print(P.fmt_fim(rows[0], cfg, ts, res), flush=True)
        else:
            key = dict(k=k, rho=rho, method=name, variant="default", train_seed=NAN)
            self.log.remove(dataset=self.ident["dataset"], attribute=self.ident["attribute"], scope="query",
                            preset=self.pname, gpu=self.gpu, **key)
            res = P.baseline_query(case, k, rho, name, self.limit)
            a, r = res["peaks"]["select"]
            self.row("query", "select", res["secs"], a, r, status="ok" if res["seeds"] is not None else "missing",
                     reason=res["reason"], info={**res["info"], "suite_wall_s": res["wall"]}, **key)
            if res["judge"] is not None:
                a, r = res["peaks"]["verify"]
                self.row("query", "verify", res["verify_s"], a, r, info={"rule": "certified_equal_spread",
                                                                         "passed": bool(res["judge"]["passed"])}, **key)
            print(P.fmt_baseline(P.record_baseline(self.rec, case, k, rho, name, res, self.limit, self.extra), res),
                  flush=True)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None, help=f"default: every built large-tier graph {LARGE}")
    ap.add_argument("--attributes", nargs="+", default=None, help="default: each dataset's default attribute")
    ap.add_argument("--ks", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--rhos", nargs="+", type=float, default=[RHO_DEFAULT])
    ap.add_argument("--variants", nargs="+", choices=list(P.VARIANTS), default=["v2-b"])
    ap.add_argument("--start", choices=list(P.STARTS), default=None, help="default: tuning.json's start")
    ap.add_argument("--proposer", choices=["simple", "learned"], default="simple")
    ap.add_argument("--methods", nargs="*", default=list(SUITE), help="run_suite baselines (none: --methods)")
    ap.add_argument("--no-fim", action="store_true", help="baselines only")
    ap.add_argument("--tuning", default=None, help="tune_v2.py's output (default results/v2/tuning.json)")
    ap.add_argument("--untuned", action="store_true", help="development only: config defaults, rows tuned=False")
    ap.add_argument("--out", default=os.path.join(RESULTS_V2, "t4.csv"))
    ap.add_argument("--runtime-out", default=os.path.join(RESULTS_V2, "t4_runtime.csv"))
    ap.add_argument("--gpu-busy-gb", type=float, default=1.5, help="warn if this much GPU memory is in use at start")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimate, then stop")
    P.add_pipeline_args(ap)
    args = ap.parse_args()
    P.check_args(ap, args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    bad = [m for m in args.methods if m not in SUITE]
    if bad:
        ap.error(f"unknown methods {bad}; the suite is {SUITE}")
    names = args.datasets or [x for x in LARGE if D.available(x)]
    unknown = [x for x in names if x not in D.LOADABLE]
    if unknown:
        ap.error(f"unknown datasets {unknown}; loadable: {D.LOADABLE}")
    for x in names:
        if TIER.get(x) != "large":
            print(f"note: {x} is in the {TIER.get(x)} tier; T4 (claim C3) is defined on the large tier")
    ks = sorted({int(k) for k in args.ks})
    rhos = sorted({float(r) for r in args.rhos})
    tuning = P.load_tuning(args.tuning, required=not args.untuned)
    if args.untuned:
        tuning = dict(tuning, tuned=False)
    tuned = {"delta": float(tuning["delta"]), "phi0": float(tuning["phi0"])}
    start = args.start or tuning["start"]
    cfgs = [] if args.no_fim else [P.FimConfig(v, start, args.proposer) for v in args.variants]

    dev = device()
    pname, preset = preset_config(args.preset)
    print(f"T4 runtime: device={dev} gpu={gpu_name()} preset={pname} {preset} code_tag={code_tag()}", flush=True)
    print(f"tuning: delta={tuned['delta']} phi0={tuned['phi0']} start={start} tuned={tuning.get('tuned', True)}",
          flush=True)
    print("One GPU job at a time: nothing else may use the GPU while this runs.", flush=True)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 2**30
        if used > args.gpu_busy_gb:
            print(f"WARNING: {used:.1f} GB of GPU memory is already in use (another job?); timings and peaks "
                  f"assume this script is the only GPU job", flush=True)
    models = P.load_models(args, dev, starts=[c.start for c in cfgs], proposers=[c.proposer for c in cfgs])
    dropped = [c.name() for c in cfgs if c.learned and not P.seeds_for(c, models)]
    if dropped:
        print(f"note: no checkpoints for {dropped}; those configurations are not run", flush=True)
    cfgs = [c for c in cfgs if not c.learned or P.seeds_for(c, models)]

    rec = RecorderV2(args.out, n_sims=args.n_sims, preset=pname)
    log = P.CsvLog(args.runtime_out, preset=pname, gpu=gpu_name(), code_tag=code_tag())
    store = P.read_json(args.ref_file)
    jobs, tot_c, tot_q = [], 0.0, 0.0
    for name in names:
        if not D.available(name):
            print(f"{name}: not built ({D.built_path(name)}); skipped. Build it with experiments/v2/build_datasets.py")
            continue
        for attr in (args.attributes or [D.DEFAULT_ATTRIBUTE[name]]):
            ident = {"dataset": name, "attribute": attr, "graph": name}
            todo = P.plan_queries(rec, ident, ks, rhos, cfgs, models, args.methods, tuned)
            if not todo:
                print(f"{name} / {attr}: every query already in {args.out}")
                continue
            ref_cached = any(kk.startswith(f"{name}|{attr}|") for kk in store)
            c, qm = estimate_min(name, todo, args.reference_runs, args.time_limit, ref_cached)
            tot_c, tot_q = tot_c + c, tot_q + qm
            n_fim = sum(q[2] == P.FIM for q in todo)
            print(f"{name} / {attr}: {len(todo)} queries ({n_fim} FIM-GAT, {len(todo) - n_fim} baseline); "
                  f"rough estimate {c:.0f} min cached + {qm:.0f} min queries", flush=True)
            jobs.append((ident, todo))
    if not jobs:
        print("nothing to do")
        return
    print(f"ROUGH ESTIMATE (UNVERIFIED, 8 GB laptop): {(tot_c + tot_q) / 60:.1f} h in total "
          f"({tot_c / 60:.1f} h cached stages, {tot_q / 60:.1f} h queries; LocalSearch and Simple-v2 may run to "
          f"--time-limit {args.time_limit:g}s per query)", flush=True)
    if args.dry_run:
        return

    if not args.no_warmup:
        t = time.time()
        try:
            warm_up(dev, models, args.methods, cfgs)
            print(f"warm-up done in {time.time() - t:.0f}s", flush=True)
        except Exception as e:                          # timings then include compilation; say so and go on
            print(f"WARNING: warm-up failed ({type(e).__name__}: {e}); first timings include numba compilation",
                  flush=True)
            P.free()

    extra = {"test": "t4", "tuned": bool(tuning.get("tuned", True)), "tuning_tag": tuning.get("code_tag", "")}
    for ident, todo in jobs:
        run = T4Run(ident, todo, models, args, rec, log, dev, pname, preset, tuned, extra)
        try:
            run.run()
        except (RuntimeError, MemoryError) as e:
            if not P.is_oom(e):
                raise
            print(f"{ident['dataset']} / {ident['attribute']}: out of memory while preparing ({type(e).__name__}); "
                  f"graph skipped", flush=True)
        except KeyError as e:                           # attribute not on this dataset
            print(f"{ident['dataset']} / {ident['attribute']}: {e}; skipped", flush=True)
        del run
        P.free()


if __name__ == "__main__":
    main()
