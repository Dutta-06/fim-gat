"""Test T3 (docs/design_v2.md sections 10, 18.3 and 21): with the same verifier and the same wall-clock budgets, does
the learned proposer make the improvement phase faster or better than the simple hybrid?

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore run_t3.py --dry-run
    PYTHONIOENCODING=utf-8 python -u -W ignore run_t3.py

Fixed before any T3 run (design section 21):
  gate      runs only if tune_v2.py's W3 rule said to train (tuning.json "w3") and train_proposer.py has written
            proposer checkpoints; otherwise it says why and stops, and summarize_v2.py reports T3 as not run under the
            pre-registered rule. --ignore-gate is for development only.
  graphs    the large-tier test graphs under T1's attributes (run_t1_t2.T1_ATTRS), every budget of config.BUDGETS, rho
            config.T3_RHO, the tuned delta and start.
  search    the labelled-only improvement phase (variant v2-a: improve(audit=False)), run once per proposer up to the
            largest of the budgets (default config.TIME_BUDGETS), with no round limit:
              simple      the simple hybrid with setpoint-scaled prices (improve()'s defaults; T3's baseline)
              learned     LearnedProposer, with the same exhaustive fallback, per train seed
              exhaustive  local_search.best_swap every round, for reference
            The set reached by each budget is read off the search's own history (the time of every round), so one run
            serves every budget. Each such set is scored by forward Monte Carlo (RecorderV2 rows in t3.csv,
            deployment "search", with budget_s, reached_s, rr_disp, moves and the RR-disparity trace of the run).
  recall    recall@C (config.T3_RECALL_C) of the simple and of each learned proposer on up to --recall-states states
            evenly spaced along the simple hybrid's path (t3_recall.csv).
The pass rule is applied by summarize_v2.py: (i) the learned proposer reaches the simple hybrid's final RR disparity at
least config.T3_SPEEDUP_MIN times sooner on more than half of the test graphs, or (ii) its Monte Carlo disparity at
config.T3_PRIMARY_BUDGET seconds is lower (paired two-sided Wilcoxon over test graphs, p < 0.05, negative median).
A fail drops the learned proposer; the claim is not reframed. Resumable. One GPU job at a time.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pipeline_v2 as P                                                   # noqa: E402
from common_v2 import NAN, RecorderV2, code_tag, device, gpu_name, preset_config, progress   # noqa: E402
from run_t1_t2 import T1_ATTRS                                            # noqa: E402
from fimgat.v2 import datasets as D                                       # noqa: E402
from fimgat.v2 import imitation as I                                      # noqa: E402
from fimgat.v2.certify import path_states                                 # noqa: E402
from fimgat.v2.config import (BUDGETS, RESULTS_V2, T3_RECALL_C, T3_RHO, TIER, TIME_BUDGETS)   # noqa: E402
from fimgat.v2.cover import CoverState                                    # noqa: E402
from fimgat.v2.improve import FLOOR_RTOL, improve                         # noqa: E402
from fimgat.v2.train import select_seeds_v2                               # noqa: E402

LARGE = tuple(x for x in D.NAMES if TIER.get(x) == "large")
PROPOSERS = ("simple", "exhaustive", "learned")
MAX_ROUNDS = 10 ** 6                      # time, not a round count, ends a T3 search


def done(rec, ident, k, rho, name, ts, budgets, tuned):
    return all(rec.done(**ident, k=int(k), rho=float(rho), method=P.FIM, variant="v2-a", proposer=name,
                        train_seed=ts, budget_s=float(b), deployment="search", **tuned) for b in budgets)


def start_set(case, k, rho, floor, start, model, args):
    """T1's start: the reference set (greedy start) or the constructor's set (learned start)."""
    ref, _ = case.ref_for(k)
    if start != "learned":
        return list(ref)
    rounds = int(model["ck"].get("rounds", 3))
    return [int(v) for v in select_seeds_v2(model["policy"], case.ctx, k, rho, floor, rounds,
                                            n_samples=args.n_samples, mu=P.mu_ref(model["ck"], k))]


def search(case, k, rho, start, proposer, budget):
    """One labelled-only search up to `budget` seconds: (path of seed sets, trace [[secs, rr_disp]] aligned with
    the path, seconds, stopped before the budget (a local optimum))."""
    ref, _ = case.ref_for(k)
    floor = case.floor_for(k, rho)
    t0 = time.perf_counter()
    _, hist = improve(case.est, list(start), floor, proposer=proposer, audit=False, max_rounds=MAX_ROUNDS,
                      max_seconds=float(budget), reference=list(ref))
    secs = time.perf_counter() - t0
    start1 = list(ref) if hist[0]["fallback"] else list(start)
    path = path_states(start1, hist)
    trace = [[float(hist[0]["secs"]), float(hist[0]["disp"])]] + \
        [[float(h["secs"]), float(h["disp"])] for h in hist[1:] if h.get("move")]
    return path, trace, secs, secs < 0.999 * float(budget)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None, help=f"default: every built large-tier test graph {LARGE}")
    ap.add_argument("--attributes", nargs="+", default=None, help="default T1_ATTRS of each dataset")
    ap.add_argument("--ks", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--budgets", nargs="+", type=float, default=list(TIME_BUDGETS),
                    help="wall-clock budgets in seconds (default config.TIME_BUDGETS)")
    ap.add_argument("--proposers", nargs="+", choices=PROPOSERS, default=list(PROPOSERS))
    ap.add_argument("--recall-states", type=int, default=4, help="states along the simple path for recall@C")
    ap.add_argument("--tuning", default=None, help="tune_v2.py's output (default results/v2/tuning.json)")
    ap.add_argument("--untuned", action="store_true", help="development only: config defaults, rows tuned=False")
    ap.add_argument("--ignore-gate", action="store_true", help="development only: run although W3 said no")
    ap.add_argument("--out", default=os.path.join(RESULTS_V2, "t3.csv"))
    ap.add_argument("--recall-out", default=os.path.join(RESULTS_V2, "t3_recall.csv"))
    ap.add_argument("--dry-run", action="store_true")
    P.add_pipeline_args(ap)
    args = ap.parse_args()
    P.check_args(ap, args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tuning = P.load_tuning(args.tuning, required=not args.untuned)
    if args.untuned:
        tuning = dict(tuning, tuned=False)
    w3 = tuning.get("w3") or {}
    if not w3.get("train") and not args.ignore_gate:
        print(f"T3 not run under the pre-registered rule: the W3 rule did not ask for a learned proposer "
              f"({w3.get('reason', 'no W3 result in the tuning file')}).", flush=True)
        return
    tuned = {"delta": float(tuning["delta"]), "phi0": float(tuning["phi0"])}
    start = tuning["start"]
    rho = float(T3_RHO)
    budgets = sorted({float(b) for b in args.budgets})
    ks = sorted({int(k) for k in args.ks})
    dev = device()
    pname, preset = preset_config(args.preset)
    print(f"T3: device={dev} gpu={gpu_name()} preset={pname} code_tag={code_tag()}; start={start}, rho={rho}, "
          f"budgets {budgets} s", flush=True)
    models = P.load_models(args, dev, starts=[start], proposers=["learned"] if "learned" in args.proposers else [])
    seeds = sorted(models)
    if "learned" in args.proposers and not seeds:
        print("T3 needs trained proposers (experiments/v2/train_proposer.py); none were found.", flush=True)
        return
    if start == "learned" and not seeds:
        print("the tuned start is the constructor's, but no constructor checkpoint was found.", flush=True)
        return

    names = args.datasets or [x for x in LARGE if D.available(x) and x in D.splits()["test"]]
    rec = RecorderV2(args.out, n_sims=args.n_sims, preset=pname)
    rlog = P.CsvLog(args.recall_out, code_tag=code_tag(), preset=pname, gpu=gpu_name())
    starts_per = seeds if start == "learned" else [P.NO_TRAIN]
    jobs, total = [], 0.0
    for name in names:
        if name not in D.LOADABLE or not D.available(name):
            print(f"{name}: not built; skipped", flush=True)
            continue
        for attr in (args.attributes or T1_ATTRS.get(name, (D.DEFAULT_ATTRIBUTE[name],))):
            ident = {"dataset": name, "attribute": attr, "graph": name}
            todo = []
            for k in ks:
                for ts in starts_per:
                    # the simple search also supplies the recall states: rerun it (without new rows) if they are missing
                    no_recall = "simple" in args.proposers and not rlog.find(
                        dataset=name, attribute=attr, k=k, start_seed=ts, proposer="simple", status="ok")
                    for prop in args.proposers:
                        for pts in (seeds if prop == "learned" else [ts]):
                            if start == "learned" and prop == "learned" and pts != ts:
                                continue                    # a learned proposer runs from its own seed's start
                            if not done(rec, ident, k, rho, prop, pts, budgets, tuned) or (
                                    prop == "simple" and no_recall):
                                todo.append((k, ts, prop, pts))
            if not todo:
                print(f"{name} / {attr}: every T3 query already in {args.out}", flush=True)
                continue
            hours = len(todo) * budgets[-1] / 3600.0
            total += hours
            print(f"{name} / {attr}: {len(todo)} searches of up to {budgets[-1]:g} s; at most {hours:.1f} h of search, "
                  f"plus the cached stages and Monte Carlo", flush=True)
            jobs.append((ident, todo))
    if not jobs:
        print("nothing to do", flush=True)
        return
    print(f"AT MOST {total:.1f} h of search in total (searches stop early at a local optimum)", flush=True)
    if args.dry_run:
        return

    extra = {"test": "t3", "tuned": bool(tuning.get("tuned", True)), "tuning_tag": tuning.get("code_tag", ""),
             "ignore_gate": bool(args.ignore_gate)}
    bar = progress(sum(len(todo) for _, todo in jobs), "T3", unit="search")
    for ident, todo in jobs:
        bar.set_description_str(f"T3 {ident['graph']}/{ident['attribute']}")
        print(f"\n=== {ident['graph']} / {ident['attribute']}: {len(todo)} searches ===", flush=True)
        case = P.Case(lambda i=ident: D.load(i["dataset"], i["attribute"]), sorted({q[0] for q in todo}), args, dev,
                      preset, need_features=True, need_rr_greedy=start == "learned", phi0=tuned["phi0"],
                      delta=tuned["delta"], hook=lambda st, s, a, r, info: print(f"  {st:12s} {s:8.1f}s", flush=True))
        case.prepare()
        print("  " + case.describe(), flush=True)
        for k in sorted({q[0] for q in todo}):
            ref, total_ref = case.ref_for(k)
            floor = case.floor_for(k, rho)
            simple_paths = {}
            for kk, ts, prop, pts in todo:
                if kk != k:
                    continue
                model = models.get(pts if prop == "learned" else ts)
                s0 = start_set(case, k, rho, floor, start, models.get(ts), args)
                proposer = (P.learned_proposer(model, case.ctx, rho, total_ref, case.delta) if prop == "learned"
                            else prop)
                path, trace, secs, local_opt = search(case, k, rho, s0, proposer, budgets[-1])
                if prop == "simple":
                    simple_paths[ts] = path
                moves_at = [t for t, _ in trace[1:]]
                mc = NAN
                for b in budgets:
                    if rec.done(**ident, k=int(k), rho=rho, method=P.FIM, variant="v2-a", proposer=prop,
                                train_seed=pts, budget_s=float(b), deployment="search", **tuned):
                        continue                            # a rerun for the recall states only
                    n_t = int(np.searchsorted(moves_at, b, side="right"))
                    reached, rr_disp = trace[n_t]
                    row = rec.add(case.g, k, rho, P.FIM, path[n_t], reached, ref_seeds=ref, variant="v2-a",
                                  train_seed=pts, communities=case.comms, proposer=prop, start=start,
                                  deployment="search", budget_s=b, reached_s=reached, rr_disp=rr_disp, moves=n_t,
                                  search_s=secs, local_optimum=local_opt, trace=json.dumps(trace), floor=floor,
                                  total_ref_rr=total_ref, **case.base_cols(), **extra)
                    mc = row["disparity"]
                print(f"    k={k} {prop:10s} s{pts}: {len(trace) - 1} moves in {secs:.1f}s "
                      f"({'local optimum' if local_opt else 'time'}), RR disparity {trace[0][1]:.4f} -> "
                      f"{trace[-1][1]:.4f}; MC disparity at {budgets[-1]:g}s {mc:.4f}", flush=True)
                bar.update(1)
            for ts, path in simple_paths.items():           # recall@C on states along the simple hybrid's path
                if rlog.find(dataset=ident["dataset"], attribute=ident["attribute"], k=k, start_seed=ts,
                             proposer="simple", status="ok"):
                    continue
                idx = sorted({int(round(x)) for x in np.linspace(0, max(len(path) - 2, 0), args.recall_states)})
                states = [CoverState(case.est, path[i]) for i in idx]
                floor_acc = floor + FLOOR_RTOL * abs(floor)
                kw = dict(rho=rho, total_ref=total_ref, floor=floor_acc, return_info=True)
                rows = [("simple", NAN, I.recall_at_c("simple", case.ctx, states, list(T3_RECALL_C), **kw))]
                for pts in (seeds if "learned" in args.proposers else []):
                    if start == "learned" and pts != ts:
                        continue
                    rows.append(("learned", pts, I.recall_at_c(models[pts]["proposer"], case.ctx, states,
                                                              list(T3_RECALL_C), mu=0.0, **kw)))
                for prop, pts, out in rows:
                    rlog.add({**ident, "k": k, "rho": rho, "start_seed": ts, "train_seed": pts, "proposer": prop,
                              "n_states": out["n_states"], "n_improvable": out["n_improvable"], "status": "ok",
                              **{f"recall_{c}": out["recall"][c] for c in T3_RECALL_C}})
        del case
        P.free()
    bar.close()


if __name__ == "__main__":
    main()
