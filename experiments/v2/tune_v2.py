"""Validation tuning of FIM-GAT v2 on the validation split only (datasets.splits()["val"], pokec_t). Writes
results/v2/tuning.json, which run_t1_t2.py and run_t4.py read (docs/design_v2.md sections 18 and 19).

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore tune_v2.py --dry-run
    PYTHONIOENCODING=utf-8 python -u -W ignore tune_v2.py

Four settings, in this order (each uses the ones before it):
  delta   the RR floor margin of every method that searches against the floor (design section 19; it was section
          16 item 4), from config.DELTA_GRID. The method (v2-b, greedy start, simple hybrid, certified deployment,
          the default phi0) runs at rho = RHO_C1 and every budget; the score is the mean Monte Carlo labelled
          disparity of the deployed sets (all carry the certificate). A small delta leaves the search room, but its
          final sets fail the certificate and the reference is deployed; a large one certifies but leaves little
          room. The lowest score wins; ties go to the smaller delta.
  phi0    the auditor's conductance cap, from config.MAX_CONDUCTANCE_GRID. The method (v2-b, greedy start, simple
          hybrid, certified deployment) runs at rho = RHO_C2 and every budget on the validation graph under its
          labelled attribute (region); the score is the mean Monte Carlo disparity over the held-out attributes
          (gender, age) of the sets deployed by the path rule ("certified-path", T2's deployment, which keeps the
          epsilon-constraint phase that phi0 acts on; design section 21). The lowest score wins; ties go to the
          smaller cap.
  start   the T1 rule of design section 18.2. The method with the constructor's start (every train seed that has a
          checkpoint) against the greedy start, at rho = RHO_C1 and every budget, scored by the mean Monte Carlo
          labelled disparity of the deployed sets (each carries the certificate, so all are at equal spread). The
          constructor wins only if its mean is strictly lower. Without a checkpoint the start is greedy, and the
          reason is recorded.
  w3      the learned-proposer gate of design section 18.3: diagnostics.learning_decision over
          diagnostics.measure_proposer at states along the simple hybrid's labelled-only path (v2-a, greedy start)
          at rho = RHO_C1, grid 16 x 8, with improve_share from the method's runs. The split has no large-tier
          validation graph, so this runs on the medium-tier pokec_t (design section 19).

Every number comes from the validation graph's own RR samples and its own Monte Carlo stream; nothing from a test
graph. Each query's row goes to results/v2/tuning_runs.csv (RecorderV2), so every decision can be re-derived, and a
rerun reuses them (resumable).
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
from fimgat.v2 import datasets as D                                       # noqa: E402
from fimgat.v2.certify import path_states                                 # noqa: E402
from fimgat.v2.config import (BUDGETS, DELTA_GRID, MAX_CONDUCTANCE_GRID, RESULTS_V2, RHO_C1, RHO_C2,   # noqa: E402
                              rr_floor)
from fimgat.v2.diagnostics import learning_decision, measure_proposer     # noqa: E402
from fimgat.v2.improve import improve                                     # noqa: E402

STEPS = ("delta", "phi0", "start", "w3")
W3_GRID = (16, 8)


def mean_of(rows, col):
    vals = [float(r[col]) for r in rows if r.get("status") == "ok" and r.get(col) == r.get(col)]
    return float(np.mean(vals)) if vals else NAN


def run_method(rec, case, ident, k, rho, cfg, ts, model, limit, step, extra, deployment="certified"):
    """One query of `cfg` (resumable): returns its row of `deployment` ("certified": Theorem 2's rule, T1's
    deployment; "certified-path": the path rule, T2's)."""
    key = dict(P.fim_key(ident, k, rho, cfg, ts, deployment=deployment,
                         tuned={"delta": case.delta, "phi0": case.phi0}), tuning_step=step)
    got = rec.find(**key)
    if got:
        return got[-1]
    res = P.fim_query(case, k, rho, cfg, model, limit)
    rows = P.record_fim(rec, case, k, rho, cfg, ts, res, model, limit, dict(extra, tuning_step=step))
    print(P.fmt_fim(rows[0], cfg, ts, res), flush=True)
    return next(r for r in rows if r.get("deployment") == deployment)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=D.splits()["val"][0], help="validation graph (default: the val split)")
    ap.add_argument("--attribute", default=None, help="labelled attribute (default: the dataset's default)")
    ap.add_argument("--ks", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--delta-grid", nargs="+", type=float, default=list(DELTA_GRID))
    ap.add_argument("--phi0-grid", nargs="+", type=float, default=list(MAX_CONDUCTANCE_GRID))
    ap.add_argument("--w3-states", type=int, default=4, help="states measured per budget along the path")
    ap.add_argument("--out", default=P.tuning_path(), help="the tuning file the test scripts read")
    ap.add_argument("--runs-out", default=os.path.join(RESULTS_V2, "tuning_runs.csv"))
    ap.add_argument("--dry-run", action="store_true")
    P.add_pipeline_args(ap)
    args = ap.parse_args()
    P.check_args(ap, args)
    if args.dataset not in D.splits()["val"]:
        ap.error(f"{args.dataset} is not in the validation split {D.splits()['val']}: tuning never uses test graphs")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ks = sorted({int(k) for k in args.ks})
    dev = device()
    pname, preset = preset_config(args.preset)
    attr = args.attribute or D.DEFAULT_ATTRIBUTE.get(args.dataset, "degree")
    ident = {"dataset": args.dataset, "attribute": attr, "graph": args.dataset}
    print(f"tuning on {args.dataset} / {attr}: device={dev} gpu={gpu_name()} preset={pname} code_tag={code_tag()}",
          flush=True)
    models = P.load_models(args, dev, starts=["learned"] if "start" in args.steps else [])
    n_q = (len(args.delta_grid) * len(ks) if "delta" in args.steps else 0) + \
          (len(args.phi0_grid) * len(ks) if "phi0" in args.steps else 0) + \
          ((len(models) + 1) * len(ks) if "start" in args.steps else 0)
    w3 = f", plus the W3 measurements ({args.w3_states} states x {len(ks)} budgets)" if "w3" in args.steps else ""
    print(f"plan: {n_q} method queries ({len(models)} constructor checkpoints){w3}", flush=True)
    if args.dry_run:
        return
    if not D.available(args.dataset):
        sys.exit(f"{args.dataset} is not built: run experiments/v2/build_datasets.py --only {args.dataset}")

    rec = RecorderV2(args.runs_out, n_sims=args.n_sims, preset=pname)
    limit = P.time_limit(args)
    extra = {"test": "tuning", "tuned": False}

    def hook(stage, secs, alloc, res, info):
        print(f"  {stage:12s} {secs:8.1f}s", flush=True)
    case = P.Case(lambda: D.load(args.dataset, attr), ks, args, dev, preset, need_features=bool(models),
                  need_rr_greedy=bool(models), hook=hook).prepare()
    print("  " + case.describe(), flush=True)
    old = P.read_json(args.out)
    # a run of some steps keeps what earlier runs decided (for example the start step after the constructor exists,
    # docs/cloud_runbook.md section 5); the steps run now overwrite their own keys only
    out = {**old, "validation": {"dataset": args.dataset, "attribute": attr, "ks": ks, "rho_phi0": RHO_C2,
                                 "rho_start": RHO_C1, "graph_n": int(case.g.n), "R": case.R, "cert_R": case.R_B},
           "delta": old.get("delta", P.DELTA_MARGIN), "phi0": old.get("phi0", P.MAX_CONDUCTANCE),
           "start": old.get("start", "greedy")}
    greedy = P.FimConfig("v2-b", "greedy", "simple")
    case.phi0 = float(out["phi0"])
    bar = progress(n_q, "tuning " + "+".join(x for x in STEPS if x in args.steps))

    def query(*a, **kw):                            # run_method, counted on the bar
        row = run_method(*a, **kw)
        bar.update(1)
        return row

    if "delta" in args.steps:                       # 1. the RR floor margin
        scores = {}
        for d in args.delta_grid:
            case.delta = float(d)
            rows = [query(rec, case, ident, k, RHO_C1, greedy, P.NO_TRAIN, None, limit, "delta", extra)
                    for k in ks]
            scores[str(d)] = mean_of(rows, "disparity")
            ref_share = float(np.mean([str(r.get("deploy_is_reference")) in ("True", "1", "1.0") for r in rows]))
            print(f"  delta={d}: mean labelled disparity {scores[str(d)]:.5f} (reference deployed in "
                  f"{ref_share:.0%} of budgets)", flush=True)
        ok = {float(x): v for x, v in scores.items() if v == v}
        out["delta"] = min(ok, key=lambda x: (ok[x], x)) if ok else P.DELTA_MARGIN
        out["delta_scores"] = scores
        print(f"  -> delta = {out['delta']}", flush=True)
    case.delta = float(out["delta"])

    if "phi0" in args.steps:                        # 2. the auditor's conductance cap
        scores = {}
        for phi in args.phi0_grid:
            case.phi0 = float(phi)
            rows = [query(rec, case, ident, k, RHO_C2, greedy, P.NO_TRAIN, None, limit, "phi0", extra,
                          deployment="certified-path") for k in ks]
            scores[str(phi)] = mean_of(rows, "heldout_disparity")
            print(f"  phi0={phi}: mean held-out-attribute disparity {scores[str(phi)]:.5f}", flush=True)
        ok = {float(p): s for p, s in scores.items() if s == s}
        out["phi0"] = min(ok, key=lambda p: (ok[p], p)) if ok else P.MAX_CONDUCTANCE
        out["phi0_scores"] = scores
        print(f"  -> phi0 = {out['phi0']}", flush=True)
    case.phi0 = float(out["phi0"])

    if "start" in args.steps:                       # 3. the start set (T1 rule)
        g_rows = [query(rec, case, ident, k, RHO_C1, greedy, P.NO_TRAIN, None, limit, "start", extra) for k in ks]
        g_score = mean_of(g_rows, "disparity")
        learned = P.FimConfig("v2-b", "learned", "simple")
        l_rows = [query(rec, case, ident, k, RHO_C1, learned, ts, models[ts], limit, "start", extra)
                  for ts in P.seeds_for(learned, models) for k in ks]
        l_score = mean_of(l_rows, "disparity")
        if not models:
            out["start"], why = "greedy", "no constructor checkpoint"
        elif l_score < g_score:
            out["start"], why = "learned", f"constructor mean {l_score:.5f} < greedy {g_score:.5f}"
        else:
            out["start"], why = "greedy", f"constructor mean {l_score:.5f} >= greedy {g_score:.5f}"
        out["start_scores"] = {"greedy": g_score, "learned": l_score, "n_learned_seeds": len(models)}
        out["start_reason"] = why
        print(f"  -> start = {out['start']} ({why})", flush=True)

    if "w3" in args.steps:                          # 4. the learned-proposer gate
        method_rows = rec.find(dataset=args.dataset, attribute=attr, method=P.FIM, variant="v2-b", start=out["start"],
                               deployment="certified", phi0=case.phi0, delta=case.delta, status="ok")
        tot = [float(r["select_s"]) + float(r["improve_s"]) + float(r["verify_s"]) for r in method_rows]
        share = float(np.mean([float(r["improve_s"]) / t for r, t in zip(method_rows, tot) if t > 0])) if tot else None
        meas, t0 = [], time.perf_counter()
        for k in ks:
            ref, total_ref = case.ref_for(k)
            floor = rr_floor(RHO_C1, total_ref, case.delta)
            _, hist = improve(case.est, list(ref), floor, proposer="simple", audit=False, max_rounds=args.max_rounds,
                              reference=list(ref))
            path = path_states(ref, hist)
            idx = sorted(set(np.linspace(0, max(len(path) - 2, 0), args.w3_states).round().astype(int)))
            for i in idx:
                meas.append(measure_proposer(case.est, path[i], floor, grids=(W3_GRID,), n_time=32, device=dev))
        d = learning_decision(meas, grid=W3_GRID, improve_share=share)
        d.update(improve_share=share, n_measured=len(meas), secs=time.perf_counter() - t0,
                 graph_tier=P.tier_of(case.g), note="no large-tier validation graph exists; measured on " + args.dataset)
        out["w3"] = d
        print(f"  -> W3: train a learned proposer = {d['train']} ({d['reason']})", flush=True)

    bar.close()
    run = {"steps": list(args.steps), "code_tag": code_tag(), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "gpu": gpu_name(), "preset": pname}
    done = set(old.get("steps", [])) | set(args.steps)
    out.update(code_tag=run["code_tag"], when=run["when"], tuned=True, steps=[s for s in STEPS if s in done],
               gpu=run["gpu"], preset=pname, history=list(old.get("history", [])) + [run])
    if "start" not in done:
        out["start_reason"] = "provisional: the start step has not run (it waits for the constructor; cloud runbook)"
    P.write_json(args.out, out)
    print(f"wrote {args.out}:\n{json.dumps({k: v for k, v in out.items() if k != 'w3'}, indent=1, default=str)}",
          flush=True)


if __name__ == "__main__":
    main()
