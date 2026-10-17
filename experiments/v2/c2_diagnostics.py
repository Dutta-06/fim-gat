"""Post-hoc diagnostics of the C2 null (docs/design_v2.md section 22). EXPLORATORY: specified on 2026-10-17, after T2
had failed and before these diagnostics ran. They test an explanation of the failure; they never replace T2's
pre-registered verdict, and their results are reported as exploratory.

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore c2_diagnostics.py --dry-run
    PYTHONIOENCODING=utf-8 python -u -W ignore c2_diagnostics.py --rhos 0.97        # T2's primary floor first
    PYTHONIOENCODING=utf-8 python -u -W ignore c2_diagnostics.py --rhos 0.99
    PYTHONIOENCODING=utf-8 python -u -W ignore c2_diagnostics.py --summarize

The finding to test. On T2's six 100,000-node samples of com-Youtube, option (b) deployed the same set as option (a)
in all 60 cases: the epsilon-constraint phase may raise labelled disparity by at most one standard error, kappa =
D_a + SE(D_a), about 0.0004 against disparities near 0.38, and it accepted no move. Across T1 and T2 the phase acted
more often where that room was larger. Three diagnostics, on the same 60 states (T2's samples, budgets and floors,
T2's R = 64 and the tuned delta and phi0):

  room      phase 2 exactly as improve.hive_search runs it (community-price proposals, multi-scale audits, the
            exhaustive fallback), from the same phase-1 end, with the cap kappa = D_a (1 + f), f in FRACTIONS, and the
            pre-registered kappa = D_a + SE as the control. The control must reproduce T2's deployed (b) set and the
            labelled-only run T2's (a) set (column matches_t2). Each run is deployed by T2's path rule on B and scored by
            forward Monte Carlo; summarize compares it with (a)'s deployment, paired by case. Prediction: with more room
            the phase accepts moves; whether hidden-community disparity then falls is what coverage explains.
  coverage  Lemma W1.1's terms for every deployed set of a phase-2 run (theory section 9.1): the hidden disparity D_H,
            the family's D_F, the gap bounds from the hidden extremes h+ and h- to the auditor's family, and the best
            Jaccard overlap of a family member with h+ and with h-. Per sample, the conductance of every evaluated
            ground-truth community against the auditor's cap phi0: communities above phi0 are outside the auditor's
            search class (theory section 9.6, item 7).
  price     at the phase-1 end (a single-swap local optimum of labelled disparity), floor-feasible single swaps are
            scanned in increasing order of their labelled-disparity cost (up to SCAN_MAX). Reported: the cost of the
            cheapest swap that lowers the community disparity of (F) the auditor's first-audit family and (H) the
            hidden family on sample A's roots, and the largest decrease of each a single swap can buy within labelled
            cost budgets of one SE and of 1%, 5% and 20% of D_a. This is the exchange rate between labelled fairness
            and community fairness that the cap decides whether to pay.

Evaluation integrity: ground-truth communities only score sets and price swaps for this diagnostic; no set is chosen
or deployed from them. Every deployed set still comes from the search sample A and the certificate on B.

Outputs, in results/v2/parts/c2diag/: room.csv (RecorderV2 rows; variant = run label, deployment = "certified-path"
or "final"), coverage.csv, price.csv, and summary.md (--summarize). Resumable: a finished case is skipped.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pipeline_v2 as P                                                   # noqa: E402
from common_v2 import NAN, RecorderV2, code_tag, device, mc_seed_for, preset_config, progress   # noqa: E402
from run_t1_t2 import t2_jobs                                             # noqa: E402
from fimgat.v2 import imitation as I                                      # noqa: E402
from fimgat.v2.auditor import audit_bands, conductance, node_exposure, size_bands   # noqa: E402
from fimgat.v2.certify import certify_sets, choose, deploy_candidates, path_states  # noqa: E402
from fimgat.v2.config import (AUDIT_BAND_RATIO, AUDIT_MAX_SIZE, BUDGETS, CERT_ALPHA, KAPPA_SE,   # noqa: E402
                              MIN_COMMUNITY, RESULTS_V2, RHO_C1, RHO_C2, T2_RR_PER_NODE, T2_SAMPLE_SIZE, T2_SAMPLES)
from fimgat.v2.cover import CoverState                                    # noqa: E402
from fimgat.v2.hidden import transfer_bound                               # noqa: E402
from fimgat.v2.improve import FLOOR_RTOL, improve                         # noqa: E402
from fimgat.v2.metrics_v2 import simulate                                 # noqa: E402
from fimgat.v2.proposer import community_moves                            # noqa: E402
from fimgat.v2.verify import CommunityFamily, disparity_se                # noqa: E402

FRACTIONS = (0.01, 0.05, 0.20)         # room runs: kappa = D_a (1 + f)
COST_FRACTIONS = (0.01, 0.05, 0.20)    # price budgets, as fractions of D_a (plus one SE)
SCAN_MAX = 5000                        # swaps scanned per case in the price diagnostic, cheapest first
OUT = os.path.join(RESULTS_V2, "parts", "c2diag")
LABELS = ["a", "b-1se"] + [f"b-{int(round(f * 100))}pct" for f in FRACTIONS]


# ------------------------------------------------------------------ helpers

def best_jaccard(h, family):
    """Largest Jaccard overlap of node set h with any member of family (0 when none overlaps)."""
    h = set(int(v) for v in h)
    best = 0.0
    for c in family:
        c = set(int(v) for v in c)
        inter = len(h & c)
        if inter:
            best = max(best, inter / len(h | c))
    return best


def t2_seeds(t2, graph, k, rho, variant):
    """T2's deployed (certified-path) set of one case, or None."""
    if t2 is None:
        return None
    r = t2[(t2.graph == graph) & (t2.k == k) & np.isclose(t2.rho, rho) & (t2.variant == variant)
           & (t2.deployment == "certified-path") & (t2.method == P.FIM)]
    return sorted(int(v) for v in json.loads(r.iloc[0].seeds)) if len(r) else None


def first_audit(case, state):
    """The auditor's family at state, exactly as improve()'s first audit draws it (multi-scale, seed 0)."""
    g, est = case.g, case.est
    cap = max(min(int(AUDIT_MAX_SIZE), g.n // 2), 1)
    found = audit_bands(g, node_exposure(est, state.exposure()), size_bands(MIN_COMMUNITY, cap, AUDIT_BAND_RATIO),
                        max_conductance=case.phi0, seed=0)
    return CommunityFamily(est, [c for c in found if c is not None and len(c)])


def price(case, state, floor, d_a, se, fam_f, scan_max=SCAN_MAX):
    """The labelled cost of community fairness at a labelled local optimum (see the module docstring)."""
    est = case.est
    floor_acc = floor + FLOOR_RTOL * abs(floor)
    imp, cols = I.improvement_matrix(state, floor_acc, device=est.device)
    cost = -imp                                          # labelled-disparity change of every swap; +inf: breaks floor
    flat = np.argsort(cost, axis=None, kind="stable")
    flat = flat[np.isfinite(cost.ravel()[flat])][:int(scan_max)]
    seeds = np.asarray(state.seeds, dtype=np.int64)
    fams = {"F": fam_f, "H": CommunityFamily(est, case.comms)}
    base = {nm: fam.state_disp(state) for nm, fam in fams.items()}
    budgets = {"se": se, **{f"{int(round(f * 100))}pct": f * d_a for f in COST_FRACTIONS}}
    first = {nm: math.nan for nm in fams}
    best = {(nm, b): 0.0 for nm in fams for b in budgets}
    for idx in flat:
        r, c = divmod(int(idx), cost.shape[1])
        dc = float(cost[r, c])
        u, v = int(seeds[r]), int(cols[c])
        for nm, fam in fams.items():
            if len(fam) < 2:
                continue
            delta = fam.disp_after_swap(state, u, v) - base[nm]
            if delta < -1e-12 and math.isnan(first[nm]):
                first[nm] = dc
            for b, cap in budgets.items():
                if dc <= cap and delta < best[(nm, b)]:
                    best[(nm, b)] = delta
    last = float(cost.ravel()[flat[-1]]) if len(flat) else math.nan
    full = len(flat) < int(scan_max)                     # every feasible swap was scanned
    out = {"n_feasible_scanned": int(len(flat)), "scan_complete": bool(full), "max_cost_scanned": last,
           "family_F_size": len(fams["F"]), "family_H_size": len(fams["H"]),
           "D_F_at_a": base["F"], "D_H_at_a_rr": base["H"]}
    for nm in fams:
        out[f"price_{nm}"] = first[nm]                   # NaN: no community-improving swap among those scanned
        for b, cap in budgets.items():
            out[f"best_dD{nm}_within_{b}"] = best[(nm, b)]
            out[f"budget_{b}"] = cap
            out[f"truncated_{b}"] = bool(not full and cap > last)
    return out


# ------------------------------------------------------------------ one case

def run_case(case, k, rho, args, rec, cov_log, price_log, t2, extra):
    g, est = case.g, case.est
    ref, _ = case.ref_for(k)
    floor = case.floor_for(k, rho)
    limit = P.time_limit(args)
    kw = dict(max_rounds=args.max_rounds, reference=list(ref), max_conductance=case.phi0,
              audit_max_size=AUDIT_MAX_SIZE, rng_seed=0)
    t0 = time.perf_counter()
    s_a, h_a = improve(est, list(ref), floor, proposer="simple", audit=False, max_seconds=limit, **kw)
    start1 = list(ref)
    p1 = path_states(start1, h_a)
    st_a = CoverState(est, s_a)
    d_a, se = float(st_a.disp), float(disparity_se(st_a))
    t_phase1 = time.perf_counter() - t0
    runs = [("a", None, math.nan), ("b-1se", d_a + KAPPA_SE * se, math.nan)] + \
        [(f"b-{int(round(f * 100))}pct", d_a * (1.0 + f), f) for f in FRACTIONS]
    for label, kappa, frac in runs:
        t1 = time.perf_counter()
        if kappa is None:
            paths, final, fam, acc, rounds = [p1], sorted(int(v) for v in s_a), [], 0, 0
        else:
            s_b, h_b = improve(est, list(s_a), floor, communities=None, audit=True, disp_cap=kappa,
                               max_seconds=limit, return_family=True,
                               proposer=lambda st, fm, rng: community_moves(st, fm), **kw)
            paths = [p1, path_states(list(s_a), h_b)]
            final = sorted(int(v) for v in s_b)
            fam = h_b[-1].get("communities", [])
            acc = int(sum(bool(h.get("accepted")) for h in h_b[1:]))
            rounds = len(h_b) - 1
        cands = deploy_candidates(paths, n_mid=args.deploy_mid)
        cert = certify_sets(case.est_b, cands, ref, float(rho), alpha=CERT_ALPHA)
        i = choose(cert["certificates"], "path")
        c = cert["certificates"][i]
        dep = [int(v) for v in c["seeds"]]
        secs = time.perf_counter() - t1 + (t_phase1 if label == "a" else 0.0)
        want = t2_seeds(t2, g.name, k, rho, {"a": "v2-a", "b-1se": "v2-b"}.get(label, ""))
        common = dict(kappa=NAN if kappa is None else kappa, fraction=frac, room=NAN if kappa is None else kappa - d_a,
                      disp_a=d_a, se_a=se, phase2_accepted=acc, phase2_rounds=rounds, family_size=len(fam),
                      deploy_position=int(c["position"]), deploy_is_reference=bool(c["is_reference"]),
                      deployed_is_final=dep == final, matches_t2=(NAN if want is None else bool(want == dep)),
                      **case.base_cols(), **extra)
        for deployment, seeds in (("certified-path", dep), ("final", final)):
            rec.add(g, k, rho, P.FIM, seeds, secs, ref_seeds=ref, variant=label, communities=case.comms,
                    deployment=deployment, **P.cert_cols(case.judge(seeds, k, rho)), **common)
        if fam:                                          # Lemma W1.1's terms for the deployed set
            prob, _ = simulate(g, dep, n_sims=case.ns, seed=mc_seed_for(g))
            tb = transfer_bound(np.asarray(prob, dtype=np.float64), case.comms, fam)
            cov_log.add({"graph": g.name, "k": k, "rho": rho, "variant": label, "kind": "transfer",
                         "D_H": tb["D_H"], "D_F": tb["D_F"], "gap_max": tb["gap_max"], "gap_min": tb["gap_min"],
                         "bound": tb["bound"], "holds": tb["holds"], "family_size": len(fam),
                         "jaccard_hmax": best_jaccard(case.comms[tb["h_max"]], fam) if tb["h_max"] >= 0 else NAN,
                         "jaccard_hmin": best_jaccard(case.comms[tb["h_min"]], fam) if tb["h_min"] >= 0 else NAN,
                         "size_hmax": len(case.comms[tb["h_max"]]) if tb["h_max"] >= 0 else NAN,
                         "size_hmin": len(case.comms[tb["h_min"]]) if tb["h_min"] >= 0 else NAN,
                         "status": "ok"})
        runs_dep[label] = dep
    t2_ = time.perf_counter()
    pr = price(case, st_a, floor, d_a, se, first_audit(case, st_a), args.scan_max)
    price_log.add({"graph": g.name, "k": k, "rho": rho, "disp_a": d_a, "se_a": se, "secs": time.perf_counter() - t2_,
                   "status": "ok", **pr})
    moved = {lab: rec.find(graph=g.name, k=k, rho=rho, variant=lab, deployment="final")[-1]["phase2_accepted"]
             for lab in LABELS[1:]}
    same = {lab: runs_dep[lab] == runs_dep["a"] for lab in LABELS[1:]}
    print(f"  {g.name} k={k} rho={rho}: D_a {d_a:.4f}, SE {se:.5f}; phase-2 moves {moved}; deployed = (a): {same}; "
          f"price F {pr['price_F']:.5f}, H {pr['price_H']:.5f}", flush=True)


runs_dep = {}                                            # label -> deployed set of the current case (for the log)


def case_done(rec, price_log, graph, k, rho):
    return bool(price_log.find(graph=graph, k=k, rho=rho, status="ok")) and all(
        rec.done(graph=graph, k=k, rho=rho, method=P.FIM, variant=lab, deployment="final") for lab in LABELS)


# ------------------------------------------------------------------ summary

def summarize(out_dir):
    from scipy.stats import wilcoxon
    room = pd.read_csv(os.path.join(out_dir, "room.csv"))
    cov = pd.read_csv(os.path.join(out_dir, "coverage.csv"))
    pr = pd.read_csv(os.path.join(out_dir, "price.csv"))
    room, cov, pr = (x[x.status == "ok"] for x in (room, cov, pr))
    md = ["# C2 diagnostics (EXPLORATORY; docs/design_v2.md section 22)\n",
          "Specified after T2's pre-registered null and before these runs. T2's verdict stands; nothing here "
          "replaces it.\n"]
    dep = room[room.deployment == "certified-path"]
    fin = room[room.deployment == "final"]
    md.append("## Room: does the phase act when it may spend more labelled fairness?\n")
    md.append("Paired with (a)'s deployment in each case (difference = run - (a); negative = lower).\n")
    md.append("| rho | run | cases | room (median) | phase-2 moves > 0 | deployed set differs from (a) | "
              "hidden-community disparity diff (median) | wins / losses | p | labelled disparity diff (median) | "
              "spread ratio diff (median) | control reproduces T2 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    key = ["graph", "k", "rho"]
    for rho in sorted(dep.rho.unique()):
        a = dep[(dep.rho == rho) & (dep.variant == "a")].set_index(key)
        for lab in LABELS[1:]:
            b = dep[(dep.rho == rho) & (dep.variant == lab)].set_index(key)
            f = fin[(fin.rho == rho) & (fin.variant == lab)].set_index(key)
            j = b.join(a, lsuffix="", rsuffix="_a", how="inner")
            if j.empty:
                continue
            dh = j.comm_disparity - j.comm_disparity_a
            dl = j.disparity - j.disparity_a
            ds = j.es_ratio - j.es_ratio_a
            nz = dh[dh != 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) >= 2 else math.nan
            ctrl = j.matches_t2.dropna().astype(str).isin(["True", "1", "1.0"]).mean() if lab == "b-1se" else math.nan
            md.append(f"| {rho:g} | {lab} | {len(j)} | {j.room.median():.5f} | {(f.phase2_accepted > 0).mean():.0%} | "
                      f"{(j.seeds != j.seeds_a).mean():.0%} | {dh.median():+.4f} | {(dh < 0).sum()} / {(dh > 0).sum()} | "
                      f"{p:.3g} | {dl.median():+.4f} | {ds.median():+.4f} | "
                      + ("-" if lab != "b-1se" else f"{ctrl:.0%}") + " |")
        ctrl_a = a.matches_t2.dropna().astype(str).isin(["True", "1", "1.0"]).mean() if len(a) else math.nan
        md.append(f"\n(a) reproduces T2's (a) deployment in {ctrl_a:.0%} of cases at rho {rho:g}.\n")
    md.append("## Coverage: does the auditor find the communities that set the hidden disparity?\n")
    t = cov[cov.kind == "transfer"]
    if len(t):
        md.append("| rho | run | cases | D_H (median) | D_F (median) | gap to h+ | gap to h- | bound non-vacuous (< 1) | "
                  "Jaccard with h+ | Jaccard with h- | family size |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for (rho, lab), x in t.groupby(["rho", "variant"]):
            md.append(f"| {rho:g} | {lab} | {len(x)} | {x.D_H.median():.4f} | {x.D_F.median():.4f} | "
                      f"{x.gap_max.median():.4f} | {x.gap_min.median():.4f} | {(x.bound < 1).mean():.0%} | "
                      f"{x.jaccard_hmax.median():.3f} | {x.jaccard_hmin.median():.3f} | {x.family_size.median():.0f} |")
    s = cov[cov.kind == "gt_conductance"]
    if len(s):
        md.append("\nGround-truth communities against the auditor's conductance cap phi0 (per sample):\n")
        md.append("| sample | communities | conductance median | within phi0 |")
        md.append("|---|---|---|---|")
        for _, r in s.iterrows():
            md.append(f"| {r.graph} | {int(r.n_communities)} | {r.conductance_median:.3f} | {r.share_within_phi0:.0%} |")
    md.append("\n## Price: what does community fairness cost in labelled fairness, one swap at a time?\n")
    md.append("At the labelled-only optimum. price = labelled-disparity cost of the cheapest floor-feasible swap that "
              "lowers the community disparity (F: the auditor's first family; H: the hidden communities on sample "
              "A). Shown also in units of SE and of D_a.\n")
    md.append("| rho | cases | price F (median) | in SE | / D_a | none found | price H (median) | in SE | / D_a | "
              "none found | scan complete |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for rho, x in pr.groupby("rho"):
        md.append(f"| {rho:g} | {len(x)} | {x.price_F.median():.5f} | {(x.price_F / x.se_a).median():.1f} | "
                  f"{(x.price_F / x.disp_a).median():.3%} | {x.price_F.isna().mean():.0%} | {x.price_H.median():.5f} | "
                  f"{(x.price_H / x.se_a).median():.1f} | {(x.price_H / x.disp_a).median():.3%} | "
                  f"{x.price_H.isna().mean():.0%} | {x.scan_complete.mean():.0%} |")
    budgets = ["se"] + [f"{int(round(f * 100))}pct" for f in COST_FRACTIONS]
    md.append("\nLargest community-disparity decrease one swap can buy within a labelled-cost budget (median over "
              "cases; budgets: one SE, then 1%, 5%, 20% of D_a):\n")
    md.append("| rho | family | " + " | ".join(budgets) + " |")
    md.append("|---|---|" + "---|" * len(budgets))
    for rho, x in pr.groupby("rho"):
        for nm in ("F", "H"):
            md.append(f"| {rho:g} | {nm} | " + " | ".join(f"{x[f'best_dD{nm}_within_{b}'].median():+.4f}"
                                                          for b in budgets) + " |")
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {path}")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--rhos", nargs="+", type=float, default=[RHO_C2, RHO_C1])
    ap.add_argument("--t2-samples", type=int, default=T2_SAMPLES)
    ap.add_argument("--t2-seeds", nargs="+", type=int, default=None)
    ap.add_argument("--t2-size", type=int, default=T2_SAMPLE_SIZE)
    ap.add_argument("--t2-communities", choices=["all", "top5000"], default="all")
    ap.add_argument("--t2-csv", default=os.path.join(RESULTS_V2, "t2.csv"),
                    help="T2's rows (with every parts/*/t2.csv) for the reproduction check")
    ap.add_argument("--scan-max", type=int, default=SCAN_MAX)
    ap.add_argument("--tuning", default=None)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--summarize", action="store_true", help="write summary.md from the rows and stop")
    ap.add_argument("--dry-run", action="store_true")
    P.add_pipeline_args(ap)
    ap.set_defaults(ref_file=os.path.join(RESULTS_V2, "parts", "t2", "references.json"))   # T2's cached references
    args = ap.parse_args()
    P.check_args(ap, args)
    if args.summarize:
        summarize(args.out_dir)
        return
    if args.R is None:
        args.R = T2_RR_PER_NODE                         # T2's R, so the states are T2's
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tuning = P.load_tuning(args.tuning)
    dev = device()
    pname, preset = preset_config(args.preset)
    ks, rhos = sorted({int(k) for k in args.ks}), sorted({float(r) for r in args.rhos})
    jobs = list(t2_jobs(args))
    print(f"C2 diagnostics (EXPLORATORY): {len(jobs)} samples x {len(ks)} budgets x {len(rhos)} floors; runs "
          f"{LABELS}; R={args.R}; delta={tuning['delta']} phi0={tuning['phi0']}; code_tag={code_tag()}; device={dev}",
          flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    rec = RecorderV2(os.path.join(args.out_dir, "room.csv"), n_sims=args.n_sims, preset=pname)
    cov_log = P.CsvLog(os.path.join(args.out_dir, "coverage.csv"), code_tag=code_tag())
    price_log = P.CsvLog(os.path.join(args.out_dir, "price.csv"), code_tag=code_tag())
    todo = {job.ident["graph"]: [(k, r) for k in ks for r in rhos
                                 if not case_done(rec, price_log, job.ident["graph"], k, r)] for job in jobs}
    n_left = sum(len(v) for v in todo.values())
    print(f"cases left: {n_left} of {len(jobs) * len(ks) * len(rhos)} (rough: 5-15 min each on a 4-CPU VM, plus about "
          f"25 min per sample for its cached stages)", flush=True)
    if args.dry_run or not n_left:
        return
    from summarize_v2 import load
    t2 = load(args.t2_csv)
    extra = {"test": "c2diag", "exploratory": True, "tuning_tag": tuning.get("code_tag", "")}
    bar = progress(n_left, "C2 diagnostics", unit="case")
    for job in jobs:
        cases = todo[job.ident["graph"]]
        if not cases:
            continue
        print(f"\n=== {job.ident['graph']}: {len(cases)} cases ===", flush=True)
        case = P.Case(job.loader, sorted({k for k, _ in cases}), args, dev, preset, phi0=tuning["phi0"],
                      delta=tuning["delta"], hook=lambda st, s, a, r, info: print(f"  {st:12s} {s:8.1f}s", flush=True))
        case.prepare()
        case.comms = job.comms(case.g)
        print("  " + case.describe() + f"; {len(case.comms)} evaluated ground-truth communities", flush=True)
        if not cov_log.find(graph=case.g.name, kind="gt_conductance"):
            cond = np.array([conductance(case.g, c) for c in case.comms]) if case.comms else np.zeros(0)
            cov_log.add({"graph": case.g.name, "kind": "gt_conductance", "n_communities": len(cond),
                         "conductance_median": float(np.median(cond)) if len(cond) else NAN,
                         "share_within_phi0": float((cond <= case.phi0).mean()) if len(cond) else NAN,
                         "phi0": case.phi0, "status": "ok"})
        for k, rho in cases:
            runs_dep.clear()
            run_case(case, k, rho, args, rec, cov_log, price_log, t2, extra)
            bar.update(1)
        del case
        P.free()
    bar.close()


if __name__ == "__main__":
    main()
