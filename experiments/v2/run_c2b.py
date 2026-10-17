"""Test C2b (docs/design_v2.md section 25): pre-registered on 2026-10-18, before any method ran on its graph.

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore run_c2b.py --dry-run
    PYTHONIOENCODING=utf-8 python -u -W ignore run_c2b.py              # both floors; resumable
    PYTHONIOENCODING=utf-8 python -u -W ignore run_c2b.py --summarize  # the pre-registered verdicts

T2 (C2) was a null on com-Youtube, and its diagnostics traced it to coverage (section 22): YouTube's ground-truth
groups have conductance near 0.97, outside the auditor's search class (phi0 = 0.4). C2b asks T2's question where the
hidden communities are cohesive: SNAP com-Amazon (config.C2B_DATASET), with SNAP's top-5000 communities
(config.C2B_COMMUNITIES), whose median conductance inside T2-size samples is 0.06-0.07, all within phi0.

Fixed in advance (config and section 25):
  samples   T2's design: config.T2_SAMPLES BFS samples of config.T2_SAMPLE_SIZE nodes (seeds 1, 2, ...), labelled groups
            by degree, R = config.T2_RR_PER_NODE, the tuned delta and phi0, every budget, rho 0.97 (primary) and 0.99
  runs      from the same labelled-only optimum (c2_diagnostics.run_case, which reproduces T2 exactly):
              a       the labelled-only search, T2's (a)
              b-1se   the method: the epsilon-constraint phase with T2's cap kappa = D_a + 1 SE (H1)
              b-5pct  the same phase with kappa = D_a (1 + config.C2B_FRACTION) (H2)
            each deployed by T2's path rule on B and scored by forward Monte Carlo
  metric    the disparity of the hidden communities (top 5,000, at least config.EVAL_MIN_COMMUNITY members in the
            sample); secondary: over SNAP's "all" communities, and over those of them within phi0 in the sample
  tests     per case (sample, budget) at rho 0.97: (run) - (a), two-sided Wilcoxon signed-rank, each at
            config.C2B_ALPHA (Bonferroni over H1 and H2), with at least config.T2_MIN_PAIRS cases.
              H1 passes: p < C2B_ALPHA, a negative median, and no significant rise in labelled disparity (T2's rule)
              H2 passes: p < C2B_ALPHA and a negative median; its labelled cost (bounded by the cap on A) is reported
            Everything else (rho 0.99, final sets, the secondary metrics, coverage and price) is descriptive.

Outputs, in results/v2/parts/c2b/: c2b.csv (RecorderV2 rows; variant = run, deployment "certified-path" or "final"),
coverage.csv, price.csv, references.json, and summary.md (--summarize). Ground-truth communities only score sets and
price swaps; no set is chosen or deployed from them.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c2_diagnostics as C                                                # noqa: E402
import pipeline_v2 as P                                                   # noqa: E402
from summarize_v2 import ALPHA, wilcoxon                                  # noqa: E402  (T2's test and level)
from fimgat.v2.config import (BUDGETS, C2B_ALPHA, C2B_COMMUNITIES, C2B_DATASET, C2B_FRACTION, RESULTS_V2,   # noqa: E402
                              RHO_C1, RHO_C2, T2_MIN_PAIRS, T2_RR_PER_NODE, T2_SAMPLE_SIZE, T2_SAMPLES)

OUT = os.path.join(RESULTS_V2, "parts", "c2b")
ROWS = "c2b.csv"
FRACTIONS = (C2B_FRACTION,)
LABELS = C.labels_for(FRACTIONS)                 # ["a", "b-1se", "b-5pct"]
H = {"H1": "b-1se", "H2": C.labels_for(FRACTIONS)[2]}
METRICS = ("comm_disparity", "comm_disparity_all", "comm_disparity_visible", "disparity")


def comparisons(rows):
    """[(rho, deployment, run, metric, n, median, p)] of (run) - (a), paired by case (graph, k)."""
    out = []
    for rho in sorted(rows.rho.unique()):
        for dep in ("certified-path", "final"):
            f = rows[(rows.rho == rho) & (rows.deployment == dep)]
            per = {v: f[f.variant == v].groupby(["graph", "k"])[[m for m in METRICS if m in f]].mean()
                   for v in LABELS}
            for v in LABELS[1:]:
                j = per[v].join(per["a"], lsuffix="_x", rsuffix="_a", how="inner")
                for m in METRICS:
                    if m + "_x" not in j:
                        continue
                    n, med, p = wilcoxon((j[m + "_x"] - j[m + "_a"]).values)
                    out.append((float(rho), dep, v, m, n, med, p))
    return out


def verdicts(rows):
    """The pre-registered H1 and H2 verdicts: {name: dict}."""
    comp = {(rho, dep, v, m): (n, med, p) for rho, dep, v, m, n, med, p in comparisons(rows)}
    out = {}
    for name, run in H.items():
        n, med, p = comp.get((RHO_C2, "certified-path", run, "comm_disparity"), (0, math.nan, math.nan))
        nl, medl, pl = comp.get((RHO_C2, "certified-path", run, "disparity"), (0, math.nan, math.nan))
        lower = n >= T2_MIN_PAIRS and p == p and p < C2B_ALPHA and med < 0
        rise = pl == pl and pl < ALPHA and medl > 0
        ok = lower and (not rise if name == "H1" else True)
        why = ("underpowered" if n < T2_MIN_PAIRS else "no significant decrease" if not lower
               else "labelled disparity rose significantly" if (name == "H1" and rise) else "")
        out[name] = {"run": run, "cases": n, "median": med, "p": p, "labelled_median": medl, "labelled_p": pl,
                     "pass": bool(ok), "why": why}
    return out


def summarize(out_dir):
    rows = pd.read_csv(os.path.join(out_dir, ROWS), float_precision="round_trip", low_memory=False)
    rows = rows[(rows.status == "ok") & (rows.method == P.FIM)]
    v = verdicts(rows)
    md = ["# C2b: T2's test where the hidden communities are cohesive (PRE-REGISTERED; docs/design_v2.md section 25)\n",
          f"Graph {C2B_DATASET}, hidden communities: SNAP's {C2B_COMMUNITIES} (primary). Primary: rho {RHO_C2:g}, "
          f"sets deployed by the path rule; each hypothesis at alpha {C2B_ALPHA:g} (Bonferroni over two).\n"]
    for name in ("H1", "H2"):
        x = v[name]
        what = ("the method (kappa = D_a + 1 SE)" if name == "H1"
                else f"the scaled cap (kappa = D_a (1 + {C2B_FRACTION:g}))")
        md.append(f"- **{name}, {what}: {'PASS' if x['pass'] else 'FAIL'}**" + (f" ({x['why']})" if x["why"] else "")
                  + f". Hidden-community disparity, {x['run']} - a over {x['cases']} cases: median {x['median']:+.4f}, "
                  f"p {x['p']:.3g}; labelled disparity: median {x['labelled_median']:+.4f}, p {x['labelled_p']:.3g}.")
    md.append("\n## Every comparison (run - a, paired by case; negative = lower)\n")
    md.append("| rho | sets | run | metric | cases | median difference | p |")
    md.append("|---|---|---|---|---|---|---|")
    for rho, dep, run, m, n, med, p in comparisons(rows):
        md.append(f"| {rho:g} | {dep} | {run} | {m} | {n} | {med:+.4f} | {p:.3g} |")
    md.append("\n## Descriptive: room, coverage and price (as in section 22)\n")
    md += C.summary_lines(out_dir, ROWS, LABELS)
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {path}")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--rhos", nargs="+", type=float, default=[RHO_C2, RHO_C1])
    ap.add_argument("--t2-samples", type=int, default=T2_SAMPLES)
    ap.add_argument("--t2-seeds", nargs="+", type=int, default=None)
    ap.add_argument("--t2-size", type=int, default=T2_SAMPLE_SIZE)
    ap.add_argument("--scan-max", type=int, default=C.SCAN_MAX)
    ap.add_argument("--tuning", default=None)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--summarize", action="store_true", help="write summary.md (the verdicts) from the rows and stop")
    ap.add_argument("--dry-run", action="store_true")
    P.add_pipeline_args(ap)
    ap.set_defaults(ref_file=os.path.join(OUT, "references.json"))
    args = ap.parse_args()
    P.check_args(ap, args)
    if args.summarize:
        summarize(args.out_dir)
        return
    if args.R is None:
        args.R = T2_RR_PER_NODE
    args.t2_dataset, args.t2_communities, args.t2_csv = C2B_DATASET, C2B_COMMUNITIES, None
    C.run_all(args, FRACTIONS, "C2b (PRE-REGISTERED, design section 25)", test="c2b", exploratory=False,
              rows_name=ROWS, secondary=True)


if __name__ == "__main__":
    main()
