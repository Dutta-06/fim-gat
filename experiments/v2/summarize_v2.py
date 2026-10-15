"""The pre-registered analysis of FIM-GAT v2 (docs/design_v2.md sections 10, 18 and 19): results/v2/*.csv in,
results/v2/summary_v2.md and summary_*.csv out. Nothing here is chosen after seeing results; every rule below was
fixed before the first v2 run.

    cd experiments/v2 && PYTHONIOENCODING=utf-8 python -W ignore summarize_v2.py

T1 (C1). The method is FIM-GAT v2 variant v2-b with the tuned start (tuning.json), the simple proposer and certified
deployment by Theorem 2's rule (deployment "certified"). Per case (dataset, attribute, k, rho): its Monte Carlo
disparity, averaged over train seeds, against every external baseline whose set passed the certificate in that case
(internal variants are ablations, not baselines). The headline comparison is against the best certified external
baseline of each case. Wilcoxon: paired, two-sided, over test graphs (dataset x attribute) of the mean over budgets of
(method - baseline), at alpha 0.05, for each rho; rho = 0.99 is the headline and "holds across the range" means a
significant win at every rho in config.RHO_HOLDS = (0.97, 0.98, 0.99). rho = 1.0 stays in the tables, descriptively:
there almost only the reference itself can be certified (design section 21, 2026-10-14). Each ablation row (one
component changed, section 18.2) is compared with the method the same way; the "deployment" rows are the method's final
search set and its path-rule deployment.

Headline and supplement (design section 20, fixed before any result): the headline T1 test uses the test graphs
whose groups are real attributes (Rice, Twitter-political, Pokec-n, full Pokec, Instagram); the graphs with degree
groups (Epinions, Twitter-ego, com-Youtube, LinkedIn) are reported as a supplement, and all graphs together as well.

The results matrix (summary_t1_matrix.csv and the tables in summary_v2.md): one table per rho, a row per test graph, a
column per method (the method and every external baseline), each cell the mean Monte Carlo disparity over the budgets
where that method's set passed the certificate, with the number of such budgets; "-" when none did. The lower-bound
column is the mean of bounds.disparity_lower_bound over the budgets where it was computed (t1_bounds.csv). A per-budget
version is written to summary_t1_matrix_by_k.csv.

Certificate-size sensitivity (design section 20): t1_certsens.csv, written by run_t1_t2.py with --cert-R on the fixed
subset of protocol section 1, is summarised by B's size: how often the method deployed a searched set rather than the
reference, its mean disparity, and the baselines' certificate pass rate.

T2 (C2). Cases are T2 samples x budgets; at rho = 0.97 (primary) and 0.99. Variant v2-b against v2-a on the sets
deployed by the path rule (deployment "certified-path", design section 21; both variants deployed by the same rule):
primary metric comm_disparity (ground-truth communities). Pass: Wilcoxon two-sided p < 0.05 with a negative median
difference, at least config.T2_MIN_PAIRS cases, and no significant rise of labelled disparity (two-sided p < 0.05
with a positive median). v2-b0 against v2-a, Theorem 2's deployments and the final search sets are reported the same
way, as secondary. The held-out-attribute secondary metric comes from T1's v2-b and v2-a path-rule rows.

T3 (design sections 10, 18.3 and 21). Reported as "not run under the pre-registered rule" unless tuning.json's W3 rule
said to train. Otherwise, from t3.csv (run_t3.py): (i) per case (test graph, budget) the speed-up of each learned
proposer over the simple hybrid to the simple hybrid's final RR disparity, from the searches' own traces; a test graph
is faster when the median over budgets of the median over train seeds is at least config.T3_SPEEDUP_MIN; (ii) at
config.T3_PRIMARY_BUDGET seconds, a paired two-sided Wilcoxon over test graphs of the mean over budgets of (learned,
seed-averaged - simple) Monte Carlo disparity. Pass: (i) on more than half of the test graphs, or (ii) p < 0.05 with
a negative median. recall@C comes from t3_recall.csv.

T4 (C3). Per graph and method: cached seconds (prep_s), mean per-query seconds (query_s), and the time for
Q in {1, 10, 100} queries, prep + Q x query, next to the method's mean disparity and certificate pass rate. FIM-GAT's
rows are its Theorem 2 deployments (deployment "certified"), whose per-query time includes certification.

Pareto (design section 21; descriptive). Per test graph and budget, the method's curve (its certified deployments at
every rho of RHO_REPORT, seed-averaged) against every external baseline's distinct sets (T1 rows) and BSM's whole tau
sweep (t1_pareto.csv), each point (Monte Carlo spread relative to the reference, es_ratio; Monte Carlo disparity): which
baseline points the method's curve dominates, which dominate one of its points, and which are incomparable (most
often: less spread, less disparity). summary_pareto.csv holds every point; --figures draws k = 50 for every test graph
into figures/.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fimgat.v2.config import (RESULTS_V2, RHO_C1, RHO_C2, RHO_HOLDS, RHO_REPORT, T2_MIN_PAIRS,   # noqa: E402
                              T3_PRIMARY_BUDGET, T3_RECALL_C, T3_SPEEDUP_MIN, TUNING_FILE)

FIM = "FIM-GAT v2"
INTERNAL = ("LocalSearch", "Simple-v2")         # internal variants of the method (design section 18.2)
ALPHA = 0.05
QS = (1, 10, 100)
T2_PRIMARY = "certified-path"                   # T2's deployment: the path rule (design section 21)
PARETO_K = 50                                   # the budget drawn by --figures (the case studies' budget)


def wilcoxon(diff):
    """(n, median, p) of a two-sided Wilcoxon signed-rank test on paired differences (NaN p below 2 pairs or when
    every difference is 0)."""
    d = np.asarray([x for x in diff if x == x], dtype=np.float64)
    if len(d) < 2 or np.all(d == 0):
        return len(d), float(np.median(d)) if len(d) else np.nan, np.nan
    from scipy.stats import wilcoxon as w
    return len(d), float(np.median(d)), float(w(d).pvalue)


def verdict(n, med, p):
    if p != p:
        return "n/a"
    if p < ALPHA:
        return "win" if med < 0 else "loss"
    return "no difference"


def truthy(s):
    return s.astype(str).isin(["True", "1", "1.0", "true"])


KEY = ["dataset", "attribute", "graph", "k", "rho", "method", "variant", "start", "proposer", "deployment",
       "train_seed", "delta", "phi0", "cert_R", "R", "budget_s"]


def load(path):
    """A results CSV merged with the same file of every parallel job (docs/cloud_runbook.md): <dir>/<name> and each
    <dir>/parts/*/<name>, the job's name in column "part" ("" for the shared file). Rows with status "ok" only; None
    when no file exists."""
    d, name = os.path.split(path)
    frames = []
    for f in [path] + sorted(glob.glob(os.path.join(d, "parts", "*", name))):
        if os.path.exists(f) and os.path.getsize(f) > 0:
            x = pd.read_csv(f, float_precision="round_trip")
            x["part"] = "" if f == path else os.path.basename(os.path.dirname(f))
            frames.append(x)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True, sort=False)
    return df[df["status"] == "ok"].copy() if "status" in df else df


def duplicates(df):
    """Rows whose query key (KEY columns present) occurs more than once, e.g. a case run by two jobs."""
    if df is None or df.empty:
        return 0
    cols = [c for c in KEY if c in df]
    return int(df.astype({c: str for c in cols}).duplicated(subset=cols, keep="first").sum())


# ------------------------------------------------------------------ T1

def t1(d, start, out_rows, md, out_dir):
    fim = d[d.method == FIM]
    sel = lambda f, v, s, p, dep: f[(f.variant == v) & (f.start == s) & (f.proposer == p) & (f.deployment == dep)]
    case = ["dataset", "attribute", "k", "rho"]
    method = sel(fim, "v2-b", start, "simple", "certified").groupby(case).disparity.mean().rename("method")
    base = d[(d.method != FIM) & ~d.method.isin(INTERNAL)].copy()
    base = base[truthy(base.cert_passed)]
    md.append("## T1 (C1): disparity at equal spread\n")
    md.append(f"Method: v2-b, {start} start, simple hybrid, certified deployment. "
              f"{method.index.droplevel(['k', 'rho']).nunique()} test graphs (dataset x attribute), "
              f"{len(method)} cases.\n")
    if base.empty or method.empty:
        md.append("No certified external baseline or no method rows yet.\n")
        return
    bdisp = base.groupby(case + ["method"]).disparity.mean().unstack("method")
    best = bdisp.min(axis=1).rename("best_baseline")
    cases = pd.concat([method, best, bdisp], axis=1).reset_index()
    cases.to_csv(os.path.join(out_dir, "summary_t1_cases.csv"), index=False)
    for label, sub_cases in (("real-attribute graphs (headline)", cases[cases.attribute != "degree"]),
                             ("degree-group graphs (supplement)", cases[cases.attribute == "degree"]),
                             ("all test graphs", cases)):
        md.append(f"### {label}\n")
        if sub_cases.empty:
            md.append("No rows yet.\n")
            continue
        md.append("| rho | comparison | test graphs | median of mean difference | p | verdict | case wins / ties / losses |")
        md.append("|---|---|---|---|---|---|---|")
        holds = {}
        for rho in sorted(sub_cases.rho.unique()):
            c = sub_cases[sub_cases.rho == rho]
            for name in ["best_baseline"] + [m for m in bdisp.columns]:
                sub = c.dropna(subset=["method", name])
                if sub.empty:
                    continue
                diff = (sub.method - sub[name])
                per_graph = diff.groupby([sub.dataset, sub.attribute]).mean()
                n, med, p = wilcoxon(per_graph.values)
                v = verdict(n, med, p)
                holds.setdefault(name, {})[round(float(rho), 6)] = v == "win"
                wins, ties, losses = int((diff < 0).sum()), int((diff == 0).sum()), int((diff > 0).sum())
                md.append(f"| {rho:g} | {name} | {n} | {med:+.4f} | {p:.3g} | {v} | {wins} / {ties} / {losses} |")
                out_rows.append({"test": "T1", "subset": label, "rho": rho, "comparison": name, "n_graphs": n,
                                 "median_diff": med, "p": p, "verdict": v, "case_wins": wins, "case_ties": ties,
                                 "case_losses": losses})
        full = sorted(round(float(r), 6) for r in RHO_HOLDS)
        md.append("")
        md.append(f"Holds across the range (a significant win at every rho in {full}; rho = 1.0 is reported above but "
                  f"not required, design section 21):\n")
        for name, vs in holds.items():
            miss = [r for r in full if r not in vs]
            ok = not miss and all(vs[r] for r in full)
            md.append(f"- {name}: {'yes' if ok else 'no'}" + (f" (no paired cases at rho {miss})" if miss else ""))
            out_rows.append({"test": "T1-holds", "subset": label, "comparison": name, "verdict": "yes" if ok else "no",
                             "missing_rho": json.dumps(miss)})
        md.append("")

    # ablation rows (design section 18.2): one component changed against the method, same certificate
    other = "greedy" if start == "learned" else "learned"
    rows = {"start: " + other: sel(fim, "v2-b", other, "simple", "certified"),
            "proposer: exhaustive": sel(fim, "v2-b", start, "exhaustive", "certified"),
            "acceptance: (a) labelled only": sel(fim, "v2-a", start, "simple", "certified"),
            "acceptance: (b0) published rule": sel(fim, "v2-b0", start, "simple", "certified"),
            "deployment: final search set": sel(fim, "v2-b", start, "simple", "final"),
            "deployment: path rule (T2's)": sel(fim, "v2-b", start, "simple", "certified-path")}
    md.append("### Ablation (one component changed; difference = method - row, negative favours the method)\n")
    md.append("| rho | row | test graphs | mean disparity (row) | median of mean difference | p | verdict |")
    md.append("|---|---|---|---|---|---|---|")
    for rho in sorted(cases.rho.unique()):
        m_r = method.xs(rho, level="rho")
        for label, r in rows.items():
            rr = r[r.rho == rho].groupby(["dataset", "attribute", "k"]).disparity.mean()
            j = pd.concat([m_r.rename("m"), rr.rename("r")], axis=1).dropna()
            if j.empty:
                md.append(f"| {rho:g} | {label} | 0 | n/a | n/a | n/a | not run |")
                continue
            per_graph = (j.m - j.r).groupby(level=["dataset", "attribute"]).mean()
            n, med, p = wilcoxon(per_graph.values)
            md.append(f"| {rho:g} | {label} | {n} | {j.r.mean():.4f} | {med:+.4f} | {p:.3g} | {verdict(n, med, p)} |")
            out_rows.append({"test": "T1-ablation", "rho": rho, "comparison": label, "n_graphs": n,
                             "median_diff": med, "p": p, "verdict": verdict(n, med, p)})
    final = sel(fim, "v2-b", start, "simple", "final")
    if not final.empty and "es_ratio" in final:
        below = int((final.es_ratio < final.rho).sum())
        md.append(f"\nFinal search sets below the floor by Monte Carlo (es_ratio < rho): {below} of {len(final)}; "
                  f"not certified: {int((~truthy(final.cert_passed)).sum())} of {len(final)}.\n")


def t1_matrix(d_all, start, bounds, md, out_dir):
    """The results matrix: a table per rho, a row per test graph, a column per method (see the module docstring)."""
    fim = d_all[(d_all.method == FIM) & (d_all.variant == "v2-b") & (d_all.start == start)
                & (d_all.proposer == "simple") & (d_all.deployment == "certified")].copy()
    fim["method"] = "FIM-GAT v2"
    base = d_all[(d_all.method != FIM) & ~d_all.method.isin(INTERNAL)].copy()
    rows = pd.concat([fim, base])
    rows = rows[rows.status == "ok"] if "status" in rows else rows
    rows["ok"] = truthy(rows.cert_passed)
    per_k = (rows.groupby(["rho", "dataset", "attribute", "method", "k"])
             .agg(disparity=("disparity", "mean"), certified=("ok", "all")).reset_index())
    per_k.to_csv(os.path.join(out_dir, "summary_t1_matrix_by_k.csv"), index=False)
    cert = per_k[per_k.certified]
    cell = (cert.groupby(["rho", "dataset", "attribute", "method"])
            .agg(disparity=("disparity", "mean"), n_budgets=("k", "nunique")).reset_index())
    cell.to_csv(os.path.join(out_dir, "summary_t1_matrix.csv"), index=False)
    lb = None
    if bounds is not None and not bounds.empty:
        b = bounds[bounds.status == "ok"] if "status" in bounds else bounds
        lb = b.groupby(["rho", "dataset", "attribute"]).lb.mean()
    methods = ["FIM-GAT v2"] + sorted(m for m in rows.method.unique() if m != "FIM-GAT v2")
    budgets = per_k.groupby(["rho", "dataset", "attribute"]).k.nunique()
    md.append("### Results matrix (mean disparity over certified budgets; n = budgets certified; - = none; bold = lowest among the methods certified at every budget; the paired tests above are the formal comparison)\n")
    for rho in sorted(rows.rho.unique()):
        md.append(f"**rho = {rho:g}**\n")
        md.append("| test graph | " + " | ".join(methods) + " | lower bound |")
        md.append("|---|" + "---|" * (len(methods) + 1))
        c = cell[cell.rho == rho].set_index(["dataset", "attribute", "method"])
        for (ds, at), nk in budgets.xs(rho, level="rho").items():
            vals = {m: c.loc[(ds, at, m)] if (ds, at, m) in c.index else None for m in methods}
            best = min((v.disparity for v in vals.values() if v is not None and int(v.n_budgets) == nk),
                       default=None)
            out = []
            for m in methods:
                v = vals[m]
                if v is None:
                    out.append("-")
                    continue
                txt = f"{v.disparity:.4f} ({int(v.n_budgets)}/{nk})"
                full = int(v.n_budgets) == nk
                out.append(f"**{txt}**" if best is not None and full and v.disparity == best else txt)
            lbv = lb.get((rho, ds, at)) if lb is not None else None
            md.append(f"| {ds} / {at} | " + " | ".join(out) + f" | {'-' if lbv is None or lbv != lbv else f'{lbv:.4f}'} |")
        md.append("")
    rate = rows.groupby(["rho", "method"]).ok.mean().unstack("rho")
    md.append("Certificate pass rate per method (share of all its sets):\n")
    md.append("| method | " + " | ".join(f"rho {r:g}" for r in rate.columns) + " |")
    md.append("|---|" + "---|" * len(rate.columns))
    for m, r in rate.iterrows():
        md.append(f"| {m} | " + " | ".join(f"{x:.0%}" if x == x else "-" for x in r.values) + " |")
    md.append("")


def cert_sensitivity(ds, start, md):
    md.append("## Certificate-size sensitivity\n")
    if ds is None or ds.empty:
        md.append("No t1_certsens.csv rows yet.\n")
        return
    f = ds[(ds.method == FIM) & (ds.variant == "v2-b") & (ds.start == start) & (ds.proposer == "simple")
           & (ds.deployment == "certified")]
    b = ds[(ds.method != FIM) & ~ds.method.isin(INTERNAL)]
    md.append("| cert R | method: searched set deployed | method: mean disparity | baselines certified |")
    md.append("|---|---|---|---|")
    for r in sorted(ds.cert_R.unique()):
        fr, br = f[f.cert_R == r], b[b.cert_R == r]
        searched = 1.0 - truthy(fr.deploy_is_reference).mean() if len(fr) else float("nan")
        md.append(f"| {r:g} | {searched:.0%} | {fr.disparity.mean():.4f} | {truthy(br.cert_passed).mean():.0%} |")
    md.append("")


# ------------------------------------------------------------------ T2

def t2(d2, d1, start, out_rows, md):
    md.append("## T2 (C2): hidden-community disparity, (b) against (a)\n")
    if d2 is None or d2.empty:
        md.append("No T2 rows yet.\n")
        return
    fim = d2[(d2.method == FIM) & (d2.start == start) & (d2.proposer == "simple")]
    case = ["graph", "k"]
    md.append(f"Primary: rho {RHO_C2:g}, sets deployed by the path rule ({T2_PRIMARY}); every other row is secondary.\n")
    md.append("| rho | sets | comparison | metric | cases | median difference | p | verdict |")
    md.append("|---|---|---|---|---|---|---|---|")
    primary = {}
    for rho in (RHO_C2, RHO_C1):
        for dep in (T2_PRIMARY, "certified", "final"):
            f = fim[(fim.rho == rho) & (fim.deployment == dep)]
            per = {v: f[f.variant == v].groupby(case)[["comm_disparity", "disparity"]].mean() for v in
                   ("v2-b", "v2-b0", "v2-a")}
            for v in ("v2-b", "v2-b0"):
                j = per[v].join(per["v2-a"], lsuffix="_x", rsuffix="_a", how="inner").dropna()
                for metric in ("comm_disparity", "disparity"):
                    n, med, p = wilcoxon((j[metric + "_x"] - j[metric + "_a"]).values)
                    if metric == "comm_disparity":
                        v_ = verdict(n, med, p)
                        if n < T2_MIN_PAIRS:
                            v_ += f" (underpowered: {n} < {T2_MIN_PAIRS} cases)"
                    else:
                        v_ = "significant rise" if (p == p and p < ALPHA and med > 0) else "no significant rise"
                    if rho == RHO_C2 and dep == T2_PRIMARY and v == "v2-b":
                        primary[metric] = (n, v_)
                    md.append(f"| {rho:g} | {dep} | {v} - v2-a | {metric} | {n} | {med:+.4f} | {p:.3g} | {v_} |")
                    out_rows.append({"test": "T2", "rho": rho, "sets": dep, "comparison": f"{v} - v2-a",
                                     "metric": metric, "n_cases": n, "median_diff": med, "p": p, "verdict": v_})
    if primary:
        (n, vc), (_, vd) = primary.get("comm_disparity", (0, "n/a")), primary.get("disparity", (0, "n/a"))
        ok = vc == "win" and n >= T2_MIN_PAIRS and vd == "no significant rise"
        md.append(f"\nT2 pass (pre-registered rule): {'yes' if ok else 'no'} (hidden communities: {vc}; labelled "
                  f"disparity: {vd}; {n} cases).\n")
        out_rows.append({"test": "T2-pass", "rho": RHO_C2, "sets": T2_PRIMARY, "verdict": "pass" if ok else "fail",
                         "n_cases": n})
    if d1 is not None and not d1.empty and "heldout_disparity" in d1:
        md.append("\nSecondary, held-out attributes (T1 rows, path-rule deployments):\n")
        f = d1[(d1.method == FIM) & (d1.start == start) & (d1.proposer == "simple") & (d1.deployment == T2_PRIMARY)]
        for rho in (RHO_C2, RHO_C1):
            g = f[f.rho == rho]
            b = g[g.variant == "v2-b"].groupby(["dataset", "attribute", "k"]).heldout_disparity.mean()
            a = g[g.variant == "v2-a"].groupby(["dataset", "attribute", "k"]).heldout_disparity.mean()
            n, med, p = wilcoxon((b - a).dropna().values)
            md.append(f"- rho {rho:g}: v2-b - v2-a held-out-attribute disparity over {n} cases: median {med:+.4f}, "
                      f"p {p:.3g} ({verdict(n, med, p)})")
    md.append("")


# ------------------------------------------------------------------ Pareto (descriptive, design section 21)

CASE_COLS = ["dataset", "attribute", "k"]


def pareto_points(d1, dp, start):
    """Every point of the spread-disparity plane, per (dataset, attribute, k): source "method" (its Theorem 2
    deployments per rho, seed-averaged), "baseline" (every external baseline's distinct sets in T1, with the floors
    they were recorded at) and "bsm_sweep" (t1_pareto.csv, one per tau). Columns: CASE_COLS, source, method, rho, rhos,
    tau, es_ratio (Monte Carlo spread / reference spread), disparity (Monte Carlo, labelled groups)."""
    parts = []
    fim = d1[(d1.method == FIM) & (d1.variant == "v2-b") & (d1.start == start) & (d1.proposer == "simple")
             & (d1.deployment == "certified")]
    if not fim.empty:
        m = (fim.groupby(CASE_COLS + ["rho"]).agg(es_ratio=("es_ratio", "mean"), disparity=("disparity", "mean"))
             .reset_index())
        m["source"], m["method"], m["tau"], m["rhos"] = "method", FIM, np.nan, m.rho.map(lambda r: json.dumps([r]))
        parts.append(m)
    base = d1[(d1.method != FIM) & ~d1.method.isin(INTERNAL)]
    if not base.empty:
        b = (base.groupby(CASE_COLS + ["method", "seeds"])
             .agg(es_ratio=("es_ratio", "first"), disparity=("disparity", "first"),
                  rhos=("rho", lambda s: json.dumps(sorted({float(x) for x in s})))).reset_index()
             .drop(columns="seeds"))
        b["source"], b["tau"], b["rho"] = "baseline", np.nan, np.nan
        parts.append(b)
    if dp is not None and not dp.empty:
        s = dp[dp.method == "BSM"].drop_duplicates(CASE_COLS + ["seeds"])
        s = s[CASE_COLS + ["tau", "es_ratio", "disparity"]].copy()
        s["source"], s["method"], s["rho"], s["rhos"] = "bsm_sweep", "BSM (every tau)", np.nan, "[]"
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    pts = pd.concat(parts, ignore_index=True)
    return pts.dropna(subset=["es_ratio", "disparity"]).reset_index(drop=True)


def dominance(pts):
    """Adds dominated_by_method and dominates_method: whether some method point of the same case has at least the
    point's spread ratio and at most its disparity (one strictly), and the reverse. False for the method's points."""
    pts = pts.copy()
    pts["dominated_by_method"], pts["dominates_method"] = False, False
    for _, g in pts.groupby(CASE_COLS):
        m = g[g.source == "method"][["es_ratio", "disparity"]].to_numpy(dtype=np.float64)
        if not len(m):
            continue
        for i in g.index[g.source != "method"]:
            x, y = float(pts.at[i, "es_ratio"]), float(pts.at[i, "disparity"])
            pts.at[i, "dominated_by_method"] = bool(np.any((m[:, 0] >= x) & (m[:, 1] <= y)
                                                           & ((m[:, 0] > x) | (m[:, 1] < y))))
            pts.at[i, "dominates_method"] = bool(np.any((x >= m[:, 0]) & (y <= m[:, 1])
                                                        & ((x > m[:, 0]) | (y < m[:, 1]))))
    return pts


def pareto(d1, dp, start, md, out_dir, figures=False):
    md.append("## Spread-disparity trade-off (descriptive)\n")
    pts = pareto_points(d1, dp, start) if d1 is not None else pd.DataFrame()
    if pts.empty or not (pts.source == "method").any():
        md.append("No method rows yet.\n")
        return
    pts = dominance(pts)
    pts.to_csv(os.path.join(out_dir, "summary_pareto.csv"), index=False)
    md.append("Per test graph, over budgets: the external baselines' sets (T1 rows and BSM's whole tau sweep) that the "
              f"method's curve (its deployments at rho in {list(RHO_REPORT)}) dominates, that dominate one of its points, "
              "and that are incomparable, with how many of those have less spread than every method point. Point "
              "estimates from Monte Carlo; the paired tests of T1 remain the formal comparison.\n")
    md.append("| test graph | baseline sets | dominated by the method | dominate a method point | incomparable "
              "| of which below the method's least spread |")
    md.append("|---|---|---|---|---|---|")
    for (ds, at), g in pts.groupby(["dataset", "attribute"]):
        ext = g[g.source != "method"]
        if ext.empty:
            continue
        low = 0
        for _, c in ext.groupby("k"):
            mk = g[(g.source == "method") & (g.k == c.k.iloc[0])]
            if len(mk):
                low += int(((~c.dominated_by_method) & (~c.dominates_method)
                            & (c.es_ratio < mk.es_ratio.min())).sum())
        n, dom, domd = len(ext), int(ext.dominated_by_method.sum()), int(ext.dominates_method.sum())
        inc = n - dom - domd + int((ext.dominated_by_method & ext.dominates_method).sum())
        md.append(f"| {ds} / {at} | {n} | {dom} ({dom / n:.0%}) | {domd} ({domd / n:.0%}) | {inc} | {low} |")
    md.append("")
    if figures:
        paths = pareto_figures(pts, os.path.join(out_dir, "figures"))
        md.append(f"Figures (k = {PARETO_K}): " + (", ".join(os.path.relpath(p, out_dir) for p in paths) or "none")
                  + "\n")


def pareto_figures(pts, fig_dir, k=PARETO_K):
    """One spread-disparity figure per test graph at budget k (light, for print). Colours: the validated reference
    palette's first three categorical slots (all-pairs safe), blue = the method, orange = BSM's sweep, aqua = the
    other baselines, told apart by marker shape and named in the legend; summary_pareto.csv is the table view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, ink2, muted, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    shapes = ("^", "v", "D", "P", "X", "<", ">", "h", "p", "d", "8", "*")
    os.makedirs(fig_dir, exist_ok=True)
    out = []
    for (ds, at), g in pts[pts.k == k].groupby(["dataset", "attribute"]):
        m = g[g.source == "method"].sort_values("rho")
        if m.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.6, 3.7), dpi=200)
        fig.patch.set_facecolor(surf)
        ax.set_facecolor(surf)
        # the method's curve: its deployments joined in floor order (the trade-off the user sets with rho)
        ax.plot(m.es_ratio, m.disparity, color=blue, lw=1.5, marker="o", ms=6, mec=surf, mew=1.2,
                solid_joinstyle="round", solid_capstyle="round", label="FIM-GAT v2 (one point per floor)", zorder=4)
        for _, r in m.iterrows():                       # direct labels on the method's points only: its floors
            ax.annotate(f"ρ={r.rho:g}", (r.es_ratio, r.disparity), xytext=(5, 4), textcoords="offset points",
                        fontsize=6.5, color=ink2)
        s = g[g.source == "bsm_sweep"]
        if len(s):                                      # points, not a line: tau does not order the plane
            ax.scatter(s.es_ratio, s.disparity, marker="s", s=22, color=orange, edgecolors=surf, linewidths=0.8,
                       label="BSM, every tau (one point each)", zorder=2)
        b = g[g.source == "baseline"]
        for i, name in enumerate(sorted(b.method.unique())):
            bb = b[b.method == name]
            ax.scatter(bb.es_ratio, bb.disparity, marker=shapes[i % len(shapes)], s=34, color=aqua, edgecolors=surf,
                       linewidths=0.9, label=name, zorder=3)
        ax.grid(True, color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(axis)
        ax.tick_params(colors=muted, labelsize=7)
        ax.set_xlabel("spread / reference spread (Monte Carlo; 1 = the efficiency reference)", color=ink2,
                      fontsize=7.5)
        ax.set_ylabel("disparity, labelled groups (Monte Carlo)", color=ink2, fontsize=7.5)
        ax.set_title(f"{ds} / {at}, k = {k}", color=ink, fontsize=8.5, loc="left")
        ax.legend(fontsize=6, frameon=False, labelcolor=ink2, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0)                    # outside the plot: it never hides a point
        fig.tight_layout()
        path = os.path.join(fig_dir, f"pareto_{ds}_{at}_k{k}.png")
        fig.savefig(path, facecolor=surf)
        plt.close(fig)
        out.append(path)
    return out


# ------------------------------------------------------------------ T3

def _reach(trace, target):
    """First time (s) at which a search trace ([[secs, rr_disp], ...], JSON) is at or below target; inf if never."""
    for secs, disp in (json.loads(trace) if isinstance(trace, str) else trace):
        if float(disp) <= float(target) + 1e-12:
            return float(secs)
    return float("inf")


def t3(d3, dr, tuning, out_rows, md):
    md.append("## T3: does the learning add anything?\n")
    w3 = tuning.get("w3") or {}
    if not w3.get("train"):
        h = w3.get("H")
        share = w3.get("improve_share")
        md.append("Not run under the pre-registered rule (W3 gate: "
                  + str(w3.get("reason", "no W3 result in tuning.json"))
                  + (f"; H = {h:.2f}" if isinstance(h, (int, float)) and h == h else "")
                  + (f"; improvement share of query time {share:.0%}" if isinstance(share, (int, float)) else "")
                  + ").\n")
        if d3 is not None and not d3.empty:
            md.append("WARNING: t3.csv has rows although the W3 rule did not ask for a learned proposer; they are "
                      "not a T3 result.\n")
        return
    if d3 is None or d3.empty:
        md.append("The W3 rule asked for a learned proposer, but there are no T3 rows yet (train_proposer.py, then "
                  "run_t3.py).\n")
        return
    budgets = sorted(float(b) for b in d3.budget_s.unique())
    top = budgets[-1]
    fin = d3[np.isclose(d3.budget_s.astype(float), top)]
    sp = []
    for key, g in fin.groupby(CASE_COLS):
        # the simple search a learned run is paired with: the one from the same start (its train seed under a
        # learned start), else the single one from the greedy start (train_seed -1)
        simple = {float(r.train_seed): r for _, r in g[g.proposer == "simple"].iterrows()}
        for _, r in g[g.proposer == "learned"].iterrows():
            s = simple.get(float(r.train_seed), simple.get(-1.0))
            if s is None or len(json.loads(s.trace)) < 2:      # the simple hybrid made no move: nothing to reach
                continue
            target = float(s.rr_disp)
            t_s, t_l = _reach(s.trace, target), _reach(r.trace, target)
            sp.append({**dict(zip(CASE_COLS, key)), "train_seed": r.train_seed, "t_simple": t_s, "t_learned": t_l,
                       "speedup": t_s / t_l if t_l > 0 else float("inf")})
    sp = pd.DataFrame(sp)
    md.append(f"(i) Time to the simple hybrid's final RR disparity (searches of up to {top:g} s):\n")
    faster, n_graphs = 0, 0
    if not sp.empty:
        md.append("| test graph | cases | median speed-up (budgets x train seeds) | faster |")
        md.append("|---|---|---|---|")
        for (ds, at), g in sp.groupby(["dataset", "attribute"]):
            per_k = g.groupby("k").speedup.median()
            med = float(per_k.median())
            ok = med >= T3_SPEEDUP_MIN
            faster, n_graphs = faster + int(ok), n_graphs + 1
            md.append(f"| {ds} / {at} | {len(per_k)} | {med:.2f} | {'yes' if ok else 'no'} |")
        md.append("")
    pass_i = n_graphs > 0 and faster > n_graphs / 2
    md.append(f"Faster (median speed-up >= {T3_SPEEDUP_MIN:g}) on {faster} of {n_graphs} test graphs: "
              f"{'pass' if pass_i else 'not met'}.\n")
    b2 = min(budgets, key=lambda b: abs(b - float(T3_PRIMARY_BUDGET)))
    at_b = d3[np.isclose(d3.budget_s.astype(float), b2)]
    simple = at_b[at_b.proposer == "simple"].groupby(CASE_COLS).disparity.mean().rename("simple")
    learned = at_b[at_b.proposer == "learned"].groupby(CASE_COLS).disparity.mean().rename("learned")
    j = pd.concat([simple, learned], axis=1).dropna()
    n, med, p = (0, float("nan"), float("nan"))
    if not j.empty:
        per_graph = (j.learned - j.simple).groupby(level=["dataset", "attribute"]).mean()
        n, med, p = wilcoxon(per_graph.values)
    v = verdict(n, med, p)
    pass_ii = v == "win"
    md.append(f"(ii) Monte Carlo disparity at {b2:g} s, learned (seed mean) - simple, over {n} test graphs: median "
              f"{med:+.4f}, p {p:.3g} ({v}).\n")
    md.append(f"T3: {'pass' if pass_i or pass_ii else 'fail'} (pre-registered rule: (i) or (ii)). A fail drops the "
              "learned proposer; the claim is not reframed (design section 10).\n")
    out_rows.append({"test": "T3", "comparison": "learned - simple", "n_graphs": n, "median_diff": med, "p": p,
                     "verdict": "pass" if pass_i or pass_ii else "fail", "faster_graphs": faster,
                     "graphs_with_speedups": n_graphs})
    lrn = d3[d3.proposer == "learned"]
    if lrn.train_seed.nunique() > 1:
        sd = lrn.groupby(CASE_COLS + ["budget_s"]).disparity.std(ddof=1).groupby("budget_s").mean()
        md.append("Train-seed standard deviation of the learned proposer's Monte Carlo disparity (mean over cases): "
                  + ", ".join(f"{b:g} s: {s:.4f}" for b, s in sd.items()) + "\n")
    if dr is not None and not dr.empty:
        cols = [f"recall_{c}" for c in T3_RECALL_C if f"recall_{c}" in dr]
        rec = dr.groupby("proposer")[cols].mean()
        md.append("recall@C on states along the simple hybrid's path (share of improvable states whose top-C "
                  "proposals hold an improving swap):\n")
        md.append("| proposer | " + " | ".join(c.replace("recall_", "C = ") for c in cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        for name, r in rec.iterrows():
            md.append(f"| {name} | " + " | ".join(f"{x:.2f}" for x in r.values) + " |")
        md.append("")


# ------------------------------------------------------------------ T4

def t4(d4, md, out_dir):
    md.append("## T4 (C3): runtime, cached and per query\n")
    if d4 is None or d4.empty:
        md.append("No T4 rows yet.\n")
        return
    keys = ["dataset", "attribute", "method", "variant"]
    if "deployment" in d4:                      # FIM-GAT: its deployed set only; the other rows have no deploy step
        d4 = d4[(d4.method != FIM) | (d4.deployment == "certified")]
    agg = d4.groupby(keys).agg(prep_s=("prep_s", "max"), query_s=("query_s", "mean"),
                               disparity=("disparity", "mean"),
                               cert_pass=("cert_passed", lambda s: float(truthy(s).mean()))).reset_index()
    for q in QS:
        agg[f"time_Q{q}"] = agg.prep_s + q * agg.query_s
    agg.to_csv(os.path.join(out_dir, "summary_t4.csv"), index=False)
    md.append("| dataset | method | variant | cached s | per query s | " + " | ".join(f"Q={q} s" for q in QS)
              + " | disparity | certified |")
    md.append("|---|---|---|---|---|" + "---|" * len(QS) + "---|---|")
    for _, r in agg.iterrows():
        md.append(f"| {r.dataset}/{r.attribute} | {r.method} | {r.variant} | {r.prep_s:.1f} | {r.query_s:.1f} | "
                  + " | ".join(f"{r[f'time_Q{q}']:.0f}" for q in QS) + f" | {r.disparity:.4f} | {r.cert_pass:.0%} |")
    md.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=RESULTS_V2)
    ap.add_argument("--out", default=None, help="default <dir>/summary_v2.md")
    ap.add_argument("--figures", action="store_true",
                    help=f"also draw the spread-disparity figures (k = {PARETO_K}) into <dir>/figures/")
    args = ap.parse_args()
    tuning = {}
    tp = os.path.join(args.dir, TUNING_FILE)
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as f:
            tuning = json.load(f)
    start = tuning.get("start", "greedy")
    md = ["# FIM-GAT v2 results (pre-registered analysis)\n",
          f"Tuning (validation only): delta={tuning.get('delta')}, phi0={tuning.get('phi0')}, start={start}, "
          f"W3 train={tuning.get('w3', {}).get('train')} (code_tag {tuning.get('code_tag')}).\n"]
    rows = []
    d1, d2, d3, d4 = (load(os.path.join(args.dir, f)) for f in ("t1.csv", "t2.csv", "t3.csv", "t4.csv"))
    bounds = load(os.path.join(args.dir, "t1_bounds.csv"))
    dsens = load(os.path.join(args.dir, "t1_certsens.csv"))
    dp = load(os.path.join(args.dir, "t1_pareto.csv"))
    dr = load(os.path.join(args.dir, "t3_recall.csv"))
    if d1 is not None and "tuned" in d1 and (~truthy(d1.tuned)).any():
        md.append(f"WARNING: {int((~truthy(d1.tuned)).sum())} T1 rows were made untuned (development runs).\n")
    for label, d in (("T1", d1), ("T2", d2), ("T3", d3), ("T4", d4), ("Pareto", dp)):
        n_dup = duplicates(d)
        if n_dup:
            md.append(f"WARNING: {n_dup} {label} rows repeat a query already recorded (two jobs ran the same case); "
                      f"means over budgets and seeds count them twice. Remove one copy before reporting.\n")
    if d1 is not None:
        t1(d1, start, rows, md, args.dir)
        t1_matrix(d1, start, bounds, md, args.dir)
        pareto(d1, dp, start, md, args.dir, figures=args.figures)
    t2(d2, d1, start, rows, md)
    t3(d3, dr, tuning, rows, md)
    t4(d4, md, args.dir)
    cert_sensitivity(dsens, start, md)
    pd.DataFrame(rows).to_csv(os.path.join(args.dir, "summary_tests.csv"), index=False)
    out = args.out or os.path.join(args.dir, "summary_v2.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
