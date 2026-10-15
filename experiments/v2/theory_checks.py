"""Numerical checks of docs/theory_v2.md on a synthetic graph with a high-precision ground truth.

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore theory_checks.py            # about 5 min on the 8 GB laptop (273 s)
    PYTHONIOENCODING=utf-8 python -u -W ignore theory_checks.py --quick    # smoke run, about 25 s; writes results/v2/theory_quick/

Graph: the 2,000-node DC-SBM of the test suite (weighted cascade, three groups). It is synthetic on purpose:
these checks concern the statistics of the RR verifier, not any v2 dataset, so no validation or test graph
is touched (protocol section 8).

Ground truth: 8,192 RR sets per node, drawn in chunks of 512 from seeds used nowhere else, accumulated per
seed set (group-rate sd about 2e-4). Every quantity below is compared with it.

1 search      improve(proposer="exhaustive", audit=False) from the floor-meeting greedy-family start, on
              search samples A with R in {8, 64, 512} per node, with the plain rule (z=0) and noise-aware
              rules (z=1, 2). Records every visited set's estimated (A) and true disparity: the optimism
              gap (true - estimated at the end), the true gain, and how many accepted moves were not true
              improvements.
2 certify     certify.certified_choice on independent samples B (R in {8, 64, 512}) for each final set:
              coverage (no eligible set misses the true floor; no disparity bound below the truth), power
              (how often the final set is certified), and the half-width next to the Monte Carlo
              equal-spread test's (2,000 simulations, delta method).
3 communities sd of a community's rate across independent samples against the prediction
              sqrt(r(1-r) / (R m)), for BFS-ball communities of m in {10, 30, 100, 300, 1000} nodes.

Outputs results/v2/theory/{search,certify,community_noise}.csv and summary.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fimgat.baselines import RRGreedy                     # noqa: E402
from fimgat.data import make_dcsbm                        # noqa: E402
from fimgat.diffusion import ExposureEstimator            # noqa: E402
from fimgat.graph import from_edges                       # noqa: E402
from fimgat.v2 import certify as C                        # noqa: E402
from fimgat.v2.config import RESULTS_V2, rr_floor         # noqa: E402
from fimgat.v2.improve import improve                     # noqa: E402
from fimgat.v2.reference import equal_spread_test         # noqa: E402
from fimgat.v2.train import exact_rates                   # noqa: E402

RHO = 0.99
OUT = os.path.join(RESULTS_V2, "theory")


def study_graph():
    """The test suite's sbm_graph (tests/conftest.py), rebuilt here."""
    g0 = make_dcsbm(n=2000, homophily=0.8, seed=11)
    wc = from_edges("sbm_wc", g0.n, np.stack([g0.src, g0.dst], 1), g0.groups, g0.group_names,
                    undirected=False, weighted_cascade=True)
    rng = np.random.default_rng(0)
    three = np.where(wc.groups == 1, 2, rng.integers(0, 2, wc.n))
    return wc.with_groups(three, ["x", "y", "z"])


def disp(rates, present):
    r = np.asarray(rates)[present]
    return float(r.max() - r.min())


def path_of(start, hist):
    s, out = set(int(v) for v in start), [tuple(sorted(int(v) for v in start))]
    for h in hist[1:]:
        if h["accepted"]:
            rem, add = h["move"]
            s = (s - set(rem)) | set(add)
            out.append(tuple(sorted(s)))
    return out


def truth(g, sets, R_total, chunk, seed0):
    """Group rates (n_sets, G) and totals from R_total RR sets per node, streamed in chunks."""
    n_chunks = max(1, R_total // chunk)
    acc = np.zeros((len(sets), g.n_groups))
    for c in range(n_chunks):
        est = ExposureEstimator(g, R=chunk, seed=seed0 + c, device="cpu")
        for i, s in enumerate(sets):
            acc[i] += exact_rates(est, list(s))[0]
        del est
        print(f"    truth chunk {c + 1}/{n_chunks}", flush=True)
    rates = acc / n_chunks
    return rates, rates @ g.group_sizes.astype(np.float64)


def bfs_ball(g, center, m):
    from collections import deque
    nbr = [[] for _ in range(g.n)]
    for u, v in zip(g.src.tolist(), g.dst.tolist()):
        nbr[u].append(v)
        nbr[v].append(u)
    seen, q, out = {center}, deque([center]), []
    while q and len(out) < m:
        u = q.popleft()
        out.append(u)
        for w in nbr[u]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    return np.array(out[:m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--RA", type=int, nargs="+", default=[8, 64, 512])
    ap.add_argument("--RB", type=int, nargs="+", default=[8, 64, 512])
    ap.add_argument("--zs", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--truth-R", type=int, default=8192)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.RA, args.RB, args.truth_R = 2, [8, 64], [64], 1024
    out_dir = OUT + "_quick" if args.quick else OUT      # a smoke run never overwrites the study's outputs
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    g = study_graph()
    present = g.group_sizes > 0
    k = args.k
    print(f"graph n={g.n} m={g.m} groups={g.group_sizes.tolist()} k={k}", flush=True)

    # ---------------------------------------------------------------- 1 search on samples A
    runs, sets = [], {}

    def sid(s):
        return sets.setdefault(tuple(sorted(int(v) for v in s)), len(sets))

    for rep in range(args.reps):
        for RA in args.RA:
            est = ExposureEstimator(g, R=RA, seed=1000 + 37 * rep + RA, device="cpu")
            gr = RRGreedy(est)
            ref = gr.greedy(k)
            ref_tot = exact_rates(est, ref)[1]
            floor = rr_floor(RHO, ref_tot)
            start = ref
            for cand in (gr.parity(k), gr.welfare(k), gr.maximin(k)):
                if exact_rates(est, cand)[1] >= floor and disp(exact_rates(est, cand)[0], present) < \
                        disp(exact_rates(est, start)[0], present):
                    start = cand
            for z in args.zs:
                ts = time.perf_counter()
                seeds, hist = improve(est, start, floor, proposer="exhaustive", audit=False, max_rounds=500, z=z)
                secs = time.perf_counter() - ts
                path = path_of(start, hist)
                runs.append({"rep": rep, "RA": RA, "z": z, "secs": secs, "ref": sid(ref), "start": sid(start),
                             "final": sid(seeds), "path": [sid(s) for s in path],
                             "est_disp": [disp(exact_rates(est, list(s))[0], present) for s in path],
                             "est_slack": (exact_rates(est, seeds)[1] - RHO * ref_tot) / ref_tot,
                             "moves": len(path) - 1})
                print(f"  rep {rep} RA={RA} z={z}: moves {len(path) - 1}, est disp "
                      f"{runs[-1]['est_disp'][0]:.4f} -> {runs[-1]['est_disp'][-1]:.4f} ({secs:.1f}s)", flush=True)

    print(f"search done ({time.time() - t0:.0f}s); {len(sets)} distinct sets; computing truth", flush=True)
    order = sorted(sets, key=sets.get)
    t_rates, t_tot = truth(g, order, args.truth_R, 512, 777000)
    t_disp = np.array([disp(r, present) for r in t_rates])

    rows = []
    for r in runs:
        td = [float(t_disp[i]) for i in r["path"]]
        steps = list(zip(td, td[1:]))
        rows.append({"rep": r["rep"], "RA": r["RA"], "z": r["z"], "moves": r["moves"], "secs": r["secs"],
                     "est_start": r["est_disp"][0], "est_final": r["est_disp"][-1],
                     "true_start": td[0], "true_final": td[-1],
                     "est_gain": r["est_disp"][0] - r["est_disp"][-1], "true_gain": td[0] - td[-1],
                     "optimism": td[-1] - r["est_disp"][-1],
                     "nonimproving_moves": int(sum(b >= a for a, b in steps)),
                     "worsening_moves": int(sum(b > a + 1e-4 for a, b in steps)),
                     "est_slack": r["est_slack"],
                     "true_slack": float((t_tot[r["final"]] - RHO * t_tot[r["ref"]]) / t_tot[r["ref"]])})
    search = pd.DataFrame(rows)
    search.to_csv(os.path.join(out_dir, "search.csv"), index=False)

    # ---------------------------------------------------------------- 2 certificates on samples B
    crow = []
    for rep in range(args.reps):
        for RB in args.RB:
            est_b = ExposureEstimator(g, R=RB, seed=5000 + 41 * rep + RB, device="cpu")
            for r in [x for x in runs if x["rep"] == rep]:
                final, ref = list(order[r["final"]]), list(order[r["ref"]])
                out = C.certified_choice(est_b, [final], ref, RHO, alpha=0.05)
                cf = out["certificates"][0]
                T_true = t_tot[r["final"]] - RHO * t_tot[r["ref"]]
                crow.append({"rep": rep, "RA": r["RA"], "z": r["z"], "RB": RB,
                             "eligible": cf["eligible"], "identical_to_ref": cf["is_reference"],
                             "floor_lcb_rel": cf["floor_lcb"] / t_tot[r["ref"]],
                             "floor_est_rel": cf["floor_estimate"] / t_tot[r["ref"]],
                             "floor_halfwidth_rel": cf["floor_halfwidth"] / t_tot[r["ref"]],
                             "true_slack": T_true / t_tot[r["ref"]],
                             "floor_miss": bool(cf["eligible"] and T_true < 0),
                             "disp_ucb": cf["disp_ucb"], "disp_true": float(t_disp[r["final"]]),
                             "disp_miss": bool(cf["disp_ucb"] < t_disp[r["final"]]),
                             "deployed_is_ref": out["is_reference"]})
            del est_b
    for r in runs:                                   # Monte Carlo equal-spread test, for comparison
        final, ref = list(order[r["final"]]), list(order[r["ref"]])
        es = equal_spread_test(g, final, ref, rho=RHO, n_sims=2000, seed=4242 + r["rep"], alpha=0.05)
        for c in crow:
            if c["rep"] == r["rep"] and c["RA"] == r["RA"] and c["z"] == r["z"]:
                c.update(mc_passed=es["passed"], mc_halfwidth_rel=es["ratio"] - es["lower"], mc_ratio=es["ratio"])
    cert = pd.DataFrame(crow)
    cert.to_csv(os.path.join(out_dir, "certify.csv"), index=False)

    # ---------------------------------------------------------------- 3 community-rate noise
    rng = np.random.default_rng(3)
    S = RRGreedy(ExposureEstimator(g, R=64, seed=31337, device="cpu")).greedy(k)
    comms = {m: bfs_ball(g, int(rng.integers(g.n)), m) for m in (10, 30, 100, 300, 1000)}
    nrow = []
    for R in (8, 64, 512):
        vals = {m: [] for m in comms}
        for rep in range(20 if not args.quick else 5):
            est = ExposureEstimator(g, R=R, seed=60000 + 97 * rep + R, device="cpu")
            cov = C.set_coverage(est, S).reshape(g.n, R).mean(axis=1)      # all nodes are roots here
            for m, nodes in comms.items():
                vals[m].append(float(cov[nodes].mean()))
            del est
        for m, v in vals.items():
            r_bar = float(np.mean(v))
            nrow.append({"R": R, "m": m, "rate": r_bar, "sd_observed": float(np.std(v, ddof=1)),
                         "sd_predicted": float(np.sqrt(max(r_bar * (1 - r_bar), 1e-12) / (R * m)))})
    noise = pd.DataFrame(nrow)
    noise.to_csv(os.path.join(out_dir, "community_noise.csv"), index=False)

    # ---------------------------------------------------------------- summary
    agg = search.groupby(["RA", "z"]).agg(moves=("moves", "mean"), est_gain=("est_gain", "mean"),
                                          true_gain=("true_gain", "mean"), optimism=("optimism", "mean"),
                                          nonimproving=("nonimproving_moves", "sum"), total_moves=("moves", "sum"),
                                          worsening=("worsening_moves", "sum"), true_final=("true_final", "mean"),
                                          true_slack_min=("true_slack", "min")).reset_index()
    cagg = cert.groupby(["RB"]).agg(n=("eligible", "size"), floor_miss=("floor_miss", "sum"),
                                    disp_miss=("disp_miss", "sum"), certified=("eligible", "mean"),
                                    halfwidth_rel=("floor_halfwidth_rel", "median"),
                                    mc_halfwidth_rel=("mc_halfwidth_rel", "median"),
                                    mc_passed=("mc_passed", "mean")).reset_index()
    summary = {"graph": {"n": int(g.n), "m": int(g.m), "groups": g.group_sizes.tolist(), "k": k, "rho": RHO,
                         "truth_R": args.truth_R},
               "search": agg.to_dict(orient="records"), "certify": cagg.to_dict(orient="records"),
               "community_noise": noise.to_dict(orient="records"), "secs": time.time() - t0}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    pd.set_option("display.width", 200)
    print("\nSEARCH (means over reps)\n", agg.round(5).to_string(index=False))
    print("\nCERTIFY\n", cagg.round(5).to_string(index=False))
    print("\nCOMMUNITY NOISE\n", noise.round(5).to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
