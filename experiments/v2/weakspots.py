"""Numerical checks of the weak-spot theory (docs/theory_v2.md, W1-W4) on synthetic graphs only.

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore weakspots.py --part all            # about 20 min (8 GB laptop)
    PYTHONIOENCODING=utf-8 python -u -W ignore weakspots.py --part w3 --quick     # smoke run

No v2 validation or test graph is touched (protocol section 8). Ground truth is a separate RR sample of
R_truth sets per node (node-level exposure), drawn from seeds used nowhere else.

w1  Hidden communities (claim C2). Planted-block graphs (30 blocks; cohesive or loose), whose blocks are hidden
    communities the methods never see; three labelled groups correlated with the blocks; rho in {0.99, 0.97}.
    From the floor-meeting greedy-family start, on a search sample: (a) labelled disparity only, (b) hive mind
    at one scale, (b') hive mind with multi-scale audits, (e) the epsilon-constraint phase from (a)'s optimum
    (disp_cap = D_a + EPS_CAP, community-price proposals), (c) a mixture of campaigns against the labelled
    groups and (b')'s family. Reports true labelled and hidden-community disparity, true floor slack, and
    the terms of the transfer bound (with the Jaccard of the matches to the hidden extremes).
w2  Lower bounds. Graphs with many tiny hard-to-reach groups (isolated pairs; pendant pairs) and the
    three-group graph: the bound of Proposition W2.3 against the disparity local search reaches.
w3  Proposer cost model. DC-SBM graphs of 2k, 8k and 32k nodes (3 each), k in {25, 50, 100}, rho in
    {0.99, 0.97}, descending from the greedy reference: along exhaustive paths, the hit rate and progress
    ratio per grid of the simple proposer with the published and with setpoint-scaled prices, the costs,
    the model's speed-up and headroom; then the measured time and final disparity of exhaustive search, the
    hybrid and the simple proposer alone (each with both price settings), and the RR-sampling and greedy
    times (Amdahl).
w4  Price loop. The simple proposer from the greedy reference, with and without the exhaustive fallback,
    with the published gains at setpoints 0 (the spread price is inert), 0.25%, 0.5%, 1% and with gains
    scaled to the setpoint: proposals rejected by the floor, the spread price, moves, final disparity.
Outputs results/v2/weakspots/{w1,w2,w3_states,w3_runs,w4}.csv and summary.json (weakspots_quick/ with --quick).
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
from fimgat.v2 import auditor as AUD                      # noqa: E402
from fimgat.v2 import bounds as BND                       # noqa: E402
from fimgat.v2 import certify as C                        # noqa: E402
from fimgat.v2 import diagnostics as DIA                  # noqa: E402
from fimgat.v2 import improve as IMPM                     # noqa: E402
from fimgat.v2 import prices as PRC                       # noqa: E402
from fimgat.v2.config import RESULTS_V2, rr_floor         # noqa: E402
from fimgat.v2.cover import CoverState                    # noqa: E402
from fimgat.v2.hidden import community_rates, transfer_bound   # noqa: E402
from fimgat.v2.improve import FLOOR_RTOL, improve         # noqa: E402
from fimgat.v2.local_search import best_swap              # noqa: E402
from fimgat.v2.mixture import mixture_campaigns           # noqa: E402

RHO = 0.99


# ------------------------------------------------------------------ shared helpers

def wc_graph(name, n, edges, groups, names):
    return from_edges(name, n, edges, groups, names, undirected=True, weighted_cascade=True)


def three_groups(seed, n, avg_degree=8):
    """The test suite's DC-SBM recipe at size n: weighted cascade, three groups (minority -> group 2)."""
    g0 = make_dcsbm(n=n, homophily=0.8, avg_degree=avg_degree, seed=seed)
    wc = from_edges(f"sbm{n}_{seed}", g0.n, np.stack([g0.src, g0.dst], 1), g0.groups, g0.group_names,
                    undirected=False, weighted_cascade=True)
    rng = np.random.default_rng(seed)
    return wc.with_groups(np.where(wc.groups == 1, 2, rng.integers(0, 2, wc.n)), ["x", "y", "z"])


def truth_exposure(g, sets, R_truth, chunk, seed0):
    """Node-level activation probabilities (len(sets), n) from R_truth RR sets per node, in chunks."""
    acc = np.zeros((len(sets), g.n))
    done, i = 0, 0
    while done < R_truth:
        r = min(chunk, R_truth - done)
        est = ExposureEstimator(g, R=r, seed=seed0 + i, device="cpu")
        for j, s in enumerate(sets):
            acc[j] += C.set_coverage(est, list(s)).reshape(g.n, r).sum(axis=1)
        done, i = done + r, i + 1
        del est
    return acc / R_truth


def group_rates(g, a):
    return np.bincount(g.groups, weights=a, minlength=g.n_groups) / np.maximum(g.group_sizes, 1)


def disp(r):
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    return float(r.max() - r.min()) if len(r) >= 2 else 0.0


def sample_disp(est, seeds):
    st = CoverState(est, seeds)
    return disp(np.asarray(st.rates)[st.present]), st.total


def floor_start(est, k, rho=RHO):
    """(reference, floor, start): RR-greedy reference, RR floor, the lowest-disparity floor-meeting set of the
    greedy family (as experiments/v2/theory_checks.py)."""
    gr = RRGreedy(est)
    ref = gr.greedy(k)
    ref_tot = CoverState(est, ref).total
    floor = rr_floor(rho, ref_tot)
    start, best = ref, sample_disp(est, ref)[0]
    for cand in (gr.parity(k), gr.welfare(k), gr.maximin(k)):
        d, t = sample_disp(est, cand)
        if t >= floor and d < best:
            start, best = cand, d
    return ref, floor, start


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ W1

def planted_graph(seed, n_blocks=30, smin=20, smax=300, deg_in=6, cross=1.0, n_groups=3, dom=0.6):
    """Blocks of power-law sizes in [smin, smax]; deg_in / 2 within-block partners per node and one
    cross-block partner with probability cross / 2 (undirected: mean cross degree about cross, block
    conductance about cross / (deg_in + cross)); labelled groups follow a block's dominant group with
    probability dom."""
    rng = np.random.default_rng(seed)
    sizes = np.clip((smin * (1.0 - rng.random(n_blocks)) ** (-1.0 / 1.2)).astype(np.int64), smin, smax)
    n = int(sizes.sum())
    block = np.repeat(np.arange(n_blocks), sizes)
    members = [np.flatnonzero(block == b) for b in range(n_blocks)]
    edges = []
    for mem in members:
        src = np.repeat(mem, deg_in // 2)
        edges.append(np.stack([src, rng.choice(mem, len(src))], 1))
    src = np.flatnonzero(rng.random(n) < cross / 2.0)
    dst = rng.integers(0, n, len(src))
    keep = block[src] != block[dst]
    edges.append(np.stack([src[keep], dst[keep]], 1))
    dominant = rng.integers(0, n_groups, n_blocks)
    groups = np.where(rng.random(n) < dom, dominant[block], rng.integers(0, n_groups, n))
    g = wc_graph(f"planted_{seed}", n, np.concatenate(edges), groups, [f"g{i}" for i in range(n_groups)])
    return g, members


COHESION = {"loose": dict(deg_in=6, cross=2.0), "cohesive": dict(deg_in=8, cross=1.0)}


EPS_CAP = 0.002      # epsilon-constraint phase: labelled D may rise this far above the labelled-only optimum


def _jaccard_best(h, family):
    h = np.asarray(h)
    return max((len(np.intersect1d(h, c)) / len(np.union1d(h, c)) for c in family), default=0.0)


def _w1_row(base, name, g, blocks, a, sig_ref, d_sample, family, rho=RHO, hist=None):
    """Truth metrics of one exposure vector, with the transfer-bound terms for `family`."""
    tb = transfer_bound(a, blocks, family) if family else {}
    row = {**base, "method": name, "D_lab": disp(group_rates(g, a)), "D_hidden": disp(community_rates(a, blocks)),
           "slack": float((a.sum() - rho * sig_ref) / sig_ref), "D_lab_sample": d_sample,
           "family": len(family) if family else 0}
    if hist is not None:
        row.update(moves=sum(h["accepted"] for h in hist), comm_disp_first=hist[0]["comm_disp"],
                   comm_disp_last=hist[-1]["comm_disp"])
    if tb:
        row.update(tb_DF=tb["D_F"], tb_gap_max=tb["gap_max"], tb_gap_min=tb["gap_min"], tb_bound=tb["bound"],
                   tb_holds=tb["holds"], jac_hmax=_jaccard_best(blocks[tb["h_max"]], family),
                   jac_hmin=_jaccard_best(blocks[tb["h_min"]], family))
    return row


def run_w1(args, out):
    from fimgat.v2.proposer import community_moves
    rows = []
    cases = [(c, s, k, rho) for c in args.w1_cohesion for s in args.w1_graphs for k in args.w1_ks
             for rho in args.w1_rhos]
    graphs = {}
    for (cohesion, gs, k, rho) in cases:
        if (cohesion, gs) not in graphs:
            g0, b0 = planted_graph(gs, **COHESION[cohesion])
            graphs[(cohesion, gs)] = (g0, b0, float(np.median([AUD.conductance(g0, b) for b in b0])))
        g, blocks, phi = graphs[(cohesion, gs)]
        if True:
            t0 = time.time()
            est = ExposureEstimator(g, R=args.w1_R, seed=7000 + 13 * gs + k, device="cpu")
            ref, floor, start = floor_start(est, k, rho)
            kw = dict(max_rounds=args.w1_rounds, audit_every=5, return_family=True)
            s_a, h_a = improve(est, start, floor, proposer="exhaustive", audit=False, **kw)
            s_b, h_b = improve(est, start, floor, proposer="exhaustive", audit=True, **kw)
            s_m, h_m = improve(est, start, floor, proposer="exhaustive", audit=True, audit_band_ratio=2.0, **kw)
            # epsilon-constraint phase from the labelled-only optimum: lower D_C while D <= D_a + EPS_CAP
            cap = sample_disp(est, s_a)[0] + EPS_CAP
            s_e, h_e = improve(est, s_a, floor, proposer=lambda st, fam, rng: community_moves(st, fam), audit=True,
                               audit_band_ratio=2.0, disp_cap=cap, fallback="exhaustive", **kw)
            fams = {"hive": h_b[-1]["communities"], "hive_multi": h_m[-1]["communities"],
                    "hive_eps": h_e[-1]["communities"]}
            fam_m = fams["hive_multi"]
            mix = mixture_campaigns(est, s_m, floor, communities=fam_m, T=args.w1_T)
            named = [("reference", ref), ("start", start), ("labelled", s_a), ("hive", s_b), ("hive_multi", s_m),
                     ("hive_eps", s_e)]
            sets = [s for _, s in named] + mix["sets"]
            A = truth_exposure(g, sets, args.truth_R, args.truth_chunk, 900000 + 101 * gs + k)
            sig_ref = A[0].sum()
            base = {"cohesion": cohesion, "block_conductance": phi, "graph": gs, "n": g.n, "k": k, "rho": rho,
                    "n_blocks": len(blocks)}
            hists = {"labelled": h_a, "hive": h_b, "hive_multi": h_m, "hive_eps": h_e}
            for i, (name, s) in enumerate(named):
                rows.append(_w1_row(base, name, g, blocks, A[i], sig_ref, sample_disp(est, s)[0],
                                    fams.get(name, fam_m), rho, hists.get(name)))
            lam = np.asarray(mix["weights"])
            Am = A[len(named):]
            a_mix = lam @ Am
            per_h = np.array([disp(community_rates(a, blocks)) for a in Am])
            per_l = np.array([disp(group_rates(g, a)) for a in Am])
            slack_mix = np.array([(a.sum() - rho * sig_ref) / sig_ref for a in Am])
            row = _w1_row(base, "mixture", g, blocks, a_mix, sig_ref, float(mix["mix_disp"]), fam_m, rho)
            sup = lam > 1e-6                                 # campaigns the lottery actually deploys
            row.update(slack=float(slack_mix[sup].min()), mix_slack_all_sets=float(slack_mix.min()),
                       mix_support=int(sup.sum()), mix_sets=len(lam),
                       mix_D_hidden_expost_mean=float(lam @ per_h), mix_D_hidden_expost_max=float(per_h[lam > 1e-6].max()),
                       mix_D_lab_expost_mean=float(lam @ per_l), mix_audited_disp=float(mix["mix_disp"]),
                       mix_best_single_audited=float(mix["best_single_disp"]),
                       mix_uniform_audited=float(mix["uniform_disp"]), mix_regret_bound=float(mix["regret_bound"]))
            rows.append(row)
            log(f"  w1 {cohesion} graph {gs} (n={g.n}, block conductance {phi:.2f}) k={k} rho={rho}: families "
                f"{len(fams['hive'])}/{len(fam_m)}/{len(fams['hive_eps'])}, mixture support "
                f"{int((lam > 1e-6).sum())}/{len(lam)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "w1.csv"), index=False)
    return df


# ------------------------------------------------------------------ W2

def tiny_group_graph(seed, kind, n_core=2000, n_tiny=60, tiny_size=2):
    """Three-group DC-SBM core plus n_tiny groups of tiny_size extra nodes: isolated ("isolated") or a
    connected pair hanging off one random core node ("pendant")."""
    core = three_groups(seed, n_core)
    rng = np.random.default_rng(seed + 1)
    edges = [np.stack([core.src, core.dst], 1)]
    n = n_core + n_tiny * tiny_size
    groups = np.concatenate([core.groups, 3 + np.repeat(np.arange(n_tiny), tiny_size)])
    if kind == "pendant":
        extra = []
        for t in range(n_tiny):
            a = n_core + t * tiny_size
            for j in range(1, tiny_size):
                extra += [(a + j - 1, a + j), (a + j, a + j - 1)]
            u = int(rng.integers(n_core))
            extra += [(u, a), (a, u)]
        edges.append(np.array(extra))
    names = ["x", "y", "z"] + [f"t{t}" for t in range(n_tiny)]
    return from_edges(f"tiny_{kind}_{seed}", n, np.concatenate(edges), groups, names, undirected=False,
                      weighted_cascade=True)


def run_w2(args, out):
    rows = []
    specs = [("three_groups", lambda s: three_groups(s, 2000))] + \
            [(kind, (lambda s, kind=kind: tiny_group_graph(s, kind))) for kind in ("isolated", "pendant")]
    for kind, make in specs:
        for gs in args.w2_graphs:
            g = make(gs)
            for k in args.w2_ks:
                t0 = time.time()
                est = ExposureEstimator(g, R=args.w2_R, seed=3000 + 17 * gs + k, device="cpu")
                ref, floor, start = floor_start(est, k)
                seeds, _ = improve(est, start, floor, proposer="exhaustive", audit=False, max_rounds=args.w2_rounds)
                A = truth_exposure(g, [ref, seeds], args.truth_R, args.truth_chunk, 800000 + 31 * gs + k)
                sig_ref = A[0].sum()
                est_b = ExposureEstimator(g, R=args.w2_R_bound, seed=4000 + 19 * gs + k, device="cpu")
                lb = BND.disparity_lower_bound(est_b, k, RHO * sig_ref, delta=0.05)
                D_true = disp(group_rates(g, A[1]))
                rows.append({"kind": kind, "graph": gs, "n": g.n, "groups": g.n_groups, "k": k,
                             "lb": lb["lb"], "mean_floor": lb["mean_floor"], "maximin_ub": lb["maximin_ub"],
                             "u_k1": lb["u_k1"], "min_U": lb["min_U"], "D_local_search_true": D_true,
                             "D_local_search_sample": sample_disp(est, seeds)[0],
                             "D_reference_true": disp(group_rates(g, A[0])),
                             "slack_true": float((A[1].sum() - RHO * sig_ref) / sig_ref),
                             "gap": D_true - lb["lb"]})
                log(f"  w2 {kind} graph {gs} k={k}: lb {lb['lb']:.4f}, local search {D_true:.4f} "
                    f"({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "w2.csv"), index=False)
    return df


# ------------------------------------------------------------------ W3

GRIDS = ((4, 2), (8, 4), (16, 8), (32, 16), (64, 32))
SETPOINT = 0.005          # W4's setpoint-scaled prices: setpoint 0.5%, kp = 1 / setpoint, ki = 0.5 / setpoint
SCALED = dict(price_setpoint=SETPOINT, price_gains=(1.0 / SETPOINT, 0.5 / SETPOINT))


def scaled_scores(state, floor):
    """The simple proposer's scores with setpoint-scaled prices (a fresh controller, as simple_scores)."""
    from fimgat.v2.objective import price_weights
    ctrl = PRC.PIController(kp=SCALED["price_gains"][0], ki=SCALED["price_gains"][1], sizes=state.sizes,
                            setpoint=SETPOINT)
    alpha, beta, mu = ctrl.update(state.rates, state.present, state.total, floor)
    w = price_weights(alpha, beta, mu, state.sizes, state.present)
    add = np.asarray(state.gains(), dtype=np.float64) @ w
    drop = np.zeros(state.index.n)
    drop[np.asarray(state.seeds, dtype=np.int64)] = -(np.asarray(state.removal_loss(), dtype=np.float64) @ w)
    return add, drop


def run_w3(args, out):
    srows, rrows = [], []
    for n in args.w3_sizes:
        for gs in args.w3_graphs:
            g = three_groups(500 + n + 7 * gs, n)
            for (k, rho) in [(k, rho) for k in args.w3_ks for rho in args.w3_rhos]:
                t0 = time.time()
                ts = time.perf_counter()
                est = ExposureEstimator(g, R=args.w3_R, seed=6000 + n + k + 11 * gs, device="cpu")
                t_rr = time.perf_counter() - ts
                ts = time.perf_counter()
                RRGreedy(est).greedy(k)
                t_greedy = time.perf_counter() - ts
                ref, floor, _ = floor_start(est, k, rho)
                start = ref                                  # descend from the greedy reference
                floor_acc = floor * (1 + FLOOR_RTOL)
                # measured runs from the same start (timed, nothing else running)
                for name, kw in (("exhaustive", dict(proposer="exhaustive")),
                                 ("hybrid", dict(proposer="simple", fallback="exhaustive")),
                                 ("simple", dict(proposer="simple")),
                                 ("hybrid_scaled", dict(proposer="simple", fallback="exhaustive", **SCALED)),
                                 ("simple_scaled", dict(proposer="simple", **SCALED))):
                    ts = time.perf_counter()
                    seeds, hist = improve(est, start, floor, audit=False, max_rounds=args.w3_rounds, **kw)
                    secs = time.perf_counter() - ts
                    d, _ = sample_disp(est, seeds)
                    local_opt = best_swap(CoverState(est, seeds), floor_acc) is None
                    rrows.append({"n": n, "graph": gs, "k": k, "rho": rho, "method": name, "secs": secs,
                                  "t_rr_sample": t_rr, "t_greedy": t_greedy,
                                  "moves": sum(h["accepted"] for h in hist),
                                  "fallback_moves": sum(h.get("fallback_move", False) for h in hist),
                                  "D_start": hist[0]["disp"], "D_final": d, "local_opt": local_opt})
                    if name == "exhaustive":
                        path, st = [list(start)], CoverState(est, start)
                        for h in hist[1:]:
                            if h["move"]:
                                (u,), (v,) = h["move"]
                                st.apply_swap(u, v)
                                path.append(list(st.seeds))
                for step, s in enumerate(path):
                    for scorer, fn in (("simple", None), ("scaled", scaled_scores)):
                        m = DIA.measure_proposer(est, s, floor_acc, grids=GRIDS, scores=fn, n_time=args.w3_ntime)
                        row = {"n": n, "graph": gs, "k": k, "rho": rho, "scorer": scorer, "step": step,
                               "improvable": m["improvable"], "best_all": m["best_all"], "T_ex": m["T_ex"],
                               "c_scores": m["c_scores"], "c_s": m["c_s"]}
                        for (a, b) in GRIDS:
                            row[f"hit_{a}x{b}"] = m["pi"][(a, b)]
                            row[f"q_{a}x{b}"] = m["q"][(a, b)]
                        srows.append(row)
                log(f"  w3 n={n} graph {gs} k={k} rho={rho}: path {len(path) - 1} moves ({time.time() - t0:.0f}s)")
    sdf, rdf = pd.DataFrame(srows), pd.DataFrame(rrows)
    sdf.to_csv(os.path.join(out, "w3_states.csv"), index=False)
    rdf.to_csv(os.path.join(out, "w3_runs.csv"), index=False)
    return sdf, rdf


def w3_model(sdf):
    """Model inputs and predictions per (n, k, rho, scorer, grid), pooled over graphs and improvable states."""
    rows = []
    for (n, k, rho, scorer), d in sdf.groupby(["n", "k", "rho", "scorer"]):
        imp = d[d.improvable]
        if imp.empty:
            continue
        T_ex, c_scores, c_s = imp.T_ex.mean(), imp.c_scores.mean(), imp.c_s.mean()
        for (a, b) in GRIDS:
            pi = float(imp[f"hit_{a}x{b}"].mean())
            hits = imp[imp[f"hit_{a}x{b}"]]
            q = float(hits[f"q_{a}x{b}"].mean()) if len(hits) else 0.0
            c_round = c_scores + a * b * c_s
            rows.append({"n": n, "k": k, "rho": rho, "scorer": scorer, "grid": f"{a}x{b}", "C": a * b,
                         "states": len(imp),
                         "pi": pi, "q": q, "T_ex": T_ex, "c_round": c_round, "Tex_over_cround": T_ex / c_round,
                         "speedup_model": DIA.hybrid_speedup(pi, q, c_round, T_ex),
                         "speedup_cap": 1.0 / (1.0 - pi) if pi < 1 else float("inf"),
                         "fallback_share": DIA.fallback_share(pi, c_round, T_ex),
                         "headroom": DIA.learning_headroom(pi, q, c_round, T_ex)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ W4

def run_w4(args, out):
    rows = []
    for (n, gs) in [(n, gs) for n in args.w4_sizes for gs in args.w4_graphs]:
        g = three_groups(900 + n + 7 * gs, n)
        for k in args.w4_ks:
            est = ExposureEstimator(g, R=args.w4_R, seed=8000 + n + k + 11 * gs, device="cpu")
            ref, floor, _ = floor_start(est, k)
            start = ref                                      # descend from the greedy reference
            floor_acc = floor * (1 + FLOOR_RTOL)
            configs = [("default", sp, (1.0, 0.5)) for sp in args.w4_setpoints] + \
                      [("scaled", sp, (1.0 / sp, 0.5 / sp)) for sp in args.w4_setpoints if sp > 0]
            for fb in (None, "exhaustive"):
                for gains_name, sp, gains in configs:
                    stats = {"proposed": 0, "below_floor": 0, "mu": []}
                    orig_moves, orig_update = IMPM.simple_moves, PRC.PIController.update

                    def spy_moves(state, w, na, nd, _o=orig_moves, _s=stats):
                        moves = _o(state, w, na, nd)
                        for u, v in moves:
                            _s["proposed"] += 1
                            _s["below_floor"] += int(state.total + state.swap_delta(u, v)[1] < floor_acc)
                        return moves

                    def spy_update(self, *a, _o=orig_update, _s=stats, **kw):
                        res = _o(self, *a, **kw)
                        _s["mu"].append(res[2])
                        return res

                    IMPM.simple_moves, PRC.PIController.update = spy_moves, spy_update
                    try:
                        ts = time.perf_counter()
                        seeds, hist = improve(est, start, floor, proposer="simple", audit=False, fallback=fb,
                                              price_setpoint=sp, price_gains=gains, max_rounds=args.w4_rounds)
                        secs = time.perf_counter() - ts
                    finally:
                        IMPM.simple_moves, PRC.PIController.update = orig_moves, orig_update
                    mu = np.asarray(stats["mu"]) if stats["mu"] else np.zeros(1)
                    sign = np.sign(np.diff(mu)) if len(mu) > 2 else np.zeros(1)
                    rows.append({"n": n, "graph": gs, "k": k, "fallback": fb or "none", "gains": gains_name,
                                 "setpoint": sp,
                                 "kp": gains[0], "ki": gains[1], "secs": secs,
                                 "moves": sum(h["accepted"] for h in hist),
                                 "proposer_moves": sum(h["accepted"] and not h.get("fallback_move", False)
                                                       for h in hist),
                                 "mu_reversals": int(np.sum(sign[1:] * sign[:-1] < 0)) if len(sign) > 1 else 0,
                                 "fallback_moves": sum(h.get("fallback_move", False) for h in hist),
                                 "proposed": stats["proposed"],
                                 "below_floor_share": stats["below_floor"] / max(stats["proposed"], 1),
                                 "mu_mean": float(mu.mean()), "mu_max": float(mu.max()),
                                 "mu_zero_share": float(np.mean(mu == 0.0)),
                                 "D_start": hist[0]["disp"], "D_final": sample_disp(est, seeds)[0],
                                 "slack_final_sample": (CoverState(est, seeds).total - floor) / floor})
            log(f"  w4 n={n} graph {gs} k={k} done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "w4.csv"), index=False)
    return df


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "w1", "w2", "w3", "w4"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    # full settings
    args.truth_R, args.truth_chunk = 1024, 128
    args.w1_graphs, args.w1_ks, args.w1_R, args.w1_rounds, args.w1_T = [0, 1, 2], [25, 50], 64, 150, 30
    args.w1_cohesion, args.w1_rhos = ["cohesive", "loose"], [0.99, 0.97]
    args.w1_graphs = list(range(10))
    args.w2_graphs, args.w2_ks, args.w2_R, args.w2_R_bound, args.w2_rounds = [0, 1], [25, 50], 64, 256, 150
    args.w3_sizes, args.w3_ks, args.w3_R, args.w3_rounds, args.w3_ntime = [2000, 8000, 32000], [25, 50], 16, 300, 64
    args.w3_graphs, args.w4_graphs = [0, 1, 2], [0, 1, 2]
    args.w3_ks, args.w3_rhos = [25, 50, 100], [0.99, 0.97]
    args.w4_sizes, args.w4_ks, args.w4_R, args.w4_rounds = [2000, 8000], [25, 50], 16, 300
    args.w4_setpoints = [0.0, 0.0025, 0.005, 0.01]
    if args.quick:
        args.truth_R, args.truth_chunk = 128, 64
        args.w1_graphs, args.w1_ks, args.w1_T, args.w1_rounds, args.w1_cohesion = [0], [25], 6, 30, ["cohesive"]
        args.w1_rhos = [0.99]
        args.w2_graphs, args.w2_ks, args.w2_R_bound, args.w2_rounds = [0], [25], 64, 30
        args.w3_sizes, args.w3_ks, args.w3_rounds, args.w3_ntime = [2000], [25], 40, 16
        args.w3_graphs, args.w4_graphs = [0], [0]
        args.w3_ks, args.w3_rhos = [25], [0.97]
        args.w4_sizes, args.w4_ks, args.w4_rounds, args.w4_setpoints = [2000], [25], 40, [0.0, 0.005]
    out = os.path.join(RESULTS_V2, "weakspots_quick" if args.quick else "weakspots")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    summary_path = os.path.join(out, "summary.json")
    summary = json.load(open(summary_path, encoding="utf-8")) if os.path.exists(summary_path) else {}
    parts = ["w1", "w2", "w3", "w4"] if args.part == "all" else [args.part]
    pd.set_option("display.width", 220)
    for part in parts:
        tp = time.time()
        log(f"== {part}")
        if part == "w1":
            df = run_w1(args, out)
            summary["w1"] = df.groupby(["cohesion", "rho", "method"])[["D_lab", "D_hidden", "slack"]].mean().round(5) \
                              .reset_index().to_dict(orient="records")
            log(df.groupby(["cohesion", "rho", "method"])[["D_lab", "D_hidden", "slack", "family",
                                                   "tb_bound"]].mean().round(4).to_string())
        elif part == "w2":
            df = run_w2(args, out)
            summary["w2"] = df.groupby(["kind", "k"])[["lb", "D_local_search_true", "gap"]].mean().round(5) \
                              .reset_index().to_dict(orient="records")
            log(df.groupby(["kind", "k"])[["lb", "mean_floor", "maximin_ub", "D_local_search_true", "gap"]]
                .mean().round(4).to_string())
        elif part == "w3":
            sdf, rdf = run_w3(args, out)
            model = w3_model(sdf)
            model.to_csv(os.path.join(out, "w3_model.csv"), index=False)
            summary["w3_model"] = model.round(5).to_dict(orient="records")
            summary["w3_runs"] = rdf.round(5).to_dict(orient="records")
            log(model.round(3).to_string(index=False))
            log(rdf.round(4).to_string(index=False))
        else:
            df = run_w4(args, out)
            summary["w4"] = df.round(5).to_dict(orient="records")
            log(df.round(4).to_string(index=False))
        summary.setdefault("secs", {})[part] = time.time() - tp
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, default=float)
    log(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
