"""Train the learned proposer (docs/design_v2.md sections 7, 18.3 and 21) by imitation of exhaustive single-swap local
search (fimgat.v2.imitation), only when the W3 rule says so.

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore train_proposer.py --dry-run
    PYTHONIOENCODING=utf-8 python -u -W ignore train_proposer.py >> ../../results/v2/train_proposer.log 2>&1

The gate: tune_v2.py's results/v2/tuning.json must say "w3": {"train": true} (design section 18.3). --ignore-gate is
for development only: it trains anyway and writes under results/v2/quick/, so it never shadows a real checkpoint.

Fixed before any run (design section 21):
  graphs   the train split (pokec_z) under region, gender, age and degree, as the constructor; validation: the
           validation split (pokec_t) under region, gender and degree. RR samples only (every node a root, R of the
           tier), never Monte Carlo; never a test graph.
  states   imitation.generate_imitation_data along exhaustive single-swap local-search trajectories from the greedy-family
           starts (Greedy, Parity-Greedy, Welfare-Greedy; those meeting the floor) at every budget of config.BUDGETS and
           rho in (RHO_C2, RHO_C1), at improve()'s acceptance floor with the tuned delta, --states-per-query states each,
           a quarter of the moves random improving swaps (explore). Validation states: the same floors, no
           exploration. Features carry spread price 0: LearnedProposer's own controller gives 0 while the floor holds
           (theory W4.2), so training and inference see the same inputs.
  model    PolicyV2, initialised from the constructor checkpoint of the same train seed when one exists, else fresh;
           every layer is trained with imitation.train_proposer (listwise loss toward the exact improvements, AdamW),
           features computed on the fly. The epoch kept is the one with the best validation recall@128 (the share of
           improvable validation states whose 128 best proposals hold an improving swap, as many swaps as improve()'s
           16 x 8 grid); the starting policy can win.
Outputs: models/proposer_s{seed}.pt for every --seeds (the train.load_policy format, which pipeline_v2.load_models
reads); train_proposer_log.csv (one row per evaluation). The checkpoint meta also holds the simple proposer's and the
learned proposer's validation recall@C (config.T3_RECALL_C), and the learned proposer's measured per-round cost relative
to the simple proposer's (cost_ratio), with W3's headroom over it.
Resumable: a seed whose checkpoint exists is skipped unless --force. One GPU job at a time; ask before multi-hour runs.
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pipeline_v2 as P                                                              # noqa: E402
from common_v2 import code_tag, device, gpu_name, peak_gpu_gb, preset_config, reset_peak   # noqa: E402
from train_constructor import build_contexts, load_bases, log, read_rows, write_rows  # noqa: E402
from fimgat.baselines import RRGreedy                                                # noqa: E402
from fimgat.v2 import datasets as D                                                  # noqa: E402
from fimgat.v2 import imitation as I                                                 # noqa: E402
from fimgat.v2.config import (BUDGETS, DELTA_MARGIN, MODELS_V2, N_TRAIN_SEEDS, RESULTS_V2, RHO_C1,   # noqa: E402
                              RHO_C2, RR_PER_NODE, T3_RECALL_C, TIER, rr_floor)
from fimgat.v2.cover import CoverState                                               # noqa: E402
from fimgat.v2.features import IN_DIM                                                # noqa: E402
from fimgat.v2.improve import FLOOR_RTOL                                             # noqa: E402
from fimgat.v2.model import PolicyV2                                                 # noqa: E402
from fimgat.v2.train import _plain, exact_rates, load_policy                         # noqa: E402

QUICK_DIR = os.path.join(RESULTS_V2, "quick")
TRAIN_ATTRS = ("region", "gender", "age", "degree")
VAL_ATTRS = ("region", "gender", "degree")
RHOS = (RHO_C2, RHO_C1)            # floors of the training states
SELECT_C = 128                     # validation recall@C that selects the epoch
STARTS = ("greedy", "parity", "welfare")     # RRGreedy methods: LocalSearch's greedy-family starts
EXPLORE = 0.25
# UNVERIFIED rough unit costs on the 8 GB laptop (nothing timed yet): order of magnitude only
SEC_STATE_PER_NNZ = 1.5e-7         # one imitation state (exact targets + one best_swap), per RR-set entry
SEC_STEP_PER_ARC = 1.0 / 0.9e6     # one training step (features, forward and backward), per arc


# ------------------------------------------------------------------ states

def acceptance_floor(rho, total_ref, delta):
    """improve()'s floor for accepted moves: config.rr_floor(rho, total_ref, delta), raised by FLOOR_RTOL."""
    f = rr_floor(rho, total_ref, delta)
    return f + FLOOR_RTOL * abs(f)


def starts_for(ctx, rg, k, floor):
    """The greedy-family starts of budget k that meet the floor exactly; Greedy's set when none does (the
    imitation data then repairs it first, without recording the repair)."""
    out = []
    for name in STARTS:
        s = [int(v) for v in (ctx.greedy_seeds[k] if name == "greedy" else getattr(rg, name)(k))]
        if exact_rates(ctx.est, s)[1] >= floor and s not in out:
            out.append(s)
    return out or [[int(v) for v in ctx.greedy_seeds[k]]]


def make_states(ctxs, labels, ks, rhos, per_query, explore, delta, seed, dev):
    """Imitation records over every (context, budget, floor), each tagged with its context's index (rec["ctx"]),
    label and floor ratio; features are left to train_proposer (on the fly)."""
    rng = np.random.default_rng(seed)
    recs, t0 = [], time.time()
    for i, ctx in enumerate(ctxs):
        rg = RRGreedy(ctx.est)
        for k in ks:
            total_ref = float(ctx.total_ref[k])
            for rho in rhos:
                floor = acceptance_floor(rho, total_ref, delta)
                got = I.generate_imitation_data(ctx, ctx.est, starts_for(ctx, rg, k, floor), floor, rho, total_ref,
                                                per_query, rng, mu=None, explore=explore, device=dev, ctx_id=i)
                for r in got:
                    r["label"] = labels.get(id(ctx), str(i))
                recs += got
        del rg
    log(f"  {len(recs)} states from {len(ctxs)} contexts in {time.time() - t0:.0f}s")
    return recs


# ------------------------------------------------------------------ validation and cost

def recall(policy, ctxs, recs, C):
    """recall@C over the records (their own floors and targets); NaN without improvable states."""
    return I.recall_at_c(policy, ctxs, recs, C)


def cost_ratio(policy, ctxs, recs, n=8):
    """Mean time of the learned proposer's scores (features + policy) over that of the simple proposer's scores, on up
    to n records: the per-round cost ratio c of the W3 rule (theory Proposition W3.2)."""
    t_l, t_s = [], []
    for rec in recs[:n]:
        ctx = ctxs[int(rec["ctx"])]
        state = CoverState(ctx.est, rec["seeds"])
        t = time.perf_counter()
        I.policy_scores(policy, ctx, state, rec["rho"], rec["total_ref"], 0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_l.append(time.perf_counter() - t)
        t = time.perf_counter()
        I.simple_scores(state, rec["floor"])
        t_s.append(time.perf_counter() - t)
    return float(np.mean(t_l) / max(np.mean(t_s), 1e-9)) if t_l else float("nan")


# ------------------------------------------------------------------ one model

def model_path(models_dir, seed):
    return os.path.join(models_dir, f"proposer_s{seed}.pt")


def initial_policy(seed, constructor_path, args, dev):
    """(PolicyV2, where it came from): the constructor checkpoint of this seed when it exists, else a fresh net."""
    if constructor_path and os.path.exists(constructor_path):
        pol, ck = load_policy(constructor_path, dev)
        hp = ck.get("hparams", {})
        return pol, {"init": os.path.basename(constructor_path), "hidden": hp.get("hidden", 64),
                     "layers": hp.get("layers", 3), "heads": hp.get("heads", 4)}
    torch.manual_seed(seed)
    pol = PolicyV2(IN_DIM, args.hidden, args.layers, args.heads).to(dev)
    return pol, {"init": "fresh", "hidden": args.hidden, "layers": args.layers, "heads": args.heads}


def train_one(seed, path, train_ctxs, recs, val_ctxs, val_recs, simple_recall, settings, args, dev, ckpt_flag):
    log(f"\n=== {os.path.basename(path)}: seed {seed} ===")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pol, init = initial_policy(seed, settings["constructor_pattern"].format(seed=seed), args, dev)
    log(f"  initial policy: {init['init']}")
    reset_peak()
    t0 = time.time()
    hist = I.train_proposer(pol, train_ctxs, recs, args.epochs, args.lr, dev, temp=args.temp, seed=seed,
                            val_fn=lambda p: recall(p, val_ctxs, val_recs, SELECT_C), eval_every=1,
                            checkpoint=ckpt_flag, on_the_fly=True)
    secs = time.time() - t0
    alloc, reserved = peak_gpu_gb()
    best = max(hist, key=lambda e: (e["val"] if e["val"] == e["val"] else -math.inf))
    learned = recall(pol, val_ctxs, val_recs, list(T3_RECALL_C))
    c = cost_ratio(pol, val_ctxs, val_recs)
    w3 = settings.get("w3") or {}
    H = w3.get("H")
    meta = dict(role="proposer", seed=seed, init=init["init"], train_graphs=settings["train_graphs"],
                attributes=settings["attributes"], val_graphs=settings["val_graphs"],
                val_attributes=settings["val_attributes"], ks=settings["ks"], rhos=list(RHOS),
                states=len(recs), val_states=len(val_recs), states_per_query=args.states_per_query, explore=EXPLORE,
                epochs=args.epochs, lr=args.lr, temp=args.temp, delta=settings["delta"], R=settings["R"],
                rr_seed=args.rr_seed, val_rr_seed=args.val_rr_seed, selected_epoch=int(best["epoch"]),
                best_val_recall=float(best["val"]), val_recall={str(k): float(v) for k, v in learned.items()},
                simple_val_recall={str(k): float(v) for k, v in simple_recall.items()}, cost_ratio=c,
                w3_H=H, w3_H_over_cost=(float(H) / c if isinstance(H, (int, float)) and c == c and c > 0 else None),
                checkpointing=bool(ckpt_flag), preset=settings["preset"], gpu=gpu_name(), code_tag=code_tag(),
                peak_alloc_gb=alloc, peak_reserved_gb=reserved, train_time=secs, ignore_gate=bool(args.ignore_gate),
                saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    ck = {"policy": pol.state_dict(), "rounds": 3, "mu_ref": {},
          "hparams": {"seed": seed, "in_dim": IN_DIM, "hidden": init["hidden"], "layers": init["layers"],
                      "heads": init["heads"], "role": "proposer"},
          "log": _plain(hist), "train_time": float(secs), "meta": _plain(meta)}
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)
    rows = [r for r in read_rows(settings["log"]) if r.get("model") != os.path.basename(path)]
    for e in hist:
        rows.append({"script": "train_proposer", "model": os.path.basename(path), "seed": seed, "epoch": e["epoch"],
                     f"val_recall_{SELECT_C}": e["val"], "loss": e["loss"], "steps": e["steps"], "secs": e["secs"],
                     f"simple_val_recall_{SELECT_C}": simple_recall.get(SELECT_C, float("nan")),
                     "selected_epoch": int(best["epoch"]), "init": init["init"], "code_tag": code_tag(),
                     "gpu": gpu_name(), "preset": settings["preset"], "ignore_gate": bool(args.ignore_gate)})
    write_rows(rows, settings["log"])
    log(f"  saved {path}: validation recall@{SELECT_C} {best['val']:.3f} (epoch {best['epoch']}; simple proposer "
        f"{simple_recall.get(SELECT_C, float('nan')):.3f}), cost ratio {c:.2f}, {secs:.0f}s, peak GPU "
        f"{alloc:.2f}/{reserved:.2f} GB")
    del pol
    P.free()


# ------------------------------------------------------------------ main

def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help=f"default 0..{N_TRAIN_SEEDS - 1}")
    ap.add_argument("--attributes", nargs="+", default=list(TRAIN_ATTRS))
    ap.add_argument("--val-attributes", nargs="+", default=list(VAL_ATTRS))
    ap.add_argument("--ks", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--states-per-query", type=int, default=8, help="training states per (graph, attribute, k, rho)")
    ap.add_argument("--val-states-per-query", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.1, help="listwise target temperature (imitation.proposer_loss)")
    ap.add_argument("--hidden", type=int, default=64, help="fresh nets only (a constructor init keeps its own)")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--R", type=int, default=None, help="RR sets per node (default config.RR_PER_NODE of the tier)")
    ap.add_argument("--rr-seed", type=int, default=3000, help="RR seed of the training contexts")
    ap.add_argument("--val-rr-seed", type=int, default=4000, help="RR seed of the validation contexts")
    ap.add_argument("--data-seed", type=int, default=0, help="seed of the exploration moves")
    ap.add_argument("--checkpointing", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--tuning", default=None, help="tune_v2.py's output (default results/v2/tuning.json)")
    ap.add_argument("--constructor-pattern", default=os.path.join(MODELS_V2, "constructor_s{seed}.pt"),
                    help="initial weights per seed when the file exists")
    ap.add_argument("--models-dir", default=None, help="default results/v2/models (--ignore-gate: results/v2/quick)")
    ap.add_argument("--log", default=None, help="default results/v2/train_proposer_log.csv")
    ap.add_argument("--ignore-gate", action="store_true", help="development only: train although W3 said no")
    ap.add_argument("--force", action="store_true", help="retrain seeds whose checkpoint exists")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    tuning = P.load_tuning(args.tuning, required=False)
    w3 = tuning.get("w3") or {}
    if not w3.get("train") and not args.ignore_gate:
        why = w3.get("reason", "no W3 result in the tuning file; run tune_v2.py first")
        log(f"not trained: the W3 rule did not ask for a learned proposer ({why}). T3 is then reported as not run "
            f"under the pre-registered rule (design section 18.3).")
        return
    root = QUICK_DIR if args.ignore_gate else RESULTS_V2
    models_dir = args.models_dir or (os.path.join(QUICK_DIR, "models") if args.ignore_gate else MODELS_V2)
    log_path = args.log or os.path.join(root, "train_proposer_log.csv")
    seeds = args.seeds if args.seeds is not None else list(range(N_TRAIN_SEEDS))
    todo = [(s, model_path(models_dir, s)) for s in seeds]
    todo = [(s, p) for s, p in todo if args.force or not os.path.exists(p)]
    if not todo:
        log("nothing to train (every checkpoint exists; --force retrains)")
        return
    splits = D.splits()
    train_names, val_names = list(splits["train"]), list(splits["val"])
    delta = float(tuning.get("delta", DELTA_MARGIN))
    ks = sorted({int(k) for k in args.ks})
    n_states = len(train_names) * len(args.attributes) * len(ks) * len(RHOS) * args.states_per_query
    n_val = len(val_names) * len(args.val_attributes) * len(ks) * args.val_states_per_query
    log(f"train {train_names} x {args.attributes}, validate {val_names} x {args.val_attributes}, ks {ks}, rho {RHOS},"
        f" delta {delta}; seeds {[s for s, _ in todo]} -> {models_dir}" + (" (IGNORING THE W3 GATE)"
                                                                            if args.ignore_gate else ""))
    log(f"plan: about {n_states} training and {n_val} validation states, {args.epochs} epochs per seed")
    if args.dry_run:
        return
    os.makedirs(models_dir, exist_ok=True)
    dev = device()
    pname, preset = preset_config(args.preset)
    log(f"device {dev} ({gpu_name()}), preset {pname}, code {code_tag()}; run one GPU job at a time")
    train_bases, val_bases = load_bases(train_names), load_bases(val_names)
    R = args.R or RR_PER_NODE[TIER.get(train_names[0], "medium")]
    if R is None:
        raise SystemExit("no RR_PER_NODE for the training tier: pass --R")
    if args.checkpointing == "auto":
        ckpt_flag = any(2 * g.m > preset["checkpoint_edges"] for g in train_bases.values())
    else:
        ckpt_flag = args.checkpointing == "on"
    m_train = float(np.mean([g.m for g in train_bases.values()]))
    log("\npreparing contexts")
    train_ctxs, labels, s1 = build_contexts(train_bases, args.attributes, ks, dev, R, args.rr_seed,
                                            label_graph=len(train_names) > 1)
    val_ctxs, vlabels, s2 = build_contexts(val_bases, args.val_attributes, ks, dev, R, args.val_rr_seed,
                                           label_graph=len(val_names) > 1)
    nnz = float(np.mean([c.est.nnz for c in train_ctxs]))
    log(f"rough estimate (UNVERIFIED unit costs): states {(n_states + n_val) * SEC_STATE_PER_NNZ * nnz / 60:.0f} min, "
        f"then about {args.epochs * n_states * SEC_STEP_PER_ARC * m_train / 60:.0f} min per seed")
    log("generating training states (exhaustive local-search trajectories, a quarter random improving moves)")
    recs = make_states(train_ctxs, labels, ks, RHOS, args.states_per_query, EXPLORE, delta, args.data_seed, dev)
    log("generating validation states")
    val_recs = make_states(val_ctxs, vlabels, ks, RHOS, args.val_states_per_query, 0.0, delta, args.data_seed + 1,
                           dev)
    if not recs or not val_recs:
        raise SystemExit("no improvable state was found: the searches start at local optima; nothing to imitate")
    simple_recall = recall("simple", val_ctxs, val_recs, list(T3_RECALL_C))
    log("simple proposer, validation recall@C: " + ", ".join(f"C={c}: {v:.3f}" for c, v in simple_recall.items()))
    settings = dict(train_graphs=train_names, attributes=list(args.attributes), val_graphs=val_names,
                    val_attributes=list(args.val_attributes), ks=ks, delta=delta, R=R, preset=pname, log=log_path,
                    constructor_pattern=args.constructor_pattern, w3=w3)
    for s, path in todo:
        train_one(s, path, train_ctxs, recs, val_ctxs, val_recs, simple_recall, settings, args, dev, ckpt_flag)
    log(f"\ndone: {len(todo)} proposer(s); log {log_path}")


if __name__ == "__main__":
    main()
