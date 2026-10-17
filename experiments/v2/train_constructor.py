"""Train the FIM-GAT v2 constructor (fimgat.v2.train.V2Trainer) on the train split under several group definitions.

The constructor has to serve any group definition (docs/design_v2.md section 4), so it trains on the train split
(datasets.splits()["train"] = pokec_z) under four of them at once: region, gender, age and degree. Each definition
is its own prepare_v2 context (graph.with_groups), all built with the same RR seed: RR sets depend only on the
topology, R and the seed (every node is a root at this size), so the four contexts share identical RR sets and
differ only in their group bookkeeping and group features. fit() cycles through them, with k drawn from
config.BUDGETS and rho from the trainer's range. Checkpoints are selected on the validation district pokec_t
(region, gender, degree) from RR samples only (V2Trainer.validate), never on test graphs or on Monte Carlo.

Outputs (results/v2/):
  models/constructor_s{seed}.pt        config.N_TRAIN_SEEDS seeds (--seeds), community adversary on
  models/constructor_noadv_s{seed}.pt  the no-adversary ablation (gamma_comm = 0), --noadv-seeds (default seed 0)
  train_log.csv                        one row per epoch and model: step means of loss, disparity, adversary
                                       community disparity, spread ratio, mu and DMoN loss; the collapse diagnostic
                                       eff_comm (exp-entropy of the adversary's community mass, in [1, K]; mean and
                                       min over the epoch) with n_comm; disparity per group definition; the
                                       validation score at validation epochs; peak GPU memory since the run began
Checkpoints go through trainer.save(path, **meta); meta holds train_time, graph, attributes, code_tag and the
settings, and save() adds mu_ref (k -> dual price), which run_t1_t2.py passes to select_seeds_v2.

Resumable: a model whose checkpoint exists is skipped unless --force. Checkpoints are written to a temp file and
then renamed, so an interrupted save never looks finished. Retraining a model replaces its rows in the log.
--quick (2 epochs, 4 steps, seed 0, k in {25, 50}) writes under results/v2/quick/, so a smoke run never shadows
real checkpoints.

One GPU job at a time on the 8 GB card (docs/research_protocol.md section 3). From the repo root:
    mkdir -p results/v2
    cd experiments/v2 && PYTHONIOENCODING=utf-8 python -u -W ignore train_constructor.py >> ../../results/v2/train_constructor.log 2>&1
"""
import argparse
import gc
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

from common_v2 import code_tag, device, gpu_name, peak_gpu_gb, preset_config, progress, reset_peak, timed  # noqa: E402
from fimgat.v2 import datasets as D                                                            # noqa: E402
from fimgat.v2.config import BUDGETS, MODELS_V2, N_TRAIN_SEEDS, RESULTS_V2, RR_PER_NODE, TIER   # noqa: E402
from fimgat.v2.train import V2Trainer, prepare_v2                                              # noqa: E402

NAN = float("nan")
TRAIN_ATTRS = ("region", "gender", "age", "degree")
VAL_ATTRS = ("region", "gender", "degree")
STEP_KEYS = ("loss", "disp", "comm_disp", "ratio", "mu", "dmon", "n_comm", "eff_comm")   # V2Trainer.step output
QUICK_DIR = os.path.join(RESULTS_V2, "quick")

# UNVERIFIED: rough costs on the 8 GB laptop, scaled by hand from v1's Pokec-z training (research_protocol
# section 6: about 5.5 min for 400 checkpointed steps plus 8 validations; Pokec-z has about 0.9M arcs). The cost
# of the v2 adversary steps is a guess. Order of magnitude only.
SEC_STEP_PER_ARC = 1.2 / 0.9e6       # one training step (policy step plus two adversary steps), per arc
SEC_SELECT_PER_ARC = 0.6 / 0.9e6     # one select_seeds_v2 call during validation (16 samples), per arc
SEC_PREP_PER_SET = 1.2e-5            # prepare_v2 per RR set (sampling, features, greedy reference)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ graphs and contexts

def with_attribute(base, attr):
    """base under the group definition `attr`: same topology, name (so the same RR and MC streams) and meta,
    with meta["attribute"] updated, because FIMGraph.with_groups copies meta unchanged."""
    attrs, names = base.meta["attrs"], base.meta["attr_names"]
    if attr not in attrs:
        raise KeyError(f"{base.name} has no attribute {attr!r}; available: {sorted(attrs)}")
    g = base.with_groups(attrs[attr], names[attr])
    g.meta["attribute"] = attr
    return g


def load_bases(names):
    missing = [x for x in names if not D.available(x)]
    if missing:
        raise SystemExit(f"not built: {missing}. Pokec districts: experiments/build_pokec.py (pokec_z) and "
                         f"experiments/v2/build_datasets.py --only pokec_t")
    return {x: D.load(x) for x in names}


def build_contexts(bases, attrs, ks, dev, R, rr_seed, label_graph):
    """One prepare_v2 context per (graph, attribute), all with the same rr_seed. Returns (ctxs, labels, secs),
    labels mapping id(ctx) to the name used in the per-definition log columns."""
    ctxs, labels, secs = [], {}, 0.0
    for name, base in bases.items():
        for a in attrs:
            g = with_attribute(base, a)
            ctx, s = timed(lambda: prepare_v2(g, ks, dev, R=R, rr_seed=rr_seed))
            secs += s
            ctxs.append(ctx)
            labels[id(ctx)] = f"{name}_{a}" if label_graph else a
            log(f"  {name}:{a:8s} G={g.n_groups:3d}  RR sets={ctx.est.n_sets:,} nnz={ctx.est.nnz:,} "
                f"(mean RR size {ctx.est.mean_rr_size:.2f})  {s:.1f}s")
    return ctxs, labels, secs


# ------------------------------------------------------------------ per-epoch log

def _mean(vals):
    v = np.asarray([NAN if x is None else x for x in vals], dtype=np.float64)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else NAN


def _min(vals):
    v = np.asarray([NAN if x is None else x for x in vals], dtype=np.float64)
    v = v[~np.isnan(v)]
    return float(v.min()) if v.size else NAN


def read_rows(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def write_rows(rows, path):
    """Atomic rewrite (temp file, then os.replace), as RecorderV2.save."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                         # Windows: the CSV is briefly open elsewhere
            if i == 9:
                raise
            time.sleep(0.5)


class EpochLog:
    """Per-epoch rows of one training run in the shared log CSV, rewritten atomically after every epoch.

    Steps are grouped into epochs of steps_per_epoch (the value handed to fit). A validation epoch's row is
    written once its validation score is known; any other epoch's row when the next epoch starts. Rows of an
    earlier run of the same model file are dropped first, so a retrain replaces its log."""

    def __init__(self, path, steps_per_epoch, base, labels):
        self.path, self.per_epoch = path, int(steps_per_epoch)
        self.base, self.labels = dict(base), dict(labels)
        self.names = sorted(set(self.labels.values()))
        self.rows = [r for r in read_rows(path) if r.get("model") != self.base["model"]]
        self.stats, self.epoch, self.pending = [], 0, None
        self.best_epoch, self.best_val = None, NAN
        self.t0 = time.time()
        self.bar = None                                 # a progress bar over the run's steps (common_v2.progress)

    def on_step(self, ctx, k, rho, T, out):
        if self.pending is not None:                    # the previous epoch was not a validation epoch
            self._flush()
        self.stats.append({**out, "k": int(k), "rho": float(rho), "T": float(T),
                           "label": self.labels.get(id(ctx), "?")})
        if self.bar is not None:
            self.bar.update(1)
            if out.get("loss") is not None:
                self.bar.set_postfix_str(f"loss {float(out['loss']):.4f}", refresh=False)
        if len(self.stats) >= self.per_epoch:
            self.epoch += 1
            self.pending, self.stats = self._summary(self.stats), []

    def on_validate(self, v, fallback):
        if self.pending is None:
            return
        v = float(v)
        # V2Trainer.fit's rule for keeping a snapshot, to name the epoch whose state fit restores
        if self.best_epoch is None or math.isnan(v) or math.isnan(self.best_val) or v > self.best_val:
            self.best_epoch, self.best_val = self.pending["epoch"], v
        self.pending.update(val=v, val_fallback=float(fallback), selected_epoch=self.best_epoch)
        self._flush()

    def close(self):
        if self.pending is not None:
            self._flush()
        if self.stats:                                  # an epoch cut short by an error
            self.epoch += 1
            self.pending, self.stats = self._summary(self.stats, partial=True), []
            self._flush()

    def _summary(self, stats, partial=False):
        row = {**self.base, "epoch": self.epoch, "steps": len(stats), "partial": bool(partial),
               "T": stats[-1]["T"], "k_mean": _mean([s["k"] for s in stats]),
               "rho_mean": _mean([s["rho"] for s in stats])}
        for key in STEP_KEYS:
            row[key] = _mean([s.get(key) for s in stats])
        row["eff_comm_min"] = _min([s.get("eff_comm") for s in stats])
        row["n_comm_min"] = _min([s.get("n_comm") for s in stats])
        for lab in self.names:
            mine = [s for s in stats if s["label"] == lab]
            row[f"disp_{lab}"] = _mean([s.get("disp") for s in mine])
            row[f"ratio_{lab}"] = _mean([s.get("ratio") for s in mine])
        alloc, reserved = peak_gpu_gb()
        row.update(val=NAN, val_fallback=NAN, selected_epoch=NAN if self.best_epoch is None else self.best_epoch,
                   secs=time.time() - self.t0, peak_alloc_gb=alloc, peak_reserved_gb=reserved)
        return row

    def _flush(self):
        row, self.pending = self.pending, None
        self.rows.append(row)
        write_rows(self.rows, self.path)
        if math.isnan(row["val"]):                      # fit() prints the validation epochs itself
            log(f"  ep {row['epoch']:4d}  L={row['loss']:.4f}  disp={row['disp']:.4f}  comm={row['comm_disp']:.4f}"
                f"  spread={row['ratio']:.3f}  mu={row['mu']:.3f}  comms={row['n_comm']:.1f}"
                f" eff={row['eff_comm']:.2f} (min {row['eff_comm_min']:.2f})  T={row['T']:.3f}  ({row['secs']:.0f}s)")


class LoggedTrainer(V2Trainer):
    """V2Trainer whose step() and validate() also report to an EpochLog. fit() is inherited unchanged: it calls
    self.step once per step and self.validate after the last step of each validation epoch."""

    epoch_log = None

    def step(self, ctx, k, rho, T):
        out = super().step(ctx, k, rho, T)
        if self.epoch_log is not None:
            self.epoch_log.on_step(ctx, k, rho, T, out)
        return out

    def validate(self, val_ctxs, ks, rhos=(0.99,)):
        v = super().validate(val_ctxs, ks, rhos)
        if self.epoch_log is not None:
            self.epoch_log.on_validate(v, self.val_fallback)
        return v


# ------------------------------------------------------------------ one model

def model_path(models_dir, variant, seed):
    return os.path.join(models_dir, f"constructor_s{seed}.pt" if variant == "constructor"
                        else f"constructor_noadv_s{seed}.pt")


def train_one(variant, seed, gamma, path, train_ctxs, val_ctxs, labels, settings, args, dev, ckpt_flag):
    log(f"\n=== {os.path.basename(path)}: seed {seed}, gamma_comm {gamma} ===")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                              # V2Trainer seeds torch again before building the nets
    tr = LoggedTrainer(dev, seed=seed, gamma_comm=gamma, checkpoint=ckpt_flag)
    base = {"script": "train_constructor", "model": os.path.basename(path), "variant": variant, "seed": seed,
            "gamma_comm": gamma, "train_graphs": "+".join(settings["train_graphs"]),
            "attributes": "+".join(settings["attributes"]), "code_tag": code_tag(), "gpu": gpu_name(),
            "preset": settings["preset"], "quick": bool(args.quick)}
    elog = EpochLog(settings["log"], settings["steps_per_epoch"], base, labels)
    elog.bar = progress(settings["epochs"] * settings["steps_per_epoch"], os.path.basename(path), unit="step")
    tr.epoch_log = elog
    reset_peak()
    try:
        best = tr.fit(train_ctxs, val_ctxs, settings["ks"], settings["epochs"],
                      steps_per_epoch=settings["steps_per_epoch"], eval_every=settings["eval_every"], verbose=True)
    finally:
        elog.close()
        elog.bar.close()
    alloc, reserved = peak_gpu_gb()
    names = settings["train_graphs"]
    meta = dict(train_time=float(tr.train_time), graph=names[0] if len(names) == 1 else list(names),
                attributes=list(settings["attributes"]), code_tag=code_tag(), variant=variant, seed=seed,
                gamma_comm=gamma, val_graphs=list(settings["val_graphs"]),
                val_attributes=list(settings["val_attributes"]), ks=list(settings["ks"]), epochs=settings["epochs"],
                steps_per_epoch=settings["steps_per_epoch"], eval_every=settings["eval_every"], R=settings["R"],
                rr_seed=settings["rr_seed"], val_rr_seed=settings["val_rr_seed"], best_val=float(best),
                selected_epoch=elog.best_epoch, checkpointing=bool(ckpt_flag), preset=settings["preset"],
                gpu=gpu_name(), peak_alloc_gb=alloc, peak_reserved_gb=reserved, quick=bool(args.quick),
                saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = path + ".tmp"
    tr.save(tmp, **meta)
    os.replace(tmp, path)
    mu = ", ".join(f"{k}: {v:.3f}" for k, v in tr.mu_table().items())
    log(f"  saved {path}: best val {best:.4f} (epoch {elog.best_epoch}), train {tr.train_time:.0f}s, "
        f"peak GPU {alloc:.2f}/{reserved:.2f} GB (allocated/reserved); mu_ref {{{mu}}}")
    del tr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ main

def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help=f"constructor seeds (default 0..{N_TRAIN_SEEDS - 1}; --quick: 0)")
    ap.add_argument("--noadv-seeds", nargs="*", type=int, default=None,
                    help="seeds of the no-adversary ablation (default 0; pass the flag alone for none)")
    ap.add_argument("--attributes", nargs="+", default=list(TRAIN_ATTRS), help="training group definitions")
    ap.add_argument("--val-attributes", nargs="+", default=list(VAL_ATTRS), help="validation group definitions")
    ap.add_argument("--ks", nargs="+", type=int, default=None, help="budgets (default config.BUDGETS)")
    ap.add_argument("--epochs", type=int, default=None, help="default 40 (--quick: 2)")
    ap.add_argument("--steps-per-epoch", type=int, default=None,
                    help="default 3 x the number of training contexts, 12 (--quick: 4)")
    ap.add_argument("--eval-every", type=int, default=None, help="validate every N epochs (default 5; --quick: 1)")
    ap.add_argument("--R", type=int, default=None, help="RR sets per node (default config.RR_PER_NODE of the tier)")
    ap.add_argument("--rr-seed", type=int, default=1000, help="RR seed of the training contexts")
    ap.add_argument("--val-rr-seed", type=int, default=2000, help="RR seed of the validation contexts")
    ap.add_argument("--gamma-comm", type=float, default=0.5, help="adversary weight of the constructor runs")
    ap.add_argument("--checkpointing", choices=("auto", "on", "off"), default="auto",
                    help="activation checkpointing; auto: 2m > the preset's checkpoint_edges")
    ap.add_argument("--preset", default=None, help="memory preset (8gb / 16gb / 40gb); default from the GPU")
    ap.add_argument("--models-dir", default=None, help="default results/v2/models (--quick: results/v2/quick/models)")
    ap.add_argument("--log", default=None, help="default results/v2/train_log.csv (--quick: results/v2/quick/...)")
    ap.add_argument("--force", action="store_true", help="retrain models whose checkpoint exists")
    ap.add_argument("--quick", action="store_true", help="a few steps of one seed, written under results/v2/quick")
    return ap.parse_args()


def main():
    args = parse()
    q = args.quick
    splits = D.splits()
    train_names, val_names = list(splits["train"]), list(splits["val"])
    seeds = args.seeds if args.seeds is not None else ([0] if q else list(range(N_TRAIN_SEEDS)))
    noadv = args.noadv_seeds if args.noadv_seeds is not None else [0]
    ks = [int(k) for k in (args.ks or (BUDGETS[:2] if q else BUDGETS))]
    n_ctx = len(train_names) * len(args.attributes)
    epochs = args.epochs or (2 if q else 40)
    spe = args.steps_per_epoch or (4 if q else 3 * n_ctx)
    eval_every = args.eval_every or (1 if q else 5)
    if min(epochs, spe, eval_every) < 1:
        raise SystemExit("--epochs, --steps-per-epoch and --eval-every must be >= 1")
    root = QUICK_DIR if q else RESULTS_V2
    models_dir = args.models_dir or (os.path.join(QUICK_DIR, "models") if q else MODELS_V2)
    log_path = args.log or os.path.join(root, "train_log.csv")
    os.makedirs(models_dir, exist_ok=True)

    runs = [("constructor", s, float(args.gamma_comm)) for s in seeds] + [("noadv", s, 0.0) for s in noadv]
    todo = []
    for variant, s, gamma in runs:
        path = model_path(models_dir, variant, s)
        if os.path.exists(path) and not args.force:
            log(f"{os.path.basename(path)} exists: skipped (--force retrains)")
        else:
            todo.append((variant, s, gamma, path))
    if not todo:
        log("nothing to train")
        return

    dev = device()
    pname, preset = preset_config(args.preset)
    log(f"device {dev} ({gpu_name()}), preset {pname}, code {code_tag()}; run one GPU job at a time")
    if dev == "cpu":
        log("WARNING: no CUDA device; training on the CPU is very slow")
    train_bases, val_bases = load_bases(train_names), load_bases(val_names)
    tiers = {TIER.get(x) for x in train_names + val_names}
    R = args.R or RR_PER_NODE[TIER[train_names[0]]]
    if R is None:
        raise SystemExit(f"no RR_PER_NODE for tiers {tiers}: pass --R")
    if args.checkpointing == "auto":
        ckpt_flag = any(2 * g.m > preset["checkpoint_edges"] for g in train_bases.values())
    else:
        ckpt_flag = args.checkpointing == "on"

    # rough estimate before anything long
    n_val_ep = sum(1 for e in range(epochs) if (e + 1) % eval_every == 0 or e == epochs - 1)
    m_train = float(np.mean([g.m for g in train_bases.values()]))
    t_step = SEC_STEP_PER_ARC * m_train
    t_val = sum(SEC_SELECT_PER_ARC * g.m * len(args.val_attributes) * len(ks) for g in val_bases.values())
    t_prep = SEC_PREP_PER_SET * R * (sum(g.n for g in train_bases.values()) * len(args.attributes)
                                     + sum(g.n for g in val_bases.values()) * len(args.val_attributes))
    t_run = epochs * spe * t_step + n_val_ep * t_val
    log(f"train {train_names} x {args.attributes}, validate {val_names} x {args.val_attributes}, ks {ks}, R={R}, "
        f"checkpointing {ckpt_flag}")
    log(f"{len(todo)} model(s) to train: {[os.path.basename(p) for *_, p in todo]}; {epochs} epochs x {spe} steps, "
        f"validation every {eval_every} epoch(s)")
    log(f"rough estimate (UNVERIFIED unit costs): preparing contexts {t_prep / 60:.1f} min, then about "
        f"{t_run / 60:.0f} min per model ({epochs * spe} steps x {t_step:.2f}s + {n_val_ep} validations x "
        f"{t_val:.0f}s); total about {(t_prep + len(todo) * t_run) / 3600:.1f} h")

    log("\npreparing training contexts (identical RR sets across definitions: same graph, R and seed)")
    train_ctxs, labels, s_train = build_contexts(train_bases, args.attributes, ks, dev, R, args.rr_seed,
                                                 label_graph=len(train_names) > 1)
    log("preparing validation contexts")
    val_ctxs, _, s_val = build_contexts(val_bases, args.val_attributes, ks, dev, R, args.val_rr_seed,
                                        label_graph=len(val_names) > 1)
    alloc, reserved = peak_gpu_gb()
    log(f"contexts ready in {s_train + s_val:.0f}s; GPU peak so far {alloc:.2f}/{reserved:.2f} GB")
    # UNVERIFIED: all contexts (4 training + 3 validation on Pokec districts at R = 16) are assumed to fit next to
    # a training step on the 8 GB card; if not, lower --R or train with fewer --attributes.

    settings = dict(train_graphs=train_names, attributes=list(args.attributes), val_graphs=val_names,
                    val_attributes=list(args.val_attributes), ks=ks, epochs=epochs, steps_per_epoch=spe,
                    eval_every=eval_every, R=R, rr_seed=args.rr_seed, val_rr_seed=args.val_rr_seed, preset=pname,
                    log=log_path)
    t0 = time.time()
    for variant, s, gamma, path in todo:
        train_one(variant, s, gamma, path, train_ctxs, val_ctxs, labels, settings, args, dev, ckpt_flag)
    log(f"\ndone: {len(todo)} model(s) in {(time.time() - t0) / 60:.1f} min; log {log_path}")


if __name__ == "__main__":
    main()
