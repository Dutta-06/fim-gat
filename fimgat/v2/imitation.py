"""Learned-proposer training by imitation of exhaustive single-swap local search (docs/design_v2.md section 7).

improvement_matrix       exact labelled-disparity improvement disp(S) - disp(S - u + v) of every swap (seed u out,
                         candidate v in), from the CoverState count primitives with the formula of
                         local_search.best_swap: post-swap hits = hits - L_u + G_v + O_uv (loss, gain and overlap
                         counts), rates = hits * gw. Swaps whose RR total falls below the floor get -inf.
improvement_targets      the same, reduced without storing the (k, n) matrix: the best improvement of adding each
                         node (max over seeds) and of dropping each seed (max over candidates), 0 if none improves.
generate_imitation_data  states along exhaustive local-search trajectories (best_swap steps, plus a share of random
                         improving swaps for diversity), each with its v2 dynamic features and targets.
proposer_loss            listwise softmax cross-entropy (written as a KL) of the add scores over non-seeds and the
                         drop scores over seeds toward softmax(target / temp), targets scaled by their state maximum.
train_proposer           fine-tunes a PolicyV2 on those states (both heads, or the heads only).
recall_at_c              share of improvable states whose top-C proposed swaps contain an improving one.

Conventions follow the improvement phase (improve.py, proposer.py): a state is featurised as round t = 2 of 3 with
the signed prices objective.signed_prices(rates, present, sizes, mu); the drop head scores seeds, higher = better
to remove; a swap improves when it keeps the RR total at or above the floor and lowers the labelled disparity by
more than 1e-12 (verify.accept without a community family). acceptance_floor gives improve()'s floor for accepted
moves. Targets ignore the community disparity D_C, which the verifier still checks at inference. Everything is
scored on RR samples only, never with Monte Carlo (evaluation integrity rule).
"""
# UNVERIFIED: nothing in this module has been executed; shapes, dtypes and devices were checked by reading the
# callee code only.
import copy
import math
import time

import numpy as np
import torch

from .config import PRICE_KI, PRICE_KP, PRICE_SETPOINT, RHO_DEFAULT, rr_floor
from .cover import CoverState
from .features import dynamic_from_state
from .improve import FLOOR_RTOL
from .local_search import TOL, _candidates, best_swap
from .objective import price_weights, signed_prices
from .prices import PIController
from .proposer import _pairs, _top
from .verify import accept, evaluate_state, evaluate_swap

ROUNDS = 3              # constructor rounds in training; an improvement state is fed as its last round (t = 2)
MAX_ELEMS = 1 << 23     # float64 elements per (seeds x candidates x groups) block, as local_search.best_swap


# ------------------------------------------------------------------ small helpers

def acceptance_floor(rho, total_ref):
    """The floor improve() applies to accepted moves: config.rr_floor raised by improve.FLOOR_RTOL (relative)."""
    f = rr_floor(rho, total_ref)
    return f + FLOOR_RTOL * abs(f)


def mu_for(ckpt, k, default=0.0):
    """Spread price of budget k in a checkpoint's mu_ref (int or str keys). A budget the checkpoint was not
    trained on takes the price of the nearest trained budget (in log k); no table gives `default`."""
    table = {}
    for key, v in (ckpt.get("mu_ref") or {}).items():
        try:
            table[int(key)] = float(v)
        except (TypeError, ValueError):
            continue
    table = {kk: v for kk, v in table.items() if kk > 0}
    if not table:
        return float(default)
    k = max(int(k), 1)
    if k in table:
        return table[k]
    return table[min(table, key=lambda kk: (abs(math.log(kk / k)), kk))]


def _mu_of(mu, k):
    if mu is None:
        return 0.0
    if callable(mu):
        return float(mu(int(k)))
    if isinstance(mu, dict):
        return mu_for({"mu_ref": mu}, k)
    return float(mu)


def _device(state, device):
    return torch.device(device) if device is not None else torch.device(getattr(state.est, "device", "cpu"))


def _counts(state):
    """(Lc (k, G), Gc (n, G), (oi, ov, oc)): the integer count primitives of a CoverState, computed once."""
    return state.loss_counts(), state.gain_counts(), state.overlap_counts()


def _columns(state, cand):
    """Sorted distinct non-seed ids: every non-seed when cand is None, else cand without its seeds."""
    if cand is None:
        return np.flatnonzero(~state.is_seed).astype(np.int64)
    c = np.unique(np.asarray(cand, dtype=np.int64).ravel())
    if c.size and (c[0] < 0 or c[-1] >= state.index.n):
        raise ValueError(f"candidate ids must lie in [0, {state.index.n})")
    return c[~state.is_seed[c]]


def _rows(state, rows):
    """Distinct positions in state.seeds (default all, in order)."""
    if rows is None:
        return np.arange(state.k, dtype=np.int64)
    r = np.asarray(rows, dtype=np.int64).ravel()
    if r.size and (r.min() < 0 or r.max() >= state.k):
        raise ValueError(f"rows are positions in state.seeds and must lie in [0, {state.k})")
    if len(np.unique(r)) != len(r):
        raise ValueError("rows must be distinct")
    return r


def _np(t):
    if torch.is_tensor(t):
        t = t.detach().double().reshape(-1).cpu().numpy()
    return np.asarray(t, dtype=np.float64).ravel()


# ------------------------------------------------------------------ exact improvements

def _blocks(state, floor, ri, cols, counts, device, max_elems):
    """Yield (a0, a1, b0, b1, imp): imp is the float64 torch (a1 - a0, b1 - b0) block of improvements of the swaps
    seeds[ri[a0:a1]] out x cols[b0:b1] in; -inf where the post-swap RR total is below floor.

    Post-swap hits per present group are hits - L_u + G_v + O_uv (exact integers in float64); rates = hits * gw
    and total = rates @ sizes, the expressions of CoverState, so values agree with verify.evaluate_swap to
    rounding. Groups that are not present hold no RR set or no node and add exactly 0 to the total."""
    Lc, Gc, (oi, ov, oc) = counts
    ix = state.index
    nr, nc = len(ri), len(cols)
    if nr == 0 or nc == 0:
        return
    P = np.flatnonzero(state.present)
    Gp = len(P)
    floor, d0 = float(floor), float(state.disp)
    dev = _device(state, device)

    def t64(a):
        return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64), dtype=torch.float64, device=dev)

    base = t64(state.hits[P])
    Lr, Gv = t64(Lc[ri][:, P]), t64(Gc[cols][:, P])
    gw, sz = t64(ix.gw[P]), t64(state.sizes[P])
    # overlap entries whose seed row and candidate column are both in this request, grouped by row
    row_of = np.full(state.k, -1, dtype=np.int64)
    row_of[ri] = np.arange(nr, dtype=np.int64)
    col_of = np.full(ix.n, -1, dtype=np.int64)
    col_of[cols] = np.arange(nc, dtype=np.int64)
    e_row, e_col = row_of[oi], col_of[ov]
    keep = np.flatnonzero((e_row >= 0) & (e_col >= 0))
    keep = keep[np.argsort(e_row[keep], kind="stable")]
    e_row, e_col = e_row[keep], e_col[keep]
    O = t64(oc[keep][:, P]) if Gp else None
    er_t, ec_t = torch.as_tensor(e_row, device=dev), torch.as_tensor(e_col, device=dev)
    bounds = np.searchsorted(e_row, np.arange(nr + 1))
    width = max(Gp, 1)
    cv = int(min(nc, max(1, max_elems // width)))
    cu = int(min(nr, max(1, max_elems // (cv * width))))
    for a0 in range(0, nr, cu):
        a1 = min(nr, a0 + cu)
        e0, e1 = int(bounds[a0]), int(bounds[a1])
        for b0 in range(0, nc, cv):
            b1 = min(nc, b0 + cv)
            bv = b1 - b0
            if Gp == 0:                                     # no present group: every total is 0
                tot = torch.zeros(a1 - a0, bv, dtype=torch.float64, device=dev)
                disp = torch.zeros_like(tot)
            else:
                hits = base - Lr[a0:a1, None, :] + Gv[None, b0:b1, :]          # (bu, bv, Gp)
                if e1 > e0:
                    c = ec_t[e0:e1]
                    sel = torch.nonzero((c >= b0) & (c < b1)).flatten()
                    if sel.numel():
                        e = sel + e0
                        hits.view(-1, Gp).index_add_(0, (er_t[e] - a0) * bv + (ec_t[e] - b0), O[e])
                rates = hits * gw
                tot = rates @ sz
                disp = rates.amax(-1) - rates.amin(-1) if Gp >= 2 else torch.zeros_like(tot)
            yield a0, a1, b0, b1, (d0 - disp).masked_fill(tot < floor, -math.inf)


def improvement_matrix(state, floor, device=None, cand=None, rows=None, max_elems=MAX_ELEMS):
    """Exact labelled-disparity improvement disp(S) - disp(S - u + v) of every swap: (imp, cand).

    imp: float64 numpy (len(rows), len(cand)). rows are positions in state.seeds (default all, in order); columns
    are the sorted non-seed candidates (default every non-seed; seeds in a given cand are dropped). Swaps whose
    RR total falls below floor get -inf. A swap improves when imp > 1e-12, the tolerance of verify.accept. Chunked
    over seeds and candidates in torch float64 (device: default the estimator's). The matrix itself is
    k x |cand| float64 (about 1.6 MB per seed at 200k candidates); improvement_targets reduces it on the fly."""
    ri, cols = _rows(state, rows), _columns(state, cand)
    out = np.full((len(ri), len(cols)), -np.inf)
    for a0, a1, b0, b1, imp in _blocks(state, floor, ri, cols, _counts(state), device, max_elems):
        out[a0:a1, b0:b1] = imp.cpu().numpy()
    return out, cols


def improvement_targets(state, floor, device=None, cand=None, max_elems=MAX_ELEMS, counts=None):
    """(add (n,), drop (k,)) float64: the best improvement of adding each node (max over the seed it replaces) and
    of dropping each seed (max over candidates), 0 where no swap improves by more than 1e-12. drop follows
    state.seeds; add is 0 on seeds and, with cand, outside cand. With cand=None every non-seed is scored through
    local_search's distinct columns: non-seeds in no uncovered RR set and no uniquely covered set all give the same
    column, so one stands in for them and they share its value."""
    ix, k = state.index, state.k
    add, drop = np.zeros(ix.n), np.zeros(k)
    if k == 0 or k >= ix.n:
        return add, drop
    counts = _counts(state) if counts is None else counts
    _, Gc, (_, ov, _) = counts
    idle = None
    if cand is None:
        cols = _candidates(state, Gc, ov)
        idle = ~state.is_seed                                # the idle nodes of local_search._candidates
        idle[np.flatnonzero(Gc.any(axis=1))] = False
        idle[np.unique(ov)] = False
    else:
        cols = _columns(state, cand)
    col_best, row_best = np.full(len(cols), -np.inf), np.full(k, -np.inf)
    for a0, a1, b0, b1, imp in _blocks(state, floor, np.arange(k, dtype=np.int64), cols, counts, device,
                                       max_elems):
        col_best[b0:b1] = np.maximum(col_best[b0:b1], imp.amax(0).cpu().numpy())
        row_best[a0:a1] = np.maximum(row_best[a0:a1], imp.amax(1).cpu().numpy())
    add[cols] = np.where(col_best > TOL, col_best, 0.0)
    if idle is not None and idle.any():
        add[idle] = add[int(np.argmax(idle))]                # _candidates keeps the first idle node
    drop = np.where(row_best > TOL, row_best, 0.0)
    return add, drop


# ------------------------------------------------------------------ imitation data

@torch.no_grad()
def state_features(ctx, state, rho, total_ref, mu):
    """(n, 14) float32 CPU dynamic features of a CoverState as the learned proposer sees it: round t = 2 of 3,
    w = objective.signed_prices(rates, present, sizes, mu)[0]."""
    w = signed_prices(state.rates, state.present, state.sizes, float(mu))[0]
    return dynamic_from_state(ctx, state, w, float(rho), float(total_ref), ROUNDS - 1, ROUNDS).float().cpu()


def _record(ctx, state, add_t, drop_t, floor, rho, total_ref, mu, ctx_id, start, step):
    n = state.index.n
    seeds = np.asarray(state.seeds, dtype=np.int64)
    drop_full = np.zeros(n, dtype=np.float32)
    drop_full[seeds] = drop_t
    rec = {"ctx": int(ctx_id), "k": int(state.k), "rho": float(rho), "floor": float(floor),
           "total_ref": float(total_ref), "mu": None if mu is None else float(mu), "seeds": seeds.tolist(),
           "disp": float(state.disp), "total": float(state.total), "start": int(start), "step": int(step),
           "add": torch.from_numpy(add_t.astype(np.float32)), "drop": torch.from_numpy(drop_full),
           "is_seed": torch.from_numpy(state.is_seed.copy()), "dyn": None}
    if mu is not None:
        rec["dyn"] = state_features(ctx, state, rho, total_ref, mu)
    return rec


def _random_improving(state, floor, drop_t, rng, device, counts, max_elems):
    """A uniformly drawn improving swap (u, v): a seed with an improving swap, then one of its improving
    candidates; checked with verify.accept (no family). None if the draw fails."""
    rows = np.flatnonzero(drop_t > 0)
    if rows.size == 0:
        return None
    i = int(rng.choice(rows))
    cols = _columns(state, None)
    row = np.full(len(cols), -np.inf)
    for _, _, b0, b1, imp in _blocks(state, floor, np.array([i], dtype=np.int64), cols, counts, device, max_elems):
        row[b0:b1] = imp[0].cpu().numpy()
    good = np.flatnonzero(row > TOL)
    if good.size == 0:
        return None
    u, v = int(state.seeds[i]), int(cols[int(rng.choice(good))])
    if not accept(evaluate_state(state), evaluate_swap(state, u, v), floor):
        return None
    return u, v


def generate_imitation_data(ctx, est, starts, floor, rho, total_ref, n_states, rng, mu=0.0, explore=0.25,
                            max_steps=None, device=None, ctx_id=0, max_elems=MAX_ELEMS):
    """Imitation states along exhaustive single-swap local-search trajectories: a list of records (dicts).

    From each start (a list of k node ids) the state moves by local_search.best_swap(state, floor), or, with
    probability `explore`, by a random improving swap (_random_improving), until no swap improves or the start's
    share of n_states is recorded. Each visited state at or above the floor with an improving swap becomes a
    record: ctx (= ctx_id), k, rho, floor, total_ref, mu, seeds, disp, total, start (index), step; add (n,) and
    drop (n,) float32 targets (improvement_targets; drop on the seed rows, 0 elsewhere); is_seed (n,) bool; dyn,
    the (n, 14) float32 CPU features of state_features with price mu, or None when mu is None (then call
    featurize_records, e.g. once per checkpoint price table).

    floor: the floor on est that defines an improving swap; acceptance_floor(rho, total_ref) matches improve().
    States below it are repaired by best_swap and not recorded. Starts are taken in order and each gets an equal
    share of the states still to record, so a trajectory that ends early leaves its share to the next ones.
    rng: numpy Generator (None: default_rng(0)). est: the CoverState estimator (None: ctx.est)."""
    est = ctx.est if est is None else est
    rng = np.random.default_rng(0) if rng is None else rng
    floor, n_states = float(floor), int(n_states)
    starts = [[int(v) for v in s] for s in starts]
    records = []
    for si, start in enumerate(starts):
        left = n_states - len(records)
        if left <= 0:
            break
        quota = -(-left // (len(starts) - si))
        cap = 20 * quota + 200 if max_steps is None else int(max_steps)
        state = CoverState(est, start)
        got = step = 0
        while got < quota and step < cap:
            move = None
            if state.total >= floor:
                counts = _counts(state)
                add_t, drop_t = improvement_targets(state, floor, device, counts=counts, max_elems=max_elems)
                if drop_t.max(initial=0.0) <= 0.0:
                    break                                    # single-swap local optimum
                records.append(_record(ctx, state, add_t, drop_t, floor, rho, total_ref, mu, ctx_id, si, step))
                got += 1
                if got >= quota:
                    break
                if explore > 0 and rng.random() < explore:
                    move = _random_improving(state, floor, drop_t, rng, device, counts, max_elems)
            if move is None:
                out = best_swap(state, floor, device=device)
                if out is None:
                    break
                move = (out[0], out[1])
            state.apply_swap(*move)
            step += 1
    return records


def featurize_records(ctx_list, records, mu):
    """(Re)compute rec["dyn"] and rec["mu"] of every record in place and return the records. mu: a price, a dict
    k -> price (a checkpoint's mu_ref) or a callable k -> price. ctx_list: one GraphContext or a list indexed by
    rec["ctx"]."""
    ctxs = ctx_list if isinstance(ctx_list, (list, tuple)) else [ctx_list]
    for rec in records:
        ctx = ctxs[int(rec.get("ctx", 0))]
        m = _mu_of(mu, rec["k"])
        rec["dyn"] = state_features(ctx, CoverState(ctx.est, rec["seeds"]), rec["rho"], rec["total_ref"], m)
        rec["mu"] = m
    return records


# ------------------------------------------------------------------ loss and training

def _listwise(scores, target, pool, temp):
    """KL(softmax(t / (temp max t)) || softmax(s)) over pool; None if the pool has < 2 entries or no positive t."""
    s, t = scores[pool], target[pool].to(scores.dtype)
    if t.numel() < 2:
        return None
    top = float(t.max())
    if not top > 0.0:
        return None
    q = torch.softmax(t / (float(temp) * top), 0)
    return (torch.xlogy(q, q) - q * torch.log_softmax(s, 0)).sum()


def proposer_loss(add_scores, drop_scores, targets, is_seed, temp=0.1):
    """Listwise imitation loss: softmax cross-entropy of the add scores over non-seeds toward softmax(t_add / temp'),
    plus the same for the drop scores over seeds (higher = better to remove), with temp' = temp * the largest
    target of that pool. Scaling by the maximum makes temp relative: improvements range over orders of magnitude
    between states and graphs. Written as a KL (cross-entropy minus the target's entropy, a constant), so the
    value is 0 at a perfect fit and the gradients are those of the cross-entropy. targets: (add (n,), drop (n,))
    or a dict with "add" and "drop" (improvement_targets values; drop read on seeds). A pool whose targets are
    all 0 adds nothing; if both do, the result is a zero that keeps the autograd graph."""
    add_t, drop_t = (targets["add"], targets["drop"]) if isinstance(targets, dict) else targets
    dev = add_scores.device
    seed = torch.as_tensor(is_seed, device=dev).bool()
    add_t, drop_t = torch.as_tensor(add_t, device=dev), torch.as_tensor(drop_t, device=dev)
    loss = add_scores.sum() * 0.0 + drop_scores.sum() * 0.0
    for scores, t, pool in ((add_scores, add_t, ~seed), (drop_scores, drop_t, seed)):
        term = _listwise(scores, t, pool, temp)
        if term is not None:
            loss = loss + term
    return loss


def train_proposer(policy, ctx_list, data, epochs, lr, device, freeze_backbone=False, temp=0.1, weight_decay=1e-4,
                   clip=1.0, seed=0, val_fn=None, eval_every=1, checkpoint=None, verbose=True, on_the_fly=False):
    """Fine-tune a PolicyV2 on featurised imitation records; returns the log (one dict per evaluation).

    One AdamW step per record, records shuffled every epoch (np.random.default_rng(seed)); gradients clipped at
    `clip`. freeze_backbone trains only the add and drop heads, with the frozen encoder in eval mode (the heads have
    no dropout). val_fn(policy) -> float, higher is better (e.g. mean validation recall_at_c): evaluated before
    training, every eval_every epochs and after the last, and the best state is restored (the untouched policy
    can win). checkpoint: sets policy.checkpoint (activation checkpointing) when not None. Records need rec["dyn"]
    (featurize_records), or on_the_fly=True, which computes each record's features at its step (state_features with
    the record's price, 0 when it has none) instead of keeping an (n, 14) block per record in memory; each ctx's
    edges must be on ctx.device, as V2Trainer."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ctxs = list(ctx_list) if isinstance(ctx_list, (list, tuple)) else [ctx_list]
    for c in ctxs:
        if c.edge_device != c.device:
            raise RuntimeError("training needs ctx's edge list on ctx.device (chunked mode is inference-only)")
        if c.device.type != dev.type:
            raise ValueError(f"ctx lives on {c.device}, training runs on {dev}")
    data = list(data)
    if not on_the_fly and any(r.get("dyn") is None for r in data):
        raise ValueError("records without features: call featurize_records first, or pass on_the_fly=True")
    policy.to(dev)
    if checkpoint is not None:
        policy.checkpoint = bool(checkpoint)
    heads = set(id(p) for h in (policy.add_head, policy.drop_head) for p in h.parameters())
    for p in policy.parameters():
        p.requires_grad_(not freeze_backbone or id(p) in heads)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    log, best, best_state, t0 = [], -math.inf, None, time.time()

    def evaluate(ep, stats):
        nonlocal best, best_state
        v = float(val_fn(policy)) if val_fn is not None else float("nan")
        log.append({"epoch": ep, "val": v, **stats, "secs": time.time() - t0})
        if val_fn is not None and (best_state is None or v > best):
            best, best_state = v, copy.deepcopy(policy.state_dict())
        if verbose:
            print(f"    ep {ep:3d}  loss={stats['loss']:.4f} ({stats['steps']} steps)  val={v:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    try:
        evaluate(0, {"loss": float("nan"), "steps": 0})
        for ep in range(1, int(epochs) + 1):
            policy.train(not freeze_backbone)
            losses = []
            for i in rng.permutation(len(data)):
                rec = data[int(i)]
                ctx = ctxs[int(rec.get("ctx", 0))]
                dyn = rec.get("dyn")
                if dyn is None:                          # on_the_fly: the record's features, not kept
                    dyn = state_features(ctx, CoverState(ctx.est, rec["seeds"]), rec["rho"], rec["total_ref"],
                                         rec.get("mu") or 0.0)
                feats = torch.cat([ctx.static, dyn.to(device=ctx.device, dtype=ctx.static.dtype)], 1)
                add, drop = policy.score(ctx, feats)
                loss = proposer_loss(add, drop, (rec["add"], rec["drop"]), rec["is_seed"], temp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, clip)
                opt.step()
                losses.append(float(loss.detach()))
            stats = {"loss": float(np.mean(losses)) if losses else float("nan"), "steps": len(losses)}
            if ep % max(int(eval_every), 1) == 0 or ep == int(epochs):
                evaluate(ep, stats)
            elif verbose:
                print(f"    ep {ep:3d}  loss={stats['loss']:.4f} ({stats['steps']} steps)"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
        if best_state is not None:
            policy.load_state_dict(best_state)
    finally:
        for p in policy.parameters():
            p.requires_grad_(True)
        policy.eval()
    return log


# ------------------------------------------------------------------ recall of improving swaps

@torch.no_grad()
def policy_scores(policy, ctx, state, rho, total_ref, mu, dyn=None):
    """(add (n,), drop (n,)) float64 numpy scores of a PolicyV2 at a CoverState, in eval mode, from the features
    the learned proposer uses (state_features; dyn reuses precomputed ones)."""
    if dyn is None:
        dyn = state_features(ctx, state, rho, total_ref, mu)
    feats = torch.cat([ctx.static, dyn.to(device=ctx.static.device, dtype=ctx.static.dtype)], 1)
    was = policy.training
    policy.eval()
    try:
        add, drop = policy.score(ctx, feats)
    finally:
        policy.train(was)
    return _np(add), _np(drop)


def simple_scores(state, floor):
    """(add (n,), drop (n,)) of the simple proposer: gains() @ w on every node and -(removal_loss() @ w) on the
    seed rows (the cheapest removal ranks first), with the prices of improve()'s first round: a fresh PIController
    with improve()'s default setpoint and gains (config.PRICE_SETPOINT, PRICE_KP, PRICE_KI; design section 18)."""
    ctrl = PIController(kp=PRICE_KP, ki=PRICE_KI, sizes=state.sizes, setpoint=PRICE_SETPOINT)
    alpha, beta, mu = ctrl.update(state.rates, state.present, state.total, floor)
    w = price_weights(alpha, beta, mu, state.sizes, state.present)
    add = np.asarray(state.gains(), dtype=np.float64) @ w
    drop = np.zeros(state.index.n)
    drop[np.asarray(state.seeds, dtype=np.int64)] = -(np.asarray(state.removal_loss(), dtype=np.float64) @ w)
    return add, drop


def _first_improving_rank(state, floor, add_sc, drop_sc, cols, c_max, device, counts):
    """1-based rank of the first improving swap among the c_max best (seed u, candidate v) swaps by
    drop[u] + add[v], in LearnedProposer's order (proposer._top and _pairs); inf if none of them improves.
    The c_max best pairs by a sum always lie within the c_max best drops x the c_max best adds."""
    seeds = np.asarray(state.seeds, dtype=np.int64)
    pos = _top(drop_sc[seeds], np.arange(len(seeds), dtype=np.int64), c_max)      # positions in state.seeds
    adds = _top(add_sc, cols, c_max)                                               # node ids
    if len(pos) == 0 or len(adds) == 0:
        return math.inf
    imp = np.full((len(pos), len(adds)), -np.inf)
    for a0, a1, b0, b1, blk in _blocks(state, floor, pos, adds, counts, device, MAX_ELEMS):
        imp[a0:a1, b0:b1] = blk.cpu().numpy()
    drops = seeds[pos]
    row = {int(u): i for i, u in enumerate(drops)}
    col = {int(v): j for j, v in enumerate(adds)}
    for rank, (u, v) in enumerate(_pairs(drops, adds, drop_sc[drops], add_sc[adds])[:c_max], 1):
        if imp[row[u], col[v]] > TOL:
            return rank
    return math.inf


def recall_at_c(policy, ctx, states, C, rho=None, total_ref=None, floor=None, mu=None, cand=None, rng=None,
                device=None, return_info=False):
    """Share of improvable states whose top-C proposed swaps contain an improving swap.

    policy: a PolicyV2, scored as the learned proposer (policy_scores), or "simple" (simple_scores). The top-C
    proposals are the C swaps (seed u, candidate v) with the largest drop[u] + add[v], in LearnedProposer's order.
    A state counts only if some swap improves (labelled disparity down by more than 1e-12, RR total at or above
    the floor); the others are single-swap local optima and are left out of the share.
    states: CoverStates, or imitation records (dicts with seeds, k, rho, total_ref, floor, mu and optionally ctx,
    dyn and add / drop targets, which are reused when the arguments below are left at None). ctx: a GraphContext,
    or a list indexed by record["ctx"]. rho None: the record's, else RHO_DEFAULT. total_ref None: the record's,
    else ctx.total_ref[k]. floor None: the record's, else acceptance_floor(rho, total_ref). mu None: the record's,
    else 0. cand: None (every non-seed), an int (that many random non-seeds per state, from rng; default
    default_rng(0)), or an array of ids; adds are ranked within cand.
    C: an int (returns a float) or a sequence (returns {C: share}); NaN when no state is improvable. return_info
    returns {"recall", "n_states", "n_improvable", "hits", "per_state"} instead."""
    ctxs = ctx if isinstance(ctx, (list, tuple)) else [ctx]
    Cs = [int(C)] if np.isscalar(C) else [int(c) for c in C]
    if not Cs or min(Cs) < 1:
        raise ValueError("C must be >= 1")
    rng = np.random.default_rng(0) if rng is None else rng
    simple = isinstance(policy, str)
    if simple and policy != "simple":
        raise ValueError(f"policy must be a PolicyV2 or 'simple', not {policy!r}")
    reuse = rho is None and total_ref is None and floor is None and mu is None
    hits, n_imp, per_state = {c: 0 for c in Cs}, 0, []
    for st in states:
        rec = st if isinstance(st, dict) else None
        cx = ctxs[int(rec.get("ctx", 0))] if rec is not None else ctxs[0]
        state = CoverState(cx.est, rec["seeds"]) if rec is not None else st
        get = (lambda key, dflt: rec[key] if rec is not None and rec.get(key) is not None else dflt)
        r = float(rho) if rho is not None else float(get("rho", RHO_DEFAULT))
        tr = float(total_ref) if total_ref is not None else float(get("total_ref", cx.total_ref[state.k]))
        fl = float(floor) if floor is not None else float(get("floor", acceptance_floor(r, tr)))
        m = float(mu) if mu is not None else float(get("mu", 0.0))
        if cand is None:
            cols = _columns(state, None)
        elif np.isscalar(cand):
            non = np.flatnonzero(~state.is_seed)
            cols = np.sort(rng.choice(non, min(int(cand), len(non)), replace=False)).astype(np.int64)
        else:
            cols = _columns(state, cand)
        counts = _counts(state)
        if reuse and cand is None and rec is not None and rec.get("drop") is not None:
            improvable = bool(float(torch.as_tensor(rec["drop"]).max()) > 0.0)
        else:
            _, drop_t = improvement_targets(state, fl, device, cand=None if cand is None else cols, counts=counts)
            improvable = bool(drop_t.max(initial=0.0) > 0.0)
        info = {"improvable": improvable, "first_rank": None, "hit": {c: False for c in Cs}}
        if improvable:
            n_imp += 1
            if simple:
                add_sc, drop_sc = simple_scores(state, fl)
            else:
                dyn = rec.get("dyn") if (reuse and rec is not None) else None
                add_sc, drop_sc = policy_scores(policy, cx, state, r, tr, m, dyn=dyn)
            rank = _first_improving_rank(state, fl, add_sc, drop_sc, cols, max(Cs), device, counts)
            info["first_rank"] = None if math.isinf(rank) else int(rank)
            for c in Cs:
                info["hit"][c] = bool(rank <= c)
                hits[c] += int(rank <= c)
        per_state.append(info)
    share = {c: (hits[c] / n_imp if n_imp else float("nan")) for c in Cs}
    if return_info:
        return {"recall": share, "n_states": len(per_state), "n_improvable": n_imp, "hits": hits,
                "per_state": per_state}
    return share[Cs[0]] if np.isscalar(C) else share
