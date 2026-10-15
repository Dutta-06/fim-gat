"""Exhaustive single-swap local search on the exact RR verifier (docs/v2_interfaces.md section 9).

best_swap scores every swap (seed u out, non-seed v in) at once from three exact pieces of a
CoverState: the removal loss L_u (sets only u covers), the gains G_v (uncovered sets containing v)
and a sparse overlap O_uv (sets only u covers that also contain v, which stay covered). Candidate
rates are rates - L_u + G_v + O_uv, and totals are formed the same way. They are evaluated in float64
torch blocks of bounded size, chunked over seeds and candidates. The best few candidates are then
re-scored exactly with verify.evaluate_swap and must pass verify.accept, so every returned swap is a
verified improvement.

Non-seeds that lie in no uncovered set and in no set covered by a single seed all give the same
candidate for every u (rates - L_u), so one of them stands in for all. The scan therefore grows with
the nodes the RR sample can tell apart, not with n.

This is the required local-search baseline and the "exhaustive" proposer of improve().
"""
import time

import numpy as np
import torch

from .cover import CoverState

TOL = 1e-12          # improvement threshold (as verify.accept)
_SLACK = 1e-9        # float64 pre-filter slack; the exact re-scoring decides


class _Pool:
    """The m best candidates by (disparity, -total), deduplicated on (seed index, column)."""

    def __init__(self, m):
        self.m = int(m)
        self.d, self.t = np.zeros(0), np.zeros(0)
        self.u, self.c = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    def __len__(self):
        return len(self.d)

    def add(self, d, t, u, c):
        d, t = np.r_[self.d, d], np.r_[self.t, t]
        u, c = np.r_[self.u, u].astype(np.int64), np.r_[self.c, c].astype(np.int64)
        order = np.lexsort((-t, d))
        _, first = np.unique(np.stack([u[order], c[order]], 1), axis=0, return_index=True)
        keep = order[np.sort(first)][:self.m]
        self.d, self.t, self.u, self.c = d[keep], t[keep], u[keep], c[keep]


def _candidates(state, gain_counts, ov):
    """Candidate columns: non-seeds with a gain, non-seeds in some seed's uniquely covered sets, and
    one idle non-seed standing in for all the others."""
    act = np.flatnonzero(gain_counts.any(axis=1))            # seed rows are 0
    o_nodes = np.unique(ov)
    idle = ~state.is_seed
    idle[act] = False
    idle[o_nodes] = False
    parts = [act.astype(np.int64), o_nodes.astype(np.int64)]
    if idle.any():
        parts.append(np.array([int(np.argmax(idle))], dtype=np.int64))
    return np.unique(np.concatenate(parts))


def best_swap(state, floor, family=None, device=None, *, n_check=32, max_elems=1 << 23, deadline=None,
              accept_kw=None, exclude=None):
    """Exact best-improvement single swap: (u, v, Evaluation) or None.

    Feasible state (total >= floor): among swaps that keep total >= floor, the lowest labelled
    disparity wins (ties: larger total). With a family, candidates are ranked by labelled disparity
    and the best n_check are re-scored with the community disparity; the best accepted one wins
    (lowest disp, then comm_disp, then larger total). Every returned swap passes verify.accept,
    so it improves by more than 1e-12; None when nothing does.
    Below the floor the state is repaired first: the lowest-disparity swap that reaches the floor,
    else the swap that raises the total most (None if no swap raises it).
    deadline (a time.perf_counter() value) stops the scan early, keeping the candidates seen so far.
    accept_kw: extra keyword arguments for verify.accept (min_gain, min_gain_comm, noise); with a
    noise model, the re-scoring carries MoveDetail. exclude(u, v) -> bool skips swaps (tabu), checked
    when the best n_check candidates are re-scored.
    """
    from .verify import accept, evaluate_state, evaluate_swap
    accept_kw = dict(accept_kw or {})
    noisy = accept_kw.get("noise") is not None and accept_kw["noise"].z > 0
    if noisy:
        # the plain-best candidates are the ones most likely to be near-tie noise; re-score more of them
        n_check = max(int(n_check), 128)

    ix = state.index
    seeds = np.asarray(state.seeds, dtype=np.int64)
    k = len(seeds)
    if k == 0 or k >= ix.n:
        return None
    floor = float(floor)
    before = evaluate_state(state, family, detail=noisy)
    repair = before.total < floor
    P = np.flatnonzero(state.present)
    Gp = len(P)
    if not repair and family is None and Gp < 2:
        return None                                          # disparity is identically 0

    Lc = state.loss_counts()                                 # (k, G)
    Gc = state.gain_counts()                                 # (n, G)
    oi, ov, oc = state.overlap_counts()                      # (E,), (E,), (E, G)
    cand = _candidates(state, Gc, ov)
    nc = len(cand)
    if nc == 0:
        return None
    col_of = np.full(ix.n, -1, dtype=np.int64)
    col_of[cand] = np.arange(nc, dtype=np.int64)
    gw, sizes = ix.gw, state.sizes
    L, Gv, O = Lc * gw, Gc[cand] * gw, oc * gw
    del Gc

    # UNVERIFIED: float64 on consumer CUDA cards runs at a small fraction of float32 speed; not timed
    dev = torch.device(device) if device is not None else torch.device(getattr(state.est, "device", "cpu"))

    def t64(a):
        return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64), dtype=torch.float64, device=dev)

    Lt, Gt, Ot = t64(L @ sizes), t64(Gv @ sizes), t64(O @ sizes)
    base = Lp = Gpv = Op = blk = None                        # group blocks, only with >= 2 present groups
    if Gp >= 2:
        base, Lp, Gpv, Op = t64(state.rates[P]), t64(L[:, P]), t64(Gv[:, P]), t64(O[:, P])
    oi_t = torch.as_tensor(oi, device=dev)
    ocol_t = torch.as_tensor(col_of[ov], device=dev)
    obounds = np.searchsorted(oi, np.arange(k + 1))          # entries of seed i: [obounds[i], obounds[i+1])

    floor_lo = floor - _SLACK * max(1.0, abs(floor))
    if repair:
        d_hi = np.inf
    elif family is None:
        d_hi = before.disp - TOL + _SLACK                    # must improve D
    else:
        d_hi = before.disp + TOL + _SLACK                    # D may stay equal if D_C improves
    width = max(Gp, 1)
    cv = int(min(nc, max(1, max_elems // width)))
    cu = int(min(k, max(1, max_elems // (cv * width))))

    pool, far = _Pool(n_check), None                         # far: (total, seed index, column) in repair
    inf = float("inf")
    done = False
    for u0 in range(0, k, cu):
        u1 = min(k, u0 + cu)
        e0, e1 = int(obounds[u0]), int(obounds[u1])
        for v0 in range(0, nc, cv):
            if deadline is not None and time.perf_counter() > deadline:
                done = True
                break
            v1 = min(nc, v0 + cv)
            bv = v1 - v0
            tot = before.total - Lt[u0:u1, None] + Gt[None, v0:v1]              # (bu, bv)
            if Gp >= 2:
                blk = base - Lp[u0:u1, None, :] + Gpv[None, v0:v1, :]            # (bu, bv, Gp)
            if e1 > e0:
                cols = ocol_t[e0:e1]
                sel = torch.nonzero((cols >= v0) & (cols < v1)).flatten()
                if sel.numel():
                    e = sel + e0
                    flat = (oi_t[e] - u0) * bv + (ocol_t[e] - v0)
                    tot.view(-1).index_add_(0, flat, Ot[e])
                    if Gp >= 2:
                        blk.view(-1, Gp).index_add_(0, flat, Op[e])
            disp = blk.amax(-1) - blk.amin(-1) if Gp >= 2 else torch.zeros_like(tot)
            tot_f, disp_f = tot.reshape(-1), disp.reshape(-1)
            key = disp_f.masked_fill(~((tot_f >= floor_lo) & (disp_f <= d_hi)), inf)
            vals, idx = torch.topk(key, min(pool.m, key.numel()), largest=False)
            idx = idx[torch.isfinite(vals)]
            if idx.numel():
                tie = tot_f.masked_fill(key != vals[0], -inf).argmax()          # min key, then max total
                idx = torch.cat([idx, tie.view(1)])
                ij = idx.cpu().numpy()
                pool.add(disp_f[idx].cpu().numpy(), tot_f[idx].cpu().numpy(), ij // bv + u0, ij % bv + v0)
            if repair:
                j = int(torch.argmax(tot_f))
                tj = float(tot_f[j])
                if far is None or tj > far[0]:
                    far = (tj, j // bv + u0, j % bv + v0)
        if done:
            break

    def rescore(ui, col):
        u, v = int(seeds[ui]), int(cand[col])
        return u, v, evaluate_swap(state, u, v, family, detail=noisy)

    best = None
    for ui, col in zip(pool.u, pool.c):                      # in (disparity, -total) order
        if exclude is not None and exclude(int(seeds[ui]), int(cand[col])):
            continue
        u, v, ev = rescore(ui, col)
        ok = ev.total >= floor if repair else accept(before, ev, floor, **accept_kw)
        if ok and (best is None or (ev.disp, ev.comm_disp, -ev.total) < (best[2].disp, best[2].comm_disp,
                                                                        -best[2].total)):
            best = (u, v, ev)
    if best is None and repair and far is not None:
        u, v, ev = rescore(far[1], far[2])
        if ev.total > before.total + TOL and not (exclude is not None and exclude(u, v)):
            best = (u, v, ev)
    return best


def local_search(est, start, floor, communities=None, max_passes=1000, max_seconds=None, device=None,
                 z=0.0, min_gain=0.0):
    """Best-improvement single-swap local search from start; returns the seed list.

    Repeats best_swap (with a CommunityFamily of `communities`, if given) until no swap improves by
    more than 1e-12, which is a single-swap local optimum, or until max_passes / max_seconds.
    The time limit is checked before each pass and inside the scan. z > 0 or min_gain > 0 apply the
    noise-aware acceptance of verify.accept (docs/theory_v2.md section 3); the defaults keep the plain rule.
    """
    from .verify import CommunityFamily, noise_model

    t0 = time.perf_counter()
    deadline = None if max_seconds is None else t0 + float(max_seconds)
    state = CoverState(est, [int(s) for s in start])
    family = CommunityFamily(est, communities) if communities is not None else None
    if family is not None and len(family) == 0:
        family = None
    kw = {"min_gain": float(min_gain)}
    if z and z > 0:
        kw["noise"] = noise_model(state, family, z)
    for _ in range(int(max_passes)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        out = best_swap(state, floor, family, device, deadline=deadline, accept_kw=kw)
        if out is None:
            break
        state.apply_swap(out[0], out[1])
    return list(state.seeds)
