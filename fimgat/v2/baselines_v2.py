"""v2 baselines: Frank-Wolfe on the disparity-under-floor Lagrangian, and the full baseline suite.

FW-Disparity  Frank-Wolfe with step 1/steps from x = 0 (continuous greedy, Tsang et al. 2019 style) on the
              multilinear extension of objective.disparity_lagrangian, then sample-and-select decoding: the
              top-k of x, the last FW vertex and Gumbel-top-k samples (the v1 decode recipe) are ranked on the
              RR sample, and the lowest-disparity candidate whose exact float64 RR total meets the floor wins,
              else the reference set. The relaxation certifies nothing (design_v2 section 3), so its quality
              comes from decoding; it is the non-learned ancestor of the v2 approach.
BSM, MultiFW-MaxMin, RFIM  external baselines written from their papers (fimgat/v2/external.py; design section 19).
run_suite     every baseline of docs/v2_interfaces.md section 17 on one prepared graph: timed, retried once
              after an out-of-memory error (as v1's suite runner did), with an optional time limit.
              A method that yields no valid seed set stays in the result with its reason; it is never dropped.
"""
import gc
import math
import time

import numpy as np
import torch
from numba import njit

from .. import baselines as B
from .config import DELTA_MARGIN
from .objective import disparity, disparity_lagrangian

SUITE = ("Degree", "Greedy", "Parity-Greedy", "Welfare-Greedy", "Maximin-Greedy", "FW-Disparity",
         "LocalSearch", "Simple-v2", "AGE", "Prior", "BSM", "MultiFW-MaxMin", "RFIM")
GREEDY = {"Greedy": "greedy", "Parity-Greedy": "parity", "Welfare-Greedy": "welfare", "Maximin-Greedy": "maximin"}
STARTS = ("Parity-Greedy", "Welfare-Greedy", "Greedy")        # starts of LocalSearch and Simple-v2
AGE_MAX_N, PRIOR_MAX_N = 5000, 100_000
PRESKIP = 3.0         # skip a method only when its rough estimate exceeds the time limit this many times
LOOSE = 1e-4          # float32 batch totals this far below the floor still get the exact float64 check
FLOOR_RTOL = 1e-12    # slack when re-checking a search output: CoverState updates its float64 totals
                      # incrementally, so its sums differ from a fresh one in the last few ulps
_OOM = (getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError), MemoryError)


# ------------------------------------------------------------------ exact evaluation and checks

@njit(cache=True)
def _root_cover(ptr, nodes, is_seed, R):
    """Covered RR sets per root (sets are root-major: set j belongs to root j // R)."""
    n_sets = ptr.shape[0] - 1
    cov = np.zeros(n_sets // R, dtype=np.int64)
    for j in range(n_sets):
        for t in range(ptr[j], ptr[j + 1]):
            if is_seed[nodes[t]]:
                cov[j // R] += 1
                break
    return cov


def _exact_eval(est, seeds):
    """Float64 group rates (G,) and RR total of a binary seed set, as defined in interfaces section 1."""
    is_seed = np.zeros(est.graph.n, dtype=np.bool_)
    is_seed[np.asarray(seeds, dtype=np.int64)] = True
    cov = _root_cover(est.ptr, est.nodes, is_seed, int(est.R))
    hits = np.bincount(est.root_groups_np, weights=cov.astype(np.float64), minlength=est.graph.n_groups)
    rates = hits / (est.R * np.maximum(est.root_counts, 1.0))
    return rates, float(rates @ est.graph.group_sizes)


def _present(est):
    return (est.graph.group_sizes > 0) & (est.root_counts > 0)


def check_seeds(seeds, k, n):
    """seeds as a list of ints; ValueError unless they are exactly k distinct node ids in [0, n)."""
    if torch.is_tensor(seeds):
        seeds = seeds.detach().cpu().numpy()
    s = np.asarray(seeds).ravel()
    if s.size and s.dtype.kind not in "iu" and (s.dtype.kind != "f" or not np.all(s == np.round(s))):
        raise ValueError("non-integer node ids")
    s = s.astype(np.int64)
    if len(s) != k:
        raise ValueError(f"{len(s)} seeds for budget {k}")
    if len(s) and (s.min() < 0 or s.max() >= n):
        raise ValueError(f"node ids outside [0, {n})")
    if len(np.unique(s)) != len(s):
        raise ValueError("duplicate seeds")
    return s.tolist()


# ------------------------------------------------------------------ FW-Disparity

@torch.no_grad()
def _candidates(scores, k, n_samples, temps, seed, extra=()):
    """Distinct candidate sets (int64 arrays): top-k of the z-scored scores, top-k of each extra vector,
    and Gumbel-top-k samples at several temperatures."""
    s = (scores - scores.mean()) / (scores.std() + 1e-6)
    cands = [torch.topk(s, k).indices] + [torch.topk(e, k).indices for e in extra]
    gen = torch.Generator(device=s.device).manual_seed(seed)
    for i in range(n_samples):
        u = torch.rand(s.shape[0], generator=gen, device=s.device).clamp(1e-9, 1 - 1e-9)
        cands.append(torch.topk(s / temps[i % len(temps)] - torch.log(-torch.log(u)), k).indices)
    out, seen = [], set()
    for c in cands:
        idx = c.cpu().numpy().astype(np.int64)
        key = np.sort(idx).tobytes()
        if key not in seen:
            seen.add(key)
            out.append(idx)
    return out


@torch.no_grad()
def _batch_scores(est, cands, chunk=16):
    """Float32 disparity and RR total of every candidate, used for ranking only."""
    n, dev, present = est.graph.n, est.device, est.present
    disp, tot = [], []
    for c0 in range(0, len(cands), chunk):
        part = cands[c0:c0 + chunk]
        X = torch.zeros(len(part), n, device=dev)
        for i, idx in enumerate(part):
            X[i, torch.as_tensor(idx, device=dev)] = 1.0
        rates, total = est.batch_exposure_rates(X)
        r = rates[:, present]
        d = r.max(1).values - r.min(1).values if r.shape[1] >= 2 else torch.zeros(len(part), device=dev)
        disp.append(d.double().cpu())
        tot.append(total.double().cpu())
    return torch.cat(disp).numpy(), torch.cat(tot).numpy()


def _select(est, cands, k, floor, reference, n_exact):
    """Lowest exact disparity among candidates whose exact RR total meets the floor (ties: larger total);
    the n_exact best by float32 ranking are checked. Else the reference (RR-greedy if None)."""
    disp32, tot32 = _batch_scores(est, cands)
    present = _present(est)
    best, checked, n_ok = None, 0, 0
    for i in np.lexsort((-tot32, disp32)):
        # UNVERIFIED: LOOSE assumes float32 batch totals are within 1e-4 (relative) of the float64 ones
        if tot32[i] < floor * (1.0 - LOOSE):
            continue
        rates, total = _exact_eval(est, cands[i])
        checked += 1
        if total < floor:
            continue
        d = float(disparity(rates, present))
        if best is None or (d, -total) < (best[0], -best[1]):
            best = (d, total, i)
        n_ok += 1
        if n_ok >= n_exact:
            break
    info = {"n_candidates": len(cands), "n_exact": checked, "fallback": best is None}
    if best is None:
        return (B.RRGreedy(est).greedy(k) if reference is None else [int(v) for v in reference]), info
    return cands[best[2]].tolist(), info


def fw_disparity(est, k, rho, total_ref, steps=100, tau=0.02, penalty=10.0, mu=0.0, floor=None, reference=None,
                 n_samples=64, temps=(0.05, 0.1, 0.2, 0.4), seed=0, n_exact=8, return_info=False):
    """Frank-Wolfe (step 1/steps from x = 0) on the multilinear disparity_lagrangian, then decoding.

    total_ref: RR total of the reference set on est. floor: the acceptance floor, by default
    (rho + DELTA_MARGIN) * total_ref; the Lagrangian targets floor / total_ref, so the margin is included.
    mu: fixed spread price (the quadratic penalty does the rest). reference: returned when no candidate
    meets the floor (RR-greedy if None). Returns list[int], or (seeds, info) with return_info."""
    n, k = est.graph.n, int(k)
    if not 1 <= k <= n:
        raise ValueError(f"budget k={k} outside [1, n={n}]")
    steps = int(steps)
    total_ref = float(total_ref)
    floor = (rho + DELTA_MARGIN) * total_ref if floor is None else float(floor)
    rho_eff = floor / max(total_ref, 1e-12)
    x = torch.zeros(n, device=est.device)
    grad = torch.zeros_like(x)
    with torch.enable_grad():                          # callers may sit inside torch.no_grad()
        for _ in range(steps):
            xv = x.clone().requires_grad_(True)
            _, rates, total, _ = est(xv)
            L = disparity_lagrangian(rates, total, total_ref, rho_eff, est.present, tau, mu, penalty)
            if not getattr(L, "requires_grad", False):     # objective constant in x: nothing to follow
                break
            grad, = torch.autograd.grad(L, xv)
            v = torch.zeros_like(x)
            v[torch.topk(-grad, k).indices] = 1.0          # linear minimiser over {v in [0,1]^n, sum v = k}
            x = x + v / steps
    # final mass, ties broken by the last descent direction; the last FW vertex is also a candidate
    key = x - 1e-3 * (grad / (grad.abs().max() + 1e-12)) / max(steps, 1)
    cands = _candidates(key, k, n_samples, temps, seed, extra=[-grad])
    seeds, info = _select(est, cands, k, floor, reference, n_exact)
    return (seeds, info) if return_info else seeds


# ------------------------------------------------------------------ suite plumbing

def _is_oom(e):
    """Allocator OOM, the 'CUDA error: out of memory' RuntimeError, or a host MemoryError."""
    return isinstance(e, _OOM) or "out of memory" in str(e)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _free():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:                           # a broken context: the retry will report it
            pass


def _attempt(fn):
    """(fn(), secs, None). After an out-of-memory error, one retry (as in v1); after a second one,
    (None, secs of both attempts, reason). secs of a success counts only the attempt that succeeded,
    as in v1. Any other error propagates."""
    spent, reason = 0.0, None
    for _ in range(2):
        t0 = None
        try:
            _sync()                                    # an earlier async CUDA error surfaces here, inside the try
            t0 = time.perf_counter()
            out = fn()
            _sync()
            return out, time.perf_counter() - t0, None
        except (RuntimeError, MemoryError) as e:
            if not _is_oom(e):
                raise
            reason = "host out of memory" if isinstance(e, MemoryError) else "CUDA out of memory"
        if t0 is not None:
            spent += time.perf_counter() - t0
        _free()                                        # outside the except block: the traceback is released
    # UNVERIFIED: assumes a CUDA OOM leaves the context usable for the retry and for later methods
    return None, spent, reason


def _estimate_secs(name, g, est, k, device):
    """Rough wall-clock guess, used only to skip a method far beyond a time limit (None: no guess)."""
    # UNVERIFIED: constants scaled by hand from the v1 timings in docs/research_protocol.md section 6
    # (8 GB laptop) and never calibrated; hence the PRESKIP slack.
    gpu = torch.device(device).type == "cuda"
    if name in GREEDY:
        return k * 3e-9 * (est.nnz + g.n * g.n_groups)            # non-lazy greedy: O(nnz + n*G) per seed
    if name == "AGE":
        return (300 * math.ceil(g.n / 50) + 1000 * max(1, g.n // 100)) * (3e-3 if gpu else 3e-2)
    if name == "Prior":
        return 1000 * (0.01 + 4e-7 * g.m) * (1.0 if gpu else 20.0)
    return None


class SuiteResult(dict):
    """name -> (seeds, secs). seeds is None when the method produced no valid set, and .reason[name]
    says why. .info[name] holds per-method extras (start used, fallback to the reference, index time)."""

    def __init__(self):
        super().__init__()
        self.reason, self.info = {}, {}


def run_suite(ctx, k, rho, floor, reference, methods=None, time_limit=None, cache=None):
    """Run the v2 baseline suite on one prepared graph (ctx from fimgat.v2.train.prepare_v2, or v1
    train.prepare: only ctx.graph, ctx.est and ctx.device are used).

    floor: exact RR floor, (rho + DELTA_MARGIN) * RR total of the reference. reference: the reference
    seeds (reference_spread()["seeds"], or that dict). methods: a subset of SUITE (default all).
    time_limit: seconds per method. A method is skipped if its rough estimate exceeds PRESKIP times
    the limit, and recorded missing if it takes longer than the limit. LocalSearch and Simple-v2 get
    the remaining budget as max_seconds, plus a grace of max(1 s, 10%) for their check granularity.
    cache: a dict kept by the caller for this ctx; methods that ignore rho and the floor (Degree, the
    greedy family, AGE, Prior) are then computed once and reused across calls with other rho.
    Per-graph index structures (the RR-greedy index, the verifier's cover index) are built outside the
    per-method timings, like the RR sample; their times are reported in info. The time of LocalSearch
    and Simple-v2 includes computing and checking their greedy starts. Returns a SuiteResult."""
    g, est = ctx.graph, ctx.est
    n, k, floor = g.n, int(k), float(floor)
    if not 1 <= k <= n:
        raise ValueError(f"budget k={k} outside [1, n={n}]")
    names = list(SUITE) if methods is None else list(dict.fromkeys(methods))
    unknown = [m for m in names if m not in SUITE]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; the suite is {SUITE}")
    if isinstance(reference, dict):
        reference = reference["seeds"]
    reference = check_seeds(reference, k, n)
    present = _present(est)
    total_ref = max(_exact_eval(est, reference)[1], 1e-12)       # also compiles the kernel before any timing
    cache = {} if cache is None else cache
    if cache.setdefault("_est", est) is not est:
        raise ValueError("this cache belongs to another estimator; keep one cache per ctx")
    res, shared, memo = SuiteResult(), {}, {}

    def record(name, seeds, secs, reason=None, info=None, anytime=False):
        if reason is None:
            try:
                seeds = check_seeds(seeds, k, n)
            except ValueError as e:
                reason = f"invalid seed set ({e})"
        grace = (max(1.0, 0.1 * time_limit) if anytime else 0.0) if time_limit is not None else 0.0
        if reason is None and time_limit is not None and secs > time_limit + grace:
            reason = f"time_limit: took {secs:.1f}s > {time_limit:g}s"
        res[name] = (None if reason else list(seeds), float(secs))
        if reason:
            res.reason[name] = reason
        res.info[name] = dict(info or {})

    def preskip(name):
        if time_limit is None:
            return None
        guess = _estimate_secs(name, g, est, k, ctx.device)
        if guess is not None and guess > PRESKIP * time_limit:
            return f"skipped: estimated {guess:.0f}s > {PRESKIP:g} x time_limit {time_limit:g}s"
        return None

    def rho_free(name, fn):
        """(seeds, secs, reason) of a method independent of rho and the floor, cached per k."""
        key = (k, name)
        if key not in cache:
            why = preskip(name)
            if why:
                return None, 0.0, why
            cache[key] = _attempt(fn)
        return cache[key]

    def index():
        if "index" not in cache:
            cache["index"] = _attempt(lambda: B.RRGreedy(est))
        rg, secs, why = cache["index"]
        shared["index_s"] = secs
        return rg, why

    def start(name):
        rg, why = index()
        if rg is None:
            return None, 0.0, f"RR index unavailable ({why})"
        return rho_free(name, lambda: getattr(rg, GREEDY[name])(k))

    def search_starts():
        """([(name, seeds, total, disp)], secs, notes): greedy starts with their exact floor check."""
        if "starts" in memo:
            return memo["starts"]
        from .cover import CoverState
        _, shared["cover_s"], why = _attempt(lambda: CoverState(est, list(reference)))   # builds est._v2_cover
        if why:
            memo["starts"] = ([], 0.0, {"cover index": why})
            return memo["starts"]
        got, secs, notes = [], 0.0, {}
        for nm in STARTS:
            seeds, s, why = start(nm)
            secs += s
            if why is None:
                try:
                    seeds = check_seeds(seeds, k, n)
                except ValueError as e:
                    why = f"invalid seed set ({e})"
            if why is not None:
                notes[nm] = why
                continue
            t0 = time.perf_counter()
            rates, total = _exact_eval(est, seeds)
            secs += time.perf_counter() - t0
            got.append((nm, seeds, total, float(disparity(rates, present))))
        memo["starts"] = (got, secs, notes)
        return memo["starts"]

    def search(name, runner):
        got, pre, notes = search_starts()
        info = {key: shared[key] for key in ("index_s", "cover_s") if key in shared}
        if notes:
            info["start_notes"] = notes
        if not got:
            return record(name, None, pre, "no start available: " + "; ".join(f"{a}: {b}" for a, b in notes.items()),
                          info)
        if time_limit is not None and pre >= time_limit:
            return record(name, None, pre, f"time_limit: starts took {pre:.1f}s >= {time_limit:g}s", info)
        budget = None if time_limit is None else time_limit - pre
        out, secs, why = _attempt(lambda: runner(got, budget))
        seeds, extra = out if out is not None else (None, {})
        record(name, seeds, pre + secs, why, {**info, **extra}, anytime=True)

    def checked(seeds):
        """(seeds, meets_floor, total, disp) of a search output, or None if it is not a valid set of k nodes."""
        try:
            seeds = check_seeds(seeds, k, n)
        except ValueError:
            return None
        rates, total = _exact_eval(est, seeds)
        return seeds, bool(total >= floor - FLOOR_RTOL * abs(floor)), total, float(disparity(rates, present))

    def local(got, budget):
        """Single-swap local search from every floor-meeting start (else the reference); best result."""
        from .local_search import local_search
        t0 = time.perf_counter()
        feas = [(nm, s) for nm, s, total, _ in got if total >= floor] or [("reference", reference)]
        best, bad = None, 0
        for nm, s in feas:
            left = None if budget is None else max(0.0, budget - (time.perf_counter() - t0))
            if left == 0.0 and best is not None:
                break
            # copies: the search may keep and mutate the list it is given, and starts are shared
            got_out = checked(local_search(est, list(s), floor, max_seconds=left))
            if got_out is None:
                bad += 1
                continue
            out, ok, total, d = got_out
            if ok and (best is None or (d, -total) < (best[0], -best[1])):
                best = (d, total, out, nm)
        extra = {"n_starts": len(feas), "n_invalid": bad}
        if best is None:                  # nothing meets the floor, not even the reference
            return list(reference), {"start": "reference", "fallback": True, **extra}
        return best[2], {"start": best[3], "fallback": best[3] == "reference", **extra}

    def simple(got, budget):
        """improve(proposer="simple") from the best floor-meeting start (else the reference). An output
        that is invalid or misses the floor is replaced by the reference and flagged."""
        from .improve import improve
        feas = [(d, -total, i) for i, (_, _, total, d) in enumerate(got) if total >= floor]
        nm, s = got[min(feas)[2]][:2] if feas else ("reference", reference)
        seeds, hist = improve(est, list(s), floor, proposer="simple", max_seconds=budget, reference=list(reference))
        fallback = nm == "reference" or bool(hist and hist[0].get("fallback", False))
        extra = {"start": nm, "rounds": len(hist)}
        got_out = checked(seeds)
        if got_out is None or not got_out[1]:
            # UNVERIFIED: improve() should never return this (contract section 10) unless even the
            # reference misses the floor (rho + DELTA_MARGIN > 1); the reference is the honest fallback
            return list(reference), {**extra, "fallback": True, "output_rejected": True}
        return got_out[0], {**extra, "fallback": fallback}

    for name in names:
        if name == "Degree":
            record(name, *rho_free(name, lambda: B.degree(g, k)))
        elif name in GREEDY:
            seeds, secs, why = start(name)
            record(name, seeds, secs, why, {"index_s": shared.get("index_s", 0.0)})
        elif name == "FW-Disparity":
            out, secs, why = _attempt(lambda: fw_disparity(est, k, rho, total_ref, floor=floor,
                                                           reference=list(reference), return_info=True))
            seeds, info = out if out is not None else (None, {})
            record(name, seeds, secs, why, info)
        elif name == "LocalSearch":
            search(name, local)
        elif name == "Simple-v2":
            search(name, simple)
        elif name == "BSM":
            from .external import bsm
            rg, why0 = index()
            if rg is None:
                record(name, None, 0.0, f"RR index unavailable ({why0})")
                continue
            out, secs, why = _attempt(lambda: bsm(est, k, floor, reference=list(reference), rg=rg, cache=cache,
                                                  return_info=True))
            seeds, info = out if out is not None else (None, {})
            record(name, seeds, secs, why, {**info, "index_s": shared.get("index_s", 0.0)})
        elif name == "MultiFW-MaxMin":
            from .external import multifw_maxmin
            rg, why0 = index()
            if rg is None:
                record(name, None, 0.0, f"RR index unavailable ({why0})")
                continue
            record(name, *rho_free(name, lambda: multifw_maxmin(est, k, rg=rg)))
        elif name == "RFIM":
            from .external import rfim
            rg, why0 = index()
            if rg is None:
                record(name, None, 0.0, f"RR index unavailable ({why0})")
                continue
            record(name, *rho_free(name, lambda: rfim(est, k, rg=rg)))
        elif name == "AGE":
            if n > AGE_MAX_N:
                record(name, None, 0.0, f"not run: n={n} > {AGE_MAX_N}")
            else:
                record(name, *rho_free(name, lambda: B.age(g, k, device=ctx.device)))
        else:                                                        # "Prior"
            if n > PRIOR_MAX_N:
                record(name, None, 0.0, f"not run: n={n} > {PRIOR_MAX_N}")
            else:
                record(name, *rho_free(name, lambda: B.prior_repadv_gat(g, k, device=ctx.device)))
    return res
