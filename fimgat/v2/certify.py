"""Certified deployment on an independent RR sample (docs/theory_v2.md section 4).

The search (improve, local search) picks seed sets on RR sample A, so A's numbers for those sets are
optimistic (winner's curse). A second RR sample B, drawn independently and used for nothing else,
gives confidence bounds that stay valid after that selection:

floor_lcb       lower bound on T = sigma(S) - rho * sigma(S_ref). Paired per RR unit (common random
                numbers on B), by Bernstein's inequality with a data-driven variance bound (Maurer's
                inequality for sums of non-negative variables).
rate_bounds     lower and upper bounds on every group rate, by the Chernoff-Hoeffding KL inequality
                (Hoeffding 1963, Theorem 1), which holds for independent variables in [0, 1].
disparity_ucb   max_g U_g - min_g L_g over present groups.
certified_choice  every candidate whose floor bound is >= 0 is eligible, and so is the reference
                (T(S_ref) = (1 - rho) sigma(S_ref) >= 0 for rho <= 1). The deployed set is the eligible
                one with the smallest disparity upper bound. A union bound splits alpha over every
                statement, so with probability >= 1 - alpha the deployed set meets the floor and its
                true disparity is at most its upper bound.

Units. With every node a root (small and medium tiers), the RR sets are independent Bernoulli units
(N_g = R * roots in g). With sampled roots (large tier), the units are the per-root averages a_r in
[0, 1] (N_g = roots in g). That is valid for roots drawn without replacement, by Hoeffding's
comparison of sampling without and with replacement (1963, Theorem 4), but ignores the R-fold
variance reduction, so it is conservative.

The candidates and the reference must be fixed before B is looked at, and B must not be used to
choose them. Only the deployment decision may use B.

Deployment protocol (docs/design_v2.md section 18): path_states and deploy_candidates pick the candidates
from the search path; certified_equal_spread is test T1's equal-spread judgement for every method.

Deployment rules (docs/design_v2.md section 21, 2026-10-14): certify_sets computes every candidate's certificate once
and choose picks among the eligible ones, by Theorem 2's rule ("min_ucb", T1's deployment) or by the path rule
("path", T2's deployment: the furthest certified search state whose labelled-disparity bound is within
config.DEPLOY_TOL_SE standard errors of the smallest). Any rule that chooses among eligible sets keeps Theorem 2's
floor guarantee, so both deployed sets carry the certificate.
"""
import math

import numpy as np

from .config import DEPLOY_TOL_SE

KL_ITERS = 64          # bisection steps for the KL inversion (interval width 2**-64)


# ------------------------------------------------------------------ coverage on a sample

def set_coverage(est, seeds):
    """Bool (n_sets,): whether each RR set of est contains a seed. Every RR set holds its root."""
    is_seed = np.zeros(int(est.graph.n), dtype=bool)
    s = np.asarray(seeds, dtype=np.int64).ravel()
    if s.size:
        is_seed[s] = True
    hit = is_seed[np.asarray(est.nodes, dtype=np.int64)].astype(np.int32)
    return np.add.reduceat(hit, np.asarray(est.ptr[:-1], dtype=np.int64)) > 0


def _units(est, values):
    """(unit values, unit groups, units per group N_g (G,)). values: per RR set, in [0, 1]."""
    R, G = int(est.R), int(est.graph.n_groups)
    rg = np.asarray(est.root_groups_np, dtype=np.int64)
    rc = np.asarray(est.root_counts, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if est.all_roots:
        return v, np.repeat(rg, R), R * rc
    return v.reshape(int(est.n_roots), R).mean(axis=1), rg, rc.copy()


def present_groups(est):
    """Groups with members and at least one root (the verifier's definition)."""
    return (np.asarray(est.graph.group_sizes) > 0) & (np.asarray(est.root_counts) > 0)


def _unit_counts(est):
    """Units per group N_g (G,): R * roots when every node is a root, else roots (as _units)."""
    rc = np.asarray(est.root_counts, dtype=np.float64)
    return rc * int(est.R) if est.all_roots else rc.copy()


def _disp_se(rates, N, present):
    """Standard error (z = 1) of max - min of the present rates on this sample: sqrt(s_max^2 + s_min^2) with
    s_g = sqrt(max(r_g (1 - r_g), 1 / N_g) / N_g), the formula of verify.disparity_se. 0.0 below 2 present groups."""
    P = np.flatnonzero(present)
    if len(P) < 2:
        return 0.0
    r = np.clip(np.asarray(rates, dtype=np.float64)[P], 0.0, 1.0)
    n = np.maximum(np.asarray(N, dtype=np.float64)[P], 1.0)
    s = np.sqrt(np.maximum(r * (1.0 - r), 1.0 / n) / n)
    return float(math.sqrt(s[int(np.argmax(r))] ** 2 + s[int(np.argmin(r))] ** 2))


# ------------------------------------------------------------------ KL (Chernoff-Hoeffding) bounds

def _kl(p, q):
    p = np.clip(p, 0.0, 1.0)
    q = np.clip(q, 1e-300, 1.0 - 1e-16)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log(p / q), 0.0)
        b = np.where(p < 1, (1 - p) * np.log((1 - p) / (1 - q)), 0.0)
    return a + b


def kl_bounds(p_hat, n, level):
    """(lower, upper) with P(mean < lower) <= level and P(mean > upper) <= level, each one-sided,
    for the mean of n independent variables in [0, 1] whose sample mean is p_hat.

    upper = max{q >= p_hat : n * kl(p_hat, q) <= log(1/level)}, lower symmetric. Vectorised."""
    p = np.clip(np.asarray(p_hat, dtype=np.float64), 0.0, 1.0)
    n = np.asarray(n, dtype=np.float64)
    L = math.log(1.0 / float(level))
    ok = n > 0
    budget = np.where(ok, L / np.where(ok, n, 1.0), np.inf)
    lo_u, hi_u = p.copy(), np.ones_like(p)                # upper: kl(p, q) increasing on [p, 1]
    hi_l, lo_l = p.copy(), np.zeros_like(p)               # lower: kl(p, q) decreasing on [0, p]
    for _ in range(KL_ITERS):
        mid = 0.5 * (lo_u + hi_u)
        inside = _kl(p, mid) <= budget
        lo_u, hi_u = np.where(inside, mid, lo_u), np.where(inside, hi_u, mid)
        mid = 0.5 * (lo_l + hi_l)
        inside = _kl(p, mid) <= budget
        hi_l, lo_l = np.where(inside, mid, hi_l), np.where(inside, lo_l, mid)
    upper = np.where(ok, hi_u, 1.0)                       # hi_u: outermost point not yet ruled in
    lower = np.where(ok, lo_l, 0.0)
    return np.minimum(lower, p), np.maximum(upper, p)


# ------------------------------------------------------------------ group rates and disparity

def rate_bounds(est, seeds, level):
    """(rates (G,), lower (G,), upper (G,)): every one-sided bound holds with prob >= 1 - level."""
    x, grp, N = _units(est, set_coverage(est, seeds))
    G = int(est.graph.n_groups)
    rates = np.bincount(grp, weights=x, minlength=G) / np.maximum(N, 1.0)
    lo, hi = kl_bounds(rates, N, level)
    return rates, lo, hi


def _disparity_stats(est, seeds, delta):
    """(ucb, estimate, se) of D(S) on est: disparity_ucb's two values and the standard error _disp_se."""
    p = present_groups(est)
    P = int(p.sum())
    if P < 2:
        return 0.0, 0.0, 0.0
    rates, lo, hi = rate_bounds(est, seeds, delta / (2 * P))
    return (float(hi[p].max() - lo[p].min()), float(rates[p].max() - rates[p].min()),
            _disp_se(rates, _unit_counts(est), p))


def disparity_ucb(est, seeds, delta):
    """Upper confidence bound on D(S) = max_g r_g - min_g r_g (present groups), valid with
    probability >= 1 - delta (2 one-sided bounds per present group, delta split evenly).
    Returns (ucb, estimate)."""
    ucb, d_hat, _ = _disparity_stats(est, seeds, delta)
    return ucb, d_hat


# ------------------------------------------------------------------ spread floor

def floor_lcb(est, seeds, ref_seeds, rho, delta):
    """Lower confidence bound on T = sigma(S) - rho * sigma(S_ref), valid with probability >= 1 - delta.

    T_hat = sum_u w_u y_u over units, y_u = cov_u(S) - rho * cov_u(S_ref) in [-rho, 1], w_u = n_g / N_g.
    Q_hat = sum_u (w_u y_u)^2 estimates Q = sum_u E[(w_u y_u)^2] >= Var(T_hat). With probability
    >= 1 - delta/2, Q <= Q+ = (sqrt(Q_hat + c L/2) + sqrt(c L/2))^2 (Maurer 2003; c = max w_u^2,
    L = log(2/delta)). Then Bernstein with variance Q+ and range b = (1 + rho) max w_u gives
    T >= T_hat - t, t = b L/3 + sqrt((b L/3)^2 + 2 Q+ L), with probability >= 1 - delta/2 more.

    Returns {"lcb", "estimate", "halfwidth", "se", "identical"}. An identical set returns lcb = 0 with
    identical=True: T(S_ref) = (1 - rho) sigma(S_ref) >= 0 holds without sampling error.
    """
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1], got {rho}")
    a = np.unique(np.asarray(seeds, dtype=np.int64).ravel())
    b_ = np.unique(np.asarray(ref_seeds, dtype=np.int64).ravel())
    cov_s = set_coverage(est, a)
    if np.array_equal(a, b_):
        x, grp, N = _units(est, cov_s)
        sizes = np.asarray(est.graph.group_sizes, dtype=np.float64)
        w = sizes[grp] / np.maximum(N[grp], 1.0)
        val = (1.0 - rho) * float(np.dot(w, x))
        return {"lcb": 0.0, "estimate": val, "halfwidth": val, "se": 0.0, "identical": True}
    y = cov_s.astype(np.float64) - rho * set_coverage(est, b_).astype(np.float64)
    if est.all_roots:
        x, grp, N = _units(est, y)
    else:   # per-root averages of the paired differences
        R = int(est.R)
        x = y.reshape(int(est.n_roots), R).mean(axis=1)
        grp = np.asarray(est.root_groups_np, dtype=np.int64)
        N = np.asarray(est.root_counts, dtype=np.float64)
    sizes = np.asarray(est.graph.group_sizes, dtype=np.float64)
    w = sizes[grp] / np.maximum(N[grp], 1.0)
    wy = w * x
    T_hat = float(wy.sum())
    Q_hat = float(np.dot(wy, wy))
    L = math.log(2.0 / float(delta))
    c = float(np.max(w)) ** 2
    Q_plus = (math.sqrt(Q_hat + c * L / 2.0) + math.sqrt(c * L / 2.0)) ** 2
    brange = (1.0 + rho) * float(np.max(w))
    t = brange * L / 3.0 + math.sqrt((brange * L / 3.0) ** 2 + 2.0 * Q_plus * L)
    return {"lcb": T_hat - t, "estimate": T_hat, "halfwidth": t, "se": math.sqrt(Q_hat), "identical": False}


# ------------------------------------------------------------------ deployment rule

def certify_sets(est_b, candidates, ref_seeds, rho, alpha=0.05):
    """Theorem 2's statements for every candidate and the reference on B, without choosing.

    est_b: an RR sample independent of everything that chose the candidates and the reference.
    candidates: seed sets (lists of node ids), in path order when the path rule will choose (deploy_candidates);
    the reference is appended when it is not among them. Eligible: the reference, and every candidate with
    floor_lcb >= 0. alpha is split by a union bound over 2 statements (floor, disparity) per distinct set, so with
    probability >= 1 - alpha every eligible set meets sigma(S) >= rho sigma(S_ref) and every disparity bound holds.

    Returns {"certificates", "delta_each", "n_candidates"}. Certificate i has seeds (sorted), position (its index
    among the candidates; -1 for an appended reference), floor_lcb, floor_estimate, floor_halfwidth, disp_ucb,
    disp_estimate, disp_se (standard error of the labelled disparity on B, _disp_se), eligible and is_reference."""
    ref = sorted(int(v) for v in np.asarray(ref_seeds, dtype=np.int64).ravel())
    sets = [sorted(int(v) for v in np.asarray(c, dtype=np.int64).ravel()) for c in candidates]
    if not sets and not ref:
        raise ValueError("no candidates and no reference")
    distinct = {tuple(s) for s in sets} | {tuple(ref)}
    delta = float(alpha) / (2.0 * len(distinct))
    certs = []
    listed = sets + ([ref] if tuple(ref) not in {tuple(x) for x in sets} else [])
    for pos, s in enumerate(listed):
        fl = floor_lcb(est_b, s, ref, rho, delta)
        ucb, d_hat, se = _disparity_stats(est_b, s, delta)
        is_ref = s == ref
        certs.append({"seeds": s, "position": pos if pos < len(sets) else -1, "floor_lcb": fl["lcb"],
                      "floor_estimate": fl["estimate"], "floor_halfwidth": fl["halfwidth"], "disp_ucb": ucb,
                      "disp_estimate": d_hat, "disp_se": se, "eligible": bool(is_ref or fl["lcb"] >= 0.0),
                      "is_reference": is_ref})
    return {"certificates": certs, "delta_each": delta, "n_candidates": len(sets)}


def choose(certs, rule="min_ucb", tol_se=None):
    """Index into certs (certify_sets) of the set to deploy, always an eligible one.

    "min_ucb"  Theorem 2's rule: the smallest labelled-disparity upper bound; ties: the larger estimated spread,
               then the earlier certificate. T1's deployment.
    "path"     the furthest certified state along the search path (largest position) whose disparity upper bound is
               at most the smallest one plus tol_se standard errors of labelled disparity on B (the smallest-bound
               set's disp_se; default tol_se config.DEPLOY_TOL_SE). It keeps what the search's later states bought,
               for T2's option (b) the epsilon-constraint phase's community gains, unless B says they cost more than
               about tol_se standard errors of labelled disparity: the deployment analogue of the cap kappa. T2's
               deployment (docs/design_v2.md section 21).
    Both choose among eligible sets, so Theorem 2's floor guarantee holds for either, and the deployed set's
    disparity is at most its own upper bound (within tol_se standard errors of the smallest, for "path")."""
    elig = [i for i, c in enumerate(certs) if c["eligible"]]
    if not elig:
        raise ValueError("no eligible certificate: the reference must be among them")
    best = min(elig, key=lambda i: (certs[i]["disp_ucb"], -certs[i]["floor_estimate"], i))
    if rule == "min_ucb":
        return best
    if rule != "path":
        raise ValueError(f"unknown deployment rule {rule!r}: use 'min_ucb' or 'path'")
    tol = (DEPLOY_TOL_SE if tol_se is None else float(tol_se)) * float(certs[best].get("disp_se", 0.0))
    cap = float(certs[best]["disp_ucb"]) + tol
    within = [i for i in elig if certs[i]["disp_ucb"] <= cap + 1e-15]
    return max(within, key=lambda i: (certs[i].get("position", i), i))


def certified_choice(est_b, candidates, ref_seeds, rho, alpha=0.05, rule="min_ucb", tol_se=None):
    """Deploy one eligible candidate (docs/theory_v2.md, Theorem 2): certify_sets, then choose by `rule` ("min_ucb",
    the eligible set with the smallest disparity upper bound; or "path", see choose).

    Returns {"index", "seeds", "is_reference", "certificates", "delta_each", "rule"}; index is into the candidate
    list as given, or -1 when the reference (not among the candidates) is chosen."""
    out = certify_sets(est_b, candidates, ref_seeds, rho, alpha)
    certs = out["certificates"]
    best = choose(certs, rule, tol_se)
    index = best if best < out["n_candidates"] else -1
    return {"index": index, "seeds": certs[best]["seeds"], "is_reference": certs[best]["is_reference"],
            "certificates": certs, "delta_each": out["delta_each"], "rule": rule}


# ------------------------------------------------------------------ deployment protocol (design section 18)

def path_states(start, history):
    """Seed sets along one improve() history: the start, then the set after every applied move ("move" entries).
    When improve() restarted from the reference (history[0]["fallback"]), pass the reference as start."""
    cur = [int(s) for s in start]
    out = [sorted(cur)]
    for h in history[1:]:
        mv = h.get("move")
        if mv:
            rem, add = set(int(u) for u in mv[0]), [int(v) for v in mv[1]]
            cur = [s for s in cur if s not in rem] + add
            out.append(sorted(cur))
    return out


def deploy_candidates(paths, n_mid=3):
    """Theorem 2's candidates from one or more paths (lists of seed sets, e.g. path_states of each phase): the first
    and last set of every path, plus n_mid sets evenly spaced over the concatenated path, duplicates dropped. They
    are chosen on the search sample only, so certifying them on an independent sample stays valid.

    Returned in path order: by each set's last occurrence in the concatenated path (a set at the junction of two
    phases, or one a later phase returns to, counts where it last occurs), which the path rule of choose uses."""
    flat = [s for p in paths for s in p]
    picks = [s for p in paths if p for s in (p[0], p[-1])]
    if n_mid > 0 and len(flat) > 2:
        picks += [flat[i] for i in np.linspace(0, len(flat) - 1, int(n_mid) + 2)[1:-1].round().astype(int)]
    last = {}
    for i, s in enumerate(flat):
        last[tuple(sorted(int(v) for v in s))] = i
    seen, out = set(), []
    for s in picks:
        key = tuple(sorted(int(v) for v in s))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    out.sort(key=lambda s: last[tuple(s)])
    return out


def certified_equal_spread(est_b, seeds, ref_seeds, rho, alpha=0.05):
    """Test T1's equal-spread judgement, the same for every method (design section 18): does Lemma 6's paired
    floor bound, at level alpha on the independent sample est_b, certify sigma(S) >= rho * sigma(S_ref)?

    Returns {"passed", "lcb", "estimate", "halfwidth", "lcb_rel", "estimate_rel", "halfwidth_rel", "identical"};
    *_rel are divided by the reference's spread estimate on est_b. A set identical to the reference passes."""
    f = floor_lcb(est_b, seeds, ref_seeds, rho, alpha)
    rates = rate_bounds(est_b, ref_seeds, 0.5)[0]
    sizes = np.asarray(est_b.graph.group_sizes, dtype=np.float64)
    ref_total = max(float(np.nansum(np.asarray(rates, dtype=np.float64) * sizes)), 1e-12)
    return {"passed": bool(f["identical"] or f["lcb"] >= 0.0), "lcb": f["lcb"], "estimate": f["estimate"],
            "halfwidth": f["halfwidth"], "lcb_rel": f["lcb"] / ref_total, "estimate_rel": f["estimate"] / ref_total,
            "halfwidth_rel": f["halfwidth"] / ref_total, "identical": f["identical"]}
