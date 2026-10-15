"""Exact verifier for the improvement phase (docs/design_v2.md sections 5 and 7).

A move is accepted only if the RR spread stays at or above the floor, neither the labelled-group
disparity D nor the community disparity D_C gets worse, and at least one of them strictly
improves (verifier option (b)). D_C is taken over a CommunityFamily: the communities the hive-mind
auditor has found so far in this query, plus any given ones. The family only grows, so "before"
and "after" are always compared on the same family.

Communities are node-id arrays and may overlap. A community's rate is the mean per-root exposure
a_r over the RR roots it contains; nodes that are not roots are ignored, and a community with no
roots is rejected. The verifier keeps per-community sums of integer RR-set cover counts, so the
community disparity after a swap comes from the few roots the swap touches and equals a full
recomputation on the swapped state exactly.

Noise-aware acceptance (docs/theory_v2.md section 3). Every number here is exact on one RR sample,
but that sample is random. With a NoiseModel (z > 0), "strictly improves" means that a lower
confidence bound on the true improvement, certified_gain, is positive: it allows for groups that are
within sampling noise of the extremes (near-ties) and for the noise of the move's own change, which
comes only from the RR sets the move touches. z = 0 reproduces the plain rule exactly.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .objective import disparity, present_mask, to_numpy


@dataclass(frozen=True)
class MoveDetail:
    """Rates after a move and, per group / community, the second moment of the move's change in RR-set
    units (_touched): the number of RR sets that changed coverage when every node is a root, the sum of
    squared changed sets per root with sampled roots."""
    rates: np.ndarray
    touched: np.ndarray
    comm_rates: np.ndarray = field(default_factory=lambda: np.zeros(0))
    comm_touched: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass(frozen=True)
class Evaluation:
    total: float
    disp: float
    comm_disp: float = 0.0
    detail: object = None              # MoveDetail when evaluated with detail=True


@dataclass(frozen=True)
class NoiseModel:
    """Sampling-noise scales of one RR sample (docs/theory_v2.md section 3).

    level_units: independent units behind each group rate (RR sets if every node is a root, else
    roots); delta_units: R x roots per group, the scale of a move's change, whose second moment is
    MoveDetail.touched. comm_*: the same per community.
    """
    z: float
    present: np.ndarray
    level_units: np.ndarray
    delta_units: np.ndarray
    comm_level_units: np.ndarray
    comm_delta_units: np.ndarray


def noise_model(state, family=None, z=2.0):
    """NoiseModel of a CoverState's sample (and of a community family on it)."""
    est, ix = state.est, state.index
    rc = np.asarray(est.root_counts, dtype=np.float64)
    delta = ix.R * rc
    level = delta.copy() if est.all_roots else rc.copy()
    if family is not None and len(family):
        family._build()
        csize = family._size.astype(np.float64)
        cdelta = ix.R * csize
        clevel = cdelta.copy() if est.all_roots else csize.copy()
    else:
        cdelta = clevel = np.zeros(0)
    return NoiseModel(float(z), np.asarray(state.present, dtype=bool).copy(), level, delta, clevel, cdelta)


def level_halfwidth(rates, units, z):
    """z * sd of a rate estimated from `units` independent [0, 1] units (Bernoulli-type variance bound,
    floored at 1/units so a zero or full rate still carries uncertainty)."""
    r = np.clip(to_numpy(rates), 0.0, 1.0)
    n = np.maximum(np.asarray(units, dtype=np.float64), 1.0)
    return float(z) * np.sqrt(np.maximum(r * (1.0 - r), 1.0 / n) / n)


def change_halfwidth(touched, units, z):
    """z * sd of a move's estimated change, z * sqrt(touched + 1) / units (docs/theory_v2.md, Lemma 3):
    the change sums +-1 over the touched RR sets only; touched is its second moment (_touched)."""
    n = np.maximum(np.asarray(units, dtype=np.float64), 1.0)
    return float(z) * np.sqrt(np.asarray(touched, dtype=np.float64) + 1.0) / n


def disparity_se(state, z=1.0):
    """Standard error (z = 1) of the labelled disparity on the sample: sqrt(lam_max^2 + lam_min^2), the level
    half-widths of the groups at the highest and the lowest rate (noise_model; docs/theory_v2.md, Lemma 5).
    It sets the cap of T2's option (b): kappa = D + config.KAPPA_SE * disparity_se (docs/design_v2.md section 18).
    0.0 with fewer than 2 present groups."""
    nm = noise_model(state, None, z)
    P = np.flatnonzero(nm.present)
    if len(P) < 2:
        return 0.0
    r = to_numpy(state.rates)[P]
    lam = level_halfwidth(r, nm.level_units[P], z)
    i, j = int(np.argmax(r)), int(np.argmin(r))
    return float(np.sqrt(lam[i] ** 2 + lam[j] ** 2))


def certified_gain(x, x_new, lam, eta, present=None):
    """Lower bound on the true decrease of max - min (docs/theory_v2.md, Lemma 5).

    x, x_new: estimated values before / after a move; lam: half-widths with |x - true| <= lam;
    eta: half-widths with |(x_new - x) - true change| <= eta. On those events

        true gain >= min_g [max(0, (x_g - lam_g) - min_h(x_h + lam_h)) + (d_g - eta_g)]      (min side)
                   + min_g [max(0, max_h(x_h - lam_h) - (x_g + lam_g)) - d_g - eta_g]       (max side)

    with d = x_new - x. With lam = eta = 0 it equals the plain gain (max x - min x) - (max x_new - min x_new).
    Entries outside `present` are ignored; 0.0 with fewer than 2 present.
    """
    x, xn = to_numpy(x), to_numpy(x_new)
    p = present_mask(present, x.shape[-1])
    x, xn = x[p], xn[p]
    if x.size < 2:
        return 0.0
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), p.shape)[p]
    eta = np.broadcast_to(np.asarray(eta, dtype=np.float64), p.shape)[p]
    d = xn - x
    cap_min = np.min(x + lam)                       # >= true min on the level event
    cap_max = np.max(x - lam)                       # <= true max on the level event
    gain_min = np.min(np.maximum(0.0, (x - lam) - cap_min) + d - eta)
    gain_max = np.min(np.maximum(0.0, cap_max - (x + lam)) - d - eta)
    return float(gain_min + gain_max)


def accept(before, after, floor, tol=1e-12, *, min_gain=0.0, min_gain_comm=0.0, noise=None, disp_cap=None):
    """False if after breaks the floor or worsens either disparity by more than tol. Otherwise True only
    if one of them improves: by more than max(tol, min_gain) for D, max(tol, min_gain_comm) for D_C.

    With noise (a NoiseModel with z > 0) and both evaluations carrying a MoveDetail, the improvement is
    certified_gain (a lower bound allowing for sampling noise) instead of the plain difference. The
    "no worse" checks always use the plain values.

    disp_cap (docs/theory_v2.md, W1; default None = the rule above): D may rise up to max(D_before,
    disp_cap) when D_C strictly improves (an epsilon-constraint phase that lowers D_C under a cap on D)."""
    if not all(math.isfinite(x) for x in (after.total, after.disp, after.comm_disp)):
        return False
    if after.total < floor:
        return False
    dd = after.disp - before.disp
    dc = after.comm_disp - before.comm_disp
    d_limit = before.disp if disp_cap is None else max(before.disp, float(disp_cap))
    if after.disp - d_limit > tol or dc > tol:
        return False
    g_d, g_c = -dd, -dc
    b, a = before.detail, after.detail
    if noise is not None and noise.z > 0 and b is not None and a is not None:
        g_d = certified_gain(b.rates, a.rates, level_halfwidth(b.rates, noise.level_units, noise.z),
                             change_halfwidth(a.touched, noise.delta_units, noise.z), noise.present)
        if len(b.comm_rates) >= 2 and len(a.comm_rates) == len(b.comm_rates) == len(noise.comm_level_units):
            g_c = certified_gain(b.comm_rates, a.comm_rates,
                                 level_halfwidth(b.comm_rates, noise.comm_level_units, noise.z),
                                 change_halfwidth(a.comm_touched, noise.comm_delta_units, noise.z))
    return bool(g_d > max(tol, float(min_gain)) or g_c > max(tol, float(min_gain_comm)))


class CommunityFamily:
    """A growing family of (possibly overlapping) communities, scored on the estimator's RR roots.

    communities[c] holds the node ids given, members[c] the sorted root positions (into est.roots).
    Near-duplicates are judged on root sets, since a community's rate depends only on its roots.
    """

    def __init__(self, est, communities=(), dedupe_jaccard=0.9):
        self.est, self.R, self.n_roots = est, int(est.R), int(est.n_roots)
        self.dedupe_jaccard = float(dedupe_jaccard)
        self.communities, self.members = [], []
        self._idx = None
        self._cache = {}
        for c in (communities if communities is not None else ()):
            self.add(c)

    def __len__(self):
        return len(self.members)

    def _roots_of(self, nodes):
        nodes = np.unique(np.asarray(nodes, dtype=np.int64).ravel())
        roots = self.est.roots                                 # sorted node ids
        pos = np.searchsorted(roots, nodes)
        ok = pos < len(roots)
        pos, nodes = pos[ok], nodes[ok]
        return pos[roots[pos] == nodes].astype(np.int64), nodes

    def _near_duplicate(self, idx):
        a, t, mask = len(idx), self.dedupe_jaccard, None
        for m in self.members:
            b = len(m)
            if min(a, b) < t * max(a, b):                      # Jaccard <= min/max < t
                continue
            if mask is None:
                mask = np.zeros(self.n_roots, dtype=bool)
                mask[idx] = True
            inter = int(mask[m].sum())
            if inter >= t * (a + b - inter):
                return True
        return False

    def add(self, nodes):
        """Add a community; False (not added) if it has no roots or is a near-duplicate of a member."""
        if nodes is None:
            return False
        idx, _ = self._roots_of(nodes)
        if len(idx) == 0 or self._near_duplicate(idx):
            return False
        self.communities.append(np.unique(np.asarray(nodes, dtype=np.int64).ravel()))
        self.members.append(idx)
        self._idx = None
        self._cache.clear()
        return True

    def _build(self):
        if self._idx is None:
            lens = np.array([len(m) for m in self.members], dtype=np.int64)
            self._size = lens.astype(np.float64)
            self._idx = np.concatenate(self.members) if self.members else np.zeros(0, dtype=np.int64)
            self._comm_of = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
            # root -> communities containing it (CSR), for incremental swaps
            self._rcomm = self._comm_of[np.argsort(self._idx, kind="stable")]
            self._rptr = np.zeros(self.n_roots + 1, dtype=np.int64)
            np.cumsum(np.bincount(self._idx, minlength=self.n_roots), out=self._rptr[1:])

    def rates(self, exposure):
        """Mean of exposure (n_roots,) over each community's roots: (C,)."""
        if not self.members:
            return np.zeros(0)
        self._build()
        e = to_numpy(exposure)
        return np.bincount(self._comm_of, weights=e[self._idx], minlength=len(self)) / self._size

    def disp(self, exposure):
        """max - min of community rates; 0.0 with fewer than 2 communities."""
        r = self.rates(exposure)
        return float(r.max() - r.min()) if len(r) >= 2 else 0.0

    # ---------------------------------------------------------- exact paths on a CoverState

    def _check(self, state):
        if state.est is not self.est:
            raise ValueError("state and community family use different estimators")

    def _cov_sums(self, state):
        """Per-community sums of root cover counts (exact integers in float64), cached by seed set."""
        self._check(state)
        key = tuple(sorted(int(s) for s in state.seeds))
        sums = self._cache.get(key)
        if sums is None:
            self._build()
            # UNVERIFIED: CoverState.exposure() is root_cov / R (contract section 5), so rint recovers root_cov
            cov = np.rint(to_numpy(state.exposure()) * self.R)
            sums = np.bincount(self._comm_of, weights=cov[self._idx], minlength=len(self))
            if len(self._cache) >= 8:
                self._cache.clear()
            self._cache[key] = sums
        return sums

    def _disp_of(self, sums):
        r = sums / (self.R * self._size)
        return float(r.max() - r.min())

    def _scatter(self, ridx, vals):
        """Per-community sums of vals over the given root positions (a root counts for every community
        that contains it): (C,)."""
        ridx = np.asarray(ridx, dtype=np.int64)
        out = np.zeros(len(self))
        if len(ridx):
            starts = self._rptr[ridx]
            cnt = self._rptr[ridx + 1] - starts
            n_hit = int(cnt.sum())
            if n_hit:
                # positions in _rcomm of every (touched root, community) pair
                seg = np.repeat(starts - (np.cumsum(cnt) - cnt), cnt) + np.arange(n_hit)
                d = np.repeat(np.asarray(vals, dtype=np.float64), cnt)
                out = np.bincount(self._rcomm[seg], weights=d, minlength=len(self))
        return out

    def state_rates(self, state):
        """Exact community rates (C,) of a CoverState."""
        if not self.members:
            return np.zeros(0)
        return self._cov_sums(state) / (self.R * self._size)

    def state_disp(self, state):
        """Exact community disparity of a CoverState; 0.0 with fewer than 2 communities."""
        return self._disp_of(self._cov_sums(state)) if len(self) >= 2 else 0.0

    def disp_after_swap(self, state, u, v):
        """Exact community disparity after swapping seed u out and v in, without mutating state.
        Only the roots returned by state.swap_root_delta(u, v) change."""
        if len(self) < 2:
            return 0.0
        sums = self._cov_sums(state)
        ridx, dcov = state.swap_root_delta(u, v)
        return self._disp_of(sums + self._scatter(ridx, dcov))

    def after_change(self, state, loss, gain):
        """(community rates (C,), the change's second moment per community (C,), as in _touched) after the
        coverage of the RR sets `loss` / `gain` flips. The rates equal disp_after_swap's to the last bit."""
        self._build()
        sums = self._cov_sums(state)
        ridx, dcov = state.root_delta_from(loss, gain)
        rates = (sums + self._scatter(ridx, dcov)) / (self.R * self._size)
        tidx, tcnt = state.root_touch_from(np.concatenate([np.asarray(loss, dtype=np.int64),
                                                           np.asarray(gain, dtype=np.int64)]))
        return rates, self._scatter(tidx, _unit_moment(state, tcnt))


# ------------------------------------------------------------ evaluating states and moves

def _unit_moment(state, per_root):
    """Touched RR sets per root -> each root's share of the change's second moment (in RR-set units).
    Every node a root: the RR sets are the independent units, so a root contributes its count. Sampled
    roots: the roots are the units (their between-root variance counts too), so it contributes the
    count squared, which bounds (R x its exposure change)^2 (docs/theory_v2.md, Lemma 3)."""
    return per_root if state.est.all_roots else np.asarray(per_root, dtype=np.float64) ** 2


def _touched(state, sets, G):
    """Per-group second moment of a move's change, given the RR sets whose coverage changes."""
    sets = np.asarray(sets, dtype=np.int64)
    if state.est.all_roots:
        return np.bincount(state.index.set_group[sets], minlength=G).astype(np.float64)
    roots, cnt = state.root_touch_from(sets)
    return np.bincount(state.index.root_groups[roots], weights=_unit_moment(state, cnt), minlength=G)


def evaluate_state(state, family=None, detail=False):
    comm = family.state_disp(state) if family is not None else 0.0
    md = None
    if detail:
        cr = family.state_rates(state) if family is not None and len(family) else np.zeros(0)
        md = MoveDetail(to_numpy(state.rates).copy(), np.zeros(len(state.rates)), cr, np.zeros(len(cr)))
    return Evaluation(float(state.total), float(state.disp), float(comm), md)


def evaluate_swap(state, u, v, family=None, detail=False):
    """Exact Evaluation after swapping seed u out and v in; state is not mutated. detail=True also
    returns the post-swap rates and the touched-set counts (MoveDetail) for noise-aware acceptance."""
    if not detail:
        d_rates, d_total = state.swap_delta(u, v)
        rates = to_numpy(state.rates) + to_numpy(d_rates)
        comm = family.disp_after_swap(state, u, v) if family is not None else 0.0
        return Evaluation(float(state.total + d_total), disparity(rates, state.present), float(comm))
    d_rates, d_total, loss, gain = state.swap_delta_detail(u, v)
    rates = to_numpy(state.rates) + to_numpy(d_rates)
    touched = _touched(state, np.concatenate([loss, gain]), len(rates))     # loss and gain are disjoint
    comm, cr, ct = 0.0, np.zeros(0), np.zeros(0)
    if family is not None and len(family):
        cr, ct = family.after_change(state, loss, gain)
        comm = float(cr.max() - cr.min()) if len(cr) >= 2 else 0.0
    return Evaluation(float(state.total + d_total), disparity(rates, state.present), float(comm),
                      MoveDetail(rates, touched, cr, ct))


def evaluate_move(state, removes, adds, family=None, detail=False):
    """Exact Evaluation after removing the seeds `removes` and adding `adds`, on a copy of state."""
    removes = [int(u) for u in np.atleast_1d(np.asarray(removes, dtype=np.int64))]
    adds = [int(v) for v in np.atleast_1d(np.asarray(adds, dtype=np.int64))]
    if len(removes) != len(adds):
        raise ValueError(f"a move must keep the budget: {len(removes)} removes, {len(adds)} adds")
    s = state.copy()
    for u in removes:
        s.remove(u)
    for v in adds:
        s.add(v)
    ev = evaluate_state(s, family)
    if not detail:
        return ev
    changed = np.flatnonzero((s.counts > 0) != (state.counts > 0))
    touched = _touched(state, changed, len(state.rates))
    cr, ct = np.zeros(0), np.zeros(0)
    if family is not None and len(family):
        family._build()
        cr = family.state_rates(s)
        tidx, tcnt = state.root_touch_from(changed)
        ct = family._scatter(tidx, _unit_moment(state, tcnt))
    return Evaluation(ev.total, ev.disp, ev.comm_disp, MoveDetail(to_numpy(s.rates).copy(), touched, cr, ct))
