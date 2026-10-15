"""Hidden-community transfer bound (docs/theory_v2.md, Lemma W1.1).

Claim C2 says the hive-mind auditor lowers disparity on communities nobody labelled. The verifier only
controls the communities the auditor found (the family F). For any exposure vector a (true activation
probabilities, an estimate, or the expected exposure of a mixture of campaigns) and any hidden family H,

    D_H(a) <= D_F(a) + gap(h+, F) + gap(h-, F),

where h+ / h- are the hidden communities with the highest / lowest rate and gap(h, F) bounds how far h's
rate can be from the closest found community's: for node sets h, c with overlap I = h & c,

    |r_h - r_c| <= sd_h(a) sqrt(|h \\ c| / |I|) + sd_c(a) sqrt(|c \\ h| / |I|)        (Cauchy-Schwarz)
    |r_h - r_c| <= (|h \\ c| / |h| + |c \\ h| / |c|) * osc_{h | c}(a)                  (always finite)

with sd the population standard deviation and osc = max - min of a over the set. The three terms are what
the design can act on: what the verifier controls (D_F), whether the auditor finds communities close to
the hidden extremes, and how unevenly exposure spreads inside them.
"""
import math

import numpy as np

from .objective import to_numpy


def _as_sets(communities):
    if isinstance(communities, dict):
        communities = list(communities.values())
    return [np.unique(np.asarray(list(c) if isinstance(c, (set, frozenset)) else c, dtype=np.int64).ravel())
            for c in communities]


def community_rates(a, communities):
    """Mean of a (n,) over each community (node-id arrays; negative ids ignored): (C,)."""
    a = to_numpy(a).ravel()
    out = []
    for c in _as_sets(communities):
        c = c[(c >= 0) & (c < len(a))]
        out.append(float(a[c].mean()) if len(c) else math.nan)
    return np.asarray(out, dtype=np.float64)


def pair_gap_bound(a, h, c):
    """Upper bound on |r_h - r_c| (mean of a over node sets h and c): the smaller of the
    Cauchy-Schwarz and the oscillation bounds of Lemma W1.1."""
    a = to_numpy(a).ravel()
    h, c = _as_sets([h, c])
    if len(h) == 0 or len(c) == 0:
        return math.inf
    inter = np.intersect1d(h, c, assume_unique=True)
    ni = len(inter)
    hc, ch = len(h) - ni, len(c) - ni
    both = np.union1d(h, c)
    osc = float(a[both].max() - a[both].min())
    bound = (hc / len(h) + ch / len(c)) * osc
    if ni > 0:
        bound = min(bound, float(a[h].std()) * math.sqrt(hc / ni) + float(a[c].std()) * math.sqrt(ch / ni))
    return bound


def _closest(a, h, family):
    best, arg = math.inf, -1
    for j, c in enumerate(family):
        g = pair_gap_bound(a, h, c)
        if g < best:
            best, arg = g, j
    return best, arg


def transfer_bound(a, hidden, family):
    """Lemma W1.1 for exposure a (n,), hidden communities and the audited family (node-id arrays).

    Returns {"D_H", "D_F", "h_max", "h_min", "gap_max", "gap_min", "match_max", "match_min", "bound",
    "holds"}: D_H <= bound = D_F + gap_max + gap_min, where gap_max (gap_min) is the smallest gap bound
    between the highest- (lowest-) rate hidden community and any family member, match_* its index in the
    family. With an empty family the bound is infinite."""
    a = to_numpy(a).ravel()
    H, F = _as_sets(hidden), _as_sets(family)
    rh = community_rates(a, H)
    ok = np.flatnonzero(np.isfinite(rh))
    if len(ok) < 2:
        return {"D_H": 0.0, "D_F": math.nan, "h_max": -1, "h_min": -1, "gap_max": 0.0, "gap_min": 0.0,
                "match_max": -1, "match_min": -1, "bound": 0.0, "holds": True}
    i_max, i_min = int(ok[np.argmax(rh[ok])]), int(ok[np.argmin(rh[ok])])
    D_H = float(rh[i_max] - rh[i_min])
    if not F:
        return {"D_H": D_H, "D_F": math.nan, "h_max": i_max, "h_min": i_min, "gap_max": math.inf,
                "gap_min": math.inf, "match_max": -1, "match_min": -1, "bound": math.inf, "holds": True}
    rf = community_rates(a, F)
    rf = rf[np.isfinite(rf)]
    D_F = float(rf.max() - rf.min()) if len(rf) >= 2 else 0.0
    g_max, j_max = _closest(a, H[i_max], F)
    g_min, j_min = _closest(a, H[i_min], F)
    bound = D_F + g_max + g_min
    return {"D_H": D_H, "D_F": D_F, "h_max": i_max, "h_min": i_min, "gap_max": g_max, "gap_min": g_min,
            "match_max": j_max, "match_min": j_min, "bound": bound, "holds": bool(D_H <= bound + 1e-12)}
