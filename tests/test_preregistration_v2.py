"""Checks of the pre-registration amendment of 2026-10-10 (docs/design_v2.md section 18).

  defaults          improve() takes its weak-spot defaults from config (multi-scale audits, exhaustive fallback,
                    setpoint-scaled prices); the confirmed floors are RHO_C1 = 0.99 and RHO_C2 = 0.97
  disparity_se      the standard error behind T2's cap kappa: the noise model's formula, and an upper bound on
                    the spread of D across independent RR samples
  hive_search       option (b) of T2: a labelled-only phase, then the epsilon-constraint phase under
                    kappa = D_a + KAPPA_SE * SE; the floor holds and D never exceeds kappa
  deployment        path_states rebuilds the search path, deploy_candidates keeps both ends without duplicates,
                    certified_equal_spread passes the reference and fails a weak set
  capped floor      LearnedProposer and V2Trainer aim at config.rr_floor, which never exceeds total_ref
  learning_decision the W3 rule that gates training of a learned proposer
"""
import inspect
import math

import numpy as np
import pytest

from fimgat.baselines import RRGreedy
from fimgat.diffusion import ExposureEstimator
from fimgat.graph import from_edges

pytestmark = pytest.mark.spec

CFG = pytest.importorskip("fimgat.v2.config")
IMP = pytest.importorskip("fimgat.v2.improve")
CER = pytest.importorskip("fimgat.v2.certify")
DIA = pytest.importorskip("fimgat.v2.diagnostics")
VER = pytest.importorskip("fimgat.v2.verify")
PRC = pytest.importorskip("fimgat.v2.prices")
COV = pytest.importorskip("fimgat.v2.cover")


def _blocks_graph(seed=0, n_blocks=8, size=40):
    """Dense blocks with a few random cross edges; the three groups follow the blocks 60% of the time."""
    rng = np.random.default_rng(seed)
    n = n_blocks * size
    block = np.repeat(np.arange(n_blocks), size)
    edges = [(v, w) for v in range(n) for w in rng.choice(np.flatnonzero(block == block[v]), 4, replace=False)]
    edges += [(v, int(rng.integers(n))) for v in range(n) if rng.random() < 0.5]
    groups = np.where(rng.random(n) < 0.6, block % 3, rng.integers(0, 3, n))
    return from_edges("blocks", n, np.array(edges), groups, ["a", "b", "c"], undirected=True, weighted_cascade=True)


@pytest.fixture(scope="module")
def blocks():
    g = _blocks_graph()
    est = ExposureEstimator(g, R=16, seed=0, device="cpu")
    ref = [int(v) for v in RRGreedy(est).greedy(10)]
    floor = CFG.rr_floor(CFG.RHO_C1, COV.CoverState(est, ref).total)
    return g, est, ref, floor


# ------------------------------------------------------------------ the approved defaults

def test_improve_defaults_are_the_approved_decisions():
    p = inspect.signature(IMP.improve).parameters
    assert p["audit_band_ratio"].default == CFG.AUDIT_BAND_RATIO == 2.0
    assert p["fallback"].default == CFG.PROPOSER_FALLBACK == "exhaustive"
    assert p["price_setpoint"].default == CFG.PRICE_SETPOINT == pytest.approx(2 * CFG.DELTA_MARGIN)
    assert p["price_gains"].default is None
    assert CFG.PRICE_KP * CFG.PRICE_SETPOINT == pytest.approx(1.0)
    assert CFG.PRICE_KI * CFG.PRICE_SETPOINT == pytest.approx(0.5)
    assert CFG.RHO_C1 == CFG.RHO_DEFAULT == 0.99 and CFG.RHO_C2 == 0.97
    assert CFG.RHO_C1 in CFG.RHO_REPORT and CFG.RHO_C2 in CFG.RHO_REPORT
    assert CFG.MAX_CONDUCTANCE in CFG.MAX_CONDUCTANCE_GRID
    assert CFG.KAPPA_SE == 1.0 and CFG.T2_MIN_PAIRS == 20
    assert CFG.rr_floor(1.0, 100.0) == 100.0                              # the margin never lifts the floor above S_ref


def test_simple_proposer_uses_the_scaled_controller(blocks, monkeypatch):
    _, est, ref, floor = blocks
    made = []
    orig = PRC.PIController.__init__

    def spy(self, *args, **kw):
        orig(self, *args, **kw)
        made.append((self.kp, self.ki, self.setpoint))

    monkeypatch.setattr(PRC.PIController, "__init__", spy)
    IMP.improve(est, ref, floor, proposer="simple", audit=False, max_rounds=2)
    assert made and made[-1] == pytest.approx((CFG.PRICE_KP, CFG.PRICE_KI, CFG.PRICE_SETPOINT))
    made.clear()
    IMP.improve(est, ref, floor, proposer="simple", audit=False, max_rounds=2, price_setpoint=0.0,
                price_gains=(1.0, 0.5))                                   # the published controller
    assert made[-1] == pytest.approx((1.0, 0.5, 0.0))


# ------------------------------------------------------------------ the standard error behind kappa

def test_disparity_se_is_the_noise_model_formula(blocks):
    _, est, ref, _ = blocks
    state = COV.CoverState(est, ref)
    nm = VER.noise_model(state, None, 1.0)
    P = np.flatnonzero(nm.present)
    r = np.asarray(state.rates, dtype=np.float64)[P]
    units = nm.level_units[P]
    sd = np.sqrt(np.maximum(r * (1 - r), 1.0 / units) / units)
    want = math.sqrt(sd[np.argmax(r)] ** 2 + sd[np.argmin(r)] ** 2)
    assert VER.disparity_se(state) == pytest.approx(want, rel=1e-12)
    assert VER.disparity_se(state, z=2.0) == pytest.approx(2 * want, rel=1e-12)


def test_disparity_se_is_zero_with_one_group():
    g = _blocks_graph(3)
    one = g.with_groups(np.zeros(g.n, dtype=np.int64), ["all"])
    est = ExposureEstimator(one, R=4, seed=0, device="cpu")
    assert VER.disparity_se(COV.CoverState(est, list(range(5)))) == 0.0


def test_disparity_se_bounds_the_sampling_spread_of_d(blocks):
    """The Bernoulli-type variance bound makes SE an upper bound on the sd of each group rate; D = max - min is
    1-Lipschitz in every rate, so across independent RR samples sd(D) stays at or below about SE."""
    g, _, ref, _ = blocks
    d, se = [], []
    for s in range(40):
        st = COV.CoverState(ExposureEstimator(g, R=4, seed=100 + s, device="cpu"), ref)
        d.append(float(st.disp))
        se.append(VER.disparity_se(st))
    assert 0.0 < np.std(d, ddof=1) <= 1.25 * np.mean(se)


# ------------------------------------------------------------------ option (b) of T2: hive_search

def test_hive_search_runs_the_labelled_phase_then_the_capped_phase(blocks):
    _, est, ref, floor = blocks
    seeds, info = IMP.hive_search(est, ref, floor, max_rounds=60)
    h_a, h_b = info["phase1"], info["phase2"]
    assert all(h["family_size"] == 0 for h in h_a)                         # phase 1: labelled disparity only
    s_a = CER.path_states(info["start1"], h_a)[-1]
    st_a = COV.CoverState(est, s_a)
    assert info["start1"] == ref and info["disp_a"] == pytest.approx(float(st_a.disp), abs=1e-12)
    assert info["se_a"] == pytest.approx(VER.disparity_se(st_a), rel=1e-12)
    assert info["kappa"] == pytest.approx(info["disp_a"] + CFG.KAPPA_SE * info["se_a"], rel=1e-12)
    assert all(h["disp"] <= info["kappa"] + 1e-12 for h in h_b)            # D may rise, never above kappa
    assert all(h["total"] >= floor for h in h_a + h_b)
    assert CER.path_states(s_a, h_b)[-1] == sorted(seeds)
    assert h_b[-1]["family_size"] >= 2 and len(info["communities"]) == h_b[-1]["family_size"]
    for a, b in zip(h_b, h_b[1:]):                                          # (D_C, D) falls within an epoch
        if b["accepted"] and a["family_size"] == b["family_size"]:
            assert b["comm_disp"] < a["comm_disp"] - 1e-12 or (
                b["comm_disp"] <= a["comm_disp"] + 1e-12 and b["disp"] < a["disp"] - 1e-12)


def test_hive_search_restart_records_the_reference_as_its_start(blocks):
    g, est, ref, floor = blocks
    weak = list(range(10))                                                 # all in one block
    assert COV.CoverState(est, weak).total < floor
    seeds, info = IMP.hive_search(est, weak, floor, reference=ref, max_rounds=5)
    assert info["phase1"][0]["fallback"] and info["start1"] == ref
    s_a = CER.path_states(info["start1"], info["phase1"])[-1]
    assert CER.path_states(s_a, info["phase2"])[-1] == sorted(seeds)


def test_hive_search_owns_the_phase_switches(blocks):
    _, est, ref, floor = blocks
    for key, val in (("audit", True), ("disp_cap", 0.1), ("return_family", False)):
        with pytest.raises(ValueError):
            IMP.hive_search(est, ref, floor, **{key: val})


# ------------------------------------------------------------------ the deployment protocol

def test_path_states_follows_multi_node_moves(blocks):
    _, est, ref, _ = blocks
    floor = 0.5 * COV.CoverState(est, ref).total                           # loose, so random moves can pass

    def pairs(state, family, rng):                                          # random 2-for-2 moves
        seeds = np.asarray(state.seeds)
        non = np.flatnonzero(~state.is_seed)
        return [(tuple(rng.choice(seeds, 2, replace=False)), tuple(rng.choice(non, 2, replace=False)))
                for _ in range(40)]

    seeds, hist = IMP.improve(est, ref, floor, proposer=pairs, audit=False, fallback=None, max_rounds=8)
    path = CER.path_states(ref, hist)
    assert len(path) == 1 + sum(h["move"] is not None for h in hist[1:])
    assert any(h["move"] is not None and len(h["move"][0]) == 2 for h in hist[1:])
    assert path[0] == sorted(ref) and path[-1] == sorted(seeds)
    assert all(len(set(s)) == len(ref) for s in path)


def test_deploy_candidates_keep_both_ends_without_duplicates():
    p1 = [[1, 2], [1, 3], [3, 4], [4, 5], [5, 6]]
    p2 = [[6, 5], [6, 7], [7, 8]]
    c = CER.deploy_candidates([p1, p2], n_mid=3)
    for s in ([1, 2], [5, 6], [7, 8]):
        assert s in c
    assert len({tuple(s) for s in c}) == len(c) and all(s == sorted(s) for s in c)
    assert len(c) <= 2 * 2 + 3
    assert CER.deploy_candidates([p1], n_mid=0) == [[1, 2], [5, 6]]
    assert CER.deploy_candidates([[[3, 1]]], n_mid=3) == [[1, 3]]
    assert CER.deploy_candidates([[], p2], n_mid=0) == [[5, 6], [7, 8]]


def test_certified_equal_spread_passes_the_reference_and_fails_a_weak_set(blocks):
    g, _, ref, _ = blocks
    est_b = ExposureEstimator(g, R=16, seed=777, device="cpu")              # independent of the search sample
    same = CER.certified_equal_spread(est_b, ref, ref, CFG.RHO_C1)
    assert same["passed"] and same["identical"]
    weak = list(range(10))                                                 # all in one block
    out = CER.certified_equal_spread(est_b, weak, ref, CFG.RHO_C1)
    assert not out["passed"] and not out["identical"] and out["lcb"] < 0
    assert out["lcb"] == pytest.approx(out["estimate"] - out["halfwidth"])
    total_ref = float(COV.CoverState(est_b, ref).total)
    assert out["estimate_rel"] == pytest.approx(out["estimate"] / total_ref, rel=1e-6)


def test_certified_choice_over_the_hive_path_is_eligible(blocks):
    g, est, ref, floor = blocks
    seeds, info = IMP.hive_search(est, ref, floor, max_rounds=30)
    s_a = CER.path_states(info["start1"], info["phase1"])[-1]
    paths = [CER.path_states(info["start1"], info["phase1"]), CER.path_states(s_a, info["phase2"])]
    cands = CER.deploy_candidates(paths)
    assert sorted(ref) in cands and sorted(seeds) in cands
    est_b = ExposureEstimator(g, R=16, seed=778, device="cpu")
    pick = CER.certified_choice(est_b, cands, ref, CFG.RHO_C1, alpha=CFG.CERT_ALPHA)
    chosen = next(c for c in pick["certificates"] if c["seeds"] == pick["seeds"])
    assert chosen["eligible"] and CER.certified_equal_spread(est_b, pick["seeds"], ref, CFG.RHO_C1)["passed"]


# ------------------------------------------------------------------ the capped floor of the learned parts

def test_learned_proposer_aims_at_the_capped_floor():
    LP = pytest.importorskip("fimgat.v2.proposer").LearnedProposer
    assert LP(None, None, 1.0, 100.0).floor(None) == 100.0
    assert LP(None, None, 0.99, 100.0).floor(None) == pytest.approx(99.25)


def test_trainer_dual_ascent_targets_the_capped_floor(monkeypatch):
    T = pytest.importorskip("fimgat.v2.train")
    targets = []
    orig = T.disparity_lagrangian

    def spy(rates, total, ref, rho_eff, *args, **kw):
        targets.append(float(rho_eff))
        return orig(rates, total, ref, rho_eff, *args, **kw)

    monkeypatch.setattr(T, "disparity_lagrangian", spy)
    ctx = T.prepare_v2(_blocks_graph(4), [10], "cpu", R=4, rr_seed=0)
    tr = T.V2Trainer("cpu", seed=0, hidden=16, heads=2, gamma_comm=0.0)
    out = tr.step(ctx, 10, 1.0, 0.05)
    assert targets[-1] == 1.0                                               # not 1 + DELTA_MARGIN
    assert tr.mu[(id(ctx), 10)] == pytest.approx(max(0.0, 1.0 - out["ratio"]), abs=1e-12)
    tr.step(ctx, 10, 0.99, 0.05)
    assert targets[-1] == pytest.approx(0.99 + CFG.DELTA_MARGIN)


# ------------------------------------------------------------------ the W3 decision rule

def _meas(improvable, hit, q, T_ex, c_scores=1e-3, c_s=1e-5, grid=(16, 8)):
    return {"improvable": improvable, "T_ex": T_ex, "c_scores": c_scores, "c_s": c_s,
            "pi": {grid: hit}, "q": {grid: q if hit else 0.0}}


def test_learning_decision_follows_the_w3_rule():
    always = [_meas(True, True, 1.0, 1.0) for _ in range(5)]                # the simple proposer never misses
    d = DIA.learning_decision(always)
    assert not d["train"] and d["pi"] == 1.0 and d["H"] == pytest.approx(1.0)
    misses = [_meas(True, i % 2 == 0, 0.5, 1.0) for i in range(10)]         # half the rounds fall back
    d = DIA.learning_decision(misses)
    c_round = 1e-3 + 128 * 1e-5
    assert d["pi"] == 0.5 and d["q"] == 0.5 and d["c_round"] == pytest.approx(c_round)
    assert d["H"] == pytest.approx(DIA.learning_headroom(0.5, 0.5, c_round, 1.0))
    assert d["train"] and d["H"] >= CFG.LEARN_HEADROOM_MIN
    assert not DIA.learning_decision(misses, cost_ratio=d["H"] / 1.4)["train"]   # too costly a proposer
    small = DIA.learning_decision(misses, improve_share=0.1)
    assert not small["train"] and "Amdahl" in small["reason"]
    assert DIA.learning_decision(misses, improve_share=0.5)["train"]
    none = DIA.learning_decision([_meas(False, False, 0.0, 1.0)] * 3)
    assert not none["train"] and none["n_states"] == 0
    assert DIA.learning_decision(misses + [_meas(False, False, 0.0, 9.0)])["T_ex"] == pytest.approx(1.0)
