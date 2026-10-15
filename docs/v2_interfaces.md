# FIM-GAT v2 interfaces (implementation contract)

Companion to `docs/design_v2.md`. It fixes every module, public name, signature and convention, so that v2 can be written in parallel and later checked against `tests/test_spec_v2.py`, whose header summarises the same contract. When this file and the tests disagree, **the tests win**; report the conflict rather than working around it.

## 0. Ground rules

- v2 lives in `fimgat/v2/`, `experiments/v2/`, `results/v2/` and `data/v2/`. **v1 modules are not modified,** with two exceptions: the empty-community guard in `fimgat/metrics.py`, and the removal of dead code on 2026-10-10 (the v1 trainer, its adversaries and losses, and the SPA loaders; `docs/design_v2.md` §19). v2 may import any v1 module.
- Python 3.11, torch 2.6, torch_geometric 2.7, numba 0.63, numpy, scipy, pandas, networkx, sklearn. Nothing else, unless an experiment script checks for an optional package and reports clearly when it is missing.
- Style follows v1: short module docstrings that say what and why, sparse comments, numpy/torch vectorisation, numba for loops over RR sets, and no printing inside library code (experiments print).
- **Exactness:** the verifier state (`cover.py`) works in **float64 numpy**, on the same RR sets as `ExposureEstimator`. The estimator and training work in float32 torch.
- Any assumption that can't be checked without running code is marked with a comment `# UNVERIFIED: ...`.

## 1. Shared conventions

- `graph`: v1 `fimgat.graph.FIMGraph`. Arcs point in the direction influence flows. Weighted cascade: `p(u->v) = 1/in_degree(v)`, built with `from_edges(..., weighted_cascade=True)`.
- `est`: v1 `fimgat.diffusion.ExposureEstimator`. Useful fields: `est.graph`, `est.R`, `est.roots` (sorted node ids), `est.n_roots`, `est.all_roots`, `est.ptr` and `est.nodes` (CSR of the RR sets, root-major: set j belongs to root `j // R`), `est.root_groups_np`, `est.root_counts` (float per group), `est.evaluate_set(seeds) -> (rates np (G,), total float)`, and `est.batch_exposure_rates(X)`.
- **Group rate** of group g: `rate_g = (1 / (R * root_counts[g])) * sum over sets j rooted in g of covered_j`. **Total** `= rates @ graph.group_sizes`. This matches `est.evaluate_set` exactly, in float64.
- **Present groups:** `present = (graph.group_sizes > 0) & (est.root_counts > 0)`.
- **Disparity** `= max(rates[present]) - min(rates[present])`. It is 0.0 when fewer than 2 groups are present.
- **Per-root exposure** `a_r = (number of covered sets among root r's R sets) / R`.
- **Communities** are arrays of node ids and may overlap. A community's rate is the mean of `a_r` over the roots it contains. Communities with no root are ignored.
- **Floor:** `floor = (rho + delta) * total_ref`, with `total_ref` the RR total of the reference set and `delta = config.DELTA_MARGIN` (§2). Acceptance compares `state.total >= floor`.
- Budgets `config.BUDGETS = (25, 50, 100, 150, 200)` apply everywhere.

## 2. `fimgat/v2/config.py`

```python
BUDGETS = (25, 50, 100, 150, 200)
DIFFUSION = "weighted_cascade"
SNAP_GT = "com-youtube"
RHO_DEFAULT = 0.99
RHO_REPORT = (0.97, 0.98, 0.99, 1.0)
REFERENCE_RUNS = 5
DELTA_MARGIN = 0.0025          # RR floor safety margin; tuned on validation only
PRICE_TAU = 0.02
MIN_COMMUNITY = 10             # tuned on validation only
MAX_CONDUCTANCE = 0.3          # tuned on validation only
N_TRAIN_SEEDS = 5
TIME_BUDGETS = (10, 60, 300)   # seconds per query, test T3
TIER = {"rice_facebook": "small", "twitter_political": "small", "pokec_z": "medium", "pokec_n": "medium",
        "pokec_t": "medium", "epinions": "medium", "twitter_ego": "medium", "pokec_full": "large",
        "instagram": "large", "snap_youtube": "large", "linkedin": "large"}
N_SIMS = {"small": 5000, "medium": 2000, "large": 1000}
RR_PER_NODE = {"small": 512, "medium": 16, "large": None}   # large: from the memory preset (full_R, full_max_roots)
ROOT, DATA_V2, RAW_V2, RESULTS_V2, MODELS_V2              # absolute paths: repo, data/v2, data/v2/raw, results/v2, results/v2/models
OLD_PROJECT = r"C:\Users\Odwitiyo\Desktop\xAI Project\Baseline_Test"   # read-only
```

## 3. `fimgat/v2/objective.py`

- `disparity(rates, present) -> float`. Accepts numpy or torch.
- `smooth_max(r, present, tau)` and `smooth_min(r, present, tau)` (torch): `tau*logsumexp(r/tau)` and `-tau*logsumexp(-r/tau)` over present entries, with **no** `-log G` correction. Hence `smooth_disparity = smooth_max - smooth_min` lies in `[true, true + 2*tau*log(G_present)]`.
- `smooth_disparity(rates_t, present, tau) -> torch scalar`. `present` may be a bool numpy array or tensor.
- `signed_prices(rates, present, sizes, mu, tau=PRICE_TAU) -> (w, alpha, beta)`, all numpy (G,):
  - `alpha = softmax(r/tau)` and `beta = softmax(-r/tau)` over present groups, 0 elsewhere;
  - `w = mu * sizes / sizes[present].sum() + beta - alpha` (entries of absent groups are 0).
- `disparity_lagrangian(rates, total, total_ref, rho, present, tau, mu, penalty, comm_rates=None, comm_present=None, gamma=0.5) -> torch scalar` (to minimise):
  - `smooth_disparity(rates) + gamma * smooth_disparity(comm_rates)` (the second term only if given);
  - `+ mu * relu(rho - total/total_ref) + 0.5 * penalty * relu(rho - total/total_ref)**2`.

## 4. `fimgat/v2/prices.py`

- `PIController(kp=1.0, ki=0.5, tau=PRICE_TAU)`: `.update(rates, present, total, floor) -> (alpha, beta, mu)` and `.reset()`.
  - Spread error `e = (floor - total) / floor`. Integral with anti-windup: `I = max(0, I + e)`. Price `mu = max(0, kp*e + ki*I)`. (The spec test requires: below the floor `mu` is non-decreasing and positive; after enough steps comfortably above the floor, `mu == 0`.)
  - Damped group prices: `dev = r - center` over present groups, where `center` is the size-weighted mean rate. Then `alpha = s * relu(dev)/sum(relu(dev))`, `beta = s * relu(-dev)/sum(relu(-dev))`, with strength `s = D/(D + tau)` and D the current disparity (`s = 0` if `D = 0`).

## 5. `fimgat/v2/cover.py` (exact incremental verifier state)

```python
class CoverState:
    def __init__(self, est, seeds): ...
    est, seeds (list[int], order preserved), counts (np.int32 per RR set: number of seeds in the set),
    rates (np.float64 (G,)), total (float), root_cov (np.int64 per root), present (bool (G,)), sizes (float64 (G,))
    disp  (property)                   # disparity(rates, present)
    swap_detail(u, v) -> (loss_sets, gain_sets)   # sets losing / gaining coverage if u is swapped out and v in
    swap_delta(u, v) -> (d_rates np (G,), d_total float)   # exact; ValueError if u not a seed or v already a seed
    swap_root_delta(u, v) -> (root_idx np.int64, d_cov np.int64)  # change in root_cov (for community rates)
    apply_swap(u, v)                   # updates counts, root_cov, rates, total; seeds[i] = v in place of u
    add(v); remove(u)                  # used by multi-node moves; keep every structure consistent
    removal_loss() -> np (k, G)        # exact rate lost per group by removing each seed (sets with count == 1)
    gains() -> np (n, G)               # exact rate gained per group by adding each node (sets with count == 0); rows of seeds = 0
    exposure() -> np (n_roots,)        # root_cov / R
    copy() -> CoverState               # deep copy of the mutable arrays; shares the static structures
```

- Static structures are built once per estimator and cached on it as `est._v2_cover`:
  - `node_ptr`, `node_sets` (for each node, the RR sets containing it);
  - `set_group` (`np.repeat(est.root_groups_np, est.R)`);
  - `set_w = 1/(R*root_counts[set_group])`;
  - `set_root = np.arange(n_sets) // R`;
  - `root_index` (node id → root position, or -1).
- Swap logic:
  - **loss:** sets containing u with `count == 1` that do **not** contain v;
  - **gain:** sets containing v with `count == 0`;
  - sets containing both are unchanged.
- Use numba for the hot loops, e.g. `gains()` over all nnz. `swap_delta` can be numpy (`np.isin` with `assume_unique=True`).

## 6. `fimgat/v2/verify.py`

```python
@dataclass(frozen=True)
class Evaluation: total: float; disp: float; comm_disp: float = 0.0

def accept(before, after, floor, tol=1e-12) -> bool:
    # False if after.total < floor; False if either disparity got worse by more than tol;
    # True only if at least one of them improved by more than tol.

class CommunityFamily:
    def __init__(self, est, communities=(), dedupe_jaccard=0.9): ...
    add(nodes) -> bool                 # False if empty (no roots) or near-duplicate (Jaccard >= dedupe_jaccard) of a member
    __len__()
    rates(exposure) -> np (C,)         # mean a_r over each community's roots
    disp(exposure) -> float            # max - min over communities; 0.0 if fewer than 2
    disp_after_swap(state, u, v) -> float   # exact community disparity after the swap, via state.swap_root_delta

def evaluate_state(state, family=None) -> Evaluation
def evaluate_swap(state, u, v, family=None) -> Evaluation   # exact, without mutating state
def evaluate_move(state, removes, adds, family=None) -> Evaluation  # multi-node move, via state.copy()
```

The family only **grows** within a query. At every round, "before" and "after" are both computed on the current family.

## 7. `fimgat/v2/reference.py`

- `mc_spreads(graph, seeds, n_sims, seed) -> np (n_sims,)`: the spread of each simulation, `grp.sum(1)` from `fimgat.diffusion.monte_carlo`.
- `reference_spread(graph, k, runs=REFERENCE_RUNS, R=16, seed=0, n_sims=2000, device="cpu", max_roots=None, mc_seed=12345) -> {"seeds", "spread", "run_spreads", "run_seeds"}`.
  - Run i uses `ExposureEstimator(graph, R, seed=seed*1000 + i, device, max_roots)` and `RRGreedy(est).greedy(k)`.
  - Each run is scored by `mc_spreads(..., seed=mc_seed).mean()`; the best run wins.
- `equal_spread_test(graph, seeds, ref_seeds, rho=RHO_DEFAULT, n_sims=2000, seed=12345, alpha=0.05) -> {"passed", "ratio", "lower", "se"}`.
  - Paired simulations with the same `seed`: `X = mc_spreads(seeds)`, `Y = mc_spreads(ref_seeds)`, `ratio = X.mean()/Y.mean()`.
  - Delta-method standard error: `se = std(X - ratio*Y, ddof=1)/(sqrt(n)*Y.mean())`.
  - `lower = ratio - z_{1-alpha}*se` (one-sided); `passed = lower >= rho`.
  - When the two seed sets are identical, `se = 0` and the ratio is exactly 1.

## 8. `fimgat/v2/auditor.py` (hive mind: cohesive-community search)

- `conductance(graph, nodes) -> float`: on the undirected, unweighted, de-duplicated graph (v1 `features.undirected_csr`), `cut(S, V\S) / min(vol(S), vol(V\S))`, returning 1.0 if either volume is 0.
- `node_exposure(est, root_exposure) -> np (n,)`: roots get their own value. A non-root gets the mean over its root neighbours (undirected), or the global root mean if it has none.
- `find_community(graph, exposure, side="under", min_size=MIN_COMMUNITY, max_conductance=MAX_CONDUCTANCE, seed=0, alpha=0.15, n_starts=8, max_size=None) -> np.ndarray | None`:
  - Score `s_v = 1 - exposure_v` ("under") or `exposure_v` ("over").
  - Starts are the top `n_starts // 2` nodes by score, plus `n_starts // 2` drawn with `np.random.default_rng(seed)` from the top 5%.
  - From each start, run a local personalised PageRank (Andersen–Chung–Lang push, numba; teleport alpha; `eps = 1/(20 * max_size)`, with `max_size` defaulting to `n//2`).
  - Sweep prefixes ordered by `ppr/deg`. Keep prefixes with `min_size <= |C| <= max_size` and `conductance <= max_conductance`, computed incrementally during the sweep.
  - Among all starts and prefixes, maximise the scan statistic `sqrt(|C|) * (mean_C(s) - mean_V(s))`.
  - Return the sorted node ids, or None if nothing qualifies.
  - Deterministic for a given `seed`. **Must never call `fimgat.metrics.communities`** (leakage rule).
- `audit(graph, exposure, min_size, max_conductance, seed) -> (under, over)`.

## 9. `fimgat/v2/local_search.py` (required baseline; also the "exhaustive" proposer)

- `best_swap(state, floor, family=None, device=None) -> (u, v, Evaluation) | None`.
  - Exact best-improvement single swap. For each seed u:
    - candidate rates for every non-seed v are `rates - L_u + G_v + O_uv`;
    - `L_u` = removal loss, `G_v` = gains;
    - `O_uv` = rate regained on sets uniquely covered by u that also contain v (a sparse correction).
  - Totals are computed the same way. Candidates are filtered by `total >= floor`, and the minimum disparity is chosen (ties: larger total).
  - Vectorised in torch, chunked over u.
  - With a family, candidates are ranked by labelled disparity, and the best few are re-checked with `accept` including `comm_disp`.
- `local_search(est, start, floor, communities=None, max_passes=1000, max_seconds=None, device=None) -> list[int]`.
  - Repeat `best_swap` while it improves by more than 1e-12. Returns a single-swap local optimum.

## 10. `fimgat/v2/improve.py` (the improvement phase)

```python
def improve(est, start, floor, communities=None, proposer="simple", max_rounds=50, max_seconds=None,
            reference=None, audit=True, audit_every=5, min_size=MIN_COMMUNITY, max_conductance=MAX_CONDUCTANCE,
            n_add=16, n_drop=8, rng_seed=0) -> (seeds, history)
```

- **Fallback:** if `CoverState(start).total < floor`, restart from `reference`. If `reference` is None, use `RRGreedy(est).greedy(k)`. Then `history[0]["fallback"] = True`.
- **Family:** a `CommunityFamily` seeded with `communities`. If `audit`, add `auditor.audit(...)` on `node_exposure(est, state.exposure())` at round 0 and every `audit_every` rounds.
- **Proposers:**
  - `"simple"`: signed prices from `PIController`, add scores `gains() @ w` (top `n_add` non-seeds), drop scores `removal_loss() @ w` (smallest `n_drop` seeds), giving `n_add*n_drop` swaps;
  - `"exhaustive"`: `local_search.best_swap`;
  - a **callable** `proposer(state, family, rng) -> list[move]`, where a move is `(u, v)` or `(removes: tuple, adds: tuple)` (the learned proposer).
- **Rounds:** every candidate is scored exactly (`evaluate_swap` / `evaluate_move`). Apply the best candidate that `accept`s: lowest disp, then comm_disp, then higher total. Stop after a round with no accepted move. If audit is on, re-audit once before stopping. Also stop on `max_rounds` or `max_seconds`, checked after every candidate batch.
- **History:** one dict per round with keys `round, total, disp, comm_disp, fallback, accepted, n_candidates, family_size, secs`. Labelled `disp` must never increase.

## 11. `fimgat/v2/proposer.py`

- `simple_moves(state, w, n_add, n_drop) -> list[(u, v)]`, used by `improve("simple")`.
- `class LearnedProposer(policy, ctx, rho, total_ref, n_add=16, n_drop=8, n_multi=4, multi_size=(2, 3))`. It is callable as `(state, family, rng) -> list[move]`:
  - build v2 dynamic features from the state (`features_v2.dynamic_from_state`);
  - run `policy.score(...)` to get `(add_scores, drop_scores)`;
  - return top swaps, plus `n_multi` multi-node moves combining top drops with top adds.

## 12. `fimgat/v2/features.py`

- `N_DYNAMIC_V2 = 14`. Static features are v1's 9 (`GraphContext.static`).
- `build_context(graph, est, device, edge_device=None) -> GraphContext` (the v1 class), plus `ctx.total_ref = {}` filled by the caller.
- `dynamic_v2(ctx, x, a_roots, rates, total, gains, w, rho, t, rounds, k, removal=None) -> torch (n, 14)`. Columns:
  0. `x*n/(10k)`
  1. exposure `a` (via `ctx.full_exposure`)
  2. own-group normalised rate `rate_own/(total/n)`
  3. own-group signed price `w_own*G`
  4. `signlog(gains @ w * total_ref[k])`
  5. `log1p(gains.sum(1))` (z-scored)
  6. own-group rate gain, `log1p` and z-scored as in v1
  7. `signlog(reach_rate @ w)`
  8. `nbr_frac @ w`
  9. `rho`
  10. floor slack `total/total_ref[k] - rho`
  11. round `t/max(rounds-1,1)`
  12. `signlog(removal @ w)` for seeds, 0 otherwise
  13. `is_seed` (`x > 0.5`)

  Here `signlog(z) = sign(z)*log1p(|z|)`, z-scored with v1's `_tz`. Every group quantity enters as own-group or w-weighted, which keeps the model **invariant to group relabelling**.
- `dynamic_from_state(ctx, state, w, rho, total_ref, t, rounds) -> torch (n, 14)`, used at inference for binary seed sets via a `CoverState`.

## 13. `fimgat/v2/model.py`

- `PolicyV2(in_dim=N_STATIC+N_DYNAMIC_V2, hidden=64, layers=3, heads=4, dropout=0.1, checkpoint=False)`.
  - Same GATv2 backbone as v1 `SeedPolicy`, with residual pre-norm, explicit self-loops and a mean-context readout.
  - Two heads: `forward(feats, edge_index, edge_attr) -> (add_scores, drop_scores)`.
  - `forward_chunked(...)` with the same semantics as v1.
  - `score(ctx, feats)` picks full or chunked like `train.policy_scores`.
- `CommunityAdversaryV2(in_dim=N_STATIC, K=16, hidden=32, layers=2)`. `forward(static, edge_index) -> S` gives soft memberships (n, K), a softmax over K.
  - `dmon_loss(S, edge_index, n)`: negative modularity plus the collapse regulariser `sqrt(K)/n * ||sum_i S_i|| - 1` (Tsitsulin et al. 2023).
  - `size_penalty(S, min_size)`: `sum_k relu(min_size - S[:,k].sum())/min_size`.
- `soft_community_rates(S_roots, a_roots) -> (K,)`: `(S^T a)/(S^T 1)`.

## 14. `fimgat/v2/train.py`

- `prepare_v2(graph, ks, device, R, rr_seed, max_roots=None, edge_device=None) -> ctx`. Same as v1 `prepare` (it can call it), but features come from `features.build_context`. `ctx.total_ref[k]` is the RR total of RR-greedy, as in v1.
- `rollout_v2(policy, ctx, k, rho, T, rounds, mu, grad_round=None) -> list[(add_scores, x)]`. This is v1's "replace" dynamics with `dynamic_v2` features. Signed prices come from `objective.signed_prices(rates, present, sizes, mu)`.
- `class V2Trainer(device, seed=0, rounds=3, lr=3e-3, adv_lr=3e-3, K_comm=16, gamma_comm=0.5, min_size=MIN_COMMUNITY, penalty=10.0, checkpoint=False)`.
  - `step(ctx, k, rho, T)`: two adversary steps, which maximise `smooth_disparity(soft community rates) - dmon_loss - size_penalty` with the policy output detached. Then one policy step, which minimises `disparity_lagrangian` including the community term with the adversary detached. `mu` is updated per `(ctx id, k)` by dual ascent: `mu = max(0, mu + eta*(rho - total/total_ref))`.
  - `rho` is sampled per step from `U(0.95, 1.0)`, and k from `BUDGETS`. T anneals 1.0 → 0.1 with a cosine schedule.
  - `validate(val_ctxs, ks, rhos=(0.99,)) -> float`: mean over cases of `-disparity` for `select_seeds_v2`, plus `-1` for each case whose floor fails.
  - `fit(train_ctxs, val_ctxs, ks, epochs, steps_per_epoch=None, eval_every=5, verbose=True) -> best` (restores the best state). Also `save(path, **meta)` and `load(path)`.
- `select_seeds_v2(policy, ctx, k, rho, floor, rounds, n_samples=64) -> list[int]`. Candidates are the top-k of each round's add scores plus Gumbel-top-k samples (as in v1 `decode`). Each is scored on the RR sample and the lowest-disparity candidate meeting `floor` wins. If none meets it, return `ctx.greedy_seeds[k]`.

## 15. `fimgat/v2/metrics_v2.py`

- `evaluate_v2(graph, seeds, n_sims=2000, seed=12345, communities=None, min_comm=10, return_sims=False) -> dict`. Keys:
  - `influence`, `influence_se`, `rates` (list over present labelled groups), `disparity`, `min_rate`, `max_rate`;
  - `heldout_disparity` and `heldout_worst_rel` (mean over the other attributes in `graph.meta["attrs"]`);
  - `comm_disparity`, `comm_worst_rel`, `comm_q10_rel`, `n_comm_eval`.
  - Community metrics use the given `communities` (ground truth, overlapping allowed; members' node_prob averaged). Otherwise they use v1 `metrics.communities(graph)` restricted to size >= `min_comm`. They are NaN when nothing qualifies.
  - With `return_sims`, also returns `sims` (per-simulation spread).

## 16. `fimgat/v2/datasets.py`

- `NAMES = ("rice_facebook", "twitter_political", "pokec_z", "pokec_n", "pokec_full", "epinions", "twitter_ego", "instagram", "snap_youtube", "linkedin")`. `pokec_t` (validation district, Trenčín + Považská Bystrica) is loadable but deliberately **not** in `NAMES`, which is the test registry.
- `SOURCES`: name → list of raw file paths or URLs. `SOURCES["linkedin"] = [OLD_PROJECT/soc-linkedin.edges]` only; **never** the `.nodes` file.
- `splits() -> {"train": ["pokec_z"], "val": ["pokec_t"], "test": [every name in NAMES except pokec_z]}`. `HELD_OUT = tuple(splits()["test"])`.
  - This is a simplification of design §8: we train and validate on Pokec districts only, and everything else is zero-shot test. That keeps every test graph cleanly held out, with no overlap between samples and full graphs.
- `available(name) -> bool` (the built file exists). `load(name, attribute=None) -> FIMGraph`, always weighted cascade:
  - `meta = {"dataset", "attribute", "attrs": {attr: groups}, "tier", "attr_names": {attr: names}}`.
  - Default attributes: pokec → "region", rice → "college", twitter_political → "leaning", instagram → "gender", the others → "degree".
  - Every graph carries `attrs["degree"] = degree_groups(full graph)`.
- `degree_groups(graph, n_bins=2) -> np.int64 (n,)`: quantile bins of undirected degree (low → high). With `n_bins=2`, `<= median` is 0 and the rest is 1.
- `sample(name, size, seed) -> FIMGraph`: an induced subgraph on a BFS ball from a seeded random start. Weighted-cascade probabilities are recomputed inside the sample. `meta["full_ids"]` holds the original ids, and `meta["attrs"]["degree"]` is taken **from the full graph**.
- `ground_truth_communities(name, which="top5000") -> list[np.ndarray]` for `snap_youtube`. Ids are remapped to graph ids; sizes are >= 3; overlap is allowed.
- Built files: `data/v2/<name>.npz` with `n`, `arcs` (directed influence arcs `src -> dst`, before weighted cascade), `undirected` (bool), one int array per attribute plus `<attr>_names`, and `orig_id`. YouTube communities go in `data/v2/snap_youtube_cmty.npz` (`ptr`, `members`). Pokec loads through `fimgat.data.load_pokec` from `data/pokec/` (`pokec_t` from `data/pokec/pokec_t.npz`).
- Build functions: `build_all(raw_dir=RAW_V2, only=None)` and `build_<name>(...)`. They are called by `experiments/v2/build_datasets.py`, which also downloads raw files with urllib into `data/v2/raw/<name>/`.

## 17. `fimgat/v2/baselines_v2.py`

- `fw_disparity(est, k, rho, total_ref, steps=100, tau=0.02, penalty=10.0) -> list[int]`: Frank–Wolfe on the multilinear `disparity_lagrangian` (Tsang-style), then decoding that picks the lowest-disparity candidate meeting the floor.
- `run_suite(ctx, k, rho, floor, reference, methods=None, time_limit=None) -> dict[name, (seeds, secs)]`:
  - `Degree`, `Greedy`, `Parity-Greedy`, `Welfare-Greedy`, `Maximin-Greedy`, `FW-Disparity`;
  - `LocalSearch`: best of the Parity, Welfare and Greedy starts;
  - `Simple-v2`: `improve(proposer="simple")` from the best floor-meeting start among Parity/Welfare/Greedy;
  - `AGE` (n <= 5000) and `Prior` (n <= 100000).
  - A method that fails (OOM, or `time_limit`) is recorded as missing with its reason, never silently dropped.

## 18. `experiments/v2/`

- `common_v2.py`: `RecorderV2(path, n_sims)`.
  - Rows carry `dataset, attribute, k, rho, method, variant, train_seed, seeds, time_s, prep_s, code_tag, gpu, preset`, all `evaluate_v2` metrics, and the equal-spread test fields (`es_passed, es_ratio, es_lower`).
  - `done(**key)` must include `rho` and `variant`.
  - The CSV is written atomically: temp file, then replace.
- `build_datasets.py`, `train_constructor.py`, `run_t1_t2.py`, `train_proposer.py`, `run_t3.py`, `run_t4.py` and `summarize_v2.py` have argparse CLIs and write under `results/v2/`. Every script is resumable. Scripts that take hours print an estimate first.
- `external/`: one adapter per external baseline (IMM/OPIM, RFIM, BSM, DQ4FairIM, S3D, CrossWalk). Each fetches or builds the upstream code, exports the graph, runs it and parses the seeds. All are marked UNVERIFIED until run.

## 19. As-built amendments (lead decisions, 2026-09-27; these override earlier sections)

The core library was written against sections 0–17. These are the real APIs and the decisions taken on the authors'
questions. New code must use these.

- **config**: `rr_floor(rho, total_ref, delta=DELTA_MARGIN) = min(rho + delta, 1) * total_ref`. **Every** script uses it for
  the RR floor, so at rho = 1.0 the reference still qualifies. `AUDIT_MAX_SIZE = 20000` caps auditor communities (push
  cost scales with max_size/alpha); it is tuned on validation only.
- **objective**: extra helpers `to_numpy`, `present_mask`, `smooth_max/min` and
  `price_weights(alpha, beta, mu, sizes, present)`. The simple proposer uses PIController (damped) prices through
  `price_weights`; the learned proposer and training use `signed_prices` (softmax).
- **prices**: `PIController(kp, ki, tau, sizes=None)`, and `update(..., sizes=None)`. Pass `sizes=state.sizes` so the
  centre is size-weighted, as section 4 requires.
- **cover**:
  - `cover_index(est)` returns a `CoverIndex` (fields `gw`, `root_index`, `node_ptr`, `node_sets`, ...).
  - `CoverState` also has `is_seed`, `hits`, `k`, `index`, and integer-count methods `loss_counts()`, `gain_counts()`,
    `overlap_counts() -> (oi, ov, oc)`.
  - Coverage is binary and float64. It is exact against the RR definition, and within about 1e-6 relative of the
    float32-clamped `est.evaluate_set`, which is why spec-test totals use rel = 1e-5.
  - CoverState raises on duplicate or out-of-range seeds.
- **local_search**:
  - `best_swap(state, floor, family=None, device=None, *, n_check=32, max_elems=1<<23, deadline=None)`.
  - Below the floor it runs in **repair mode**: the lowest-disparity swap reaching the floor, else the largest-total
    swap.
- **auditor**:
  - The sweep ranks every node the push touched by `(p + alpha*r)/deg` (a tighter PPR lower bound). **Ratified.**
  - Starts exclude nodes of degree 0.
  - Decision: `find_community` returns **None when the best scan statistic is <= 0** (no genuinely under- or
    over-exposed cohesive community). A reviewer will implement this.
  - `audit(..., max_size=None)` accepts a cap. Callers pass `config.AUDIT_MAX_SIZE`.
- **improve**:
  - Signature adds `audit_max_size=AUDIT_MAX_SIZE` and `return_family=False`. With `return_family=True`, the
    CommunityFamily's communities are attached to the last history entry as `communities` (lists of node ids), for the
    collapse diagnostics. A reviewer will implement both.
  - Accepted moves must reach `floor_acc = floor*(1 + 5e-5)` (FLOOR_RTOL). The fallback uses the exact floor.
  - `history[0]` is the start state, and every entry carries `fallback`.
- **proposer**:
  - `LearnedProposer(policy, ctx, rho, total_ref, ..., mu_ref=None)` uses `mu_ref` (the checkpoint's per-k dual price,
    `ckpt["mu_ref"][k]`) when given, so prices match training. A reviewer will implement this.
  - Drop head: **higher = better to remove.**
  - An improvement state is featurised as round `t = 2` of `rounds = 3`.
- **features** (`fimgat/v2/features.py`, not `features_v2`):
  - `IN_DIM = 23`.
  - Column 4 = `signlog(((1-x)*gains/n_g) @ w * total_ref[k])`, and column 12 = `signlog((removal/n_g) @ w * total_ref[k])`,
    z-scored among seeds only. `gains` and `removal` are in `est.group_gains` units; `removal` defaults to `x*gains`.
  - `build_context` sets `ctx.present_np`.
- **model**: `PolicyV2.forward -> (add, drop)`; `score(ctx, feats)`; `CommunityAdversaryV2`, `dmon_loss`, `size_penalty`,
  `soft_community_rates` as in section 13.
- **train**:
  - `exact_rates(est, seeds) -> (rates float64, total)`.
  - `prepare_v2(graph, ks, device, R=16, rr_seed=0, max_roots=None, edge_device=None)`. It runs greedy once at max(ks);
    exact float64 `total_ref`.
  - `select_seeds_v2(policy, ctx, k, rho, floor, rounds, n_samples=64, mu=0.0, T=0.05, ..., return_info=False)`. Pass
    `mu=trainer.mu_for(k)` or `ckpt["mu_ref"][k]`.
  - `V2Trainer(..., adv_steps=2, tau, dual_lr=1.0, rho_range=(0.95,1.0), hidden, layers, heads, truncate=True)`.
    `gamma_comm=0` is the no-adversary ablation. `step()` returns a dict including `eff_comm` (the scale-free collapse
    diagnostic, in [1, K]) and `n_comm`. `save(path, **meta)` stores `mu_ref`. `load_policy(path, device) ->
    (PolicyV2, ckpt)`.
  - The drop head is **not** trained by V2Trainer; `fimgat/v2/imitation.py` trains it.
- **reference**:
  - `mc_spreads` sorts seeds.
  - `reference_spread` also returns `best_run` and `spread_se`.
  - `equal_spread_test(..., sims=None, ref_sims=None)` also returns `spread` and `ref_spread`; `passed = lower >= rho`.
  - v1 `monte_carlo` draws are **not** common across different seed sets, so the pairing does not reduce variance. A
    live-edge CRN kernel is planned (`fimgat/v2/crn.py`, reviewer task) and will become the default stream for
    `equal_spread_test` and `evaluate_v2`.
- **metrics_v2**:
  - `simulate(graph, seeds, n_sims, seed) -> (node_prob, grp)` validates its inputs.
  - Given communities are used in full (`min_comm` applies only to the Louvain fallback).
  - Held-out attributes exclude `meta["attribute"]` and any array identical to `graph.groups`.
- **datasets**:
  - Extras: `POKEC`, `LOADABLE`, `DEFAULT_ATTRIBUTE`, `built_path`, `raw_files`, `undirected_degree`,
    `degree_group_names`, `BUILDERS`, `build_all(raw_dir, only, force=False)`.
  - `sample(name, size, seed, attribute=None, full=None)` names samples `f"{name}_bfs{size}_s{seed}"`.
  - npz `arcs` stores every directed arc (both directions for undirected data).
  - Epinions ("a trusts b") and ego-Twitter ("a follows b") are **reversed**, so influence flows b -> a. **Ratified.**
  - Missing categories become an explicit "unknown" group. Samples inherit the source dataset's tier.
- **baselines_v2**:
  - `fw_disparity(est, k, rho, total_ref, ..., floor=None, reference=None, return_info=False)`.
  - `run_suite(ctx, k, rho, floor, reference, methods=None, time_limit=None, cache=None) -> SuiteResult` (a dict
    name -> (seeds or None, secs), plus `.reason` and `.info`). Keep **one `cache` dict per ctx** across rho values.
  - Helpers `check_seeds`, `SUITE`, `GREEDY`, `STARTS`.
- **common_v2**:
  - `RecorderV2(path, n_sims=None, preset=None, alpha=0.05)` with `.add(graph, k, rho, method, seeds, secs,
    ref_seeds=None, variant="default", train_seed=NaN, prep_s=NaN, communities=None, mc_seed=None, n_sims=None,
    **extra)`, `.add_missing(...)`, `.add_suite(graph, k, rho, suite, ref_seeds, variant="default", prep_s=NaN,
    communities=None, skip_done=True, **extra)`, `.done(**key)` (must include rho and variant), `.find`, `.remove`.
  - Baselines use `variant="default"` and `train_seed=NaN`. Rows carry `status` ("ok"/"missing") and `reason`. `done()`
    counts missing rows unless `status="ok"` is in the key.
  - The evaluation MC stream is `mc_seed_for(graph) = crc32(graph.name) % 2**31`. It differs from reference_spread's
    12345, which avoids a winner's curse.
  - Helpers: `preset_config`, `device`, `gpu_name`, `timed`, `reset_peak`, `peak_gpu_gb`, `code_tag`, `n_sims_for`,
    `fmt_row`.

## 20. Theory amendments (2026-10-10; `docs/theory_v2.md`). These override earlier sections where they differ.

- **certify** (new, `fimgat/v2/certify.py`): certified deployment on an RR sample B that is independent of everything that chose the candidates.
  - `set_coverage(est, seeds) -> bool (n_sets,)`.
  - `kl_bounds(p_hat, n, level) -> (lower, upper)`, one-sided at `level` each.
  - `rate_bounds(est, seeds, level) -> (rates, lower, upper)`.
  - `disparity_ucb(est, seeds, delta) -> (ucb, estimate)`.
  - `floor_lcb(est, seeds, ref_seeds, rho, delta) -> {"lcb", "estimate", "halfwidth", "se", "identical"}`. A set identical to the reference returns lcb = 0.
  - `certified_choice(est_b, candidates, ref_seeds, rho, alpha=CERT_ALPHA) -> {"index", "seeds", "is_reference", "certificates", "delta_each"}`. `index` is -1 when the reference (not among the candidates) is deployed. Each certificate carries `floor_lcb`, `floor_estimate`, `floor_halfwidth`, `disp_ucb`, `disp_estimate`, `eligible` and `is_reference`.
  - Units are RR sets when every node is a root, else per-root averages (conservative).
  - Stage 1 scripts (T1/T2) build B with `ExposureEstimator(graph, R, seed=<a seed used nowhere else>, max_roots)`. They pass the start, final and a few intermediate sets of each query, deploy `out["seeds"]`, and record the certificate fields next to the Monte Carlo evaluation.
- **verify**:
  - `Evaluation(total, disp, comm_disp=0.0, detail=None)`. `MoveDetail(rates, touched, comm_rates, comm_touched)`.
  - `evaluate_state / evaluate_swap / evaluate_move(..., detail=False)`. With `detail=True` they also return post-move rates and, per group / per community, the second moment of the move's change in RR-set units (`touched`): the number of RR sets whose coverage changed when every node is a root, the sum of squared changed sets per root with sampled roots (theory Lemma 3). Totals and disparities are bit-identical to the `detail=False` path.
  - `NoiseModel` and `noise_model(state, family=None, z=2.0)`.
  - `level_halfwidth(rates, units, z)`, `change_halfwidth(touched, units, z)`, `certified_gain(x, x_new, lam, eta, present=None)`.
  - `accept(before, after, floor, tol=1e-12, *, min_gain=0.0, min_gain_comm=0.0, noise=None)`. The defaults reproduce the earlier rule exactly. With `noise` (z > 0) and MoveDetail on both evaluations, "improves" means certified_gain > max(tol, min_gain); the "no worse" checks stay plain.
  - `CommunityFamily.after_change(state, loss, gain) -> (rates, touched)`.
- **cover**: `swap_delta_detail(u, v) -> (d_rates, d_total, loss_sets, gain_sets)`, `root_delta_from(loss, gain)`, `root_touch_from(sets)`. `swap_delta` and `swap_root_delta` are unchanged in value.
- **improve**:
  - Keyword-only additions: `z=None` (default `config.Z_ACCEPT`), `min_gain=0.0`, `min_gain_comm=0.0`, `tabu=None` (default `config.TABU`, True), `max_audits=config.MAX_AUDITS` (None).
  - Tabu: no seed set is visited twice (63-bit Zobrist hash of the set).
  - History entries add `move` (`[removes, adds]` of the applied move, or None) and `n_tabu` (candidates skipped by tabu).
- **local_search**: `best_swap(..., accept_kw=None, exclude=None)`, where `exclude(u, v) -> bool` is applied to the re-scored pool. With a noise model the pool is at least 128. `local_search(..., z=0.0, min_gain=0.0)`.
- **config**: `Z_ACCEPT = 0.0` (tuned on validation only), `TABU = True`, `MAX_AUDITS = None`, `CERT_ALPHA = 0.05`.
- **tests**: `tests/test_theory_v2.py` (marker `spec`) is the executable form of the theory's lemmas.

## 21. Weak-spot amendments (2026-10-10; `docs/theory_v2.md` Part II). These override earlier sections where they differ.

Every option below is off by default, so the published behaviour is unchanged.
- **improve**:
  - Keyword-only additions:
    - `audit_band_ratio=None`: multi-scale audits;
    - `fallback=None`: `"exhaustive"` makes a round whose proposals contain no accepted move fall back to `best_swap` before widening or stopping;
    - `price_setpoint=0.0` and `price_gains=None` (a `(kp, ki)` pair) for the simple proposer's `PIController`;
    - `disp_cap=None`: the epsilon-constraint phase.
  - `audit_max_size=AUDIT_MAX_SIZE` and `return_family=False` are implemented as specified in the earlier improve section. The cap is applied as `min(audit_max_size, n // 2)`, so graphs below 2 * AUDIT_MAX_SIZE nodes behave as before.
  - History entries add `fallback_move` (the move came from the exhaustive fallback). The restart-from-reference flag keeps its key, `fallback`.
  - With `disp_cap`, candidates are ranked by D_C first, and `disp` need not be monotone; `(comm_disp, disp)` decreases lexicographically within an epoch.
- **verify.accept**: `disp_cap=None`. D may rise up to `max(D_before, disp_cap)` when D_C strictly improves; D_C may never rise.
- **auditor**: `size_bands(min_size, max_size, ratio=2.0) -> [(lo, hi), ...]` and `audit_bands(graph, exposure, bands, max_conductance, seed, alpha, n_starts) -> [id arrays]` (under before over, band by band).
- **prices.PIController**: `setpoint=0.0`. The error is `((1 + setpoint) * floor - total) / floor`.
- **proposer**: `community_moves(state, family, n_add=16, n_drop=8, tau=None) -> [(u, v), ...]`: swaps steered by damped prices over the community family (empty below 2 communities).
- **hidden** (new):
  - `community_rates(a, communities)`;
  - `pair_gap_bound(a, h, c)`;
  - `transfer_bound(a, hidden, family) -> {"D_H", "D_F", "h_max", "h_min", "gap_max", "gap_min", "match_max", "match_min", "bound", "holds"}`.
- **mixture** (new):
  - `action_rates(state, family, use_groups)`;
  - `root_weights(state, family, w, use_groups)`;
  - `best_response(state, root_w, floor, n_add, n_drop, max_moves)`;
  - `lp_mixture(rates)`;
  - `mixture_campaigns(est, start, floor, communities=(), T=30, ...) -> {"sets", "weights", "rates", "mix_rates", "mix_disp", "uniform_disp", "campaign_disp", "best_single_disp", "regret_bound", "n_actions", "T", "moves"}`.
- **bounds** (new):
  - `coverage_upper`;
  - `group_rate_upper(est, k, exclude_own=False, delta=0.05)`;
  - `lower_bound_from(U, U0, k, mean_floor)`;
  - `disparity_lower_bound(est, k, floor_total, delta=0.05) -> {"lb", "mean_floor", "maximin_ub", "u_k1", "min_U", "U", "U0", "n_groups", "k"}`.
- **diagnostics** (new):
  - `hybrid_speedup`, `fallback_share` and `learning_headroom`;
  - `measure_proposer(est, seeds, floor, grids, scores=None, ...)`;
  - `pi_loop_stable` and `pi_loop_radius`.
- **tests**: `tests/test_theory_v2_w.py` (marker `spec`). The spec-suite test of `return_family` now runs.

## 22. Pre-registration amendments (2026-10-10; `docs/design_v2.md` §18). These override earlier sections where they differ.

The user approved these before any v2 run.

- **config**:
  - `RHO_C1 = 0.99` (= `RHO_DEFAULT`) and `RHO_C2 = 0.97`;
  - `AUDIT_BAND_RATIO = 2.0`, `MAX_CONDUCTANCE_GRID = (0.3, 0.4, 0.5)`, `PROPOSER_FALLBACK = "exhaustive"`;
  - `PRICE_SETPOINT = 2 * DELTA_MARGIN`, `PRICE_KP = 1 / PRICE_SETPOINT`, `PRICE_KI = 0.5 / PRICE_SETPOINT`;
  - `KAPPA_SE = 1.0`, `T2_MIN_PAIRS = 20`, `LEARN_HEADROOM_MIN = 1.5`, `LEARN_SHARE_MIN = 0.2`.
- **improve**:
  - The defaults of §21's options now come from config: `audit_band_ratio=AUDIT_BAND_RATIO`, `fallback=PROPOSER_FALLBACK`, `price_setpoint=PRICE_SETPOINT`, and `price_gains=None`, meaning `(PRICE_KP, PRICE_KI)`. The published behaviour is `audit_band_ratio=None, fallback=None, price_setpoint=0.0, price_gains=(1.0, 0.5)`.
  - New `hive_search(est, start, floor, proposer="exhaustive", communities=None, kappa_se=KAPPA_SE, max_rounds=50, max_seconds=None, rng_seed=0, **kw) -> (seeds, info)`, which is T2's option (b):
    - phase 1 is `improve(audit=False)` with `proposer`;
    - phase 2 is `improve(audit=True, disp_cap=kappa, return_family=True)` with `community_moves` proposals, from phase 1's result;
    - `kappa = D_a + kappa_se * verify.disparity_se(state_a)`;
    - `max_seconds` covers both phases, and `kw` goes to both `improve()` calls; passing `audit`, `disp_cap` or `return_family` in `kw` raises `ValueError`;
    - `info` has `phase1` and `phase2` (the two histories), `start1` (phase 1's actual start: the reference after a restart), `kappa`, `disp_a`, `se_a` and `communities`.
- **verify**: `disparity_se(state, z=1.0)` returns `sqrt(lam_max^2 + lam_min^2)`, the noise-model half-widths of the groups at the highest and the lowest rate, and 0.0 with fewer than 2 present groups.
- **certify**:
  - `path_states(start, history) -> [sorted seed lists]`: the start, then the set after every applied move. Pass the reference as `start` when `history[0]["fallback"]`.
  - `deploy_candidates(paths, n_mid=3) -> [sorted seed lists]`: the first and last set of every path, plus `n_mid` sets evenly spaced over the joined paths, without duplicates.
  - `certified_equal_spread(est_b, seeds, ref_seeds, rho, alpha=0.05) -> {"passed", "lcb", "estimate", "halfwidth", "lcb_rel", "estimate_rel", "halfwidth_rel", "identical"}`. This is the equal-spread judge for every method; `*_rel` values are divided by the reference's spread estimate on `est_b`.
- **diagnostics**: `learning_decision(measurements, grid=(16, 8), cost_ratio=1.0, improve_share=None, headroom_min=None, share_min=None) -> {"train", "H", "H_over_cost", "pi", "q", "c_round", "T_ex", "n_states", "reason"}`, the W3 rule over `measure_proposer` outputs (improvable states pooled).
- **proposer.LearnedProposer and train.V2Trainer**: both aim at `config.rr_floor(rho, total_ref)`, which is capped at `total_ref`. `LearnedProposer.floor(state)` returns it, and the trainer's dual ascent and validation use `rr_floor(rho, 1.0)` and `rr_floor(rho, total_ref)`.
- **imitation.simple_scores**: uses the scaled controller (`PRICE_KP`, `PRICE_KI`, `PRICE_SETPOINT`), as `improve()` does.
- **experiments/v2/run_t4.py**:
  - `--variants` takes {v2-b, v2-b0, v2-a}; v2-b is `hive_search`;
  - new options `--start {learned, greedy}`, `--deploy-mid`, `--cert-seed` (default 9001, which must differ from every RR seed that chose a set), `--cert-R` and `--cert-max-roots`;
  - new cached stage `cert_sample`;
  - FIM-GAT deploys `certified_choice` over `deploy_candidates`, and every method gets the `cert_*` columns from `certified_equal_spread`;
  - the Monte Carlo verify stream is gone, so the recorder's `es_*` columns carry the Monte Carlo test;
  - a greedy start with the simple proposer runs under `train_seed = -1`.
- **tests**: `tests/test_preregistration_v2.py` (marker `spec`, 15 tests). Two older test files now pass the published settings explicitly:
  - `test_theory_v2.py`: the Proposition 3 construction, which also scripts `audit_bands` and passes `fallback=None`;
  - `test_theory_v2_w.py`: W4.2, which passes `price_setpoint=0.0`.

## 23. Scripts and external baselines (2026-10-10; `docs/design_v2.md` §19). These override earlier sections where they differ.

The user delegated these choices before any v2 run.

- **config** adds:
  - `CERT_SEED = 9001`, `CERT_NNZ_BUDGET = 60_000_000` and `CERT_R_MAX = 512`;
  - `T2_SAMPLES = 4`, `T2_SAMPLE_SIZE = 100_000` and `EVAL_MIN_COMMUNITY = 10`;
  - `LB_MAX_WORK = 2e10`, `TUNING_FILE = "tuning.json"` and `DELTA_GRID = (0.0025, 0.005, 0.01, 0.02)`.
- **experiments/v2/pipeline_v2.py** (new; shared by `run_t1_t2.py`, `run_t4.py` and `tune_v2.py`):
  - `FimConfig(variant, start, proposer)`, with variant ∈ {v2-b, v2-b0, v2-a}, start ∈ {learned, greedy} and proposer ∈ {simple, exhaustive, learned}. It provides `.cols()`, `.needs()` and `.learned`.
  - `Case(loader, ks, args, dev, preset, need_features, need_rr_greedy, comms, phi0, delta, hook)`:
    - `.prepare()` runs the stages load, reference (JSON store), rr_sample, cert_sample, features, rr_greedy and cover_index;
    - `.ref_for(k)` returns `(ref, total_ref)`;
    - `.floor_for(k, rho)` is `rr_floor(rho, total_ref, delta)`;
    - `.judge(seeds, k, rho)` is `certified_equal_spread` on B;
    - `.base_cols()`, `.prep_for(method, cfg, model)` and `.describe()`.
  - `cert_config(graph, est, R, max_roots, args) -> (R_B, max_roots_B, rule)` implements the memory rule. `rr_config` and `tier_of` (samples are tiered by size) sit beside it.
  - `fim_query(case, k, rho, cfg, model, limit) -> dict` returns:
    - the start, final and deployed sets, with the judges of the deployed and final sets;
    - `select_s`, `improve_s` and `verify_s`, with per-stage `peaks`;
    - the search and deployment information.

    `baseline_query(case, k, rho, name, limit) -> dict` is its baseline counterpart.
  - **Recording:**
    - `record_fim` writes two RecorderV2 rows per query, deployment "certified" and "final"; `record_baseline` writes one.
    - `plan_queries(rec, ident, ks, rhos, cfgs, models, methods, tuned)` lists the pending queries. The resume keys include `delta` and `phi0`, so rows made under other tuned values never count as done.
    - Helpers: `fim_key`, `baseline_key`, `cert_cols`, `load_models`, `seeds_for`, `load_tuning`, `CsvLog`, `add_pipeline_args` and `check_args`.
- **experiments/v2/run_t1_t2.py** (new): `--test t1|t2`, plus `--datasets`, `--attributes`, `--ks`, `--rhos`, `--configs method ablation`, `--methods`, `--no-fim`, `--no-bounds`, `--t2-samples`, `--t2-size`, `--t2-communities all|top5000`, `--tuning`, `--untuned`, `--out`, `--bounds-out`, `--dry-run` and the pipeline options. It writes `results/v2/t1.csv`, `t2.csv` and `t1_bounds.csv`.
- **experiments/v2/tune_v2.py** (new): `--steps delta phi0 start w3`, plus `--delta-grid`, `--phi0-grid`, `--w3-states`, `--out` (default `results/v2/tuning.json`) and `--runs-out` (`tuning_runs.csv`). It runs only on `datasets.splits()["val"]`.
- **experiments/v2/summarize_v2.py** (new): `--dir` and `--out`. It writes `summary_v2.md`, `summary_tests.csv`, `summary_t1_cases.csv` and `summary_t4.csv`.
- **experiments/v2/run_t4.py** is rewritten on the pipeline:
  - `--variants`, `--start` (default: tuning's), `--proposer simple|learned`, `--tuning` and `--untuned`;
  - its stage log is `pipeline_v2.CsvLog`.
- **fimgat/v2/external.py** (new):
  - `bsm(est, k, floor, reference=None, rg=None, taus=TAUS, cache=None, return_info=False)`, with `bsm_tsgreedy(rg, k, tau, greedy_seeds, sat_seeds, opt_g)` and `saturate(rg, k, iters=12)`;
  - `multifw_maxmin(est, k, iters=20, mw_steps=100, rounds=20, seed=0, rg=None, return_info=False)`;
  - `eval_rates(rg, seeds) -> (rates, total)`, where rg is an `fimgat.baselines.RRGreedy` index.
- **fimgat/v2/baselines_v2.py:**
  - `SUITE` adds "BSM" and "MultiFW-MaxMin".
  - `run_suite` runs them on the suite's RR index. BSM's Greedy and Saturate parts are cached per k; MultiFW, which ignores ρ and the floor, is cached per k.
- **proposer.LearnedProposer:** `delta=DELTA_MARGIN`. Its floor is `rr_floor(rho, total_ref, delta)`.
- **tests:**
  - `tests/test_external_v2.py` (6 tests);
  - `tests/test_scripts_v2.py` (3 tests: the end-to-end chain on synthetic stand-ins, tuning refusing a test graph, and the T4 stage log).

  Both use the marker `spec`.

## 24. Reviewer-gap amendments (2026-10-10; `docs/design_v2.md` §20)

- **fimgat/v2/external.py** adds RFIM:
  - `rfim(est, k, partitions=None, rg=None, iters=12, seed=0, return_info=False)`, with `info` {"min_ratio", "c", "Q", "opt"};
  - `rfim_partitions(graph, seed=0, lp_iters=20) -> [(labels, alpha)]`: the labelled groups and a label-propagation partition, each with its inverse;
  - `label_propagation(graph, iters=20, seed=0)`;
  - `rfim_alpha(labels, n, rng)` (the three-level scheme), `rfim_inverse(alpha, rng)` and `rfim_weights(est, partitions) -> W (n_sets, Q)`.
- **fimgat/v2/baselines_v2.py:** `SUITE` adds "RFIM" (computed once per k; it ignores ρ and the floor).
- **experiments/v2/pipeline_v2.py:**
  - for v2-b, `fim_query`'s `search` holds `_communities`, the auditor's final family. Keys starting with "_" never reach a results row.
  - `baseline_key` takes every tuned value except `phi0`.
  - `run_t1_t2.py --cert-R` adds `cert_R` to the resume keys.
- **experiments/v2/interpret_v2.py** (new): `CASES` (pre-registered), `--case DATASET ATTRIBUTE K RHO`, `--t1`, `--tuning` and `--out-dir` (default `results/v2/cases`). It writes `groups.csv`, `communities.csv` and `case.md` for each case.
- **experiments/v2/summarize_v2.py:**
  - T1 Wilcoxon tables for the real-attribute graphs (the headline), the degree-group graphs and all graphs (`summary_tests.csv` gains `subset`);
  - the results matrix: `summary_t1_matrix.csv`, `summary_t1_matrix_by_k.csv`, a table per ρ, and certificate pass rates;
  - the certificate-size sensitivity table, from `t1_certsens.csv`.
- **docs/paper_notes.md** (new): reproducibility, limitations and ethics drafts.

## 25. Pre-run fixes (2026-10-14; `docs/design_v2.md` §21). These override earlier sections where they differ.

The user approved these before any v2 run.

- **config** adds:
  - `RHO_HOLDS = (0.97, 0.98, 0.99)`;
  - `DEPLOY_RULES = ("min_ucb", "path")` and `DEPLOY_TOL_SE = KAPPA_SE`;
  - `T2_RR_PER_NODE = 64`, and changes `T2_SAMPLES` from 4 to 6;
  - `T3_RHO = RHO_C1`, `T3_SPEEDUP_MIN = 1.5`, `T3_PRIMARY_BUDGET = 60` and `T3_RECALL_C = (8, 32, 128)`.
- **certify:**
  - `certify_sets(est_b, candidates, ref_seeds, rho, alpha=0.05) -> {"certificates", "delta_each", "n_candidates"}`. Every certificate also has `position` (its index among the candidates; -1 for an appended reference) and `disp_se` (the standard error of labelled disparity on B).
  - `choose(certs, rule="min_ucb", tol_se=None) -> index`. Rule "min_ucb" is Theorem 2's: the smallest disparity upper bound. Rule "path" picks the eligible certificate with the largest position whose bound is at most the smallest plus `tol_se` (default `DEPLOY_TOL_SE`) times that set's `disp_se`.
  - `certified_choice(..., rule="min_ucb", tol_se=None)` returns `rule` as well; its default behaviour is unchanged.
  - `deploy_candidates` returns its sets in path order, each by its last occurrence in the joined paths.
- **external:** `bsm_sweep(est, k, rg=None, taus=TAUS, cache=None) -> [(tau, seeds)]`, BSM-TSGreedy at every τ with no floor. BSM's Greedy and Saturate parts are cached under the same key as `bsm`.
- **imitation:** `train_proposer(..., on_the_fly=False)`. With `True`, a record without `dyn` gets its features at its step from `state_features` with the record's `mu` (0 when it has none).
- **experiments/v2/pipeline_v2.py:**
  - `DEPLOYMENTS = {"certified": "min_ucb", "certified-path": "path"}`.
  - `fim_query` certifies once and chooses by both rules. It adds `deployed_path`, `judge_path` and `deploy_path` to `deployed`, `judge` and `deploy`, which stay Theorem 2's. The deploy dicts also carry `rule`, `position`, `disp_se` and `n_eligible`.
  - `record_fim` writes three rows per query, `certified`, `certified-path` and `final`, in that order.
- **experiments/v2/run_t1_t2.py:**
  - T2 uses `R = T2_RR_PER_NODE` unless `--R` is given (printed as a deviation); R is in T2's resume key, and the run prints A's and B's memory.
  - T1 records BSM's sweep into `--pareto-out` (default `results/v2/t1_pareto.csv`) once per graph and budget, unless `--no-pareto` is given.
  - Sweep rows are RecorderV2 rows (method "BSM", variant `tau=<τ>`, ρ = `RHO_C1`) with `tau`, `b_ratio`, `sweep_s` and `cert_passed_<ρ>` for every ρ of `RHO_REPORT`.
- **experiments/v2/tune_v2.py:** `run_method(..., deployment="certified")`; the φ₀ step scores the `certified-path` rows.
- **experiments/v2/train_proposer.py** (new). Options:
  - `--seeds`, `--attributes`, `--val-attributes` and `--ks`;
  - `--states-per-query`, `--val-states-per-query`, `--epochs`, `--lr` and `--temp`;
  - `--hidden`, `--layers` and `--heads`;
  - `--R`, `--rr-seed`, `--val-rr-seed` and `--data-seed`;
  - `--checkpointing`, `--preset`, `--tuning` and `--constructor-pattern`;
  - `--models-dir`, `--log`, `--ignore-gate`, `--force` and `--dry-run`.

  It writes `models/proposer_s{seed}.pt`: `{"policy", "rounds", "mu_ref": {}, "hparams" (with "role": "proposer"), "log", "train_time", "meta"}`, loadable by `train.load_policy`. It also writes `train_proposer_log.csv`.
- **experiments/v2/run_t3.py** (new). Options: `--datasets`, `--attributes`, `--ks`, `--budgets`, `--proposers`, `--recall-states`, `--tuning`, `--untuned`, `--ignore-gate`, `--out` (`t3.csv`), `--recall-out` (`t3_recall.csv`), `--dry-run`, and the pipeline options.
  - `t3.csv` rows carry: variant "v2-a", deployment "search", `proposer`, `budget_s`, `reached_s`, `rr_disp`, `moves`, `search_s`, `local_optimum`, `trace` (JSON `[[secs, rr_disp], ...]`), `floor` and `total_ref_rr`.
  - `t3_recall.csv` rows carry: `start_seed`, `train_seed`, `proposer`, `n_states`, `n_improvable` and `recall_<C>`.
- **experiments/v2/summarize_v2.py:**
  - T1: the "holds" verdict uses `RHO_HOLDS` and names missing ρ; the ablation table gains the row "deployment: path rule".
  - T2: the primary comparison uses `certified-path`, with a pass line.
  - T3: a new section with the pre-registered rule.
  - T4: keeps only FIM-GAT's `certified` rows.
  - Pareto: new `pareto_points`, `dominance`, `pareto` and `pareto_figures`, writing `summary_pareto.csv`; `--figures` draws them into `figures/`.
- **tests:** `tests/test_prerun_fixes_v2.py` (marker `spec`). `tests/test_scripts_v2.py` now checks three rows per FIM-GAT query, the BSM sweep and the new summary sections.
