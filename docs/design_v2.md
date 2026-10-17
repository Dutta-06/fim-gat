# FIM-GAT v2 design

Status (2026-10-10): **pre-registered; the code and the experiment scripts exist; no v2 experiment has run.** This file records the design agreed with the user, the evidence behind each decision, and the pre-registered tests with their amendments. Each amendment is dated and kept in place, so the record shows what was decided when. `CLAUDE.md`, `docs/research_protocol.md` and `README.md` describe the current state.

**Update 2026-10-10.** The v2 core code now exists, and the spec tests pass. No v2 experiment has run yet, and the v2 datasets are not built. The integrity analysis R1–R3 (noise and the winner's curse, termination, the floor guarantee) is in `docs/theory_v2.md`. §16 lists its consequences. Part II of the theory notes (the weak spots W1–W4) adds the seven proposals of §17. Some of both were approved later the same day (below).

**Decisions, 2026-10-10.** Before any v2 experiment, the user approved the pre-registration amendment recorded in §18:
- T2 tests the ε-constraint phase;
- verified local search is part of the method, so C1 is judged against external baselines;
- learning is an accelerator, gated by the W3 rule;
- ρ = 0.99 for C1 and 0.97 for C2;
- certified deployment (§16 item 1) and §17 items 1, 2, 6 and 7.

§18 overrides the earlier sections where they differ, and they carry pointers to it.

**Delegation, 2026-10-10.** The user then delegated the remaining engineering and open methodology choices. §19 records the scripts, the settings decided, the external baselines and the analysis plan. §19 overrides earlier sections where they differ.

**Reviewer gaps, 2026-10-10.** §20 records what was added to close the gaps a TCSS review would likely raise: the RFIM port, case studies, the real-attribute headline, a sensitivity run, and the paper notes.

**Pre-run fixes, 2026-10-14.** A pre-run evaluation for TCSS found design flaws that would have biased tests or left them without a runner. The user approved fixing them ("make all the fixes so long as they are feasible and will not hurt this work"). §21 records them. Still no v2 experiment has run.

---

## 1. Goal and claims

v1 raised the worst-off group's coverage (maximin). v2 targets **disparity**, the gap between the best-served and worst-served group, while keeping total spread equal to the best efficiency-only method. It has to run fast enough to deploy on large real networks.

Claims to test, all fixed before any v2 run:

| # | Claim | Where it must hold |
|---|---|---|
| C1 | At **equal spread** (§2), lower disparity than every baseline that also reaches equal spread | all datasets, all budgets |
| C2 | Lower disparity on **real communities nobody labelled** (held-out ground-truth communities), through the hive-mind auditor | datasets with ground-truth communities or held-out attributes |
| C3 | **Faster end-to-end** than fair baselines of comparable quality, from a loaded graph to verified seeds | large tier only (§8). The user does not consider small-graph speed-ups meaningful. |

*Amended 2026-10-10 (§18):*
- *C1 is judged against external baselines. Local search and Simple-v2 are internal variants of the method (§9).*
- *C2 is tested through the ε-constraint phase, with ρ = 0.97 as the primary floor.*
- *C3 is claimed only for many fairness queries per graph, with per-graph and per-query costs reported apart.*

## 2. Definitions

- **Group rate** r_g(S): the fraction of group g reached by seed set S, as an expected value under Independent Cascade.
- **Disparity** D(S) = max_g r_g(S) − min_g r_g(S), over labelled groups.
- **Community disparity** D_C(S): the same quantity over a family of communities. §5 defines which family.
- **Reference spread** σ_ref(G, k): the best spread from an efficiency-only method on graph G with budget k. It is the best of at least 5 RR-Greedy runs, each on its own RR sample (IMM/OPIM once added), scored by forward Monte Carlo. It depends only on G and k, so it is computed once and reused by every fairness query.
- **Equal spread**: σ(S) ≥ ρ · σ_ref with 95% confidence, from a paired Monte Carlo comparison with common random numbers, checked **per graph and budget**. Headline ρ = 0.99 (**confirmed by the user on 2026-10-10**; C2's primary floor is 0.97, §18.1). Results are also reported at ρ ∈ {0.97, 0.98, 1.0} with full disparity-versus-spread curves. A win counts only if it holds across that range.
  - *2026-10-10 (§18.2): every method's equal spread is now judged by the RR certificate `certify.certified_equal_spread` on an independent RR sample. The Monte Carlo comparison above is still reported, for continuity with v1, but no longer decides.*
  - *2026-10-14 (§21.3): "across that range" means ρ ∈ {0.97, 0.98, 0.99} (`config.RHO_HOLDS`). ρ = 1.0 is still reported, descriptively, because almost only the reference itself can be certified there.*
- **User-facing setting:** ρ (the spread floor). β is retired (§6). ρ = 1 means pure spread; ρ → 0 means pure disparity.

## 3. Problem and its Lagrangian

  minimise D(S) (and D_C(S), §5)   subject to |S| = k and σ(S) ≥ ρ σ_ref.

With the epigraph form and multipliers, the value of seeding a node becomes a weighted sum of its effect on each group or community, with signed weights

  **w_g = μ n_g + β_g − α_g**

- β: a distribution over the most **under**-served groups (push them up).
- α: a distribution over the most **over**-served groups (push them down).
- μ ≥ 0: the price of spread, which rises when spread falls below the floor.

Consequences, which the design depends on:
1. v1's maximin objective is the special case α ≡ 0. With nothing pushing the top down, tiny groups get over-served (v1 disparity on SPA region: 0.46).
2. Negative weights turn the seed player's problem into a difference of submodular functions. Greedy's (1 − 1/e) guarantee no longer applies, and no simple method has one. That is the opening for search and learning.
3. Relaxing to fractional seeds tells us nothing. The LP relaxation's lower bound on disparity was exactly 0 in a scratch test. Methods that optimise fractional seeds and then round (FW-Fair, Tsang et al.'s MultiFW) get all their quality from rounding, and nothing can certify their disparity.
4. Plain dual ascent with a best-response seed player oscillates ("bang-bang"), and the balanced solution is a mixture of seed sets. The prices therefore have to be damped and should be used only to **guide** search, never to decide it (§6, §7).

## 4. Architecture

Per graph, cached and reused by every query: RR sample, static features, reference spread σ_ref(G, k).

Per query (group definition, k, ρ):

```
constructor (v1 policy, 3 rounds, retrained)  ──►  start set S0 (must meet the floor; else fall back to the reference set)
        │
        ▼
improvement rounds t = 1..T:
    auditor: find the most under- and over-served cohesive communities around S_t  (§5)
    prices:  α, β, μ  (predicted by the adversary network, refined by a PI controller)  (§6)
    proposals: exhaustive single swaps where affordable; otherwise pruned swaps
               + multi-node moves + whole-set proposals from the constructor  (§7)
    verifier: exact RR evaluation; accept only if the floor holds (with margin δ) and
              neither D nor D_C gets worse, and at least one of them improves  (§7)
        │
        ▼
final check: forward Monte Carlo on the few best verified sets; deploy the lowest-disparity
             set whose equal-spread test passes (the reference set always passes)
             [2026-10-10, approved (§16 item 1, §18.2): certify.certified_choice on an independent
              RR sample, over the start, the end and three states of the search path, replaces
              this; forward Monte Carlo stays the evaluation metric]
```

*2026-10-10 (§18):*
- *The improvement phase is `improve.hive_search`: the labelled-only search, then the ε-constraint phase, in which D may rise up to κ while D_C falls.*
- *The start set is the constructor's or the reference, whichever won on validation.*

Component mapping from v1:

| v1 component | v2 | Verdict |
|---|---|---|
| RR sampling, `ExposureEstimator` | training estimator; scores whole sets exactly (as the verifier does) | keep |
| `group_gains` (closed form) | "add" scores; add the exact **removal loss** of each current seed | keep + extend |
| — | incremental swap scoring: track, for each RR set, how many seeds cover it; a swap u→v only touches sets containing u or v | new |
| 9 static features | unchanged | keep |
| 12 dynamic features | deficit weights → signed prices; add each seed's removal loss, floor slack and ρ; drop β. Group features stay **relative**, so any group definition works | modify |
| GATv2 `SeedPolicy` | per-node score used as add score (non-seeds) and drop score (seeds); full encoder once per query, then a light head per round | keep + cheaper per-round head |
| `soft_topk`, 3 "replace" rounds | the constructor; whole-set proposals in every round | keep (new role) |
| sample-and-select decoding | becomes the verifier, with floor filter and disparity objective | keep (generalised) |
| J_β (spread + softmin) | smooth disparity-under-floor for training; exact disparity for verification; J_β kept only as a comparison variant | replace |
| `CommunityAdversary` (per-node λ, collapses on Pokec) | community-level two-sided auditor (§5) | replace |
| `ExposureCritic` | not needed | drop (or ablation only) |
| `finetune` | mostly redundant with the improvement phase | optional |
| ensemble | several models contribute proposals; the verifier chooses | keep |
| FW-Fair, greedy family, AGE, prior model, MC metrics | kept; efficient local search added as a **required** baseline | keep + add |

## 5. Hive-mind auditor (the adversarial core)

The auditor is part of **what is verified** (option (b), chosen by the user). The adversarial component therefore affects every accepted move, and its contribution is measured directly (test T2). It does not depend on whether the learned proposer pays off.

- **Two-sided.** The auditor finds (i) the most under-exposed and (ii) the most over-exposed cohesive community around the current seed set.
- **Cohesive and not tiny.** Each community must have conductance ≤ φ₀ and size ≥ m. The minimum size m is the fix for v1's collapse onto single nodes (max/mean weight up to about 18,000 on Pokec-z). It also avoids the tiny-community failure reported by Liu et al. (FAccT 2023). m and φ₀ are tuned on validation data only.
  - *2026-10-10:* m also has a statistical floor (theory Proposition 5). Resolving community-rate differences of ε at confidence z needs m ≳ 2z²·r(1−r)/(R·ε²), and twice that under the noise-aware rule. For example, m ≥ 326 at R = 16, r = 0.07, ε = 0.01. With m = 10 and R = 16, a community's rate has a standard deviation of about 0.02 at r ≈ 0.07 and 0.04 at r ≈ 0.35 (measured in theory §5.3), so its D_C decisions are mostly noise. The proposal is in §16, item 5.
- **Best response by local clustering.** Run a personalised-PageRank sweep over the exposure-deficit field (1 − a_v), or over a_v for the over-exposed side, keeping the best sweep cut that satisfies the size and conductance rules. Cost is O(edges touched), with Cheeger-type guarantees on cohesion.
- **Learned part.** An adversary GNN amortises the auditor. It predicts soft community memberships and the prices (α, β, μ) from graph structure and the current exposure, so the improvement phase starts close to the right prices instead of iterating from scratch. Iterating from scratch took about 100 rounds in a scratch test (§11).
- **Community family for D_C.** At round t, D_C is computed over the union of all communities the auditor has found so far in this query. That family only grows, so acceptance decisions are always compared on the same set of communities and the target doesn't move.
- *2026-10-10 (theory W1):* under the acceptance rule of §7 the auditor's communities can constrain moves but rarely drive them, and on planted communities the hive mind as designed did not lower hidden-community disparity. A single-scale audit also passes over small extreme communities. See §17, items 1–3.
  - *Approved 2026-10-10 (§18.1):*
    - *multi-scale audits are the default;*
    - *φ₀ is chosen from {0.3, 0.4, 0.5} on validation graphs, against held-out attributes;*
    - *T2's option (b) adds the ε-constraint phase, in which the auditor's communities drive moves.*
- **Leakage rule.** The auditor never uses the partitions used for evaluation: Louvain with seed 0 (`metrics.communities`) or any ground-truth community set. Held-out ground-truth communities (SNAP) and held-out attributes are the test of C2.

## 6. Prices (the "thermostat")

- The prices steer proposals; they never decide acceptance. The verifier decides.
- Update rule: a **PI controller**. The proportional term in each group's price scales with that group's distance from the centre, which damps the price as the group approaches it; the integral term keeps the spread floor in force. An augmented-Lagrangian penalty is the alternative if PI turns out unstable.
- Initial prices come from the adversary network (§5).
- β is retired as a model input; ρ and the current floor slack take its place.
- *2026-10-10 (theory W4):* during improvement every state meets the floor, so the PI controller's spread price is identically 0. Even with a setpoint, the gains (1, 0.5) are about 100 times too small next to the group prices. See §17, item 7.
  - *Approved 2026-10-10 (§18): setpoint-scaled gains are the default (`config.PRICE_SETPOINT`, `PRICE_KP`, `PRICE_KI`).*

## 7. Verifier and proposers

**Verifier (exact, on the RR sample):**
- Swaps are scored incrementally using the RR-set cover counts: cost O(RR memberships of u + v) per swap. Multi-node moves and whole sets use batch evaluation.
- **Acceptance rule:** σ ≥ (ρ + δ) σ_ref on the RR sample, D and D_C both no worse, and at least one strictly better.
  - *2026-10-10 (§18.1): in the ε-constraint phase of T2's option (b), D may rise up to κ = D_a + 1 SE when D_C strictly improves. (D_C, D) then falls lexicographically.*
- δ is a safety margin for the gap between RR and Monte Carlo spread. It is set from validation data as z × the standard deviation of that gap. On SPA it would have been about 0.75%.
  - *2026-10-10:* with certified deployment, δ no longer carries the guarantee. It only decides how often certification succeeds (`docs/theory_v2.md` §4). The rule for setting it is in §16, item 4.
- Quality never goes below the start set, and the result is valid whenever search is stopped.
- *2026-10-10:* the search never revisits a seed set (tabu, on by default). Without that, re-scoring D_C after family growth can make it cycle (theory Proposition 3).
- *2026-10-10:* optional noise-aware acceptance (`z`, `min_gain`) uses the certified gain of theory Lemma 5. It is off until tuned on validation. On a synthetic graph it removed every false move, but it did not lower the true disparity; the sample size mattered more (theory §5.1; §16, items 2–3).

**Proposers:**
- **Exhaustive single swaps** wherever affordable. We don't learn what is cheap to compute exactly.
- **Simple proposer** (the baseline for test T3): adds ranked by signed-price gain, drops ranked by smallest removal loss.
- *2026-10-10 (theory W3):* `improve(..., fallback="exhaustive")` (the hybrid) keeps exhaustive search's local-optimality guarantee. Without it, the simple proposer stopped before a local optimum in 40 of 54 synthetic runs. See §17, item 6.
  - *Approved 2026-10-10 (§18.3): the hybrid with scaled prices is the default and is T3's baseline. A learned proposer is trained only if the W3 rule says so.*
- **Learned proposer** (Stage 2), used where brute force breaks, i.e. at large scale:
  - (i) **pruning**: predict the few hundred nodes worth trying as adds or drops;
  - (ii) **warm start**: the constructor's start set, so fewer passes are needed;
  - (iii) **moves the simple proposer can't make**: multi-node moves and whole-set jumps.
- **Training the learned proposer:**
  - The constructor trains on the smooth disparity-under-floor Lagrangian (softmax − softmin of rates with signed prices, ρ sampled per step, backprop through one random round as in v1).
  - The pruning head imitates exhaustive local search. It is trained to rank highly the swaps that actually improved the result on training graphs.

## 8. Datasets

SPA is **dropped** from the v2 evaluation. That was decided before any v2 experiment, for reasons that don't depend on FIM-GAT: the SPA graphs are synthetic, and scratch analysis showed that plain local search already reaches the RR-estimation noise floor on them, so no method can be separated. Published v1 SPA results stay in `results/` as the record of v1.

| Tier | Graph | Size | Groups | Source |
|---|---|---|---|---|
| small | Rice-Facebook | 1,205 nodes, 42,443 edges | real: college, age, major | github.com/redfox-codder/fair_influen_max (data/) |
| small | Twitter-political | 3,753 nodes, 6,993 edges | real: political leaning (3) | github.com/ahmadkhajehnejad/CrossWalk (data/twitter) |
| medium | Pokec-z, Pokec-n | 67,435 / 66,082 nodes | real: region, gender, age | existing `data/pokec/*.npz` |
| medium | Epinions | 75,879 nodes, 508,837 edges | structural: degree groups, detected communities | SNAP soc-Epinions1 (raw copy in Baseline_Test/data, read-only) |
| medium | Twitter (ego) | 81,306 nodes, 1,768,149 directed edges | structural | SNAP ego-Twitter (raw copy in Baseline_Test/data, read-only) |
| large | Pokec (full) | 1.63M nodes, 30.6M edges | real: region, gender, age | existing `data/pokec/pokec_full.npz` |
| large | Instagram-Activities | 553,628 nodes, ~652.9k edges | real: gender | github.com/nicolaslanzetti/fairness-sim-ot (data/instagram_ds) |
| large | **SNAP com-Youtube** (chosen 2026-09-27) | 1,134,890 nodes, 2,987,624 edges | ground-truth communities: user-created groups; overlapping, and most users are in none (held out, for C2) | snap.stanford.edu/data/com-Youtube.html (`com-youtube.ungraph.txt.gz`, `com-youtube.top5000.cmty.txt.gz`, `com-youtube.all.cmty.txt.gz`) |
| large | LinkedIn | 6.73M nodes, 19.4M edges (line counts) | structural | Baseline_Test/soc-linkedin.edges (edges only; the `.nodes` file holds real profile handles and must not be copied) |

Rules:
- **Degree groups** are computed on the **full** graph before any sampling. (In the old 2,000-node LinkedIn BFS sample, 29% of nodes had degree 1, largely because of where the sample stopped.)
- Subgraph samples are reproducible (fixed seeds), with several per network, split into train / validation / test.
- `pokec_n`, every full graph, and every ground-truth community set are **never** used for training, tuning or checkpoint selection.
- **Validation (replaces SPA graphs 14–15):** held-out subgraph samples from each medium network, plus a separate Pokec district built from the raw dumps. Every hyperparameter (m, φ₀, δ, PI gains, T, pruning size) is tuned here only.
- **Diffusion model: weighted cascade everywhere** (decided 2026-09-27): p(u→v) = 1 / in-degree(v), with undirected graphs stored as two arcs. Uniform p isn't used. (Uniform p = 0.1 would be supercritical on Twitter: about 2 activations per reached user.)
- **Budgets: k ∈ {25, 50, 100, 150, 200} on every dataset and for every method** (decided 2026-09-27).
- **Why com-Youtube:** it is a real *social* friendship network, which suits influence spread better than DBLP (co-authorship). Its communities are user-created groups, so they are real. It has about as many people as full Pokec but about a tenth of the edges, so it fits 8–16 GB cards. com-LiveJournal (3,997,962 nodes, 34,681,189 edges; also social, with user-created groups) is the optional extension once the 40 GB machine is arranged. Because YouTube communities overlap and cover only part of the graph, evaluation must support overlapping membership and nodes that belong to no community.

## 9. Baselines

| Baseline | Why | Source |
|---|---|---|
| RR-Greedy; IMM / OPIM-C | spread reference and runtime reference | repo; github.com/tangj90/OPIM |
| Parity-, Welfare-, Maximin-Greedy | cheap fair baselines; Parity/Welfare win most cases at ρ = 0.99 today | repo |
| **Efficient single-swap local search** | the quality bar: 30–57% lower disparity than the above in scratch tests (§11) | new (incremental swap scoring) |
| **Simple-proposer v2** (same verifier, no learning) | isolates what the learning adds (test T3) | new |
| FW-Fair (Tsang-style Frank–Wolfe on our objective); Tsang et al. code | the non-learned ancestors of our approach | repo; github.com/bwilder0/fair_influmax_code_release |
| RFIM (worst case over given partitions) | closest label-free competitor | github.com/TianyouG/RFIM |
| BSM | constrained utility/fairness formulation | github.com/yhwang1990/code-bsm-release |
| DQ4FairIM | closest learned inductive competitor | github.com/akratiiet/DQ4FairIM |
| S3D (optimal-transport fairness) | recent NeurIPS baseline | github.com/nicolaslanzetti/fairness-sim-ot |
| AGE, CrossWalk, v1 prior model | learned-embedding baselines | repo; github.com/ahmadkhajehnejad/CrossWalk |

Baselines that cannot run at large scale are reported as such, with their time or memory limits.

*2026-10-10 (§18.2):*
- *Efficient local search and Simple-v2 are internal variants of the method, reported in the ablation table, not baselines for C1.*
- *Every baseline's equal spread is judged by the same RR certificate as FIM-GAT's.*

## 10. Pre-registered tests and decision rules

**Amended 2026-10-10 (§18).** These definitions replace T1–T4 below where they differ; the original text follows for the record.
- **T1 (C1).**
  - Disparity at equal spread, ρ = 0.99, against every external baseline (§9) whose set is certified at equal spread in that case.
  - Equal spread is judged for every method by `certify.certified_equal_spread`, on an RR sample B that no method saw.
  - FIM-GAT deploys by certified choice over its search path and falls back to the reference, so its deployed set always carries the certificate. The rule that a failed case counts as a loss can therefore no longer trigger; the cost of certifying shows up as disparity instead.
  - FIM-GAT's start set (constructor or greedy) is fixed on validation.
  - Also reported: ρ ∈ {0.97, 0.98, 1.0}, the curves, the Monte Carlo equal-spread test, and the ablation table of §18.2.
- **T2 (C2).**
  - Option (b) (`hive_search`) against option (a); option (b0), the published rule, is reported as well.
  - Primary floor ρ = 0.97, secondary 0.99. At least 20 paired cases.
  - Metrics, pass rule and fail rule are as below. "Without a significant rise in labelled-group disparity" is tested on Monte Carlo disparity, with the same paired Wilcoxon test.
  - *2026-10-14 (§21.1–§21.2): the primary comparison uses the path rule's deployments, the same rule for every variant. The samples' search sample has 64 RR sets per node, and there are 6 samples.*
- **T3.**
  - The baseline is the simple hybrid with scaled prices.
  - The learned proposer is trained and tested only if the W3 rule of §18.3 says so on large-tier validation graphs. Otherwise T3 reports the measured headroom and "not run under the pre-registered rule".
  - *2026-10-14 (§21.4): the training script, the T3 runner and the operational pass rule now exist.*
- **T4 (C3).**
  - Per-graph (cached) and per-query costs for every method, and the time for Q ∈ {1, 10, 100} queries per graph.
  - Certification counts as per-query time, and its sample B as a cached stage. `experiments/v2/run_t4.py` implements this.

**Original (2026-09-27):**

All tests use all test graphs, at least 5 training seeds for learned variants, and paired two-sided Wilcoxon tests over test graphs of seed-averaged values at α = 0.05.

- **T1 (claim C1).** Disparity at equal spread (ρ = 0.99, confidence test per case) against every baseline that meets equal spread in that case. A case where FIM-GAT fails equal spread counts as a loss. Also report ρ ∈ {0.97, 0.98, 1.0} and the curves.
- **T2 (hive mind, claim C2).** Verifier option (b) against option (a), where (a) checks labelled-group disparity only.
  - Primary metric: disparity over **held-out** ground-truth communities.
  - Secondary metrics: labelled-group disparity and held-out-attribute disparity.
  - Pass: (b) lowers the primary metric without a significant rise in labelled-group disparity.
  - Fail: report it, and ask the user how to frame the adversary. Do not drop it.
- **T3 (Risk 1: does the learning add anything?).** On the large tier, with the same verifier and the same wall-clock budgets (for example 10 s, 60 s and 300 s per query):
  - Compare the learned proposer, the simple proposer, and exhaustive local search where it finishes.
  - Report disparity reached against time, and recall@C of improving swaps.
  - Pass: the learned proposer reaches the simple proposer's final disparity in clearly less time, or lower disparity in the same time, on most test graphs.
  - Fail: report it and drop the learned proposer. Do not reframe the claim afterwards.
- **T4 (claim C3).** End-to-end runtime on the large tier. Report the per-graph cached cost (RR sampling, reference spread) and the per-query cost separately for every method. Baselines are measured the same way.

## 11. Evidence behind the design (scratch analyses, 2026-09-27)

Throw-away scripts in the session scratchpad, not archived in the repo; the key numbers are here. SPA test graphs 16–23, k = 25, floor 0.99 × Greedy, independent Monte Carlo, unless stated otherwise.

- **Headroom exists.** Single-swap local search at the floor cut disparity relative to the best baseline meeting the floor (graphs matched so both sides meet it) by 37% (ethnicity), 30% (age), 37% (status), 57% (gender) and 6% (region). The gains held on fresh samples (disparity 0.0334 on the optimised RR sample, 0.0361 on an independent one, 0.0337 in MC).
- **Granularity floor.** Region's 2–5-person groups keep disparity near the mean reach rate (about 0.12) for any method.
- **Relaxations are uninformative.** The LP lower bound was 0. An exact MILP found no feasible solution in 60 s at n = 500.
- **Plain thermostat fails.** Dual ascent with an exact greedy best response met the floor in 1/40 cases after 3 rounds and 31/40 after 100. When it met the floor, disparity was 2–12× worse than the baselines.
- **Rotation over floor-feasible sets gains little.** An LP mixture of sets for repeated campaigns gained −22% (ethnicity), −9% (age), and nothing on status or gender, and each individual campaign was unfair (disparity 0.03–0.09). Not adopted.
- **Multi-node moves add little on small graphs.** Destroy-and-repair (2–3 nodes) improved only 0–2 of 8 graphs per attribute. Single-swap local search already reaches the noise floor on SPA. The learned proposer's case therefore rests on large-scale time-to-quality (T3).
- **Noise.** Greedy reruns fell below 99% of the reference Greedy 12% of the time (hence the best-of-5 reference in §2). RR-versus-MC spread differed by 0.5% (s.d.), hence the margin δ.
- **Sample size (Pokec-n region, k = 50).** Every method kept its quality from R = 32 down to R = 2 RR sets per node, while prep time fell from 26 s to 1.8 s. FIM-GAT is not uniquely sample-efficient. v1 FIM-GAT reached only 94–96% of Greedy's spread, below the floor.

## 12. Docs to update after approval

Done on 2026-10-10. `CLAUDE.md`, `docs/research_protocol.md` and `README.md` describe v2. The v1 versions and the v1 record were archived outside the repo (`Desktop\FIM-GAT_archive\removed_2026-10-10\`).

## 13. Open decisions (for the user)

1. The equal-spread threshold. **Decided 2026-10-10:** ρ = 0.99 for C1 and 0.97 for C2 (§18).
2. Go-ahead for Stage 1 code (§14). **Done:** the v2 code dates from 2026-09-27/28 and 2026-10-10.
3. The six proposals of §16, which follow from the theory notes R1–R3 (2026-10-10). **Item 1 approved 2026-10-10 (§18); items 2–6 decided under the user's delegation (§19).**
4. The seven proposals of §17, which follow from the weak-spot analysis W1–W4 (2026-10-10). **Items 1, 2, 6 and 7 approved 2026-10-10 (§18); items 3–5 decided under the user's delegation (§19).**

Decided on 2026-10-10: the pre-registration amendment of §18 (Decisions 1–3 and their dependencies).

Decided on 2026-09-27:
- weighted cascade everywhere;
- SNAP com-Youtube;
- k ∈ {25, 50, 100, 150, 200} everywhere;
- the 16 / 40 GB machine is deferred ("we'll see to later"), so the large tier runs on 8 GB presets until then.

## 14. Stages

0. **Stage 0: test suites (done, 2026-09-27; §15).** Smoke and stress tests of the v1 foundation, and v2 specification tests, all written before any v2 code.
1. **Stage 1 (reuses most of v1).**
   - Data loaders and splits (§8).
   - Disparity-under-floor objective with signed prices.
   - Community-level auditor.
   - Exact verifier with incremental swap scoring.
   - Simple proposer.
   - Efficient local-search baseline.
   - v1 policy retrained as the constructor.
   - Runs T1 and T2 on the small and medium tiers.
2. **Stage 2.** Learned pruning and multi-node proposals, and predicted prices. Runs T3.
3. **Stage 3.** Large tier: full Pokec, Instagram, the SNAP graph, LinkedIn. Runs T3 and T4, on the 16 GB machine where needed.

Compute for each stage is measured at the start of that stage. Any run expected to take hours is confirmed with the user first.

## 15. Test suites

The suites were written before any v2 code (Stage 0, 2026-09-27) and have grown with it. `tests/README.md` lists them, with how to run them and their current counts.

What the smoke suite establishes against **exact** ground truth (every live-edge world enumerated on tiny graphs, and autograd):
- the RR estimator and Monte Carlo are correct, including under weighted cascade;
- the closed-form group gains and objective gradient match autograd;
- soft top-k keeps the budget exactly, and its gradients never change it;
- features don't change when groups are relabelled, the basis of "any group definition";
- the policy is equivariant to node permutations, and chunked inference matches full-graph inference.

The stress suite's Stage 0 finding, the v1 `metrics.evaluate` crash when no Louvain community has 10 members, is guarded. Both `metrics.evaluate` and v2's `metrics_v2.evaluate_v2` report NaN community metrics in that case.

## 16. Consequences of the theory notes (2026-10-10; item 1 approved, items 2–6 decided under delegation, §19)

`docs/theory_v2.md` analyses three weak points of the design. Its §5 checks each one against a high-precision ground truth on a synthetic graph.
- **Noise (R1).** The verifier's numbers are exact on one RR sample, and the search selects on that same sample.
  - With 8 RR sets per node, the plain rule reported about twice the real disparity gain, and 7 of its 23 accepted moves made the true disparity worse.
  - With 512 RR sets per node, the estimate was right to within 3·10⁻⁴.
- **Cycling (R2).** Re-scoring D_C after the auditor adds a community can make the search return to a set it left. This is proven by an explicit construction. Tabu, now on by default, prevents it. With thresholds and an audit cap, the number of moves is bounded.
- **The floor (R3).** Meeting ρ + δ on the search sample does not imply the true floor: the final set was clearly below it in 6 of 90 runs. A certificate on an independent RR sample is valid by theorem and had no failures in 270 checks.

Already in the code, with defaults that keep the published behaviour (except tabu):
- `certify.py`;
- noise-aware acceptance (`Z_ACCEPT = 0`, off);
- tabu (on);
- thresholds and an audit cap (off).

Interfaces are in `v2_interfaces.md` §20, and the tests in `tests/test_theory_v2.py`.

Proposed changes, each needing the user's approval:

| # | Change | Evidence (`theory_v2.md`) | Cost or risk |
|---|---|---|---|
| 1 | Deploy with `certify.certified_choice` on an independent RR sample B, instead of the Monte Carlo equal-spread test (§4, final check). Candidates are the start set, the final set and a few states along the path. Monte Carlo stays the evaluation metric. | Theorem 2. §5.2: 0 failures in 270 checks (165 distinct set–sample pairs). At R_B = 512 it certified 36 of 46 distinct sets that truly met the floor; the Monte Carlo test passed 19. | One more RR sample per query. When nothing is certified, the reference is deployed, which costs fairness but never breaks the floor. A small B certifies little (R_B = 64: 11 of 46), so B is sized per tier on validation queries. |
| 2 | Size the search sample first. Measure on validation queries how the held-out final disparity and floor safety change with R at each tier, and raise `RR_PER_NODE` where memory allows. | §5.1: with the plain rule, the true final disparity was 0.0152 at R = 8 and 0.0122 at R = 64 and 512. Floor violations stopped by R = 512. | Memory and time grow linearly in R. |
| 3 | Validate `Z_ACCEPT` ∈ {0, 1}, choosing on held-out (B) disparity, never on A's estimate. | §5.1: z = 1 removed every worsening move and most of the phantom gain, kept nearly all of the true gain, and did not lower the true disparity. | None for the default. z = 2 is too strict at small R. |
| 4 | Set δ per tier from δ ≳ (t_B + 2√Q̂_B + 2√Q̂_A)/σ_ref, measured on validation queries. It is 0.25% now. | §4 remarks. §5.2: about 0.35–0.55% on the study graph. | A larger δ leaves less room for fairness. |
| 5 | Set `MIN_COMMUNITY` per tier from m ≳ 2z²r(1−r)/(Rε²) instead of 10, doubled under the noise-aware rule. Score the auditor's communities on RR sets the auditor did not use. | Proposition 5. §5.3: at m = 10 and R = 16, a community's rate has a standard deviation of 0.02–0.04. | Fewer, larger communities. Held-out scoring needs a second RR sample, which can be B. |
| 6 | Report the certificate with every deployed set: eligible or not, the floor bound, the disparity bound, and whether the reference was deployed. | §4. | Reporting only. The disparity bound is valid but loose, so it is reported next to the Monte Carlo disparity, not instead of it. |

Not proposed:
- certifying D_C, which the same machinery allows but which is not built yet;
- thresholds `min_gain` > 0 by default, because tabu and `max_rounds` already guarantee termination.

The adversary's role is unchanged by all six proposals. Proposal 5 changes only the auditor's minimum community size and the sample its communities are scored on.

## 17. Consequences of the weak-spot analysis W1–W4 (2026-10-10; items 1, 2, 6 and 7 approved, items 3–5 decided under delegation, §19)

`docs/theory_v2.md` Part II (§8–§14) analyses the design's four weak spots, with proofs and a synthetic numerical study (`experiments/v2/weakspots.py`, outputs in `results/v2/weakspots/`). In short:
- **W1, claim C2.** As designed, the hive mind did not lower hidden-community disparity in any setting of a planted-community study: the acceptance rule lets it constrain the search but rarely drive it. An ε-constraint phase lowered hidden disparity by 10.5% of its mean (p = 0.005; the median case by 3.1%) at ρ = 0.97 on cohesive communities, without raising labelled disparity.
- **W2.** The problem is NP-hard, even under weighted cascade with ρ near 1. A computable lower bound certifies near-optimality only when hard-to-reach groups outnumber the seeds.
- **W3, claim C3.** A proposer with an exhaustive fallback keeps the local-optimality guarantee and is at most 1/(1 − π) times faster. On 32,000-node graphs it was 1.2× faster (1.7× with scaled prices). A learned proposer's headroom over that hybrid was 1.0–2.1 for k ≤ 50, and up to 4.9 at k = 100.
- **W4.** The spread price is identically zero during improvement, and the PI gains are two orders of magnitude too small to matter. Setpoint-scaled gains fix both (p ≤ 0.031).

Already in the code, all off by default until §18 (interfaces in `v2_interfaces.md` §21, tests in `tests/test_theory_v2_w.py`). Since §18, multi-scale audits, the exhaustive fallback and the scaled prices are on by default:
- multi-scale audits;
- ε-constraint acceptance and community-price proposals;
- the mixture of campaigns;
- diagnostics: the transfer bound, the lower bounds, the proposer cost model and the PI stability check;
- the exhaustive fallback;
- the price setpoint and gains;
- `return_family` and `audit_max_size`, which were already specified.

Proposed changes, each needing the user's approval:

| # | Change | Evidence (`theory_v2.md`) | Cost or risk |
|---|---|---|---|
| 1 | Verifier option (b) for T2 becomes the labelled-only search followed by the ε-constraint phase, with κ = the labelled optimum plus about one noise unit. The published rule is still reported, as option (b0). Run T2 at ρ = 0.97 as well as 0.99. The pre-registration is amended before T2 runs. | W1.5. §9.6: −10.5% of the mean hidden disparity (median case −3.1%), better in 13 of 20 cases, p = 0.005, labelled disparity unchanged (p = 0.58). (b0) had no effect in any setting. | Relaxes "never make either worse": labelled disparity may rise by up to κ − D*. |
| 2 | Multi-scale audits (band ratio 2) become the hive mind's default. φ₀ is tuned on validation data against the cohesion of the target communities. | W1.2. §9.6: 19 against 4 communities per run. Communities with conductance 0.25 were invisible at φ₀ = 0.3. | More communities, so more D_C constraints and more auditor time (still O(edges touched)). |
| 3 | Report Lemma W1.1's terms in T2: D_F, the two gaps, and the Jaccard of the matches to the hidden extremes. | W1.1; §9.6. | Reporting only. |
| 4 | Offer the mixture of campaigns only as an explicit ex-ante mode, with a certificate per campaign (Theorem 2). | W1.3–W1.4. §9.6: ex-ante disparity −8% to −11% of the mean in three of four settings, but campaigns are individually less fair, and 5–6 of 20 mixtures deployed a campaign below the true floor. | A product decision; not a headline. |
| 5 | Report `bounds.disparity_lower_bound` and the gap to it in every T1 table. | W2.3; §10.1. | Reporting only. The bound is 0 for a few large groups. |
| 6 | T3's simple-proposer baseline becomes the hybrid with scaled prices. Apply the W3 decision rule on large-tier validation graphs before training a learned proposer. | W3.1–W3.2. §11.1: the hybrid reached a local optimum in 54 of 54 runs; headroom for learning is 1.0–2.1 for k ≤ 50 and grows to 3.5–4.9 at k = 100 on 32,000 nodes. | A stronger baseline makes T3 harder to pass, honestly. |
| 7 | A setpoint-scaled spread price for the simple proposer: setpoint about δ–2δ, k_p = 1/setpoint, k_i = 0.5/setpoint, tuned on validation data. Give `LearnedProposer` and the trainer the same capped floor as `improve` (`config.rr_floor`); at ρ = 1 they aim above it. | W4.2, W4.4. §12.1: floor-breaking proposals fell from 93–95% to 57–86%, and final disparity improved (p ≤ 0.031). | A limit cycle is possible (W4.4), but harmless for correctness (W4.1). |

Not proposed:
- making the mixture the default;
- any change to the adversary's role. Proposal 1 strengthens it, because the auditor's communities can then drive moves, not only constrain them.

## 18. Decisions of 2026-10-10 (pre-registration amendment)

The user approved these decisions on 2026-10-10, before any v2 experiment ran and before the v2 datasets were built. Where this section and an earlier one differ, this section wins; the earlier text stays, with a pointer here.

The decisions fix how the contribution is positioned: certified fair influence maximisation against an adversarial community auditor, with the auditor as the headline. Claims come in three layers:
1. guaranteed by proof (the floor certificate, termination, the hardness and bounds of `theory_v2.md`);
2. C1, a strong empirical claim against external baselines;
3. C2 and C3, pre-registered and reported whichever way they come out.

### 18.1 Decision 1: T2 tests the ε-constraint phase

- **Option (b) is `improve.hive_search`.**
  - Phase 1 is the labelled-only search, identical to option (a).
  - Phase 2 starts from (a)'s result and runs the ε-constraint phase (theory Proposition W1.5): community-price proposals, multi-scale audits and the exhaustive fallback.
  - The cap is κ = D_a + 1 · SE(D_a) (`config.KAPPA_SE`, `verify.disparity_se` on the search sample). Labelled disparity can therefore rise above (a)'s result by at most about one standard error, and only through moves that strictly lower D_C.
- **Option (b0)**, the published rule (`improve(audit=True)`: neither D nor D_C may rise), is reported next to (a) and (b) as a secondary comparison.
- **Floors.** ρ = 0.97 is the primary floor for C2 (`config.RHO_C2`); ρ = 0.99 is reported as secondary.
- **φ₀**, the auditor's conductance cap, is chosen from {0.3, 0.4, 0.5} (`config.MAX_CONDUCTANCE_GRID`) on validation graphs, scored by disparity over held-out attributes. SNAP ground-truth communities are never used for tuning or selection.
- **Sample size.** T2 needs at least 20 paired cases, counted as test subgraphs × budgets (`config.T2_MIN_PAIRS`). If the test graphs with held-out structure give fewer, more test subgraph samples are drawn before T2 runs, never after results are seen.

### 18.2 Decision 2: verified local search is part of the method

- **C1 is judged against external baselines only.** The complete FIM-GAT pipeline is compared with FW-Fair / MultiFW, the greedy family, RFIM, BSM, DQ4FairIM, S3D and CrossWalk, plus AGE and the v1 prior model where they run. Efficient local search and Simple-v2 become internal variants (§9).
- **T1's start set is fixed on validation.** It is the constructor's set if that beat the greedy start on validation disparity at equal spread, with the same search and deployment. Otherwise it is the greedy start, the reference set itself.
- **Ablation table (main text).** Each internal variant differs from the method in one component, and every one is reported at every T1 budget and floor with T1's columns:

  | Component | Method | Variants reported |
  |---|---|---|
  | Start | T1's start | the other start (constructor or greedy) |
  | Proposer | simple hybrid (or learned, Decision 3) | exhaustive local search; simple hybrid if the learned proposer is used |
  | Acceptance | (b), the ε-constraint phase | (a) labelled only; (b0) the published rule |
  | Deployment | certified choice over the path | the final search set, undeployed (its Monte Carlo disparity and floor violations) |

- **One judge for equal spread.** Every method's equal spread is judged the same way: `certify.certified_equal_spread`, the paired floor bound of Lemma 6 at α = 0.05, on an RR sample B that no method saw (one per graph).
  - FIM-GAT deploys with `certify.certified_choice` (Theorem 2) on B, over `certify.deploy_candidates` of its search path: the start, the end, and three states between. The choice's union bound implies the per-set certificate.
  - The paired Monte Carlo test of §2 is still computed and reported for continuity with v1, but it no longer decides.

### 18.3 Decision 3: learning is an accelerator; the adversary is the headline

- **The W3 rule gates training.** Before any learned proposer is trained, `diagnostics.learning_decision` runs on large-tier validation graphs, on states along the simple hybrid's search path, at the 16 × 8 grid.
  - Train only if the headroom H / c ≥ 1.5 (`config.LEARN_HEADROOM_MIN`), where c is the learned proposer's per-round cost relative to the simple proposer's (c = 1 until a prototype is timed).
  - The improvement phase must also take at least 20% of per-query time (`config.LEARN_SHARE_MIN`; Amdahl).
  - If either condition fails, the headroom is reported, no proposer is trained, and T3 is reported as not run under this rule.
- **T3's baseline** is the simple proposer with the exhaustive fallback and setpoint-scaled prices: the defaults of `improve()`. The learned proposer runs with the same fallback.
- **C3 covers many fairness queries per graph only.** Every method's per-graph (cached) cost and per-query cost are reported separately, with the time for Q queries, prep + Q × query, at Q ∈ {1, 10, 100}. No single-query speed-up is claimed.

### 18.4 Dependencies approved with the decisions

- ρ = 0.99 for C1 (`config.RHO_C1`, which closes §13 item 1) and ρ = 0.97 for C2.
- §16 item 1: certified deployment, with B sized per tier on validation queries.
- §17 items 1, 2, 6 and 7:
  - the ε-constraint phase for T2;
  - multi-scale audits as the default;
  - the hybrid as T3's baseline, with the W3 rule;
  - setpoint-scaled prices, and the capped floor for the learned proposer and the trainer.

Still open at the time: §16 items 2–6 and §17 items 3–5. They were decided later the same day under the user's delegation (§19).

### 18.5 Operational details

These were chosen while implementing the decisions and can be changed until the first v2 run, not after.

| Setting | Value | Where |
|---|---|---|
| ε-cap κ | D_a + 1 standard error | `config.KAPPA_SE` |
| T2 paired cases | at least 20 | `config.T2_MIN_PAIRS` |
| φ₀ grid | 0.3, 0.4, 0.5 | `config.MAX_CONDUCTANCE_GRID` |
| W3 thresholds | H / c ≥ 1.5; improvement ≥ 20% of query time | `config.LEARN_HEADROOM_MIN`, `LEARN_SHARE_MIN` |
| Deployment candidates | start, end, 3 states between | `run_t4.py --deploy-mid` |
| Certificate sample B | the search sample's size until sized on validation (§18.7); its own RR seed, 9001 | `run_t4.py --cert-R`, `--cert-max-roots`, `--cert-seed` |
| Q for C3 | 1, 10, 100 | analysis |

SE(D) is a conservative scale. On the synthetic block graphs of `tests/test_preregistration_v2.py`, the standard deviation of D across 200 independent RR samples was 0.53–0.90 of the mean SE (4 graphs, R = 4 and 16). κ therefore allows a rise of about one to two real standard deviations.

### 18.6 What changed in the code

- **`improve()` defaults** now come from config:
  - multi-scale audits (`AUDIT_BAND_RATIO = 2`);
  - the exhaustive fallback (`PROPOSER_FALLBACK`);
  - setpoint-scaled prices (`PRICE_SETPOINT` = 2δ, k_p = 1/setpoint, k_i = 0.5/setpoint).

  The published behaviour is still available as `audit_band_ratio=None, fallback=None, price_setpoint=0.0, price_gains=(1.0, 0.5)`.
- **New functions:** `improve.hive_search`, `verify.disparity_se`, `certify.path_states`, `certify.deploy_candidates`, `certify.certified_equal_spread` and `diagnostics.learning_decision`.
- **Capped floor.** `LearnedProposer` and `V2Trainer` aim at `config.rr_floor`, which never exceeds the reference total, as `improve()` does. `imitation.simple_scores` uses the scaled controller.
  - The learned proposer's own price controller keeps the published gains, because no proposer has been trained. §17 item 7 asks to check the same scaling for it; that check happens only if the W3 rule leads to training one.
- **`experiments/v2/run_t4.py`:**
  - variants v2-b, v2-b0 and v2-a;
  - `--start`;
  - the cached stage `cert_sample`;
  - certified deployment for FIM-GAT;
  - the certificate as every baseline's deploy-time check.

  An end-to-end check on a 1,500-node synthetic graph ran every configuration: greedy and learned start, simple and learned proposer, all three variants, and four baselines.
- **Tests.** `tests/test_preregistration_v2.py` adds 15 tests. Three older tests now pin the published settings they are about: W4.2, and the cycling construction of Proposition 3, whose scripted auditor now also covers the multi-scale call.

### 18.7 What B's size and δ do to certified deployment (settle before T1)

Certified deployment deploys a search state only if its spread clears ρ σ_ref by more than the certificate's half-width on B. The search itself stops at (ρ + δ) σ_ref on its own sample, with δ = 0.25%. The guarantee holds whatever B and δ are, because the reference never breaks the floor. How much fairness FIM-GAT keeps, however, depends on both.

A what-if on the 1,500-node synthetic graph of the `run_t4.py` check (`hive_search` from the greedy reference; k ∈ {10, 20, 40}, search sample R_A ∈ {8, 64}, ρ ∈ {0.97, 0.99}; Monte Carlo with 2,000 runs) gave:

| B (RR sets per node) | δ = 0.25% | δ = 1% | δ = 2% | Mean disparity reduction against the reference (range over the 3 δ) |
|---|---|---|---|---|
| 8 (the search sample's size) | 1 of 12 deployments not the reference | 1 of 12 | 4 of 12 | 5–19% |
| 64 | 6 of 12 | 10 of 12 | 11 of 12 | 41–58% |
| 512 | 8 of 12 | 9 of 12 | 9 of 12 | 57–63% |

- No deployed set was below the true floor by more than Monte Carlo noise (0 of 108). 2 of 108 were below ρ by less than 0.005.
- The final search set itself was below the true floor in 2–3 of 6 cases with R_A = 8, and in 0–2 of 6 with R_A = 64. This is the winner's curse of R3, which the certificate catches.
- In the end-to-end check (R = 8 for both samples), FIM-GAT deployed the reference in all 12 queries. The certificate also rejected LocalSearch's sets in all 4 queries; each one's Monte Carlo spread was below its floor (ratio 0.951–0.988).

B's size matters most, and sizing it per tier on validation queries is part of the approved §16 item 1. δ matters second; it is §16 item 4, still open. Both should be settled on validation queries before T1. `run_t4.py` prints a note while B is left at the search sample's size. This is one synthetic graph, so these numbers illustrate the effect and do not set any value.

## 19. Decisions under the user's delegation (2026-10-10)

On 2026-10-10 the user said that building validation datasets seemed redundant, since datasets can simply be downloaded later. They asked for the experiment scripts first, and gave authority to "transform this codebase into what we need". The choices below were made under that delegation, before any v2 experiment ran and before the v2 datasets were built. The user can review or reverse any of them until the first v2 test run; after that, none may change. Where this section and an earlier one differ, this section wins.

### 19.1 The scripts

| Script | Role |
|---|---|
| `experiments/v2/pipeline_v2.py` | The per-graph pipeline that every script shares (the `Case` class). It holds the reference, the search sample A, the certificate sample B, the cover index and the learned parts' context. It runs a FIM-GAT query (start → search → certified deployment) or a baseline query, judges every set with the same certificate, and records the rows. |
| `experiments/v2/tune_v2.py` | Validation tuning on `pokec_t` only, in this order: δ, then φ₀, then the start set, then the W3 gate. It writes `results/v2/tuning.json`, refuses any graph outside the validation split, and records every query in `tuning_runs.csv`. |
| `experiments/v2/run_t1_t2.py` | T1 runs every test graph × attribute × budget × ρ, with the method, the ablation rows and the baseline suite. T2 runs on com-Youtube samples. Both read `tuning.json`; `--untuned` exists for development only and marks its rows `tuned=False`. |
| `experiments/v2/run_t4.py` | T4, rewritten on the pipeline. Its queries are T1's, logged stage by stage with memory. |
| `experiments/v2/summarize_v2.py` | The pre-registered analysis (§19.4). It writes `summary_v2.md` and the `summary_*.csv` files. |

The run order is: build the datasets, train the constructor, tune, run T1, run T2, run T4, then summarise. The constructor training is optional; without checkpoints the start is greedy. `docs/research_protocol.md` §1 has the commands.

### 19.2 Settings decided

- **δ (was §16 item 4) is tuned on validation.** The grid is `config.DELTA_GRID` = {0.25%, 0.5%, 1%, 2%}. The method runs at ρ = 0.99, and the score is the Monte Carlo disparity of its deployed sets; ties go to the smaller δ. Every method that searches against the RR floor uses the same δ (`Case.floor_for`).
  - *Why:* with certified deployment, a small δ leaves the search room, but its final sets miss the certificate and the reference is deployed. A large δ certifies but leaves no room. Only data can place that trade-off.
- **The size of the certificate sample B follows a memory rule, not tuning** (`pipeline_v2.cert_config`).
  - B may hold up to `config.CERT_NNZ_BUDGET` = 60M RR-set entries, about 22 bytes each in RAM (measured on `pokec_z`).
  - Roots grow first, up to every node, because with sampled roots the certificate's units are roots. RR sets per root grow next, up to `config.CERT_R_MAX` = 512. B is never smaller than A.
  - *Why:* a larger B can only add certifying power, and the guarantee holds at any size (§18.7).
- **Left at their current values:**
  - the search sample's R per tier (§16 item 2);
  - `MIN_COMMUNITY` = 10 (§16 item 5; the ε-phase evidence of W1 used it);
  - `Z_ACCEPT` = 0 (§16 item 3).
- **Reporting** (§16 item 6, §17 items 3 and 5): the more complete choice in each case.
  - Every set's certificate is recorded (`cert_*` columns), and FIM-GAT's deployment fields too (`deploy_*`).
  - Lower bounds go to `t1_bounds.csv` whenever affordable (`config.LB_MAX_WORK`). Their floor is ρ times the reference's Monte Carlo spread minus 2 standard errors.
  - Lemma W1.1's terms are not computed: rows carry the family size and the RR community disparity only.
  - The mixture of campaigns (§17 item 4) is not used.
- **T1's attributes:** every real attribute of each test dataset, and degree groups for the structural graphs (`run_t1_t2.T1_ATTRS`).
- **T2's sample:**
  - 4 BFS samples of com-Youtube, 100,000 nodes each, with sample seeds 1–4 (`config.T2_SAMPLES`, `T2_SAMPLE_SIZE`). Each sample's tier follows its size.
  - The metric uses every user-created community with at least `config.EVAL_MIN_COMMUNITY` = 10 members inside the sample.
  - The primary comparison uses the deployed sets; the final search sets are reported as the mechanism.
- **The W3 gate runs on `pokec_t`,** a medium-tier graph, because the split has no large-tier validation graph. `tuning.json` records this.

### 19.3 External baselines

From the feasibility review of the six repositories (licences, entry points, scale):

| Baseline | Status | Why |
|---|---|---|
| BSM (Wang et al., EDBT 2024) | **in the suite** (`external.bsm`) | BSM-TSGreedy with Saturate, written from the paper on our RR sets (the code is Apache-2.0; it was read, not copied). The adapter takes the largest τ in {1.0, 0.9, …, 0.1} whose set meets the RR floor on A, else Greedy's set. |
| Tsang et al. MaxMin (IJCAI 2019) | **in the suite** (`external.multifw_maxmin`, "MultiFW-MaxMin") | The multiobjective Frank-Wolfe with an inner multiplicative-weights LP and dependent rounding, written from the paper. The released code has no licence, and its group objective counts only one hop of spread. |
| RFIM (Gao & Ito, SIGMOD 2026) | **pending** | MIT-licensed C++ that needs GCC. This machine has no C++ compiler, and WSL holds only Docker's internal distribution. It needs a WSL Ubuntu install or a gcc container. |
| S3D (NeurIPS 2024) | not implemented | Two groups only, with a uniform edge probability. It would have to be ported from the paper. |
| CrossWalk (AAAI 2022) | not implemented | Small graphs only (about 3,500 nodes, k-medoids over all node pairs), with GPL code. |
| DQ4FairIM | reported as not runnable | The released code imports a simulator module that never existed in its repository, and it builds a dense n × n matrix. |

### 19.4 The analysis (`summarize_v2.py`)

- **T1:**
  - **Unit:** a test graph, meaning a dataset under one attribute. Its value is the mean over budgets of (method − baseline) disparity, using only budgets where the baseline's set is certified.
  - **Test:** a paired two-sided Wilcoxon test at α = 0.05, for each ρ.
  - **Headline:** against the best certified external baseline of each case. "Holds across the range" means a significant win at every ρ in `config.RHO_REPORT`.
  - **Ablation rows** are compared with the method the same way.
- **T2:**
  - **Unit:** a case, meaning one sample at one budget.
  - **Pass:** (b) − (a) on the hidden-community disparity of the deployed sets, with two-sided p < 0.05, a negative median, at least 20 cases, and no significant rise in labelled disparity.
  - **Also reported:** (b0), and the final search sets.
- **T4:** for each graph and method, the cached seconds, the per-query seconds, and prep + Q × query at Q ∈ {1, 10, 100}, next to disparity and the certificate pass rate.

### 19.5 Checks and environment

- **Tests:**
  - `tests/test_external_v2.py` (6 tests) checks the ports against brute force: the incremental gains, the truncated cover gain, Saturate's bound, BSM's constraint and τ choice, and the rounding's marginals.
  - `tests/test_scripts_v2.py` (3 tests) runs tune → T1 → T2 → summarise, and T4, end to end on synthetic stand-ins, including resumption and the refusal to tune on a test graph.
- **Host memory:** this laptop has 15.5 GB of RAM, but only about 1–3 GB was free during development, with other applications open. Two development runs were killed by host out-of-memory errors. Real runs, especially on the large tier, need those applications closed or the 16 / 40 GB machine.

## 20. Closing the gaps reviewers would flag (2026-10-10, under the delegation)

When asked whether the work would suit IEEE TCSS, the review listed five gaps reviewers would likely raise. The user asked for them to be closed now. Everything below was done before any v2 experiment ran, so the choices are pre-registered.

| Gap | What was done |
|---|---|
| RFIM, the closest label-free robust competitor, was missing | `external.rfim`, written from the paper (Gao & Ito, SIGMOD 2026; the MIT-licensed C++ needs GCC, which this machine lacks). It is S-HIST's Saturate over Q weighted-coverage objectives on our RR samples, with exactly k seeds (the paper's `-sratio=1` setting) and OPT_q from greedy / (1 − 1/e). It is in the suite as "RFIM" and tested in `tests/test_external_v2.py`. |
| No social-systems interpretation | `experiments/v2/interpret_v2.py`, a case study that shows who gains and who loses (reach of every group of every attribute, labelled and held out, under the reference, the method and the best certified baseline), and what the auditor found (each community's size, conductance, most over-represented group per attribute, and reach). |
| Four test graphs have only degree groups | The headline T1 test uses the real-attribute graphs; the degree-group graphs are a supplement, and all graphs together are reported too (`summarize_v2.py`). |
| Sensitivity | A certificate-size sensitivity run on a fixed subset (below). Floors and budgets are already in every table. |
| Reproducibility, ethics and limitations | `docs/paper_notes.md`: a reproducibility statement and a limitations and ethics draft. The licence for a code release is the user's decision and is still open. |

**RFIM's partitions in our setting.** RFIM is given four partitions, mirroring the paper's Q = 4 construction:
- our labelled groups, in place of the paper's ground truth;
- a label-propagation partition;
- the "inverse" of each, where each community's α becomes 1 − α with probability 0.75.

The α values follow the paper's three-level scheme. No Louvain or Leiden partition is used, because the community metrics are evaluated on Louvain, and giving it to a method would leak. RFIM ignores the spread floor, like Tsang et al.'s MaxMin, and is judged by the same certificate as every method.

**Case studies, pre-registered.** `interpret_v2.CASES` covers every T1 graph with a real attribute, at k = 50 and ρ = 0.99, using the method's smallest train seed: 11 cases. Any other case is marked "not pre-registered" in its output. The script reruns the method's query and reports whether it reproduced the recorded set.

**Certificate-size sensitivity, pre-registered.** `rice_facebook` / college and `pokec_n` / region, at ρ = 0.99 and every budget, for the method and every external baseline. B is set to 16, 64 and 256 RR sets per node (`--cert-R`), and the runs write to `results/v2/t1_certsens.csv`; protocol §1 has the commands. Rows record `cert_R`, and the resume keys include it.

**The results matrix.** `summarize_v2.py` writes one table per ρ, with a row per test graph and a column per method. Each cell is the mean Monte Carlo disparity over the budgets where that method was certified, shown with that number of budgets. A lower-bound column comes from `t1_bounds.csv`. A per-budget CSV and the certificate pass rate of each method complete it. Bold marks the lowest cell among methods certified at every budget; the paired tests remain the formal comparison.

## 21. Pre-run fixes (2026-10-14, approved by the user)

A pre-run evaluation for an IEEE TCSS submission (2026-10-14) found problems in the pre-registered rules:
- some rules would bias a test toward a null result;
- one rule made its test unwinnable;
- one test had no runner.

The user approved the fixes: "make all the fixes so long as they are feasible and will not hurt this work". No v2 experiment had run, so the fixes are pre-registered. None may change after the first v2 test run. T1's headline is unchanged: the method, its deployment and its tests.

### 21.1 T2's deployment: the path rule

- **The flaw.** `certify.certified_choice` deploys the eligible candidate with the smallest *labelled*-disparity upper bound (Theorem 2's rule).
  - For option (b), the candidates include phase 1's end, which is the labelled optimum.
  - The ε-constraint phase may raise labelled disparity by up to κ while it lowers D_C.
  - So the rule usually redeploys phase 1's end, and (b)'s deployed set mostly equals (a)'s.
  - T2's primary comparison uses the deployed sets, so it was biased toward "no difference" whatever the auditor does.
- **The fix.** A second deployment rule, `certify.choose(..., "path")`. Among the eligible candidates, it deploys the one **furthest along the search path** whose labelled-disparity upper bound is within `config.DEPLOY_TOL_SE` (= `KAPPA_SE`, 1) standard errors (on B) of the smallest bound.
  - It is the deployment analogue of κ.
  - It keeps what the search's later states bought, unless B says they cost more than about one standard error of labelled disparity.
- **Validity.** Both rules choose among eligible sets, so Theorem 2's floor guarantee holds for either (theory §4, remark of 2026-10-14). The deployed set's disparity is at most its own upper bound.
- **Where it is used.**
  - Every FIM-GAT query records three rows:
    - `certified`: Theorem 2's rule, which is T1's deployment, unchanged;
    - `certified-path`: the path rule;
    - `final`: the undeployed search set.
  - T2's primary comparison uses `certified-path` for (b), (a) and (b0) alike. Theorem 2's deployments and the final sets are secondary.
  - T2's held-out-attribute secondary metric uses the T1 path-rule rows.
  - `tune_v2.py` scores φ₀ on the path-rule rows, because φ₀ acts only through the ε phase.
  - The T1 ablation table gains a row "deployment: path rule".
- **Mechanics.**
  - `certify.certify_sets` computes each candidate's certificate once, with its `position` on the path and its `disp_se`.
  - `choose` applies a rule.
  - `certified_choice(..., rule="min_ucb")` keeps its old behaviour by default.
  - `deploy_candidates` returns the candidates in path order, each set at its last occurrence.

### 21.2 T2's sample: R and sample count

- **R = 64** RR sets per node for the T2 samples' search sample A (`config.T2_RR_PER_NODE`).
  - The W1 evidence for the ε phase used R = 64.
  - The 100,000-node samples would otherwise get the medium tier's R = 16. There, a 10-node community's rate has a standard deviation of about 0.04 (theory Proposition 5, §5.3), so the auditor's small communities would be mostly noise.
  - `--R` overrides it. Any other value is printed as a deviation and must be reported.
  - R is part of T2's resume key.
- **Memory.** A and B are bound by host RAM: about 22 bytes per entry, and B is never smaller than A. The script prints both sizes. Run T2 with other applications closed, or on the 16 GB machine.
- **6 samples** (`config.T2_SAMPLES`) give 30 cases instead of exactly the minimum 20, so one failed case no longer makes T2 underpowered.

### 21.3 "Holds across the range"

The criterion of §2 is now a significant win at every ρ in `config.RHO_HOLDS` = {0.97, 0.98, 0.99}.
- At ρ = 1.0 the certificate requires σ(S) ≥ σ(S_ref) with confidence. Almost only the reference itself passes, for the method and every baseline alike, so requiring a win there made the criterion unwinnable by construction.
- ρ = 1.0 stays in every table (`RHO_REPORT`), reported descriptively.
- `summarize_v2.py` now prints the verdict for every comparison, and names any ρ without paired cases instead of omitting the line.

### 21.4 T3: the runner and the operational rule

The W3 gate (§18.3) could say "train", but no script trained a proposer or ran T3. Added:

- **`experiments/v2/train_proposer.py`.**
  - **Gate:** it runs only if `tuning.json` says `"w3": {"train": true}`. `--ignore-gate` is for development only and writes under `results/v2/quick/`.
  - **Graphs:** training on the train split under region, gender, age and degree; validation on the validation split under region, gender and degree.
  - **States:** `imitation.generate_imitation_data` along exhaustive local-search trajectories:
    - from the greedy-family starts that meet the floor;
    - at every budget and ρ ∈ {0.97, 0.99}, at `improve()`'s acceptance floor with the tuned δ;
    - with a quarter of the moves random improving swaps.
  - **Features** carry spread price 0. That is what `LearnedProposer`'s own controller gives while the floor holds (theory W4.2).
  - **Model:** it starts from the constructor checkpoint of the same seed when one exists, else fresh, and is trained with `imitation.train_proposer` (features on the fly). The epoch with the best validation recall@128 is kept; the starting policy can win.
  - **Output:** `models/proposer_s{seed}.pt` in the `train.load_policy` format. It also records the simple proposer's validation recall and the learned proposer's measured cost ratio.
- **`experiments/v2/run_t3.py`.** Same gate, and it also needs trained checkpoints.
  - **Scope:** the large-tier test graphs under T1's attributes, every budget, ρ = `config.T3_RHO` (0.99), and the tuned δ and start.
  - **Searches:** the labelled-only search (v2-a), run once per proposer up to the largest of `config.TIME_BUDGETS` (10, 60, 300 s), with no round limit. The proposers are:
    - the simple hybrid, T3's baseline;
    - each learned proposer, with the same exhaustive fallback;
    - exhaustive search.
  - **Scoring:** the set reached by each budget is read off the search's own timed history and scored by forward Monte Carlo (`t3.csv`, with the RR-disparity trace).
  - **Recall:** recall@C for C ∈ {8, 32, 128} is measured on states along the simple path (`t3_recall.csv`).
- **Pass rule** (`summarize_v2.py`). It makes §10's wording operational: "clearly less time, or lower disparity in the same time, on most test graphs".
  - **(i) Speed.** For each case, the speed-up of the learned proposer in reaching the simple hybrid's final RR disparity is read from the traces. A learned run is paired with the simple run from the same start.
    - A test graph is *faster* when the median over budgets of the median over train seeds is at least `config.T3_SPEEDUP_MIN` = 1.5, the W3 threshold.
    - (i) holds when more than half of the test graphs are faster.
  - **(ii) Same-time quality.** At `config.T3_PRIMARY_BUDGET` = 60 s, a paired two-sided Wilcoxon test over test graphs of the mean over budgets of (learned, seed-averaged − simple) Monte Carlo disparity. (ii) holds at p < 0.05 with a negative median.
  - T3 passes if (i) or (ii) holds. A fail drops the learned proposer, and the claim is not reframed (§10).
- **`imitation.train_proposer`** gains `on_the_fly=True`, which computes features per step instead of keeping an n × 14 block per state. The whole chain ran for the first time on synthetic graphs (`tests/test_prerun_fixes_v2.py`).

### 21.5 The spread–disparity trade-off (reporting)

Nine of the thirteen suite methods ignore the spread floor, and BSM, RFIM and MultiFW-MaxMin optimise maximin rather than disparity. At ρ = 0.99 most of them fail the certificate and drop out of T1's comparisons, which a reviewer could read as a win by construction. Added, as description only:
- **`external.bsm_sweep`:** BSM-TSGreedy's set at every τ, with no floor.
  - `run_t1_t2.py --test t1` records it once per graph and budget in `t1_pareto.csv`.
  - Each row has the Monte Carlo spread ratio and disparity, B's spread ratio, and the certificate at every ρ of `RHO_REPORT`.
  - `--no-pareto` skips it.
- **`summarize_v2.py`:** per test graph, it sorts the external sets (T1 rows and BSM's sweep) against the method's curve, which is its deployments at every ρ. Each set is dominated by the method's curve, dominates one of its points, or is incomparable.
  - `summary_pareto.csv` holds every point.
  - `--figures` draws one spread–disparity figure per test graph at k = 50.
  - The paired tests of T1 remain the formal comparison.

### 21.6 A reporting bug fixed

`summarize_v2.py`'s T4 table averaged FIM-GAT's deployed and final rows together. Its per-query time therefore understated certification, and its disparity mixed the deployed and the undeployed set. It now keeps FIM-GAT's Theorem 2 deployments only.

### 21.7 What this does not change

- T1's method, deployment, baselines and tests; the certificate; the search; the auditor; every tuned setting.
- The adversary's role. §21.1 and §21.2 give it a fair test; they do not change what it does.
- Not done, because each needs a download, a design change or the user's decision:
  - running CrossWalk's released code;
  - a DQ4FairIM reimplementation;
  - S3D;
  - IMM/OPIM-C as a runtime reference;
  - a second diffusion model.

## 22. The C2 null and its post-hoc diagnostics (2026-10-17; exploratory)

**The pre-registered result stands.** T2 failed by its own rule (cloud job ALL, `results/v2/summary_v2.md`). On six 100,000-node samples of com-Youtube, option (b) deployed the same set as option (a) in all 60 cases. The ε-constraint phase may raise labelled disparity by at most κ − D_a = 1 SE, which was about 0.0004 against labelled disparities of 0.18–0.46, and it accepted no move.

**The user's framing decision (2026-10-17).** Report C2 as a finding about what the method cannot do, and why, rather than only as a claim that could not be made. Three conditions apply:
- the pre-registered verdict is stated first;
- the mechanism is shown with data and labelled post hoc;
- the explanation is tested.

**The evidence already in hand** (descriptive):
- Across the 224 runs below ρ = 1, the phase changed the final set in 0% of the runs in the lowest fifth of room (κ − D_a) and in 56% of the highest fifth. The rank correlation is 0.42 (p = 5·10⁻¹¹).
- The median room was 0.0025 where the phase acted and 0.0007 where it did not.
- On the full com-Youtube and Instagram graphs it almost never acted despite 3–5 times T2's room. That is consistent with an auditor that finds no cohesive community there (theory §9.6, item 7).
- The cross-graph pattern is confounded: graph size drives both the room and the density of communities.

**The diagnostics, fixed before they run** (`experiments/v2/c2_diagnostics.py`; tested in `tests/test_c2_diagnostics.py`, where the control reproduces T2's deployed sets exactly). They use T2's 60 states: samples, budgets, floors, R = 64, the tuned δ and φ₀.
1. **Room (the intervention).**
   - What runs: phase 2 exactly as `hive_search` runs it, from the same phase-1 end, with κ = D_a (1 + f) for f ∈ {1%, 5%, 20%}. The pre-registered κ = D_a + 1 SE is the control.
   - Scoring: each run is deployed by T2's path rule on B and scored by forward Monte Carlo, paired with (a)'s deployment.
   - Prediction: the phase accepts moves once it has room.
2. **Coverage.**
   - Lemma W1.1's terms for every deployed set: D_H, D_F, the gap bounds from the hidden extremes h⁺ and h⁻ to the auditor's family, and their best Jaccard overlaps.
   - Per sample, the conductance of every evaluated ground-truth community against φ₀.
   - This separates "no room" from "wrong communities".
3. **Price.**
   - At the labelled-only optimum, floor-feasible single swaps are scanned in increasing labelled cost (at most 5,000 of them).
   - It reports the cost of the cheapest swap that lowers community disparity: for the auditor's first-audit family, and for the hidden communities on sample A.
   - It also reports the most a single swap can buy within labelled-cost budgets of one SE and of 1%, 5% and 20% of D_a.
   - This is the exchange rate between labelled and community fairness that κ decides whether to pay.

**Reading, fixed in advance.** The outcomes map onto findings as follows:

| Outcome | Finding |
|---|---|
| The phase acts with room, and hidden-community disparity falls | the method can protect unlabelled communities at a stated, measured price |
| It acts, but hidden disparity does not fall | coverage is the bottleneck; the auditor's communities miss the hidden extremes |
| It stays inert even with 20% room | no community-improving moves exist at the labelled optimum on these graphs |

A relative cap gives no room when D_a ≈ 0. On T2's samples D_a is 0.18–0.46.

**Reporting.** Every result is exploratory. Paired tests are reported descriptively, and no number here replaces T2's verdict. A confirmatory test of a scaled cap would be a new pre-registered test on new samples.

**Not used for any choice.** Ground-truth communities only score and price, as in T2.

**Outcome (2026-10-18; cloud job c2diag, `results/v2/parts/c2diag/summary.md`; exploratory).** The run is valid: the control runs reproduced T2's deployed sets in 120 of 120 rows for (a) and 120 of 120 for (b-1se). No fixes or deviations.
- **Room does not make the phase act.**
  - At κ = D_a(1 + 20%), phase 2 accepted moves in 7 of 30 cases at ρ = 0.97 and 3 of 30 at ρ = 0.99.
  - The deployed set differed from (a)'s in 3 of the 60 cases.
  - The median change in hidden-community disparity is 0 at every room.
- **Price rules out row 3 of the reading.** Community-improving moves exist, and they are cheap.
  - In every case, a floor-feasible single swap lowers the hidden-community disparity (in-sample, on A) at a labelled cost of 0.9–1.4 SE (about 0.1% of D_a).
  - Within 5% of D_a, one swap lowers it by a median 14% (ρ = 0.97) and 8% (ρ = 0.99).
  - For the auditor's own first family, a lowering swap within 1 SE of labelled cost existed in 53% and 50% of the cases, yet the phase accepted none at the pre-registered κ: its proposals did not offer those swaps.
- **Coverage is the bottleneck (row 2).**
  - The auditor's family of 7–8 communities barely overlaps the hidden extremes: the best Jaccard with h⁺ is at most 0.006, and with h⁻ the median is 0.001–0.002.
  - Lemma W1.1's bound is vacuous in 100% of cases.
  - The ground-truth communities have median conductance 0.97, and none lies within φ₀ = 0.4. The φ₀ grid tops out at 0.5, so no choice of φ₀ would admit them.
- **Reading.** The hive-mind auditor protects structurally cohesive communities (conductance ≤ φ₀). com-Youtube's user-created groups are not cohesive, so the auditor cannot see the groups that set the hidden disparity. Its ε phase also leaves affordable improvements to its own family untaken.
- **Caveats.**
  - One graph family (60 cases).
  - The hidden-community price is in-sample, on sample A.
  - The extreme groups are small (median 12 members for h⁻, 43 for h⁺).
  - T2's pre-registered null stands.

## 23. GPU runs on Kaggle (2026-10-17, under the delegation)

The user chose Kaggle's free notebooks (GPU T4 x2) for the GPU part and runs each job by hand. `kaggle/run_job.py` defines the jobs, and `kaggle/README.md` is the user's guide. The choices below are operational: nothing pre-registered changes.

- **One preset.** Every pipeline script runs at `--preset 8gb`.
  - *Why:* the cloud made every row at 8gb (the preset a CPU machine detects). The preset sets the large tier's search sample A (R = 8, 150,000 roots), and it is not part of a row's resume key, so another preset would silently mix two samples within one graph.
  - *Cost:* a 16 GB T4 is under-used, so T4's timings are those of the 8 GB preset on a T4, which is conservative.
  - The trainers run at 8gb too. There the preset only switches activation checkpointing, which is on for `pokec_z` under either preset.
- **Constructor training on two GPUs.** Seeds 0–2 train on one GPU; seeds 3–4 and the no-adversary ablation (seed 0) train on the other. Each model's training is unchanged (seeds, data, steps). The two processes share the CPUs, so the training wall times in the checkpoints were measured under that sharing.
- **The timing tests run alone,** one process on one GPU.
  - T4 computes its references afresh, because they are one of its timed cached stages.
  - T3 reuses T1's stored references, because its clock times only the search. A reference key names the graph, attribute, sample sizes and seeds, so a matching key holds the same values.
- **CPU and GPU sessions.**
  - Full Pokec's T1 rows that need no constructor run in CPU sessions (jobs C1), which never load a checkpoint.
  - The learned start's rows run afterwards in a GPU session (job G2), as the missing rows of the same job folders.
  - Resume keys exclude the hardware, and every row records its machine (`gpu`), as before.
- **The cloud's job folders.** G2 writes the learned start's rows into the cloud's folders (`results/v2/parts/t1-*`), so each graph's rows stay in one file and no query runs twice. If the start step picks the constructor, G2 also reruns T2 (whose method rows then change) and the eight case studies made with the greedy start.
- **Session limits.** A Kaggle session ends at 12 h. The runner starts no command after 11 h, and it stops a running command in time to save the output. Every script resumes, so a second session continues the job.
- **Outputs.** A job's zip holds only files under `results/` and `data/cache/` that differ from the bundle. `kaggle/import_outputs.py` merges results CSVs by query key, so two outputs that extend one file lose no rows and count none twice.
- **Progress bars.** The long scripts show a progress bar with an ETA (`common_v2.progress`, tqdm), one line a minute in a log, which the runner relays every 5 minutes. It is output only and changes no result, but it changes `code_tag`, as the C2 diagnostics script did. So the Kaggle rows carry a different `code_tag` from the cloud's, and neither change touches a computation.
- **Not bundled:**
  - LinkedIn: its node file holds personal data. Dropping it, or preparing an edges-only build, is the user's call.
  - Raw downloads, and the raw Pokec dumps.

## 24. Constructor fixes after Kaggle G1 (2026-10-17, approved by the user)

The first GPU run, Kaggle job G1 (attempt 1, `results/v2/attempts/G1-attempt1/`), trained the constructor as specified: five seeds and the no-adversary ablation, 40 epochs each. It then ran the start step. Two mechanical failures made that step compare greedy with greedy:

- **Decoding discarded the constructor's sets.** `select_seeds_v2` returned RR-greedy whenever no candidate met the floor exactly. At ρ = 0.99, with floor min(ρ + δ, 1) × reference, no candidate ever did: training reached 0.96–0.98 of the reference. Every validation query fell back, and so did all 25 learned-start queries of the start step.
- **Checkpoint selection could not rank epochs.** A fallback scored as the greedy set's disparity, so every validation epoch scored −0.0423, and the strict rule kept epoch 5 of 40.

The user chose to fix both and rerun, keeping attempt 1 on record. No test row had used the constructor, so these are validation-stage changes. The rerun is a second look at the start rule, so the paper reports both attempts and the change between them.

**The fixes:**
1. **Repair before fallback** (`train.repair_to_floor`, and `select_seeds_v2` with `n_repair = config.REPAIR_CANDIDATES`). When no candidate meets the floor:
   - the 4 candidates with the largest RR totals are each repaired on sample A by spread-gaining swaps until they meet it;
   - each repair step pairs the 16 seeds that lose the least spread with the 16 non-seeds that add the most, and applies the largest exact gain among the 16 most promising pairs;
   - the lowest-disparity repaired set becomes the start;
   - greedy stays the fallback when repair fails.

   Rows record `select_repaired` and `repair_swaps`, so the share of each start that repair replaced is reported.
2. **Selection that cannot tie on failures** (`V2Trainer.validate` and `fit`). A fallback scores as a failure (−1, like a set below the floor), and an exact tie keeps the later epoch. Training logs also record `val_repaired`, the share of repaired validation queries.

**Unchanged:**
- the training objective, data and schedule;
- the validation graphs and floor;
- the start rule: the constructor's start must have a strictly lower mean validation disparity than greedy's;
- everything that runs from the greedy start.

The learned proposer is retrained as well, because it starts from the constructor checkpoints.

**The rerun** is Kaggle job `G1b-train`, with the same phases. Its output alone feeds G2–G4, and attempt 1's output is ignored. Attempt 1's decision is kept in `tuning.json` under `attempts`, and its rows are kept in its archive folder. If the repaired constructor still loses, the start stays greedy and both attempts are reported.
