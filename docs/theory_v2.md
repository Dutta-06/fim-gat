# FIM-GAT v2: theory notes (R1–R3, W1–W4)

Status: written 2026-10-10, after the v2 design (`docs/design_v2.md`) and the v2 core code.

**Part I (§0–§7)** covers the three integrity questions agreed with the user:
- **R1. Noise and the winner's curse.** Does the verifier accept improvements that are only sampling luck?
- **R2. Termination.** Can the improvement loop cycle, and does it always stop?
- **R3. The spread floor.** Does a deployed seed set really reach ρ × the reference spread?

**Part II (§8–§14)** covers the design's weak spots: hidden communities (W1), lower bounds and hardness (W2), the learned proposer (W3) and the price thermostat (W4). Its summary is §8.

Every result is stated with its assumptions. A proof follows it, or it is marked as a heuristic. Each one is backed by an executable check:
- Part I: `tests/test_theory_v2.py` and the numerical study `experiments/v2/theory_checks.py`, with outputs in `results/v2/theory/`;
- Part II: `tests/test_theory_v2_w.py` and `experiments/v2/weakspots.py`, with outputs in `results/v2/weakspots/`.

Both test files run with the spec suite.

---

## 0. Summary

**R1, noise.** Every number the verifier uses is exact *on one random sample* of reverse-reachable (RR) sets. The search then picks, among many candidates, the one that looks best on that same sample, so it partly selects luck. There are two mechanisms:
- **Multiplicity.** Picking the best of M candidates adds up to about τ√(2 ln M) of phantom gain per move, where τ is the noise of one move's *change*. Every candidate is scored on the same RR sets (common random numbers), so τ comes only from the few RR sets a move touches. That makes τ much smaller than the noise of the rates. But M is large (about 5·10⁴ swaps per round on the study graph), so the term is not negligible.
- **Near-ties.** When two groups are within sampling noise of each other at the top or bottom, the search can "improve" disparity by lifting whichever one *looks* lowest. The other, truly tied group stays where it is, so the true disparity does not change.

On the study graph both mechanisms are of the same order (Proposition 4), and §5 does not separate them.

The fix is a **certified gain** (Lemma 5): a provable lower bound on the true improvement, given half-widths for the rates and for the move's change. It equals the plain gain when there is no noise.

A related limit is **resolution** (Proposition 5). A sample of N units per group cannot resolve disparity differences below about z·√(2r(1−r)/N). The noise-aware rule compares intervals, so it needs twice that N. This has a direct consequence for the hive mind: with R = 16 RR sets per node, a 10-node community's rate has a standard deviation of about 0.02–0.04, comparable to or larger than the disparities being fought over. Communities need hundreds of nodes, or the sample needs more RR sets.

Measured against ground truth (§5.1):
- With 8 RR sets per node, the plain rule reported nearly twice the real gain, and 7 of its 23 accepted moves made the true disparity worse.
- With z = 1 there were no such moves at any sample size, and nearly all of the real gain was kept.
- The noise-aware rule does **not** lower the true final disparity. What it buys is honesty and safety.
- The sample size mattered more than the choice between z = 0 and z = 1. With the plain rule, the true final disparity was 0.0152 at R = 8 and 0.0122 at both R = 64 and R = 512. Every floor violation came from R ≤ 64.
- Gains are not comparable across R, because each R starts from a different set, chosen on its own sample.

**R2, termination.**
- The improvement loop keeps the spread floor and never increases labelled disparity on its sample (Proposition 1).
- Within one community family it never revisits a seed set (Proposition 2).
- When the hive mind adds a community, the community disparity is re-scored, and the loop **can** return to a set it left. `tests/test_theory_v2.py` builds an explicit example that the pre-2026-10-10 code follows (Proposition 3).
- A no-revisit rule (tabu, now the default) removes this. With minimum-improvement thresholds and a cap on auditor calls, the number of accepted moves has an explicit bound (Theorem 1).

**R3, the floor.** Enforcing ρ + δ on the search sample guarantees nothing about the true spread. In the numerical study, the final set was clearly below the true floor in 6 of 90 runs (by 0.09–0.55%), all with small search samples, and sat on it within the ground truth's noise in 2 more (§5.1). The fix is **certified deployment** (Theorem 2):
1. Draw a second RR sample B, independent of everything that chose the candidate sets.
2. Certify on B, using a paired Bernstein lower bound on σ(S) − ρ·σ(S_ref) and Chernoff–Hoeffding (KL) bounds on every group rate.
3. Deploy the eligible set with the smallest disparity upper bound. The reference is always eligible.

With probability at least 1 − α, the deployed set meets the true floor and its true disparity is at most the reported bound. The bound is non-asymptotic and needs no normal approximation, which the current Monte Carlo test does not provide.

In 270 certificates (165 distinct set–sample pairs) checked against ground truth there were no failures. With 512 RR sets per node, the certificate certified 36 of the 46 distinct final sets that truly met the floor, not counting the reference itself, which needs no certificate. The current Monte Carlo test passed 19 of them. Neither passed any of the 7 distinct sets that were below the floor (§5.2). The disparity bound is valid but loose at these sample sizes.

**What changed in the code** (§6):
- `certify.py` (new);
- `verify.certified_gain` and noise-aware `accept`, with change noise that also covers sampled roots;
- tabu, thresholds and an audit cap in `improve`;
- `z` and `min_gain` in `local_search`;
- new constants in `config`.

Defaults keep the published v1 behaviour except tabu. The noise-aware rule is off (`Z_ACCEPT = 0`) until it is tuned on the v2 validation split, as the protocol requires.

---

## 1. Setup and notation

- Graph G with n nodes, Independent Cascade probabilities p_e (weighted cascade in v2), budget k, and seed set S with |S| = k.
- a_v(S) = Pr(v is activated). Groups g with sizes n_g; P is the set of *present* groups (members and at least one RR root).
- Group rate r_g(S) = (1/n_g) Σ_{v∈g} a_v(S). Spread σ(S) = Σ_v a_v(S) = Σ_g n_g r_g(S). Disparity D(S) = max_{g∈P} r_g − min_{g∈P} r_g.
- A community family F is a set of node sets. A community's rate is the mean of a_r over its RR roots. D_C^F(S) is max − min over F.
- **RR sample.** For each root r, R RR sets are drawn with independent coin flips. The roots are every node on the small and medium tiers, and a group-stratified sample without replacement on the large tier (m_g roots in group g).
  - X_rj(S) = 1[S ∩ RR_rj ≠ ∅]. Exposure estimate â_r = (1/R) Σ_j X_rj.
  - r̂_g = (1/m_g) Σ_{r∈g} â_r, σ̂ = Σ_g n_g r̂_g, and D̂, D̂_C are computed from these. This is exactly `CoverState` and `exact_rates` (float64, from integer counts).
- **Units.** When every node is a root, the units are the RR sets: N_g = R·m_g independent indicators. With sampled roots, the units are the per-root averages â_r ∈ [0, 1], so N_g = m_g.
- The search sample is A, the certification sample is B, and ρ is the spread floor ratio.

---

## 2. R2: invariants, cycling and termination

The loop analysed is `improve()` (`fimgat/v2/improve.py`). Each round proposes moves, scores them exactly on A, and applies the best move that `verify.accept` allows. The community family F grows only when the auditor adds a community. An **epoch** is a maximal run of rounds with a fixed F. Arithmetic is treated as exact: tol = 10⁻¹² only absorbs float rounding, because rates are exact multiples of 1/N_g.

**Proposition 1 (invariants).** Let f be the floor and S_ref the set that `improve` falls back to (RR-greedy on A by default). Let S_0 be the state after the fallback check. Assume f ≤ σ̂_A(S_ref), as when the pipeline sets f = `rr_floor`(ρ, σ̂_A(S_ref)); `improve` itself accepts any floor. Then:
- (a) σ̂_A(S_0) ≥ f.
- (b) Every accepted move ends at σ̂_A ≥ f(1 + FLOOR_RTOL).
- (c) D̂_A never increases along accepted moves.
- (d) Stopping at any round returns a state satisfying (a)–(c).

*Proof.* (a) If the start is below f, the loop restarts from S_ref, whose A-total is at least f by assumption. With the pipeline's floor, `rr_floor` = min(ρ + δ, 1)·σ̂_A(S_ref) ≤ σ̂_A(S_ref). Parts (b) and (c) are the first two checks of `accept`. Part (d) holds because history entries are states. ∎

**Proposition 2 (no revisit within an epoch).** Within one epoch no seed set occurs twice.

*Proof.* With F fixed, every accepted move has ΔD̂ ≤ 0 and ΔD̂_C^F ≤ 0, and at least one of them is < 0. Hence Φ = D̂ + D̂_C^F strictly decreases along the epoch, and a repeated set would repeat Φ. ∎

**Proposition 3 (revisits across epochs are possible).** With the plain rule and no tabu, the loop can visit S_P → S_Q → S_P.

*Construction* (executable as `test_without_tabu_the_search_revisits_a_set`):
1. Take a 5-node clique P (nodes 0–4) and a 20-node path Q (nodes 5–24) under weighted cascade, with one labelled group, so D ≡ 0. Let S_P = {0}, S_Q = {5} and F₁ = {P, Q}.
2. Let e_P be P's rate under S_P and e_Q be Q's rate under S_Q. On the test sample e_P ≈ 0.5 and e_Q ≈ 0.1.
3. Then D_C^{F₁}(S_P) = e_P and D_C^{F₁}(S_Q) = e_Q, because a disconnected component has rate 0. Since e_Q < e_P, the move S_P → S_Q is accepted.
4. Now the auditor adds C₃ = {5}, giving F₂. Node 5 is S_Q's seed, so C₃'s rate is 1 under S_Q and 0 under S_P. Hence D_C^{F₂}(S_Q) = 1 > e_P = D_C^{F₂}(S_P), and the move S_Q → S_P is accepted.

The re-scoring after family growth is the only cause. Each step is a legitimate improvement *on the family at that time*, but nothing stops the loop from oscillating while the family keeps growing. ∎

**Theorem 1 (termination and a bound on moves).** Suppose accepted moves must improve D̂ by more than ε_D or D̂_C by more than ε_C (`min_gain`, `min_gain_comm` > 0), and let E be the number of family-growth events (at most `max_audits`). Then

  #accepted moves ≤ ⌊D̂_A(S_0)/ε_D⌋ + (E + 1)·⌊1/ε_C⌋,

and `improve` stops within 2·(that bound) + 3 rounds. With tabu, no seed set is ever visited twice, whatever the thresholds.

*Proof.*
- Call a move a D-move if D̂ falls by more than ε_D, and a C-move otherwise. A C-move must then cut D̂_C by more than ε_C.
- With noise-aware acceptance, the certified gain is at most the plain gain (Lemma 5(i)), so the same counts hold.
- D̂ ≥ 0 never increases (Proposition 1), so there are at most ⌊D̂_0/ε_D⌋ D-moves in total.
- Within an epoch, D̂_C^F ∈ [0, 1] never increases, so there are at most ⌊1/ε_C⌋ C-moves per epoch, and there are E + 1 epochs.
- A round without an accepted move does one of three things:
  - widens the proposer, which happens once in total and only for the simple proposer;
  - re-audits and continues, at most once between two accepted moves, because the re-audit flag is reset only on acceptance;
  - stops.

  With B accepted moves that is at most B + 1 + (B + 1) + 1 = 2B + 3 rounds. Periodic audits (`audit_every`) run inside rounds, so they add no rounds; their family growth counts in E.
- Tabu: every accepted state's Zobrist hash is recorded, and candidates leading to a recorded hash are skipped.
  - The keys are drawn from [1, 2⁶³ − 1], so the hash has 63 bits, and a false block needs a 63-bit collision.
  - There are at most rounds × candidates checks against at most rounds stored hashes, so the probability is about (rounds² × candidates)/2⁶³, which is negligible. ∎

*Remark.* Without thresholds (ε = 0, the published default), termination rests on `max_rounds`, plus tabu, which makes the loop finite on its own, though the bound is astronomically large. The thresholds make the bound meaningful. §3 says how large they should be.

---

## 3. R1: noise, selection and the winner's curse

**Lemma 1 (what the sample is).**
- *Every node a root:* the X_rj(S) are independent across (r, j) and E[X_rj] = a_r(S). Hence E[r̂_g] = r_g and E[σ̂] = σ.
- *Sampled roots* (without replacement within each group, independently of the coins): E[r̂_g] = r_g, and the units â_r are conditionally independent given the roots.

*Proof.* RR sampling (Borgs et al. 2014): Pr(S ∩ RR(r) ≠ ∅) = a_r(S). Fresh coins make sets independent. Simple random sampling makes the root mean unbiased. ∎

**Lemma 2 (bounds for a fixed set).** For a set S chosen independently of the sample, and units in [0, 1] with sample mean p̂ over N units, let

  U = max{q ≥ p̂ : N·kl(p̂, q) ≤ ln(1/δ)}  and  L = min{q ≤ p̂ : N·kl(p̂, q) ≤ ln(1/δ)},

where kl is the Bernoulli Kullback–Leibler divergence. Then Pr(r_g > U) ≤ δ and Pr(r_g < L) ≤ δ.

*Proof.* Hoeffding (1963, Thm 1): for independent variables in [0, 1] with mean μ, Pr(X̄ ≥ μ + t) ≤ exp(−N·kl(μ + t, μ)), and symmetrically for the lower tail. The event {r_g > U} equals {r_g > p̂ and N·kl(p̂, r_g) > ln(1/δ)}, which is a lower-tail event of probability ≤ δ. For sampled roots, condition on the roots: the moment generating function is E[exp(Σ_i c_{R_i})] with c_r = ln E[e^{λâ_r}]. That is a convex function of a without-replacement sample, so Hoeffding (1963, Thm 4) bounds it by the with-replacement value, for which Theorem 1 applies. ∎

`tests/test_theory_v2.py::test_kl_bounds_coverage_is_exact_binomial_safe` checks the coverage against exact binomial tails. The level noise is s_g ≈ √(r_g(1−r_g)/N_g).

**Lemma 3 (common random numbers).** Fix a swap S → S' = S − u + v. Only RR sets that contain u or v can change coverage. Let ΔX_j = X_j(S') − X_j(S) ∈ {−1, 0, 1}, and let t_i be the number of changed sets rooted at root i.
- *Every node a root* (the RR sets are the units, N_g = R·m_g): Δr̂_g = (1/N_g) Σ_{j∈g} ΔX_j. Then E[Δr̂_g] = Δr_g and Var(Δr̂_g) ≤ E[c_g]/N_g², where c_g = Σ_{i∈g} t_i is the number of group-g RR sets whose coverage changes.
- *Sampled roots* (the roots are the units): Δr̂_g = (1/m_g) Σ_{i∈g} Δâ_i, where Δâ_i = (1/R) Σ_{j rooted at i} ΔX_j, so |Δâ_i| ≤ t_i/R. Then E[Δr̂_g] = Δr_g and Var(Δr̂_g) ≤ E[Σ_{i∈g} t_i²]/(R·m_g)².

The sample gives c_g and every t_i exactly (`swap_detail`, `root_touch_from`).

*Proof.*
- Every node a root: independence (Lemma 1), and Var(ΔX) ≤ E[ΔX²] = Pr(ΔX ≠ 0).
- Sampled roots: unbiasedness is as in Lemma 1. The variance of a mean of m_g roots drawn without replacement is at most that of m_g draws with replacement, so Var(Δr̂_g) ≤ (1/m_g)·E[Δâ²] ≤ E[t²]/(R²·m_g). Each sampled root is marginally uniform over its group, so m_g·E[t²] = E[Σ_{i∈g} t_i²]. ∎

With sampled roots the bound includes the between-root variance. When a move's changes concentrate on a few roots, Σ t_i² can be up to R times c_g = Σ t_i, so the every-node formula would understate the noise.

When every node is a root, the change noise τ_g = √c_g/N_g relates to the level noise s_g as τ_g/s_g = √(p_touch/(r(1−r))), where p_touch is the fraction of the group's RR sets the move touches. On the study graph at R = 8, a random swap touches a median of 0.66% of the RR sets, so τ/s ≈ 0.17. Comparing two neighbouring sets on one sample is far more precise than estimating either one.

**Lemma 4 (a fixed set's disparity is over-estimated).** For S fixed, E[D̂(S)] ≥ D(S).

*Proof.* max is convex and min is concave, so Jensen gives E[max r̂] ≥ max r and E[min r̂] ≤ min r. ∎

For two groups tied at the maximum, the bias is about s/√π. Disparity read off a fresh sample is therefore pessimistic, and selection (below) pushes in the other direction. Which effect wins is an empirical question, answered in §5.

**Proposition 4 (where selection comes from).**
- (a) *Multiplicity.* Take M candidates with equal true gain. If their errors e_i are sub-Gaussian with variance proxy τ² (any dependence), then E[max_i e_i] ≤ τ√(2 ln M), the standard maximal inequality. The chosen candidate's estimate therefore exceeds its truth by at most that much on average.
  - The change errors here are sparse ±1 sums. Those are Bernstein-type rather than sub-Gaussian with a proxy equal to their variance, and a Bernstein maximal inequality adds a term of order (ln M)/N_g.
  - With common random numbers, τ is the change noise of Lemma 3, much smaller than s, but M is large. On the study graph at R = 8, τ ≈ 0.17·s, and the exhaustive proposer scores M = 25 × 1,975 ≈ 4.9·10⁴ swaps. That gives τ√(2 ln M) ≈ 0.8·s per group, plus about 0.2·s from the Bernstein term. This is the same order as (b).
- (b) *Near-ties.* Let groups 1 and 2 have equal true rates at the minimum, and let estimation noise make group 1 look lower by G, with G ~ |N(0, 2s²)| and E[G] = 2s/√π ≈ 1.13·s. A move that lifts group 1 then has estimated gain G and true gain 0, because group 2 is still at the minimum. Its error is at the scale of the *level* noise s, not the change noise τ.
- In general, |D̂(S) − D(S)| ≤ 2·max_g |r̂_g(S) − r_g(S)| holds deterministically. The phantom gain of a whole search from S₀ to S_f, (D̂ − D)(S₀) − (D̂ − D)(S_f), is therefore at most (D̂ − D)(S₀) + 2‖r̂ − r‖∞(S_f): the start set's estimation error plus the final set's level error. This is the scale at which the sample cannot tell groups apart.

Both (a) and (b) are upper-bound or idealised calculations. §5 measures their combined effect, not each one.

**Lemma 5 (certified gain).** Let x and x' be estimated values before and after a move, d = x' − x, and r, r' the true values. Suppose that for every group g, |x_g − r_g| ≤ λ_g (level event) and |d_g − (r'_g − r_g)| ≤ η_g (change event). Then

  D(r) − D(r') ≥ I := min_g [max(0, (x_g − λ_g) − min_h(x_h + λ_h)) + d_g − η_g]
            + min_g [max(0, max_h(x_h − λ_h) − (x_g + λ_g)) − d_g − η_g].

Moreover:
- (i) I ≤ the plain gain D(x) − D(x'), with equality when λ = η = 0.
- (ii) Call a group b a *possible minimum* if x_b − λ_b ≤ min_h(x_h + λ_h): on the level event it may be the true minimum. Call it a *possible maximum* if x_b + λ_b ≥ max_h(x_h − λ_h).
  - If a possible minimum b is not raised by more than η_b, the minimum-side term is at most d_b − η_b ≤ 0, so I is at most the maximum-side term. The maximum side is symmetric.
  - A move that acts only on a tied end is therefore never certified. A move that also moves the other end far enough can be.
- (iii) If some group g is both a possible minimum and a possible maximum, then I ≤ −2η_g < 0 for every move, so nothing can be certified. This includes the case where all intervals [x_h − λ_h, x_h + λ_h] overlap pairwise.

*Proof.*
1. Write the true minimum-side gain as min r' − min r = min_g [(r_g − r_{t⁻}) + (r'_g − r_g)], where t⁻ is the true argmin.
2. On the level event, r_{t⁻} = min r ≤ min_h(x_h + λ_h) and r_g ≥ x_g − λ_g. Together with r_g ≥ r_{t⁻}, this gives r_g − r_{t⁻} ≥ max(0, (x_g − λ_g) − min_h(x_h + λ_h)).
3. On the change event, r'_g − r_g ≥ d_g − η_g. This gives the first term.
4. The maximum side is symmetric: max r − max r' = min_g [(r_{t⁺} − r_g) + (r_g − r'_g)], with r_{t⁺} ≥ max_h(x_h − λ_h) and r_g ≤ x_g + λ_g.
5. For (i), each term only decreases as λ or η grows, and at λ = η = 0 the expression is the plain gain.
6. For (ii), the minimum-side term at b has first part 0 by hypothesis, so it equals d_b − η_b. For (iii), add the minimum-side term and the maximum-side term at g: (d_g − η_g) + (−d_g − η_g) = −2η_g. ∎

`test_certified_gain_*` checks:
- the reduction to the plain gain;
- validity on 4,000 random instances;
- the tie example;
- rules (ii) and (iii).

The verifier uses the following half-widths, for groups and in the same way for communities on their own roots (`verify.noise_model`, `verify._touched`, `accept(..., noise=)`):
- **λ_g = z·√(max(x_g(1−x_g), 1/N_g)/N_g)**, with N_g the level units of §1;
- **η_g = z·√(c_g + 1)/(R·m_g)** when every node is a root;
- **η_g = z·√(Σ_{i∈g} t_i² + 1)/(R·m_g)** with sampled roots (Lemma 3).

Per move these are plug-in confidence statements. Under the adaptive search they are *not* rigorous, because the sample also chose the move. They are a principled guard, and the rigorous guarantee is the deployment certificate of §4.

**Proposition 5 (resolution limits; a normal-approximation heuristic).** Take two groups with rates near r, each estimated from N units, whose true rates differ by ε.
- (a) Any test that tells them apart at confidence z needs N ≳ 2z²·r(1−r)/ε².
- (b) The noise-aware verifier treats them as tied while their intervals overlap, |x_a − x_b| ≤ λ_a + λ_b. It therefore needs N ≳ 4z²·r(1−r)/ε², twice as many units.

*Derivation.*
- The difference of two independent estimates has standard deviation √(2r(1−r)/N), so (a) needs z·√(2r(1−r)/N) ≤ ε.
- Under (b), the gap must exceed λ_a + λ_b ≈ 2z·√(r(1−r)/N).

This is a heuristic, not a theorem: it uses the normal approximation and assumes equal sizes and rates.

For a community of m roots with R RR sets each (every node a root), N = R·m. So (a) gives

  **m ≳ 2z²·r(1−r)/(R·ε²)**,

and (b) doubles it. The table gives (a); the noise-aware verifier needs twice these sizes.

| r | ε | z | R = 16 | R = 64 | R = 512 |
|---|---|---|---|---|---|
| 0.07 | 0.01 | 2 | m ≥ 326 | m ≥ 82 | m ≥ 11 |
| 0.07 | 0.005 | 2 | m ≥ 1,302 | m ≥ 326 | m ≥ 41 |
| 0.33 | 0.01 | 2 | m ≥ 1,106 | m ≥ 277 | m ≥ 35 |

For labelled groups with thousands of members the limit is mild. For example, Pokec-like rates (r ≈ 0.07) with 20,000 roots and R = 16 resolve ε ≈ 1.3·10⁻³ under (a), or 1.8·10⁻³ under (b). For the auditor's communities it is not mild: `MIN_COMMUNITY = 10` at the medium tier (R = 16) leaves a community-rate standard deviation of about 0.02 at r ≈ 0.07 and 0.04 at r ≈ 0.35. §5.3 measures this. The auditor's communities are also *selected* for looking extreme on the same noisy exposure field, which is another winner's curse.

*Design rule:* set the minimum community size from (a) with the plain rule and from (b) with the noise-aware rule, or raise R. Preferably also score the auditor's communities on RR sets the auditor did not see.

**Choosing the thresholds of Theorem 1.**
- *With the noise-aware rule (z > 0),* Lemma 5 already subtracts the near-tie and change noise. The thresholds ε_D and ε_C then only have to make the bound on moves meaningful, and they can be small. With ε_D = 0.002, for example, the bound is ⌊D̂₀/0.002⌋, which is at most 24 D-moves on the study graph.
- *With the plain rule,* a threshold is the only guard against noise. To stop near-tie moves it would have to reach the resolution scale z·√(2r(1−r)/N) of Proposition 5, which also blocks real small gains. Lemma 5 is the better tool, because it pays the noise price only where groups are actually tied.

---

## 4. R3: the spread floor and certified deployment

Enforcing σ̂_A(S) ≥ (ρ + δ)·σ̂_A(S_ref) on the search sample means the search ends near that boundary ("floor hugging"), where A's noise is most likely to favour the set. It therefore says nothing reliable about σ(S) ≥ ρ·σ(S_ref). §5 shows true-floor violations. The guarantee has to come from a sample the search never saw.

**Lemma 6 (paired floor bound).** Let S and S_ref be fixed, independent of B. The estimate is

  T̂ = Σ_u w_u y_u, with y_u = X_u(S) − ρX_u(S_ref) ∈ [−ρ, 1] and w_u = n_g/N_g,

and it has mean T = σ(S) − ρσ(S_ref). Let Q̂ = Σ_u (w_u y_u)², c = max w_u², b = (1 + ρ) max w_u and ℓ = ln(2/δ). Define

  Q⁺ = (√(Q̂ + cℓ/2) + √(cℓ/2))²  and  t = bℓ/3 + √((bℓ/3)² + 2Q⁺ℓ).

Then Pr(T < T̂ − t) ≤ δ.

*Proof.*
1. Q = Σ_u E[(w_u y_u)²] ≥ Var(T̂), and the terms q_u = (w_u y_u)² lie in [0, c], so E[q_u²] ≤ c·E[q_u].
2. Maurer (2003, Thm 1) for sums of independent non-negative variables gives Pr(Q − Q̂ ≥ s) ≤ exp(−s²/(2cQ)). Solving for Q gives Pr(Q > Q⁺) ≤ δ/2.
3. Bernstein's inequality with variance Q and range b gives Pr(T̂ − T > t(Q)) ≤ δ/2, and t(·) increases in its variance argument.
4. Hence Pr(T < T̂ − t(Q⁺)) ≤ Pr(Q > Q⁺) + Pr(T̂ − T > t(Q)) ≤ δ.
5. Sampled roots use per-root units, and the moment-generating-function steps hold under sampling without replacement by Hoeffding's Theorem 4, as in Lemma 2. ∎

**Pairing matters.** A unit contributes to Q̂ only when S and S_ref cover it differently, or as (1−ρ)² when both do. Near-identical sets therefore get tight bounds.

The project's forward simulator (`_mc_forward`) does not pair different sets: their random streams desynchronise (`reference.py` docstring). The current equal-spread test also uses a normal approximation.

Forward Monte Carlo *could* be made paired and rigorous:
- sampling live-edge worlds and evaluating both sets on each world pairs them exactly;
- an empirical-Bernstein bound makes the result rigorous.

But a simulation's paired spread difference can range over n, whereas an RR unit's paired value lies in [−ρ, 1]. That makes the range term of a Monte Carlo bound much larger. RR sampling gives the pairing and the small range directly.

**Theorem 2 (certified deployment).** Let the reference S_ref and the candidates S_1, …, S_m be chosen without using B; they may all be computed from A. Let B be independent of A and of those sets. `certify.certified_choice(B, candidates, S_ref, ρ, α)` does the following:
1. It splits α over 2 statements per distinct set: Lemma 6 for the floor, and Lemma 2 for each present group's lower and upper rate bound, which give UCB_D = max_g U_g − min_g L_g.
2. It declares S_ref and every candidate with floor bound ≥ 0 eligible.
3. It deploys the eligible set with the smallest UCB_D.

Then with probability at least 1 − α:
- (i) every eligible set satisfies σ(S) ≥ ρ·σ(S_ref);
- (ii) every reported UCB_D is at least the set's true disparity.

In particular the deployed set S* satisfies σ(S*) ≥ ρ·σ(S_ref) and D(S*) ≤ min over eligible sets of UCB_D. That is at most the bound of the start set, if the start is a candidate and is eligible, and at most the bound of the reference.

*Proof.* Conditionally on (A, S_ref, S_1, …, S_m), every set is fixed and B is still independent of it, so Lemmas 2 and 6 apply to each statement at its share of α. A union bound covers them all. The reference is eligible without sampling error, because T(S_ref) = (1−ρ)σ(S_ref) ≥ 0 for ρ ≤ 1. ∎

`test_certificate_coverage_against_exact_ic` checks coverage against *exact* Independent Cascade (all 2¹² live-edge worlds) over 300 independent B samples. §5 checks it at scale against the high-precision ground truth.

**Remarks.**
- *What is guaranteed:* true IC expectations (spread and group rates), not one campaign's realisation. Community disparity D_C is not certified yet; the same Lemma 2 machinery applies to communities with their own units.
- *Choosing δ on A.* δ no longer carries the guarantee; it now only decides how often the certificate succeeds. A set found on A fails certification when its true slack, (σ(S) − ρσ(S_ref))/σ(S_ref), is below the certificate's relative half-width. Its A-slack also overstates the true slack by about the A-noise of the paired difference, √Q̂_A/σ_ref, inflated by selection near the boundary.
  - A workable rule is **δ ≳ (t_B + 2√Q̂_B + 2√Q̂_A)/σ_ref**. Its three terms are the certificate's half-width, B's own noise, and A's overstatement, where √Q̂ is Lemma 6's standard error on each sample.
  - Set δ per tier from validation queries.
  - On the study graph the rule gives about 0.35% with A and B both at 512 RR sets per node, and about 0.55% with A at 64. Both agree with the measured thresholds (§5.2).
- *Candidates.* Use the start set, the final set and a few states along the path. All have D̂_A non-increasing, but A's optimism can reorder them in truth, and Bonferroni over a handful of sets costs little.
- *Other selection rules (2026-10-14, `design_v2.md` §21.1).* Step 3 of Theorem 2 may be replaced by any rule that chooses among the eligible sets using B, because (i) and (ii) hold simultaneously for every eligible set on the same event of probability ≥ 1 − α. The chosen set then meets the floor, and its disparity is at most its own UCB_D, rather than at most the smallest one.
  - The **path rule** (`certify.choose(..., "path")`) deploys the eligible candidate furthest along the search path whose UCB_D is at most min UCB_D + c · SE_B. Here c = `DEPLOY_TOL_SE` and SE_B is the smallest-bound set's standard error of D̂ on B.
  - Its disparity guarantee is therefore min UCB_D + c · SE_B.
  - T2 uses it, so that the ε-constraint phase's states are not discarded for a labelled-disparity difference within noise.

---

## 5. Numerical checks

`experiments/v2/theory_checks.py` uses the 2,000-node DC-SBM of the test suite: weighted cascade, three groups of 627–723 nodes, k = 25, ρ = 0.99 and δ = 0.0025. Ground truth comes from 8,192 RR sets per node drawn from separate seeds (group-rate standard deviation about 2·10⁻⁴). The graph is synthetic on purpose: it touches no v2 validation or test data. Outputs: `results/v2/theory/{search,certify,community_noise}.csv` and `summary.json`.

### 5.1 Search on a noisy sample (R1)

Ten replicates per row. Each row is `improve(proposer="exhaustive", audit=False)` from the floor-meeting greedy-family start, on a search sample with R RR sets per node.
- *Gains* are the decrease of disparity from start to finish, as a mean per run.
- *Phantom* is estimated minus true gain.
- *Worsening moves* are accepted moves that raised the true disparity by more than 10⁻⁴.

The rules share their start sets within each R, but the start is chosen on A, so it differs across R. Runs with z = 0, 1 and 2 often end at the same set: the 90 runs contain 55 distinct final sets, counted per replicate.

| R (search) | rule | accepted moves | worsening moves | est. gain | true gain | phantom | true start D | true final D | final set below true floor |
|---|---|---|---|---|---|---|---|---|---|
| 8 | plain (z = 0) | 23 | **7** | 0.0112 | 0.0059 | **0.0053** | 0.0210 | 0.0152 | 3/10 |
| 8 | z = 1 | 6 | 0 | 0.0068 | 0.0052 | 0.0015 | 0.0210 | 0.0158 | 2/10 |
| 8 | z = 2 | 0 | 0 | 0 | 0 | 0 | 0.0210 | 0.0210 | 2/10 |
| 64 | plain | 26 | **2** | 0.0171 | 0.0156 | 0.0015 | 0.0278 | 0.0122 | 1/10 |
| 64 | z = 1 | 18 | 0 | 0.0162 | 0.0153 | 0.0009 | 0.0278 | 0.0125 | 0/10 |
| 64 | z = 2 | 16 | 0 | 0.0156 | 0.0147 | 0.0009 | 0.0278 | 0.0130 | 0/10 |
| 512 | plain | 39 | 0 | 0.0259 | 0.0256 | 0.0003 | 0.0378 | 0.0122 | 0/10 |
| 512 | z = 1 | 38 | 0 | 0.0259 | 0.0256 | 0.0003 | 0.0378 | 0.0122 | 0/10 |
| 512 | z = 2 | 29 | 0 | 0.0248 | 0.0245 | 0.0002 | 0.0378 | 0.0133 | 0/10 |

What this shows:
1. **Small samples chase noise.** At R = 8 the plain rule reports almost twice the real gain, and 7 of its 23 accepted moves made the true disparity *worse*. At R = 512 estimate and truth agree to 3·10⁻⁴.
2. **z = 1 removes every worsening move at every sample size and cuts the phantom gain about 3.5× at R = 8, while keeping nearly all of the true gain** (0.0052 vs 0.0059; 0.0153 vs 0.0156; equal at R = 512).
   - It does **not** lower the true final disparity. The plain rule's real and spurious moves net out slightly positive (final D 0.0152 vs 0.0158 at R = 8, 0.0122 vs 0.0125 at R = 64).
   - Its value is honest reporting, fewer and safer moves, and no regressions, not better outcomes.
   - z = 2 is too strict at small R. At R = 8 the group rates (about 0.32, with 5,000–5,800 units per group) have λ ≈ 0.013 at z = 2.
     - In 8 of the 10 R = 8 starts, some group was both a possible minimum and a possible maximum; in 6 of them all three intervals overlapped. By Lemma 5(iii), no move could be certified there at all.
     - In the other 2 starts only the minimum was tied. By Lemma 5(ii), a move had to lower the top group by more than its change half-width plus a tied group's, or raise both tied groups by more than theirs. No candidate swap did.
3. **The sample size matters more than the choice between z = 0 and z = 1.** Gains are not comparable across R, because each R starts from a different set: its true disparity averaged 0.021, 0.028 and 0.038 at R = 8, 64 and 512. The outcomes are comparable:
   - With the plain rule, the true final disparity was 0.0152 at R = 8 and 0.0122 at both R = 64 and R = 512.
   - The final set broke the true floor in 3, 1 and 0 of 10 runs at R = 8, 64 and 512.
   - Going from R = 8 to 64 lowered the plain rule's final disparity by 0.003. That is five times the largest gap between z = 0 and z = 1 at any R (0.0006).
   - A rule that is too strict costs more: z = 2 at R = 8 made no moves and ended 0.006 above the plain rule.
4. **The floor on A guarantees nothing.** Every final set met ρ + δ on A, with an A-slack of at least 0.25%. In truth:
   - In 6 of 90 runs the final set was clearly below the floor, by 0.09–0.55%. That is 6 to 37 standard deviations of the ground truth's paired slack, which is about 1.3·10⁻⁴.
   - In 2 more runs, which share one set, it was 0.013% below, about one standard deviation. That set sits on the floor.
   - All 8 were found with R ≤ 64.
   - Two of the sets (from rep 1 and rep 4) were *start* sets chosen on A from the greedy family, on which no move was made. So it is not only the search: any choice made on A near the boundary is exposed.

   A's estimate overstates the true slack by a median of 0.17%, 0.04% and 0.004% at R = 8, 64 and 512. The 90th percentiles are 0.49%, 0.28% and 0.09%.
5. **Lemma 4's over-estimate persists under mild selection.** Where the search made no moves (R = 8, z = 2), the final sets are the start sets, chosen on A among four candidates. The sample still over-estimated their disparity by 0.0013 on average. Where the search worked hard (R = 8, plain), selection reversed this into a 0.0040 under-estimate.

### 5.2 Certified deployment (R3)

There are 270 certificates: 90 final sets × B samples with R_B ∈ {8, 64, 512}, each B independent of A. Each B serves all 9 runs of its replicate, and runs often share their final set, so the 270 certificates are 165 distinct set–sample pairs.

**Validity.** No eligible set was below the true floor, and no disparity upper bound was below the true disparity. There were 0 failures in each case, against a guarantee of at most 5% per certificate.

**Power.** 82 of the 90 runs ended at or above the floor in truth. Two of these ended at the reference itself: at R = 8 with z = 2 the start was the greedy set and nothing moved. Those sets are eligible by construction, so they are left out below. That leaves 80 runs, which contain 46 distinct sets.

| check | certified (of 80 runs) | certified (of 46 distinct sets) | median half-width (relative to σ_ref) |
|---|---|---|---|
| RR certificate, R_B = 8 | 4 (5%) | 3 (7%) | 1.4% |
| RR certificate, R_B = 64 | 16 (20%) | 11 (24%) | 0.45% |
| RR certificate, R_B = 512 | **63 (79%)** | **36 (78%)** | **0.15%** |
| Monte Carlo equal-spread test (2,000 simulations, normal approximation) | 30 (38%) | 19 (41%) | 0.31% |

Neither the Monte Carlo test nor any certificate passed any of the 8 runs (7 distinct sets) that ended below the floor in the ground truth.

At R_B = 512 (10⁶ RR sets on this graph), the rigorous certificate is valid and about twice as tight as the asymptotic Monte Carlo test.

**The true slack decides success.**

| true slack | certificate at R_B = 512 | Monte Carlo test |
|---|---|---|
| [0, 0.1%] | 0 of 4 | 2 of 4 |
| (0.1%, 0.2%] | 4 of 16 | 0 of 16 |
| (0.2%, 0.3%] | 21 of 22 | 9 of 22 |
| above 0.3% | 38 of 38 | 19 of 38 |

Given its true slack, a set's certificate on B does not depend on A. It succeeds when the true slack exceeds about 0.25%, which is the half-width plus B's own noise.

The size of A still matters. A's overstatement (§5.1, item 4) leaves the selected set with less true slack than its A-slack suggests. With R_B = 512, the share of final sets certified (counting runs) was:
- 26 of 30 when A also had 512 RR sets per node;
- 22 of 30 when A had 64;
- 15 of 28 when A had 8.

**Choosing δ.** The rule of §4 is δ ≳ (t_B + 2√Q̂_B + 2√Q̂_A)/σ_ref. Here √Q̂/σ_ref ≈ 0.05% at 512 RR sets per node, which gives:
- about 0.35% with A and B both at 512 RR sets per node;
- about 0.55% with A at 64.

The data agree. Certification is routine above a true slack of about 0.25%, and A's 90th-percentile overstatement is 0.09% at 512 and 0.28% at 64.

A larger δ costs fairness, because the search has less room above the floor. The value belongs to the validation split.

The half-width shrinks roughly as 1/√(number of RR units): 1.4% → 0.45% → 0.15% for each 8× step. Pokec-z (67,435 nodes, medium tier, R = 16) has about as many units as R = 512 has here. However, its lower exposure rates make relative errors larger, so its half-width has to be measured on validation queries.

**The disparity bound is valid but loose.** Its median excess over the true disparity is 0.047, 0.016 and 0.0055 at R_B = 8, 64 and 512, while the true disparities are 0.008–0.042. Each group gets its own share of α, and the bound adds the worst ends of two KL intervals. It is a guarantee, not an estimate: report it next to the Monte Carlo disparity, not instead of it.

### 5.3 Community-rate noise (Proposition 5)

The table gives the standard deviation of a community's rate across 20 independent samples. Communities are BFS balls with rates of 0.34–0.46, under a fixed seed set.

| community size m | R = 8 | R = 64 | R = 512 |
|---|---|---|---|
| 10 | 0.048 | 0.021 | 0.0058 |
| 30 | 0.023 | 0.0095 | 0.0034 |
| 100 | 0.013 | 0.0046 | 0.0018 |
| 300 | 0.0067 | 0.0025 | 0.0008 |
| 1,000 | 0.0048 | 0.0012 | 0.0005 |

The prediction √(r(1−r)/(R·m)) is an upper bound in 14 of 15 cells. One cell exceeds it by 12%, within the error of a standard deviation estimated from 20 samples. A 10-node community at the medium tier's R = 16 therefore has a rate standard deviation of about 0.04 at these rates. That is as large as the largest group disparity in §5.1 (0.042) and three times the typical final one (0.012–0.013).

---

## 6. Consequences for the design, and what changed in the code

| Change | Where | Default |
|---|---|---|
| Certified deployment: floor and disparity bounds on an independent RR sample, deploy rule | `fimgat/v2/certify.py` (new) | to be called by the Stage 1 scripts at deploy time (`v2_interfaces.md` §20) |
| Certified gain (Lemma 5); `NoiseModel`; `accept(..., min_gain, min_gain_comm, noise)`; `evaluate_*(..., detail=True)`; `_touched` (change noise with sampled roots, Lemma 3) | `fimgat/v2/verify.py` | z = 0: identical to the published rule |
| `swap_delta_detail`, `root_delta_from`, `root_touch_from` (no numeric change) | `fimgat/v2/cover.py` | n/a |
| Tabu (Zobrist hash), `min_gain`, `min_gain_comm`, `z`, `max_audits`; history gains `move` and `n_tabu` | `fimgat/v2/improve.py` | tabu **on**, z = 0, no thresholds, no audit cap |
| `best_swap(..., accept_kw, exclude)`; `local_search(..., z, min_gain)` | `fimgat/v2/local_search.py` | plain rule |
| `Z_ACCEPT`, `TABU`, `MAX_AUDITS`, `CERT_ALPHA` | `fimgat/v2/config.py` | 0.0, True, None, 0.05 |

Design recommendations, for the user to approve, in `design_v2.md` §16. Each is backed by the section in brackets.
1. **Certified deployment.** Replace the deploy-time Monte Carlo floor check with `certified_choice` on an independent RR sample B. The candidates are the start set, the final set and a few states along the path. Keep forward Monte Carlo as the *evaluation* metric, so results stay comparable with baselines and v1. [Theorem 2. §5.2: no failures in 270 checks (165 distinct set–sample pairs). At R_B = 512 it certified 36 of 46 distinct sets that truly met the floor; the Monte Carlo test passed 19.]
2. **Spend compute on the sample before the rule.** With the plain rule, the true final disparity fell from 0.0152 at R = 8 to 0.0122 at R = 64, and floor violations stopped by R = 512. At each tier, measure on validation queries how the held-out final disparity and floor safety change with R, and raise `RR_PER_NODE` where memory allows. [§5.1, item 3]
3. **Noise-aware acceptance: z = 1 is the candidate to validate, against z = 0.** Choose on held-out disparity (sample B), never on A's own estimate. z = 1 is the rule to use if the search's claimed improvement must be real. It does not buy a lower true disparity, and z = 2 is too strict at small R. [§5.1, item 2]
4. **Set δ from the rule in §4,** δ ≳ (t_B + 2√Q̂_B + 2√Q̂_A)/σ_ref, measured per tier on validation queries. That is about 0.35–0.55% on the study graph, against the current 0.25%. [§4 remarks, §5.2]
5. **Set `MIN_COMMUNITY` from Proposition 5** for each tier's R, instead of 10. Use (a) with the plain rule and (b), which is twice as large, with the noise-aware rule. Score the auditor's communities on RR sets the auditor did not use. [§3, §5.3]
6. **Report the certificate** with every deployed set: whether it is eligible, the floor bound, the disparity bound, and whether the reference was deployed instead. [§4]

## 7. Limits of this analysis

- Only the deployment certificate is rigorous after selection. The noise-aware acceptance is a principled guard whose per-move statements are conditional on a fixed candidate list.
- Proposition 5 and the magnitudes in Proposition 4 are normal-approximation estimates, not theorems. The study measures the combined winner's curse; it does not separate multiplicity from near-ties.
- The sampled-roots certificate is conservative, because it ignores the R-fold variance reduction.
- D_C is not certified.
- The reported evaluation, forward Monte Carlo with the equal-spread test, remains asymptotic.
- The cycling example needs a hand-crafted family. How often re-scoring causes revisits in real runs is unknown; tabu makes the question moot.
- The numerical study has the following scope:
  - one synthetic graph (2,000 nodes, group rates about 0.32, higher than Pokec's);
  - one budget (k = 25) and one floor (ρ = 0.99);
  - the exhaustive proposer, with the auditor off.
  
  The noise-aware rule has not yet been run with the GAT proposer or with the hive mind's communities. The medium-tier figures in §5.2 are extrapolations by unit count.
- The ground truth is itself a sample. Its paired-slack standard deviation is about 1.3·10⁻⁴, which is why one set at −0.013% is reported as "on the floor", not as a violation.

---

# Part II. The weak spots (W1–W4)

Status: written 2026-10-10, after Part I, at the user's request to complete W1–W4. Part I asked whether the results can be trusted. Part II asks where the idea is weakest, and what the mathematics says about it:

- **W1.** Does the hive mind protect communities nobody labelled (claim C2)?
- **W2.** How far are we from the best possible disparity, and can we know?
- **W3.** Can the learned proposer ever pay off (claim C3)?
- **W4.** Can the price "thermostat" stabilise?

The conventions are those of Part I. Every result states its assumptions, and either has a proof or is marked as a heuristic. The tests are in `tests/test_theory_v2_w.py`. The numerical study is `experiments/v2/weakspots.py`, which uses synthetic graphs only, with outputs in `results/v2/weakspots/`.

## 8. Summary of Part II

**W1, hidden communities (claim C2).**
- **Transfer bound.** Lemma W1.1 bounds hidden-community disparity by what the verifier controls, plus how well the audited family matches the hidden extremes. It held in all 394 rows of the study where the auditor had found communities, but it is loose, and it is vacuous whenever the auditor finds nothing near the hidden extremes.
- **Scale bias.** The auditor's scan statistic passes over small extreme communities (Proposition W1.2). Multi-scale audits fix that, as a test demonstrates.
- **One campaign or a mixture.** A single campaign has no game-theoretic guarantee (Proposition W1.3). A mixture of campaigns has one, ex ante (Theorem W1.4).
- **The decisive finding is structural.** The acceptance rule ("never make either worse") lets the hive mind constrain the search but rarely drive it (Proposition W1.5).
  - On planted-community graphs, the hive mind as designed did not lower hidden-community disparity in any setting.
  - An ε-constraint phase lowers community disparity while labelled disparity stays within 0.002 of its optimum. At ρ = 0.97 on cohesive graphs it lowered hidden-community disparity by 0.020 on average, 10.5% of the labelled-only mean (better in 13 of 20 cases, p = 0.005; the median case improved by 3.1%), without raising labelled disparity. It did nothing at ρ = 0.99, or when the hidden communities were only loosely cohesive.
  - The mixture lowered ex-ante hidden disparity by 8–11% of the mean in three of the four settings. The price: campaigns that are fair only on average, higher labelled disparity, and more floor violations.
- **Verdict on C2.** The design as written does not support it. On synthetic data, the ε-constraint variant supports it at ρ = 0.97. Both need the user's decision (§13).

**W2, how far from optimal.**
- **Hardness.** Minimising disparity under the floor is NP-hard, even under weighted cascade with ρ near 1 (Proposition W2.1). The design's architecture, heuristic search with an exact checker, is the right one, and optimality can only be judged against bounds.
- **Lower bound.** A computable lower bound (Proposition W2.3) certifies near-optimality when there are more hard-to-reach groups than seeds: in that study, local search came within 13% of the best possible in every case. Otherwise the bound is typically 0.
- **What we cannot certify.** For the typical v2 case of a few large groups, we cannot certify how far from optimal we are.

**W3, the learned proposer (claim C3).**
- **The hybrid's ceiling.** A proposer with an exhaustive fallback keeps exhaustive search's local-optimality guarantee, and can be at most 1/(1 − π) times faster (Proposition W3.1).
- **The fallback matters.** Without it, the simple proposer stopped early in 40 of 54 runs.
- **Speed-up grows with size.** On synthetic graphs of 2,000–32,000 nodes, the hybrid's speed-up grew to 1.2× at 32,000 nodes, and to 1.7× with W4's scaled prices.
- **Room for learning grows with k and n.** At equal cost, a learned proposer's headroom over the scaled hybrid (Proposition W3.2) was 1.0–2.1 for k ≤ 50, but up to 4.9 for k = 100 (3.5–4.9 on 32,000 nodes). The case for learning rests on large budgets and the large tier, and must be measured there before any training.

**W4, the thermostat.**
- **It cannot do harm.** The prices cannot break correctness (W4.1), and the group-price loop cannot oscillate in D (W4.3).
- **The spread price does nothing.** It is identically zero during improvement (W4.2), so proposals ignore the floor: 93–95% of the simple proposer's swaps broke it.
- **The gains are mis-scaled.** They are two orders of magnitude too small to help, even with a setpoint. Gains scaled to the setpoint fix this: fewer floor-breaking proposals, more moves, and lower disparity, with p ≤ 0.031 at every setpoint tested.
- **Bang-bang explained.** The "bang-bang" behaviour of plain dual ascent has the same root as W1.3: with a non-convex choice set, balance may only be reachable by mixing (W4.5).

---

## 9. W1: hidden communities (claim C2)

**Setting.** Let a ∈ [0, 1]^n be an exposure vector: the true activation probabilities a_v(S), an estimate of them, or the expected exposure of a lottery over campaigns.
- A community c ⊆ V has rate r_c(a) = mean_{v∈c} a_v.
- A family 𝓕 has disparity D_𝓕(a) = max_{c∈𝓕} r_c − min_{c∈𝓕} r_c.
- H is the hidden family, for example the held-out ground-truth communities.
- F is the family the auditor has found. The verifier keeps D̂_F from rising.

### 9.1 What transfers from the audited family to the hidden one

**Lemma W1.1 (transfer bound).** For node sets h and c with overlap I = h ∩ c, and any a ∈ [0, 1]^n:
- (a) if I ≠ ∅, then |r_h − r_c| ≤ sd_h(a)·√(|h∖c|/|I|) + sd_c(a)·√(|c∖h|/|I|), where sd is the population standard deviation of a over the set;
- (b) |r_h − r_c| ≤ (|h∖c|/|h| + |c∖h|/|c|)·osc_{h∪c}(a), where osc is max − min of a over the set.

Let Γ(h, c) be the smaller of the two bounds, and let h⁺ and h⁻ be the hidden communities with the highest and the lowest rate. Then for every family F,

  D_H(a) ≤ D_F(a) + min_{c∈F} Γ(h⁺, c) + min_{c∈F} Γ(h⁻, c).

*Proof.*
1. For I ⊆ A, write r_A − r_I = Σ_{v∈A} w_v (a_v − r_A) with w_v = 1/|A| − 1[v∈I]/|I|. The weights sum to 0, which is what allows subtracting r_A.
2. Cauchy–Schwarz gives |r_A − r_I| ≤ √(|A|·var_A(a))·√(Σ_v w_v²). Since Σ_v w_v² = |A∖I|/(|A||I|), this is sd_A(a)·√(|A∖I|/|I|). Applying it to A = h and to A = c, and using the triangle inequality, gives (a).
3. r_A − r_I = (|A∖I|/|A|)(r_{A∖I} − r_I), and |r_{A∖I} − r_I| ≤ osc. The triangle inequality gives (b). When I = ∅, the right-hand side of (b) is 2·osc, which is still at least |r_h − r_c|.
4. For any c, c' ∈ F: r_{h⁺} ≤ r_c + Γ(h⁺, c) ≤ max_F r + Γ(h⁺, c), and r_{h⁻} ≥ r_{c'} − Γ(h⁻, c') ≥ min_F r − Γ(h⁻, c'). Subtracting gives the claim. ∎

`test_pair_gap_bound_is_valid` checks (a) and (b) on 3,000 random instances; `test_transfer_bound_holds_on_random_families` checks the final inequality.

**What it says.**
- The verifier controls only the first term, D_F, and only on its sample.
- The hidden disparity is controlled only if F contains, for each of the two hidden extremes, a community that overlaps it heavily (the second and third terms).
- The bound is tighter where exposure is even inside those communities (the sd factors).
- If F holds nothing near the hidden extremes, the bound is vacuous, and nothing about D_H follows from D_F. The numerical check finds exactly this (§9.6).

**Class version.** Let 𝒞 be the auditor's whole search class: cohesive communities of size at least m. Then D_H ≤ D_𝒞 + Γ(h⁺, 𝒞) + Γ(h⁻, 𝒞), and Γ(h, 𝒞) = 0 when h ∈ 𝒞.
- A perfect auditor over 𝒞 would therefore protect every hidden community that is itself cohesive and large enough.
- The difference between D_F and D_𝒞 is the auditor's *search gap*.

### 9.2 The auditor's scale bias

**Proposition W1.2.** Suppose the auditor maximises the scan statistic √|C|·dev(C) over a set of candidate communities, where dev(C) ≥ 0 is the community's deficit for the "under" side and its excess for the "over" side. Among candidates with sizes in [b, βb], the winner C* satisfies dev(C*) ≥ dev(C°)/√β, where C° is the candidate with the largest deviation in that range. Over sizes [m, M] the factor is √(m/M).

*Proof.* √|C*|·dev(C*) ≥ √|C°|·dev(C°), so dev(C*) ≥ √(|C°|/|C*|)·dev(C°) ≥ √(b/(βb))·dev(C°). ∎

**Consequence.** With m = 10 and communities up to 1,000 nodes, the winner's deviation can be a tenth of the most extreme community's.
- The √|C| weighting is deliberate: it turns the statistic into a z-score, so a large community with a modest deviation counts as more significant (Proposition 5).
- Auditing per size band (`auditor.size_bands`, ratio 2) keeps that weighting within each band, and makes sure every scale is examined. Within a band, the winner is within a factor √2 of the band's extreme.
- `test_multiscale_audit_finds_the_small_extreme_community` demonstrates this. A 15-node community at rate 0.10 sits next to a 200-node community at rate 0.30. The single-scale audit returned the 200-node community, because its statistic is 1.57 against 1.20. The band [10, 20] returned the small one.

### 9.3 One campaign, or a mixture of campaigns

**Proposition W1.3 (a single campaign has no game-theoretic guarantee).** There are instances where every floor-feasible single seed set has disparity 1, while a 50/50 mixture of two of them has expected-rate disparity 0.

*Proof.*
1. Take two disjoint stars, each a centre with three leaves, as the two groups. Under weighted cascade every centre→leaf edge has p = 1.
2. With k = 1 and the floor σ ≥ 4, only the two centres are feasible.
3. Each centre activates its own star fully and the other star not at all, so D = 1.
4. Seeding each centre with probability 1/2 gives both groups an expected rate of 1/2. ∎

`test_two_stars_single_campaigns_are_unfair_mixtures_are_not` checks this with exact Independent Cascade.

So the learner-versus-auditor game behind the hive mind has, in general, only a mixed equilibrium. Kearns et al. (2018) obtain subgroup-fairness guarantees for exactly this structure by letting the learner randomise. For one deterministic campaign no comparable guarantee exists: **the single-campaign hive mind is a heuristic with respect to the game.** A mixture mode recovers a guarantee:

**Theorem W1.4 (mixture of campaigns).** Let A be a finite audited family (present labelled groups and/or communities) and 𝒮 the set of floor-feasible k-sets. For a lottery P over 𝒮 let r_a(P) = E_{S∼P} r_a(S), D_A(P) = max_a r_a(P) − min_a r_a(P), and V_A = min_P D_A(P). Run T rounds of the following:
- The auditor plays two Hedge instances over A with step η = √(8 ln|A|/T). One has gains r_a(S_t) and looks for the highest rate; the other has gains 1 − r_a(S_t) and looks for the lowest. Their mixtures are q⁺_t and q⁻_t.
- The learner answers with an ε-best response S_t ∈ 𝒮, meaning Σ_a (q⁺_t − q⁻_t)_a r_a(S_t) ≤ min_{S∈𝒮} Σ_a (q⁺_t − q⁻_t)_a r_a(S) + ε.

Then the uniform lottery P̄ over S_1, …, S_T satisfies

  D_A(P̄) ≤ V_A + ε + √(2 ln|A| / T).

*Proof.*
1. For a lottery P and auditor mixtures q = (q⁺, q⁻), let L(P, q) = Σ_a (q⁺ − q⁻)_a r_a(P). Then max_q L(P, q) = D_A(P), attained by point masses on the argmax and the argmin.
2. L is bilinear, and both strategy sets are compact convex polytopes: a simplex over the finite set 𝒮, and a product of two simplices. The minimax theorem (von Neumann; Sion 1958) therefore gives V_A = max_q min_P L(P, q). In particular, min_{S∈𝒮} L(S, q) ≤ V_A for every q.
3. Hedge with gains in [0, 1] has regret at most √(T ln|A|/2) (Cesa-Bianchi & Lugosi 2006, Theorem 2.2). Applied to both instances, T·D_A(P̄) = max_a Σ_t r_a(S_t) − min_a Σ_t r_a(S_t) ≤ Σ_t L(S_t, q_t) + 2√(T ln|A|/2).
4. By the ε-best-response property and step 2, Σ_t L(S_t, q_t) ≤ T·(V_A + ε). Dividing by T gives the claim. ∎

**Remarks.**
- The guarantee is *ex ante*: it concerns expected rates over the lottery. Every campaign meets the floor, but individual campaigns can be unfair, and `mixture.mixture_campaigns` reports both. This is the ex-ante notion that Becker et al. (2022) study for influence maximisation through randomisation.
- ε is the error of the learner's best-response oracle. Here that oracle is a swap local search (`mixture.best_response`), so the guarantee is relative to an oracle whose error is unknown. With an exact oracle (the two stars), the mixture reaches V_A = 0 (`test_mixture_campaigns_reaches_the_mixed_value`).
- The deployed weights come from an LP over the distinct answers that minimises D_A of the expected rates (`mixture.lp_mixture`). It is at least as good as the uniform lottery, so it inherits the bound. If the LP fails, the code keeps the uniform lottery over the T answers.
- In the study, with |A| ≈ 22 actions and T = 30, the regret term √(2 ln|A|/T) is about 0.45, larger than every disparity measured. There the theorem is qualitative, and the measured disparities of the LP mixture are what count.
- This is Freund and Schapire's (1999) construction (no-regret against best response). Kearns et al. (2018) use the same framework, with the roles reversed in their algorithm.

### 9.4 The acceptance rule lets the hive mind constrain, but not drive

**Proposition W1.5.**
- (a) Under `verify.accept`, a move that lowers D̂_C is accepted only if D̂ does not rise by more than 10⁻¹². D̂ is the max − min of continuous group rates, and almost every swap moves the extreme groups' rates. So at a state near a labelled optimum, moves that lower D_C are almost always rejected. The first sentence is the rule; the rest is a heuristic description, measured in §9.6.
- (b) **The ε-constraint phase.** Fix a cap κ, and accept a move when σ̂ ≥ floor, D̂_after ≤ max(D̂_before, κ), D̂_C does not rise, and at least one of D̂ and D̂_C strictly falls (`accept(..., disp_cap=κ)`). Then within an epoch:
  - every accepted move strictly lowers the pair (D̂_C, D̂) in lexicographic order;
  - every state has D̂ ≤ max(D̂_0, κ);
  - no seed set recurs;
  - with the thresholds of Theorem 1, there are at most ⌊1/ε_C⌋ C-moves, and at most ⌊max(D̂_0, κ)/ε_D⌋ D-moves between two C-moves.

*Proof of (b).*
1. If D̂_C falls, the pair falls lexicographically, whatever D̂ does.
2. Otherwise D̂_C is unchanged (it may not rise), so D̂ must fall.
3. D̂ never exceeds max(D̂_before, κ), so by induction it never exceeds max(D̂_0, κ).
4. A strictly decreasing order cannot repeat a set.
5. The counts follow as in Theorem 1, with D̂ ∈ [0, max(D̂_0, κ)] restarting after each C-move. ∎

The ε-constraint method is the standard way to trade one objective against another under a cap.
- In the study it starts from the labelled-only optimum D̂*_a, with κ = D̂*_a + 0.002, about one level-noise unit at R = 64.
- Its proposals come from damped prices over the community family (`proposer.community_moves`).
- It relaxes the design's "never make either worse" rule, so it is **a proposal for the user, not a default**. *The user approved it for T2's option (b) on 2026-10-10, with κ = D̂*_a + 1 SE (`improve.hive_search`; `design_v2.md` §18.1).*

`test_epsilon_phase_respects_the_cap_and_the_lexicographic_order` and `test_accept_disp_cap_trades_labelled_for_community_disparity` cover (b).

### 9.5 What changed in the code (W1)

All of these were off by default when Part II was written. §13 says what the decisions of 2026-10-10 changed.
- `auditor.size_bands` and `auditor.audit_bands`, plus `improve(..., audit_band_ratio=r)`: multi-scale audits.
- `improve(..., disp_cap=κ)` with `verify.accept(..., disp_cap=κ)`: the ε-constraint phase. With a cap, proposals are ranked by D_C first. The exhaustive fallback still pre-filters and ranks on D, so in this phase it adds only D-lowering moves: it is not an exhaustive search for the ε-rule.
- `proposer.community_moves`: proposals steered by damped community prices.
- `mixture.mixture_campaigns`, `mixture.lp_mixture` and `mixture.best_response`: the mixture of campaigns.
- `hidden.transfer_bound` and `hidden.pair_gap_bound`: Lemma W1.1 as a diagnostic.
- `improve(..., return_family=True)` and `audit_max_size`: these were already specified in `v2_interfaces.md` and are now implemented.

### 9.6 Numerical check (W1)

**Setup** (`weakspots.py --part w1`):
- *Graphs.* Planted-block graphs with 30 blocks of 20–300 nodes (power-law sizes), n = 1,454–2,830, under weighted cascade.
- *Labelled groups.* Three groups correlated with the blocks: each member takes its block's dominant group with probability 0.6 and a uniformly random group otherwise (0.73 in total).
- *Hidden communities.* The blocks; no method sees them.
- *Cohesion.* Two levels: block conductance 0.11–0.13 ("cohesive"), and 0.25–0.27 ("loose", just under the auditor's φ₀ = 0.3).
- *Cases.* 10 graphs × k ∈ {25, 50} gives 20 cases per cohesion level and floor, with ρ ∈ {0.99, 0.97}.
- *Samples.* The search sample has R = 64; the ground truth has 1,024 RR sets per node.

**Methods,** all from the floor-meeting greedy-family start:
- **labelled**: verifier option (a), T2's control.
- **hive**: option (b) as designed: single-scale audits and the published acceptance rule.
- **hive, multi-scale**: option (b) with multi-scale audits.
- **ε-constraint phase**: from the labelled-only optimum, with κ = D̂*_a + 0.002, multi-scale audits, community-price proposals and the exhaustive fallback.
- **mixture**: Theorem W1.4 over the labelled groups and the multi-scale family, T = 30, rated by its expected (ex-ante) rates.

Cohesive graphs. "Better" counts the cases (out of 20) where the hidden-community disparity is below the labelled-only one; p is a two-sided Wilcoxon signed-rank test on the paired differences.

| ρ = 0.97 | true hidden-community disparity | change vs labelled | better | p | change in labelled disparity | p | below the true floor |
|---|---|---|---|---|---|---|---|
| labelled (option a) | 0.187 | — | — | — | — | — | 1 |
| hive (option b, as designed) | 0.186 | −0.001 | 8 | 0.19 | −0.0001 | 0.19 | 0 |
| hive, multi-scale | 0.192 | +0.005 | 5 | 0.76 | −0.0002 | 0.08 | 2 |
| **ε-constraint phase** | **0.168** | **−0.020 (−10.5%)** | **13** | **0.005** | −0.0001 | 0.58 | 1 |
| mixture (ex ante) | 0.166 | −0.021 (−11.1%) | 15 | 0.0007 | +0.0014 | 0.005 | 6 |
| greedy reference | 0.180 | −0.007 | 13 | 0.26 | +0.0089 | < 0.001 | 0 |

At ρ = 0.99 on cohesive graphs:
- hive: −0.002 (7 better, p = 0.68);
- hive, multi-scale: +0.004 (p = 0.70);
- ε-constraint phase: +0.001. It made a move in only 5 of 20 cases, too few for the test.
- mixture: −0.015 (12 better, p = 0.010), with labelled disparity up by 0.0021 (p = 0.012), and 5 of 20 mixtures deploying a campaign below the true floor.

On loose graphs the auditor found about one community per run (mean family size 0.8–1.6), and no hive variant changed the hidden disparity by more than 0.002. The mixture changed it by −0.014 at ρ = 0.97 (14 better, p = 0.015, labelled disparity unchanged at p = 0.99) and by +0.004 at ρ = 0.99 (p = 0.39). In both cases, 5 of its 20 mixtures deployed a campaign below the true floor.

What this shows:
1. **The hive mind as designed does not lower hidden-community disparity in any of the four settings.** It adds constraints, but the acceptance rule rarely lets D_C drive the search (Proposition W1.5(a)).
2. **Multi-scale audits find far more communities** (19 against 4 per run on cohesive graphs), but under the published rule they still change nothing.
3. **The ε-constraint phase is the first configuration in which C2's mechanism works.**
   - On cohesive graphs at ρ = 0.97 it lowered hidden-community disparity by 0.020 on average, 10.5% of the labelled-only mean, better in 13 of 20 cases (p = 0.005), with labelled disparity unchanged (p = 0.58). That is T2's pass criterion, on this synthetic family.
   - The effect is concentrated. The median case improved by 3.1%; eight cases improved by more than 6%, three did not change, and four got worse (by up to 10%).
   - It made 8.4 moves on average there, but only 1.0 at ρ = 0.99, where the floor leaves almost no room, and it found little to work with on loose graphs.
4. **The mixture lowers ex-ante hidden-community disparity by 8–11% of the mean where it works, at a cost.**
   - Labelled disparity rises on cohesive graphs (+0.0014 to +0.0021, p ≤ 0.012).
   - Its individual campaigns are less fair than the single labelled-only set: their mean hidden disparity is 0.229 against 0.187 at cohesive ρ = 0.97.
   - Every campaign sits on the sample floor, so 5–6 of 20 mixtures deployed (with positive weight) a campaign below the true floor (worst −0.73%).
   - So it is an ex-ante product with these costs, not a free improvement, and each campaign would need certified deployment (Theorem 2).
5. **Labelled fairness does not transfer.** The labelled-only optimum's hidden disparity is no lower than the greedy reference's (0.187 against 0.180 at cohesive ρ = 0.97).
6. **Lemma W1.1 held in all 394 rows where the auditor's family was not empty, but it is loose.**
   - For the labelled, multi-scale and ε-phase sets, it was non-vacuous (bound < 1) in 60–68% of cohesive cases, with families of 19–36 communities. With the single-scale family (4 communities) it was non-vacuous in 22%, and on loose graphs never.
   - Even where it is non-vacuous it is loose: a mean of 0.37–0.45 against a hidden disparity of about 0.18–0.19. Over all cohesive rows the mean is 0.75–0.79.
   - The slack comes from the most-exposed hidden block. Its gap is about 0.5, and its best-matching found community has Jaccard 0.45–0.50. The least-exposed block is matched well: gap about 0.04, Jaccard 0.63–0.73. **The auditor's "over" side is the weak half.**
7. **Loosely cohesive hidden communities are invisible to the auditor.** At conductance 0.25–0.27 against φ₀ = 0.3, it found about one community per run, so the search gap of §9.1 dominates. φ₀ must be tuned on validation data against the cohesion of the communities that C2 is meant to protect.

---

## 10. W2: how far from the best possible?

**Proposition W2.1 (hardness).** Minimising D(S) subject to |S| = k and σ(S) ≥ ρσ_ref is NP-hard, even under weighted cascade, with a floor that every set containing one particular node meets, and with ρ arbitrarily close to 1.

*Proof.*
1. *Influence maximisation under weighted cascade is NP-hard.* Reduce from exact cover by 3-sets, restricted to instances where every element lies in exactly three sets; this restriction is still NP-complete (Gonzalez 1985).
   - For a universe U with |U| = 3q and sets T_1, …, T_m, create set nodes s_i and element nodes e_u, with an edge s_i → e_u whenever u ∈ T_i. Every element node has in-degree 3, so weighted cascade gives every edge p = 1/3. Set nodes have no in-edges.
   - Let S consist of s set nodes and q − s element nodes, and let c_u be the number of seeded sets containing u. Then σ(S) = q + Σ_{u∉S} f(c_u) with f(c) = 1 − (2/3)^c.
   - f is concave with f(0) = 0 and f(1) = 1/3, so f(c) ≤ c/3, with equality only for c ∈ {0, 1}.
   - Since Σ_u c_u = 3s, we get σ(S) ≤ q + s ≤ 2q. Equality holds exactly when s = q and every element is covered once, that is, when S is an exact cover. So the question "is max_{|S|=q} σ(S) ≥ 2q?" decides the cover problem.
2. *The gadget.* Take an IM instance (H, k', θ) as in step 1. Add a node b₀ with edges b₀ → x to L ≥ n_H + k' + 1 new nodes x ∈ K; each x has in-degree 1, so p = 1 under weighted cascade.
   - The groups are A = V_H and B = {b₀} ∪ K. The budget is k = k' + 1, and the floor is σ ≥ L + 1.
   - A set without b₀ reaches at most n_H + k < L + 1 nodes, so it is infeasible.
   - A set with b₀ is feasible, has r_B = 1, and has r_A = σ_H(S ∩ V_H)/n_H, so D(S) = 1 − σ_H(S ∩ V_H)/n_H.
   - Hence min D = 1 − max_{|S'| = k'} σ_H(S')/n_H, and deciding "min D ≤ 1 − θ/n_H" decides IM.
3. *Relative floor.* RR-greedy seeds b₀ first, since its marginal spread L + 1 exceeds anything in H. So σ_ref ∈ [L + 1, L + 1 + n_H], and the floor is ρσ_ref with ρ = (L + 1)/σ_ref ≥ (L + 1)/(L + 1 + n_H), which tends to 1 as L grows. ∎

**Remarks.**
- With edges of probability 1, an RR sample is the instance itself. So under general Independent Cascade, the problem the verifier optimises on its sample is NP-hard as well.
- Evaluating σ exactly is #P-hard (Chen, Wang & Wang 2010).
- The exact optimum is therefore out of reach in general. This justifies the design's heuristic search with an exact checker, and it means optimality can only be judged against bounds.
- `test_hardness_gadget_reduces_disparity_to_influence_maximisation` checks the gadget identity with exact Independent Cascade.

**Lemma W2.2 (floor-forced mean).** If the labelled groups partition V, then max_g r_g(S) ≥ σ(S)/n for every S. So every S with σ(S) ≥ F has D(S) ≥ F/n − min_g r_g(S).

*Proof.* σ(S)/n = Σ_g (n_g/n)·r_g(S) is a weighted mean of the rates. ∎

**Proposition W2.3 (granularity bound).** Let U_g ≥ max_{|S|=k} r_g(S) and U⁰_g ≥ max_{|S|=k, S∩g=∅} r_g(S). Let u_(k+1) be the (k+1)-th smallest U⁰_g, or ∞ when there are at most k groups. Then every |S| = k has min_g r_g(S) ≤ M := min(u_(k+1), min_g U_g), and

  every S meeting the floor F has D(S) ≥ F/n − M.

*Proof.*
1. The groups partition V, so a k-set contains members of at most k groups.
2. Among the k + 1 groups with the smallest U⁰, at least one therefore has no seed, and its rate is at most its U⁰, which is at most u_(k+1).
3. Also, min_g r_g ≤ r_h ≤ U_h for every h.
4. Combine with Lemma W2.2. ∎

**Computing U and U⁰** (`bounds.group_rate_upper`):
- Run greedy on the group's RR sets; for U⁰, exclude the group's own members. Keep the online bound for monotone submodular coverage: OPT ≤ f(A) + the k largest marginal gains at A, for every greedy prefix A (Leskovec et al. 2007).
- The true optimum is a fixed set, and its sample mean is at most that bound divided by the number of RR sets (B/n_sets, the same sample mean whichever unit convention is used). Lemma 2's KL upper bound increases with the sample mean, so it bounds the optimum's true rate with probability ≥ 1 − δ/(2G).
- `bounds.disparity_lower_bound` returns the bound, valid with probability ≥ 1 − δ.
- Tests: `test_lower_bound_never_exceeds_the_exact_optimum` (brute force, exact IC), `test_lower_bound_is_positive_when_groups_outnumber_seeds`, and `test_group_rate_upper_covers_the_exact_maxima`.

**When it bites.**
- When more than k groups cannot be reached well without a seed of their own (U⁰ small), u_(k+1) is small and the bound approaches the floor-forced mean rate F/n. This is the SPA-region situation recorded in design §11 (2–5-person groups).
- With at most k groups, u_(k+1) = ∞ and min_g U_g lies above the mean rate, so the bound is typically 0: it is positive only if some group cannot reach F/n even with all k seeds. For few large groups we have no lower bound beyond the statistical resolution of Proposition 5, and the gap to the optimum cannot be certified.

### 10.1 Numerical check (W2)

**Setup** (`weakspots.py --part w2`). Three kinds of graph:
- *three groups*: the 2,000-node three-group DC-SBM;
- *isolated pairs*: the same, plus 60 isolated pairs of nodes as 60 more groups;
- *pendant pairs*: the same, plus 60 pairs that each hang off one random node.

Two graphs of each kind, k ∈ {25, 50}, ρ = 0.99. The bound comes from 256 RR sets per node (δ = 0.05). Local search runs from the floor-meeting greedy-family start on 64 RR sets per node, and the truth uses 1,024.

| graphs | k | lower bound | local search (true disparity) | bound / achieved |
|---|---|---|---|---|
| three groups | 25 | 0 | 0.014 | — |
| three groups | 50 | 0 | 0.006 | — |
| isolated pairs | 25 | 0.316 | 0.359 | 88% |
| isolated pairs | 50 | 0.379 | 0.424 | 89% |
| pendant pairs | 25 | 0 | 0.337 | — |
| pendant pairs | 50 | 0 | 0.390 | — |

What it shows:
- **With more hard-to-reach groups than seeds, the bound certifies that local search is within 13% of the best achievable disparity in every case** (12% on average). At most 0.041–0.046 of improvement is left, however good the method. Here u_(k+1) = 0.015, the KL bound for a group that no allowed seed reaches in the sample.
- **With pendant pairs the bound is 0,** even though local search ends at about the floor-forced mean (0.337 against 0.336 at k = 25). The pigeonhole step ignores that the floor limits where seeds can go: a pendant pair's best unseeded rate needs a seed at or next to its attachment node, which the floor makes costly. A tighter bound would have to put the floor inside U⁰, which is a hard problem in itself.
- **With three groups the bound is 0, as predicted.** There is no certificate of near-optimality.

---

## 11. W3: can a learned proposer pay off? (claim C3)

**The model.** One accepted move takes one round.
- Exhaustive search (`local_search.best_swap`) costs T_ex per round and makes the best swap's progress, Δ_ex.
- A proposer checks the grid of its n_add best adds × n_drop best drops, C = n_add·n_drop exact checks, so its round costs c_round = c_score + C·c_s. With probability π its grid contains an improving swap, and its best one then makes progress q·Δ_ex, with q ≤ 1.
- On a miss, the **hybrid** (`improve(..., fallback="exhaustive")`) runs exhaustive search. The current simple proposer instead widens its grid once and then stops.

**Proposition W3.1 (the hybrid).** The hybrid ends at a single-swap local optimum, like exhaustive search. If π, q and the costs are constant along the descent (a modelling assumption, not a theorem), its speed-up over exhaustive search, measured in disparity progress per second, is

  S = (πq + 1 − π)·T_ex / (c_round + (1 − π)·T_ex) ≤ 1/(1 − π).

*Proof.*
1. It stops only when the exhaustive fallback finds no improving swap, which is a local optimum.
2. Per round, its expected cost is c_round + (1 − π)·T_ex and its expected progress is (πq + 1 − π)·Δ_ex. Exhaustive search spends T_ex for Δ_ex.
3. The numerator of S is at most T_ex, and the denominator is at least (1 − π)·T_ex. ∎

So, under this model, **no proposer that falls back to exhaustive search on every miss can make the search more than 1/(1 − π) times faster**. A 10× speed-up needs π ≥ 0.9 at the chosen grid.

**Proposition W3.2 (headroom for learning).** Under the same model, a proposer whose round costs at least κ·c_round can be at most H/κ times faster than the simple hybrid, whatever its recall, provided it proposes single swaps (q ≤ 1; multi-node moves, which the learned proposer can also make, could exceed q = 1), where

  H = 1/((1 − φ)(πq + 1 − π)),  φ = (1 − π)·T_ex/(c_round + (1 − π)·T_ex),

computed for the simple proposer. φ is the share of the simple hybrid's time spent in fallbacks.

*Proof.* Set the learned proposer's π and q to 1, which can only help it, and its cost to κ·c_round. The ratio of the two times per unit of progress is then [(c_round + (1 − π)·T_ex)/(πq + 1 − π)]/(κ·c_round) = H/κ. ∎

**Two more limits.**
- *Without a fallback,* a miss ends the search. If each round misses independently with probability 1 − π (a heuristic), a descent of J moves completes with probability π^J, so the current simple proposer stops early on long descents.
- *Amdahl's bound:* if improvement takes a share f of a query's end-to-end time, no proposer can speed the query up by more than 1/(1 − f). Claim C3 is about end-to-end time (test T4).

**Decision rule.** All of this can be computed on validation states before any training:
1. Measure π, q, c_round and T_ex for the simple proposer (`diagnostics.measure_proposer`), and the improvement share f.
2. Let κ be the learned proposer's measured inference cost relative to the simple scores.
3. If H/κ is close to 1, or f is small, a learned proposer cannot pay off, and T3 is not worth running as designed.

### 11.1 Numerical check (W3)

**Setup** (`weakspots.py --part w3`):
- *Graphs.* Three-group DC-SBMs of 2,000, 8,000 and 32,000 nodes, three graphs of each size.
- *Settings.* k ∈ {25, 50, 100}, ρ ∈ {0.99, 0.97}, R = 16. Descents start from the greedy reference.
- *States.* 189 improvable states along the exhaustive paths.
- *Scorers.* "simple" uses the published prices; "scaled" uses W4's setpoint-scaled prices (setpoint 0.5%, k_p = 200, k_i = 100).
- *Grid.* 16×8 (128 checks), `improve`'s default.

Means over the six (k, ρ) settings of each size, at the 16×8 grid. Measured speed-ups are medians, over the 18 cases per size, of the time exhaustive search took to reach a local optimum divided by the hybrid's time. The "scaled" hit rate is measured with a fresh one-step controller, while the scaled runs accumulate the integral term.

| nodes | T_ex / c_round | hit rate π, simple | π, scaled | model speed-up (simple / scaled) | measured hybrid speed-up (simple / scaled) | headroom H (simple / scaled) |
|---|---|---|---|---|---|---|
| 2,000 | 1.4 | 0.56 | 0.85 | 0.80 / 1.09 | 0.72 / 0.90 | 1.7 / 1.3 |
| 8,000 | 2.9–3.1 | 0.50 | 0.82 | 1.07 / 1.74 | 0.89 / 1.20 | 2.9 / 1.8 |
| 32,000 | 5.5–5.7 | 0.55 | 0.82 | 1.83 / 2.92 | 1.19 / 1.70 | 4.0 / 2.3 |

What it shows:
- **The fallback carries the guarantee.**
  - The hybrid ended at a single-swap local optimum in 54 of 54 runs. Its final disparity equals exhaustive search's on average (mean difference at most 2·10⁻⁴). The two can stop at different local optima, so they differed in 22 of 54 single runs (29 of 54 with scaled prices), by up to 0.0066.
  - Without the fallback, the simple proposer reached a local optimum in only 14 of 54 runs and ended worse in 44. This is the early-stopping risk of §11. With scaled prices: 32 of 54, and worse in 27.
- **The hybrid's advantage grows with graph size,** because T_ex/c_round grows from 1.4 to 5.6. The model overstates the measured speed-up by a factor of 1.1–1.7, as expected from its stationarity assumption, but it gets the trend right.
- **Recall is the binding factor.** At 16×8, the published simple proposer hits only about half the time. W4's scaled prices raise this to 0.82–0.85, and to 0.91 at ρ = 0.99. Most of the simple proposer's misses come from the inert spread price (§12).
- **Headroom for learning grows with k and n.** Against the scaled hybrid, a learned proposer with the same per-round cost has H = 1.0–2.1 for k ≤ 50 (the means per size are 1.3–2.3), but up to 4.9 for k = 100 (3.5–4.9 on 32,000 nodes). It gains less if its inference costs more than the simple scores (κ > 1).
- **The improvement phase is cheap at these sizes.** It took 0.09–1.0 s, against 0.06–0.35 s for RR sampling and 0.07–3.4 s for the greedy reference (k up to 100). If the reference is cached per graph (design §2), improvement is most of the per-query time, but that is about a second at 32,000 nodes.
- **What this means for T3.** On these graphs a learned proposer's headroom over the scaled hybrid is small for k ≤ 50 (at most 2.1×), and it grows with k and n. T3 should therefore compare the learned proposer with the scaled hybrid, not with the simple proposer as it stands. The case for learning rests on large budgets and the large tier, where T_ex/c_round is larger, and the decision rule of §11 has to be applied there, on validation graphs, before any training.

---

## 12. W4: the price thermostat

**Proposition W4.1 (prices cannot break correctness).** Every result in §2 (Propositions 1–3, Theorem 1) and Proposition W1.5 depends only on the acceptance rule. Prices only decide which candidates are scored. So any price sequence, stable or not, preserves the floor, monotonicity and termination. ∎

**Proposition W4.2 (the spread price is inert in the improvement phase).** In `improve`, μ_t = max(0, k_p·e_t + k_i·I_t), with e_t = (F − σ̂_t)/F and I_t = max(0, I_{t−1} + e_t). Every state meets the floor (Proposition 1), so e_t ≤ 0, I_t = 0 and μ_t = 0 for all t. The same holds for `imitation.simple_scores`, and for the learned proposer's controller when ρ + δ ≤ 1 and it is given the same reference total.

At ρ = 1 it does not hold. `LearnedProposer` uses the floor (ρ + δ)·total_ref, which is above `improve`'s min(ρ + δ, 1)·total_ref, so its μ is positive near the floor and winds up. The trainer uses the same uncapped ρ + δ. §13 lists this inconsistency.

*Proof.* Induction on t. ∎

`test_spread_price_is_inert_in_improve` records μ in every round. Consequence: the simple proposer's weights are β − α only, so its proposals ignore spread, and at the floor most of them break it (§12.1).

**Proposition W4.3 (the group-price loop cannot oscillate in D).** The group prices α and β are memoryless functions of the current rates (`prices.damped_prices`): there is no integral term and nothing to wind up. Under the published acceptance rule, accepted moves never raise D̂ (Proposition 1), so D̂ is a Lyapunov function of the closed loop, whatever the prices propose. With a cap (W1.5(b)), the pair (D̂_C, D̂) in lexicographic order plays that role within an epoch. ∎

**Proposition W4.4 (a spread setpoint, linearised).** Give the spread price a setpoint s* > 0, so that e_t = ((1 + s*)F − σ̂_t)/F (`PIController(setpoint=s*)`), and linearise the response as e_{t+1} = e_t − γ(μ_t − μ̄). For γ > 0, the loop is stable if and only if 0 < γk_p < 2, k_i > 0 and γ(2k_p + k_i) < 4.

*Proof.* On (e_t, I_{t−1}), measured as deviations from the fixed point, the loop is linear with characteristic polynomial λ² − (2 − γ(k_p + k_i))λ + (1 − γk_p). Jury's conditions for a quadratic p(λ) = λ² + a₁λ + a₀ are |a₀| < 1, p(1) > 0 and p(−1) > 0. Here p(1) = γk_i and p(−1) = 4 − γ(2k_p + k_i), which gives the three inequalities. ∎

`test_jury_criterion_matches_the_spectral_radius` checks this. With the default gains (1, 0.5), the condition is γ < 1.6.

**Scale.**
- The spread price enters the weights as μ·n_g/n, next to group prices whose total is D/(D + τ), about 0.2–0.4 at the disparities seen here.
- With e ≈ s* ≈ 0.005 and k_p = 1, μ ≈ 0.005: two orders of magnitude too small to matter.
- Gains scaled to the setpoint (k_p = 1/s*, k_i = 0.5/s*) make μ ≈ 1 at a deficit of s*. Then γk_p is large and the linear condition fails.
- Because a single move changes σ̂ in a step, the response is relay-like, so a bounded limit cycle is expected rather than divergence. By W4.1 it is harmless for correctness; §12.1 measures it.

**Proposition W4.5 (why plain dual ascent with a best response is bang-bang).** Let S₁ meet the floor with slack +0.05 and D = 0.10, and let S₂ break it (slack −0.03) with D = 0.02. Run dual ascent μ ← max(0, μ − η·slack(S_t)) with the best response S_t = argmin_S [D(S) − μ·slack(S)]. It never settles:
- S₂ is the best response for μ < 1 and S₁ for μ > 1, so μ circles μ* = 1 and the pure iterates alternate;
- the iterates spend a share 3/8 of the time on S₁ and 5/8 on S₂, and the running average of the slack tends to 0: a mixture that meets the floor on average.

*Proof.*
1. D(S₁) − 0.05μ < D(S₂) + 0.03μ exactly when μ > 1.
2. For μ < 1, the step is +0.03η; for μ > 1 it is −0.05η. So once μ reaches 1 it stays within η·[−0.05, +0.03] of it, and both sets recur.
3. The increments cancel in the long run, so 0.03·n₂ ≈ 0.05·n₁, which gives the 3/8 share and an average slack of 0.
4. In general, when the dual optimum lies where the best response switches between sets (a positive duality gap), the pure iterates cycle, while their averages approach a solution of the convexified problem over lotteries, which meets the floor on average (primal averaging; Nedić & Ozdaglar 2009 treat the convex case). ∎

`test_best_response_dual_ascent_is_bang_bang` checks the example. This is the mechanism behind the design's evidence that "plain dual ascent with a best response oscillates" (design §11). It is analogous to Theorem W1.4, but not the same object. W1.4's lottery mixes only floor-feasible sets and minimises the disparity of expected rates. This average mixes in a set that breaks the floor, and meets the floor only on average.

### 12.1 Numerical check (W4)

**Setup** (`weakspots.py --part w4`). DC-SBMs of the same kind as W3's, with 2,000 and 8,000 nodes (three graphs each, different seeds), k ∈ {25, 50}, ρ = 0.99, R = 16, starting from the greedy reference. The simple proposer runs with and without the exhaustive fallback, giving 24 cases per configuration.

| prices | setpoint | spread price μ (mean) | proposals below the floor (no fallback / fallback) | moves the proposer finds itself (no fallback / fallback) | final disparity (no fallback / fallback) |
|---|---|---|---|---|---|
| default | 0 | 0, exactly | 95% / 93% | 1.75 / 1.00 | 0.0115 / 0.0097 |
| default | 0.25–1% | 0.002–0.014 | 95% / 91–92% | 1.75 / 1.00–1.17 | 0.0112–0.0115 / 0.0097 |
| scaled (k_p = 1/s*, k_i = 0.5/s*) | 0.25% | 0.88–0.99 | 86% / 77% | 2.42 / 2.25 | 0.0099 / 0.0097 |
| scaled | 0.5% | 1.09–1.24 | 84% / 74% | 2.42 / 2.33 | 0.0098 / 0.0097 |
| scaled | 1% | 1.31–1.51 | 71% / 57% | 2.33 / 2.42 | 0.0093 / 0.0091 |

Paired over the 24 cases (both fallback settings), scaled prices against the default prices at setpoint 0:

| setpoint | final disparity better | worse | Wilcoxon p |
|---|---|---|---|
| 0.25% | 6 | 1 | 0.031 |
| 0.5% | 7 | 0 | 0.016 |
| 1% | 10 | 0 | 0.002 |

The spread price reversed direction at most 0.17 times per run on average, so no oscillation was seen. The runs were short (2–3 accepted moves), however, so the limit cycle that W4.4 allows could not have shown.

What it shows:
- **Proposition W4.2 is confirmed.** With the default controller, μ was exactly 0 in every round, and 93–95% of the simple proposer's swaps broke the floor.
- **The default gains cannot fix it, even with a setpoint:** the mean of μ stays at or below 0.014 (single runs reach 0.034), as the scale argument predicts.
- **Gains scaled to the setpoint work.** They make μ about 1–1.5, cut the floor-breaking share to 57–86%, let the proposer find more moves itself (with the fallback, more than twice as many), and lower the final disparity, significantly at every setpoint tested.
- **The thermostat is not unstable; it is mis-scaled.** Through W3, this is also most of the simple proposer's recall problem.

---

## 13. Consequences for the design, and what changed in the code (Part II)

| Change | Where | Default |
|---|---|---|
| Size bands and multi-scale audits | `auditor.size_bands`, `auditor.audit_bands`; `improve(..., audit_band_ratio)` | off |
| ε-constraint acceptance | `verify.accept(..., disp_cap)`; `improve(..., disp_cap)` (ranks by D_C first) | off |
| Community-price proposals | `proposer.community_moves` | used only when passed as the proposer |
| Mixture of campaigns | `fimgat/v2/mixture.py` (new) | not wired into the pipeline |
| Transfer bound (Lemma W1.1) | `fimgat/v2/hidden.py` (new) | diagnostic |
| Lower bounds (Proposition W2.3) | `fimgat/v2/bounds.py` (new) | diagnostic |
| Proposer cost model, measurement, PI stability | `fimgat/v2/diagnostics.py` (new) | diagnostic |
| Exhaustive fallback (the hybrid) | `improve(..., fallback="exhaustive")` | off |
| Spread-price setpoint and gains | `PIController(setpoint)`; `improve(..., price_setpoint, price_gains)` | 0 and (1, 0.5): unchanged |
| `return_family`, `audit_max_size` (already specified) | `improve` | as in `v2_interfaces.md` |

*2026-10-10 (`design_v2.md` §18): the user approved recommendations 1, 2, 6 and 7 below. The table shows the defaults when Part II was written. Since then:*
- *multi-scale audits, the exhaustive fallback and the setpoint-scaled prices are `improve()`'s defaults;*
- *the ε-constraint phase runs in `improve.hive_search`;*
- *`LearnedProposer` and the trainer use the capped floor;*
- *the learned proposer's own price controller keeps the published gains. Whether to scale it is checked only if the W3 rule leads to training one.*

Design recommendations, for the user to approve, in `design_v2.md` §17. Each is backed by the section in brackets.
1. **C2 needs the ε-constraint phase.** The published acceptance rule rarely lets the hive mind drive the search [W1.5, §9.6].
   - Proposal: verifier option (b) for T2 becomes the labelled-only search followed by the ε-constraint phase, with κ = the labelled optimum plus about one noise unit. The published rule is still reported as option (b0).
   - Run T2 at ρ = 0.97 as well as 0.99, because at 0.99 the floor leaves almost no room.
   - This must be decided before T2 runs; the pre-registration is amended, not reinterpreted.
2. **Multi-scale audits** (band ratio 2) should become the hive mind's default. Tune φ₀ on validation data against the cohesion of the target communities, because loosely cohesive communities were invisible at φ₀ = 0.3 [W1.2, §9.6].
3. **Report Lemma W1.1's terms in T2:** D_F, the two gaps, and the Jaccard of the best matches to the hidden extremes, so that a pass or a fail can be explained. The auditor's "over" side is the weaker half and is worth improving [§9.6].
4. **Offer the mixture of campaigns only as an explicit ex-ante mode,** with a certificate per campaign (Theorem 2), never as the headline. Its campaigns are individually less fair and more often break the true floor [W1.4, §9.6].
5. **Report `bounds.disparity_lower_bound`** and the gap to it in every T1 table, and say where it is 0 [W2.3, §10.1].
6. **Make the hybrid with scaled prices the simple-proposer baseline in T3.** Apply the W3 decision rule (π, q, costs and headroom H on validation graphs of the large tier) before training a learned proposer [W3.1–W3.2, §11.1].
7. **Scale the spread price to its setpoint** (setpoint about δ–2δ, k_p = 1/setpoint, k_i = 0.5/setpoint) for the simple proposer, tuned on validation data [W4.2, W4.4, §12.1].
   - Check the same scaling for the learned proposer's price features.
   - Give `LearnedProposer` and the trainer the same capped floor as `improve` (`config.rr_floor`); at ρ = 1 they currently aim above it [W4.2].

## 14. Limits of Part II

- **Synthetic graphs only:** planted blocks for W1 and DC-SBMs for W2–W4. Real communities overlap, cover few nodes and vary in cohesion, so C2 still has to be tested on held-out ground truth (T2).
- **No learned proposer exists yet.** W3's headroom for learning is a bound, not a measurement.
- **The speed-up model assumes π and q are stationary.** It overstated the measured speed-ups by a factor of 1.1–1.7.
- **The largest graph had 32,000 nodes.** Statements about the large tier (≥ 500,000 nodes) are extrapolations.
- **The mixture's oracle is a heuristic local search,** so the ε of Theorem W1.4 is unknown in practice.
- **Proposition W2.1 under weighted cascade** relies on exact cover by 3-sets remaining NP-complete when every element occurs exactly three times (Gonzalez 1985).
- **Untuned parameters.** The ε-constraint phase's κ (D*_a + 0.002) and the scaled gains are first choices, not tuned values; both belong to the validation split.
- **Short W4 runs** (2–3 accepted moves) could not show the limit cycles that W4.4 allows.
- **Ground-truth noise.** The truth used 1,024 RR sets per node, so its own noise in a community rate is at most about 0.001–0.0035 for blocks of 300 down to 20 nodes. That is small next to the hidden disparities of about 0.18 and the ε-phase's −0.020, but not next to the smallest paired differences (for example the hive mind's −0.001).
