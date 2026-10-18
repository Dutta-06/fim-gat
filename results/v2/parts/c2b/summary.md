# C2b: T2's test where the hidden communities are cohesive (PRE-REGISTERED; docs/design_v2.md section 25)

Graph snap_amazon, hidden communities: SNAP's top5000 (primary). Primary: rho 0.97, sets deployed by the path rule; each hypothesis at alpha 0.025 (Bonferroni over two).

- **H1, the method (kappa = D_a + 1 SE): FAIL** (no significant decrease). Hidden-community disparity, b-1se - a over 30 cases: median +0.0000, p nan; labelled disparity: median +0.0000, p nan.
- **H2, the scaled cap (kappa = D_a (1 + 0.05)): FAIL** (no significant decrease). Hidden-community disparity, b-5pct - a over 30 cases: median +0.0000, p nan; labelled disparity: median +0.0000, p nan.

## Every comparison (run - a, paired by case; negative = lower)

| rho | sets | run | metric | cases | median difference | p |
|---|---|---|---|---|---|---|
| 0.97 | certified-path | b-1se | comm_disparity | 30 | +0.0000 | nan |
| 0.97 | certified-path | b-1se | comm_disparity_all | 30 | +0.0000 | nan |
| 0.97 | certified-path | b-1se | comm_disparity_visible | 30 | +0.0000 | nan |
| 0.97 | certified-path | b-1se | disparity | 30 | +0.0000 | nan |
| 0.97 | certified-path | b-5pct | comm_disparity | 30 | +0.0000 | nan |
| 0.97 | certified-path | b-5pct | comm_disparity_all | 30 | +0.0000 | nan |
| 0.97 | certified-path | b-5pct | comm_disparity_visible | 30 | +0.0000 | nan |
| 0.97 | certified-path | b-5pct | disparity | 30 | +0.0000 | nan |
| 0.97 | final | b-1se | comm_disparity | 30 | +0.0000 | nan |
| 0.97 | final | b-1se | comm_disparity_all | 0 | +nan | nan |
| 0.97 | final | b-1se | comm_disparity_visible | 0 | +nan | nan |
| 0.97 | final | b-1se | disparity | 30 | +0.0000 | nan |
| 0.97 | final | b-5pct | comm_disparity | 30 | +0.0000 | nan |
| 0.97 | final | b-5pct | comm_disparity_all | 0 | +nan | nan |
| 0.97 | final | b-5pct | comm_disparity_visible | 0 | +nan | nan |
| 0.97 | final | b-5pct | disparity | 30 | +0.0000 | nan |
| 0.99 | certified-path | b-1se | comm_disparity | 30 | +0.0000 | nan |
| 0.99 | certified-path | b-1se | comm_disparity_all | 30 | +0.0000 | nan |
| 0.99 | certified-path | b-1se | comm_disparity_visible | 30 | +0.0000 | nan |
| 0.99 | certified-path | b-1se | disparity | 30 | +0.0000 | nan |
| 0.99 | certified-path | b-5pct | comm_disparity | 30 | +0.0000 | nan |
| 0.99 | certified-path | b-5pct | comm_disparity_all | 30 | +0.0000 | nan |
| 0.99 | certified-path | b-5pct | comm_disparity_visible | 30 | +0.0000 | nan |
| 0.99 | certified-path | b-5pct | disparity | 30 | +0.0000 | nan |
| 0.99 | final | b-1se | comm_disparity | 30 | +0.0000 | nan |
| 0.99 | final | b-1se | comm_disparity_all | 0 | +nan | nan |
| 0.99 | final | b-1se | comm_disparity_visible | 0 | +nan | nan |
| 0.99 | final | b-1se | disparity | 30 | +0.0000 | nan |
| 0.99 | final | b-5pct | comm_disparity | 30 | +0.0000 | nan |
| 0.99 | final | b-5pct | comm_disparity_all | 0 | +nan | nan |
| 0.99 | final | b-5pct | comm_disparity_visible | 0 | +nan | nan |
| 0.99 | final | b-5pct | disparity | 30 | +0.0000 | nan |

## Descriptive: room, coverage and price (as in section 22)

## Room: does the phase act when it may spend more labelled fairness?

Paired with (a)'s deployment in each case (difference = run - (a); negative = lower).

| rho | run | cases | room (median) | phase-2 moves > 0 | deployed set differs from (a) | hidden-community disparity diff (median) | wins / losses | p | labelled disparity diff (median) | spread ratio diff (median) | control reproduces T2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.97 | b-1se | 30 | 0.00020 | 0% | 0% | +0.0000 | 0 / 0 | nan | +0.0000 | +0.0000 | - |
| 0.97 | b-5pct | 30 | 0.00075 | 0% | 0% | +0.0000 | 0 / 0 | nan | +0.0000 | +0.0000 | - |

| 0.99 | b-1se | 30 | 0.00020 | 0% | 0% | +0.0000 | 0 / 0 | nan | +0.0000 | +0.0000 | - |
| 0.99 | b-5pct | 30 | 0.00094 | 0% | 0% | +0.0000 | 0 / 0 | nan | +0.0000 | +0.0000 | - |

## Coverage: does the auditor find the communities that set the hidden disparity?

| rho | run | cases | D_H (median) | D_F (median) | gap to h+ | gap to h- | bound non-vacuous (< 1) | Jaccard with h+ | Jaccard with h- | family size |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.97 | b-1se | 30 | 0.0268 | 0.8171 | 0.2114 | 0.0060 | 37% | 0.000 | 0.000 | 24 |
| 0.97 | b-5pct | 30 | 0.0268 | 0.8171 | 0.2114 | 0.0060 | 37% | 0.000 | 0.000 | 24 |
| 0.99 | b-1se | 30 | 0.0265 | 0.8165 | 0.1640 | 0.0055 | 40% | 0.000 | 0.000 | 26 |
| 0.99 | b-5pct | 30 | 0.0265 | 0.8165 | 0.1640 | 0.0055 | 40% | 0.000 | 0.000 | 26 |

Ground-truth communities against the auditor's conductance cap phi0 (per sample):

| sample | set | communities | conductance median | within phi0 |
|---|---|---|---|---|
| snap_amazon_bfs100000_s1 | evaluated | 70 | 0.064 | 100% |
| snap_amazon_bfs100000_s1 | all | 4047 | 0.538 | 25% |
| snap_amazon_bfs100000_s2 | evaluated | 225 | 0.068 | 100% |
| snap_amazon_bfs100000_s2 | all | 5527 | 0.538 | 29% |
| snap_amazon_bfs100000_s3 | evaluated | 51 | 0.103 | 100% |
| snap_amazon_bfs100000_s3 | all | 3918 | 0.547 | 23% |
| snap_amazon_bfs100000_s4 | evaluated | 278 | 0.041 | 100% |
| snap_amazon_bfs100000_s4 | all | 5973 | 0.506 | 35% |
| snap_amazon_bfs100000_s5 | evaluated | 211 | 0.046 | 100% |
| snap_amazon_bfs100000_s5 | all | 6030 | 0.500 | 35% |
| snap_amazon_bfs100000_s6 | evaluated | 240 | 0.045 | 100% |
| snap_amazon_bfs100000_s6 | all | 6030 | 0.484 | 38% |

## Price: what does community fairness cost in labelled fairness, one swap at a time?

At the labelled-only optimum. price = labelled-disparity cost of the cheapest floor-feasible swap that lowers the community disparity (F: the auditor's first family; H: the hidden communities on sample A). Shown also in units of SE and of D_a.

| rho | cases | price F (median) | in SE | / D_a | none found | price H (median) | in SE | / D_a | none found | scan complete |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.97 | 30 | 0.00016 | 0.8 | 0.975% | 0% | 0.00026 | 1.6 | 2.332% | 0% | 100% |
| 0.99 | 30 | 0.00027 | 1.2 | 1.150% | 3% | 0.00047 | 2.4 | 2.592% | 17% | 100% |

Largest community-disparity decrease one swap can buy within a labelled-cost budget (median over cases; budgets: one SE, then 1%, 5%, 20% of D_a):

| rho | family | se | 1pct | 5pct | 20pct |
|---|---|---|---|---|---|
| 0.97 | F | -0.0001 | -0.0001 | -0.0022 | -0.0392 |
| 0.97 | H | +0.0000 | +0.0000 | -0.0015 | -0.0092 |
| 0.99 | F | +0.0000 | +0.0000 | -0.0008 | -0.0008 |
| 0.99 | H | +0.0000 | +0.0000 | -0.0015 | -0.0028 |
