# C2 diagnostics (EXPLORATORY; docs/design_v2.md section 22)

Specified after T2's pre-registered null and before these runs. T2's verdict stands; nothing here replaces it.

## Room: does the phase act when it may spend more labelled fairness?

Paired with (a)'s deployment in each case (difference = run - (a); negative = lower).

| rho | run | cases | room (median) | phase-2 moves > 0 | deployed set differs from (a) | hidden-community disparity diff (median) | wins / losses | p | labelled disparity diff (median) | spread ratio diff (median) | control reproduces T2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.97 | b-1se | 30 | 0.00039 | 0% | 0% | +0.0000 | 0 / 0 | nan | +0.0000 | +0.0000 | 100% |
| 0.97 | b-1pct | 30 | 0.00371 | 7% | 3% | +0.0000 | 0 / 1 | nan | +0.0000 | +0.0000 | - |
| 0.97 | b-5pct | 30 | 0.01854 | 13% | 3% | +0.0000 | 0 / 1 | nan | +0.0000 | +0.0000 | - |
| 0.97 | b-20pct | 30 | 0.07416 | 23% | 7% | +0.0000 | 1 / 1 | 1 | +0.0000 | +0.0000 | - |

(a) reproduces T2's (a) deployment in 100% of cases at rho 0.97.

| 0.99 | b-1se | 30 | 0.00035 | 0% | 0% | +0.0000 | 0 / 0 | nan | +0.0000 | +0.0000 | 100% |
| 0.99 | b-1pct | 30 | 0.00414 | 3% | 3% | +0.0000 | 0 / 1 | nan | +0.0000 | +0.0000 | - |
| 0.99 | b-5pct | 30 | 0.02070 | 10% | 3% | +0.0000 | 1 / 0 | nan | +0.0000 | +0.0000 | - |
| 0.99 | b-20pct | 30 | 0.08279 | 10% | 3% | +0.0000 | 1 / 0 | nan | +0.0000 | +0.0000 | - |

(a) reproduces T2's (a) deployment in 100% of cases at rho 0.99.

## Coverage: does the auditor find the communities that set the hidden disparity?

| rho | run | cases | D_H (median) | D_F (median) | gap to h+ | gap to h- | bound non-vacuous (< 1) | Jaccard with h+ | Jaccard with h- | family size |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.97 | b-1pct | 30 | 0.4786 | 0.3295 | 1.5242 | 0.4484 | 0% | 0.000 | 0.001 | 8 |
| 0.97 | b-1se | 30 | 0.4786 | 0.3295 | 1.5242 | 0.4484 | 0% | 0.000 | 0.001 | 8 |
| 0.97 | b-20pct | 30 | 0.4786 | 0.3295 | 1.5242 | 0.4484 | 0% | 0.001 | 0.001 | 8 |
| 0.97 | b-5pct | 30 | 0.4786 | 0.3387 | 1.5242 | 0.4484 | 0% | 0.001 | 0.001 | 8 |
| 0.99 | b-1pct | 30 | 0.4773 | 0.3343 | 1.4934 | 0.4460 | 0% | 0.000 | 0.002 | 7 |
| 0.99 | b-1se | 30 | 0.4773 | 0.3343 | 1.4934 | 0.4460 | 0% | 0.000 | 0.002 | 7 |
| 0.99 | b-20pct | 30 | 0.4773 | 0.3343 | 1.4934 | 0.4180 | 0% | 0.000 | 0.002 | 7 |
| 0.99 | b-5pct | 30 | 0.4773 | 0.3343 | 1.4934 | 0.4180 | 0% | 0.000 | 0.002 | 7 |

Ground-truth communities against the auditor's conductance cap phi0 (per sample):

| sample | communities | conductance median | within phi0 |
|---|---|---|---|
| snap_youtube_bfs100000_s1 | 1001 | 0.970 | 0% |
| snap_youtube_bfs100000_s2 | 918 | 0.968 | 0% |
| snap_youtube_bfs100000_s3 | 833 | 0.970 | 0% |
| snap_youtube_bfs100000_s4 | 1003 | 0.970 | 0% |
| snap_youtube_bfs100000_s5 | 991 | 0.969 | 0% |
| snap_youtube_bfs100000_s6 | 1091 | 0.970 | 0% |

## Price: what does community fairness cost in labelled fairness, one swap at a time?

At the labelled-only optimum. price = labelled-disparity cost of the cheapest floor-feasible swap that lowers the community disparity (F: the auditor's first family; H: the hidden communities on sample A). Shown also in units of SE and of D_a.

| rho | cases | price F (median) | in SE | / D_a | none found | price H (median) | in SE | / D_a | none found | scan complete |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.97 | 30 | 0.00028 | 0.6 | 0.087% | 0% | 0.00039 | 0.9 | 0.110% | 0% | 100% |
| 0.99 | 30 | 0.00028 | 0.7 | 0.075% | 13% | 0.00047 | 1.4 | 0.111% | 0% | 100% |

Largest community-disparity decrease one swap can buy within a labelled-cost budget (median over cases; budgets: one SE, then 1%, 5%, 20% of D_a):

| rho | family | se | 1pct | 5pct | 20pct |
|---|---|---|---|---|---|
| 0.97 | F | -0.0001 | -0.0030 | -0.0139 | -0.0201 |
| 0.97 | H | -0.0005 | -0.0056 | -0.0692 | -0.0702 |
| 0.99 | F | -0.0000 | -0.0012 | -0.0089 | -0.0089 |
| 0.99 | H | +0.0000 | -0.0028 | -0.0355 | -0.0355 |
