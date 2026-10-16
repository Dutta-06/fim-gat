# FIM-GAT v2 results (pre-registered analysis)

Tuning (validation only): delta=0.005, phi0=0.4, start=greedy, W3 train=True (code_tag f53a3e10824a).

## T1 (C1): disparity at equal spread

Method: v2-b, greedy start, simple hybrid, certified deployment. 4 test graphs (dataset x attribute), 80 cases.

### real-attribute graphs (headline)

| rho | comparison | test graphs | median of mean difference | p | verdict | case wins / ties / losses |
|---|---|---|---|---|---|---|
| 0.97 | best_baseline | 4 | -0.0075 | 0.875 | no difference | 15 / 0 / 5 |
| 0.97 | BSM | 4 | -0.0992 | 0.125 | no difference | 19 / 0 / 1 |
| 0.97 | Degree | 4 | -0.0739 | 0.125 | no difference | 20 / 0 / 0 |
| 0.97 | FW-Disparity | 4 | -0.0135 | 0.25 | no difference | 15 / 0 / 5 |
| 0.97 | Greedy | 4 | -0.0607 | 0.125 | no difference | 20 / 0 / 0 |
| 0.97 | Maximin-Greedy | 4 | -0.0274 | 0.125 | no difference | 13 / 0 / 0 |
| 0.97 | MultiFW-MaxMin | 4 | -0.0667 | 0.125 | no difference | 19 / 0 / 1 |
| 0.97 | Parity-Greedy | 4 | -0.0152 | 0.125 | no difference | 16 / 0 / 2 |
| 0.97 | Prior | 1 | -0.0940 | nan | n/a | 2 / 0 / 0 |
| 0.97 | RFIM | 2 | -0.2637 | 0.5 | no difference | 6 / 0 / 0 |
| 0.97 | Welfare-Greedy | 4 | -0.0540 | 0.125 | no difference | 20 / 0 / 0 |
| 0.98 | best_baseline | 4 | -0.0075 | 0.25 | no difference | 15 / 0 / 5 |
| 0.98 | BSM | 4 | -0.0884 | 0.125 | no difference | 20 / 0 / 0 |
| 0.98 | Degree | 4 | -0.0682 | 0.125 | no difference | 20 / 0 / 0 |
| 0.98 | FW-Disparity | 4 | -0.0127 | 0.125 | no difference | 16 / 0 / 4 |
| 0.98 | Greedy | 4 | -0.0550 | 0.125 | no difference | 20 / 0 / 0 |
| 0.98 | Maximin-Greedy | 4 | -0.0271 | 0.125 | no difference | 6 / 0 / 0 |
| 0.98 | MultiFW-MaxMin | 3 | -0.0666 | 0.25 | no difference | 12 / 0 / 1 |
| 0.98 | Parity-Greedy | 4 | -0.0133 | 0.125 | no difference | 14 / 0 / 2 |
| 0.98 | RFIM | 2 | -0.3103 | 0.5 | no difference | 4 / 0 / 0 |
| 0.98 | Welfare-Greedy | 4 | -0.0483 | 0.125 | no difference | 20 / 0 / 0 |
| 0.99 | best_baseline | 4 | -0.0077 | 0.125 | no difference | 14 / 0 / 6 |
| 0.99 | BSM | 4 | -0.0729 | 0.125 | no difference | 20 / 0 / 0 |
| 0.99 | Degree | 4 | -0.0518 | 0.125 | no difference | 17 / 0 / 0 |
| 0.99 | FW-Disparity | 4 | -0.0228 | 0.125 | no difference | 17 / 0 / 3 |
| 0.99 | Greedy | 4 | -0.0419 | 0.125 | no difference | 20 / 0 / 0 |
| 0.99 | Maximin-Greedy | 2 | -0.0595 | 0.5 | no difference | 2 / 0 / 0 |
| 0.99 | MultiFW-MaxMin | 3 | -0.0736 | 0.25 | no difference | 7 / 0 / 0 |
| 0.99 | Parity-Greedy | 4 | -0.0117 | 0.25 | no difference | 9 / 0 / 4 |
| 0.99 | Welfare-Greedy | 4 | -0.0352 | 0.125 | no difference | 20 / 0 / 0 |
| 1 | best_baseline | 4 | +0.0000 | nan | n/a | 0 / 18 / 0 |
| 1 | BSM | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |
| 1 | FW-Disparity | 4 | +0.0000 | nan | n/a | 0 / 18 / 0 |
| 1 | Greedy | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |
| 1 | Welfare-Greedy | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |

Holds across the range (a significant win at every rho in [0.97, 0.98, 0.99]; rho = 1.0 is reported above but not required, design section 21):

- best_baseline: no
- BSM: no
- Degree: no
- FW-Disparity: no
- Greedy: no
- Maximin-Greedy: no
- MultiFW-MaxMin: no
- Parity-Greedy: no
- Prior: no (no paired cases at rho [0.98, 0.99])
- RFIM: no (no paired cases at rho [0.99])
- Welfare-Greedy: no

### degree-group graphs (supplement)

No rows yet.

### all test graphs

| rho | comparison | test graphs | median of mean difference | p | verdict | case wins / ties / losses |
|---|---|---|---|---|---|---|
| 0.97 | best_baseline | 4 | -0.0075 | 0.875 | no difference | 15 / 0 / 5 |
| 0.97 | BSM | 4 | -0.0992 | 0.125 | no difference | 19 / 0 / 1 |
| 0.97 | Degree | 4 | -0.0739 | 0.125 | no difference | 20 / 0 / 0 |
| 0.97 | FW-Disparity | 4 | -0.0135 | 0.25 | no difference | 15 / 0 / 5 |
| 0.97 | Greedy | 4 | -0.0607 | 0.125 | no difference | 20 / 0 / 0 |
| 0.97 | Maximin-Greedy | 4 | -0.0274 | 0.125 | no difference | 13 / 0 / 0 |
| 0.97 | MultiFW-MaxMin | 4 | -0.0667 | 0.125 | no difference | 19 / 0 / 1 |
| 0.97 | Parity-Greedy | 4 | -0.0152 | 0.125 | no difference | 16 / 0 / 2 |
| 0.97 | Prior | 1 | -0.0940 | nan | n/a | 2 / 0 / 0 |
| 0.97 | RFIM | 2 | -0.2637 | 0.5 | no difference | 6 / 0 / 0 |
| 0.97 | Welfare-Greedy | 4 | -0.0540 | 0.125 | no difference | 20 / 0 / 0 |
| 0.98 | best_baseline | 4 | -0.0075 | 0.25 | no difference | 15 / 0 / 5 |
| 0.98 | BSM | 4 | -0.0884 | 0.125 | no difference | 20 / 0 / 0 |
| 0.98 | Degree | 4 | -0.0682 | 0.125 | no difference | 20 / 0 / 0 |
| 0.98 | FW-Disparity | 4 | -0.0127 | 0.125 | no difference | 16 / 0 / 4 |
| 0.98 | Greedy | 4 | -0.0550 | 0.125 | no difference | 20 / 0 / 0 |
| 0.98 | Maximin-Greedy | 4 | -0.0271 | 0.125 | no difference | 6 / 0 / 0 |
| 0.98 | MultiFW-MaxMin | 3 | -0.0666 | 0.25 | no difference | 12 / 0 / 1 |
| 0.98 | Parity-Greedy | 4 | -0.0133 | 0.125 | no difference | 14 / 0 / 2 |
| 0.98 | RFIM | 2 | -0.3103 | 0.5 | no difference | 4 / 0 / 0 |
| 0.98 | Welfare-Greedy | 4 | -0.0483 | 0.125 | no difference | 20 / 0 / 0 |
| 0.99 | best_baseline | 4 | -0.0077 | 0.125 | no difference | 14 / 0 / 6 |
| 0.99 | BSM | 4 | -0.0729 | 0.125 | no difference | 20 / 0 / 0 |
| 0.99 | Degree | 4 | -0.0518 | 0.125 | no difference | 17 / 0 / 0 |
| 0.99 | FW-Disparity | 4 | -0.0228 | 0.125 | no difference | 17 / 0 / 3 |
| 0.99 | Greedy | 4 | -0.0419 | 0.125 | no difference | 20 / 0 / 0 |
| 0.99 | Maximin-Greedy | 2 | -0.0595 | 0.5 | no difference | 2 / 0 / 0 |
| 0.99 | MultiFW-MaxMin | 3 | -0.0736 | 0.25 | no difference | 7 / 0 / 0 |
| 0.99 | Parity-Greedy | 4 | -0.0117 | 0.25 | no difference | 9 / 0 / 4 |
| 0.99 | Welfare-Greedy | 4 | -0.0352 | 0.125 | no difference | 20 / 0 / 0 |
| 1 | best_baseline | 4 | +0.0000 | nan | n/a | 0 / 18 / 0 |
| 1 | BSM | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |
| 1 | FW-Disparity | 4 | +0.0000 | nan | n/a | 0 / 18 / 0 |
| 1 | Greedy | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |
| 1 | Welfare-Greedy | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |

Holds across the range (a significant win at every rho in [0.97, 0.98, 0.99]; rho = 1.0 is reported above but not required, design section 21):

- best_baseline: no
- BSM: no
- Degree: no
- FW-Disparity: no
- Greedy: no
- Maximin-Greedy: no
- MultiFW-MaxMin: no
- Parity-Greedy: no
- Prior: no (no paired cases at rho [0.98, 0.99])
- RFIM: no (no paired cases at rho [0.99])
- Welfare-Greedy: no

### Ablation (one component changed; difference = method - row, negative favours the method)

| rho | row | test graphs | mean disparity (row) | median of mean difference | p | verdict |
|---|---|---|---|---|---|---|
| 0.97 | start: learned | 0 | n/a | n/a | n/a | not run |
| 0.97 | proposer: exhaustive | 0 | n/a | n/a | n/a | not run |
| 0.97 | acceptance: (a) labelled only | 0 | n/a | n/a | n/a | not run |
| 0.97 | acceptance: (b0) published rule | 0 | n/a | n/a | n/a | not run |
| 0.97 | deployment: final search set | 4 | 0.0458 | +0.0021 | 0.25 | no difference |
| 0.97 | deployment: path rule (T2's) | 4 | 0.0490 | -0.0001 | 0.625 | no difference |
| 0.98 | start: learned | 0 | n/a | n/a | n/a | not run |
| 0.98 | proposer: exhaustive | 0 | n/a | n/a | n/a | not run |
| 0.98 | acceptance: (a) labelled only | 0 | n/a | n/a | n/a | not run |
| 0.98 | acceptance: (b0) published rule | 0 | n/a | n/a | n/a | not run |
| 0.98 | deployment: final search set | 4 | 0.0469 | +0.0024 | 0.25 | no difference |
| 0.98 | deployment: path rule (T2's) | 4 | 0.0476 | +0.0024 | 0.25 | no difference |
| 0.99 | start: learned | 0 | n/a | n/a | n/a | not run |
| 0.99 | proposer: exhaustive | 0 | n/a | n/a | n/a | not run |
| 0.99 | acceptance: (a) labelled only | 0 | n/a | n/a | n/a | not run |
| 0.99 | acceptance: (b0) published rule | 0 | n/a | n/a | n/a | not run |
| 0.99 | deployment: final search set | 4 | 0.0633 | +0.0014 | 0.375 | no difference |
| 0.99 | deployment: path rule (T2's) | 4 | 0.0633 | +0.0014 | 0.5 | no difference |
| 1 | start: learned | 0 | n/a | n/a | n/a | not run |
| 1 | proposer: exhaustive | 0 | n/a | n/a | n/a | not run |
| 1 | acceptance: (a) labelled only | 0 | n/a | n/a | n/a | not run |
| 1 | acceptance: (b0) published rule | 0 | n/a | n/a | n/a | not run |
| 1 | deployment: final search set | 4 | 0.1008 | +0.0281 | 0.125 | no difference |
| 1 | deployment: path rule (T2's) | 4 | 0.1260 | +0.0000 | nan | n/a |

Final search sets below the floor by Monte Carlo (es_ratio < rho): 17 of 80; not certified: 19 of 80.

### Results matrix (mean disparity over certified budgets; n = budgets certified; - = none; bold = lowest among the methods certified at every budget; the paired tests above are the formal comparison)

**rho = 0.97**

| test graph | FIM-GAT v2 | AGE | BSM | Degree | FW-Disparity | Greedy | Maximin-Greedy | MultiFW-MaxMin | Parity-Greedy | Prior | RFIM | Welfare-Greedy | lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rice_facebook / age | **0.0156 (5/5)** | - | 0.1768 (5/5) | 0.0870 (5/5) | 0.0286 (5/5) | 0.0720 (5/5) | 0.0357 (3/5) | 0.0853 (5/5) | 0.0320 (5/5) | - | - | 0.0663 (5/5) | 0.0000 |
| rice_facebook / college | **0.0058 (5/5)** | - | 0.0160 (5/5) | 0.0499 (5/5) | 0.0199 (5/5) | 0.0422 (5/5) | 0.0160 (5/5) | 0.0188 (5/5) | 0.0198 (5/5) | - | 0.1651 (3/5) | 0.0331 (5/5) | 0.0000 |
| rice_facebook / major | 0.1474 (5/5) | - | 0.3875 (5/5) | 0.3425 (5/5) | **0.1408 (5/5)** | 0.3230 (5/5) | 0.2999 (4/5) | 0.5424 (5/5) | 0.1592 (4/5) | - | 0.5272 (3/5) | 0.3041 (5/5) | 0.0000 |
| twitter_political / leaning | **0.0270 (5/5)** | - | 0.0641 (5/5) | 0.1033 (5/5) | 0.0563 (5/5) | 0.0920 (5/5) | 0.0324 (1/5) | 0.0908 (5/5) | 0.0340 (4/5) | 0.0952 (2/5) | - | 0.0842 (5/5) | 0.0000 |

**rho = 0.98**

| test graph | FIM-GAT v2 | AGE | BSM | Degree | FW-Disparity | Greedy | Maximin-Greedy | MultiFW-MaxMin | Parity-Greedy | Prior | RFIM | Welfare-Greedy | lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rice_facebook / age | **0.0186 (5/5)** | - | 0.1591 (5/5) | 0.0870 (5/5) | 0.0311 (5/5) | 0.0720 (5/5) | 0.0386 (1/5) | 0.0853 (5/5) | 0.0344 (4/5) | - | - | 0.0663 (5/5) | 0.0000 |
| rice_facebook / college | **0.0071 (5/5)** | - | 0.0160 (5/5) | 0.0499 (5/5) | 0.0200 (5/5) | 0.0422 (5/5) | 0.0284 (2/5) | 0.0188 (5/5) | 0.0198 (5/5) | - | 0.1952 (2/5) | 0.0331 (5/5) | 0.0000 |
| rice_facebook / major | **0.1398 (5/5)** | - | 0.4209 (5/5) | 0.3425 (5/5) | 0.1417 (5/5) | 0.3230 (5/5) | 0.3249 (2/5) | 0.4600 (3/5) | 0.1839 (3/5) | - | 0.5966 (2/5) | 0.3041 (5/5) | 0.0000 |
| twitter_political / leaning | **0.0353 (5/5)** | - | 0.0716 (5/5) | 0.1033 (5/5) | 0.0597 (5/5) | 0.0920 (5/5) | 0.0324 (1/5) | - | 0.0340 (4/5) | - | - | 0.0842 (5/5) | 0.0000 |

**rho = 0.99**

| test graph | FIM-GAT v2 | AGE | BSM | Degree | FW-Disparity | Greedy | Maximin-Greedy | MultiFW-MaxMin | Parity-Greedy | Prior | RFIM | Welfare-Greedy | lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rice_facebook / age | **0.0245 (5/5)** | - | 0.1349 (5/5) | 0.0870 (5/5) | 0.0367 (5/5) | 0.0720 (5/5) | - | 0.0832 (1/5) | 0.0338 (3/5) | - | - | 0.0663 (5/5) | 0.0000 |
| rice_facebook / college | **0.0087 (5/5)** | - | 0.0160 (5/5) | 0.0499 (5/5) | 0.0190 (5/5) | 0.0422 (5/5) | 0.0336 (1/5) | 0.0188 (5/5) | 0.0198 (5/5) | - | - | 0.0331 (5/5) | 0.0000 |
| rice_facebook / major | **0.1723 (5/5)** | - | 0.4334 (5/5) | 0.3425 (5/5) | 0.2254 (5/5) | 0.3230 (5/5) | 0.3145 (1/5) | 0.2215 (1/5) | 0.2068 (2/5) | - | - | 0.3041 (5/5) | 0.0000 |
| twitter_political / leaning | **0.0557 (5/5)** | - | 0.0911 (5/5) | 0.1185 (2/5) | 0.0890 (5/5) | 0.0920 (5/5) | - | - | 0.0269 (3/5) | - | - | 0.0842 (5/5) | 0.0000 |

**rho = 1**

| test graph | FIM-GAT v2 | AGE | BSM | Degree | FW-Disparity | Greedy | Maximin-Greedy | MultiFW-MaxMin | Parity-Greedy | Prior | RFIM | Welfare-Greedy | lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rice_facebook / age | **0.0690 (5/5)** | - | - | - | 0.0779 (4/5) | - | - | - | - | - | - | - | 0.0000 |
| rice_facebook / college | **0.0377 (5/5)** | - | - | - | 0.0345 (4/5) | - | - | - | - | - | - | - | 0.0000 |
| rice_facebook / major | **0.3040 (5/5)** | - | - | - | **0.3040 (5/5)** | - | - | - | - | - | - | - | 0.0000 |
| twitter_political / leaning | **0.0931 (5/5)** | - | 0.1349 (1/5) | - | **0.0931 (5/5)** | 0.1349 (1/5) | - | - | - | - | - | 0.1349 (1/5) | 0.0000 |

Certificate pass rate per method (share of all its sets):

| method | rho 0.97 | rho 0.98 | rho 0.99 | rho 1 |
|---|---|---|---|---|
| AGE | 0% | 0% | 0% | 0% |
| BSM | 100% | 100% | 100% | 5% |
| Degree | 100% | 100% | 85% | 0% |
| FIM-GAT v2 | 100% | 100% | 100% | 100% |
| FW-Disparity | 100% | 100% | 100% | 90% |
| Greedy | 100% | 100% | 100% | 5% |
| Maximin-Greedy | 65% | 30% | 10% | 0% |
| MultiFW-MaxMin | 100% | 65% | 35% | 0% |
| Parity-Greedy | 90% | 80% | 65% | 0% |
| Prior | 10% | 0% | 0% | 0% |
| RFIM | 30% | 20% | 0% | 0% |
| Welfare-Greedy | 100% | 100% | 100% | 5% |

## Spread-disparity trade-off (descriptive)

Per test graph, over budgets: the external baselines' sets (T1 rows and BSM's whole tau sweep) that the method's curve (its deployments at rho in [0.97, 0.98, 0.99, 1.0]) dominates, that dominate one of its points, and that are incomparable, with how many of those have less spread than every method point. Point estimates from Monte Carlo; the paired tests of T1 remain the formal comparison.

| test graph | baseline sets | dominated by the method | dominate a method point | incomparable | of which below the method's least spread |
|---|---|---|---|---|---|
| rice_facebook / age | 132 | 99 (75%) | 6 (5%) | 27 | 0 |
| rice_facebook / college | 123 | 57 (46%) | 7 (6%) | 59 | 0 |
| rice_facebook / major | 131 | 87 (66%) | 14 (11%) | 32 | 2 |
| twitter_political / leaning | 125 | 65 (52%) | 12 (10%) | 48 | 2 |

## T2 (C2): hidden-community disparity, (b) against (a)

Primary: rho 0.97, sets deployed by the path rule (certified-path); every other row is secondary.

| rho | sets | comparison | metric | cases | median difference | p | verdict |
|---|---|---|---|---|---|---|---|
| 0.97 | certified-path | v2-b - v2-a | comm_disparity | 30 | +0.0000 | nan | n/a |
| 0.97 | certified-path | v2-b - v2-a | disparity | 30 | +0.0000 | nan | no significant rise |
| 0.97 | certified-path | v2-b0 - v2-a | comm_disparity | 30 | +0.0000 | 0.455 | no difference |
| 0.97 | certified-path | v2-b0 - v2-a | disparity | 30 | +0.0000 | 0.122 | no significant rise |
| 0.97 | certified | v2-b - v2-a | comm_disparity | 30 | +0.0000 | 0.317 | no difference |
| 0.97 | certified | v2-b - v2-a | disparity | 30 | +0.0000 | 0.317 | no significant rise |
| 0.97 | certified | v2-b0 - v2-a | comm_disparity | 30 | +0.0000 | 0.414 | no difference |
| 0.97 | certified | v2-b0 - v2-a | disparity | 30 | +0.0000 | 0.122 | no significant rise |
| 0.97 | final | v2-b - v2-a | comm_disparity | 30 | +0.0000 | nan | n/a |
| 0.97 | final | v2-b - v2-a | disparity | 30 | +0.0000 | nan | no significant rise |
| 0.97 | final | v2-b0 - v2-a | comm_disparity | 30 | +0.0000 | 0.455 | no difference |
| 0.97 | final | v2-b0 - v2-a | disparity | 30 | +0.0000 | 0.122 | no significant rise |
| 0.99 | certified-path | v2-b - v2-a | comm_disparity | 30 | +0.0000 | nan | n/a |
| 0.99 | certified-path | v2-b - v2-a | disparity | 30 | +0.0000 | nan | no significant rise |
| 0.99 | certified-path | v2-b0 - v2-a | comm_disparity | 30 | +0.0000 | 0.566 | no difference |
| 0.99 | certified-path | v2-b0 - v2-a | disparity | 30 | +0.0001 | 0.0037 | significant rise |
| 0.99 | certified | v2-b - v2-a | comm_disparity | 30 | +0.0000 | 0.317 | no difference |
| 0.99 | certified | v2-b - v2-a | disparity | 30 | +0.0000 | 0.317 | no significant rise |
| 0.99 | certified | v2-b0 - v2-a | comm_disparity | 30 | +0.0000 | 0.37 | no difference |
| 0.99 | certified | v2-b0 - v2-a | disparity | 30 | +0.0001 | 0.00359 | significant rise |
| 0.99 | final | v2-b - v2-a | comm_disparity | 30 | +0.0000 | nan | n/a |
| 0.99 | final | v2-b - v2-a | disparity | 30 | +0.0000 | nan | no significant rise |
| 0.99 | final | v2-b0 - v2-a | comm_disparity | 30 | +0.0000 | 0.566 | no difference |
| 0.99 | final | v2-b0 - v2-a | disparity | 30 | +0.0001 | 0.0037 | significant rise |

T2 pass (pre-registered rule): no (hidden communities: n/a; labelled disparity: no significant rise; 30 cases).


Secondary, held-out attributes (T1 rows, path-rule deployments):

- rho 0.97: v2-b - v2-a held-out-attribute disparity over 0 cases: median +nan, p nan (n/a)
- rho 0.99: v2-b - v2-a held-out-attribute disparity over 0 cases: median +nan, p nan (n/a)

## T3: does the learning add anything?

The W3 rule asked for a learned proposer, but there are no T3 rows yet (train_proposer.py, then run_t3.py).

## T4 (C3): runtime, cached and per query

No T4 rows yet.

## Certificate-size sensitivity

No t1_certsens.csv rows yet.

