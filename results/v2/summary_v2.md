# FIM-GAT v2 results (pre-registered analysis)

Tuning (validation only): delta=0.005, phi0=0.4, start=greedy, W3 train=True (code_tag f53a3e10824a).

## T1 (C1): disparity at equal spread

Method: v2-b, greedy start, simple hybrid, certified deployment. 11 test graphs (dataset x attribute), 220 cases.

### real-attribute graphs (headline)

| rho | comparison | test graphs | median of mean difference | p | verdict | case wins / ties / losses |
|---|---|---|---|---|---|---|
| 0.97 | best_baseline | 8 | -0.0044 | 0.383 | no difference | 28 / 0 / 12 |
| 0.97 | BSM | 8 | -0.0132 | 0.0234 | win | 34 / 0 / 6 |
| 0.97 | Degree | 5 | -0.0714 | 0.0625 | no difference | 25 / 0 / 0 |
| 0.97 | FW-Disparity | 8 | -0.0123 | 0.0156 | win | 35 / 0 / 5 |
| 0.97 | Greedy | 8 | -0.0360 | 0.00781 | win | 40 / 0 / 0 |
| 0.97 | Maximin-Greedy | 6 | -0.0173 | 0.0625 | no difference | 14 / 0 / 2 |
| 0.97 | MultiFW-MaxMin | 8 | -0.0076 | 0.0156 | win | 27 / 0 / 4 |
| 0.97 | Parity-Greedy | 8 | -0.0130 | 0.00781 | win | 33 / 0 / 3 |
| 0.97 | Prior | 1 | -0.0940 | nan | n/a | 2 / 0 / 0 |
| 0.97 | RFIM | 3 | -0.1597 | 0.25 | no difference | 10 / 0 / 0 |
| 0.97 | Welfare-Greedy | 8 | -0.0280 | 0.00781 | win | 40 / 0 / 0 |
| 0.98 | best_baseline | 8 | -0.0041 | 0.148 | no difference | 27 / 0 / 13 |
| 0.98 | BSM | 8 | -0.0125 | 0.0156 | win | 37 / 0 / 3 |
| 0.98 | Degree | 5 | -0.0680 | 0.0625 | no difference | 25 / 0 / 0 |
| 0.98 | FW-Disparity | 8 | -0.0114 | 0.00781 | win | 35 / 0 / 5 |
| 0.98 | Greedy | 8 | -0.0310 | 0.00781 | win | 40 / 0 / 0 |
| 0.98 | Maximin-Greedy | 6 | -0.0212 | 0.156 | no difference | 6 / 0 / 2 |
| 0.98 | MultiFW-MaxMin | 6 | -0.0066 | 0.0625 | no difference | 18 / 0 / 2 |
| 0.98 | Parity-Greedy | 8 | -0.0129 | 0.00781 | win | 30 / 0 / 3 |
| 0.98 | RFIM | 3 | -0.1896 | 0.25 | no difference | 6 / 0 / 0 |
| 0.98 | Welfare-Greedy | 8 | -0.0230 | 0.00781 | win | 40 / 0 / 0 |
| 0.99 | best_baseline | 8 | -0.0044 | 0.0391 | win | 27 / 0 / 13 |
| 0.99 | BSM | 8 | -0.0100 | 0.00781 | win | 39 / 0 / 1 |
| 0.99 | Degree | 5 | -0.0412 | 0.0625 | no difference | 22 / 0 / 0 |
| 0.99 | FW-Disparity | 8 | -0.0103 | 0.00781 | win | 35 / 0 / 5 |
| 0.99 | Greedy | 8 | -0.0243 | 0.00781 | win | 40 / 0 / 0 |
| 0.99 | Maximin-Greedy | 2 | -0.0595 | 0.5 | no difference | 2 / 0 / 0 |
| 0.99 | MultiFW-MaxMin | 5 | -0.0101 | 0.188 | no difference | 12 / 0 / 1 |
| 0.99 | Parity-Greedy | 8 | -0.0090 | 0.0234 | win | 20 / 0 / 8 |
| 0.99 | Welfare-Greedy | 8 | -0.0172 | 0.00781 | win | 40 / 0 / 0 |
| 1 | best_baseline | 8 | +0.0000 | 1 | no difference | 1 / 33 / 1 |
| 1 | BSM | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |
| 1 | Degree | 1 | -0.0057 | nan | n/a | 1 / 0 / 1 |
| 1 | FW-Disparity | 8 | +0.0000 | nan | n/a | 0 / 33 / 0 |
| 1 | Greedy | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |
| 1 | Welfare-Greedy | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |

Holds across the range (a significant win at every rho in [0.97, 0.98, 0.99]; rho = 1.0 is reported above but not required, design section 21):

- best_baseline: no
- BSM: yes
- Degree: no
- FW-Disparity: yes
- Greedy: yes
- Maximin-Greedy: no
- MultiFW-MaxMin: no
- Parity-Greedy: yes
- Prior: no (no paired cases at rho [0.98, 0.99])
- RFIM: no (no paired cases at rho [0.99])
- Welfare-Greedy: yes

### degree-group graphs (supplement)

| rho | comparison | test graphs | median of mean difference | p | verdict | case wins / ties / losses |
|---|---|---|---|---|---|---|
| 0.97 | best_baseline | 3 | +0.0002 | 0.75 | no difference | 7 / 0 / 8 |
| 0.97 | BSM | 3 | -0.0026 | 0.5 | no difference | 13 / 0 / 2 |
| 0.97 | Degree | 3 | -0.0049 | 0.5 | no difference | 11 / 0 / 2 |
| 0.97 | FW-Disparity | 3 | -0.0067 | 0.25 | no difference | 14 / 0 / 1 |
| 0.97 | Greedy | 3 | -0.0078 | 0.25 | no difference | 15 / 0 / 0 |
| 0.97 | Maximin-Greedy | 3 | -0.0003 | 1 | no difference | 10 / 0 / 4 |
| 0.97 | MultiFW-MaxMin | 3 | -0.0072 | 0.5 | no difference | 12 / 0 / 1 |
| 0.97 | RFIM | 1 | -0.0088 | nan | n/a | 5 / 0 / 0 |
| 0.97 | Welfare-Greedy | 3 | -0.0078 | 0.25 | no difference | 15 / 0 / 0 |
| 0.98 | best_baseline | 3 | -0.0002 | 0.75 | no difference | 8 / 0 / 7 |
| 0.98 | BSM | 3 | -0.0021 | 0.25 | no difference | 12 / 0 / 3 |
| 0.98 | Degree | 3 | -0.0031 | 0.5 | no difference | 6 / 0 / 2 |
| 0.98 | FW-Disparity | 3 | -0.0062 | 0.25 | no difference | 15 / 0 / 0 |
| 0.98 | Greedy | 3 | -0.0060 | 0.25 | no difference | 15 / 0 / 0 |
| 0.98 | Maximin-Greedy | 3 | -0.0001 | 0.75 | no difference | 7 / 0 / 4 |
| 0.98 | MultiFW-MaxMin | 3 | -0.0057 | 0.5 | no difference | 10 / 0 / 1 |
| 0.98 | RFIM | 1 | -0.0070 | nan | n/a | 5 / 0 / 0 |
| 0.98 | Welfare-Greedy | 3 | -0.0060 | 0.25 | no difference | 15 / 0 / 0 |
| 0.99 | best_baseline | 3 | -0.0012 | 0.5 | no difference | 9 / 0 / 6 |
| 0.99 | BSM | 3 | -0.0033 | 0.5 | no difference | 11 / 0 / 4 |
| 0.99 | FW-Disparity | 3 | -0.0045 | 0.25 | no difference | 15 / 0 / 0 |
| 0.99 | Greedy | 3 | -0.0047 | 0.25 | no difference | 15 / 0 / 0 |
| 0.99 | Maximin-Greedy | 2 | +0.0006 | 0.5 | no difference | 1 / 0 / 6 |
| 0.99 | MultiFW-MaxMin | 2 | -0.0041 | 0.5 | no difference | 8 / 0 / 0 |
| 0.99 | RFIM | 1 | -0.0057 | nan | n/a | 5 / 0 / 0 |
| 0.99 | Welfare-Greedy | 3 | -0.0047 | 0.25 | no difference | 15 / 0 / 0 |
| 1 | best_baseline | 3 | +0.0000 | 1 | no difference | 0 / 12 / 1 |
| 1 | BSM | 1 | +0.0000 | nan | n/a | 0 / 1 / 0 |
| 1 | FW-Disparity | 3 | +0.0000 | nan | n/a | 0 / 13 / 0 |
| 1 | Greedy | 2 | -0.0001 | 1 | no difference | 1 / 1 / 0 |
| 1 | Welfare-Greedy | 2 | +0.0001 | 1 | no difference | 0 / 1 / 1 |

Holds across the range (a significant win at every rho in [0.97, 0.98, 0.99]; rho = 1.0 is reported above but not required, design section 21):

- best_baseline: no
- BSM: no
- Degree: no (no paired cases at rho [0.99])
- FW-Disparity: no
- Greedy: no
- Maximin-Greedy: no
- MultiFW-MaxMin: no
- RFIM: no
- Welfare-Greedy: no

### all test graphs

| rho | comparison | test graphs | median of mean difference | p | verdict | case wins / ties / losses |
|---|---|---|---|---|---|---|
| 0.97 | best_baseline | 11 | -0.0008 | 0.413 | no difference | 35 / 0 / 20 |
| 0.97 | BSM | 11 | -0.0048 | 0.0137 | win | 47 / 0 / 8 |
| 0.97 | Degree | 8 | -0.0297 | 0.0156 | win | 36 / 0 / 2 |
| 0.97 | FW-Disparity | 11 | -0.0104 | 0.00293 | win | 49 / 0 / 6 |
| 0.97 | Greedy | 11 | -0.0214 | 0.000977 | win | 55 / 0 / 0 |
| 0.97 | Maximin-Greedy | 9 | -0.0024 | 0.0547 | no difference | 24 / 0 / 6 |
| 0.97 | MultiFW-MaxMin | 11 | -0.0072 | 0.0137 | win | 39 / 0 / 5 |
| 0.97 | Parity-Greedy | 8 | -0.0130 | 0.00781 | win | 33 / 0 / 3 |
| 0.97 | Prior | 1 | -0.0940 | nan | n/a | 2 / 0 / 0 |
| 0.97 | RFIM | 4 | -0.0902 | 0.125 | no difference | 15 / 0 / 0 |
| 0.97 | Welfare-Greedy | 11 | -0.0152 | 0.000977 | win | 55 / 0 / 0 |
| 0.98 | best_baseline | 11 | -0.0012 | 0.175 | no difference | 35 / 0 / 20 |
| 0.98 | BSM | 11 | -0.0041 | 0.00293 | win | 49 / 0 / 6 |
| 0.98 | Degree | 8 | -0.0288 | 0.0156 | win | 31 / 0 / 2 |
| 0.98 | FW-Disparity | 11 | -0.0081 | 0.000977 | win | 50 / 0 / 5 |
| 0.98 | Greedy | 11 | -0.0177 | 0.000977 | win | 55 / 0 / 0 |
| 0.98 | Maximin-Greedy | 9 | -0.0019 | 0.129 | no difference | 13 / 0 / 6 |
| 0.98 | MultiFW-MaxMin | 9 | -0.0057 | 0.0195 | win | 28 / 0 / 3 |
| 0.98 | Parity-Greedy | 8 | -0.0129 | 0.00781 | win | 30 / 0 / 3 |
| 0.98 | RFIM | 4 | -0.1053 | 0.125 | no difference | 11 / 0 / 0 |
| 0.98 | Welfare-Greedy | 11 | -0.0115 | 0.000977 | win | 55 / 0 / 0 |
| 0.99 | best_baseline | 11 | -0.0025 | 0.0137 | win | 36 / 0 / 19 |
| 0.99 | BSM | 11 | -0.0056 | 0.00195 | win | 50 / 0 / 5 |
| 0.99 | Degree | 5 | -0.0412 | 0.0625 | no difference | 22 / 0 / 0 |
| 0.99 | FW-Disparity | 11 | -0.0102 | 0.000977 | win | 50 / 0 / 5 |
| 0.99 | Greedy | 11 | -0.0128 | 0.000977 | win | 55 / 0 / 0 |
| 0.99 | Maximin-Greedy | 4 | -0.0062 | 0.625 | no difference | 3 / 0 / 6 |
| 0.99 | MultiFW-MaxMin | 7 | -0.0043 | 0.0469 | win | 20 / 0 / 1 |
| 0.99 | Parity-Greedy | 8 | -0.0090 | 0.0234 | win | 20 / 0 / 8 |
| 0.99 | RFIM | 1 | -0.0057 | nan | n/a | 5 / 0 / 0 |
| 0.99 | Welfare-Greedy | 11 | -0.0084 | 0.000977 | win | 55 / 0 / 0 |
| 1 | best_baseline | 11 | +0.0000 | 1 | no difference | 1 / 45 / 2 |
| 1 | BSM | 2 | +0.0000 | nan | n/a | 0 / 2 / 0 |
| 1 | Degree | 1 | -0.0057 | nan | n/a | 1 / 0 / 1 |
| 1 | FW-Disparity | 11 | +0.0000 | nan | n/a | 0 / 46 / 0 |
| 1 | Greedy | 3 | +0.0000 | 1 | no difference | 1 / 2 / 0 |
| 1 | Welfare-Greedy | 3 | +0.0000 | 1 | no difference | 0 / 2 / 1 |

Holds across the range (a significant win at every rho in [0.97, 0.98, 0.99]; rho = 1.0 is reported above but not required, design section 21):

- best_baseline: no
- BSM: yes
- Degree: no
- FW-Disparity: yes
- Greedy: yes
- Maximin-Greedy: no
- MultiFW-MaxMin: yes
- Parity-Greedy: yes
- Prior: no (no paired cases at rho [0.98, 0.99])
- RFIM: no
- Welfare-Greedy: yes

### Ablation (one component changed; difference = method - row, negative favours the method)

| rho | row | test graphs | mean disparity (row) | median of mean difference | p | verdict |
|---|---|---|---|---|---|---|
| 0.97 | start: learned | 0 | n/a | n/a | n/a | not run |
| 0.97 | proposer: exhaustive | 11 | 0.0279 | -0.0005 | 0.147 | no difference |
| 0.97 | acceptance: (a) labelled only | 11 | 0.0298 | -0.0002 | 0.164 | no difference |
| 0.97 | acceptance: (b0) published rule | 11 | 0.0330 | -0.0010 | 0.00195 | win |
| 0.97 | deployment: final search set | 11 | 0.0279 | +0.0000 | 0.557 | no difference |
| 0.97 | deployment: path rule (T2's) | 11 | 0.0291 | -0.0000 | 0.219 | no difference |
| 0.98 | start: learned | 0 | n/a | n/a | n/a | not run |
| 0.98 | proposer: exhaustive | 11 | 0.0296 | -0.0002 | 0.52 | no difference |
| 0.98 | acceptance: (a) labelled only | 11 | 0.0323 | -0.0003 | 0.0273 | win |
| 0.98 | acceptance: (b0) published rule | 11 | 0.0348 | -0.0019 | 0.0137 | win |
| 0.98 | deployment: final search set | 11 | 0.0301 | +0.0000 | 0.359 | no difference |
| 0.98 | deployment: path rule (T2's) | 11 | 0.0304 | +0.0000 | 0.688 | no difference |
| 0.99 | start: learned | 0 | n/a | n/a | n/a | not run |
| 0.99 | proposer: exhaustive | 11 | 0.0380 | -0.0001 | 0.7 | no difference |
| 0.99 | acceptance: (a) labelled only | 11 | 0.0403 | -0.0000 | 0.0781 | no difference |
| 0.99 | acceptance: (b0) published rule | 11 | 0.0414 | -0.0004 | 0.00684 | win |
| 0.99 | deployment: final search set | 11 | 0.0386 | -0.0000 | 0.91 | no difference |
| 0.99 | deployment: path rule (T2's) | 11 | 0.0386 | -0.0000 | 0.742 | no difference |
| 1 | start: learned | 0 | n/a | n/a | n/a | not run |
| 1 | proposer: exhaustive | 11 | 0.0666 | +0.0000 | nan | n/a |
| 1 | acceptance: (a) labelled only | 11 | 0.0666 | +0.0000 | nan | n/a |
| 1 | acceptance: (b0) published rule | 11 | 0.0666 | +0.0000 | nan | n/a |
| 1 | deployment: final search set | 11 | 0.0547 | +0.0059 | 0.000977 | loss |
| 1 | deployment: path rule (T2's) | 11 | 0.0666 | +0.0000 | nan | n/a |

Final search sets below the floor by Monte Carlo (es_ratio < rho): 50 of 220; not certified: 55 of 220.

### Results matrix (mean disparity over certified budgets; n = budgets certified; - = none; bold = lowest among the methods certified at every budget; the paired tests above are the formal comparison)

**rho = 0.97**

| test graph | FIM-GAT v2 | AGE | BSM | Degree | FW-Disparity | Greedy | Maximin-Greedy | MultiFW-MaxMin | Parity-Greedy | Prior | RFIM | Welfare-Greedy | lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| epinions / degree | **0.0496 (5/5)** | - | 0.0522 (5/5) | 0.0515 (3/5) | 0.0570 (5/5) | 0.0558 (5/5) | 0.0520 (5/5) | 0.0567 (5/5) | - | - | - | 0.0556 (5/5) | 0.0000 |
| instagram / gender | **0.0005 (5/5)** | - | 0.0053 (5/5) | 0.0158 (5/5) | 0.0109 (5/5) | 0.0124 (5/5) | 0.0003 (2/5) | 0.0026 (5/5) | 0.0080 (5/5) | - | 0.0214 (4/5) | 0.0121 (5/5) | 0.0000 |
| pokec_n / age | **0.0147 (5/5)** | - | 0.0309 (5/5) | - | 0.0382 (5/5) | 0.0503 (5/5) | - | 0.0111 (2/5) | 0.0567 (3/5) | - | - | 0.0435 (5/5) | 0.0000 |
| pokec_n / gender | **0.0174 (5/5)** | - | 0.0200 (5/5) | - | 0.0249 (5/5) | 0.0291 (5/5) | 0.0002 (1/5) | 0.0017 (1/5) | 0.0274 (5/5) | - | - | 0.0281 (5/5) | 0.0000 |
| pokec_n / region | 0.0067 (5/5) | - | **0.0032 (5/5)** | - | 0.0182 (5/5) | 0.0281 (5/5) | - | 0.0013 (3/5) | 0.0187 (5/5) | - | - | 0.0219 (5/5) | 0.0000 |
| rice_facebook / age | **0.0156 (5/5)** | - | 0.1768 (5/5) | 0.0870 (5/5) | 0.0286 (5/5) | 0.0720 (5/5) | 0.0357 (3/5) | 0.0853 (5/5) | 0.0320 (5/5) | - | - | 0.0663 (5/5) | 0.0000 |
| rice_facebook / college | **0.0058 (5/5)** | - | 0.0160 (5/5) | 0.0499 (5/5) | 0.0199 (5/5) | 0.0422 (5/5) | 0.0160 (5/5) | 0.0188 (5/5) | 0.0198 (5/5) | - | 0.1651 (3/5) | 0.0331 (5/5) | 0.0000 |
| rice_facebook / major | 0.1474 (5/5) | - | 0.3875 (5/5) | 0.3425 (5/5) | **0.1408 (5/5)** | 0.3230 (5/5) | 0.2999 (4/5) | 0.5424 (5/5) | 0.1592 (4/5) | - | 0.5272 (3/5) | 0.3041 (5/5) | 0.0000 |
| snap_youtube / degree | 0.0293 (5/5) | - | **0.0293 (5/5)** | 0.0441 (5/5) | 0.0339 (5/5) | 0.0386 (5/5) | 0.0219 (4/5) | 0.0201 (3/5) | - | - | - | 0.0379 (5/5) | 0.0000 |
| twitter_ego / degree | **0.0058 (5/5)** | - | 0.0084 (5/5) | 0.0107 (5/5) | 0.0125 (5/5) | 0.0136 (5/5) | 0.0061 (5/5) | 0.0134 (5/5) | - | - | 0.0146 (5/5) | 0.0136 (5/5) | 0.0000 |
| twitter_political / leaning | **0.0270 (5/5)** | - | 0.0641 (5/5) | 0.1033 (5/5) | 0.0563 (5/5) | 0.0920 (5/5) | 0.0324 (1/5) | 0.0908 (5/5) | 0.0340 (4/5) | 0.0952 (2/5) | - | 0.0842 (5/5) | 0.0000 |

**rho = 0.98**

| test graph | FIM-GAT v2 | AGE | BSM | Degree | FW-Disparity | Greedy | Maximin-Greedy | MultiFW-MaxMin | Parity-Greedy | Prior | RFIM | Welfare-Greedy | lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| epinions / degree | **0.0501 (5/5)** | - | 0.0522 (5/5) | 0.0515 (3/5) | 0.0573 (5/5) | 0.0558 (5/5) | 0.0520 (5/5) | 0.0567 (5/5) | - | - | - | 0.0556 (5/5) | 0.0000 |
| instagram / gender | **0.0012 (5/5)** | - | 0.0053 (5/5) | 0.0158 (5/5) | 0.0115 (5/5) | 0.0124 (5/5) | 0.0000 (1/5) | 0.0026 (5/5) | 0.0080 (5/5) | - | 0.0221 (2/5) | 0.0121 (5/5) | 0.0000 |
| pokec_n / age | **0.0234 (5/5)** | - | 0.0395 (5/5) | - | 0.0391 (5/5) | 0.0503 (5/5) | - | - | 0.0583 (2/5) | - | - | 0.0435 (5/5) | 0.0000 |
| pokec_n / gender | **0.0204 (5/5)** | - | 0.0226 (5/5) | - | 0.0249 (5/5) | 0.0291 (5/5) | 0.0002 (1/5) | 0.0017 (1/5) | 0.0274 (5/5) | - | - | 0.0281 (5/5) | 0.0000 |
| pokec_n / region | 0.0104 (5/5) | - | **0.0089 (5/5)** | - | 0.0185 (5/5) | 0.0281 (5/5) | - | 0.0008 (1/5) | 0.0187 (5/5) | - | - | 0.0219 (5/5) | 0.0000 |
| rice_facebook / age | **0.0186 (5/5)** | - | 0.1591 (5/5) | 0.0870 (5/5) | 0.0311 (5/5) | 0.0720 (5/5) | 0.0386 (1/5) | 0.0853 (5/5) | 0.0344 (4/5) | - | - | 0.0663 (5/5) | 0.0000 |
| rice_facebook / college | **0.0071 (5/5)** | - | 0.0160 (5/5) | 0.0499 (5/5) | 0.0200 (5/5) | 0.0422 (5/5) | 0.0284 (2/5) | 0.0188 (5/5) | 0.0198 (5/5) | - | 0.1952 (2/5) | 0.0331 (5/5) | 0.0000 |
| rice_facebook / major | **0.1398 (5/5)** | - | 0.4209 (5/5) | 0.3425 (5/5) | 0.1417 (5/5) | 0.3230 (5/5) | 0.3249 (2/5) | 0.4600 (3/5) | 0.1839 (3/5) | - | 0.5966 (2/5) | 0.3041 (5/5) | 0.0000 |
| snap_youtube / degree | **0.0303 (5/5)** | - | 0.0336 (5/5) | 0.0634 (1/5) | 0.0341 (5/5) | 0.0386 (5/5) | 0.0094 (1/5) | 0.0088 (1/5) | - | - | - | 0.0379 (5/5) | 0.0000 |
| twitter_ego / degree | 0.0076 (5/5) | - | 0.0084 (5/5) | 0.0115 (4/5) | 0.0138 (5/5) | 0.0136 (5/5) | **0.0061 (5/5)** | 0.0134 (5/5) | - | - | 0.0146 (5/5) | 0.0136 (5/5) | 0.0000 |
| twitter_political / leaning | **0.0353 (5/5)** | - | 0.0716 (5/5) | 0.1033 (5/5) | 0.0597 (5/5) | 0.0920 (5/5) | 0.0324 (1/5) | - | 0.0340 (4/5) | - | - | 0.0842 (5/5) | 0.0000 |

**rho = 0.99**

| test graph | FIM-GAT v2 | AGE | BSM | Degree | FW-Disparity | Greedy | Maximin-Greedy | MultiFW-MaxMin | Parity-Greedy | Prior | RFIM | Welfare-Greedy | lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| epinions / degree | 0.0525 (5/5) | - | 0.0522 (5/5) | - | 0.0570 (5/5) | 0.0558 (5/5) | **0.0520 (5/5)** | 0.0567 (5/5) | - | - | - | 0.0556 (5/5) | 0.0000 |
| instagram / gender | **0.0020 (5/5)** | - | 0.0076 (5/5) | 0.0158 (5/5) | 0.0115 (5/5) | 0.0124 (5/5) | - | 0.0026 (5/5) | 0.0076 (4/5) | - | - | 0.0121 (5/5) | 0.0000 |
| pokec_n / age | **0.0351 (5/5)** | - | 0.0478 (5/5) | - | 0.0454 (5/5) | 0.0503 (5/5) | - | - | 0.0582 (1/5) | - | - | 0.0435 (5/5) | 0.0000 |
| pokec_n / gender | **0.0236 (5/5)** | - | 0.0261 (5/5) | - | 0.0272 (5/5) | 0.0291 (5/5) | - | 0.0017 (1/5) | 0.0274 (5/5) | - | - | 0.0281 (5/5) | 0.0000 |
| pokec_n / region | **0.0152 (5/5)** | - | 0.0182 (5/5) | - | 0.0256 (5/5) | 0.0281 (5/5) | - | - | 0.0187 (5/5) | - | - | 0.0219 (5/5) | 0.0000 |
| rice_facebook / age | **0.0245 (5/5)** | - | 0.1349 (5/5) | 0.0870 (5/5) | 0.0367 (5/5) | 0.0720 (5/5) | - | 0.0832 (1/5) | 0.0338 (3/5) | - | - | 0.0663 (5/5) | 0.0000 |
| rice_facebook / college | **0.0087 (5/5)** | - | 0.0160 (5/5) | 0.0499 (5/5) | 0.0190 (5/5) | 0.0422 (5/5) | 0.0336 (1/5) | 0.0188 (5/5) | 0.0198 (5/5) | - | - | 0.0331 (5/5) | 0.0000 |
| rice_facebook / major | **0.1723 (5/5)** | - | 0.4334 (5/5) | 0.3425 (5/5) | 0.2254 (5/5) | 0.3230 (5/5) | 0.3145 (1/5) | 0.2215 (1/5) | 0.2068 (2/5) | - | - | 0.3041 (5/5) | 0.0000 |
| snap_youtube / degree | **0.0332 (5/5)** | - | 0.0376 (5/5) | - | 0.0344 (5/5) | 0.0386 (5/5) | - | - | - | - | - | 0.0379 (5/5) | 0.0000 |
| twitter_ego / degree | **0.0089 (5/5)** | - | 0.0121 (5/5) | - | 0.0136 (5/5) | 0.0136 (5/5) | 0.0054 (2/5) | 0.0115 (3/5) | - | - | 0.0146 (5/5) | 0.0136 (5/5) | 0.0000 |
| twitter_political / leaning | **0.0557 (5/5)** | - | 0.0911 (5/5) | 0.1185 (2/5) | 0.0890 (5/5) | 0.0920 (5/5) | - | - | 0.0269 (3/5) | - | - | 0.0842 (5/5) | 0.0000 |

**rho = 1**

| test graph | FIM-GAT v2 | AGE | BSM | Degree | FW-Disparity | Greedy | Maximin-Greedy | MultiFW-MaxMin | Parity-Greedy | Prior | RFIM | Welfare-Greedy | lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| epinions / degree | **0.0562 (5/5)** | - | - | - | **0.0562 (5/5)** | 0.0595 (1/5) | - | - | - | - | - | 0.0591 (1/5) | 0.0000 |
| instagram / gender | **0.0130 (5/5)** | - | - | 0.0150 (2/5) | 0.0183 (1/5) | - | - | - | - | - | - | - | 0.0000 |
| pokec_n / age | **0.0489 (5/5)** | - | - | - | **0.0489 (5/5)** | - | - | - | - | - | - | - | 0.0000 |
| pokec_n / gender | **0.0296 (5/5)** | - | - | - | 0.0355 (4/5) | - | - | - | - | - | - | - | 0.0000 |
| pokec_n / region | **0.0288 (5/5)** | - | - | - | **0.0288 (5/5)** | - | - | - | - | - | - | - | 0.0000 |
| rice_facebook / age | **0.0690 (5/5)** | - | - | - | 0.0779 (4/5) | - | - | - | - | - | - | - | 0.0000 |
| rice_facebook / college | **0.0377 (5/5)** | - | - | - | 0.0345 (4/5) | - | - | - | - | - | - | - | 0.0000 |
| rice_facebook / major | **0.3040 (5/5)** | - | - | - | **0.3040 (5/5)** | - | - | - | - | - | - | - | 0.0000 |
| snap_youtube / degree | **0.0387 (5/5)** | - | - | - | 0.0514 (3/5) | - | - | - | - | - | - | - | 0.0000 |
| twitter_ego / degree | **0.0136 (5/5)** | - | 0.0074 (1/5) | - | **0.0136 (5/5)** | 0.0074 (1/5) | - | - | - | - | - | 0.0074 (1/5) | 0.0000 |
| twitter_political / leaning | **0.0931 (5/5)** | - | 0.1349 (1/5) | - | **0.0931 (5/5)** | 0.1349 (1/5) | - | - | - | - | - | 0.1349 (1/5) | 0.0000 |

Certificate pass rate per method (share of all its sets):

| method | rho 0.97 | rho 0.98 | rho 0.99 | rho 1 |
|---|---|---|---|---|
| AGE | 0% | 0% | 0% | 0% |
| BSM | 100% | 100% | 100% | 4% |
| Degree | 69% | 60% | 40% | 4% |
| FIM-GAT v2 | 100% | 100% | 100% | 100% |
| FW-Disparity | 100% | 100% | 100% | 84% |
| Greedy | 100% | 100% | 100% | 5% |
| Maximin-Greedy | 55% | 35% | 16% | 0% |
| MultiFW-MaxMin | 80% | 56% | 38% | 0% |
| Parity-Greedy | 65% | 60% | 51% | 0% |
| Prior | 10% | 0% | 0% | 0% |
| RFIM | 27% | 20% | 9% | 0% |
| Welfare-Greedy | 100% | 100% | 100% | 5% |

## Spread-disparity trade-off (descriptive)

Per test graph, over budgets: the external baselines' sets (T1 rows and BSM's whole tau sweep) that the method's curve (its deployments at rho in [0.97, 0.98, 0.99, 1.0]) dominates, that dominate one of its points, and that are incomparable, with how many of those have less spread than every method point. Point estimates from Monte Carlo; the paired tests of T1 remain the formal comparison.

| test graph | baseline sets | dominated by the method | dominate a method point | incomparable | of which below the method's least spread |
|---|---|---|---|---|---|
| epinions / degree | 112 | 33 (29%) | 23 (21%) | 56 | 7 |
| instagram / gender | 114 | 34 (30%) | 12 (11%) | 68 | 0 |
| pokec_n / age | 117 | 45 (38%) | 7 (6%) | 65 | 7 |
| pokec_n / gender | 119 | 33 (28%) | 29 (24%) | 57 | 9 |
| pokec_n / region | 119 | 27 (23%) | 36 (30%) | 56 | 11 |
| rice_facebook / age | 132 | 99 (75%) | 6 (5%) | 27 | 0 |
| rice_facebook / college | 123 | 57 (46%) | 7 (6%) | 59 | 0 |
| rice_facebook / major | 131 | 87 (66%) | 14 (11%) | 32 | 2 |
| snap_youtube / degree | 115 | 24 (21%) | 13 (11%) | 78 | 8 |
| twitter_ego / degree | 112 | 45 (40%) | 7 (6%) | 60 | 2 |
| twitter_political / leaning | 125 | 65 (52%) | 12 (10%) | 48 | 2 |

Figures (k = 50): figures/pareto_epinions_degree_k50.png, figures/pareto_instagram_gender_k50.png, figures/pareto_pokec_n_age_k50.png, figures/pareto_pokec_n_gender_k50.png, figures/pareto_pokec_n_region_k50.png, figures/pareto_rice_facebook_age_k50.png, figures/pareto_rice_facebook_college_k50.png, figures/pareto_rice_facebook_major_k50.png, figures/pareto_snap_youtube_degree_k50.png, figures/pareto_twitter_ego_degree_k50.png, figures/pareto_twitter_political_leaning_k50.png

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

- rho 0.97: v2-b - v2-a held-out-attribute disparity over 40 cases: median +0.0000, p 0.145 (no difference)
- rho 0.99: v2-b - v2-a held-out-attribute disparity over 40 cases: median +0.0000, p 0.799 (no difference)

## T3: does the learning add anything?

The W3 rule asked for a learned proposer, but there are no T3 rows yet (train_proposer.py, then run_t3.py).

## T4 (C3): runtime, cached and per query

No T4 rows yet.

## Certificate-size sensitivity

| cert R | method: searched set deployed | method: mean disparity | baselines certified |
|---|---|---|---|
| 16 | 80% | 0.0206 | 32% |
| 64 | 90% | 0.0166 | 56% |
| 256 | 100% | 0.0121 | 60% |

