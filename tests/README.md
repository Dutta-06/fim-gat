# FIM-GAT test suites

Three suites, selected with pytest markers (`pytest.ini`). Run from the repo root with the project interpreter:

```bash
PYTHONIOENCODING=utf-8 python -m pytest                  # smoke + spec (default; stress excluded)
PYTHONIOENCODING=utf-8 python -m pytest -m smoke         # smoke only
PYTHONIOENCODING=utf-8 python -m pytest -m stress        # stress only (opt-in; minutes)
PYTHONIOENCODING=utf-8 python -m pytest -m spec          # v2 contract, pipeline and theory checks
```

Never use `python3` (a different interpreter; see `CLAUDE.md`).

| Suite | Files | What it checks | Runtime (8 GB laptop) |
|---|---|---|---|
| **smoke** | `test_smoke_foundation.py`, `test_smoke_data.py` | The v1 components that v2 reuses: the RR estimator and Monte Carlo against **exact** Independent Cascade (all live-edge worlds enumerated on tiny graphs); weighted cascade; RR reproducibility; closed-form gains and objective gradient against autograd; exact-budget soft top-k; feature invariance under group relabelling (the basis of "any group definition"); policy permutation equivariance; chunked vs full inference; greedy family; metrics. Data: existing Pokec builds; old-project raw files present and untouched; integrity rules for every v2 dataset (weighted cascade, no self-loops, documented sizes), which skip until each dataset is built. | about a minute |
| **stress** | `test_stress_foundation.py`, `test_stress_edgecases.py` | Every budget {25, 50, 100, 150, 200} × every method on Pokec-z/n and full Pokec (8 GB preset); 512 × 200 batch evaluation (the v2 verifier's load); adversarial graphs (star, path, disconnected, isolated nodes, one group, 60 singleton groups, an unreachable group) with k from 1 to n − 1. Asserts validity, finite metrics and GPU memory under the card; **records** timings and memory in `tests/_reports/stress_results.jsonl`. | Pokec-z/n ≈ 10 min, full Pokec ≈ 15–20 min, edge cases ≈ 1 min |
| **spec** | `test_spec_v2.py`, `test_spec_v2_pipeline.py`, `test_theory_v2.py`, `test_theory_v2_w.py`, `test_preregistration_v2.py`, `test_external_v2.py`, `test_scripts_v2.py`, `test_prerun_fixes_v2.py` | The v2 contract from `docs/design_v2.md`, including exact incremental swap scoring, the acceptance rule (verifier option (b)), the reference spread and equal-spread test, the two-sided signed prices and the PI controller, the hive-mind auditor (finds planted communities, respects size and cohesion, never uses the evaluation partition), a monotone improvement phase with floor fallback and time budget, local-search optimality, and the dataset registry, splits and held-out rules. The interfaces are listed in the file header. `test_spec_v2_pipeline.py` covers the end-to-end v2 pipeline. The theory checks are listed below the table. | 2026-10-10: **117 passed**, 3 skipped (unbuilt datasets), about 1.5 min |

`test_theory_v2.py` is the executable form of `docs/theory_v2.md` Part I (R1–R3):
- the certified gain is a valid lower bound, reduces to the plain gain, and obeys the tie rules;
- with sampled roots, a move's change noise counts per-root second moments;
- the KL rate bounds and the deploy certificate cover the truth, checked against exact Independent Cascade;
- the search cycles without tabu and not with it;
- the move bound of Theorem 1 holds.

`test_theory_v2_w.py` does the same for Part II (W1–W4):
- the transfer bound;
- multi-scale audits;
- the two-stars mixture;
- the epsilon-constraint phase;
- the hardness gadget, and the lower bound against brute force with exact Independent Cascade;
- the proposer cost model and the hybrid;
- the price loop (the spread price is inert under the published controller and acts under the scaled one; Jury's criterion; bang-bang dual ascent).

`test_preregistration_v2.py` checks the decisions of 2026-10-10 (`docs/design_v2.md` §18):
- `improve()`'s defaults equal the approved config values, and the simple proposer uses the scaled controller;
- `verify.disparity_se` is the noise-model formula, and it bounds the spread of D across independent RR samples;
- `improve.hive_search` runs the labelled phase and then the capped phase: D stays at or below κ = D_a + 1 SE, the floor holds, (D_C, D) falls within an epoch, and a restart records the reference as its start;
- `certify.path_states` rebuilds the search path, including multi-node moves;
- `deploy_candidates` keeps both ends without duplicates;
- `certified_equal_spread` passes the reference and fails a weak set, and certified choice over a `hive_search` path is consistent with it;
- `LearnedProposer` and `V2Trainer` aim at the capped floor;
- `diagnostics.learning_decision` follows the W3 rule.

`test_external_v2.py` checks the external baselines written from their papers (`fimgat/v2/external.py`: BSM, MultiFW-MaxMin and RFIM, including RFIM's label propagation, three-level weights and weighted coverage) against brute force:
- the incremental per-group gains;
- the truncated cover gain;
- Saturate's bound;
- BSM's constraint and its choice of τ;
- the marginals of the dependent rounding;
- MultiFW-MaxMin's output.

`test_prerun_fixes_v2.py` checks the pre-run fixes of 2026-10-14 (`docs/design_v2.md` §21):
- the deployment rules (`certify.choose`: Theorem 2's rule and the path rule), with candidates in path order;
- the new constants (`RHO_HOLDS`, T2's R and sample count, T3's rule);
- BSM's τ sweep, which contains the BSM adapter's choice;
- the summary's dominance, its T3 rule and its T4 filter, on synthetic rows;
- `train_proposer.py` → `run_t3.py` → `summarize_v2.py` end to end on synthetic stand-ins, the first execution of `fimgat.v2.imitation`, and the W3 gate refusing to train.

`test_c2_diagnostics.py` runs the exploratory C2 diagnostics (`experiments/v2/c2_diagnostics.py`, design §22) on a synthetic com-Youtube stand-in. Its control runs must reproduce T2's deployed sets exactly.

`test_kaggle_v2.py` checks the Kaggle runner (`kaggle/`, design §23):
- every job command names a real script, real options, built datasets, their attributes and the suite's methods, and the pipeline scripts run at the 8gb preset;
- the CPU jobs never use the constructor, and the timing jobs use one lane;
- outputs round-trip through export, resume and import, results CSVs merge without losing or repeating rows, and nothing outside `results/` and `data/cache/` is written;
- the session deadline, the run-once markers and the conditional steps;
- the notebook's cells parse, and the bundle never takes LinkedIn or raw data.
- job G1's chain on synthetic stand-ins: `train_constructor.py` (its first execution) writes checkpoints that `tune_v2.py --steps start` and T1's learned-start rows load.

`test_scripts_v2.py` runs the experiment scripts end to end on synthetic stand-ins, with `fimgat.v2.datasets` patched and every output under `tmp_path`:
- `tune_v2` → `run_t1_t2` (T1 with BSM's τ sweep, T2) → `summarize_v2` (with the Pareto table and a figure), three rows per FIM-GAT query, and resumption;
- tuning refuses a test graph;
- `run_t4`'s stage log;
- the case study (`interpret_v2`) reruns the method exactly.

The default run (smoke + spec) gives 167 passed, 10 skipped (unbuilt datasets), in about 3 min (2026-10-14, after the pre-run fixes of design §21).

Isolation: the tests never write to `data/` or `results/`. Louvain caching is redirected to a temporary folder (existing caches are copied in read-only). The old project (`Desktop/xAI Project/Baseline_Test`) is only read, and a test checks that its files are unchanged.

Stress failures are informative. They mark v1 edge-case bugs that must be fixed or guarded before v2 reuses that component; they are not flaky timing checks.
