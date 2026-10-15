# Cloud job ALL: status

Branch `claude/fimgat-all`. VM: 4 vCPU, 15 GB RAM, no GPU. Python 3.13 (`.venv`), torch 2.6.0+cu124 (cuda False), numba 0.63.1.

## Step 1: job A (in progress)
- Tests: 169 passed, 13 skipped (before datasets were built). Log: `results/v2/parts/A/tests.txt`.
- Datasets built (rice_facebook, twitter_political, epinions, twitter_ego, pokec_t, instagram, snap_youtube). Build took ~3 min. Log: `results/v2/parts/A/build_log.txt`.
- **WARNING (data, not fixed, researcher decision):** `rice_facebook` builds with n = 1,216 (edges 42,443 as expected), while the design/test expect n = 1,205. The 11 extra nodes are attributed users with no edges (`isolated=11`); the builder's documented rule is "node set = attributed ids". After the build, `tests/test_smoke_data.py::test_v2_dataset_integrity[rice_facebook]` FAILS (`1216 == 1205`). This is a dataset-definition question, not an environment bug, so I changed neither the builder nor the test (rule 1). Runs on rice_facebook use n = 1,216. If the researcher decides to drop the isolates, the rice rows must be rerun.
- Setup note: `cloud_setup.sh` tried a CPU torch wheel from download-r2.pytorch.org (403 from the proxy); the already-installed torch 2.6.0+cu124 works on CPU, so nothing is lost.
- Tuning dry-run plan: 35 method queries (0 constructor checkpoints) + W3 measurements (4 states x 5 budgets), on pokec_t / region, CPU, preset 8gb.

## Step 1 result: tuning done (18 min wall, CPU, ~12 GB free during the run)
- Validation: pokec_t / region, ks 25–200. Command: `tune_v2.py --steps delta phi0 w3`, exit 0. Files: `tuning.json`, `tuning_runs.csv`, `references.json`.
- **δ = 0.005** (scores, lower is better: 0.0025 → 0.04162, 0.005 → 0.03678, 0.01 → 0.03808, 0.02 → 0.03808).
- **φ₀ = 0.4** (0.3 → 0.04315, 0.4 → 0.04117, 0.5 → 0.04365).
- start = "greedy" (provisional: the start step waits for the constructor on a GPU).
- **W3 decision: train = true** (H = 8.12, H/cost = 8.12 ≥ 1.5; π = 0.389, q = 1.0, c_round = 0.0292, T_ex = 0.340, 18 states; note: no large-tier validation graph, measured on pokec_t). So the learned proposer should be trained (`train_proposer.py`, then `run_t3.py`) on the GPU machine; neither is run here (rule 3).
- Job A's pull request is not opened separately: job ALL opens one at the end.
