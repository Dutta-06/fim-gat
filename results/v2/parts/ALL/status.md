# Cloud job ALL: status

Branch `claude/fimgat-all`. VM: 4 vCPU, 15 GB RAM, no GPU. Python 3.13 (`.venv`), torch 2.6.0+cu124 (cuda False), numba 0.63.1.

## Step 1: job A (in progress)
- Tests: 169 passed, 13 skipped (before datasets were built). Log: `results/v2/parts/A/tests.txt`.
- Datasets built (rice_facebook, twitter_political, epinions, twitter_ego, pokec_t, instagram, snap_youtube). Build took ~3 min. Log: `results/v2/parts/A/build_log.txt`.
- **WARNING (data, not fixed, researcher decision):** `rice_facebook` builds with n = 1,216 (edges 42,443 as expected), while the design/test expect n = 1,205. The 11 extra nodes are attributed users with no edges (`isolated=11`); the builder's documented rule is "node set = attributed ids". After the build, `tests/test_smoke_data.py::test_v2_dataset_integrity[rice_facebook]` FAILS (`1216 == 1205`). This is a dataset-definition question, not an environment bug, so I changed neither the builder nor the test (rule 1). Runs on rice_facebook use n = 1,216. If the researcher decides to drop the isolates, the rice rows must be rerun.
- Setup note: `cloud_setup.sh` tried a CPU torch wheel from download-r2.pytorch.org (403 from the proxy); the already-installed torch 2.6.0+cu124 works on CPU, so nothing is lost.
