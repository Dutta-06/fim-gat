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

## Step 2: T1 headline, small graphs (running; job `t1-small`)
- Plan (dry run): 4 attribute runs (rice_facebook college/age/major, twitter_political leaning), 240 queries each (20 FIM-GAT, 220 baseline), BSM sweep at 5 budgets (laptop estimate 0.6 h total, UNVERIFIED; the dry run does not list the rho values).
- Measured: ~173 rows in the first 30 min (rice_facebook / college, where FW-Disparity costs ~16 s per query); extrapolated total ~2-3 h on this VM. Chunks are cut at 6900 s by `timeout`; each chunk is committed and pushed (`chunks.log`).
- Memory: ~1-3 GB used of 15 GB.
- Chunk 1 (115 min, `timeout` 6900 s, rc=124 as designed) wrote 625 rows (~5.4 rows/min); chunk 2 resumed automatically (rows are skipped when their key exists). Committed and pushed after each chunk.

## Step 2 result: T1 headline, small graphs: DONE
- `t1-small`: 1,120 rows, all status ok, none missing (rice_facebook college/age/major, twitter_political leaning; 280 rows each). 2 chunks (115 min + 106 min), exit 0. Peak RAM small (a few GB). AGE is the slow baseline: ~60 s per k on Rice, ~490 s per k on twitter_political (computed once per k, reused across rho).
- `summarize_v2.py` ran (1.5 s); `results/v2/summary_v2.md` committed.

## Step 3: T2 (running; job `t2`)
- Dry-run plan: 6 BFS samples of snap_youtube (100,000 nodes each, seeds 1-6), attribute `degree`; 30 FIM-GAT queries per sample (configs v2-b, v2-a, v2-b0 with greedy start, simple proposer); laptop estimate 0.5 h per sample (3.0 h total, UNVERIFIED). R = 64 RR sets per node (pre-registered, not lowered).
- Interim summary after t1-small (4 test graphs, 80 cases; NOT the final verdict): in the results matrix, FIM-GAT v2 has the lowest mean disparity among the methods certified at every budget in 11 of the 12 (graph, rho) cells at rho in {0.97, 0.98, 0.99}; the exception is rice_facebook / major at rho = 0.97 (FW-Disparity 0.1408 vs FIM-GAT 0.1474). (On twitter_political at rho = 0.99, Parity-Greedy shows 0.0269 against FIM-GAT's 0.0557, but it is certified at only 3 of 5 budgets, so the matrix does not rank it.) At rho = 1.0 FIM-GAT is lowest or tied in all 4 graphs. FIM-GAT's certificate pass rate is 100% at every rho; AGE's is 0%. The paired tests cannot read "significant" yet: with 4 graphs the p-values are multiples of 1/8 and the smallest possible two-sided exact p is 0.125; the verdicts need the three Pokec-n graphs. "Holds across the range" is therefore "no" for every comparison at this stage. Ablation rows for "start: learned", "proposer: exhaustive" and "acceptance: (a)/(b0)" are not run yet (step 5).
- **Incident (22:09 UTC):** the VM rebooted (`uptime` = 0 min; the worker process was restarted) while T2 sample `snap_youtube_bfs100000_s1` was finishing its `reference` stage (1418.5 s, saved to `results/v2/parts/t2/references.json`) and entering `rr_sample` (R = 64, 100,000 nodes). `dmesg` was empty after the reboot, so the cause is unknown: it may have been an out-of-memory event in `rr_sample`, or an unrelated infrastructure restart. The repo, `.venv`, data and the saved reference survived. All background jobs died and are being restarted from the saved rows (T1-small was already complete and pushed).
- T2 restarted at 22:12 UTC after the reboot (sample 1's reference was loaded from `references.json`). Probe before the restart (R = 1 and 4 on the sample, then the real run): mean RR-set size 11.6, so A at R = 64 holds 74.0M entries (~1.5 GB) and B 74.1M (B is never smaller than A); `rr_sample` took 244 s and the job's RSS was 1.5 GB at that step, so the reboot was not a memory problem at that step. Peak RSS so far 5.3 GB of 16 GB (logged every 30 s in `results/v2/parts/t2/mem.log`, with a watchdog that kills the job if MemAvailable < 800 MB; it has not fired).
- T2 rates measured: per sample, `reference` ~24 min + `rr_sample` ~4 min (both cached/one-off per sample) + 30 queries at ~5-60 s each; so ~55 min per sample, ~5.5 h for 6 samples.
