# Research protocol: read before running experiments, changing the method, or moving hardware

This file holds the rules and procedures for running FIM-GAT v2 without corrupting its results. `CLAUDE.md` holds the summary, `docs/design_v2.md` §18–§19 the pre-registration record, and `experiments/v2/` the scripts. The v1 protocol and the v1 record were archived on 2026-10-10, outside the repo, in `Desktop\FIM-GAT_archive\removed_2026-10-10\`.

## 1. The v2 pre-registration (2026-10-10)

The user approved the amendment of design §18 on 2026-10-10, before any v2 experiment ran. They then delegated the remaining choices, which design §19 records. No v2 experiment has run yet.

**Run order** (design §19.1; from `experiments/v2`, with `PYTHONIOENCODING=utf-8 python -u -W ignore`). Each script prints its plan and a rough estimate first, and `--dry-run` stops there. Ask the user before any run expected to take hours.
1. `build_datasets.py` downloads and builds the v2 graphs. That includes the validation district `pokec_t`, which comes from the local Pokec dumps.
2. `train_constructor.py` is optional. Without constructor checkpoints the start set is greedy.
3. `tune_v2.py` settles δ, φ₀, the start set and the W3 gate on `pokec_t` only, and writes `results/v2/tuning.json`.
   - If and only if its W3 rule says to train, `train_proposer.py` trains the learned proposer next. It is a multi-hour GPU job, so ask the user first; without the W3 go-ahead it refuses (design §21.4).
4. `run_t1_t2.py --test t1`, then `run_t1_t2.py --test t2`.
   - T1 also records BSM's τ sweep (`t1_pareto.csv`).
   - T2 runs at 64 RR sets per node on 6 samples. Close other applications or use the 16 GB machine (design §21.2); a smaller `--R` is a reported deviation.
5. `run_t4.py` on the large tier. One GPU job at a time.
   - Then, only if the W3 rule trained a proposer, `run_t3.py` (test T3, design §21.4). It says why and stops otherwise.
6. The certificate-size sensitivity run (design §20), on its fixed subset. Each `--cert-R` gets its own resume key:
   ```
   for R in 16 64 256; do
     python run_t1_t2.py --test t1 --datasets rice_facebook --attributes college --rhos 0.99 --configs method \
       --cert-R $R --no-bounds --out ../../results/v2/t1_certsens.csv
     python run_t1_t2.py --test t1 --datasets pokec_n --attributes region --rhos 0.99 --configs method \
       --cert-R $R --no-bounds --out ../../results/v2/t1_certsens.csv
   done
   ```
7. `interpret_v2.py` writes the pre-registered case studies (design §20) to `results/v2/cases/`.
8. `summarize_v2.py --figures` writes `results/v2/summary_v2.md`, the pre-registered analysis of design §19.4, §20 and §21, plus the spread–disparity figures (`results/v2/figures/`).

**Before any v2 test run**
- **The certificate's sample B needs no tuning.** Its size follows a memory rule (design §19.2; `--cert-nnz`, default 60M entries). With a small B, certified deployment mostly deploys the reference set (design §18.7).
- **δ, φ₀ and T1's start set come from `tune_v2.py`** on validation (design §19.2).
- **`tune_v2.py` also runs the W3 rule** (`diagnostics.learning_decision`). There is no large-tier validation graph, so it runs on `pokec_t`. Train a learned proposer only if the rule says so, and record its output either way.
- **Free host memory first** (§4).
- **Confirm that T2 has at least 20 paired cases.** Since 2026-10-14 it plans 30 (6 samples × 5 budgets). If it has fewer, add test subgraph samples now, not after seeing results.
- **Last call for changes.** The values in design §18.5, §19.2 and §21 may still change now. None may change after the first v2 test run.

**Tuning rule**
- Every tuned value comes from the validation split (`datasets.splits()["val"]`) only: δ, φ₀, m, the price setpoint, T, the pruning size and the start set. `tune_v2.py` refuses any other graph.
- Never tune or select on test rows, on `pokec_n`, on any full graph, or on any ground-truth community set. SNAP com-Youtube communities are for evaluation only.

**The tests** (design §10, as amended)
- **T1 (C1):**
  - ρ = 0.99;
  - every method's equal spread is judged by the same RR certificate;
  - comparison against external baselines only;
  - the ablation table goes in the main text;
  - "holds across the range" means a significant win at ρ = 0.97, 0.98 and 0.99; ρ = 1.0 is reported descriptively (design §21.3).
- **T2 (C2):**
  - option (b) `hive_search` against (a), with (b0) also reported;
  - ρ = 0.97 primary, 0.99 secondary;
  - the primary comparison uses the path rule's deployments (`deployment == "certified-path"`) for every variant (design §21.1);
  - at least 20 paired cases;
  - paired two-sided Wilcoxon tests at α = 0.05.
- **T3:** the simple hybrid with scaled prices is the baseline. T3 runs only if the W3 rule trains a learned proposer. Pass: (i) at least 1.5× sooner to the simple hybrid's final RR disparity on more than half of the test graphs, or (ii) lower Monte Carlo disparity at 60 s (Wilcoxon). A fail drops the learned proposer (design §21.4).
- **T4 (C3):** per-graph and per-query costs are reported apart, with Q ∈ {1, 10, 100} queries per graph; run it with `experiments/v2/run_t4.py`.
- A failed test is reported as failed. The adversary is not dropped or reframed without asking the user (design §10).

**Reporting checklist**
1. Report every test graph, budget and floor. Use rates, not counts, and give the seed standard deviation of the learned parts.
2. Give every method's equal-spread verdict from the certificate, with the Monte Carlo equal-spread test next to it for continuity. The CSVs record each set's certificate (design §19.2).
3. For FIM-GAT, report the final search set as well as the deployed one: the "deployment" row of the ablation table.
4. For T2, report (a), (b0) and (b) side by side, with the primary and secondary metrics and κ for every case.
5. For T3, report the W3 rule's inputs and output, even when it says not to train.
6. For T4, report cached and per-query seconds and memory for each stage (`results/v2/t4_runtime.csv`), with the GPU and the preset.
7. Every row carries `code_tag`, `gpu` and `preset`; `RecorderV2` adds them.
8. The T1 headline is the real-attribute graphs; the degree-group graphs are a supplement (design §20).
9. Report the 11 pre-registered case studies, all of them. Any other case is marked as not pre-registered.
10. Report the certificate-size sensitivity table, and use `docs/paper_notes.md` for the reproducibility, limitations
    and ethics sections, filling in its **[after results]** points.
11. Report the spread–disparity trade-off (`summary_pareto.csv` and the figures) next to T1, so that the baselines that ignore the floor are shown, not only dropped (design §21.5).
12. For T2, report both deployment rules and the final search sets; the path rule is primary (design §21.1).

## 2. Preserving results

- **There is no git.** `RecorderV2` and the stage logs rewrite their CSV atomically after every row, so an interrupted run never leaves a truncated file.
- **A rerun into existing files silently skips what is already there.**
  - Resume keys skip every query whose row exists. The keys include the tuned δ and φ₀, but not the preset, the hardware or the code.
  - Before a rerun after a code change, copy `results/v2/` to `results/v2_backup_<date>/`, or write to a fresh `--out`.
- **Provenance.** Every row stores its seed set, so any metric can be recomputed. It also stores `code_tag`, a SHA-1 of `fimgat/`, `experiments/build_pokec.py` and `experiments/v2/`, along with `gpu` and `preset`.
- **Checkpoints** live in `results/v2/models/constructor_s{seed}.pt`. A stale checkpoint loads without complaint, or crashes after a feature change. Delete it when the features or the policy change.

## 3. Launching, monitoring and stopping long runs (Windows, Git Bash)

- **Launch** from `experiments/v2/` in the background, unbuffered, appending to a log. For example:
  `cd /c/Users/Odwitiyo/Desktop/FIM-GAT/experiments/v2 && PYTHONIOENCODING=utf-8 python -u -W ignore run_t1_t2.py --test t1 >> ../../results/v2/t1_log.txt 2>&1`
- **Monitor** with a filtered tail of the log (`Traceback|Error|out of memory|===`) plus a pandas row count of the CSV.
- **To stop** a run, stop the background shell task **and** the Python process: run `tasklist /FI "IMAGENAME eq python.exe"`, then `taskkill /PID <pid> /F` for that PID only. Chained commands (`a; b`) continue after `a` is killed.
- **One GPU job at a time on the 8 GB card.** In v1, two concurrent jobs caused `RuntimeError: CUDA error: out of memory`. They also make timings and memory peaks meaningless.

## 4. Memory: GPU and host

- **GPU.**
  - `nvidia-smi` runs in WDDM mode here: it shows no per-process memory, and it can report near 0 MiB while torch holds memory. Use `python -c "import torch; print(torch.cuda.mem_get_info())"` instead.
  - Inside code, read `torch.cuda.max_memory_allocated()` and `max_memory_reserved()` after `reset_peak_memory_stats()`.
  - `run_t4.py` logs peak allocated and reserved memory for every stage.
- **Host RAM.** This laptop has 15.5 GB, but only about 1–3 GB was free while other applications were open, and two development runs died of host out-of-memory errors. Close those applications before real runs, or use the 16 / 40 GB machine for the large tier. The large items:
  - An RR sample takes about 22 bytes per entry (measured on `pokec_z`). The certificate sample B holds up to 60M entries, about 1.3 GB.
  - Per-node, per-group gain matrices take n × G floats. These are RRGreedy's, at every greedy step, and BSM's, in float32. For full Pokec with region groups that is about 1.2 GB in float32.
  - The full-Pokec graph and its context take several GB.

## 5. Memory presets (`experiments/v2/common_v2.PRESETS`)

- `full_R` and `full_max_roots` set the search sample A for the large tier (`pipeline_v2.rr_config`). Smaller graphs use `config.RR_PER_NODE` with every node a root.
- `chunk_edges` bounds the edges per chunk of CPU-edge inference. The `--edge-device auto` option moves message passing to the CPU above it.
- `checkpoint_edges` turns on activation checkpointing in `train_constructor.py` (`--checkpointing auto`).
- The `16gb` and `40gb` presets were scaled from 8 GB measurements and have **never been run**. A hardware change needs a fresh `--out` folder (§2).

## 6. Timings measured on this laptop (v1 runs; for estimates only)

No v2 stage has been timed on real data yet; `run_t4.py` will measure them. From v1:
- **Full Pokec:**
  - RR sampling of 150k roots × 8 takes 160–190 s.
  - Greedy-family baselines take 6–54 s and Frank–Wolfe about 4.5 s.
  - One v1 policy inference took about 27 s (6.2 GB peak, 3M-edge chunks).
- **Pokec-z:**
  - One v1 policy trained in about 5.4–5.7 min with checkpointing (3.1 GB peak, ad-hoc check).
  - Monte Carlo evaluation (2,000 simulations) took about 0.5 s.

## 7. Moving to the 16 / 40 GB machine (details unknown; ask the user)

Claude Code cloud sessions (CPU-only Linux VMs) have their own manual, `docs/cloud_runbook.md` (2026-10-15).

- **What we don't know.** The access method, OS and environment of the target machine are unknown.
- **Setup.**
  - Copy `data/` (`data/pokec/*.npz` is 117 MB, plus the v2 builds) so that neither the raw dumps nor the downloads are needed.
  - Install `requirements.txt` with the matching CUDA build of torch.
  - On Linux the Windows-specific notes (`python3`, cp1252) do not apply. RFIM's C++ (design §19.3) also builds there.
- **Estimate**, scaled from the full-Pokec run (1.2M RR sets giving 19.4M memberships):
  - the `16gb` preset (400k roots × 12 = 4.8M sets) needs about 78M memberships;
  - the `40gb` preset needs about 250M.
- **Reproducibility.**
  - `sample_rr_sets` splits roots into `diffusion.RR_CHUNKS` = 16 chunks with per-chunk seeds, on any machine (since 2026-10-15; before, `min(16, NUMBA_NUM_THREADS)` chunks, which gave other samples on machines with fewer than 16 numba threads). The same seeds now give the same RR sets everywhere, and this laptop's samples are unchanged.
  - CUDA `index_add_` is nondeterministic, so same-seed reruns differ at seed-noise level.
  - Run every method of a test together on one machine, into one folder: all methods must share one RR sample.
