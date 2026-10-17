# Cloud runbook: running FIM-GAT v2 on Claude Code cloud sessions

Written 2026-10-15. On that day the user said to run everything that a Claude Code cloud session can carry, as soon as possible, so that local compute is not needed. This file is the background for those sessions; their operational brief, with the exact commands of every job, is `docs/cloud_mission.md`. `docs/research_protocol.md` still governs what is run and how it is reported. Nothing here changes a pre-registered setting (design §18–§21).

## 1. What a cloud session can and cannot carry

The Claude Code docs (code.claude.com, read 2026-10-15) describe each cloud session as:
- an Ubuntu VM with about 4 vCPUs, 16 GB RAM and 30 GB disk;
- **no GPU** (none is documented);
- background commands that run up to 30 minutes by default and 2 hours at most, unless the environment variables `BASH_DEFAULT_TIMEOUT_MS` / `BASH_MAX_TIMEOUT_MS` raise those limits;
- a VM that may pause after a few minutes without activity, and may be reclaimed;
- files that survive only through commits **pushed to GitHub**.

| Runs in the cloud | Waits for a GPU machine |
|---|---|
| tests; dataset builds (except LinkedIn and full Pokec); `tune_v2.py`; T1 on Rice, Twitter-political, Pokec-n, Epinions, Twitter-ego, Instagram and com-Youtube; T2; the certificate-size sensitivity run; the case studies of those graphs; the summary | `train_constructor.py`; `train_proposer.py` and `run_t3.py` (only if the W3 rule says train); T1, case studies and T4 on full Pokec and LinkedIn; T4 runtime claims in general (timings must come from the hardware the paper reports) |

Integrity across machines:
- RR samples are machine-independent since 2026-10-15 (`diffusion.RR_CHUNKS`); Monte Carlo already was.
- The same seeds therefore give the same samples on a cloud VM and on any other machine.
- CPU runs are deterministic.
- Rows record `gpu` = "cpu" and the provenance `code_tag`.

## 2. One-time settings (the user, in claude.ai/code)

1. **Repository.** The private GitHub repository of this project, with the Claude GitHub App installed on it, so sessions can clone it and push their branches.
2. **Network access.** Custom: keep the defaults and add `snap.stanford.edu` (Pokec, Epinions, ego-Twitter and com-Youtube downloads) and `download.pytorch.org` (CPU PyTorch wheels; without it, setup falls back to PyPI's larger default wheel).
3. **Optional environment variable.** `BASH_MAX_TIMEOUT_MS=21600000` (6 h) allows longer chunks. Without it, chunks stay under 2 h.
4. **Optional setup script.** `bash scripts/cloud_setup.sh`. It is cached only if it finishes in about 5 minutes; otherwise every session runs it itself.

## 3. Rules for every cloud session

1. Start with `bash scripts/cloud_setup.sh && source .venv/bin/activate`. Use `python` from the venv. The laptop-only advice in `CLAUDE.md` (`python3`, `PYTHONIOENCODING`, CRLF) does not apply on the VM.
2. Run only your job (§4). Never tune on test graphs and never change a pre-registered value; `docs/research_protocol.md` applies in full.
3. Print the script's `--dry-run` plan first and put it in the session's first report.
4. Run in resumable chunks, then push. The scripts skip every row that already exists:
   - run each chunk as `timeout 6600 python -u -W ignore <script> <args> >> results/v2/parts/<job>/log.txt 2>&1`;
   - after every chunk, run `git add -A results/v2 && git commit -m "<job>: chunk N" && git push`;
   - repeat until the script reports nothing left to do.

   Push often: an unpushed result is lost if the VM is reclaimed.
5. Write only your job's files: `run_t1_t2.py --job <job>` puts every output under `results/v2/parts/<job>/`. Never edit another job's files or the shared ones (`tuning.json`, `references.json`) after session A.
6. Finish with a short report:
   - the rows written and the rows missing, each with its reason;
   - wall-clock time and peak memory;
   - any warning or deviation, for example a smaller `--R`.

## 4. Sessions

All commands run from the repository root. Session A comes first. The B sessions run in parallel, each starting from a branch that contains session A's `tuning.json` (main, after A is merged). Session C comes last.

### Session A: setup, data, tuning

1. Setup, then `python -m pytest -q`. Expect only skips for unbuilt or local-only datasets; report any failure before going on.
2. Build the datasets:
   ```
   python -u -W ignore experiments/v2/build_datasets.py --only rice_facebook twitter_political epinions twitter_ego pokec_t instagram snap_youtube
   ```
   - Check the size warnings.
   - Commit `data/v2/*.npz` and `data/pokec/pokec_t.npz`; each must be under 100 MB (`ls -la`). If one is over, do not commit it and report it.
   - The raw downloads (`data/v2/raw/`) are never committed.
3. Tuning, without the start step:
   ```
   python -u -W ignore experiments/v2/tune_v2.py --steps delta phi0 w3 --dry-run
   python -u -W ignore experiments/v2/tune_v2.py --steps delta phi0 w3
   ```
   - The start step waits for the constructor, which needs a GPU (§5).
   - Run in chunks as in §3, then commit `results/v2/tuning.json`, `tuning_runs.csv` and `references.json`.
4. Push, open a pull request to main, and report δ, φ₀ and the W3 decision.

### B sessions: one job each (after A is merged)

| Job | Command (`python -u -W ignore experiments/v2/run_t1_t2.py` …) |
|---|---|
| `t1-small` | `--test t1 --datasets rice_facebook twitter_political --job t1-small` |
| `t1-pokec-n-region` | `--test t1 --datasets pokec_n --attributes region --job t1-pokec-n-region` |
| `t1-pokec-n-gender` | `--test t1 --datasets pokec_n --attributes gender --job t1-pokec-n-gender` |
| `t1-pokec-n-age` | `--test t1 --datasets pokec_n --attributes age --job t1-pokec-n-age` |
| `t1-epinions` | `--test t1 --datasets epinions --job t1-epinions` |
| `t1-twitter-ego` | `--test t1 --datasets twitter_ego --job t1-twitter-ego` |
| `t1-instagram` | `--test t1 --datasets instagram --job t1-instagram` (large tier: check memory in the dry run and the first chunk) |
| `t1-youtube` | `--test t1 --datasets snap_youtube --job t1-youtube` (large tier: as above) |
| `t2-a` | `--test t2 --t2-seeds 1 2 3 --job t2-a` |
| `t2-b` | `--test t2 --t2-seeds 4 5 6 --job t2-b` |
| `certsens` | for R in 16 64 256: `--test t1 --datasets rice_facebook --attributes college --rhos 0.99 --configs method --cert-R $R --no-bounds --no-pareto --job certsens --out results/v2/parts/certsens/t1_certsens.csv`, then the same with `--datasets pokec_n --attributes region` |

T2 runs at 64 RR sets per node (design §21.2). If a sample does not fit in 16 GB, report it rather than lowering `--R`; a lower R is a deviation the user must approve.

### Session C: merge and summarise

After the B branches are merged:
1. Run `python experiments/v2/summarize_v2.py --figures`. It reads every `results/v2/parts/*/` file and warns about duplicated queries.
2. Run `interpret_v2.py` for the pre-registered cases whose graphs ran (Rice, Twitter-political, Pokec-n, Instagram). It reads the merged T1 rows.
3. Commit `results/v2/summary_v2.md`, the `summary_*.csv` files and `results/v2/figures/`.

## 5. What waits for the GPU machine

Since 2026-10-17 these run on Kaggle's GPU notebooks: `kaggle/README.md` has the jobs and their order (design §23).

- **The constructor (`train_constructor.py`, 5 seeds), trained before the start step of tuning**, which then runs as `tune_v2.py --steps start`. Until then T1 and T2 use the provisional greedy start. If the constructor wins on validation, the T1 rows of the greedy start become the "start" ablation row, and the method's learned-start rows are added (they have their own resume keys). This order was fixed on 2026-10-15, before any test result: the start rule stays validation-only.
- **The learned proposer and T3**, only if session A's W3 result says train.
- **T1 on full Pokec (three attributes) and on LinkedIn, their case studies, and T4.**
