# FIM-GAT v2: mission brief for a Claude Code cloud session

You are a Claude Code session on a cloud VM: Ubuntu, about 4 vCPUs, 16 GB RAM, **no GPU**. The repository `Dutta-06/fim-gat` is checked out.

FIM-GAT is a fair influence-maximisation research project. Its experiments are **pre-registered**: the claims, tests, settings and analysis were fixed before any run. Your task is to run the CPU-feasible parts of that pipeline on this VM, as fast as the rules allow, and push every result to GitHub.

The researcher launching you may add one line under this brief: `JOB: <name>`. Then run only that job: A (§4), a B job (§5) or C (§6). If there is no such line, or it says `JOB: ALL`, your job is **ALL** (§4b): the whole list, in priority order, in this one session.

## 1. Rules that override speed

1. **Change nothing pre-registered.** That covers:
   - `fimgat/v2/config.py`;
   - `results/v2/tuning.json`, once session A has written it;
   - the method's code;
   - script arguments beyond those given here.

   Never tune or choose anything on a test graph. Ground-truth communities are used only by the scripts' own evaluation.
2. **Report everything.** Every row a script writes stays, including rows with status "missing" and their reason. Never delete, rerun-to-replace or filter results.
3. **Stay inside the CPU scope.** Never run `train_constructor.py`, `train_proposer.py`, `run_t3.py` or `run_t4.py`. Never build or run `linkedin` or `pokec_full`. These need the GPU machine.
4. **Fix environment problems, never results.** A script may crash from a genuine bug or a Linux environment problem. If so:
   - make the smallest fix;
   - run the affected tests (`python -m pytest -q tests/<file>`);
   - commit the fix on its own with a message starting `fix:`;
   - record it in your `status.md`.

   If you cannot fix it with confidence, stop and report it.
5. **Never commit raw downloads, or any file over 95 MB.** `data/v2/raw/` is gitignored.
6. **Git:**
   - work only on the branch `claude/fimgat-<job>` (for example `claude/fimgat-a`, `claude/fimgat-t1-small`);
   - push after every chunk;
   - never push to `main` and never force-push;
   - finish with a pull request into `main`.
7. **Be honest.** If something cannot be done (memory, network, time), say so in `status.md` and in your final message. Never estimate a result you did not compute.

## 2. Setup, for every job

```
git fetch origin
git checkout claude/fimgat-<job> 2>/dev/null || git checkout -b claude/fimgat-<job>   # an earlier session's progress, if any
bash scripts/cloud_setup.sh && source .venv/bin/activate
mkdir -p results/v2/parts/<job>
```

If `origin/claude/fimgat-<job>` exists, check it out and continue from it instead of starting over. The scripts skip every row that already exists.

Use `python` from `.venv`. `CLAUDE.md` describes the project, but its Windows-laptop advice (`python3`, `PYTHONIOENCODING`, CRLF, the old project's paths) does not apply here. Reference reading, only as needed:
- `docs/cloud_runbook.md`;
- `docs/research_protocol.md`;
- `docs/design_v2.md` §18–§21.

## 3. How to run a long script

Every experiment script is resumable. Run each one in chunks:

1. Start it in the background with your longest allowed command timeout, wrapped in `timeout` a few minutes shorter than that limit:
   ```
   timeout <S> python -u -W ignore <script> <args> >> results/v2/parts/<job>/log.txt 2>&1
   ```
   Use S = 1500 if your limit is 30 minutes, or S = 6900 if it is 2 hours.
2. When it ends, commit and push:
   ```
   git add -A results/v2 data && git commit -m "<job>: chunk N" && git push -u origin claude/fimgat-<job>
   ```
3. If the exit code was 124 (timed out), run the same command again. Repeat until it exits 0.
4. A crash with a traceback is not a timeout: apply rule 4.
5. After each chunk, update `results/v2/parts/<job>/status.md`:
   - what is done and what remains;
   - minutes per chunk and an ETA from your measured rate;
   - peak memory (`free -g` while it runs);
   - any warning.

Before the first chunk, run the same command once with `--dry-run` and put its plan in `status.md`. Its time estimates were guessed for a laptop GPU; replace them with your measured rate.

## 4. Job A: datasets and tuning (runs alone, first)

1. **Tests.**
   ```
   python -m pytest -q -p no:cacheprovider > results/v2/parts/A/tests.txt 2>&1
   ```
   Expect only passes and skips (skips for unbuilt or local-only datasets). If anything fails, apply rule 4 before going on.
2. **Datasets.**
   ```
   python -u -W ignore experiments/v2/build_datasets.py --only rice_facebook twitter_political epinions twitter_ego pokec_t instagram snap_youtube >> results/v2/parts/A/build_log.txt 2>&1
   ```
   - **Downloads:** they come from GitHub and snap.stanford.edu. If a download fails, your network allowlist lacks a domain. Write that in `status.md`, push and stop.
   - **Size warnings:** report every `WARNING` line of the build log (sizes that differ from the sources').
   - **Check:** run `python -m pytest -q tests/test_smoke_data.py`.
   - **Commit:** `data/v2/*.npz` and `data/pokec/pokec_t.npz`; check each is under 95 MB.
3. **Tuning** (validation district only; the start step waits for a GPU, by design):
   ```
   python -u -W ignore experiments/v2/tune_v2.py --steps delta phi0 w3 --dry-run
   python -u -W ignore experiments/v2/tune_v2.py --steps delta phi0 w3
   ```
   Run the second command in chunks (§3). It writes `results/v2/tuning.json`, `results/v2/tuning_runs.csv` and `results/v2/references.json`. Commit them all.
4. **Finish.**
   - Write in `status.md`:
     - the chosen δ and φ₀, with the scores per grid value;
     - the W3 decision (train or not, H and the reason);
     - the test summary and the dataset sizes;
     - the total time.
   - Push, and open a pull request titled "Cloud job A: datasets and tuning".

## 4b. Job ALL: everything, in one session (the default)

**Branch.** Work on `claude/fimgat-all`. If that branch exists on origin, continue from it: every step below skips what is already done, so a relaunched session picks up where the last one stopped.

**Stay in the loop.** Nobody is watching overnight. Do not end your turn while work remains: waiting for a running chunk is part of the job. While a chunk runs, check its log every few minutes, and as soon as it ends, commit, push and start the next chunk or step.
- Stop only when the list is done, or at a blocker you cannot fix under rule 4: network, memory, or a failing test.
- Record any blocker in `status.md`, push, then go on with the steps that do not depend on it.
- Keep `results/v2/parts/ALL/status.md` current after every chunk: the step you are on, what is done, the rate you measured, and what remains.

The order puts the most important results first. Run each `run_t1_t2.py` step in chunks as §3 describes. EXT is the list of external baselines:

```
EXT="Degree Greedy Parity-Greedy Welfare-Greedy Maximin-Greedy FW-Disparity AGE Prior BSM MultiFW-MaxMin RFIM"
T1="python -u -W ignore experiments/v2/run_t1_t2.py --test t1"
```

1. **Job A, §4 steps 1–3:** tests, datasets, tuning. Use `results/v2/parts/A/` for its logs. Do not open a pull request yet.
2. **T1 headline, small graphs:**
   ```
   $T1 --datasets rice_facebook twitter_political --configs method --methods $EXT --job t1-small
   ```
   Then run `python -u -W ignore experiments/v2/summarize_v2.py` and commit `results/v2/summary_v2.md`, so the first results are visible early.
3. **T2, the adversary's test:**
   ```
   python -u -W ignore experiments/v2/run_t1_t2.py --test t2 --t2-seeds 1 2 3 4 5 6 --job t2
   ```
   Then run the summary again.
4. **T1 headline, Pokec-n:** three runs, one for each `<a>` in region, gender and age:
   ```
   $T1 --datasets pokec_n --attributes <a> --configs method --methods $EXT --job t1-pokec-n-<a>
   ```
   Then run the summary again.
5. **T1 ablations**, into the same job folders:
   ```
   $T1 --datasets rice_facebook twitter_political --configs ablation --methods LocalSearch Simple-v2 --job t1-small
   ```
   Then the same three runs with `--datasets pokec_n --attributes <a> ... --job t1-pokec-n-<a>`.
6. **Certificate-size sensitivity:** the `certsens` row of §5.
7. **Supplements**, with every configuration and the whole suite (no `--configs`, no `--methods`):
   ```
   $T1 --datasets epinions --job t1-epinions
   $T1 --datasets twitter_ego --job t1-twitter-ego
   ```
   Then `--datasets instagram --job t1-instagram` and `--datasets snap_youtube --job t1-youtube`. The last two are large graphs: if one runs out of memory, record it and go on.
8. **Finish:**
   - run job C's two commands (§6);
   - write the final `status.md`;
   - push, and open one pull request titled "Cloud job ALL".

Your final message (§7) covers the whole list: each step, and whether it finished or what remains.

## 5. B jobs: the experiments (start after job A is done; they can run in parallel)

**Prerequisite: job A's outputs.**
- If `results/v2/tuning.json` is missing from your checkout (A's pull request is not merged yet), take A's outputs from its branch:
  ```
  git fetch origin claude/fimgat-a && git checkout origin/claude/fimgat-a -- results/v2/tuning.json results/v2/tuning_runs.csv data/v2 data/pokec/pokec_t.npz
  ```
- Check that `tuning.json` lists `delta`, `phi0` and `w3` under `"steps"`. If it does not, job A is not finished: stop and say so.

**Your command** is `python -u -W ignore experiments/v2/run_t1_t2.py` followed by your job's arguments:

| JOB | Arguments |
|---|---|
| `t1-small` | `--test t1 --datasets rice_facebook twitter_political --job t1-small` |
| `t1-pokec-n-region` | `--test t1 --datasets pokec_n --attributes region --job t1-pokec-n-region` |
| `t1-pokec-n-gender` | `--test t1 --datasets pokec_n --attributes gender --job t1-pokec-n-gender` |
| `t1-pokec-n-age` | `--test t1 --datasets pokec_n --attributes age --job t1-pokec-n-age` |
| `t1-epinions` | `--test t1 --datasets epinions --job t1-epinions` |
| `t1-twitter-ego` | `--test t1 --datasets twitter_ego --job t1-twitter-ego` |
| `t1-instagram` | `--test t1 --datasets instagram --job t1-instagram` |
| `t1-youtube` | `--test t1 --datasets snap_youtube --job t1-youtube` |
| `t2-a` | `--test t2 --t2-seeds 1 2 3 --job t2-a` |
| `t2-b` | `--test t2 --t2-seeds 4 5 6 --job t2-b` |
| `certsens` | three runs, one for each R in 16 64 256: `--test t1 --datasets rice_facebook --attributes college --rhos 0.99 --configs method --cert-R <R> --no-bounds --no-pareto --job certsens --out results/v2/parts/certsens/t1_certsens.csv`; then the same three with `--datasets pokec_n --attributes region` |

- Start with `--dry-run` (its plan goes in `status.md`), then run in chunks (§3).
- `t1-instagram` and `t1-youtube` are large graphs. Watch memory in the first chunk; if the VM runs out (MemoryError, a killed process), record it and stop. Those graphs then wait for the GPU machine.
- T2 runs at 64 RR sets per node (pre-registered). If a sample does not fit in memory, record it and stop. **Never lower `--R`**: that is a deviation only the researcher can approve.
- **Finish:**
  - Write in `status.md`:
    - the rows written, and the missing ones with their reasons;
    - the wall time and the peak memory;
    - any fix or warning.
  - Push, and open a pull request titled "Cloud job <job>".

## 6. Job C: the summary (after the B pull requests are merged into main)

```
python -u -W ignore experiments/v2/summarize_v2.py --figures
python -u -W ignore experiments/v2/interpret_v2.py
```

- `summarize_v2.py` merges every `results/v2/parts/*/` file.
- `interpret_v2.py` runs the pre-registered case studies whose graphs exist. It skips the full-Pokec cases, which are not built here.

Commit `results/v2/summary_v2.md`, `results/v2/summary_*.csv`, `results/v2/figures/` and `results/v2/cases/`, then open a pull request titled "Cloud job C: summary".

In `status.md` and your final message, copy:
- the T1 verdicts (headline, real-attribute graphs, ρ = 0.99 and "holds across the range");
- the T2 pass line;
- every WARNING the summary printed (untuned rows, repeated queries).

## 7. Your final message

Keep it under 200 words:
- the job, and whether it finished or what remains;
- rows written and missing, with reasons;
- wall time and peak memory;
- fixes and deviations, if any;
- the pull request link.
