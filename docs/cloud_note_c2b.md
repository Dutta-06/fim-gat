# Cloud note: test C2b (pre-registered)

T2 (claim C2) was a null on com-Youtube. Its diagnostics found why: the auditor cannot see YouTube's user groups, which are not cohesive (median conductance 0.97 against the auditor's cap of 0.4).

**C2b** asks T2's question on SNAP com-Amazon, whose top-5,000 ground-truth communities are cohesive (median conductance 0.06–0.07, all within the cap). It tests two hypotheses:
- **H1:** does the method, as pre-registered, lower their disparity?
- **H2:** does it when its room is scaled to 5% of the labelled disparity?

The design, the pass rules and the predictions are fixed in `docs/design_v2.md` §25, written before any method ran on the graph. The script is `experiments/v2/run_c2b.py`. This is a **pre-registered** test: run it exactly as given.

## Rules

1. **Change nothing from earlier runs.** Leave every existing file under `results/v2/` as it is.
2. **Run the script as given.** Pass no arguments beyond those below, and change none of its constants or `config.C2B_*`. Its outputs go to `results/v2/parts/c2b/` only.
3. **Bugs.** For a genuine bug:
   - make the smallest fix;
   - run `python -m pytest -q -p no:cacheprovider tests/test_c2b.py tests/test_c2_diagnostics.py`;
   - commit the fix on its own as `fix: ...`, and report it.

   Never "fix" a result. A fix that changes what a run computes invalidates the pre-registration: stop and report instead.
4. **Nothing else.** No GPU jobs and no other experiments.
5. **Git.** Work on a new branch `claude/fimgat-c2b` from the updated `main`. Push after every chunk, never to `main`, and finish with a pull request.

## Steps

1. Get the code:
   ```
   git fetch origin && git checkout main && git pull
   git checkout -b claude/fimgat-c2b
   source .venv/bin/activate || (bash scripts/cloud_setup.sh && source .venv/bin/activate)
   python -m pytest -q -p no:cacheprovider tests/test_c2b.py tests/test_c2_diagnostics.py
   ```
   The tests must pass.

   The data is in git (`data/v2/snap_amazon.npz`, `data/v2/snap_amazon_cmty.npz`). Only if it is missing, build it: `python experiments/v2/build_datasets.py --only snap_amazon` (about 10 MB from snap.stanford.edu).
2. Dry run:
   ```
   python -u -W ignore experiments/v2/run_c2b.py --dry-run
   ```
   Put its plan in `results/v2/parts/c2b/status.md`. Expect 6 samples × 5 budgets × 2 floors = 60 cases, runs `a`, `b-1se` and `b-5pct`, R = 64, δ = 0.005 and φ₀ = 0.4.
3. The run, both floors in one process:
   ```
   python -u -W ignore experiments/v2/run_c2b.py
   ```
   - Run it in the background, with your monitoring, a memory watchdog and reboot recovery. The script resumes, skipping finished cases, so a restart continues where it stopped.
   - Use chunks only if your command limit requires them.
   - Commit and push `results/v2/parts/c2b/` after every sample or chunk.
   - Keep `status.md` current: the cases done, the rate you measure, and an ETA.
4. Finish:
   ```
   python -u -W ignore experiments/v2/run_c2b.py --summarize
   ```
   Commit `summary.md` and every file in `results/v2/parts/c2b/`, push, and open a pull request titled "C2b (pre-registered)".

**Rough cost:** no references are cached for this graph, so each sample computes its own (about 15–25 min at R = 64), plus about 8 min of RR sampling and about 2–3 min per case. That is about 1 h per sample and about 6 h in all. Expect T2's memory, a peak of about 7 GB.

## Your final message

Keep it under 200 words, facts only:
1. **The verdicts:** copy the H1 and H2 lines from `summary.md`.
2. **Run details:** the cases done, wall time, peak memory, any fixes or deviations, and the pull request link.

Do not interpret beyond that. The researcher reads the outcome against the reading fixed in §25.
