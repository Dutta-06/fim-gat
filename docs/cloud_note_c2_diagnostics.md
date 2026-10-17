# Follow-up note for the cloud session that ran job ALL: the C2 diagnostics (exploratory)

You ran job ALL. Its T2 was a pre-registered null: (b) and (a) deployed the same set in all 60 cases, because κ − D_a was only about 0.0004. The researcher has decided to report C2 as a finding about **what the method cannot do, and why**. The result stays a failure, but it gets explained.

Three diagnostics will test the explanation on the same 60 T2 states:
- **room:** does the phase act when it may spend more labelled fairness?
- **coverage:** does the auditor find the communities that set the hidden disparity?
- **price:** what does community fairness cost in labelled fairness?

They are specified in `docs/design_v2.md` §22, written before they run, and implemented in `experiments/v2/c2_diagnostics.py`. The script was reviewed and tested: its control runs reproduce T2's deployed sets exactly. Everything here is **exploratory**. It never replaces T2's verdict.

## Rules

1. **Change nothing from your earlier runs.** Leave `t2.csv`, every `results/v2/parts/*/` file you wrote, and `tuning.json` exactly as they are.
2. **Run the script as given.**
   - Do not change its constants (`FRACTIONS`, `COST_FRACTIONS`, `SCAN_MAX`) or add arguments beyond those below.
   - Its outputs go to `results/v2/parts/c2diag/` only.
3. **Validity check.** In every case, the rows with `variant` "a" and "b-1se" must have `matches_t2` = True: they reproduce T2. If any case shows False, **stop**. The comparison is invalid there, and the researcher needs to know. Report the case and push what you have.
4. **Bugs.** For a genuine bug, make the smallest fix, run `python -m pytest -q tests/test_c2_diagnostics.py`, commit it on its own as `fix: ...`, and report it. Never "fix" a result.
5. **Nothing else.** No GPU jobs and no other experiments.
6. **Git.** Work on a new branch `claude/fimgat-c2diag` from the updated `main`. Push after every chunk, never to `main`, and finish with a pull request.

## Steps

1. Get the new code:
   ```
   git fetch origin && git checkout main && git pull
   git checkout -b claude/fimgat-c2diag
   source .venv/bin/activate || (bash scripts/cloud_setup.sh && source .venv/bin/activate)
   python -m pytest -q -p no:cacheprovider tests/test_c2_diagnostics.py
   ```
   The test must pass.
2. Dry run:
   ```
   python -u -W ignore experiments/v2/c2_diagnostics.py --dry-run
   ```
   Put its plan in `results/v2/parts/c2diag/status.md`.
   - It reuses T2's cached references (`results/v2/parts/t2/references.json`) and rebuilds T2's samples from `data/v2/snap_youtube.npz`.
   - Expect the same memory as T2: a peak of about 7 GB.
3. **T2's primary floor first**, for an early answer:
   ```
   python -u -W ignore experiments/v2/c2_diagnostics.py --rhos 0.97
   ```
   Run it as you ran T2:
   - in the background, with your monitoring and reboot recovery;
   - in chunks only if your command limit requires them (the script resumes: finished cases are skipped);
   - committing and pushing `results/v2/parts/c2diag/` after every sample or chunk.

   Then run `--summarize` once and push, so the ρ = 0.97 tables are on GitHub early.
4. Then the secondary floor:
   ```
   python -u -W ignore experiments/v2/c2_diagnostics.py --rhos 0.99
   ```
5. Finish:
   ```
   python -u -W ignore experiments/v2/c2_diagnostics.py --summarize
   ```
   Commit `summary.md` and the CSVs, push, and open a pull request titled "C2 diagnostics (exploratory)".

Rough time: about 25 minutes per sample for its cached stages, plus 5–15 minutes per case, with 30 cases per floor. Measure the real rate and keep an ETA in `status.md`.

## Your final message

Keep it under 250 words, facts only:
1. **Validity:** the reproduction rate (`matches_t2`) for "a" and "b-1se".
2. **Room:** for each run (b-1se, b-1pct, b-5pct, b-20pct) at each ρ:
   - the share of cases where phase 2 accepted a move;
   - the share where the deployed set differs from (a)'s;
   - the median change in hidden-community disparity, with wins and losses;
   - the median change in labelled disparity.
3. **Coverage:**
   - the median best Jaccard overlap with h⁺ and with h⁻;
   - the share of ground-truth communities within φ₀.
4. **Price:** the median cost of the cheapest community-improving swap (F and H), in units of SE and as a share of D_a, and how often none was found.
5. **Run details:** time, peak memory, fixes, deviations, and the pull request link.

Do not interpret beyond that. The researcher reads the outcome against the plan fixed in design §22.
