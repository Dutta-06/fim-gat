# C2 diagnostics (exploratory, design section 22): status

Branch `claude/fimgat-c2diag` (from `main` at eef1374). Same 4 vCPU / 15 GB / no-GPU VM as job ALL. Follows `docs/cloud_note_c2_diagnostics.md`. Nothing from the earlier runs is touched; outputs go to `results/v2/parts/c2diag/` only.

## Plan (dry run, step 2)
```
C2 diagnostics (EXPLORATORY): 6 samples x 5 budgets x 2 floors; runs ['a', 'b-1se', 'b-1pct', 'b-5pct', 'b-20pct']; R=64; delta=0.005 phi0=0.4; code_tag=539c06dcb0d8; device=cpu
cases left: 60 of 60 (rough: 5-15 min each on a 4-CPU VM, plus about 25 min per sample for its cached stages)
```
- `tests/test_c2_diagnostics.py`: 1 passed (33 s).
- T2's references are reused from `results/v2/parts/t2/references.json` (the script's default); T2's rows for the reproduction check are read from `results/v2/parts/t2/t2.csv` (merged by `summarize_v2.load`).
- Order: rho = 0.97 first (30 cases), then rho = 0.99 (30 cases), then `--summarize`.
- Measured (sample 1 at rho = 0.97): ~8 min for the cached stages (reference loaded from T2's cache, rr_sample 225 s, cert_sample 222 s) plus ~3-4 min per case, i.e. ~30 min per sample; ETA ~2.5 h for the 30 cases at rho = 0.97 and ~2.5-3 h for rho = 0.99 (the samples are rebuilt on the second run). Peak job RSS so far 6.0 GB; a watchdog kills the job if MemAvailable < 800 MB.
- Validity check (`matches_t2` for variants a and b-1se): after 5 cases (10 rows each) all True, none False (checked after every case batch; I stop and report if any False appears).

## rho = 0.97: DONE (08:53 UTC; 30 cases, 300 room rows, 0 tracebacks)
- Wall time 2 h 27 min (06:26 -> 08:53), 6 samples (~25-30 min each including cached stages), no VM reboot, peak job RSS 6.0 GB, watchdog never fired.
- **Validity:** `matches_t2` is True in 60 of 60 rows for variant "a" and 60 of 60 for "b-1se" (30 cases x 2 deployments), none False.
- `--summarize` run and pushed (`summary.md`, rho = 0.97 tables only). Next: rho = 0.99 (started 08:54 UTC), then the final `--summarize` and the pull request.

## rho = 0.99: DONE (11:19 UTC; 30 cases) and final `--summarize`
- Wall time: rho = 0.97 2 h 27 min (06:26 -> 08:53), rho = 0.99 2 h 25 min (08:54 -> 11:19 UTC); total about 4 h 55 min. No VM reboot during this job. Peak job RSS 6.9 GB; the 800 MB watchdog never fired. 0 tracebacks.
- **Validity (both floors, 60 cases):** `matches_t2` is True in 120 of 120 rows for variant "a" and 120 of 120 for "b-1se" (60 cases x 2 deployments), none False.
- Rows: `room.csv` 600 (60 cases x 5 variants x 2 deployments), `coverage.csv`, `price.csv`, `summary.md`.
- Fixes: none. Deviations: none (the script was run as given, with no extra arguments beyond `--rhos`, `--dry-run` and `--summarize`; nothing from earlier runs was touched).
- Tooling note: the periodic snapshot commits on this branch are mine (a 30-minute `git add -A results/v2/parts/c2diag`), not part of the experiment.
