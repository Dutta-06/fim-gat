# C2b (pre-registered, design section 25): status

Branch `claude/fimgat-c2b` (from `main` at a2285ea). Same 4 vCPU / 15 GB / no-GPU VM as the earlier cloud jobs. Follows `docs/cloud_note_c2b.md`. Nothing under `results/v2/` that existed before is touched; outputs go to `results/v2/parts/c2b/` only.

## Plan (dry run, step 2)
```
C2b (PRE-REGISTERED, design section 25): 6 samples x 5 budgets x 2 floors; runs ['a', 'b-1se', 'b-5pct']; R=64; delta=0.005 phi0=0.4; code_tag=4514edc87490; device=cpu
cases left: 60 of 60 (rough: 5-15 min each on a 4-CPU VM, plus about 25 min per sample for its cached stages)
```
- Tests: `tests/test_c2b.py` + `tests/test_c2_diagnostics.py`: 3 passed (35 s).
- Data: `data/v2/snap_amazon.npz` and `data/v2/snap_amazon_cmty.npz` are in git (7.3 and 4.8 MB).
- The run: `python -u -W ignore experiments/v2/run_c2b.py` (both floors in one process), no arguments, no constants changed. References are cached per sample in `results/v2/parts/c2b/references.json`; the script resumes by skipping finished cases.
- Monitoring: memory/RSS logged every 30 s in `mem.log`, a watchdog kills the job if MemAvailable < 800 MB, and `results/v2/parts/c2b/` is committed and pushed to this branch every 30 min and after every sample. No `timeout` wrapper (a lesson from job ALL).
- ETA (note's estimate, not yet measured): ~1 h per sample, ~6 h in all.
