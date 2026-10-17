# Kaggle job G1, attempt 1 (2026-10-17): superseded

The first GPU run of the constructor and the learned proposer, kept as a record. Design §24 explains why it was superseded and what changed. The job ran in 2.74 h on two T4s and completed every command; its logs are in `results/v2/parts/kaggle-G1-train/`.

## What it found

- **Constructor** (seeds 0–4 and the no-adversary ablation, `models/constructor_*.pt`):
  - At every validation epoch, every query at ρ = 0.99 fell back to greedy (`val_fallback` = 1.0). No candidate met the floor min(0.99 + δ, 1) × reference; training reached 0.96–0.98 of the reference.
  - The validation score was therefore constant (−0.0423), so the strict "better than the best" rule kept epoch 5 of 40 for every model.
  - The community adversary did not collapse: about 15 of 16 communities stayed in use.
- **Start step** (`tuning_runs_start.csv`): greedy, with the constructor's mean labelled disparity 0.0473 against greedy's 0.0364.
  - All 25 learned-start queries fell back to RR-greedy on the search sample, so the learned score is that set's, identical across seeds.
  - It is not a measurement of the constructor.
- **Proposer** (`models/proposer_*.pt`, `train_proposer_log.csv`):
  - Imitation lowered validation recall@128, so every seed kept its initialisation, the constructor checkpoint.
  - Recall@128 was 0.983, against the simple proposer's 0.300. The per-round cost ratio was 7.1–7.4.

## Files

| File | Content |
|---|---|
| `models/` | the 11 checkpoints as G1 saved them |
| `train_log_gpu0.csv`, `train_log_gpu1.csv` | per-epoch constructor logs |
| `train_proposer_log.csv` | per-epoch proposer logs |
| `tuning_runs_start.csv` | the start step's 90 rows (30 queries × 3 deployments) |

`results/v2/tuning.json` keeps this attempt's decision under `attempts`.
