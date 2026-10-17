# Running the GPU work on Kaggle

The cloud runs (2026-10-15 to 17) did everything that needs no GPU. What is left is the GPU work:
- training the GAT constructor and the learned proposer;
- tuning's start step, which decides between the constructor's start and the greedy one;
- T3 and T4;
- full Pokec, which was not built on the cloud;
- the T1 rows that use the constructor's start.

All of it runs in Kaggle notebooks, one job per session, through `kaggle/run_job.py`.

## The jobs

| Job | Accelerator | Needs the outputs of | What it runs | Rough time (unverified; scaled from the cloud's runs and v1's full-Pokec run) |
|---|---|---|---|---|
| `G1b-train` | GPU T4 x2 | nothing | smoke run of both trainers; the constructor (5 seeds and the no-adversary ablation, split over the two GPUs); tuning's start step; the learned proposer. Attempt 2 (design §24); attempt 1, `G1-train`, is superseded and its output is ignored | about 3 h (attempt 1: 2.7 h) |
| `C1-pokec-full-region` | None (CPU) | nothing | T1 on full Pokec under region: every row except the learned start's | 2–6 h |
| `C1-pokec-full-gender` | None (CPU) | nothing | the same under gender | 2–6 h |
| `C1-pokec-full-age` | None (CPU) | nothing | the same under age | 2–6 h |
| `G2-learned` | GPU T4 x2 | G1b, the three C1 | the learned start's T1 rows on every test graph (full Pokec on GPU 0; the other graphs, then T2, on GPU 1); the three full-Pokec case studies | 5–8 h (T2 adds about 8 h if the start becomes the constructor's) |
| `G3-t3` | GPU T4 x2 (one GPU used) | G1b, the three C1 | T3 on Instagram, com-Youtube and full Pokec | 3–12 h |
| `G4-t4` | GPU T4 x2 (one GPU used) | G1b | T4 on the same three graphs | 4–8 h |

**Order:**
1. Start G1b and the three C1 jobs together. Kaggle limits how many sessions run at once; if it refuses one, run the C1 jobs one after another.
2. When G1b and the C1 jobs are done, run G2.
3. Then G3, then G4.

**GPU quota.** The GPU jobs total roughly 15–30 GPU hours. The weekly quota is about 30 h; it floats, so check yours on your Kaggle account page. Plan on one to two weeks. The C1 jobs use no GPU quota.

**Time limits.** A session ends at 12 h. The runner stops starting new commands at 11 h, and ends a running one in time to save the output. If a job is not finished, its log ends with `NOT FINISHED`. Run the same job again with that output attached: everything is resumable, and the new session continues where the last one stopped.

## One-time setup

1. **On the laptop**, commit any changes, then build the bundle:
   ```
   python kaggle/make_bundle.py
   ```
   This writes `kaggle_bundle/fimgat-bundle.zip` (about 170 MB). It holds:
   - the committed repository (code, datasets and the cloud's results);
   - full Pokec;
   - a manifest that the notebook checks.

   LinkedIn and raw downloads are never included.
2. **Kaggle > Datasets > New Dataset.** Upload the zip, name the dataset `fimgat-bundle`, and keep it **Private**. Kaggle may unpack the zip; the notebook handles either form.
3. **Kaggle > Code > New Notebook > File > Import Notebook.** Upload `kaggle/fimgat_kaggle.ipynb`.
4. In the notebook's side panel:
   - **Add Input > Your Datasets > `fimgat-bundle`**;
   - **Settings > Internet: on** (Kaggle asks for a phone-verified account; the notebook installs the cloud's package versions);
   - **Accelerator:** GPU T4 x2 for the G jobs, None for the C jobs.

## Running a job

1. In the first code cell, set `JOB` (for example `JOB = "G1b-train"`).
2. Attach the outputs the job needs (the table above), through **Add Input**. There are two ways:
   - **Your Work > the notebook version that ran the job:** its Output holds `outputs_<job>.zip`;
   - **or** download the zips and add them to a private dataset (for example `fimgat-outputs`); add a new version of it as more jobs finish.

   Extra attachments are harmless: a job uses only the outputs it needs, and its own earlier ones.
3. **Save Version > Save & Run All (Commit).** The version runs in the background; you can close the browser.
4. When it finishes, open the version's **Output** and download `outputs_<JOB>.zip`. `outputs_<JOB>.json` shows each command's status without unzipping.

For several jobs at once, use one notebook per job (**File > Copy**), each with its own `JOB`.

The log in the version's page shows:
- the environment (GPUs, RAM, package versions);
- the plan;
- a line when each command starts and ends;
- every 5 minutes, each running command's progress bar: done / total, time elapsed < time remaining;
- every 30 minutes, host memory and a refresh of the output zip.

Each command's full output is inside the zip, under `results/v2/parts/kaggle-<JOB>/logs/`.

## Bringing the results back

On the laptop, after downloading the zips:
```
python kaggle/import_outputs.py path/to/outputs_G1b-train.zip           # report: what would change
python kaggle/import_outputs.py path/to/outputs_G1b-train.zip --apply
python experiments/v2/summarize_v2.py --figures
```
The importer:
- writes only under `results/` and `data/cache/`;
- merges the results CSVs row by row, so outputs that extend the same file never lose rows;
- reports anything unfinished.

Or tell Claude where the zips are, and it will import them, check them and summarise.

## Choices made for these runs (design §23)

- **Preset.** Every pipeline script runs at the 8gb preset: the cloud's rows all used it, so each large graph keeps one search sample A whatever machine computes its rows. On a T4 this under-uses the card, which makes T4's timings conservative.
- **Timing tests.** T3 and T4 use one GPU and one process: a second job on the same CPUs would distort their clocks.
- **References.**
  - T4 computes its references afresh, because they are part of what it times.
  - T3 reuses T1's stored references: same keys, same values.
- **CPU jobs.** The C1 jobs never load the constructor. Their sessions would run its rows on the CPU, so G2 computes them on the GPU.
- **Constructor training.** The two GPUs train in parallel (seeds 0–2; seeds 3–4 and the ablation). Each model's training is unchanged; only the wall time of training is shared.

Nothing pre-registered changes.

## If something goes wrong

| Message | What to do |
|---|---|
| `this job needs a GPU` | Settings > Accelerator > GPU T4 x2 |
| `the pinned versions did not install` | turn Internet on; the fallback still runs, and the versions are recorded |
| `failed the smoke tests` | send the log to Claude; nothing ran |
| `NOT FINISHED` | run the same job again with this output attached |
| the session restarted (memory) | set `SERIAL = True` and run the job again (the lanes then run one after another) |
| `no output found for [...]` | attach the outputs that job needs |
