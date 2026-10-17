# FIM-GAT: handoff notes for Claude sessions

FIM-GAT is a fair influence-maximisation research codebase. Its current version, v2, targets **lower group disparity at a spread floor** (σ ≥ ρ × reference spread). Its adversarial core is a community-level "hive-mind" auditor, and every deployed seed set carries a certificate.

It was built from scratch on 2026-09-27 as a redo of `C:\Users\Odwitiyo\Desktop\xAI Project\Baseline_Test`. **That old project must stay untouched:** its data files were copied or read and its model was ported as the `prior_repadv_gat` baseline, but nothing in it was modified.

v1 (a GATv2 seed policy trained against a graph adversary, maximin coverage) ran to completion on 2026-09-27. Its record (write-up, results, figures, run scripts and the old handoff notes) was moved out of the repo on 2026-10-10, into `C:\Users\Odwitiyo\Desktop\FIM-GAT_archive\removed_2026-10-10\`:
- `README_v1.md`, `results/`, `figures/` and `experiments/`;
- `docs_before_cleanup/` (this file, the protocol and the design docs as they were);
- `fimgat_before_cleanup/` (the modules before dead code was removed).

v1's honest outcome:
- the learned policy won clearly on many-group maximin (SPA region, about 2× the best baseline);
- it trailed Frank–Wolfe by 0–4% on Pokec;
- its adversary was not shown to beat seed noise.

## User priorities (read first)

- The user asked for maximal novelty and results, with **adversarial graph networks as the core** of the project (it is in the title). Do not drop, rename or demote the adversarial component, or change headline claims, without the user's explicit approval.
- The user values honest reporting. Keep losses and caveats visible.
- **Delegation (2026-10-10).** The user gave authority to "transform this codebase into what we need". Make engineering and open methodology calls, record each with its reason in `docs/design_v2.md` §19, and report them.
- **Current phase (2026-10-15): running, cloud first.** The user said "we need to get fim-gat done fast ... as much stuff that can be covered with claude code should be done there asap". Everything a Claude Code cloud session can carry runs there, following `docs/cloud_runbook.md`; the GPU parts (constructor and proposer training, full Pokec, T3, T4) run on Kaggle's free T4 x2 notebooks (2026-10-17: `kaggle/README.md`; the user launches each job and brings back its output). Until 2026-10-15 the phase was codebase preparation only.
- The user's compute target is 16 GB VRAM (occasionally 40 GB); development happens on an 8 GB laptop. Ask before starting multi-hour runs.
- **Before running experiments, changing the method, or moving hardware, read `docs/research_protocol.md`.** It covers the run order, the pre-registration checklist, backups, launch commands, memory and the tuning rules.

## Where to pick up

**On a Claude Code cloud session (Linux VM):** follow `docs/cloud_mission.md`, your brief (setup, rules, jobs); `docs/cloud_runbook.md` has the background. The laptop-only advice below (the `python` vs `python3` warning, `PYTHONIOENCODING`, CRLF, the old project's paths) does not apply there.

**GPU work (Kaggle, 2026-10-17):** `kaggle/README.md` has the jobs, their order and the setup. `kaggle/run_job.py` defines the jobs, `kaggle/make_bundle.py` builds the upload, and `kaggle/import_outputs.py` merges the outputs back into `results/`. Design §23 records the choices (one preset, 8gb, for every pipeline script; timing jobs on one lane).

1. **Ask the user before multi-hour local runs.** The user said to start on 2026-10-15, cloud first (`docs/cloud_runbook.md`).
2. The run order is `docs/research_protocol.md` §1:
   - `build_datasets.py`: downloads from GitHub and SNAP, and reads the raw files in the old project;
   - optionally `train_constructor.py` (multi-hour GPU; ask first; without it the start set is greedy);
   - `tune_v2.py`;
   - only if its W3 rule says train: `train_proposer.py` (multi-hour GPU; ask first);
   - `run_t1_t2.py --test t1`, then `--test t2` (T2 runs at 64 RR sets per node: close other applications or use the 16 GB machine);
   - `run_t4.py`;
   - only if a proposer was trained: `run_t3.py`;
   - the certificate-size sensitivity run;
   - `interpret_v2.py` (the pre-registered case studies);
   - `summarize_v2.py --figures`.

   Each script prints a plan and an estimate first (`--dry-run`). Show it to the user before any long run.
3. **Open choices the user may still revisit:**
   - the delegated decisions of design §19 and the pre-run fixes of §21 (reviewable until the first test run);
   - the paper's framing: if the constructor loses to the greedy start and W3 says not to train, the deployed method has no neural part, so the "adversarial GAT" title would not match the evidence (the user's call; see the memory note of 2026-10-14);
   - the licence for a code release;
   - whether to port S3D or CrossWalk;
   - optionally, a C++ toolchain (WSL Ubuntu) to cross-check the RFIM port against the official binary.
4. **Deep context:**
   - the full conversation that built v1 and v2 (session `76a74c19-3fae-49d4-9459-c5e3a3b1eda3`) is in `C:\Users\Odwitiyo\.claude\projects\c--Users-Odwitiyo-Desktop-xAI-Project-Baseline-Test\`, and resumes from the `Baseline_Test` folder;
   - the auto-memory for this folder is in `C:\Users\Odwitiyo\.claude\projects\c--Users-Odwitiyo-Desktop-FIM-GAT\memory\`.

## Status (2026-10-14)

- **Cloud results (2026-10-17; job ALL, merged as PR #1; `results/v2/summary_v2.md`, `results/v2/parts/ALL/status.md`).** 6,518 rows, the committed code unchanged (code_tag f53a3e10824a).
  - Tuning: δ = 0.005, φ₀ = 0.4, start greedy (provisional), W3 says train (H = 8.1).
  - C1, 8 real-attribute graphs: a significant win against the best certified baseline at ρ = 0.99 only (p = 0.039; at 0.97 and 0.98, no difference). It is a win at every ρ against each of BSM, FW-Disparity and the Greedy family. Its weak spot is Pokec-n / region, where BSM is lower.
  - C2 (T2): a pre-registered null. (b) and (a) deployed identical sets in 60 of 60 cases, because κ − D_a was only about 0.0004.
  - Rice was built with 1,216 nodes (11 isolates) against the specified 1,205: a pending decision.
  - The adversary's framing after the C2 null is the user's call (protocol: ask, do not drop).
  - Still to run on a GPU: the constructor, the proposer and T3, full Pokec, LinkedIn and T4.
- **Kaggle G1 (2026-10-17; `results/v2/parts/kaggle-G1-train/`, 2.7 h on 2x T4, complete).**
  - Constructor, 6 models (seeds 0-4 and the no-adversary ablation): every validation query at rho = 0.99 fell back to greedy (none of the candidates met the floor min(0.99 + delta, 1) x reference; training reached 0.96-0.98 of the reference). The validation score was therefore constant, so the saved checkpoints are epoch 5 of 40. The adversary did not collapse (about 15 of 16 communities in use).
  - Start step: **greedy** (the constructor's start mean 0.0473 against greedy's 0.0364). The learned start fell back in 25 of 25 queries, so that score is greedy-on-A's, not the constructor's.
  - Proposer: imitation lowered validation recall@128, so every seed kept its starting policy, the constructor checkpoint: recall@128 0.983 against the simple proposer's 0.300. Its per-round cost is 7.1-7.4 times the simple proposer's (W3 headroom H = 8.1).
  - Pending the user's decision (2026-10-17): keep this result, or apply a declared fix (the decoder repairs a near-feasible set; checkpoint selection cannot tie on fallbacks), retrain, and rerun the start step, keeping this attempt on record.
- **Design:** `docs/design_v2.md`.
  - **§18 is the pre-registration amendment** the user approved before any run:
    - T2 tests the ε-constraint phase (`improve.hive_search`), with (b0), the published rule, also reported;
    - verified local search is part of the method, so C1 is judged against external baselines;
    - learning is an accelerator, gated by the W3 rule (`diagnostics.learning_decision`);
    - ρ = 0.99 for C1 and 0.97 for C2;
    - every deployed set is chosen by certified deployment.
  - **§19 records the choices made under the delegation:**
    - δ is tuned on validation;
    - the certificate sample B follows a memory rule;
    - R, `MIN_COMMUNITY` and `Z_ACCEPT` are unchanged;
    - the T1/T2 samples and the analysis plan are fixed.

    Every item stays reviewable until the first v2 test run.
  - **Module contracts:** `docs/v2_interfaces.md`. Later sections override earlier ones; §20–§23 hold the 2026-10-10 amendments.
- **Theory:** `docs/theory_v2.md`.
  - **Part I (R1–R3):**
    - few RR sets per node let the verifier accept moves that are sampling luck;
    - meeting ρ + δ on the search sample does not guarantee the true floor, but a certificate on an independent RR sample does (0 failures in 270 checks);
    - tabu prevents the cycling that family growth can cause.
  - **Part II (W1–W4):**
    - under the published rule the hive mind constrains the search but rarely drives it, while an ε-constraint phase lowered hidden-community disparity (−10.5% of the mean at ρ = 0.97, p = 0.005; median case −3.1%);
    - the problem is NP-hard;
    - the hybrid proposer keeps local optimality;
    - setpoint-scaled spread prices help.

  The checks are `experiments/v2/theory_checks.py` and `weakspots.py`, with outputs in `results/v2/theory/` and `results/v2/weakspots/`.
- **External baselines** (design §19.3):
  - BSM, Tsang et al.'s MaxMin ("MultiFW-MaxMin") and RFIM are in the suite, written from their papers (`fimgat/v2/external.py`).
  - The RFIM port gets four partitions: the labelled groups and a label-propagation partition, each with its inverse. It never gets Louvain, which the evaluation uses (design §20).
  - S3D and CrossWalk are not implemented. DQ4FairIM is not runnable as released.
- **Pre-run fixes of 2026-10-14 (design §21, approved by the user after a pre-run evaluation for TCSS):**
  - T2's deployment: a second rule, the path rule (`certify.choose(..., "path")`), is T2's primary deployment, so the ε phase is no longer discarded at deployment; every FIM-GAT query now records three rows (`certified` for T1, unchanged; `certified-path`; `final`);
  - T2 runs at 64 RR sets per node (the W1 evidence's R) on 6 samples (30 cases);
  - "holds across the range" means ρ ∈ {0.97, 0.98, 0.99}; ρ = 1.0 is reported descriptively;
  - T3 has a gated trainer (`train_proposer.py`), a runner (`run_t3.py`) and an operational pass rule;
  - BSM's whole τ sweep (`t1_pareto.csv`) and spread–disparity dominance tables and figures;
  - a T4 summary bug (deployed and final rows averaged together) is fixed.
- **Reviewer gaps closed in advance (design §20):**
  - the RFIM port;
  - case studies (`interpret_v2.py`; 11 cases pre-registered);
  - real-attribute graphs as the T1 headline;
  - the results matrix;
  - a certificate-size sensitivity run;
  - `docs/paper_notes.md` (reproducibility, limitations and ethics drafts).
- **Cloud runs (2026-10-15).** The code is cloud-ready:
  - RR samples are machine-independent (`diffusion.RR_CHUNKS`);
  - datasets fall back to their SNAP downloads when the old project's local copies are absent;
  - `scripts/cloud_setup.sh` sets up a VM;
  - `run_t1_t2.py --job` writes parallel jobs to `results/v2/parts/<job>/`, which `summarize_v2.py` merges.

  The private GitHub repository and the cloud environment's network allowlist are set up by the user; session A (data and tuning) runs first, then the B jobs in parallel (`docs/cloud_runbook.md` §4).
- **Pending:**
  - the cloud's C2 diagnostics (`docs/cloud_note_c2_diagnostics.md`, design §22), once the user has pushed the commit that adds them;
  - the Kaggle jobs (`kaggle/README.md`): G1 (constructor, start step, proposer) together with C1 (full Pokec's T1 without the constructor, CPU), then G2 (the learned start's rows, full Pokec's case studies), G3 (T3) and G4 (T4);
  - LinkedIn is not bundled (its node file holds personal data): the user decides whether to drop it or prepare an edges-only build;
  - the Rice node count (1,216 built against 1,205 specified).
- **Not done (each needs a download, a design change or the user's decision):** CrossWalk's released code, a DQ4FairIM reimplementation, S3D, IMM/OPIM-C as a runtime reference, a second diffusion model.

## Code map

- **`fimgat/v2/` (the method):**
  - **Foundations:**
    - `config.py`: every pre-registered constant, with the date it was decided;
    - `datasets.py`: registry, splits (train `pokec_z`, validation `pokec_t`, everything else test), loaders and BFS samples;
    - `reference.py`: the reference spread and the Monte Carlo equal-spread test;
    - `metrics_v2.py`: forward Monte Carlo metrics only.
  - **The search:**
    - `cover.py` and `verify.py`: the exact incremental verifier and the noise model;
    - `local_search.py`: exhaustive single swaps;
    - `improve.py`: the improvement phase and `hive_search`, T2's option (b);
    - `auditor.py`: the hive-mind auditor;
    - `prices.py`: the PI controller;
    - `proposer.py`: simple, community and learned proposals.
  - **Certification and analysis:**
    - `certify.py`: Theorem 2's certified deployment and the equal-spread judge;
    - `bounds.py`, `hidden.py`, `mixture.py` and `diagnostics.py`: the theory's diagnostics and the W3 rule.
  - **Learning:**
    - `features.py`, `model.py` and `train.py`: the constructor (GATv2 policy and the `CommunityAdversaryV2` adversary);
    - `imitation.py`: the learned proposer's training. It is dormant until the W3 gate says to train.
  - **Baselines:**
    - `baselines_v2.py`: the suite;
    - `external.py`: BSM, MultiFW and RFIM.
- **`fimgat/` (the foundation that v2 reuses):**
  - `diffusion.py`: RR sampling, `ExposureEstimator` and the Monte Carlo simulator;
  - `graph.py`;
  - `baselines.py`: `RRGreedy` and its greedy family, Degree, FW-Fair, AGE and the old project's model;
  - `metrics.py`: Louvain communities for evaluation;
  - `data.py`: Pokec and DC-SBM;
  - `model.py`: `SeedPolicy` and `soft_topk`;
  - `features.py`, `decode.py` and `objective.py`;
  - `train.py`: v1's context, rollout and decoding, which only the foundation tests use.
- **`experiments/v2/`:**
  - `pipeline_v2.py`: the shared per-graph pipeline;
  - `build_datasets.py`, `train_constructor.py`, `tune_v2.py`, `train_proposer.py`, `run_t1_t2.py`, `run_t4.py`, `run_t3.py`, `interpret_v2.py` and `summarize_v2.py`;
  - `common_v2.py`: recorder, provenance and memory presets;
  - `theory_checks.py` and `weakspots.py`;
  - `experiments/build_pokec.py`: the Pokec builder, which builds `pokec_t`.
- **`kaggle/`:** the Kaggle runner (`run_job.py`), the bundle builder, the importer and the notebook (`fimgat_kaggle.ipynb`); see `kaggle/README.md`.
- **`tests/`:** see `tests/README.md`.
  - The default run (smoke + spec) gives **167 passed, 10 skipped** (unbuilt datasets) on 2026-10-14, after the pre-run fixes. Added on 2026-10-17, and passing on their own: `test_c2_diagnostics.py` (1 test) and `test_kaggle_v2.py` (9, including the first end-to-end run of `train_constructor.py`, on synthetic stand-ins).
  - Stress tests are opt-in.

## Environment

- **Interpreter:** `python` on PATH is `C:\Users\Odwitiyo\Desktop\MeitY\.venv\Scripts\python.exe` (Python 3.11, torch 2.6.0+cu124, torch_geometric 2.7.0, numba 0.63.1). FIM-GAT has no venv of its own; `requirements.txt` lists the tested versions.
- **Never call `python3`.** It is the Microsoft Store Python 3.11 app alias, a different interpreter (torch 2.5.1+cu121, numba 0.61.0), and it hung when called from this shell.
- **Command prefix:** prefix Python commands with `PYTHONIOENCODING=utf-8` (the Windows console is cp1252), and add `-W ignore`. To hide the GPU, use `CUDA_VISIBLE_DEVICES=-1`, not an empty string.
- **Hardware:** RTX 4060 Laptop with 8 GB VRAM and 15.5 GB RAM. Only about 1–3 GB of RAM was free while other applications were open, and two development runs died of host out-of-memory errors. The `16gb` / `40gb` presets (`experiments/v2/common_v2.PRESETS`) were scaled from 8 GB measurements and have **never been run**.
- **Data:**
  - Raw Pokec dumps: `...\xAI Project\Baseline_Test\data\soc-pokec-{profiles,relationships}.txt` (read only).
  - Built Pokec graphs: `data/pokec/*.npz`.
  - v2 downloads go to `data/v2/raw/`, and builds to `data/v2/`.
  - Louvain caches: `data/cache/`.
  - The old project's `soc-linkedin.nodes` holds real profile handles and must never be opened or copied.
- **Tools:** there is no C++ compiler (`g++`/`gcc`) on this machine.

## Evaluation integrity rules (keep these)

- **Metrics come only from forward Monte Carlo** (`metrics_v2.evaluate_v2`), on a stream independent of everything that chose a set. Seed selection, search, deployment and checkpoint choice use RR samples only.
- **Equal spread is judged for every method by the same certificate,** `certify.certified_equal_spread`, on an RR sample B that no method saw. The Monte Carlo equal-spread test is reported for continuity only.
- **Report all test graphs, budgets and floors.** Group metrics are **rates**, not counts. Rows store the seed set.
- **Tuning happens on the validation split only** (`tune_v2.py` refuses anything else). Never tune or select on test rows. SNAP ground-truth communities are for evaluation only.
- **No cherry-picking.** The old project cherry-picked 9 of 24 graphs and used a count-based disparity; don't reintroduce either.

## Running experiments

- **Order and commands:** see `docs/research_protocol.md` §1. In short: build the datasets, optionally train the constructor, tune, run T1, T2 and T4, then summarise. Each script prints its plan and a rough estimate first; `--dry-run` stops there.
- **Resuming:** rows are skipped when their key exists. The keys include the tuned δ and φ₀, but not the code, preset or hardware. Commit `results/v2/` (git tracks it) or use a fresh `--out` before rerunning after a code change.
- **Pitfalls:**
  - Run one GPU job at a time on the 8 GB card.
  - Bash chains like `a; b` keep going after you kill `a`, so stop the background shell task too.
  - On Windows, Python's text mode writes CRLF. `CLAUDE.md` and `docs/research_protocol.md` are CRLF; the other docs and all code are LF.
