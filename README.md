# FIM-GAT v2: certified fair influence maximisation against an adversarial community auditor

FIM-GAT chooses k seed nodes in a social network so that information spreads **as fairly as possible across groups** while total spread stays at a floor. The floor is σ(S) ≥ ρ · σ_ref, with σ_ref the best efficiency-only spread. Fairness is measured as **disparity**: the gap between the best- and worst-reached group under the Independent Cascade model with weighted-cascade probabilities.

The method has four parts:
- **Start set.** A GATv2 constructor proposes it, or it starts from the efficiency reference, whichever wins on validation.
- **Search.** An exact incremental verifier on reverse-reachable (RR) sets improves the set.
- **Adversary.** A **hive-mind community auditor** keeps finding the cohesive communities the current seeds serve worst or best. An ε-constraint phase then lowers disparity on those communities as well, which is how fairness transfers to groups nobody labelled.
- **Certified deployment.** Every deployed set is chosen on an independent RR sample, with a certificate that it meets the spread floor (Theorem 2 of `docs/theory_v2.md`).

**Status (2026-10-14):** the design is pre-registered, with the pre-run fixes of design §21, and the code and experiment scripts are written and tested. **No v2 experiment has run yet**, and of the v2 datasets only the Pokec graphs are built.

## Repository

| Path | What |
|---|---|
| `fimgat/v2/` | The method: verifier, auditor, search, certification, constructor, baselines (including BSM, Tsang et al.'s MaxMin and RFIM, written from their papers) |
| `fimgat/` | The foundation v2 reuses: RR sampling and Monte Carlo, graphs, the greedy family, Louvain evaluation, the GATv2 policy |
| `experiments/v2/` | The scripts: `build_datasets.py`, `train_constructor.py`, `tune_v2.py`, `run_t1_t2.py`, `run_t4.py`, `interpret_v2.py`, `summarize_v2.py`, all on `pipeline_v2.py` |
| `docs/design_v2.md` | The design, the pre-registered tests T1–T4, and the decisions with their dates (§18–§20) |
| `docs/theory_v2.md` | Proofs and numerical checks: noise and the winner's curse, termination, the floor certificate (R1–R3); the weak spots W1–W4 |
| `docs/v2_interfaces.md` | Module contracts; later sections override earlier ones |
| `docs/research_protocol.md` | How to run the experiments without corrupting them: run order, tuning rule, reporting checklist, memory |
| `docs/paper_notes.md` | Draft reproducibility statement, limitations and ethics |
| `tests/` | Smoke, spec and (opt-in) stress suites; see `tests/README.md` |
| `results/v2/` | Outputs; for now only the theory checks (`theory/`, `weakspots/`) |

## Setup

Use Python 3.11. Install PyTorch for your CUDA version first, then `pip install -r requirements.txt`. On the development machine, use `python`, not `python3`, and prefix commands with `PYTHONIOENCODING=utf-8`.

```bash
PYTHONIOENCODING=utf-8 python -m pytest            # smoke + spec, about 2 minutes
PYTHONIOENCODING=utf-8 python -m pytest -m stress  # opt-in, minutes, needs the Pokec builds and a GPU
```

## Running the experiments

The order is fixed by the pre-registration. `docs/research_protocol.md` §1 has the exact commands and the checklist. Every script prints its plan and an estimate first, and `--dry-run` stops there.

1. `build_datasets.py`: download and build the graphs.
2. `train_constructor.py`: optional; without it the start set is the efficiency reference.
3. `tune_v2.py`: tune on the validation district only. Only if its W3 rule says so, then `train_proposer.py`.
4. `run_t1_t2.py --test t1`, then `--test t2`.
5. `run_t4.py`, then `run_t3.py` if a proposer was trained.
6. The certificate-size sensitivity run.
7. `interpret_v2.py`: the pre-registered case studies.
8. `summarize_v2.py --figures`.

The pre-run fixes of 2026-10-14 (design §21) give T2 its own deployment rule and sample size, operationalise T3, and add BSM's whole trade-off and spread–disparity figures.

**Claude Code cloud sessions** (CPU-only VMs) run everything that needs no GPU: setup with `bash scripts/cloud_setup.sh`, then the jobs of `docs/cloud_runbook.md`.

## v1

v1 (2026-09-27) trained a GATv2 seed policy against a graph adversary for maximin coverage:
- it won clearly on many-group maximin (SPA region, about 2× the best baseline);
- it trailed Frank–Wolfe by 0–4% on Pokec;
- its adversary was not shown to beat seed noise.

Its write-up, results, figures and run scripts were moved out of the repository on 2026-10-10 to `Desktop\FIM-GAT_archive\removed_2026-10-10\`.
