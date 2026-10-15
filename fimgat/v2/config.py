"""v2 constants: the decisions agreed with the user on 2026-09-27 (docs/design_v2.md) and the v2 paths.

Values marked "tuned on validation only" may be changed only from the validation split
(datasets.splits()["val"]), never from test graphs.
"""
import os

BUDGETS = (25, 50, 100, 150, 200)
DIFFUSION = "weighted_cascade"
SNAP_GT = "com-youtube"
RHO_DEFAULT = 0.99             # headline floor, confirmed by the user on 2026-10-10 (= RHO_C1)
RHO_REPORT = (0.97, 0.98, 0.99, 1.0)
REFERENCE_RUNS = 5
DELTA_MARGIN = 0.0025          # RR floor safety margin; tuned on validation only
PRICE_TAU = 0.02
MIN_COMMUNITY = 10             # tuned on validation only
MAX_CONDUCTANCE = 0.3          # tuned on validation only
N_TRAIN_SEEDS = 5
TIME_BUDGETS = (10, 60, 300)   # seconds per query, test T3
AUDIT_MAX_SIZE = 20000         # auditor community-size cap (push cost ~ max_size / alpha); tuned on validation only

# docs/theory_v2.md (R1-R3, 2026-10-10)
Z_ACCEPT = 0.0                 # noise-aware acceptance (verify.accept); 0 = plain rule. Tuned on validation only;
                               # results/v2/theory/ shows the effect on a synthetic graph
TABU = True                    # improve() never revisits a seed set (prevents cycling across family growth)
MAX_AUDITS = None              # cap on auditor calls per query (bounds the epochs in Theorem 1); None = no cap
CERT_ALPHA = 0.05              # certify.certified_choice: total failure probability of the deploy-time certificate

# Decisions of 2026-10-10, approved by the user before any v2 run (docs/design_v2.md section 18)
RHO_C1 = 0.99                  # headline floor of claim C1 (confirmed); RHO_DEFAULT is the same
RHO_C2 = 0.97                  # primary floor of claim C2 (test T2); 0.99 is reported as secondary
AUDIT_BAND_RATIO = 2.0         # multi-scale audits are the hive mind's default (auditor.size_bands); None = one scale
MAX_CONDUCTANCE_GRID = (0.3, 0.4, 0.5)   # phi_0 candidates, tuned on validation graphs against held-out attributes
PROPOSER_FALLBACK = "exhaustive"   # a proposer miss falls back to exhaustive search (the hybrid); None = widen / stop
PRICE_SETPOINT = 2 * DELTA_MARGIN  # slack above the floor that the spread price aims for; tuned on validation only
PRICE_KP = 1.0 / PRICE_SETPOINT    # PI gains scaled to the setpoint (theory W4); the published gains were (1.0, 0.5)
PRICE_KI = 0.5 / PRICE_SETPOINT
KAPPA_SE = 1.0                 # epsilon-constraint cap of T2 option (b): labelled optimum + KAPPA_SE standard errors
T2_MIN_PAIRS = 20              # paired cases (test subgraphs x budgets) test T2 must have
LEARN_HEADROOM_MIN = 1.5       # W3 decision rule: train a learned proposer only if headroom H / cost_ratio >= this ...
LEARN_SHARE_MIN = 0.2          # ... and the improvement phase takes at least this share of the per-query time

# Settled under the user's delegation of 2026-10-10 (docs/design_v2.md section 19)
CERT_SEED = 9001               # RR seed of the certificate's sample B; differs from every RR seed that chooses a set
CERT_NNZ_BUDGET = 60_000_000   # B's size rule: as large as this many RR-set entries (about 22 bytes each in RAM) ...
CERT_R_MAX = 512               # ... roots first (up to every node), then RR sets per root up to this
T2_SAMPLES = 6                 # test T2: BFS samples of snap_youtube (x 5 budgets = 30 paired cases; was 4, which
                               # left no margin over T2_MIN_PAIRS; 2026-10-14, design section 21)
T2_SAMPLE_SIZE = 100_000       # nodes per T2 sample
EVAL_MIN_COMMUNITY = 10        # a hidden community counts in T2's metric if this many of its members are in the graph
LB_MAX_WORK = 2e10             # bounds.disparity_lower_bound is skipped above n_groups * (k + 1) * nnz of this
TUNING_FILE = "tuning.json"    # validation results of tune_v2.py, read by the test scripts (under RESULTS_V2)
DELTA_GRID = (0.0025, 0.005, 0.01, 0.02)   # RR floor margin delta, tuned on validation (tune_v2.py; §16 item 4)

# Pre-run fixes approved by the user on 2026-10-14, before any v2 test run (docs/design_v2.md section 21)
RHO_HOLDS = (0.97, 0.98, 0.99)  # "C1 holds across the range": a significant win at each of these floors. rho = 1.0
                                # stays in RHO_REPORT, reported descriptively: there almost only S_ref certifies
DEPLOY_RULES = ("min_ucb", "path")   # certify.choose: Theorem 2's rule (T1's deployment) and the path rule (T2's)
DEPLOY_TOL_SE = KAPPA_SE       # path rule: the furthest certified path state whose labelled-disparity bound is within
                               # this many standard errors (on B) of the smallest one, the deployment analogue of kappa
T2_RR_PER_NODE = 64            # RR sets per node of the T2 samples' search sample A (the R of the W1 evidence, theory
                               # section 9.6); at the medium tier's 16 a 10-node community's rate is mostly noise
T3_RHO = RHO_C1                # test T3 runs at the headline floor
T3_SPEEDUP_MIN = 1.5           # T3 pass (i): the learned proposer reaches the simple hybrid's final RR disparity at
                               # least this many times sooner (the W3 threshold), on more than half of the test graphs
T3_PRIMARY_BUDGET = 60         # T3 pass (ii): lower Monte Carlo disparity at this budget (seconds; one of TIME_BUDGETS)
T3_RECALL_C = (8, 32, 128)     # recall@C reported in T3 (128 = improve()'s 16 x 8 proposal grid)

TIER = {"rice_facebook": "small", "twitter_political": "small", "pokec_z": "medium", "pokec_n": "medium",
        "pokec_t": "medium", "epinions": "medium", "twitter_ego": "medium", "pokec_full": "large",
        "instagram": "large", "snap_youtube": "large", "linkedin": "large"}
N_SIMS = {"small": 5000, "medium": 2000, "large": 1000}
RR_PER_NODE = {"small": 512, "medium": 16, "large": None}   # large: from the memory preset (full_R, full_max_roots)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_V2 = os.path.join(ROOT, "data", "v2")
RAW_V2 = os.path.join(DATA_V2, "raw")
RESULTS_V2 = os.path.join(ROOT, "results", "v2")
MODELS_V2 = os.path.join(RESULTS_V2, "models")
OLD_PROJECT = r"C:\Users\Odwitiyo\Desktop\xAI Project\Baseline_Test"   # read-only, never modified


def rr_floor(rho, total_ref, delta=DELTA_MARGIN):
    """Floor on the RR sample: (rho + delta) * total_ref, never above total_ref itself, so at rho = 1
    the reference set still qualifies (the margin only protects floors below equal spread)."""
    return min(float(rho) + float(delta), 1.0) * float(total_ref)
