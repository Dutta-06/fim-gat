# Paper notes: reproducibility, limitations and ethics (draft)

Draft text for the paper's reproducibility statement and its limitations and ethics section, written on 2026-10-10 before any v2 result existed. The statements about results are placeholders (marked **[after results]**). The rest describes the method and the protocol, and can be used as it stands.

## Reproducibility statement

- **Pre-registration.** The claims, tests, decision rules, tuning procedure and analysis were fixed before any experiment. The dated record is `docs/design_v2.md` §10 and §18–§21. Any deviation is reported as a deviation.
- **Two deployment rules.** Both choose only certified sets, so both keep the floor guarantee:
  - Theorem 2's rule (the smallest labelled-disparity bound) is the method's deployment for C1;
  - the path rule (the furthest certified search state within one standard error of that bound) is the deployment for C2.

  Both are reported for every query.
- **Code and data.** All code is in this repository. The datasets are public:
  - Pokec, Epinions, ego-Twitter and com-Youtube from SNAP;
  - Rice-Facebook, Twitter-political and Instagram from the repositories named in `fimgat/v2/datasets.py`.

  `experiments/v2/build_datasets.py` downloads and builds them. LinkedIn is used as an edge list only.
- **Provenance.** Every result row stores its seed set, so every metric can be recomputed. It also stores a hash of the code that produced it, the GPU and the memory preset.
- **Randomness.** Every RR sample, Monte Carlo stream and training run has a fixed seed. RR samples and Monte Carlo streams are the same on every machine: both are split into a fixed number of independently seeded chunks. CUDA scatter-add is nondeterministic, so GPU reruns of the learned parts agree at seed-noise level, not bit for bit (`docs/research_protocol.md` §7).
- **Baselines.**
  - BSM, Tsang et al.'s MaxMin and RFIM were reimplemented from their papers on the same RR samples as every other method. The paper should say so, and should note the deviations: RFIM runs with exactly k seeds and our partitions; MaxMin uses its multiplicative-weights inner solver.
  - DQ4FairIM could not be run: its released code imports a simulator module that does not exist in its repository.
  - S3D (two groups only) and CrossWalk (small graphs only) were not reimplemented.
- **Licence.** Still to be chosen by the authors before release.

## Limitations

- **Groups are given, not discovered, for the headline claim.** C1 uses labelled attributes. Degree groups on the structural graphs are a proxy, reported as a supplement. C2 tests hidden communities only on com-Youtube's user-created groups, which cover a minority of users and overlap.
- **The disparity measure is one choice among several.** Max-minus-min group reach is a demographic-parity-style notion. It does not capture maximin welfare, envy-freeness or individual fairness, and lowering it can lower the reach of the best-served group. The case studies report every group's reach so that readers can judge who pays.
- **Guarantees are about the model, not the world.** The certificate guarantees the spread floor under the Independent Cascade model with weighted-cascade probabilities. Real diffusion differs, and the edge probabilities are a modelling assumption, not measurements.
- **The certificate trades fairness for safety.** When the certificate's sample is small, the method deploys the efficiency reference set rather than an uncertified fairer one (design §18.7). **[after results:** how often this happened, from the certificate pass rates and deployment fields.**]**
- **Scale.** The large tier ran on an 8 GB GPU with sampled RR roots. **[after results:** which baselines did not finish within the time limit.**]**
- **The learned components** (the constructor, and the learned proposer if the W3 gate trains one) were trained on Pokec districts only. Transfer to other networks is tested zero-shot, not assumed.

## Ethics

- **Dual use.** Seed selection that steers information towards or away from groups can serve outreach, such as public health or civic information, or targeting. The method's objective is to *reduce* reach gaps between groups, but the same machinery with a different objective could widen them. The paper should say this plainly.
- **Group attributes.** Fairness auditing needs sensitive attributes (gender, age, region). The datasets are public and anonymised. No attribute is inferred, and held-out attributes are used for evaluation only.
- **Personal data.** The LinkedIn graph is used as an anonymous edge list. Its node file, which holds real profile handles, is never opened or copied.
- **The hive-mind auditor finds communities the method treats as under- or over-served.** Those communities are structural (cohesive regions of the graph), not inferred identities, and they are never exported or labelled with sensitive attributes beyond the aggregate composition reported in the case studies.
- **Reporting.** All test graphs, budgets and floors are reported, including the tests that fail. Rates are reported, not counts.
