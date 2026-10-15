"""Case studies for the paper's social interpretation (docs/design_v2.md section 20): who gains and who loses, and what the
hive-mind auditor found.

    cd experiments/v2
    PYTHONIOENCODING=utf-8 python -u -W ignore interpret_v2.py --dry-run
    PYTHONIOENCODING=utf-8 python -u -W ignore interpret_v2.py                  # every pre-registered case

The cases are fixed before any result exists (CASES): every T1 graph with a real attribute, at k = 50 and rho = 0.99,
the method's smallest train seed. --case runs another one, and its outputs are marked "not pre-registered".

For each case, the script reads t1.csv for the recorded sets:
- the method's deployed set: v2-b, the tuned start, the simple proposer, deployment "certified";
- the best certified external baseline;
- the reference set.

It then reruns the method's query on the same samples with the same tuned values, so the search repeats exactly; a
mismatch with the recorded set is reported. This recovers the auditor's community family. Everything is scored with
forward Monte Carlo on the graph's evaluation stream.

Outputs, in results/v2/cases/<dataset>_<attribute>_k<k>_rho<rho>/:
  groups.csv       every group of every attribute (the labelled one and the held-out ones): size, and reach under the
                   reference, the method and the best baseline, with the method's change
  communities.csv  every community the auditor found: size, conductance, the most over-represented group of each
                   attribute (with its lift over its population share), and reach under the reference, the method and
                   the best baseline
  case.md          a readable summary
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pipeline_v2 as P                                                   # noqa: E402
from common_v2 import device, mc_seed_for, n_sims_for, preset_config      # noqa: E402
from fimgat.v2 import datasets as D                                       # noqa: E402
from fimgat.v2.auditor import conductance                                 # noqa: E402
from fimgat.v2.config import RESULTS_V2                                   # noqa: E402
from fimgat.v2.metrics_v2 import simulate                                 # noqa: E402

INTERNAL = ("LocalSearch", "Simple-v2")
# pre-registered on 2026-10-10, before any v2 result: every real-attribute T1 graph, k = 50, rho = 0.99
CASES = tuple((d, a, 50, 0.99) for d, attrs in (("rice_facebook", ("college", "age", "major")),
                                                ("twitter_political", ("leaning",)),
                                                ("pokec_n", ("region", "gender", "age")),
                                                ("pokec_full", ("region", "gender", "age")),
                                                ("instagram", ("gender",))) for a in attrs)


def truthy(v):
    return str(v) in ("True", "1", "1.0", "true")


def pick_rows(t1, dataset, attribute, k, rho, start):
    """(method row, best certified external baseline row or None) of a case."""
    c = t1[(t1.dataset == dataset) & (t1.attribute == attribute) & (t1.k == k) & (np.isclose(t1.rho, rho))
           & (t1.status == "ok")]
    m = c[(c.method == P.FIM) & (c.variant == "v2-b") & (c.start == start) & (c.proposer == "simple")
          & (c.deployment == "certified")].sort_values("train_seed")
    if m.empty:
        return None, None
    b = c[(c.method != P.FIM) & ~c.method.isin(INTERNAL)]
    b = b[b.cert_passed.map(truthy)].sort_values("disparity")
    return m.iloc[0], (b.iloc[0] if len(b) else None)


def node_probs(graph, seeds, n_sims):
    prob, _ = simulate(graph, seeds, n_sims=n_sims, seed=mc_seed_for(graph))
    return np.asarray(prob, dtype=np.float64)


def run_case(case_key, args, t1, tuning, dev, preset, models):
    dataset, attribute, k, rho = case_key
    start = tuning["start"]
    m_row, b_row = pick_rows(t1, dataset, attribute, k, rho, start)
    name = f"{dataset}_{attribute}_k{k}_rho{rho:g}"
    out_dir = os.path.join(args.out_dir, name)
    if m_row is None:
        print(f"{name}: no method row in t1.csv; skipped", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    cfg = P.FimConfig("v2-b", start, "simple")
    ts = int(m_row.train_seed)
    case = P.Case(lambda: D.load(dataset, attribute), [k], args, dev, preset, need_features=cfg.learned,
                  need_rr_greedy=cfg.start == "learned", phi0=float(tuning["phi0"]), delta=float(tuning["delta"])).prepare()
    g = case.g
    res = P.fim_query(case, k, rho, cfg, models.get(ts), None)
    recorded = sorted(json.loads(m_row.seeds))
    same = res["deployed"] == recorded
    if not same:
        print(f"{name}: WARNING the rerun deployed a different set than t1.csv records; the recorded set is "
              f"used for reach, the rerun's family for the communities", flush=True)
    comms = [np.asarray(c, dtype=np.int64) for c in res["search"].get("_communities", [])]
    ns = int(args.n_sims) if args.n_sims else n_sims_for(g)
    sets = {"reference": list(case.refs[k]["seeds"]), "method": recorded}
    if b_row is not None:
        sets["baseline"] = json.loads(b_row.seeds)
    prob = {s: node_probs(g, seeds, ns) for s, seeds in sets.items()}

    rows = []
    for attr, labels in g.meta["attrs"].items():
        labels = np.asarray(labels)
        names = g.meta.get("attr_names", {}).get(attr, [])
        size = np.bincount(labels).astype(np.float64)
        for grp in np.flatnonzero(size > 0):
            r = {"attribute": attr, "labelled": attr == attribute, "group": names[grp] if grp < len(names) else grp,
                 "size": int(size[grp])}
            for s, p in prob.items():
                r[f"reach_{s}"] = float(p[labels == grp].mean())
            r["change_method_vs_reference"] = r["reach_method"] - r["reach_reference"]
            rows.append(r)
    groups = pd.DataFrame(rows)
    groups.to_csv(os.path.join(out_dir, "groups.csv"), index=False)

    crow = []
    for i, c in enumerate(comms):
        r = {"community": i, "size": int(len(c)), "conductance": float(conductance(g, c))}
        for attr, labels in g.meta["attrs"].items():
            labels = np.asarray(labels)
            pop = np.bincount(labels, minlength=labels.max() + 1) / len(labels)
            inside = np.bincount(labels[c], minlength=len(pop)) / max(len(c), 1)
            lift = np.where(pop > 0, inside / np.maximum(pop, 1e-12), 0.0)
            top = int(np.argmax(lift))
            names = g.meta.get("attr_names", {}).get(attr, [])
            r[f"{attr}_top_group"] = names[top] if top < len(names) else top
            r[f"{attr}_top_lift"] = float(lift[top])
        for s, p in prob.items():
            r[f"reach_{s}"] = float(p[c].mean())
        crow.append(r)
    communities = pd.DataFrame(crow)
    communities.to_csv(os.path.join(out_dir, "communities.csv"), index=False)

    lab = groups[groups.labelled]
    worst_ref = lab.sort_values("reach_reference").iloc[0]
    md = [f"# Case {name}" + ("" if case_key in CASES else " (not pre-registered)"), "",
          f"Method row: train seed {ts}, deployed {'the reference' if truthy(m_row.get('deploy_is_reference')) else 'a searched set'}"
          f"; rerun reproduced the recorded set: {same}. Best certified external baseline: "
          f"{b_row.method if b_row is not None else 'none'}.", "",
          f"Labelled disparity (MC): method {m_row.disparity:.4f}"
          + (f", best baseline {b_row.disparity:.4f}" if b_row is not None else "") + ".",
          f"The worst-served labelled group under the reference, {worst_ref.group}, goes from "
          f"{worst_ref.reach_reference:.4f} to {worst_ref.reach_method:.4f} under the method.", "",
          "Groups gaining most / losing most (labelled):",
          lab.sort_values("change_method_vs_reference").tail(3)[["group", "size", "reach_reference", "reach_method"]]
          .to_string(index=False),
          lab.sort_values("change_method_vs_reference").head(3)[["group", "size", "reach_reference", "reach_method"]]
          .to_string(index=False), "",
          f"The auditor found {len(comms)} communities (communities.csv)."]
    with open(os.path.join(out_dir, "case.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"{name}: wrote {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs=4, metavar=("DATASET", "ATTRIBUTE", "K", "RHO"), default=None,
                    help="one other case (marked not pre-registered)")
    ap.add_argument("--t1", default=os.path.join(RESULTS_V2, "t1.csv"))
    ap.add_argument("--tuning", default=None)
    ap.add_argument("--out-dir", default=os.path.join(RESULTS_V2, "cases"))
    ap.add_argument("--dry-run", action="store_true")
    P.add_pipeline_args(ap)
    args = ap.parse_args()
    P.check_args(ap, args)
    cases = [(args.case[0], args.case[1], int(args.case[2]), float(args.case[3]))] if args.case else list(CASES)
    print("cases:", ", ".join(f"{d}/{a} k={k} rho={r:g}" for d, a, k, r in cases), flush=True)
    if args.dry_run:
        return
    from summarize_v2 import load as load_results       # the shared file merged with every job's part
    t1 = load_results(args.t1)
    if t1 is None:
        sys.exit(f"{args.t1} missing (and no results/v2/parts/*/t1.csv): run run_t1_t2.py --test t1 first")
    tuning = P.load_tuning(args.tuning)
    dev = device()
    _, preset = preset_config(args.preset)
    models = P.load_models(args, dev, starts=[tuning["start"]])
    for c in cases:
        if not D.available(c[0]):
            print(f"{c[0]}: not built; skipped", flush=True)
            continue
        run_case(c, args, t1, tuning, dev, preset, models)
        P.free()


if __name__ == "__main__":
    main()
