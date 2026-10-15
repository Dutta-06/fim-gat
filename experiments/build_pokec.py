"""Build Pokec graphs from the SNAP dumps (Takac & Zabovsky 2012).

    python experiments/build_pokec.py --raw-dir <folder with soc-pokec-profiles.txt
                                                 and soc-pokec-relationships.txt>

Outputs data/pokec/{pokec_z,pokec_n,pokec_full}.npz with the largest weakly
connected component of each, edge list (u -> v as listed by SNAP), and three
sensitive attributes:

    region   pokec_z / pokec_n: the two districts (as in FairGNN, Dai & Wang 2021)
             pokec_full: province (kraj), small ones merged into "other"
    gender   female / male
    age      <=20, 21-30, >30, unknown (age missing in ~30% of profiles)
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pokec")
SUBGRAPHS = {
    "pokec_z": ["zilinsky kraj, zilina", "zilinsky kraj, martin"],
    "pokec_n": ["nitriansky kraj, nitra", "nitriansky kraj, levice"],
}


def _raw(raw_dir, name):
    """raw_dir/name, or SNAP's gzipped name.gz when only that is there (pandas reads either)."""
    path = os.path.join(raw_dir, name)
    return path if os.path.exists(path) or not os.path.exists(path + ".gz") else path + ".gz"


def load_raw(raw_dir):
    prof = pd.read_csv(_raw(raw_dir, "soc-pokec-profiles.txt"), sep="\t", header=None,
                       usecols=[0, 3, 4, 7], names=["id", "gender", "region", "age"],
                       dtype={"id": np.int64, "gender": "float32", "region": "string", "age": "float32"},
                       na_values=["null"], quoting=3, engine="c")
    rel = pd.read_csv(_raw(raw_dir, "soc-pokec-relationships.txt"), sep="\t", header=None,
                      names=["u", "v"], dtype=np.int64, engine="c").to_numpy()
    return prof.set_index("id"), rel


def age_groups(age):
    a = np.nan_to_num(age, nan=0.0)
    g = np.full(len(a), 3)
    g[(a > 0) & (a <= 20)] = 0
    g[(a > 20) & (a <= 30)] = 1
    g[a > 30] = 2
    return g, np.array(["<=20", "21-30", ">30", "unknown"])


def save(name, ids, prof, rel, region_codes, region_names):
    """Induced subgraph on ids, largest WCC, relabelled 0..n-1."""
    ids = np.asarray(ids)
    index = pd.Series(np.arange(len(ids)), index=ids)
    keep = np.isin(rel[:, 0], ids) & np.isin(rel[:, 1], ids)
    e = index.loc[rel[keep].ravel()].to_numpy().reshape(-1, 2)
    n = len(ids)
    A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = connected_components(A, directed=True, connection="weak")
    big = np.flatnonzero(comp == np.bincount(comp).argmax())
    remap = -np.ones(n, dtype=np.int64)
    remap[big] = np.arange(len(big))
    e = remap[e]
    e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0)]
    p = prof.loc[ids[big]]
    gender = np.nan_to_num(p["gender"].to_numpy(), nan=0).astype(np.int64)
    age, age_names = age_groups(p["age"].to_numpy())
    region = region_codes[big]
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), n=len(big), edges=e.astype(np.int64),
                        region=region, region_names=np.array(region_names),
                        gender=gender, gender_names=np.array(["female", "male"]),
                        age=age, age_names=age_names, orig_id=ids[big])
    print(f"{name}: n={len(big)} m={len(e)} region sizes={np.bincount(region).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    prof, rel = load_raw(args.raw_dir)
    for name, districts in SUBGRAPHS.items():
        sub = prof[prof["region"].isin(districts)]
        codes = np.array([districts.index(r) for r in sub["region"]])
        save(name, sub.index.to_numpy(), prof, rel, codes, [d.split(", ")[1] for d in districts])
    if not args.skip_full:
        kraj = prof["region"].fillna("unknown").str.split(",").str[0].to_numpy()
        vals, counts = np.unique(kraj, return_counts=True)
        major = [v for v, c in zip(vals, counts) if c >= 100_000]
        names = sorted(major) + ["other"]
        codes = np.array([names.index(r) if r in major else len(names) - 1 for r in kraj])
        save("pokec_full", prof.index.to_numpy(), prof, rel, codes, names)


if __name__ == "__main__":
    main()
