"""Download the v2 raw files and build data/v2/<name>.npz and data/pokec/pokec_t.npz (docs/v2_interfaces.md s16).

    python experiments/v2/build_datasets.py                                  # everything not built yet
    python experiments/v2/build_datasets.py --only rice_facebook twitter_political
    python experiments/v2/build_datasets.py --no-download                    # use raw files already present
    python experiments/v2/build_datasets.py --only epinions --force          # rebuild a built dataset

GitHub and SNAP files are downloaded with urllib into data/v2/raw/<name>/. Epinions, ego-Twitter, LinkedIn and
the Pokec dumps are read in place from the old project, which is never written to; soc-linkedin.nodes is never
opened. pokec_t, the validation district (Trencin + Povazska Bystrica), is built with experiments/build_pokec.py's
load_raw and save, exactly as v1 built pokec_z and pokec_n. pokec_z, pokec_n and pokec_full are not rebuilt here.
Resumable: raw files and built datasets that already exist are skipped (datasets unless --force).
"""
import argparse
import gc
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fimgat.v2 import datasets as D                  # noqa: E402
from fimgat.v2.config import OLD_PROJECT, RAW_V2    # noqa: E402

ORDER = ("rice_facebook", "twitter_political", "epinions", "twitter_ego", "pokec_t", "instagram", "snap_youtube",
         "snap_amazon", "snap_dblp", "linkedin")
# (nodes, edges as the source counts them) from docs/design_v2.md section 8; None = not stated
EXPECTED = {"rice_facebook": (1_205, 42_443), "twitter_political": (3_753, 6_993), "epinions": (75_879, 508_837),
            "twitter_ego": (81_306, 1_768_149), "instagram": (553_628, None), "snap_youtube": (1_134_890, 2_987_624),
            "snap_amazon": (334_863, 925_872), "snap_dblp": (317_080, 1_049_866),   # the SNAP pages (C2b, section 25)
            "linkedin": (None, None)}
# UNVERIFIED: rough guesses for the 8 GB laptop (15.5 GB RAM); nothing here has been run yet
ESTIMATE = {"rice_facebook": "seconds", "twitter_political": "seconds", "epinions": "about 10 s",
            "twitter_ego": "under 1 min", "pokec_t": "3-6 min (reads the 30.6M-line Pokec dump; RAM about 4 GB)",
            "instagram": "under 1 min (about 13 MB download)", "snap_youtube": "1-3 min (tens of MB download)",
            "snap_amazon": "under 1 min (about 6 MB download)", "snap_dblp": "under 1 min (about 7 MB download)",
            "linkedin": "3-10 min (19.4M lines; RAM about 3-4 GB)"}


def download(url, dest):
    """Stream url to dest through a .part file, so an interrupted download is never mistaken for a raw file."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "FIM-GAT-v2-dataset-builder"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def fetch(name, raw_dir, allow_download):
    """Download missing URL sources of `name`; return the raw files still missing."""
    missing = []
    for src, local in D.raw_files(name, raw_dir):
        if os.path.exists(local):
            continue
        if D.is_url(src) and allow_download:
            t = time.time()
            print(f"  downloading {src}", flush=True)
            download(src, local)
            print(f"    {os.path.getsize(local) / 2**20:.1f} MB in {time.time() - t:.0f} s -> {local}", flush=True)
        else:
            missing.append(local)
    return missing


def build_pokec_t(raw_dir=None, allow_download=True):
    """Induced subgraph of the two Trenciansky districts, largest WCC (build_pokec.save), data/pokec/pokec_t.npz.
    The raw dumps are the old project's local copies on the development laptop, else SNAP's gzipped files,
    downloaded to <raw_dir>/pokec_t/ (datasets.SOURCES)."""
    raw_dir = D.RAW_V2 if raw_dir is None else raw_dir
    missing = fetch("pokec_t", raw_dir, allow_download)
    if missing:
        raise FileNotFoundError(f"raw Pokec dumps missing: {missing}")
    files = [p for _, p in D.raw_files("pokec_t", raw_dir)]
    raw = os.path.dirname(files[0])                     # both dumps sit in one folder (local or downloaded)
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import build_pokec                                # v1 builder, imported and not modified

    districts = list(D.POKEC_T_DISTRICTS)
    prof, rel = build_pokec.load_raw(raw)
    sub = prof[prof["region"].isin(districts)]
    counts = sub["region"].value_counts()
    absent = [d for d in districts if int(counts.get(d, 0)) == 0]
    if absent:
        raise ValueError(f"districts not found in the Pokec profiles: {absent}")
    print(f"  profiles per district: {[int(counts[d]) for d in districts]}", flush=True)
    codes = np.array([districts.index(r) for r in sub["region"]], dtype=np.int64)
    os.makedirs(build_pokec.OUT, exist_ok=True)
    build_pokec.save("pokec_t", sub.index.to_numpy(), prof, rel, codes, [d.split(", ")[1] for d in districts])


def check(name, summary):
    """Compare a build summary with the sizes the sources report (1% tolerance on edges, as the smoke test)."""
    n_exp, m_exp = EXPECTED.get(name, (None, None))
    notes = []
    if n_exp is not None and summary["n"] != n_exp:
        notes.append(f"n = {summary['n']:,}, source reports {n_exp:,}")
    if m_exp is not None and abs(summary["edges"] - m_exp) > 0.01 * m_exp:
        notes.append(f"edges = {summary['edges']:,}, source reports {m_exp:,}")
    for note in notes:
        print(f"  WARNING {name}: {note}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=ORDER, help="datasets to build (default: all of them)")
    ap.add_argument("--raw-dir", default=RAW_V2, help="where downloaded raw files live (default data/v2/raw)")
    ap.add_argument("--no-download", action="store_true", help="never download; fail if a raw file is missing")
    ap.add_argument("--force", action="store_true", help="rebuild datasets that are already built")
    args = ap.parse_args()

    names = [x for x in ORDER if args.only is None or x in args.only]
    todo = [x for x in names if args.force or not D.available(x)]
    for x in names:
        if x not in todo:
            print(f"{x}: already built ({D.built_path(x)}), skipped")
    for p in ("pokec_z", "pokec_n", "pokec_full"):
        if not D.available(p):
            print(f"note: {p} is missing; build it with python experiments/build_pokec.py "
                  f"--raw-dir \"{os.path.join(OLD_PROJECT, 'data')}\"")
    if not todo:
        return
    print("estimates (UNVERIFIED):")
    for x in todo:
        print(f"  {x}: {ESTIMATE[x]}")

    failed = {}
    for x in todo:
        t = time.time()
        print(f"{x}:", flush=True)
        try:
            if x == "pokec_t":
                build_pokec_t(args.raw_dir, not args.no_download)
            else:
                missing = fetch(x, args.raw_dir, not args.no_download)
                if missing:
                    raise FileNotFoundError(f"raw files missing: {missing}")
                s = D.build_all(args.raw_dir, only=[x], force=True)[x]
                print("  " + ", ".join(f"{k}={v}" for k, v in s.items() if k != "dataset"), flush=True)
                check(x, s)
        except Exception as e:                        # keep building the others; report at the end
            failed[x] = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {failed[x]}", flush=True)
        gc.collect()
        print(f"  {time.time() - t:.0f} s -> {D.built_path(x)}", flush=True)

    if failed:
        print("\nfailed:")
        for x, why in failed.items():
            print(f"  {x}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
