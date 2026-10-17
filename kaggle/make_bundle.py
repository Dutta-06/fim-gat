"""Build the Kaggle dataset bundle (kaggle/README.md): the committed repository plus full Pokec.

    python kaggle/make_bundle.py              # writes kaggle_bundle/fimgat-bundle.zip
    python kaggle/make_bundle.py --dry-run    # what would go in

The zip holds one folder, fimgat/:
- `git archive HEAD`: code, docs, tests, the datasets built in git, and the cloud's results;
- data/pokec/pokec_full.npz, built locally and not in git;
- BUNDLE_MANIFEST.json: the commit, the date and the hash of every file added outside git, which the notebook checks.

Only committed files go in: uncommitted changes to tracked files are refused (--allow-dirty overrides, and the
manifest says so), and untracked files are never included. Never included: LinkedIn (its node file holds real profile
handles; CLAUDE.md), raw downloads (data/v2/raw/) and the raw Pokec dumps. Keep the Kaggle dataset private.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRAS = ("data/pokec/pokec_full.npz",)
PREFIX = "fimgat/"
GIT_LOWMEM = ["-c", "core.bigFileThreshold=1m", "-c", "core.deltaBaseCacheLimit=16m", "-c", "pack.threads=1"]


def forbidden(rel):
    """Paths that must never leave the laptop: LinkedIn, raw downloads, the raw Pokec dumps."""
    r = rel.replace("\\", "/").lower()
    return "linkedin" in r or "/raw/" in f"/{r}" or "soc-pokec" in r


def git(*args):
    return subprocess.run(["git", *GIT_LOWMEM, *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "kaggle_bundle", "fimgat-bundle.zip"))
    ap.add_argument("--no-pokec-full", action="store_true", help="leave full Pokec out (G1 and the timing jobs on "
                                                                 "Instagram and com-Youtube still run)")
    ap.add_argument("--allow-dirty", action="store_true", help="bundle HEAD although tracked files have changes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    commit = git("rev-parse", "HEAD").strip()
    dirty = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
    untracked = [ln[3:] for ln in git("status", "--porcelain").splitlines()
                 if ln.startswith("??") and ln[3:].startswith(("fimgat/", "experiments/", "kaggle/", "tests/"))]
    tracked = [f for f in git("ls-files").splitlines() if f]
    bad = [f for f in tracked if forbidden(f)]
    if bad:
        sys.exit(f"refused: tracked files that must never be uploaded: {bad}")
    extras = [] if args.no_pokec_full else list(EXTRAS)
    missing = [e for e in extras if not os.path.exists(os.path.join(ROOT, e))]
    if missing:
        sys.exit(f"missing: {missing} (build it with experiments/build_pokec.py, or pass --no-pokec-full)")
    size = sum(os.path.getsize(os.path.join(ROOT, f)) for f in tracked if os.path.exists(os.path.join(ROOT, f)))
    size += sum(os.path.getsize(os.path.join(ROOT, e)) for e in extras)
    print(f"commit {commit[:12]}; {len(tracked)} tracked files + {len(extras)} extra ({size / 2**20:.0f} MB "
          f"uncompressed) -> {args.out}")
    if dirty:
        print("uncommitted changes to tracked files (they are NOT in the bundle):\n  " + "\n  ".join(dirty[:20]))
    if untracked:
        print("untracked code files (NOT in the bundle; commit them first if a job needs them):\n  "
              + "\n  ".join(untracked[:20]))
    if args.dry_run:
        return 0
    if dirty and not args.allow_dirty:
        sys.exit("refused: commit first (the bundle is HEAD, and the runs must match a commit); --allow-dirty overrides")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    subprocess.run(["git", *GIT_LOWMEM, "archive", "--format=zip", "-0", f"--prefix={PREFIX}", "-o", tmp, "HEAD"],
                   cwd=ROOT, check=True)
    manifest = {"runner": "fimgat-bundle", "commit": commit, "dirty": bool(dirty),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_files": len(tracked),
                "extras": {e: {"sha1": sha1(os.path.join(ROOT, e)), "bytes": os.path.getsize(os.path.join(ROOT, e))}
                           for e in extras},
                "excluded": ["data/v2/linkedin* (never bundled: personal data)", "data/v2/raw/ (raw downloads)",
                             "the raw Pokec dumps"]}
    with zipfile.ZipFile(tmp, "a", zipfile.ZIP_STORED) as zf:
        for e in extras:
            zf.write(os.path.join(ROOT, e), PREFIX + e)
        zf.writestr(PREFIX + "BUNDLE_MANIFEST.json", json.dumps(manifest, indent=1))
        names = zf.namelist()
    if any(forbidden(n[len(PREFIX):]) for n in names):
        os.remove(tmp)
        sys.exit("refused: the archive holds a forbidden path")
    os.replace(tmp, args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 2**20:.0f} MB, {len(names)} entries). Upload it as a "
          f"PRIVATE Kaggle dataset (kaggle/README.md, step 2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
