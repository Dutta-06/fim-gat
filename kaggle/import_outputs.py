"""Merge Kaggle outputs (outputs_<job>.zip, written by kaggle/run_job.py) into this repository.

    python kaggle/import_outputs.py Downloads/outputs_G1-train.zip                # report only: what would change
    python kaggle/import_outputs.py Downloads/outputs_G1-train.zip --apply        # write
    python kaggle/import_outputs.py Downloads/ --apply                            # every output in a folder

Only files under results/ and data/cache/ are written. Results CSVs are merged as row stores and JSON stores by key
(run_job.merge_bytes); other files are replaced by the incoming copy. Several outputs are applied oldest first.
Afterwards: python experiments/v2/summarize_v2.py --figures, then review the diff and commit.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_job as RJ                                                      # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="outputs zips, or folders holding them")
    ap.add_argument("--apply", action="store_true", help="write the changes (default: report only)")
    ap.add_argument("--root", default=RJ.ROOT, help=argparse.SUPPRESS)   # tests only
    args = ap.parse_args(argv)
    found = sorted((x for p in args.paths for x in RJ.find_outputs(p)), key=lambda x: x[0])
    if not found:
        sys.exit("no Kaggle outputs found (a zip from kaggle/run_job.py holds <job>/manifest.json)")
    seen, report = set(), []
    for updated, man, src, reader in found:
        tag = (man.get("job"), updated)
        if tag in seen:
            continue
        seen.add(tag)
        env = man.get("environment", {})
        bad = [c for c in man.get("commands", []) if not str(c.get("status", "")).startswith(RJ.DONE)]
        print(f"\n{man.get('job')} (updated {updated}; {src})\n  {'complete' if man.get('complete') else 'NOT COMPLETE'}"
              f"; {man.get('hours')} h; GPUs {env.get('gpus')}; torch {env.get('torch')}; code_tag {env.get('code_tag')}"
              f"; bundle commit {(env.get('bundle') or {}).get('commit', '?')[:12]}")
        for c in bad:
            print(f"  unfinished: {c.get('status')}: {c.get('cmd')}")
        n = 0
        for rel in man.get("files", []):
            if not RJ.safe_rel(rel):
                print(f"  refused: {rel} (outside results/ and data/cache/)")
                continue
            try:
                data = reader(rel)
            except (OSError, KeyError):
                print(f"  missing in the output: {rel}")
                continue
            RJ.merge_into(args.root, rel, data, report, apply=args.apply)
            n += 1
        print(f"  {n} files")
    if report:
        print("\nchanges" + ("" if args.apply else " (not written: pass --apply)") + ":")
        for action, rel, note in report:
            print(f"  {action:8s} {rel}" + (f"  ({note})" if note else ""))
    else:
        print("\nnothing to change: every file is already here")
    if args.apply:
        print("\nnext: python experiments/v2/summarize_v2.py --figures; then review the diff and commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
