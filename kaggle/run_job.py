"""Run one FIM-GAT v2 job in a Kaggle notebook (kaggle/README.md).

    python kaggle/run_job.py --list
    python kaggle/run_job.py --job G1-train --dry-run
    python kaggle/run_job.py --job G1-train --hours 11 --resume-from /kaggle/input --out /kaggle/working

A job is a list of phases. A phase is a list of lanes that run in parallel, one per GPU (CUDA_VISIBLE_DEVICES), and
a lane is a sequence of commands. With fewer GPUs than lanes, the lanes run one after another. Every command is one
of the repository's resumable scripts, so a job cut short continues where it stopped: give the next session this
session's output, and --resume-from copies it in before anything runs.

Output: <out>/outputs_<job>.zip holds, under the folder <job>/, every file below results/ and data/cache/ that differs
from the bundle (what this job wrote, and what it resumed from earlier outputs), the command logs (they are written to
results/v2/parts/kaggle-<job>/) and manifest.json; <out>/outputs_<job>.json is a copy of the manifest. Both are
refreshed after every command and every 30 minutes, so a session that dies keeps what it finished.
kaggle/import_outputs.py merges the zip into the repository on the laptop.

Rules carried over from docs/research_protocol.md and the cloud runs (design section 23):
- every pipeline script runs at the 8gb preset, the preset of every row the cloud recorded, so a graph's rows all
  use the same search sample A whatever machine made them;
- the timing tests (T3, T4) use one lane: a second process on the same CPUs would distort the wall-clock numbers;
- the CPU jobs (C1) never load the constructor: the learned start's rows run on the GPU, in G2;
- nothing here changes a pre-registered setting, and the scripts refuse to tune on a test graph.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = "fimgat-kaggle"
ROOTS = ("results/", "data/cache/")             # the only paths an output may hold
NEVER = ("results/v2/quick/",)                  # smoke-run output: never exported
TEMP = (".tmp", ".part")                        # files being written (atomic writers rename them when done)
V2 = "experiments/v2/"
P8 = ["--preset", "8gb"]
EXT = ["Degree", "Greedy", "Parity-Greedy", "Welfare-Greedy", "Maximin-Greedy", "FW-Disparity", "AGE", "Prior", "BSM",
       "MultiFW-MaxMin", "RFIM"]
RECORDER = {"t1.csv", "t2.csv", "t3.csv", "t4.csv", "t1_pareto.csv", "t1_certsens.csv", "tuning_runs.csv"}
KEY = ["dataset", "attribute", "graph", "k", "rho", "method", "variant", "start", "proposer", "deployment",
       "train_seed", "delta", "phi0", "cert_R", "R", "budget_s", "tuning_step"]   # summarize_v2.KEY + tuning step
FILE_KEYS = {"t1_bounds.csv": ["dataset", "attribute", "graph", "k", "rho"]}
DONE = ("ok", "done", "not needed")             # command statuses that count as finished


# ------------------------------------------------------------------ the jobs

def pipe(script, *args):
    """A pipeline script's command at the 8gb preset."""
    return [V2 + script] + [str(a) for a in args] + P8


def t1(job, datasets, attribute=None, split=False):
    """T1 commands for one job folder. split: the cloud's order (the method configuration with the external baselines,
    then the ablations), so the headline rows come first; otherwise one command that runs every missing row."""
    base = ["run_t1_t2.py", "--test", "t1", "--datasets", *datasets] + (["--attributes", attribute] if attribute
                                                                        else []) + ["--job", job]
    if not split:
        return [pipe(*base)]
    return [pipe(*base, "--configs", "method", "--methods", *EXT),
            pipe(*base, "--configs", "ablation", "--methods", "LocalSearch", "Simple-v2")]


def case(dataset, attribute, job, when=None):
    """A pre-registered case study (interpret_v2.CASES), with the reference store of its T1 job."""
    c = {"cmd": pipe("interpret_v2.py", "--case", dataset, attribute, 50, 0.99,
                     "--ref-file", f"results/v2/parts/{job}/references.json"),
         "once": f"case_{dataset}_{attribute}" + ("_learned" if when else "")}
    return dict(c, when=when) if when else c


FULL = ("region", "gender", "age")
C1_JOBS = [f"C1-pokec-full-{a}" for a in FULL]
OTHER = [("t1-small", ["rice_facebook", "twitter_political"], None), ("t1-pokec-n-region", ["pokec_n"], "region"),
         ("t1-pokec-n-gender", ["pokec_n"], "gender"), ("t1-pokec-n-age", ["pokec_n"], "age"),
         ("t1-epinions", ["epinions"], None), ("t1-twitter-ego", ["twitter_ego"], None),
         ("t1-instagram", ["instagram"], None), ("t1-youtube", ["snap_youtube"], None)]
CASES_OTHER = [("rice_facebook", "college", "t1-small"), ("rice_facebook", "age", "t1-small"),
               ("rice_facebook", "major", "t1-small"), ("twitter_political", "leaning", "t1-small"),
               ("pokec_n", "region", "t1-pokec-n-region"), ("pokec_n", "gender", "t1-pokec-n-gender"),
               ("pokec_n", "age", "t1-pokec-n-age"), ("instagram", "gender", "t1-instagram")]
LARGE = ["instagram", "snap_youtube", "pokec_full"]


def c1(attr):
    return {"gpu": False, "needs": [],
            "about": f"T1 on full Pokec under {attr}: every row except the learned start's (no constructor here)",
            "phases": [{"name": f"T1, full Pokec / {attr}: the method with the external baselines, then the "
                                f"ablations", "lanes": [{"gpu": 0, "cmds": t1(f"t1-pokec-full-{attr}", ["pokec_full"],
                                                                               attr, split=True)}]}]}


JOBS = {
    "G1-train": {
        "gpu": True, "needs": [],
        "about": "the GAT constructor (5 seeds and the no-adversary ablation), tuning's start step, the learned proposer",
        "phases": [
            {"name": "smoke: a few steps of both trainers, so a broken environment shows in minutes",
             "lanes": [{"gpu": 0, "cmds": [
                 [V2 + "train_constructor.py", "--quick"] + P8,
                 [V2 + "train_proposer.py", "--ignore-gate", "--seeds", "0", "--ks", "25", "--attributes", "region",
                  "--val-attributes", "region", "--states-per-query", "1", "--val-states-per-query", "1",
                  "--epochs", "1", "--constructor-pattern", "results/v2/quick/models/constructor_s{seed}.pt"] + P8]}]},
            {"name": "constructor: seeds 0-2 on GPU 0; seeds 3-4 and the no-adversary ablation (seed 0) on GPU 1",
             "lanes": [{"gpu": 0, "cmds": [[V2 + "train_constructor.py", "--seeds", "0", "1", "2", "--noadv-seeds",
                                            "--log", "results/v2/train_log_gpu0.csv"] + P8]},
                       {"gpu": 1, "cmds": [[V2 + "train_constructor.py", "--seeds", "3", "4", "--noadv-seeds", "0",
                                            "--log", "results/v2/train_log_gpu1.csv"] + P8]}]},
            {"name": "tuning's start step on validation (GPU 0) and the learned proposer (GPU 1; W3 said train)",
             "lanes": [{"gpu": 0, "cmds": [pipe("tune_v2.py", "--steps", "start")]},
                       {"gpu": 1, "cmds": [[V2 + "train_proposer.py"] + P8]}]}]},
    **{f"C1-pokec-full-{a}": c1(a) for a in FULL},
    "G2-learned": {
        "gpu": True, "needs": ["G1-train"] + C1_JOBS,
        "about": "every T1 row still missing (the learned start's) on every test graph, T2 if the start changed, and "
                 "the full-Pokec case studies",
        "phases": [{"name": "missing T1 rows: full Pokec and its case studies (GPU 0); the other graphs, then T2 "
                            "(GPU 1)",
                    "lanes": [{"gpu": 0, "keep_going": True,
                               "cmds": [c for a in FULL for c in t1(f"t1-pokec-full-{a}", ["pokec_full"], a)
                                        + [case("pokec_full", a, f"t1-pokec-full-{a}")]]},
                              {"gpu": 1, "keep_going": True,
                               "cmds": [c for job, ds, a in OTHER for c in t1(job, ds, a)]
                               + [pipe("run_t1_t2.py", "--test", "t2", "--t2-seeds", 1, 2, 3, 4, 5, 6, "--job", "t2")]
                               + [case(d, a, j, when="start-learned") for d, a, j in CASES_OTHER]}]}]},
    "G3-t3": {
        "gpu": True, "timing": True, "needs": ["G1-train"] + C1_JOBS,
        "about": "T3: the learned proposer against the simple hybrid at equal wall-clock budgets (large tier)",
        "phases": [{"name": "T3 on Instagram, com-Youtube and full Pokec (one lane: timing)",
                    "lanes": [{"gpu": 0, "cmds": [
                        {"call": "merge_refs", "dst": "results/v2/parts/t3/references.json"},
                        pipe("run_t3.py", "--datasets", *LARGE, "--out", "results/v2/parts/t3/t3.csv",
                             "--recall-out", "results/v2/parts/t3/t3_recall.csv",
                             "--ref-file", "results/v2/parts/t3/references.json")]}]}]},
    "G4-t4": {
        "gpu": True, "timing": True, "needs": ["G1-train"],
        "about": "T4: cached and per-query runtime on the large tier (references computed afresh: they are timed)",
        "phases": [{"name": "T4 on Instagram, com-Youtube and full Pokec (one lane: timing)",
                    "lanes": [{"gpu": 0, "cmds": [
                        pipe("run_t4.py", "--datasets", *LARGE, "--out", "results/v2/parts/t4/t4.csv",
                             "--runtime-out", "results/v2/parts/t4/t4_runtime.csv",
                             "--ref-file", "results/v2/parts/t4/references.json")]}]}]},
}
ORDER = ["1. at once, in parallel: G1-train (GPU T4 x2) and C1-pokec-full-region, -gender, -age (three CPU sessions)",
         "2. when G1-train and the C1 jobs are done: G2-learned (GPU T4 x2)",
         "3. then, one after the other: G3-t3 and G4-t4 (GPU; one GPU used, because they measure time)"]


def commands(job):
    for ph in JOBS[job]["phases"]:
        for lane in ph["lanes"]:
            yield from lane["cmds"]


# ------------------------------------------------------------------ files

def digest(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_rel(rel):
    """A path an output may hold: relative, inside results/ or data/cache/, not a smoke-run or temp file."""
    rel = rel.replace("\\", "/")
    parts = rel.split("/")
    return (bool(rel) and not rel.startswith("/") and ":" not in parts[0] and ".." not in parts
            and any(rel.startswith(r) for r in ROOTS) and not any(rel.startswith(n) for n in NEVER)
            and not rel.endswith(TEMP))


def snapshot(root=ROOT):
    """{relative path: sha1} of every exportable file under the output roots."""
    out = {}
    for top in ROOTS:
        for dirpath, _, names in os.walk(os.path.join(root, top)):
            for n in names:
                rel = os.path.relpath(os.path.join(dirpath, n), root).replace(os.sep, "/")
                if safe_rel(rel):
                    out[rel] = digest(os.path.join(root, rel))
    return out


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _keys(df, name):
    if name in RECORDER:
        cols = [c for c in KEY if c in df]
    else:
        cols = [c for c in FILE_KEYS.get(name, []) if c in df] or list(df.columns)
    return [tuple(r) for r in df[cols].astype(str).itertuples(index=False)], cols


def merge_bytes(name, local, incoming):
    """How an incoming copy of a file meets the local one: (action, bytes to write or None, note).

    A results CSV is a row store: if the incoming file holds every local row's key, it is newer and replaces the local
    file; if the reverse, the local file is kept; otherwise the two are united (local rows first; an incoming row whose
    key is already present is dropped). A JSON object is handled the same way by key (local values win a clash when the
    two are united). Any other file is replaced by the incoming copy."""
    if local is None:
        return "new", incoming, ""
    if local == incoming:
        return "same", None, ""
    if name.endswith(".csv"):
        import pandas as pd                               # pandas is on Kaggle and on the laptop
        try:
            a = pd.read_csv(io.BytesIO(local), float_precision="round_trip", low_memory=False)
            b = pd.read_csv(io.BytesIO(incoming), float_precision="round_trip", low_memory=False)
        except Exception as e:                            # an unreadable CSV: the incoming copy wins, and it is said
            return "replaced", incoming, f"unreadable CSV ({e}); took the incoming copy"
        ka, cols = _keys(a, name)
        kb, _ = _keys(b, name)
        sa, sb = set(ka), set(kb)
        if sa <= sb:
            return "newer", incoming, f"{len(sb - sa)} rows added"
        if sb <= sa:
            return "older", None, "the local file already holds every incoming row"
        keep = [k not in sa for k in kb]
        merged = pd.concat([a, b[keep]], ignore_index=True, sort=False)
        buf = io.StringIO()
        merged.to_csv(buf, index=False)
        return "united", buf.getvalue().encode("utf-8"), (f"{sum(keep)} incoming rows added, {len(kb) - sum(keep)} "
                                                         f"already present (key {cols})")
    if name.endswith(".json"):
        try:
            a, b = json.loads(local), json.loads(incoming)
        except ValueError:
            return "replaced", incoming, "unreadable JSON; took the incoming copy"
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) <= set(b):
                return "newer", incoming, ""
            if set(b) <= set(a) and all(a[k] == b[k] for k in b):
                return "older", None, "the local file already holds every incoming entry"
            merged = dict(b, **a)                         # local entries win; incoming-only entries are added
            clash = sorted(k for k in set(a) & set(b) if a[k] != b[k])
            return "united", json.dumps(merged, indent=1).encode("utf-8"), (
                f"{len(set(b) - set(a))} entries added" + (f"; {len(clash)} differ, kept the local ones: {clash[:5]}"
                                                           if clash else ""))
    return "replaced", incoming, ""


def merge_into(root, rel, data, report=None, apply=True):
    path = os.path.join(root, rel)
    local = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            local = f.read()
    action, out, note = merge_bytes(os.path.basename(rel), local, data)
    if apply and out is not None:
        write_bytes(path, out)
    if report is not None and action != "same":
        report.append((action, rel, note))
    return action


def find_outputs(where):
    """Earlier outputs in `where` (a folder searched recursively, or one zip): [(updated, manifest, source, reader)]
    for every outputs zip and every unpacked output folder (Kaggle unpacks a zip uploaded to a dataset), oldest
    first. reader(rel) returns a file's bytes."""
    found = []
    single = os.path.isfile(where)
    zips = [where] if single else sorted(glob.glob(os.path.join(where, "**", "*.zip"), recursive=True))
    for z in zips:
        try:
            zf = zipfile.ZipFile(z)
            heads = [n for n in zf.namelist() if n.count("/") == 1 and n.endswith("/manifest.json")]
        except (OSError, zipfile.BadZipFile):
            continue
        for h in heads:
            try:
                man = json.loads(zf.read(h))
            except ValueError:
                continue
            if isinstance(man, dict) and man.get("runner") == RUNNER:
                pre = h[: -len("manifest.json")]
                found.append((man.get("updated", ""), man, z, lambda rel, zf=zf, pre=pre: zf.read(pre + rel)))
    for m in ([] if single else sorted(glob.glob(os.path.join(where, "**", "manifest.json"), recursive=True))):
        man = read_json(m)
        if isinstance(man, dict) and man.get("runner") == RUNNER:
            def reader(rel, base=os.path.dirname(m)):
                with open(os.path.join(base, rel), "rb") as f:
                    return f.read()
            found.append((man.get("updated", ""), man, os.path.dirname(m), reader))
    return sorted(found, key=lambda x: x[0])


def resume(where, job, needs, root=ROOT):
    """Merge earlier outputs of this job and of the jobs it needs into the repository, oldest first."""
    wanted = set(needs) | {job}
    report, seen = [], set()
    for updated, man, src, reader in find_outputs(where):
        name = man.get("job")
        if name not in wanted:
            print(f"  ignored: output of {name} ({src}); job {job} does not use it", flush=True)
            continue
        if (name, updated) in seen:                       # the same output twice (a zip and its unpacked copy)
            continue
        seen.add((name, updated))
        n = 0
        for rel in man.get("files", []):
            if not safe_rel(rel):
                print(f"  refused: {rel} (outside results/ and data/cache/)", flush=True)
                continue
            try:
                data = reader(rel)
            except (OSError, KeyError):
                print(f"  missing in {src}: {rel}", flush=True)
                continue
            merge_into(root, rel, data, report)
            n += 1
        print(f"  resumed {n} files from {name} (updated {updated}; {src})", flush=True)
    for action, rel, note in report:
        if action in ("united", "older", "replaced"):
            print(f"    {action}: {rel}" + (f" ({note})" if note else ""), flush=True)
    missing = sorted(set(needs) - {n for n, _ in seen})
    if missing:
        print(f"  NOTE: no output found for {missing}; attach them if this job needs their results", flush=True)
    return report


def merge_refs(dst, root=ROOT):
    """Add to the reference store `dst` every entry of the other stores under results/v2. Keys are self-describing
    (graph, attribute, n, m, R, roots, runs, seeds, n_sims), so an entry is valid wherever its key matches."""
    path = os.path.join(root, dst)
    store = read_json(path) or {}
    added = 0
    for f in sorted(glob.glob(os.path.join(root, "results", "v2", "**", "references.json"), recursive=True)):
        if os.path.abspath(f) == os.path.abspath(path):
            continue
        for k, v in (read_json(f) or {}).items():
            if k not in store:
                store[k] = v
                added += 1
    write_bytes(path, json.dumps(store, indent=1).encode("utf-8"))
    return added


class Exporter:
    """Writes <out>/outputs_<job>.zip and a copy of its manifest: every exportable file that differs from the bundle."""

    def __init__(self, job, out, base, manifest, root=ROOT):
        self.job, self.out, self.base, self.manifest, self.root = job, out, base, manifest, root
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            now = snapshot(self.root)
            files = sorted(r for r, h in now.items() if self.base.get(r) != h)
            man = dict(self.manifest, files=files, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
            os.makedirs(self.out, exist_ok=True)
            zpath = os.path.join(self.out, f"outputs_{self.job}.zip")
            tmp = zpath + ".tmp"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for rel in files:
                    zf.write(os.path.join(self.root, rel), f"{self.job}/{rel}")
                zf.writestr(f"{self.job}/manifest.json", json.dumps(man, indent=1))
            os.replace(tmp, zpath)
            write_bytes(os.path.join(self.out, f"outputs_{self.job}.json"), json.dumps(man, indent=1).encode("utf-8"))
            return files


# ------------------------------------------------------------------ running

def gpus():
    """The GPUs nvidia-smi lists. The runner never imports torch: a CUDA context in this process would hold GPU
    memory for the whole session, under the timing jobs too."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else []


def env_report(root=ROOT):
    from importlib import metadata
    info = {"python": sys.version.split()[0], "gpus": gpus()}
    for m in ("torch", "torch_geometric", "numba", "numpy", "pandas", "scipy", "networkx", "scikit-learn"):
        try:
            info[m] = metadata.version(m)
        except metadata.PackageNotFoundError:
            info[m] = "missing"
    info["cpus"] = os.cpu_count()
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    bm = read_json(os.path.join(root, "BUNDLE_MANIFEST.json"))
    if bm:
        info["bundle"] = {k: bm.get(k) for k in ("commit", "dirty", "created")}
    return info


def code_tag(root=ROOT):
    """common_v2.code_tag(), in a subprocess (it imports torch and the package)."""
    try:
        r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'experiments/v2'); "
                            "from common_v2 import code_tag; print(code_tag())"], cwd=root, capture_output=True,
                           text=True, timeout=300)
        return r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else "unavailable"
    except Exception:
        return "unavailable"


def tuning_start(root=ROOT):
    return (read_json(os.path.join(root, "results", "v2", "tuning.json")) or {}).get("start")


def describe(c):
    if isinstance(c, dict) and "call" in c:
        return f"[{c['call']} {c['dst']}]"
    cmd = c["cmd"] if isinstance(c, dict) else c
    return "python " + " ".join(cmd) + ("   (only if the tuned start is learned)"
                                        if isinstance(c, dict) and c.get("when") else "")


def run_lane(lane, gpu, ctx):
    """Run one lane's commands in order. A failed command ends the lane unless the lane is keep_going (independent
    commands). Near the session's end no command starts, and a running one is stopped in time to export."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = os.path.join(ctx["logdir"], f"lane{lane['_index']}.log")
    for c in lane["cmds"]:
        left = ctx["deadline"] - time.time()
        rec = {"cmd": describe(c), "gpu": gpu}
        if isinstance(c, dict) and "call" in c:
            ctx["record"](dict(rec, status=f"ok ({merge_refs(c['dst'], ctx['root'])} reference entries added)"))
            continue
        cmd = c["cmd"] if isinstance(c, dict) else c
        once = c.get("once") if isinstance(c, dict) else None
        marker = os.path.join(ctx["markers"], once) if once else None
        if marker and os.path.exists(marker):
            ctx["record"](dict(rec, status="done in an earlier session"))
            continue
        if isinstance(c, dict) and c.get("when") == "start-learned" and tuning_start(ctx["root"]) != "learned":
            ctx["record"](dict(rec, status="not needed: the tuned start is not the constructor's"))
            continue
        if left < 20 * 60:
            ctx["record"](dict(rec, status="not started: session time nearly over (resumable)"))
            continue
        t0 = time.time()
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} GPU {gpu}: {' '.join(cmd)}\n")
            f.flush()
            p = subprocess.Popen([sys.executable, "-u", "-W", "ignore"] + cmd, cwd=ctx["root"], env=env, stdout=f,
                                 stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=max(60.0, left - 12 * 60))
                status = "ok" if rc == 0 else f"exit {rc}"
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                status = "stopped: session time (resumable: give this output to the next session)"
        if status == "ok" and marker:
            os.makedirs(os.path.dirname(marker), exist_ok=True)
            with open(marker, "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        mins = (time.time() - t0) / 60
        ctx["record"](dict(rec, status=status, minutes=round(mins, 1)))
        print(f"  [{time.strftime('%H:%M:%S')}] lane {lane['_index']} (GPU {gpu}): {os.path.basename(cmd[0])} -> "
              f"{status} ({mins:.1f} min)", flush=True)
        ctx["export"]()
        if status.startswith("exit"):
            with open(log, encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-20:]
            print("    last lines of the log:\n    " + "    ".join(tail), flush=True)
            if not lane.get("keep_going"):
                break
        if status.startswith("stopped"):
            break


def heartbeat(stop, every, export, logdir):
    while not stop.wait(every):
        try:
            import psutil
            mem = psutil.virtual_memory()
            m = f"RAM {mem.used / 2**30:.1f} of {mem.total / 2**30:.1f} GB used"
        except Exception:
            m = "RAM n/a"
        try:
            n = len(export())
            print(f"  [{time.strftime('%H:%M:%S')}] running; {m}; outputs refreshed ({n} files); logs in {logdir}",
                  flush=True)
        except Exception as e:                            # a failed refresh must not end the job
            print(f"  [{time.strftime('%H:%M:%S')}] running; {m}; output refresh failed: {e}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", choices=list(JOBS))
    ap.add_argument("--list", action="store_true", help="list the jobs and the order to run them")
    ap.add_argument("--hours", type=float, default=11.0,
                    help="stop starting work after this many hours (a Kaggle session ends at 12 h)")
    ap.add_argument("--resume-from", default=None, help="folder searched for earlier outputs (/kaggle/input)")
    ap.add_argument("--out", default=os.path.join(ROOT, "kaggle_outputs"), help="where outputs go (/kaggle/working)")
    ap.add_argument("--smoke-tests", action="store_true", help="run the fast test file first (about a minute)")
    ap.add_argument("--serial", action="store_true", help="run a phase's lanes one after another (less memory)")
    ap.add_argument("--allow-cpu", action="store_true", help="run a GPU job without a GPU (development only)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, then stop")
    args = ap.parse_args(argv)
    if args.list or not args.job:
        for name, j in JOBS.items():
            print(f"{name:22s} {'GPU' if j['gpu'] else 'CPU'}{' (timing)' if j.get('timing') else ''}: {j['about']}")
        print("\norder:\n  " + "\n  ".join(ORDER))
        return 0
    job = JOBS[args.job]
    t_start = time.time()
    deadline = t_start + args.hours * 3600
    info = env_report()
    n_gpu = len(info.get("gpus") or [])
    print(f"job {args.job}: {job['about']}\nneeds the outputs of: {', '.join(job['needs']) or 'nothing'}\n"
          f"environment: {json.dumps(info)}", flush=True)
    parallel = (lambda lanes: n_gpu >= len(lanes) > 1 and not args.serial)
    for i, ph in enumerate(job["phases"], 1):
        par = parallel(ph["lanes"])
        print(f"phase {i}: {ph['name']}" + (" [lanes in parallel]" if par else ""), flush=True)
        for j, lane in enumerate(ph["lanes"]):
            for c in lane["cmds"]:
                print(f"    lane {j} (GPU {lane['gpu'] if par else (0 if n_gpu else 'none')}): {describe(c)}")
    if args.dry_run:
        return 0
    if job["gpu"] and not n_gpu and not args.allow_cpu:
        sys.exit("this job needs a GPU: set the notebook's accelerator to GPU T4 x2 (Settings > Accelerator)")
    if not job["gpu"] and n_gpu:
        print("note: a CPU job on a GPU session; it runs, but it spends GPU quota for little gain", flush=True)
    base = snapshot()                                     # the bundle as shipped: outputs are what differs from it
    if args.resume_from:
        print(f"\nresuming from earlier outputs under {args.resume_from}", flush=True)
        resume(args.resume_from, args.job, job["needs"])
    part = os.path.join(ROOT, "results", "v2", "parts", f"kaggle-{args.job}")
    logdir = os.path.join(part, "logs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    manifest = {"runner": RUNNER, "job": args.job, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hours_limit": args.hours, "environment": dict(info, code_tag=code_tag()), "commands": []}
    export = Exporter(args.job, args.out, base, manifest)
    lock = threading.Lock()

    def record(r):
        with lock:
            manifest["commands"].append(r)

    if args.smoke_tests:
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                              "tests/test_smoke_foundation.py"], cwd=ROOT)
        manifest["smoke_tests"] = "passed" if rc == 0 else f"FAILED (exit {rc})"
        print(f"smoke tests: {manifest['smoke_tests']}", flush=True)
        if rc != 0:
            export()
            sys.exit("the environment failed the smoke tests; nothing was run (see the test output above)")
    ctx = {"root": ROOT, "deadline": deadline, "logdir": logdir, "markers": os.path.join(part, "done"),
           "record": record, "export": export}
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, 3600 if job.get("timing") else 1800, export, logdir),
                     daemon=True).start()
    complete = True
    try:
        for i, ph in enumerate(job["phases"], 1):
            print(f"\n=== phase {i}: {ph['name']} ===", flush=True)
            n_before = len(manifest["commands"])
            lanes = [dict(lane, _index=j) for j, lane in enumerate(ph["lanes"])]
            if parallel(lanes):
                threads = [threading.Thread(target=run_lane, args=(ln, ln["gpu"], ctx)) for ln in lanes]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            else:
                for ln in lanes:
                    run_lane(ln, 0 if n_gpu else None, ctx)
            ran = manifest["commands"][n_before:]
            n_cmds = sum(len(ln["cmds"]) for ln in lanes)
            if len(ran) < n_cmds or any(not r["status"].startswith(DONE) for r in ran):
                complete = False
                print("phase not complete (a command failed or ran out of time); later phases need it, so the job "
                      "stops here", flush=True)
                break
    finally:
        stop.set()
        manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["hours"] = round((time.time() - t_start) / 3600, 2)
        manifest["complete"] = complete
        files = export()
    print(f"\n{args.job}: {manifest['hours']} h, {len(files)} files in {args.out}/outputs_{args.job}.zip", flush=True)
    for r in manifest["commands"]:
        print(f"  {r['status'][:58]:58s} {r['cmd'][:160]}", flush=True)
    if not complete:
        print("NOT FINISHED: run this job again with this output attached; it resumes where it stopped.", flush=True)
        return 1
    print("FINISHED: download the zip, and attach it to the sessions of the jobs that need it.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
