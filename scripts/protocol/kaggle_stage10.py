#!/usr/bin/env python3
"""Stage 10 on Kaggle through the Kaggle API: kaggle/README.md steps 2-5 without the web UI.

Needs the Kaggle CLI (py -3 -m pip install --user kaggle) and an API token from
https://www.kaggle.com/settings/api saved to ~/.kaggle/access_token (or kaggle.json).

  py -3 scripts/protocol/kaggle_stage10.py check            # token, username, weekly GPU quota
  py -3 scripts/protocol/kaggle_stage10.py stage            # build/kaggle/upload/* + build/kaggle/kernel/
  py -3 scripts/protocol/kaggle_stage10.py upload [--only code,cache,data]   # private datasets, timed
  py -3 scripts/protocol/kaggle_stage10.py push             # GPU kernel (internet off), runs on Kaggle
  py -3 scripts/protocol/kaggle_stage10.py wait             # poll the kernel until it finishes
  py -3 scripts/protocol/kaggle_stage10.py fetch            # download -> results/stage10/kaggle_bench, compare, cost sheet
  py -3 scripts/protocol/kaggle_stage10.py all              # every step in order

Every dataset is created private. The 23 GB dataset is uploaded region by region as
uncompressed tar archives (JPG/TIF do not compress); the kernel finds its inputs whether or
not Kaggle extracts them (kaggle/stage10_kernel.py). Upload wall time is the data-transfer
measurement and goes to reports/stage10_kaggle_transfer.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "kaggle"
UPLOAD = BUILD / "upload"
KERNEL_DIR = BUILD / "kernel"
DATA_SRC = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset"
RESULT = ROOT / "results" / "stage10" / "kaggle_bench"
QUOTA_JSON = ROOT / "reports" / "stage10_kaggle_quota.json"
TRANSFER_JSON = ROOT / "reports" / "stage10_kaggle_transfer.json"
SLUGS = {"code": "cmbl-code", "cache": "cmbl-cache", "data": "uav-visloc"}
TITLES = {"code": "CMBL code bundle", "cache": "CMBL embedding and weight cache",
          "data": "UAV-VisLoc private copy"}
KERNEL_SLUG = "cmbl-stage10-benchmark"
ACCELERATOR = "NvidiaTeslaT4"


def kaggle_exe() -> list[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    cand = Path(os.environ.get("APPDATA", "")) / "Python" / f"Python{sys.version_info[0]}{sys.version_info[1]}" / "Scripts" / "kaggle.exe"
    if cand.is_file():
        return [str(cand)]
    raise SystemExit("Kaggle CLI not found: py -3 -m pip install --user kaggle")


def kaggle(*args: str, capture: bool = False, check: bool = True) -> str:
    cmd = kaggle_exe() + list(args)
    print("$ kaggle " + " ".join(args), flush=True)
    if capture:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        if check and p.returncode:
            raise SystemExit(f"kaggle {' '.join(args)} failed ({p.returncode}):\n{out}")
        return out
    rc = subprocess.call(cmd)
    if check and rc:
        raise SystemExit(f"kaggle {' '.join(args)} failed ({rc})")
    return ""


def api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("Kaggle CLI not installed: py -3 -m pip install --user kaggle")
    a = KaggleApi()
    try:
        a.authenticate()
    except (SystemExit, Exception) as e:  # noqa: BLE001
        raise SystemExit(
            "No Kaggle API token. Create one at https://www.kaggle.com/settings/api and save it to "
            f"{Path.home() / '.kaggle' / 'access_token'} (or kaggle.json). ({e})")
    return a


def username() -> str:
    a = api()
    user = a.config_values.get("username") or a.get_config_value("username")
    if not user:
        raise SystemExit("authenticated, but the Kaggle username is unknown; pass --user")
    return user


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cmd_check(args) -> None:
    user = args.user or username()
    q = api().quota_view()
    gpu = q.gpu_quota
    rec = {"checked_utc": now(), "username": user,
           "gpu_total_h": round(gpu.total_time_allowed.total_seconds() / 3600, 2) if gpu else None,
           "gpu_used_h": round(gpu.time_used.total_seconds() / 3600, 2) if gpu else None,
           "quota_refresh_utc": q.quota_refresh_time.isoformat() if q.quota_refresh_time else None}
    QUOTA_JSON.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))


def write_meta(folder: Path, user: str, key: str) -> None:
    meta = {"title": TITLES[key], "id": f"{user}/{SLUGS[key]}", "licenses": [{"name": "unknown"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def dir_link(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    try:
        os.symlink(target, link_path, target_is_directory=True)
    except OSError:
        subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link_path), str(target)], stdout=subprocess.DEVNULL)


def cmd_stage(args) -> None:
    user = args.user or username()
    bm = json.loads((BUILD / "bundle_manifest.json").read_text(encoding="utf-8"))
    for key in SLUGS:
        d = UPLOAD / SLUGS[key]
        d.mkdir(parents=True, exist_ok=True)
        write_meta(d, user, key)
    # code/cache: the zips themselves (one file each; hard link, or copy across drives)
    for key, name in (("code", bm["code_zip"]["file"]), ("cache", bm["cache_zip"]["file"])):
        d = UPLOAD / SLUGS[key]
        for old in d.glob("*.zip"):
            if old.name != name:
                old.unlink()
        if not (d / name).exists():
            try:
                os.link(BUILD / name, d / name)
            except OSError:
                shutil.copy2(BUILD / name, d / name)
    # data: junctions to the 11 region folders + the two top-level files (nothing copied)
    d = UPLOAD / SLUGS["data"]
    for r in sorted(p for p in DATA_SRC.iterdir() if p.is_dir()):
        dir_link(r, d / r.name)
    for f in DATA_SRC.iterdir():
        if f.is_file() and not (d / f.name).exists():
            shutil.copy2(f, d / f.name)
    # kernel
    km = stage_kernel(KERNEL_DIR, user, KERNEL_SLUG, args)
    print(json.dumps({"staged": str(UPLOAD), "kernel": km, "bundle_commit": bm["commit"]}, indent=2))


def stage_kernel(folder: Path, user: str, slug: str, args, job: list[str] | None = None) -> dict:
    """Kernel folder: kaggle/stage10_kernel.py (with KERNEL_ARGS set to job, if given) + metadata."""
    folder.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kaggle" / "stage10_kernel.py").read_text(encoding="utf-8")
    if job is not None:
        line = "KERNEL_ARGS: list[str] = []"
        if src.count(line) != 1:
            raise SystemExit(f"kaggle/stage10_kernel.py must contain exactly one '{line}'")
        src = src.replace(line, f"KERNEL_ARGS: list[str] = {json.dumps(job)}")
    (folder / "stage10_kernel.py").write_text(src, encoding="utf-8")
    km = {"id": f"{user}/{slug}", "title": slug, "code_file": "stage10_kernel.py",
          "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
          "enable_tpu": False, "enable_internet": bool(args.internet),
          "machine_shape": args.accelerator,  # enable_gpu alone is deprecated; this pins the hardware
          "dataset_sources": [f"{user}/{s}" for s in SLUGS.values()], "competition_sources": [],
          "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(km, indent=2), encoding="utf-8")
    return km


def dataset_exists(user: str, slug: str) -> bool:
    # "datasets status" answers 403 (not 404) for a missing dataset, so check our own list instead.
    out = kaggle("datasets", "list", "--mine", "-s", slug, "--csv", "--page-size", "100", capture=True)
    return f"{user}/{slug}".lower() in {line.split(",")[0].strip().lower() for line in out.splitlines()}


def load_transfer() -> list:
    return json.loads(TRANSFER_JSON.read_text(encoding="utf-8")) if TRANSFER_JSON.exists() else []


def cmd_upload(args) -> None:
    user = args.user or username()
    keys = args.only.split(",") if args.only else list(SLUGS)
    log = load_transfer()
    for key in keys:
        d, slug = UPLOAD / SLUGS[key], SLUGS[key]
        if not (d / "dataset-metadata.json").exists():
            raise SystemExit(f"{d} not staged; run: stage")
        meta_id = json.loads((d / "dataset-metadata.json").read_text(encoding="utf-8"))["id"]
        if meta_id != f"{user}/{slug}":
            raise SystemExit(f"{d} was staged as {meta_id}, not {user}/{slug}; rerun: stage")
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d, followlinks=True) for f in fs)
        exists = dataset_exists(user, slug)
        common = ["-p", str(d), "-r", "tar", "-t"]
        t0 = time.perf_counter()
        if exists:
            kaggle("datasets", "version", *common, "-m", f"bundle {args.note or now()}")
        else:
            kaggle("datasets", "create", *common)  # private by default
        up_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        status = wait_dataset(user, slug, args.timeout_min)
        ready_s = time.perf_counter() - t0
        rec = {"utc": now(), "dataset": f"{user}/{slug}", "mode": "version" if exists else "create",
               "bytes": size, "gb": round(size / 1e9, 3), "upload_wall_s": round(up_s, 1),
               "mb_per_s": round(size / 1e6 / up_s, 2) if up_s else None,
               "processing_wall_s": round(ready_s, 1), "final_status": status}
        log.append(rec)
        TRANSFER_JSON.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(json.dumps(rec, indent=2), flush=True)


def wait_dataset(user: str, slug: str, timeout_min: float) -> str:
    t_end = time.time() + timeout_min * 60
    while True:
        out = kaggle("datasets", "status", f"{user}/{slug}", capture=True, check=False).strip()
        low = out.lower()
        transient = "client error" in low or "forbidden" in low  # 403 until the new dataset is visible
        if "ready" in low or (not transient and ("error" in low or "fail" in low)) or time.time() > t_end:
            return out
        time.sleep(30)


def push_kernel(folder: Path, args) -> None:
    if not (folder / "kernel-metadata.json").exists():
        raise SystemExit(f"kernel not staged in {folder}")
    km = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    user = km["id"].split("/")[0]
    missing = [s for s in km["dataset_sources"] if not dataset_exists(user, s.split("/")[1])]
    if missing:  # Kaggle would start the kernel without them and it would stop at discovery
        raise SystemExit(f"input datasets not on Kaggle yet: {missing}; run: upload")
    extra = ["-t", str(args.kernel_timeout_s)] if args.kernel_timeout_s else []
    kaggle("kernels", "push", "-p", str(folder), "--accelerator", args.accelerator, *extra)


def cmd_push(args) -> None:
    push_kernel(KERNEL_DIR, args)


def kernel_id(folder: Path = KERNEL_DIR) -> str:
    return json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))["id"]


def kernel_status(kid: str) -> str:
    """'running', 'queued', 'complete', 'error', 'cancel...' or 'unknown' (never pushed, or an API error)."""
    out = kaggle("kernels", "status", kid, capture=True, check=False).strip().lower()
    if "has status" not in out:  # a private kernel that was never pushed answers 403, not 404
        return "unknown"
    return out.rsplit("kernelworkerstatus.", 1)[-1].strip(' ".')


def wait_kernel(kid: str, timeout_min: float) -> str:
    t_end, misses, last = time.time() + timeout_min * 60, 0, None
    while True:
        st = kernel_status(kid)
        if st != last:
            print(f"{now()} {kid}: {st}", flush=True)
            last = st
        if st in ("complete", "error") or st.startswith("cancel"):
            return st
        misses = misses + 1 if st == "unknown" else 0
        if misses >= 10:
            raise SystemExit(f"{kid} not found for 10 min; did push succeed? run: push")
        if time.time() > t_end:
            raise SystemExit("timed out waiting; the kernel keeps running on Kaggle, rerun the same command")
        time.sleep(60)


def cmd_wait(args) -> str:
    return wait_kernel(kernel_id(), args.timeout_min)


def cmd_fetch(args) -> None:
    kid = kernel_id()
    dl = BUILD / "output" / datetime.now().strftime("%Y%m%d_%H%M%S")
    dl.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kid, "-p", str(dl))
    bundle = next(dl.rglob("stage10_kaggle_bench.zip"), None)
    disc = next(dl.rglob("kernel_discovery.json"), None)
    if disc:
        print(disc.read_text(encoding="utf-8"))
    if bundle is None:
        log = next(dl.rglob("*.log"), None)
        raise SystemExit(f"no stage10_kaggle_bench.zip in {dl}; kernel log: {log}")
    if RESULT.exists():
        if not args.force:
            raise SystemExit(f"{RESULT} exists; pass --force to replace it")
        shutil.rmtree(RESULT)
    with zipfile.ZipFile(bundle) as z:
        z.extractall(ROOT)
    print(f"unpacked -> {RESULT}")
    compare_and_cost(args)


def compare_and_cost(args) -> None:
    py = sys.executable
    subprocess.check_call([py, str(ROOT / "scripts/protocol/stage10_compare.py"), str(RESULT), "--name", "stage10_benchmark"],
                          cwd=ROOT)
    quota = args.weekly_quota_h
    if quota is None and QUOTA_JSON.exists():
        quota = json.loads(QUOTA_JSON.read_text(encoding="utf-8")).get("gpu_total_h")
    if quota is None:
        raise SystemExit("weekly GPU quota unknown: run check, or pass --weekly-quota-h")
    subprocess.check_call([py, str(ROOT / "scripts/protocol/cost_projection.py"), "--timing", str(RESULT / "outcomes.csv"),
                           "--label", "kaggle_gpu", "--runner-timing", str(RESULT / "timing.json"),
                           "--weekly-quota-h", str(quota)], cwd=ROOT)


# ---- Stage 11: full-scale labels, one private GPU kernel per region --------------------------------

LABELS = ROOT / "results" / "full" / "labels_v1"          # git-ignored; archived with archive_outputs.py
LABEL_REGIONS = ["10", "05", "06", "11", "04", "03", "09", "01", "08", "02"]   # smallest first; 07 excluded (D2)
PILOT = ROOT / "results" / "stage5" / "pilot" / "outcomes.csv"
MANIFEST = ROOT / "datasets" / "manifests" / "manifest.csv"
GPU_BENCH = ROOT / "results" / "stage10" / "kaggle_bench" / "outcomes.csv"
LABEL_KEY = ("region", "filename", "variant", "radius_m", "offset_k")
ATTEMPTS_PER_FRAME = 9   # 3 radii x 3 pre-registered offsets


def label_dir(r: str) -> Path:
    return BUILD / f"kernel_labels_{r}"


def csv_rows(p: Path) -> list[dict]:
    import csv
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def region_frames(r: str) -> set[str]:
    return {m["filename"] for m in csv_rows(MANIFEST) if m["region_folder"] == r}


def gpu_hours(r: str) -> float:
    """Projected session hours for region r from the Stage 10 T4 benchmark (+ 0.1 h start-up)."""
    s_per_frame = 19.0
    if GPU_BENCH.is_file():
        rows = csv_rows(GPU_BENCH)
        frames = {x["filename"] for x in rows}
        setup = {x["filename"]: float(x["frame_setup_s"]) for x in rows}
        s_per_frame = (sum(float(x["runtime_s"]) for x in rows) + sum(setup.values())) / len(frames)
    return len(region_frames(r)) * s_per_frame / 3600 + 0.1


def check_labels(r: str, expect_commit: str) -> dict:
    """Acceptance checks for one region's labels; writes <LABELS>/<r>/check.json."""
    d = LABELS / r
    rows = csv_rows(d / "outcomes.csv") if (d / "outcomes.csv").is_file() else []
    env = json.loads((d / "env.json").read_text(encoding="utf-8")) if (d / "env.json").is_file() else {}
    frames = region_frames(r)
    keys = [tuple(x[k] for k in LABEL_KEY) for x in rows]
    pilot = {tuple(x[k] for k in LABEL_KEY): x for x in csv_rows(PILOT)}
    ref_sha = {x["config_sha256"] for x in pilot.values()}
    both = [(x, pilot[k]) for x, k in zip(rows, keys) if k in pilot]
    same = sum(x["outcome_primary"] == p["outcome_primary"] for x, p in both)
    acc = [(x, p) for x, p in both if x["accepted"] == "1" and p["accepted"] == "1"]

    def dist_m(x, p):
        import math
        dlat = (float(x["pred_lat"]) - float(p["pred_lat"])) * 110_574.0
        dlon = (float(x["pred_lon"]) - float(p["pred_lon"])) * 111_320.0 * math.cos(math.radians(float(p["pred_lat"])))
        return math.hypot(dlat, dlon)

    within = sum(dist_m(x, p) <= 5.0 for x, p in acc)
    rec = {
        "region": r, "checked_utc": now(), "rows": len(rows), "expected_rows": len(frames) * ATTEMPTS_PER_FRAME,
        "frames": len({x["filename"] for x in rows}), "expected_frames": len(frames),
        "complete": len(rows) == len(frames) * ATTEMPTS_PER_FRAME and {x["filename"] for x in rows} == frames,
        "no_duplicate_keys": len(keys) == len(set(keys)),
        "config_sha256": sorted({x["config_sha256"] for x in rows}),
        "config_sha256_matches_pilot": {x["config_sha256"] for x in rows} == ref_sha,
        "gpu": env.get("gpu"), "cuda_available": env.get("cuda_available"),
        "bundle_commit": env.get("bundle_commit"), "bundle_commit_expected": expect_commit,
        "pilot_overlap_attempts": len(both), "pilot_overlap_identical_outcome": same,
        "pilot_overlap_both_accepted": len(acc), "pilot_overlap_within_5m": within,
        "label_hours": round(sum(float(x["runtime_s"]) for x in rows) / 3600, 2),
    }
    # Pilot frames reappear in the full table: Stage 10 agreement criteria applied to them.
    pilot_ok = (not both) or (same >= 0.97 * len(both) and (not acc or within >= 0.95 * len(acc)))
    rec["pass"] = bool(rec["complete"] and rec["no_duplicate_keys"] and rec["config_sha256_matches_pilot"]
                       and rec["cuda_available"] and rec["bundle_commit"] == expect_commit and pilot_ok)
    d.mkdir(parents=True, exist_ok=True)
    (d / "check.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def fetch_labels(r: str, kid: str) -> None:
    dl = BUILD / "output" / f"labels_{r}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dl.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kid, "-p", str(dl))
    bundle = next(dl.rglob("cmbl_results.zip"), None)
    if bundle is None:
        disc = next(dl.rglob("kernel_discovery.json"), None)
        raise SystemExit(f"region {r}: no cmbl_results.zip in {dl}"
                         + (f"; discovery: {disc.read_text(encoding='utf-8')}" if disc else ""))
    if (LABELS / r).exists():
        shutil.rmtree(LABELS / r)
    with zipfile.ZipFile(bundle) as z:
        z.extractall(ROOT)


def label_quota_left() -> float | None:
    try:
        gpu = api().quota_view().gpu_quota
        return (gpu.total_time_allowed - gpu.time_used).total_seconds() / 3600
    except Exception:  # noqa: BLE001
        return None


def cmd_labels(args) -> None:
    """Stage 11: for each region, stage -> push -> wait -> fetch -> check. Rerunnable: a region that
    passed is skipped, one still running on Kaggle is waited for, not pushed again."""
    user = args.user or username()
    commit = json.loads((BUILD / "bundle_manifest.json").read_text(encoding="utf-8"))["commit"]
    regions = args.regions.split(",") if args.regions else LABEL_REGIONS
    for r in regions:
        if r not in LABEL_REGIONS:
            raise SystemExit(f"region {r} is not labelled (07 is excluded by D2)")
        chk = LABELS / r / "check.json"
        if chk.is_file() and json.loads(chk.read_text(encoding="utf-8"))["pass"] and not args.repush:
            print(f"region {r}: done (check passed)", flush=True)
            continue
        folder = label_dir(r)
        job = ["--label-args", f"--set full --regions {r}", "--out", f"results/full/labels_v1/{r}", "--skip-embed-bench"]
        stage_kernel(folder, user, f"cmbl-s11-labels-{r}", args, job)
        kid = kernel_id(folder)
        st = kernel_status(kid)
        if st in ("running", "queued") and not args.repush:
            print(f"region {r}: already {st} on Kaggle; waiting", flush=True)
        elif st == "complete" and not args.repush:
            print(f"region {r}: finished on Kaggle; fetching", flush=True)
        else:
            need, left = gpu_hours(r), label_quota_left()
            if left is not None and left < need:
                raise SystemExit(f"region {r} needs ~{need:.1f} GPU h but {left:.1f} h of the weekly quota are left; "
                                 "rerun after the quota refresh (see: check)")
            print(f"region {r}: pushing ({len(region_frames(r))} frames, ~{need:.1f} GPU h; quota left "
                  f"{'?' if left is None else f'{left:.1f}'} h)", flush=True)
            push_kernel(folder, args)
            st = "pushed"
        if args.no_wait and st != "complete":
            continue
        st = wait_kernel(kid, args.timeout_min)
        fetch_labels(r, kid)
        rec = check_labels(r, commit)
        print(json.dumps(rec, indent=2), flush=True)
        if not rec["pass"]:
            raise SystemExit(f"region {r}: check FAILED (kernel status {st}); see {LABELS / r / 'check.json'}")
    cmd_labels_status(args)


def cmd_labels_status(args) -> None:
    print(f"{'region':>6} {'frames':>6} {'rows':>6} {'expected':>8}  check", flush=True)
    for r in sorted(LABEL_REGIONS):
        chk = LABELS / r / "check.json"
        rec = json.loads(chk.read_text(encoding="utf-8")) if chk.is_file() else {}
        print(f"{r:>6} {len(region_frames(r)):>6} {rec.get('rows', 0):>6} {len(region_frames(r)) * ATTEMPTS_PER_FRAME:>8}  "
              f"{'PASS' if rec.get('pass') else ('FAIL' if rec else '-')}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["check", "stage", "upload", "push", "wait", "fetch", "compare", "all",
                                     "labels", "labels-status"])
    ap.add_argument("--regions", default=None, help="labels: comma list (default: all ten, smallest first)")
    ap.add_argument("--no-wait", action="store_true", help="labels: push and return (fetch later by rerunning)")
    ap.add_argument("--repush", action="store_true", help="labels: push again even if passed or running")
    ap.add_argument("--user", default=None, help="Kaggle username (default: from the token)")
    ap.add_argument("--only", default=None, help="upload: comma list of code,cache,data")
    ap.add_argument("--note", default=None, help="upload: version note for an existing dataset")
    ap.add_argument("--internet", action="store_true", help="stage: enable kernel internet (default off)")
    ap.add_argument("--accelerator", default=ACCELERATOR)
    ap.add_argument("--kernel-timeout-s", type=int, default=None)
    ap.add_argument("--timeout-min", type=float, default=240)
    ap.add_argument("--weekly-quota-h", type=float, default=None)
    ap.add_argument("--force", action="store_true", help="fetch: replace an existing results/stage10/kaggle_bench")
    args = ap.parse_args()
    steps = {"check": cmd_check, "stage": cmd_stage, "upload": cmd_upload, "push": cmd_push,
             "wait": cmd_wait, "fetch": cmd_fetch, "compare": compare_and_cost,
             "labels": cmd_labels, "labels-status": cmd_labels_status}
    if args.step == "all":
        for s in ("check", "stage", "upload", "push", "wait", "fetch"):
            print(f"\n=== {s} ===", flush=True)
            steps[s](args)
    else:
        steps[args.step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
