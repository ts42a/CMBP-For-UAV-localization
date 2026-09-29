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
import re
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
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "kaggle" / "stage10_kernel.py", KERNEL_DIR / "stage10_kernel.py")
    km = {"id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "stage10_kernel.py",
          "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
          "enable_tpu": False, "enable_internet": bool(args.internet),
          "dataset_sources": [f"{user}/{s}" for s in SLUGS.values()], "competition_sources": [],
          "kernel_sources": [], "model_sources": []}
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(km, indent=2), encoding="utf-8")
    print(json.dumps({"staged": str(UPLOAD), "kernel": km, "bundle_commit": bm["commit"]}, indent=2))


def dataset_exists(user: str, slug: str) -> bool:
    out = kaggle("datasets", "status", f"{user}/{slug}", capture=True, check=False)
    return not re.search(r"404|not found|Permission", out, re.I)


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
        if "ready" in low or "error" in low or "fail" in low or time.time() > t_end:
            return out
        time.sleep(30)


def cmd_push(args) -> None:
    if not (KERNEL_DIR / "kernel-metadata.json").exists():
        raise SystemExit("kernel not staged; run: stage")
    extra = ["-t", str(args.kernel_timeout_s)] if args.kernel_timeout_s else []
    kaggle("kernels", "push", "-p", str(KERNEL_DIR), "--accelerator", args.accelerator, *extra)


def kernel_id() -> str:
    return json.loads((KERNEL_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))["id"]


def cmd_wait(args) -> str:
    kid, t_end = kernel_id(), time.time() + args.timeout_min * 60
    while True:
        out = kaggle("kernels", "status", kid, capture=True, check=False).strip()
        print(out, flush=True)
        low = out.lower()
        if any(s in low for s in ("complete", "error", "cancel")):
            return out
        if time.time() > t_end:
            raise SystemExit("timed out waiting; the kernel keeps running on Kaggle, rerun: wait")
        time.sleep(60)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["check", "stage", "upload", "push", "wait", "fetch", "compare", "all"])
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
             "wait": cmd_wait, "fetch": cmd_fetch, "compare": compare_and_cost}
    if args.step == "all":
        for s in ("check", "stage", "upload", "push", "wait", "fetch"):
            print(f"\n=== {s} ===", flush=True)
            steps[s](args)
    else:
        steps[args.step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
