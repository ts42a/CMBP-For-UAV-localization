#!/usr/bin/env python3
"""Stage 10 Kaggle kernel entry point (pushed by scripts/protocol/kaggle_stage10.py push).

Kaggle mounts each private dataset under /kaggle/input, but the exact layout depends on the
slug, on the mount scheme, and on whether an uploaded archive was extracted. This script finds
the three inputs by content instead of by path:

  code   a folder holding kaggle/stage10_runner.py        (or cmbl-code-*.zip)
  cache  embeddings/region10_dinov2_vits14_224.npz and
         torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth   (or cmbl-cache.zip)
  data   region folders 01..11 each holding drone/ and satellite<RR>.tif  (or <RR>.tar/.zip)

Archives are extracted to a scratch folder only when no extracted copy exists. The code is
copied to /kaggle/working/CMBL (the runner writes its results inside the code root), the
frozen runner (kaggle/stage10_runner.py) is run unchanged, and the result folder is zipped to
/kaggle/working/stage10_kaggle_bench.zip for download. A Stage 11 job staged by
scripts/protocol/kaggle_stage10.py sets KERNEL_ARGS (frame set and --out) and writes cmbl_results.zip.

Local test: set CMBL_KAGGLE_INPUT / CMBL_KAGGLE_WORK / CMBL_KAGGLE_SCRATCH and pass runner
arguments after "--", e.g. "-- --device cpu --max-frames 2".
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

INPUT = Path(os.environ.get("CMBL_KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("CMBL_KAGGLE_WORK", "/kaggle/working"))
SCRATCH = Path(os.environ.get("CMBL_KAGGLE_SCRATCH", "/tmp/cmbl"))
REGIONS = [f"{i:02d}" for i in range(1, 12)]
REQUIRED_MODULES = ["torch", "torchvision", "numpy", "cv2", "yaml", "pyproj", "tifffile", "PIL"]
OUT_REL = "results/stage10/kaggle_bench"
MAX_DEPTH = 6
# Runner arguments for a staged job (a script kernel gets no argv); kaggle_stage10.py rewrites this
# line when it stages a Stage 11 region, e.g. ["--label-args", "--set full --regions 02", "--out", ...].
KERNEL_ARGS: list[str] = []
# Bundle commit the staged job must run (rewritten with KERNEL_ARGS); a kernel pushed right after a new
# cmbl-code version can still mount the previous one.
EXPECT_COMMIT = ""


def walk(root: Path, max_depth: int = MAX_DEPTH):
    """os.walk limited in depth; never descends into drone/ image folders."""
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        p = Path(dirpath)
        if len(p.parts) - root_depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = sorted(d for d in dirnames if d != "drone" and not d.startswith("."))
        yield p, dirnames, sorted(filenames)


def dir_link(target: Path, link_path: Path) -> None:
    """Symlink, or a junction on Windows without symlink rights (local tests only)."""
    try:
        os.symlink(target, link_path, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link_path), str(target)], stdout=subprocess.DEVNULL)


def extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / f".extracted_{archive.name}"
    if not marker.exists():
        t0 = time.perf_counter()
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                import stat
                seen=set()
                for member in z.infolist():
                    name=member.filename
                    target=(dest/name).resolve()
                    if ('\\' in name or ':' in name or '..' in Path(name).parts or
                            not target.is_relative_to(dest.resolve()) or
                            stat.S_ISLNK(member.external_attr>>16) or name.casefold() in seen):
                        raise ValueError(f'unsafe ZIP member: {name}')
                    seen.add(name.casefold())
                if sum(m.file_size for m in z.infolist()) > 100*1024**3:
                    raise ValueError('archive exceeds 100 GiB extraction limit')
                z.extractall(dest)
        else:
            with tarfile.open(archive) as t:
                t.extractall(dest, filter='data')
        marker.write_text(f"{time.perf_counter() - t0:.1f}\n")
        print(f"extracted {archive} -> {dest} in {marker.read_text().strip()} s", flush=True)
    return dest


def find_code(roots: list[Path]) -> Path | None:
    for root in roots:
        for p, _, files in walk(root):
            if p.name == "kaggle" and "stage10_runner.py" in files and (p.parent / "scripts").is_dir():
                return p.parent
    return None


def find_cache(roots: list[Path]) -> tuple[Path | None, Path | None]:
    emb = th = None
    for root in roots:
        for p, _, files in walk(root):
            if emb is None and "region10_dinov2_vits14_224.npz" in files:
                emb = p
            if th is None and p.name == "checkpoints" and "dinov2_vits14_pretrain.pth" in files \
                    and (p.parent / "facebookresearch_dinov2_main").is_dir() and p.parent.name == "hub":
                th = p.parent.parent
    return emb, th


def is_region_dir(p: Path) -> bool:
    return p.name in REGIONS and (p / "drone").is_dir() and any(p.glob("satellite*.tif"))


def find_data(roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        for p, _, _ in walk(root):
            if p.name not in found and is_region_dir(p):
                found[p.name] = p
    return found


def find_archives(pattern_names: set[str]) -> list[Path]:
    out = []
    for p, _, files in walk(INPUT):
        out += [p / f for f in files if f in pattern_names or any(f.startswith(n) for n in pattern_names)]
    return out


def main() -> int:
    runner_extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else list(KERNEL_ARGS)
    out_rel = runner_extra[runner_extra.index("--out") + 1] if "--out" in runner_extra else OUT_REL
    t_start = time.perf_counter()
    disc: dict = {"input": str(INPUT), "work": str(WORK), "scratch": str(SCRATCH), "extracted": []}
    WORK.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    # Dependencies first: internet is off, so a missing module is a hard stop with a clear message.
    missing = []
    for m in REQUIRED_MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{m}: {e}")
    disc["missing_modules"] = missing
    # Kaggle silently runs a GPU-requested kernel on CPU (torch "+cpu" image) when the account
    # cannot get a GPU, e.g. without phone verification; that would mislabel every timing.
    cuda = None
    if not any(m.startswith("torch:") for m in missing):
        import torch
        cuda = torch.cuda.is_available()
    disc["cuda_available"] = cuda

    t0 = time.perf_counter()
    code = find_code([INPUT])
    if code is None:
        for a in find_archives({"cmbl-code-"}):
            if a.suffix in (".zip", ".tar"):
                disc["extracted"].append(str(a))
                code = find_code([extract(a, SCRATCH / "code")])
                if code:
                    break
    emb, th = find_cache([INPUT])
    if emb is None or th is None:
        for a in find_archives({"cmbl-cache.zip", "cmbl-cache.tar"}):
            disc["extracted"].append(str(a))
            e2, t2 = find_cache([extract(a, SCRATCH / "cache")])
            emb, th = emb or e2, th or t2
    regions = find_data([INPUT])
    missing_regions = [r for r in REGIONS if r not in regions]
    if missing_regions:
        for a in find_archives({f"{r}.tar" for r in missing_regions} | {f"{r}.zip" for r in missing_regions}):
            disc["extracted"].append(str(a))
            regions.update({k: v for k, v in find_data([extract(a, SCRATCH / "data" / a.stem)]).items()
                            if k not in regions})
    disc["discovery_s"] = round(time.perf_counter() - t0, 3)
    disc["code"] = str(code) if code else None
    disc["embeddings"] = str(emb) if emb else None
    disc["torch_home"] = str(th) if th else None
    disc["regions"] = {k: str(v) for k, v in sorted(regions.items())}

    try:
        listing = {str(p): sorted(os.listdir(p))[:20] for p in [INPUT, *INPUT.iterdir()] if p.is_dir()}
    except OSError:
        listing = {}
    disc["input_listing"] = listing
    problems = []
    if missing:
        problems.append(f"missing Python modules (internet is off): {missing}")
    if cuda is False and "cpu" not in runner_extra:
        problems.append("no GPU attached (torch.cuda.is_available() is False): phone-verify the Kaggle account "
                        "and push with an accelerator, or pass '-- --device cpu' for a deliberate CPU run")
    if code is None:
        problems.append("code input not found (kaggle/stage10_runner.py)")
    else:
        need = [n for n in ("manifest.csv", "satellite_metadata.csv") if not (code / "datasets" / "manifests" / n).is_file()]
        if need:  # git-ignored locally; scripts/protocol/package_kaggle.py adds them to the code bundle
            problems.append(f"code bundle lacks datasets/manifests/{need}; rebuild it with package_kaggle.py")
        bc = code / "BUNDLE_COMMIT"
        got = bc.read_text(encoding="utf-8").strip() if bc.is_file() else "unknown"
        disc["bundle_commit"] = got
        if EXPECT_COMMIT and got != EXPECT_COMMIT:
            problems.append(f"code bundle is {got[:7]}, job expects {EXPECT_COMMIT[:7]}: the new cmbl-code "
                            "version was not mounted yet; push again in a few minutes")
    if emb is None or th is None:
        problems.append("cache input not found (embeddings/ and torch_home/hub/)")
    bench = [r for r in REGIONS if r != "07"]
    if [r for r in bench if r not in regions]:
        problems.append(f"data regions not found: {[r for r in bench if r not in regions]}")
    disc["problems"] = problems
    (WORK / "kernel_discovery.json").write_text(json.dumps(disc, indent=2), encoding="utf-8")
    print(json.dumps(disc, indent=2), flush=True)
    if problems:
        print("STOP: " + "; ".join(problems), flush=True)
        return 2

    # Read-only inputs -> one data view with the layout the repo expects (datasets/raw/UAV_VisLoc_dataset/<RR>/).
    view = SCRATCH / "UAV_VisLoc_dataset"
    view.mkdir(parents=True, exist_ok=True)
    for r, p in regions.items():
        lp = view / r
        if not (lp.exists() or lp.is_symlink()):
            dir_link(p, lp)

    # torch.hub creates <TORCH_HOME>/hub/trusted_list on every load, and /kaggle/input is read-only:
    # give it a writable hub folder that links to the read-only DINOv2 repo and weights.
    th_view = SCRATCH / "torch_home"
    (th_view / "hub").mkdir(parents=True, exist_ok=True)
    for name in ("facebookresearch_dinov2_main", "checkpoints"):
        lp = th_view / "hub" / name
        if not (lp.exists() or lp.is_symlink()):
            dir_link(th / "hub" / name, lp)

    dst = WORK / "CMBL"
    if not dst.exists():
        shutil.copytree(code, dst, ignore=shutil.ignore_patterns("__pycache__"))
    cmd = [sys.executable, str(dst / "kaggle" / "stage10_runner.py"), "--code-root", str(dst),
           "--data-root", str(view), "--emb-root", str(emb), "--torch-home", str(th_view), "--out", out_rel,
           *runner_extra]
    print("running: " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)

    res = dst / out_rel
    res.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WORK / "kernel_discovery.json", res / "kernel_discovery.json")
    bundle = WORK / ("stage10_kaggle_bench.zip" if out_rel == OUT_REL else "cmbl_results.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(res.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(dst).as_posix())
    print(f"runner exit {rc}; wrote {bundle} ({bundle.stat().st_size} bytes); "
          f"kernel total {time.perf_counter() - t_start:.1f} s", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
