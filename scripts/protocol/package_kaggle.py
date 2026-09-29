#!/usr/bin/env python3
"""Build the two small Kaggle upload bundles for Stage 10/11 (see kaggle/README.md).

  cmbl-code-<commit>.zip  committed code (git archive of scripts, configs, kaggle/), top-level folder
                          CMBL/, plus CMBL/BUNDLE_COMMIT, plus the local dataset manifests
                          (datasets/manifests/*.csv: UAV-VisLoc metadata, git-ignored because it is
                          not cleared for public redistribution; the Kaggle dataset stays private)
  cmbl-cache.zip          embeddings/ (tile-embedding caches) and torch_home/hub/ (DINOv2 repo +
                          weights), so the notebook needs no internet

The 24 GB dataset (datasets/raw/UAV_VisLoc_dataset) is uploaded separately as a private Kaggle
dataset. Nothing here is uploaded; the files are written to build/kaggle/.

  py -3 scripts/protocol/package_kaggle.py [--ref HEAD]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = ["scripts", "configs", "datasets/manifests", "kaggle", "requirements-localization.txt"]
MANIFEST_DIR = ROOT / "datasets" / "manifests"
REQUIRED_MANIFESTS = ("manifest.csv", "satellite_metadata.csv")  # read by the labeller at run time


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD", help="git ref to package (a tag for the final run)")
    ap.add_argument("--torch-hub", default=str(Path.home() / ".cache" / "torch" / "hub"))
    ap.add_argument("--out", default="build/kaggle")
    args = ap.parse_args()

    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *CODE_PATHS], cwd=ROOT, text=True).strip()
    if dirty and args.ref == "HEAD":
        print("warning: uncommitted changes under the packaged paths are NOT included:\n" + dirty, file=sys.stderr)
    commit = subprocess.check_output(["git", "rev-parse", args.ref], cwd=ROOT, text=True).strip()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    code_zip = out / f"cmbl-code-{commit[:7]}.zip"
    raw = subprocess.check_output(["git", "archive", "--format=zip", "--prefix=CMBL/", commit, *CODE_PATHS], cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item.filename))
        dst.writestr("CMBL/BUNDLE_COMMIT", commit + "\n")
        packed = set(src.namelist())
        local = sorted(MANIFEST_DIR.glob("*.csv"))
        missing = [n for n in REQUIRED_MANIFESTS if not (MANIFEST_DIR / n).is_file()]
        if missing:
            raise SystemExit(f"missing local manifests {missing}: rebuild them from the dataset (see README)")
        frozen = json.loads((ROOT / "configs" / "splits_v1.json").read_text(encoding="utf-8"))["manifest_sha256"]
        if sha256(MANIFEST_DIR / "manifest.csv") != frozen:
            raise SystemExit("datasets/manifests/manifest.csv does not match manifest_sha256 in configs/splits_v1.json; "
                             "the frozen splits were built from a different manifest, so labels would not be comparable")
        for f in local:
            arc = f"CMBL/datasets/manifests/{f.name}"
            if arc not in packed:  # a committed copy (older commits) wins; otherwise add the local file
                dst.write(f, arc)
    local_manifests = {f.name: sha256(f) for f in local}
    with zipfile.ZipFile(code_zip) as z:  # the bundled copies must be byte-identical to the local files
        bad = [n for n, h in local_manifests.items()
               if hashlib.sha256(z.read(f"CMBL/datasets/manifests/{n}")).hexdigest() != h]
    if bad:
        raise SystemExit(f"bundled manifests differ from the local files: {bad}")

    hub = Path(args.torch_hub)
    repo_dir, ckpt = hub / "facebookresearch_dinov2_main", hub / "checkpoints" / "dinov2_vits14_pretrain.pth"
    emb_dir = ROOT / "datasets" / "cache" / "embeddings"
    for p in (repo_dir, ckpt, emb_dir):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    cache_zip = out / "cmbl-cache.zip"
    n = 0
    with zipfile.ZipFile(cache_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(emb_dir.glob("region*_dinov2_vits14_224.*")):
            z.write(f, f"embeddings/{f.name}")
            n += 1
        for f in sorted(repo_dir.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and ".git" not in f.parts:
                z.write(f, f"torch_home/hub/{repo_dir.name}/{f.relative_to(repo_dir).as_posix()}")
        z.write(ckpt, f"torch_home/hub/checkpoints/{ckpt.name}")

    manifest = {
        "commit": commit, "ref": args.ref,
        "code_zip": {"file": code_zip.name, "bytes": code_zip.stat().st_size, "sha256": sha256(code_zip),
                     "local_manifests_sha256": local_manifests},
        "cache_zip": {"file": cache_zip.name, "bytes": cache_zip.stat().st_size, "sha256": sha256(cache_zip),
                      "embedding_files": n, "weights_sha256": sha256(ckpt)},
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
