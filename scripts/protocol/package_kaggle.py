#!/usr/bin/env python3
"""Build the two small Kaggle upload bundles for Stage 10/11 (see kaggle/README.md).

  cmbl-code-<commit>.zip  committed code only (git archive of scripts, configs, manifests, kaggle/),
                          top-level folder CMBL/, plus CMBL/BUNDLE_COMMIT
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
        "code_zip": {"file": code_zip.name, "bytes": code_zip.stat().st_size, "sha256": sha256(code_zip)},
        "cache_zip": {"file": cache_zip.name, "bytes": cache_zip.stat().st_size, "sha256": sha256(cache_zip),
                      "embedding_files": n, "weights_sha256": sha256(ckpt)},
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
