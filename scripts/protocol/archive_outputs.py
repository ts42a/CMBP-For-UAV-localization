#!/usr/bin/env python3
"""Archive a full-scale output folder outside git and commit only its checksum manifest.

Policy (from 2026-09-29): pilot-scale outputs stay in git as before; full-scale outputs (Stage 11
labels, features, predictions, replay, per-frame figures) are written under results/full/ or
figures/full/, which are git-ignored, then archived with this script. The committed manifest
results/manifests/<name>.json holds the archive's sha256, every file's sha256 and size, CSV row
counts and the config/protocol hashes found in the CSVs, so a copy can be checked anywhere.

  py -3 scripts/protocol/archive_outputs.py pack results/full/labels_v1 --name labels_v1
  py -3 scripts/protocol/archive_outputs.py check results/manifests/labels_v1.json [--archive <path>]

`pack` writes build/archive/<name>.tar (uncompressed; JPG/PNG/NPZ do not compress) and the
manifest. Copy the tar to at least one place off the PC (and off Kaggle) before deleting anything.
`check` re-hashes the archive, or, without it, the folder the manifest was made from.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HASH_COLS = ("config_sha256", "prereg_sha256", "protocol_sha256", "splits_sha256")
csv.field_size_limit(1 << 30)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_stream(f) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def csv_info(p: Path) -> dict:
    with p.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        hashes = {c: set() for c in HASH_COLS if c in (r.fieldnames or [])}
        n = 0
        for row in r:
            n += 1
            for c in hashes:
                hashes[c].add(row[c])
    return {"rows": n, **{c: sorted(v) for c, v in hashes.items()}}


def describe(folder: Path) -> list[dict]:
    files = []
    for p in sorted(x for x in folder.rglob("*") if x.is_file()):
        rel = p.relative_to(folder).as_posix()
        e = {"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        if p.suffix == ".csv":
            e.update(csv_info(p))
        files.append(e)
    return files


def cmd_pack(args) -> int:
    folder = (ROOT / args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    tracked = subprocess.run(["git", "ls-files", "--", str(folder)], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if tracked:
        print(f"warning: {folder} has files tracked by git; full-scale outputs belong under a git-ignored folder",
              file=sys.stderr)
    files = describe(folder)
    out = ROOT / "build" / "archive"
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / f"{args.name}.tar"
    with tarfile.open(tar_path, "w") as t:
        for e in files:
            t.add(folder / e["path"], arcname=f"{args.name}/{e['path']}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "name": args.name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_folder": folder.relative_to(ROOT).as_posix() if folder.is_relative_to(ROOT) else str(folder),
        "git_commit": commit, "note": args.note,
        "archive": {"file": tar_path.name, "bytes": tar_path.stat().st_size, "sha256": sha256_file(tar_path)},
        "n_files": len(files), "total_bytes": sum(e["bytes"] for e in files),
        "csv_rows": {e["path"]: e["rows"] for e in files if "rows" in e},
        "hashes_in_csvs": {c: sorted({h for e in files for h in e.get(c, [])}) for c in HASH_COLS
                           if any(c in e for e in files)},
        "files": files,
    }
    mdir = ROOT / "results" / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{args.name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("archive", "n_files", "total_bytes", "csv_rows", "hashes_in_csvs")}, indent=2))
    multi = {c: v for c, v in manifest["hashes_in_csvs"].items() if len(v) > 1}
    if multi:
        print(f"warning: more than one hash per column {multi} (plan Stage 11: one config hash in every output)", file=sys.stderr)
    print(f"archive {tar_path}\nmanifest results/manifests/{args.name}.json (commit this)")
    return 0


def cmd_check(args) -> int:
    m = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    want = {e["path"]: e for e in m["files"]}
    bad = []
    if args.archive:
        a = Path(args.archive)
        if sha256_file(a) != m["archive"]["sha256"]:
            bad.append("archive sha256 differs")
        seen = set()
        with tarfile.open(a) as t:
            for ti in t.getmembers():
                if not ti.isfile():
                    continue
                rel = ti.name.split("/", 1)[1]
                seen.add(rel)
                if rel not in want or sha256_stream(t.extractfile(ti)) != want[rel]["sha256"]:
                    bad.append(f"{rel}: not in manifest or sha256 differs")
        bad += [f"{p}: missing from archive" for p in want if p not in seen]
    else:
        folder = ROOT / m["source_folder"]
        have = {e["path"]: e for e in describe(folder)}
        bad += [f"{p}: missing or sha256 differs" for p, e in want.items() if have.get(p, {}).get("sha256") != e["sha256"]]
        bad += [f"{p}: not in manifest" for p in have if p not in want]
    print("OK" if not bad else "FAIL\n" + "\n".join(bad[:50]))
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("folder")
    p.add_argument("--name", required=True)
    p.add_argument("--note", default="")
    c = sub.add_parser("check")
    c.add_argument("manifest")
    c.add_argument("--archive", default=None)
    args = ap.parse_args()
    return cmd_pack(args) if args.cmd == "pack" else cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
