#!/usr/bin/env python3
"""Freeze Stage 3B region-03 tune / validation splits BEFORE any matcher tuning.

Writes configs/stage3b_splits.yaml with fixed query IDs. Re-running is a no-op
unless --force is passed (refuses by default once frozen).
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from project_paths import MANIFEST  # noqa: E402
from query_select import load_manifest_region, select_queries  # noqa: E402

OUT = ROOT / "configs" / "stage3b_splits.yaml"
STAGE3A_CSV = ROOT / "reports" / "stage3a_pilot_results.csv"

# Fixed seeds — do not change after freeze without documenting a protocol revision.
VAL_SEED = 20260924
TUNE_SEED = 42
N_VAL = 30
N_TUNE = 24


def filenames_from_csv(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return [r["filename"] for r in csv.DictReader(f)]


def select_disjoint(manifest: list[dict], n: int, seed: int, exclude: set[str]) -> list[dict]:
    """Stratified pool then RNG pick, excluding reserved filenames."""
    pool = [r for r in select_queries(manifest, min(len(manifest), n * 4), seed) if r["filename"] not in exclude]
    # Fill from full flight if pool short
    by_num = sorted(manifest, key=lambda r: int(r["num"]))
    for r in by_num:
        if r["filename"] not in exclude and r not in pool:
            pool.append(r)
    rng = np.random.default_rng(seed)
    if len(pool) > n:
        idx = rng.choice(len(pool), size=n, replace=False)
        chosen = [pool[i] for i in sorted(idx)]
    else:
        chosen = pool[:n]
    return sorted(chosen, key=lambda r: int(r["num"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        print(f"Already frozen: {OUT} (use --force to overwrite)")
        return 0

    manifest = load_manifest_region("03")
    compare_files = filenames_from_csv(STAGE3A_CSV) if STAGE3A_CSV.is_file() else []

    # Freeze VAL first (untouched for gate); then TUNE excluding VAL.
    val = select_disjoint(manifest, N_VAL, VAL_SEED, exclude=set())
    val_names = {r["filename"] for r in val}
    tune = select_disjoint(manifest, N_TUNE, TUNE_SEED, exclude=val_names)
    tune_names = {r["filename"] for r in tune}
    overlap_vt = val_names & tune_names
    if overlap_vt:
        raise SystemExit(f"BUG: tune/val overlap {overlap_vt}")

    payload = {
        "schema_version": "stage3b_splits_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "region": "03",
        "notes": (
            "Validation subset is frozen BEFORE matcher tuning. "
            "Tune hyperparameters only on tune_queries. "
            "Gate decision uses val_queries only. "
            "compare_stage3a_queries is for baseline comparison, not for choosing the matcher."
        ),
        "val_seed": VAL_SEED,
        "tune_seed": TUNE_SEED,
        "n_val": len(val),
        "n_tune": len(tune),
        "val_queries": [{"num": int(r["num"]), "filename": r["filename"]} for r in val],
        "tune_queries": [{"num": int(r["num"]), "filename": r["filename"]} for r in tune],
        "compare_stage3a_queries": [{"filename": f} for f in compare_files],
        "overlap_tune_val": sorted(overlap_vt),
        "overlap_tune_stage3a": sorted(tune_names & set(compare_files)),
        "overlap_val_stage3a": sorted(val_names & set(compare_files)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"Froze {OUT}")
    print(f"  val={len(val)} tune={len(tune)} compare_stage3a={len(compare_files)}")
    print(f"  overlap val&stage3a={len(payload['overlap_val_stage3a'])} "
          f"tune&stage3a={len(payload['overlap_tune_stage3a'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
