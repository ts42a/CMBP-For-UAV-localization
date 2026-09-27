#!/usr/bin/env python3
"""Stage 3 geographic splits: leave-one-region-group-out outer loop, group k-fold inner loop.

Groups come from reports/independent_region_groups.csv (Stage 2C). Every frame of a
region stays with its region, so contiguous frames and a shared map (08+09) never
straddle roles. Writes configs/splits_v1.json; refuses to overwrite a different file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS_CSV = ROOT / "reports" / "independent_region_groups.csv"
MANIFEST = ROOT / "datasets" / "manifests" / "manifest.csv"
OUT = ROOT / "configs" / "splits_v1.json"
LOCALIZER_DEV_REGIONS = ["03"]
ALL_REGIONS = [f"{i:02d}" for i in range(1, 12)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouping_at(threshold_m: int) -> list[list[str]]:
    with GROUPS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["record_type"] == "sensitivity" and int(r["threshold_m"]) == threshold_m:
                return [g.split("+") for g in r["member_regions"].split("|")]
    raise SystemExit(f"No sensitivity grouping at {threshold_m} m in {GROUPS_CSV}")


def build_scheme(name: str, threshold_m: int, groups: list[list[str]], counts: Counter) -> dict:
    gids = [f"G{i + 1}" for i in range(len(groups))]
    gmap = dict(zip(gids, groups))
    folds = []
    for test in gids:
        source = [g for g in gids if g != test]
        inner = [
            {"inner_id": f"{test}-i{j + 1}", "val_group": v, "train_groups": [g for g in source if g != v]}
            for j, v in enumerate(source)
        ]
        folds.append(
            {
                "fold_id": test,
                "test_group": test,
                "test_regions": gmap[test],
                "source_groups": source,
                "inner_folds": inner,
                "localizer_dev_contaminated": any(r in LOCALIZER_DEV_REGIONS for r in gmap[test]),
            }
        )
    return {
        "name": name,
        "stage2c_threshold_m": threshold_m,
        "groups": {g: {"regions": rs, "n_frames": sum(counts[r] for r in rs)} for g, rs in gmap.items()},
        "outer_folds": folds,
    }


def check_scheme(s: dict) -> list[str]:
    errs = []
    regions = [r for g in s["groups"].values() for r in g["regions"]]
    if sorted(regions) != ALL_REGIONS:
        errs.append(f"{s['name']}: regions not a partition: {sorted(regions)}")
    for f in s["outer_folds"]:
        test = set(f["test_regions"])
        src = {r for g in f["source_groups"] for r in s["groups"][g]["regions"]}
        if test & src:
            errs.append(f"{s['name']} {f['fold_id']}: test/source overlap {test & src}")
        if test | src != set(ALL_REGIONS):
            errs.append(f"{s['name']} {f['fold_id']}: fold does not cover all regions")
        for inn in f["inner_folds"]:
            v = set(s["groups"][inn["val_group"]]["regions"])
            t = {r for g in inn["train_groups"] for r in s["groups"][g]["regions"]}
            if v & t or v & test or t & test:
                errs.append(f"{s['name']} {inn['inner_id']}: role overlap")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate existing file only")
    args = ap.parse_args()

    with MANIFEST.open(encoding="utf-8", newline="") as f:
        counts = Counter(r["region_folder"] for r in csv.DictReader(f))

    schemes = {
        "primary_7group": build_scheme("primary_7group", 5000, grouping_at(5000), counts),
        "sensitivity_10group": build_scheme("sensitivity_10group", 0, grouping_at(0), counts),
    }
    errs = [e for s in schemes.values() for e in check_scheme(s)]
    if errs:
        print("\n".join(errs), file=sys.stderr)
        return 1

    payload = {
        "schema_version": "splits_v1",
        "manifest_path": "datasets/manifests/manifest.csv",
        "manifest_sha256": sha256(MANIFEST),
        "groups_source": "reports/independent_region_groups.csv",
        "groups_source_sha256": sha256(GROUPS_CSV),
        "n_frames_total": sum(counts.values()),
        "unit_of_assignment": "region (all frames of a region share its group's role)",
        "localizer_dev_regions": LOCALIZER_DEV_REGIONS,
        "localizer_dev_note": (
            "Matcher dinov2_c2f was tuned on region 03. Folds whose test group contains 03 "
            "are flagged localizer_dev_contaminated and reported separately."
        ),
        "primary_scheme": "primary_7group",
        "schemes": schemes,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)

    if args.check:
        if not OUT.is_file():
            print(f"missing {OUT}", file=sys.stderr)
            return 1
        old = json.loads(OUT.read_text(encoding="utf-8"))
        old.pop("created_at", None)
        if json.dumps(old, indent=2, sort_keys=True) != body:
            print("splits_v1.json differs from regenerated splits", file=sys.stderr)
            return 1
        print(f"splits_v1.json OK sha256={sha256(OUT)}")
        return 0

    if OUT.is_file():
        print(f"{OUT} exists; never regenerate a frozen split. Use --check.", file=sys.stderr)
        return 1
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT} sha256={sha256(OUT)}")
    for s in schemes.values():
        print(s["name"], {g: v["regions"] for g, v in s["groups"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
