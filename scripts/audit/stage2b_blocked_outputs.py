#!/usr/bin/env python3
"""Emit Stage 2B CSV deliverables when full zip is unavailable (partial download blocked).

Does not replace full_dataset_audit.py — run that after extract completes.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# reuse audit logic
from full_dataset_audit import (
    DATASET,
    DOCUMENTED_MAPS,
    DOCUMENTED_UAV,
    FULL,
    MANIFESTS,
    RAW,
    REPORTS,
    ROOT,
    SAMPLE,
    inventory_region,
    inspect_satellite,
    list_range_csv,
    walk_inventory,
    write_csv,
)

PAPER_TABLE = ROOT / "sources" / "UAV-VisLoc_arxiv_table2_extract.txt"


def load_range(root: Path) -> dict:
    p = list_range_csv(root)
    out = {}
    if not p:
        return out
    with p.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r.get("mapname", "").strip()] = r
    return out


def raw_and_sample_inventory() -> list[dict]:
    rows = []
    for base, prefix in [(RAW, "raw"), (SAMPLE, "sample")]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                rel = f"{prefix}/" + p.relative_to(base).as_posix()
                rows.append(
                    {
                        "relative_path": rel,
                        "extension": p.suffix.lower(),
                        "size_bytes": p.stat().st_size,
                    }
                )
    part = RAW / "UAV_VisLoc_dataset.zipi4suhujo.part"
    if part.exists():
        rows.append(
            {
                "relative_path": "raw/UAV_VisLoc_dataset.zipi4suhujo.part",
                "extension": ".part",
                "size_bytes": part.stat().st_size,
            }
        )
    return rows


def capture_date_rows(region_stats: list[dict]) -> list[dict]:
    rows = []
    for rs in region_stats:
        rid = rs["region_folder"]
        sat_paper = "2021-04" if rs.get("region_label") == "Taizhou-1" else ""
        rows.append(
            {
                "region_folder": rid,
                "region_label": rs.get("region_label", ""),
                "flight_date_min": rs.get("date_min", ""),
                "flight_date_max": rs.get("date_max", ""),
                "flight_date_source": "pose CSV date column",
                "flight_timezone": "unresolved",
                "satellite_acquisition_date_release_csv": "not in satellite_coordinates_range.csv",
                "satellite_acquisition_date_paper_table2": sat_paper,
                "satellite_date_source": "arXiv:2405.11936 TABLE II (month precision) when site known",
                "file_mtime_note": "not used as acquisition date",
                "comparison": (
                    "Taizhou-1: drone 2018-10 (CSV day-level) vs satellite 2021-04 (paper month) — "
                    "temporal gap; not verified as acquisition vs export"
                    if sat_paper
                    else "full folder↔site mapping pending complete release"
                ),
            }
        )
    # paper-only sites (no folder id yet)
    if PAPER_TABLE.exists():
        text = PAPER_TABLE.read_text(encoding="utf-8", errors="replace")
        for site in (
            "Changjiang-20",
            "Changjiang-23",
            "Taizhou-6",
            "Yunnan",
            "Zhuxi",
            "Donghuayuan",
            "Huzhou-3",
            "Huzhou-6",
            "Huailai",
            "Shandan",
        ):
            if site in text:
                rows.append(
                    {
                        "region_folder": "unmapped",
                        "region_label": site,
                        "flight_date_min": "",
                        "flight_date_max": "",
                        "flight_date_source": "arXiv TABLE II (month)",
                        "flight_timezone": "unresolved",
                        "satellite_acquisition_date_release_csv": "n/a",
                        "satellite_acquisition_date_paper_table2": "see TABLE II in sources/UAV-VisLoc_arxiv_table2_extract.txt",
                        "satellite_date_source": "paper only until full inventory",
                        "file_mtime_note": "n/a",
                        "comparison": "folder id assignment pending full release extract",
                    }
                )
    return rows


def flatten_issues(issues: list[dict]) -> list[dict]:
    out = []
    for i in issues:
        out.append(
            {
                "severity": i.get("severity", ""),
                "category": i.get("category", i.get("reason_code", "")),
                "reason_code": i.get("reason_code", ""),
                "region": i.get("region", ""),
                "item": i.get("item", ""),
                "detail": i.get("detail", i.get("evidence", "")),
                "evidence": i.get("evidence", ""),
            }
        )
    return out


def main() -> int:
    free = shutil.disk_usage("F:\\").free
    issues = [
        {
            "severity": "definite",
            "category": "download",
            "reason_code": "full_archive_incomplete",
            "region": "ALL",
            "item": "UAV_VisLoc_dataset.zip",
            "evidence": "Only partial gdown .part ~68%; Drive quota blocks resume",
        },
        {
            "severity": "definite",
            "category": "inventory",
            "reason_code": "full_release_not_extracted",
            "region": "ALL",
            "item": "datasets/full",
            "evidence": "Cannot verify 6742/11 counts without complete zip",
        },
    ]

    range_by = load_range(SAMPLE)
    res = inventory_region("03", SAMPLE / "03", range_by, issues)

    flight_stats = [res["flight_stats"]]
    region_stats = [res["region_stats"]]
    relationships = [res["relationship"]]
    sat_metas = [res["sat_meta"]]

    # manifest: keep sample rows (already at dataset/manifest.csv)
    manifest_path = DATASET / "manifest.csv"
    if not manifest_path.exists():
        write_csv(manifest_path, res["manifest_rows"])

    write_csv(DATASET / "full_file_inventory.csv", raw_and_sample_inventory())
    write_csv(DATASET / "flight_map_relationships.csv", relationships)
    write_csv(DATASET / "satellite_metadata.csv", sat_metas)
    write_csv(REPORTS / "flight_stats.csv", flight_stats)
    write_csv(REPORTS / "region_stats.csv", region_stats)
    write_csv(REPORTS / "capture_date_comparison.csv", capture_date_rows(region_stats))
    write_csv(REPORTS / "full_frame_issues.csv", flatten_issues(issues))

    for name in [
        "manifest.csv",
        "full_file_inventory.csv",
        "flight_map_relationships.csv",
        "satellite_metadata.csv",
        "data_dictionary.csv",
    ]:
        src = DATASET / name
        if src.exists():
            (MANIFESTS / name).write_bytes(src.read_bytes())

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "blocked_partial_download",
        "documented_uav": DOCUMENTED_UAV,
        "documented_maps": DOCUMENTED_MAPS,
        "inventoried_uav_sample_only": res["n_jpg"],
        "inventoried_maps_sample_only": 1,
        "F_free_bytes": free,
        "partial_bytes": (RAW / "UAV_VisLoc_dataset.zipi4suhujo.part").stat().st_size
        if (RAW / "UAV_VisLoc_dataset.zipi4suhujo.part").exists()
        else 0,
    }
    (REPORTS / "full_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
