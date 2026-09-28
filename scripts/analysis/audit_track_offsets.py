#!/usr/bin/env python3
"""Audit whether localizer fits land offset from the CSV position along the flight track.

Read-only: labels are never corrected (correcting ground truth from matcher output would be
circular). Per frame, takes the best patch fit (most inliers, >= --min-inliers) over all its
attempts, splits the CSV->prediction vector into along-track and cross-track components, and
compares the offset bearing with the local track bearing (from frames num-1 / num+1).

  py -3 scripts/analysis/audit_track_offsets.py results/stage5/pilot
Writes reports/gt_track_offset_audit.{md,csv}.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))
from windowed_matcher import M_PER_DEG_LAT, M_PER_DEG_LON_EQ  # noqa: E402

AXIS_WINDOW_DEG = 30.0  # |offset axis - track axis| <= 30 deg; chance = 30/90 = 1/3
FAR_M = 100.0


def enu(lat0: float, lon0: float, lat: float, lon: float) -> np.ndarray:
    return np.array([(lon - lon0) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat0)), (lat - lat0) * M_PER_DEG_LAT])


def bearing(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[0], v[1])) % 360.0)


def track_vector(man: dict, region: str, num: int) -> np.ndarray | None:
    prev, nxt = man.get((region, num - 1)), man.get((region, num + 1))
    a, b = prev or man[(region, num)], nxt or man[(region, num)]
    v = enu(a[0], a[1], b[0], b[1])
    return v if np.hypot(*v) > 1.0 else None


def boot_median_ci(x: np.ndarray, reps: int = 2000, seed: int = 20260928) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(reps, len(x)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-inliers", type=int, default=16)
    args = ap.parse_args()

    man = {}
    with (ROOT / "datasets" / "manifests" / "manifest.csv").open(encoding="utf-8", newline="") as f:
        for m in csv.DictReader(f):
            man[(m["region_folder"], int(m["num"]))] = (float(m["lat"]), float(m["lon"]))

    best: dict[tuple, dict] = {}
    with (ROOT / args.run_dir / "outcomes.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["geo_verify_status"] != "patch_similarity" or int(r["fine_inliers"]) < args.min_inliers:
                continue
            key = (r["region"], int(r["num"]))
            if key not in best or int(r["fine_inliers"]) > int(best[key]["fine_inliers"]):
                best[key] = r

    rows = []
    for (region, num), r in sorted(best.items()):
        lat, lon = man[(region, num)]
        off = enu(lat, lon, float(r["pred_lat"]), float(r["pred_lon"]))
        tv = track_vector(man, region, num)
        dist = float(np.hypot(*off))
        row = {"region": region, "filename": r["filename"], "fine_inliers": int(r["fine_inliers"]),
               "accepted": int(r["accepted"]), "offset_m": round(dist, 1), "offset_bearing_deg": round(bearing(off), 1),
               "track_bearing_deg": "", "along_track_m": "", "cross_track_m": "", "axis_diff_deg": ""}
        if tv is not None:
            t = tv / np.hypot(*tv)
            d = abs((bearing(off) - bearing(tv) + 180.0) % 360.0 - 180.0)
            row.update({"track_bearing_deg": round(bearing(tv), 1),
                        "along_track_m": round(float(off @ t), 1),
                        "cross_track_m": round(float(off[0] * t[1] - off[1] * t[0]), 1),
                        "axis_diff_deg": round(min(d, 180.0 - d), 1)})
        rows.append(row)

    out_csv = ROOT / "reports" / "gt_track_offset_audit.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_region = defaultdict(list)
    for r in rows:
        if r["along_track_m"] != "":
            by_region[r["region"]].append(r)
    lines = [
        "# Ground-truth track-offset audit (read-only)", "",
        f"Source: `{args.run_dir}/outcomes.csv`. One row per frame: its best patch fit with >= {args.min_inliers} "
        "inliers (accepted or not). Offset = CSV position -> predicted position, split into along-track "
        "(+ = ahead along the flight direction) and cross-track (+ = right of track) components. "
        f"'Far on-axis' = offset > {FAR_M:g} m within ±{AXIS_WINDOW_DEG:g}° of the track axis; chance share is 1/3.",
        "",
        "Labels are **not** corrected from this: using matcher output to move ground truth would be circular. "
        "Treat any signal as a ground-truth limitation to report.", "",
        "| region | frames | median along-track m (95% CI) | median abs cross-track m | far (>100 m) | far on-axis |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for region, rs in sorted(by_region.items()):
        al = np.array([r["along_track_m"] for r in rs], dtype=float)
        cr = np.abs(np.array([r["cross_track_m"] for r in rs], dtype=float))
        lo, hi = boot_median_ci(al)
        far = [r for r in rs if r["offset_m"] > FAR_M]
        on_axis = sum(r["axis_diff_deg"] <= AXIS_WINDOW_DEG for r in far)
        lines.append(f"| {region} | {len(rs)} | {np.median(al):.1f} ({lo:.1f}, {hi:.1f}) | {np.median(cr):.1f} | "
                     f"{len(far)} | {on_axis}/{len(far)} |")
    far_all = [r for rs in by_region.values() for r in rs if r["offset_m"] > FAR_M]
    lines += ["", f"All regions: {sum(r['axis_diff_deg'] <= AXIS_WINDOW_DEG for r in far_all)}/{len(far_all)} "
              "far offsets on the track axis (chance 1/3).", "",
              "Far offsets:", "",
              "| region | frame | inliers | accepted | offset m | offset bearing | track bearing | axis diff |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(far_all, key=lambda r: (r["region"], r["filename"])):
        lines.append(f"| {r['region']} | {r['filename']} | {r['fine_inliers']} | {r['accepted']} | {r['offset_m']} | "
                     f"{r['offset_bearing_deg']} | {r['track_bearing_deg']} | {r['axis_diff_deg']} |")
    lines += ["", "Per-frame rows: `reports/gt_track_offset_audit.csv`."]
    (ROOT / "reports" / "gt_track_offset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[8:8 + len(by_region) + 3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
