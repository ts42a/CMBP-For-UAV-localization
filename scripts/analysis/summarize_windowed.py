#!/usr/bin/env python3
"""Summarise windowed outcomes against the pre-registered Stage 5 checks, plus a gallery.

  py -3 scripts/analysis/summarize_windowed.py results/stage5/pilot --name stage5_pilot
Writes reports/<name>_summary.{md,json} and figures/<name>/ gallery panels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))
from project_paths import DATASET_ROOT  # noqa: E402
from region_map import load_region, read_tile_rgb_region  # noqa: E402
from tile_index import TileSpec  # noqa: E402

OUTCOMES = ("accepted_correct", "rejected", "accepted_incorrect")
OFFSET_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 99.0)]


def rate(rs: list[dict], outcome: str) -> float | None:
    return round(sum(r["outcome_primary"] == outcome for r in rs) / len(rs), 4) if rs else None


def breakdown(rs: list[dict]) -> dict:
    d = {"n": len(rs)}
    for o in OUTCOMES:
        d[o] = sum(r["outcome_primary"] == o for r in rs)
    d["accepted_correct_rate"] = rate(rs, "accepted_correct")
    d["accepted_incorrect_rate"] = rate(rs, "accepted_incorrect")
    acc = d["accepted_correct"] + d["accepted_incorrect"]
    d["precision"] = round(d["accepted_correct"] / acc, 4) if acc else None
    return d


def group_by(rs: list[dict], key) -> dict:
    g = defaultdict(list)
    for r in rs:
        g[key(r)].append(r)
    return {k: breakdown(v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


def secondary(rs: list[dict], tols: list[float]) -> dict:
    out = {}
    for t in tols:
        ok = sum(r["accepted"] == "1" and r["geodesic_error_m"] != "" and float(r["geodesic_error_m"]) <= t for r in rs)
        out[f"{t:g}m"] = round(ok / len(rs), 4) if rs else None
    return out


def thumb(img: np.ndarray, h: int = 360) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def latlon_to_px(t, lat: float, lon: float) -> tuple[int, int]:
    col = (lon - t.west) / t.sx - t.col0
    row = (t.north - lat) / t.sy - t.row0
    return int(col), int(row)


def gallery(rs: list[dict], out_dir: Path, per_type: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = []
    by_type = defaultdict(list)
    for r in rs:
        by_type[r["outcome_primary"]].append(r)
    for o in OUTCOMES:
        lst = sorted(by_type[o], key=lambda r: (r["region"], r["filename"], int(r["radius_m"]), int(r["offset_k"])))
        if o == "accepted_incorrect":
            picks += lst
        else:
            step = max(1, len(lst) // per_type) if lst else 1
            picks += lst[::step][:per_type]
    written = []
    cache = {}
    spec = TileSpec(1024, 512)
    for r in picks:
        reg = r["region"]
        if reg not in cache:
            cache.clear()
            cache[reg] = load_region(reg, spec)
        rasters, tiles = cache[reg]
        q = cv2.imread(str(DATASET_ROOT / reg / "drone" / r["filename"]))
        panels = [thumb(q)]
        mlat = float(r["manifest_lat"])
        mlon = float(r["manifest_lon"])
        if r["predicted_tile_id"] != "":
            t = next(t for t in tiles if t.tile_id == int(r["predicted_tile_id"]))
            img = cv2.cvtColor(read_tile_rgb_region(rasters, t), cv2.COLOR_RGB2BGR).copy()
            px, py = latlon_to_px(t, float(r["pred_lat"]), float(r["pred_lon"]))
            cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 60, 6)
            tx, ty = latlon_to_px(t, mlat, mlon)
            if 0 <= tx < img.shape[1] and 0 <= ty < img.shape[0]:
                cv2.circle(img, (tx, ty), 30, (0, 255, 0), 6)
            panels.append(thumb(img))
        lat = np.array([t.center_lat for t in tiles])
        lon = np.array([t.center_lon for t in tiles])
        tt = tiles[int(np.argmin((lat - mlat) ** 2 + ((lon - mlon) * np.cos(np.radians(mlat))) ** 2))]
        img = cv2.cvtColor(read_tile_rgb_region(rasters, tt), cv2.COLOR_RGB2BGR).copy()
        tx, ty = latlon_to_px(tt, mlat, mlon)
        cv2.circle(img, (tx, ty), 30, (0, 255, 0), 6)
        panels.append(thumb(img))
        canvas = np.hstack(panels)
        bar = np.full((40, canvas.shape[1], 3), 255, np.uint8)
        label = (f"{r['outcome_primary']} | {r['filename']} R={r['radius_m']} k={r['offset_k']} "
                 f"err={r['geodesic_error_m']} inl={r['fine_inliers']} | query | pred tile (red=pred, green=truth) | truth tile")
        cv2.putText(bar, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        name = f"{r['outcome_primary']}_{r['filename'][:-4]}_R{r['radius_m']}_k{r['offset_k']}.jpg"
        cv2.imwrite(str(out_dir / name), np.vstack([bar, canvas]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(name)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gallery-per-type", type=int, default=4)
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()

    prereg = yaml.safe_load((ROOT / "configs" / "preregistration_v1.yaml").read_text(encoding="utf-8"))
    band = prereg["stage5_go"]["prevalence_band"]
    rs = list(csv.DictReader((ROOT / args.run_dir / "outcomes.csv").open(encoding="utf-8", newline="")))
    if any(r["outcome_primary"] == "" for r in rs):
        raise SystemExit("rows without outcome_primary: accept rule was not selected when they were generated")
    hashes = {(r["config_sha256"], r["prereg_sha256"]) for r in rs}

    man = {}
    with (ROOT / "datasets" / "manifests" / "manifest.csv").open(encoding="utf-8", newline="") as f:
        for m in csv.DictReader(f):
            man[(m["region_folder"], m["filename"])] = (m["lat"], m["lon"])
    for r in rs:
        r["manifest_lat"], r["manifest_lon"] = man[(r["region"], r["filename"])]

    for r in rs:
        rel = float(r["offset_dist_m"]) / float(r["radius_m"])
        r["_bin"] = next(f"{a:g}-{b:g}R" for a, b in OFFSET_BINS if a <= rel < b)

    pooled = breakdown(rs)
    by_bin = group_by(rs, lambda r: r["_bin"])
    bin_rates = [by_bin[f"{a:g}-{b:g}R"]["accepted_correct_rate"] for a, b in OFFSET_BINS if f"{a:g}-{b:g}R" in by_bin]
    monotonic = all(bin_rates[i] >= bin_rates[i + 1] for i in range(len(bin_rates) - 1))
    by_region = group_by(rs, lambda r: r["region"])
    in_band_regions = sum(band[0] <= v["accepted_correct_rate"] <= band[1] for v in by_region.values())
    prev = pooled["accepted_correct_rate"]

    summary = {
        "run_dir": args.run_dir,
        "n_rows": len(rs),
        "n_frames": len({(r["region"], r["filename"]) for r in rs}),
        "regions": sorted({r["region"] for r in rs}),
        "single_config_and_prereg_hash": len(hashes) == 1,
        "accept_rule": sorted({r["accept_rule"] for r in rs}),
        "pooled": pooled,
        "secondary_tolerances_accepted_correct_rate": secondary(rs, [float(t) for t in prereg["tolerances_m"]["secondary"]]),
        "by_radius": group_by(rs, lambda r: int(r["radius_m"])),
        "by_offset_bin": by_bin,
        "by_truth_within_radius": group_by(rs, lambda r: int(r["truth_within_radius"])),
        "by_region": by_region,
        "by_group_7": group_by(rs, lambda r: r["group_7"]),
        "checks": {
            "prevalence_in_band": band[0] <= prev <= band[1],
            "prevalence_band": band,
            "accepted_correct_nonincreasing_with_offset": monotonic,
            "regions_in_band": f"{in_band_regions}/{len(by_region)}",
            "all_three_outcomes_present": all(pooled[o] > 0 for o in OUTCOMES),
        },
    }
    summary["checks"]["stage5_go"] = (summary["checks"]["prevalence_in_band"] and monotonic
                                      and summary["checks"]["all_three_outcomes_present"])
    if not args.no_gallery:
        summary["gallery"] = gallery(rs, ROOT / "figures" / args.name, args.gallery_per_type)

    (ROOT / "reports" / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def tbl(title: str, d: dict) -> list[str]:
        out = [f"### {title}", "", "| | n | acc. correct | rejected | acc. incorrect | acc.-correct rate | precision |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for k, v in d.items():
            out.append(f"| {k} | {v['n']} | {v['accepted_correct']} | {v['rejected']} | {v['accepted_incorrect']} | "
                       f"{v['accepted_correct_rate']} | {v['precision']} |")
        return out + [""]

    md = [f"# {args.name} — windowed outcome summary", "",
          f"Rows {len(rs)}, frames {summary['n_frames']}, regions {', '.join(summary['regions'])}. "
          f"Accept rule `{summary['accept_rule']}`; primary tolerance {prereg['tolerances_m']['primary']} m.", "",
          "## Checks", "", "```json", json.dumps(summary["checks"], indent=2), "```", ""]
    md += tbl("Pooled", {"all": pooled})
    md += [f"Secondary tolerances (accepted-correct rate): {summary['secondary_tolerances_accepted_correct_rate']}", ""]
    md += tbl("By radius (m)", summary["by_radius"])
    md += tbl("By prior offset / radius", by_bin)
    md += tbl("By truth inside window", summary["by_truth_within_radius"])
    md += tbl("By region", by_region)
    md += tbl("By 7-group", summary["by_group_7"])
    if not args.no_gallery:
        md += [f"Gallery: `figures/{args.name}/` ({len(summary['gallery'])} panels; every accepted-incorrect row included)."]
    (ROOT / "reports" / f"{args.name}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary["checks"], indent=2))
    print(json.dumps(pooled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
