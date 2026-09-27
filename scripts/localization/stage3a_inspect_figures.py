#!/usr/bin/env python3
"""Regenerate Stage 3A inspection figures from frozen pilot CSV (no re-match)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(r"F:\CMBL")
sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from tile_index import georef_from_metadata, load_sat_memmap  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "matcher_protocol.yaml"
CSV_PATH = ROOT / "reports" / "stage3a_pilot_results.csv"
FIGS = ROOT / "figures" / "stage3a"
META = ROOT / "datasets" / "manifests" / "satellite_metadata.csv"


def lonlat_to_pixel(lon: float, lat: float, georef: tuple) -> tuple[float, float]:
    west, north, sx, sy = georef
    col = (lon - west) / sx
    row = (north - lat) / sy
    return row, col


def sat_row_for_region(meta_rows: list[dict], region: str) -> dict:
    for r in meta_rows:
        path = r["satellite_path"].replace("/", "\\")
        if f"\\{region}\\" in path and "satellite09_" not in r["filename"]:
            return r
    raise FileNotFoundError(region)


def patch_around(sat, row: float, col: float, half: int = 256) -> np.ndarray:
    r0 = max(0, int(row) - half)
    c0 = max(0, int(col) - half)
    return np.asarray(sat[r0 : r0 + 2 * half, c0 : c0 + 2 * half, :3])


def save_pair(query_path: Path, sat, georef, row: dict, out: Path, label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    qim = cv2.cvtColor(cv2.imread(str(query_path)), cv2.COLOR_BGR2RGB)
    axes[0].imshow(qim)
    axes[0].set_title(f"Query {row['filename']}\nerr={float(row['geodesic_error_m']):.1f} m ({label})")
    axes[0].axis("off")

    gt_r, gt_c = lonlat_to_pixel(float(row["gt_lon"]), float(row["gt_lat"]), georef)
    pred_r, pred_c = lonlat_to_pixel(float(row["pred_lon"]), float(row["pred_lat"]), georef)

    axes[1].imshow(patch_around(sat, gt_r, gt_c))
    axes[1].set_title("Map @ CSV GT (scoring only)")
    axes[1].axis("off")

    axes[2].imshow(patch_around(sat, pred_r, pred_c))
    axes[2].set_title("Map @ matcher prediction")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    region = cfg["leakage_and_splits"]["pilot_region"]

    with META.open(encoding="utf-8", newline="") as f:
        meta_rows = list(csv.DictReader(f))
    sat_meta = sat_row_for_region(meta_rows, region)
    georef = georef_from_metadata(sat_meta)
    sat, _ = load_sat_memmap(Path(sat_meta["satellite_path"]))

    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    FIGS.mkdir(parents=True, exist_ok=True)
    errs = [float(r["geodesic_error_m"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errs, bins=30, color="steelblue", edgecolor="white")
    for th, ls in ((100, "--"), (500, ":")):
        ax.axvline(th, color="red" if th == 100 else "darkorange", ls=ls, label=f"{th} m")
    ax.set_xlabel("Geodesic error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Stage 3A pilot — region 03 (dev only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "error_histogram.png", dpi=150)
    plt.close(fig)

    # Clear old example_* panels so filenames match labels
    for old in FIGS.glob("example_*.png"):
        old.unlink()

    ranked = sorted(rows, key=lambda r: float(r["geodesic_error_m"]))
    successes_500 = [r for r in ranked if int(r["success_500m"]) == 1]
    failures = [r for r in ranked if int(r["success_500m"]) == 0]

    panels = []
    for i, r in enumerate(successes_500[:2], start=1):
        panels.append((r, "success_500m", f"example_{i}_success_500m.png"))
    # Distinct mid-band failures (exclude the two success rows)
    mid = [r for r in ranked if 500 < float(r["geodesic_error_m"]) < 2000][:2]
    for j, r in enumerate(mid, start=len(panels) + 1):
        panels.append((r, "failure_mid", f"example_{j}_failure.png"))
    # Near-miss just above 500 m (if not already used) and catastrophic tail
    used = {r["filename"] for r, _, _ in panels}
    near = next(
        r
        for r in ranked
        if 500 < float(r["geodesic_error_m"]) < 700 and r["filename"] not in used
    )
    worst = ranked[-1]
    panels.append((near, "near_miss", f"example_{len(panels)+1}_near_miss.png"))
    panels.append((worst, "catastrophic_failure", f"example_{len(panels)+1}_failure.png"))

    drone_root = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset" / region / "drone"
    for r, label, name in panels:
        save_pair(drone_root / r["filename"], sat, georef, r, FIGS / name, label)
        print("wrote", name, r["filename"], r["geodesic_error_m"], label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
