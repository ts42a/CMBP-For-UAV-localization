#!/usr/bin/env python3
"""Render review panels for accepted fixes: query | map at prediction | map at truth.

Patch-verified queries are rotated into map orientation using the fitted similarity
angle (it maps unrotated query pixels onto the north-up crop). Rows are de-duplicated by frame and predicted position
(~10 m grid) so repeated identical fixes across radii/offsets are reviewed once.
Writes PNG panels plus <out>/review_template.csv.

  py -3 scripts/analysis/verify_panels.py results/.../outcomes.csv --rule patch_20 --label patch_s2 --out figures/verifier_cmp/patch_s2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "localization"))
from label_windowed import is_accepted  # noqa: E402
from project_paths import DATASET_ROOT, MANIFEST  # noqa: E402
from region_map import load_region  # noqa: E402
from tile_index import TileSpec  # noqa: E402

H = 420
WIDTH_M = 700.0


def fit_h(img: np.ndarray, h: int = H) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def rotate_expand(img: np.ndarray, ang: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(255, 255, 255))


def map_crop(rasters, lat: float, lon: float) -> np.ndarray | None:
    for r in rasters:
        row = (r.north - lat) / r.sy
        col = (lon - r.west) / r.sx
        if 0 <= row < r.height and 0 <= col < r.width:
            m_per_px = r.sy * 110_574.0
            half = int(WIDTH_M / m_per_px / 2)
            a0, b0 = int(row) - half, int(col) - half
            out = np.full((2 * half, 2 * half, 3), 255, np.uint8)
            ra0, rb0 = max(0, a0), max(0, b0)
            ra1, rb1 = min(r.height, a0 + 2 * half), min(r.width, b0 + 2 * half)
            out[ra0 - a0 : ra1 - a0, rb0 - b0 : rb1 - b0] = np.asarray(r.sat[ra0:ra1, rb0:rb1, :3])
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            step = max(1, out.shape[0] // 700)
            return np.ascontiguousarray(out[::step, ::step])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outcomes")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=100.0)
    args = ap.parse_args()

    rows = list(csv.DictReader((ROOT / args.outcomes).open(encoding="utf-8", newline="")))
    truth = {(m["region_folder"], m["filename"]): (float(m["lat"]), float(m["lon"]))
             for m in csv.DictReader(MANIFEST.open(encoding="utf-8", newline=""))}
    seen, picks = set(), []
    for r in rows:
        if not is_accepted(args.rule, r):
            continue
        key = (r["region"], r["filename"], round(float(r["pred_lat"]), 4), round(float(r["pred_lon"]), 4))
        if key not in seen:
            seen.add(key)
            picks.append(r)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    regions = {}
    review = []
    for r in picks:
        reg = r["region"]
        if reg not in regions:
            regions[reg] = load_region(reg, TileSpec(1024, 512))[0]
        rasters = regions[reg]
        q = cv2.imread(str(DATASET_ROOT / reg / "drone" / r["filename"]))
        q = cv2.resize(q, (q.shape[1] // 4, q.shape[0] // 4), interpolation=cv2.INTER_AREA)
        if r.get("patch_query_rot_deg"):
            # the similarity maps unrotated query pixels onto the north-up map crop
            ang = -float(r["patch_sim_angle_deg"])
            q = rotate_expand(q, ang)
            qlab = f"query rotated {ang:.0f} deg (fitted)"
        else:
            qlab = "query (unrotated)"
        tlat, tlon = truth[(reg, r["filename"])]
        plat, plon = float(r["pred_lat"]), float(r["pred_lon"])
        mp = map_crop(rasters, plat, plon)
        mt = map_crop(rasters, tlat, tlon)
        panels = [fit_h(q)]
        for m, col in ((mp, (0, 0, 255)), (mt, (0, 200, 0))):
            if m is None:
                panels.append(np.full((H, H, 3), 200, np.uint8))
                continue
            c = (m.shape[1] // 2, m.shape[0] // 2)
            if col == (0, 0, 255):
                cv2.drawMarker(m, c, col, cv2.MARKER_CROSS, 40, 4)
            else:
                cv2.circle(m, c, 22, col, 4)
            panels.append(fit_h(m))
        body = np.hstack([np.pad(p, ((0, 0), (0, 6), (0, 0)), constant_values=255) for p in panels])
        err = float(r["geodesic_error_m"])
        outcome = "accepted_correct" if err <= args.tol else "accepted_incorrect"
        bar = np.full((52, body.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{args.label} {args.rule} | {outcome} | {r['filename']} R={r['radius_m']} k={r['offset_k']} "
                    f"err={err:.0f} m inliers={r['fine_inliers']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bar, f"{qlab} | map {WIDTH_M:.0f} m at PREDICTION (red +) | map {WIDTH_M:.0f} m at TRUTH (green o), north up",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        name = f"{outcome}_{r['filename'][:-4]}_R{r['radius_m']}_k{r['offset_k']}.jpg"
        cv2.imwrite(str(out / name), np.vstack([bar, body]), [cv2.IMWRITE_JPEG_QUALITY, 82])
        review.append({"verifier": args.label, "rule": args.rule, "filename": r["filename"], "radius_m": r["radius_m"],
                       "offset_k": r["offset_k"], "outcome": outcome, "error_m": round(err, 1),
                       "inliers": r["fine_inliers"], "panel": f"{args.out}/{name}", "scene_match": "", "note": ""})
    with (out / "review_template.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0]) if review else ["verifier"])
        w.writeheader()
        w.writerows(review)
    print(f"{len(picks)} unique accepted fixes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
