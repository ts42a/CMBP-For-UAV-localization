#!/usr/bin/env python3
"""Stage 4 diagnosis (dev region 03, tune frames only): does AKAZE verification fail
because the 1024 px (~300 m) tile is much larger than the drone footprint?

For each tune frame, run the frozen AKAZE+RANSAC settings against truth-centred
satellite crops of several widths, each resampled to 1024 px. Diagnostic only; nothing
here is used for labelling. Writes reports/stage4_verify_scale_diagnosis.{csv,md}.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))
from fine_verify import query_features  # noqa: E402
from frozen_matcher import load_frozen  # noqa: E402
from project_paths import DATASET_ROOT  # noqa: E402
from query_select import load_manifest_region  # noqa: E402
from region_map import load_region  # noqa: E402
from tile_index import TileSpec  # noqa: E402

WIDTHS_M = [60, 150, 300, 600, 900, 1200, 1800]
GSD_M = 0.276


def inliers(qfeat, crop_gray: np.ndarray, p: dict) -> int:
    _, kp_q, des_q = qfeat
    ak = cv2.AKAZE_create()
    kp_t, des_t = ak.detectAndCompute(crop_gray, None)
    if des_q is None or des_t is None:
        return 0
    good = [m for pr in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_q, des_t, k=2)
            if len(pr) == 2 and pr[0].distance < p["ratio_test"] * pr[1].distance for m in [pr[0]]]
    if len(good) < 8:
        return 0
    src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, p["ransac_reproj_threshold_px"])
    return int(mask.sum()) if mask is not None else 0


def main() -> int:
    frozen = load_frozen()
    p = frozen["fine_match"]
    sp = yaml.safe_load((ROOT / "configs" / "stage3b_splits.yaml").read_text(encoding="utf-8"))
    names = {q["filename"] for q in sp["tune_queries"]}
    frames = [r for r in load_manifest_region("03") if r["filename"] in names]
    rasters, _ = load_region("03", TileSpec(1024, 512))
    r0 = rasters[0]
    rows = []
    for q in frames:
        bgr = cv2.imread(str(DATASET_ROOT / "03" / "drone" / q["filename"]))
        qfeat = query_features(bgr, p)
        cr = (r0.north - float(q["lat"])) / r0.sy
        cc = (float(q["lon"]) - r0.west) / r0.sx
        row = {"filename": q["filename"]}
        for w in WIDTHS_M:
            half = int(round(w / GSD_M / 2))
            a, b = int(cr) - half, int(cc) - half
            if a < 0 or b < 0 or a + 2 * half > r0.height or b + 2 * half > r0.width:
                row[f"inl_{w}m"] = ""
                continue
            crop = np.asarray(r0.sat[a:a + 2 * half, b:b + 2 * half, :3])
            g = cv2.cvtColor(cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_AREA if 2 * half > 1024 else cv2.INTER_CUBIC), cv2.COLOR_RGB2GRAY)
            row[f"inl_{w}m"] = inliers(qfeat, g, p)
        rows.append(row)
        print(row, flush=True)

    out = ROOT / "reports" / "stage4_verify_scale_diagnosis.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["# Stage 4 diagnosis: verification vs satellite crop width (region 03 tune frames)", "",
          "Truth-centred crops resampled to 1024 px; frozen AKAZE/RANSAC settings; min_inliers = 12.", "",
          "| Crop width | frames | median inliers | frames >= 12 inliers |", "|---:|---:|---:|---:|"]
    for w_ in WIDTHS_M:
        v = [r[f"inl_{w_}m"] for r in rows if r[f"inl_{w_}m"] != ""]
        md.append(f"| {w_} m | {len(v)} | {int(np.median(v)) if v else ''} | {sum(x >= 12 for x in v)} |")
    md += ["", "300 m is the current tile footprint. Diagnostic only; truth-centred crops are not a localizer."]
    (ROOT / "reports" / "stage4_verify_scale_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
