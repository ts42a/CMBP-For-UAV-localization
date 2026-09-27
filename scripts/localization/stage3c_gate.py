#!/usr/bin/env python3
"""Stage 3C preflight gate: untouched Stage 3B val CSV only.

Human-reviewed scene matches are NOT used as the 100 m accuracy criterion.
CSV coordinates may appear only as scoring columns.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from geo_utils import geodesic_m  # noqa: E402

VAL = ROOT / "reports" / "stage3b_val_dinov2_c2f.csv"
FROZEN = ROOT / "configs" / "matcher_stage3b_frozen.yaml"
OUT = ROOT / "reports" / "stage3c_preflight_gate.json"
OUT_MD = ROOT / "reports" / "stage3c_preflight_gate.md"


def main() -> int:
    causes: list[str] = []
    if not VAL.is_file():
        causes.append(f"missing val CSV: {VAL}")
    if not FROZEN.is_file():
        causes.append(f"missing frozen config: {FROZEN}")

    rows: list[dict] = []
    if VAL.is_file():
        rows = list(csv.DictReader(VAL.open(encoding="utf-8")))

    n = len(rows)
    recomputed_ok = 0
    s100 = 0
    f100 = 0
    search_gt = 0
    loc_from_match = 0
    csv_copied = 0
    homography_succ = 0
    tile_center_succ = 0

    for r in rows:
        plat, plon = float(r["pred_lat"]), float(r["pred_lon"])
        glat, glon = float(r["gt_lat"]), float(r["gt_lon"])
        err = geodesic_m(glon, glat, plon, plat)
        reported = float(r["geodesic_error_m"])
        if abs(err - reported) <= 1.0:
            recomputed_ok += 1
        else:
            causes.append(
                f"{r['filename']}: recomputed error {err:.3f} != reported {reported:.3f}"
            )
        flag = int(err <= 100)
        if flag != int(r["success_100m"]):
            causes.append(f"{r['filename']}: success_100m flag disagrees with recomputed error")
        if flag:
            s100 += 1
        else:
            f100 += 1
        if str(r.get("search_used_gt_position", "")).lower() in {"true", "1"}:
            search_gt += 1
        # Location must come from tile centre or homography, not a copy of CSV.
        same_csv = abs(plat - glat) < 1e-10 and abs(plon - glon) < 1e-10
        if same_csv:
            csv_copied += 1
        used_h = int(r.get("used_homography") or 0)
        has_tile = str(r.get("predicted_tile_id", "")).strip() not in {"", "-1"}
        if used_h or has_tile:
            loc_from_match += 1
        if flag and used_h:
            homography_succ += 1
        if flag and not used_h and has_tile:
            tile_center_succ += 1

    if n != 30:
        causes.append(f"expected n=30 untouched val rows, got {n}")
    if recomputed_ok != n:
        causes.append("geodesic recompute mismatch")
    if search_gt:
        causes.append(f"{search_gt} rows used CSV position during search")
    if csv_copied:
        causes.append(f"{csv_copied} predictions are exact copies of CSV lat/lon")
    if loc_from_match != n:
        causes.append("some predictions lack tile id and homography (not match-derived)")
    if not (s100 >= 1 and f100 >= 1):
        causes.append(f"no mix at 100 m: success={s100} failure={f100}")

    passed = not causes
    payload = {
        "gate": "PASS" if passed else "FAIL",
        "decision": "READY" if passed else "NOT READY",
        "val_csv": str(VAL),
        "n": n,
        "success_100m": s100,
        "failure_100m": f100,
        "recomputed_geodesic_matches": recomputed_ok,
        "search_used_gt_position_count": search_gt,
        "exact_csv_copy_count": csv_copied,
        "location_from_match_or_geo_verify": loc_from_match,
        "successes_with_homography": homography_succ,
        "successes_with_tile_center": tile_center_succ,
        "human_review_not_used_as_100m_criterion": True,
        "causes": causes,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Stage 3C preflight gate",
        "",
        f"**Decision: {payload['decision']}**",
        "",
        f"- Untouched val CSV: `{VAL.name}` n={n}",
        f"- Recomputed success@100 m: **{s100}** / failure: **{f100}**",
        f"- Search used CSV position: **{search_gt}** (must be 0)",
        f"- Predictions that are exact CSV copies: **{csv_copied}** (must be 0)",
        f"- Location from tile match or homography: **{loc_from_match}/{n}**",
        f"- Among @100 m successes: homography={homography_succ}, tile-centre={tile_center_succ}",
        "- Human scene review is **not** the 100 m criterion; geodesic error vs scoring columns is.",
        "",
    ]
    if causes:
        lines.append("## Causes")
        lines.extend(f"- {c}" for c in causes)
    else:
        lines.append("Gate passed. Full-map labeling may proceed with the frozen matcher.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
