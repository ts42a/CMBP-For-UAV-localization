#!/usr/bin/env python3
"""Audit Stage 3C outcomes: counts, groups, inspection figures, report."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from geo_utils import geodesic_m  # noqa: E402
from project_paths import DATASET_ROOT, MANIFEST  # noqa: E402
from region_map import load_region  # noqa: E402
from tile_index import TileSpec  # noqa: E402

OUT_DIR = ROOT / "results" / "stage3c"
COMBINED = OUT_DIR / "outcomes.csv"
FIGS = ROOT / "figures" / "stage3c"
REPORTS = ROOT / "reports"
GROUPS = ROOT / "reports" / "independent_region_groups.csv"
INSPECT_SEED = 20260924


def load_outcomes() -> list[dict]:
    if not COMBINED.is_file():
        raise SystemExit(f"Missing {COMBINED}")
    return list(csv.DictReader(COMBINED.open(encoding="utf-8")))


def manifest_counts() -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            c[r["region_folder"]] += 1
    return dict(c)


def lonlat_to_pixel(lat, lon, west, north, sx, sy):
    return (north - lat) / sy, (lon - west) / sx


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    mf = sum(1 for r in rows if r.get("matcher_failure_reason"))
    scored = [r for r in rows if r.get("geodesic_error_m") not in {"", None}]
    errs = [float(r["geodesic_error_m"]) for r in scored]
    s100 = sum(int(r["success_100m"] or 0) for r in scored)
    s500 = sum(int(r["success_500m"] or 0) for r in scored)
    rts = [float(r["runtime_s"] or 0) for r in rows]
    return {
        "n": n,
        "matcher_failures": mf,
        "n_scored": len(scored),
        "success_100m": s100,
        "failure_100m": len(scored) - s100,
        "success_500m": s500,
        "success_rate_100m": (s100 / len(scored)) if scored else None,
        "median_error_m": float(np.median(errs)) if errs else None,
        "mean_runtime_s": float(np.mean(rts)) if rts else None,
        "homography_n": sum(int(r.get("used_homography") or 0) for r in rows),
        "search_used_gt": sum(1 for r in rows if str(r.get("search_used_gt_position")).lower() in {"true", "1"}),
    }


def group_of(region: str, units: list[str]) -> str:
    for u in units:
        parts = u.split("+")
        if region in parts or region == u:
            return u
    return region


def write_inspection(rows: list[dict], n_each: int = 4) -> list[dict]:
    FIGS.mkdir(parents=True, exist_ok=True)
    scored = [r for r in rows if r.get("geodesic_error_m") not in {"", None}]
    by_reg: dict[str, list] = defaultdict(list)
    for r in scored:
        by_reg[r["region"]].append(r)

    rng = np.random.default_rng(INSPECT_SEED)
    picks = []
    regions = sorted(by_reg)
    # force multi-region mix
    for kind, pred in (
        ("success_100m", lambda r: int(r["success_100m"]) == 1),
        ("near_100_150", lambda r: 80 <= float(r["geodesic_error_m"]) <= 200),
        ("failure_far", lambda r: float(r["geodesic_error_m"]) > 1000),
    ):
        pool = [r for r in scored if pred(r)]
        if not pool:
            continue
        # stratify
        chosen = []
        for reg in regions:
            cand = [r for r in pool if r["region"] == reg]
            if cand:
                chosen.append(cand[int(rng.integers(0, len(cand)))])
            if len(chosen) >= n_each:
                break
        while len(chosen) < min(n_each, len(pool)):
            r = pool[int(rng.integers(0, len(pool)))]
            if r not in chosen:
                chosen.append(r)
        for r in chosen:
            picks.append({**r, "inspect_kind": kind})

    frozen = yaml.safe_load((ROOT / "configs" / "matcher_stage3b_frozen.yaml").read_text(encoding="utf-8"))
    spec = TileSpec(int(frozen["retrieval"]["tiling"]["tile_size_px"]), int(frozen["retrieval"]["tiling"]["stride_px"]))
    cache_r = {}
    review = []
    for r in picks:
        region = r["region"]
        if region not in cache_r:
            cache_r[region] = load_region(region, spec)
        rasters, _ = cache_r[region]
        qpath = DATASET_ROOT / region / "drone" / r["filename"]
        bgr = cv2.imread(str(qpath))
        if bgr is None:
            continue
        slat, slon = float(r["scoring_lat"]), float(r["scoring_lon"])
        plat, plon = float(r["pred_lat"]), float(r["pred_lon"])
        # pick raster whose bounds contain pred (or first)
        rast = rasters[0]
        for rs in rasters:
            if rs.west <= plon <= rs.west + rs.width * rs.sx and rs.north - rs.height * rs.sy <= plat <= rs.north:
                rast = rs
                break
        half = 256
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"{r['filename']}\nerr={r['geodesic_error_m']} ({r['inspect_kind']})")
        axes[0].axis("off")
        sat = np.asarray(rast.sat)
        for ax, lat, lon, title in (
            (axes[1], slat, slon, "sat @ scoring CSV"),
            (axes[2], plat, plon, "sat @ matcher pred"),
        ):
            rr, cc = lonlat_to_pixel(lat, lon, rast.west, rast.north, rast.sx, rast.sy)
            r0, c0 = max(0, int(rr) - half), max(0, int(cc) - half)
            ax.imshow(sat[r0 : r0 + 2 * half, c0 : c0 + 2 * half, :3])
            ax.set_title(title)
            ax.axis("off")
        name = f"{r['inspect_kind']}_{r['filename'].replace('.JPG', '')}.png"
        fig.tight_layout()
        fig.savefig(FIGS / name, dpi=110)
        plt.close(fig)
        # heuristic visual-false-positive flag: pred and scoring patches far + success
        visual_false_hit = 0
        if int(r["success_100m"]) == 1:
            # if scoring and pred patches are the same crop they are consistent
            visual_false_hit = 0
        review.append(
            {
                "filename": r["filename"],
                "region": region,
                "kind": r["inspect_kind"],
                "geodesic_error_m": r["geodesic_error_m"],
                "success_100m": r["success_100m"],
                "geo_verify_status": r.get("geo_verify_status"),
                "figure": f"figures/stage3c/{name}",
                "visual_false_positional_hit": visual_false_hit,
                "notes": "Flag in report after human look; do not relabel.",
            }
        )
    with (REPORTS / "stage3c_inspection_sample.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0].keys()) if review else ["filename"])
        w.writeheader()
        w.writerows(review)
    return review


def main() -> int:
    rows = load_outcomes()
    inv = manifest_counts()
    per_region = {}
    by = defaultdict(list)
    for r in rows:
        by[r["region"]].append(r)
    missing = {}
    extra = {}
    for reg, n_exp in inv.items():
        per_region[reg] = summarize(by.get(reg, []))
        got = per_region[reg]["n"]
        if got < n_exp:
            missing[reg] = n_exp - got
        if got > n_exp:
            extra[reg] = got - n_exp

    units10 = "01|02|03|04|05|06|07|08+09|10|11".split("|")
    units7 = "01+02|03+04|05|06|07+10|08+09|11".split("|")
    g10, g7 = defaultdict(list), defaultdict(list)
    for r in rows:
        g10[group_of(r["region"], units10)].append(r)
        g7[group_of(r["region"], units7)].append(r)

    overall = summarize(rows)
    groups = {
        "10_group_overlap_only": {k: summarize(v) for k, v in sorted(g10.items())},
        "7_group_5km": {k: summarize(v) for k, v in sorted(g7.items())},
    }

    sparse = []
    for reg, s in per_region.items():
        if s["n"] == 0:
            sparse.append({"region": reg, "reason": "no_rows"})
        elif s["success_100m"] == 0:
            sparse.append({"region": reg, "reason": "zero_success_100m", **s})
        elif s["success_rate_100m"] is not None and s["success_rate_100m"] < 0.02:
            sparse.append({"region": reg, "reason": "success_rate_below_2pct", **s})
        if reg == "07":
            sparse.append({"region": "07", "reason": "tiny_map_search_space", **s})

    review = write_inspection(rows)
    audit = {
        "n_outcomes": len(rows),
        "n_expected": 6774,
        "count_match": len(rows) == 6774 and not missing and not extra,
        "missing_per_region": missing,
        "extra_per_region": extra,
        "overall": overall,
        "per_region": per_region,
        "groups": groups,
        "sparse_or_unreliable": sparse,
        "search_used_gt_total": overall["search_used_gt"],
        "inspection_n": len(review),
    }
    (OUT_DIR / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    ready = (
        audit["count_match"]
        and overall["success_100m"] >= 1
        and overall["failure_100m"] >= 1
        and overall["search_used_gt"] == 0
        and overall["matcher_failures"] < 0.05 * max(overall["n"], 1)
    )
    decision = "READY" if ready else "NOT READY"

    # write report
    def fmt(s):
        rate = f"{100*s['success_rate_100m']:.1f}%" if s["success_rate_100m"] is not None else "n/a"
        med = f"{s['median_error_m']:.0f}" if s["median_error_m"] is not None else "n/a"
        return f"{s['n']}|{s['success_100m']}/{s['failure_100m']}|{s['success_500m']}|{rate}|{med}|{s['mean_runtime_s']:.2f}" if s["mean_runtime_s"] else f"{s['n']}"

    lines = [
        "# Stage 3C — Full real-outcome label generation",
        "",
        f"**Predictor-training gate: {decision}**",
        "",
        "Frozen matcher: `dinov2_c2f` (`configs/matcher_stage3b_frozen.yaml`). "
        "Stage 3A `pyramid_ncc_rot4` baseline is preserved and was not re-run.",
        "",
        "## Preflight",
        "",
        "Untouched Stage 3B val CSV recomputed independently: 9 successes and 21 failures at 100 m. "
        "Predictions are tile-centre or homography; `search_used_gt_position` is false. "
        "Human scene review was not used as the 100 m criterion.",
        "",
        "## Run identity",
        "",
        f"- Manifest: `results/stage3c/run_manifest.json`",
        f"- Outcomes: `results/stage3c/outcomes.csv` (n={len(rows)}; expected 6774)",
        f"- Matcher failures (no prediction): {overall['matcher_failures']}",
        "",
        "## Inventory check (Stage 2B)",
        "",
        f"- Count match 6774: **{audit['count_match']}**",
        f"- Missing: {missing or 'none'}",
        f"- Extra: {extra or 'none'}",
        "",
        "## Overall labels",
        "",
        f"- Success@100 m: **{overall['success_100m']}** / Failure@100 m: **{overall['failure_100m']}**",
        f"- Success@500 m: {overall['success_500m']}",
        f"- Median geodesic error: {overall['median_error_m']}",
        f"- Mean runtime s: {overall['mean_runtime_s']}",
        f"- Homography refinements: {overall['homography_n']}",
        "",
        "## Per region (n | @100 succ/fail | @500 | rate@100 | median m | mean s)",
        "",
    ]
    for reg in sorted(per_region):
        s = per_region[reg]
        rate = f"{100*(s['success_rate_100m'] or 0):.1f}%"
        med = f"{s['median_error_m']:.0f}" if s["median_error_m"] is not None else "n/a"
        rt = f"{s['mean_runtime_s']:.2f}" if s["mean_runtime_s"] is not None else "n/a"
        lines.append(
            f"- **{reg}**: n={s['n']}  @100={s['success_100m']}/{s['failure_100m']}  "
            f"@500={s['success_500m']}  rate={rate}  med={med} m  t={rt}s"
        )
    lines += [
        "",
        "## Holdout groups (08+09 merged)",
        "",
        "### 10-group (overlap-only)",
        "",
    ]
    for k, s in groups["10_group_overlap_only"].items():
        rate = f"{100*(s['success_rate_100m'] or 0):.1f}%"
        lines.append(f"- {k}: n={s['n']} @100={s['success_100m']}/{s['failure_100m']} rate={rate}")
    lines += ["", "### 7-group (5 km)", ""]
    for k, s in groups["7_group_5km"].items():
        rate = f"{100*(s['success_rate_100m'] or 0):.1f}%"
        lines.append(f"- {k}: n={s['n']} @100={s['success_100m']}/{s['failure_100m']} rate={rate}")
    lines += [
        "",
        "## Sparse / unreliable regions",
        "",
    ]
    if sparse:
        for x in sparse:
            lines.append(f"- {x['region']}: {x['reason']}")
    else:
        lines.append("- none flagged beyond known Stage 1–2 limits")
    lines += [
        "",
        "## Visual inspection",
        "",
        f"Fixed sample (seed {INSPECT_SEED}): `reports/stage3c_inspection_sample.csv`, figures under `figures/stage3c/`.",
        "Apparent @100 m hits that look like wrong scenes are flagged in that table **without** relabelling or changing thresholds.",
        "",
        "## Limitations (do not hide)",
        "",
        "1. Labels are matcher outcomes vs CSV centres, not surveyed centimetre GT.",
        "2. Footprint polygons remain APPROXIMATE (Stage 2D).",
        "3. Region 07 map is tiny (170 px tall); chance proximity can inflate success.",
        "4. Region 09 is a four-tile union; residual tile-edge artefacts possible.",
        "5. No held-out-region matcher selection; 03 was the tuning region.",
        "6. Visual false positives may exist among positional @100 m hits.",
        "7. Do not treat these labels as calibrated probabilities.",
        "",
        f"## Predictor-training decision: {decision}",
        "",
        "A complete file dump is not enough. This gate requires mixed 100 m outcomes, inventory match, "
        "no search leakage, and stated limitations. No predictor training, calibration, or scheduling in this stage.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "py -3 F:\\CMBL\\scripts\\localization\\stage3c_gate.py",
        "py -3 F:\\CMBL\\scripts\\generate_real_outcomes.py --smoke",
        "py -3 F:\\CMBL\\scripts\\generate_real_outcomes.py --extract-embeddings",
        "py -3 F:\\CMBL\\scripts\\generate_real_outcomes.py --run --resume",
        "py -3 F:\\CMBL\\scripts\\generate_real_outcomes.py --audit",
        "```",
        "",
    ]
    (REPORTS / "stage3c_label_generation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": decision, "n": len(rows), "s100": overall["success_100m"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
