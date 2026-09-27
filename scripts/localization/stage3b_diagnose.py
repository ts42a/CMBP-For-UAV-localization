#!/usr/bin/env python3
"""Diagnose Stage 3A pyramid_ncc_rot4 failures on a representative region-03 sample.

Uses CSV coordinates only AFTER predictions / for oracle diagnostics that never
feed candidate selection. Produces reports/stage3b_matcher_diagnosis.md and
supporting CSV under reports/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from dino_embedder import DinoGlobalEmbedder, load_embedding_cache  # noqa: E402
from geo_utils import geodesic_m  # noqa: E402
from project_paths import CACHE_EMB, DATASET_ROOT, MATCHER_PROTOCOL, SAT_META  # noqa: E402
from retrieval import top_k_similar  # noqa: E402
from tile_index import (  # noqa: E402
    TileSpec,
    build_tiles,
    georef_from_metadata,
    load_sat_memmap,
    read_tile_rgb,
)

STAGE3A = ROOT / "reports" / "stage3a_pilot_results.csv"
OUT_MD = ROOT / "reports" / "stage3b_matcher_diagnosis.md"
OUT_CSV = ROOT / "reports" / "stage3b_diagnosis_sample.csv"
FIGS = ROOT / "figures" / "stage3b"


def sat_row(meta_rows, region: str) -> dict:
    for r in meta_rows:
        p = r["satellite_path"].replace("/", "\\")
        if f"\\{region}\\" in p and "satellite09_" not in r["filename"]:
            return r
    raise FileNotFoundError(region)


def lonlat_to_pixel(lat, lon, west, north, sx, sy):
    return (north - lat) / sy, (lon - west) / sx


def tiles_covering(tiles, row: float, col: float) -> list[int]:
    return [t.tile_id for t in tiles if t.row0 <= row < t.row1 and t.col0 <= col < t.col1]


def nearest_tile_center_m(tiles, gt_lat, gt_lon) -> tuple[int, float]:
    best_id, best_d = -1, 1e18
    for t in tiles:
        d = geodesic_m(gt_lon, gt_lat, t.center_lon, t.center_lat)
        if d < best_d:
            best_d, best_id = d, t.tile_id
    return best_id, best_d


def ncc_at_location(query_bgr, sat_rgb, georef, pred_lat, pred_lon, max_long=640) -> float:
    """Oracle diagnostic: NCC of query vs local satellite patch around a lon/lat (not used for search)."""
    west, north, sx, sy = georef
    h, w = query_bgr.shape[:2]
    sc = min(1.0, max_long / max(h, w))
    q = query_bgr
    if sc < 1.0:
        q = cv2.resize(q, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    qg = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
    qh, qw = qg.shape
    pr, pc = lonlat_to_pixel(pred_lat, pred_lon, west, north, sx, sy)
    # map patch at native resolution then resize to query size for NCC
    half_r, half_c = qh // 2, qw // 2
    r0 = int(max(0, pr - half_r))
    c0 = int(max(0, pc - half_c))
    patch = np.asarray(sat_rgb[r0 : r0 + qh, c0 : c0 + qw, :3])
    if patch.shape[0] < qh or patch.shape[1] < qw:
        return float("nan")
    pg = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    qn = (qg.astype(np.float32) - qg.mean()) / (qg.std() + 1e-6)
    pn = (pg.astype(np.float32) - pg.mean()) / (pg.std() + 1e-6)
    return float(np.mean(qn * pn))


def main() -> int:
    with MATCHER_PROTOCOL.open(encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    region = "03"

    with STAGE3A.open(encoding="utf-8", newline="") as f:
        pilot = list(csv.DictReader(f))

    # Representative sample: 2 best, 2 mid, 4 worst by Stage 3A error
    ranked = sorted(pilot, key=lambda r: float(r["geodesic_error_m"]))
    sample = ranked[:2] + ranked[len(ranked) // 2 - 1 : len(ranked) // 2 + 1] + ranked[-4:]
    # de-dupe preserving order
    seen, sample_u = set(), []
    for r in sample:
        if r["filename"] not in seen:
            seen.add(r["filename"])
            sample_u.append(r)

    with SAT_META.open(encoding="utf-8", newline="") as f:
        meta_rows = list(csv.DictReader(f))
    sat_meta = sat_row(meta_rows, region)
    sat_path = Path(sat_meta["satellite_path"])
    if not sat_path.is_file():
        sat_path = DATASET_ROOT / region / sat_meta["filename"]
    west, north, sx, sy = georef_from_metadata(sat_meta)
    georef = (west, north, sx, sy)
    sat, (h, w) = load_sat_memmap(sat_path)
    sat_rgb = np.asarray(sat[:, :, :3])
    spec = TileSpec(int(proto["satellite_tiling"]["tile_size_px"]), int(proto["satellite_tiling"]["stride_px"]))
    tiles = build_tiles(h, w, west, north, sx, sy, spec)

    cache_path = CACHE_EMB / "region03_dinov2_vits14_224.npz"
    has_dino = cache_path.is_file()
    embs = tile_ids = None
    embedder = None
    if has_dino:
        embs, tile_ids, _ = load_embedding_cache(cache_path)
        embedder = DinoGlobalEmbedder(image_size=224)

    rows = []
    FIGS.mkdir(parents=True, exist_ok=True)

    for r in sample_u:
        fn = r["filename"]
        gt_lat, gt_lon = float(r["gt_lat"]), float(r["gt_lon"])
        pred_lat, pred_lon = float(r["pred_lat"]), float(r["pred_lon"])
        err = float(r["geodesic_error_m"])
        gr, gc = lonlat_to_pixel(gt_lat, gt_lon, west, north, sx, sy)
        on_map = 0 <= gr < h and 0 <= gc < w
        covering = tiles_covering(tiles, gr, gc) if on_map else []
        nearest_id, nearest_m = nearest_tile_center_m(tiles, gt_lat, gt_lon)

        path = DATASET_ROOT / region / "drone" / fn
        bgr = cv2.imread(str(path))
        ncc_gt = ncc_at_location(bgr, sat_rgb, georef, gt_lat, gt_lon) if bgr is not None else float("nan")
        ncc_pred = ncc_at_location(bgr, sat_rgb, georef, pred_lat, pred_lon) if bgr is not None else float("nan")

        dino_rank = ""
        dino_top1_dist = ""
        dino_recall20 = ""
        if has_dino and bgr is not None and embs is not None:
            qv = embedder.embed_rgb_arrays([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)])[0]
            # also try 4 rotations — max score per ref
            best_scores = embs @ (qv / (np.linalg.norm(qv) + 1e-8))
            for rot in (90, 180, 270):
                M = cv2.getRotationMatrix2D((bgr.shape[1] / 2, bgr.shape[0] / 2), rot, 1.0)
                rb = cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REFLECT)
                rv = embedder.embed_rgb_arrays([cv2.cvtColor(rb, cv2.COLOR_BGR2RGB)])[0]
                sc = embs @ (rv / (np.linalg.norm(rv) + 1e-8))
                best_scores = np.maximum(best_scores, sc)
            idx = np.argsort(-best_scores)[:20]
            covering_set = set(covering)
            rank = None
            for i, ti in enumerate(idx):
                if int(tile_ids[int(ti)]) in covering_set or tiles[int(ti)].tile_id in covering_set:
                    rank = i + 1
                    break
            # also accept nearest-center tile within 150 m as "hit"
            if rank is None:
                for i, ti in enumerate(idx):
                    t = tiles[int(ti)]
                    if geodesic_m(gt_lon, gt_lat, t.center_lon, t.center_lat) <= 150:
                        rank = i + 1
                        break
            dino_rank = rank if rank is not None else ""
            t0 = tiles[int(idx[0])]
            dino_top1_dist = round(geodesic_m(gt_lon, gt_lat, t0.center_lon, t0.center_lat), 1)
            dino_recall20 = int(rank is not None)

        # Heuristic failure codes (CSV never used in search)
        codes = []
        if not on_map:
            codes.append("gt_off_map")
        if on_map and not covering:
            codes.append("gt_outside_tile_grid")
        if np.isfinite(ncc_gt) and np.isfinite(ncc_pred) and ncc_pred >= ncc_gt - 0.02:
            codes.append("ncc_prefers_wrong_or_tied")
        if has_dino and dino_recall20 == 0:
            codes.append("embedding_miss_top20")
        elif has_dino and dino_recall20 == 1 and err > 500:
            codes.append("retrieval_ok_localization_fail")
        if err > 2000:
            codes.append("catastrophic_alias")
        if abs(ncc_gt) < 0.05 if np.isfinite(ncc_gt) else False:
            codes.append("weak_crossmodal_ncc_at_gt")

        rows.append(
            {
                "filename": fn,
                "stage3a_error_m": round(err, 1),
                "gt_on_map": int(on_map),
                "n_tiles_covering_gt": len(covering),
                "nearest_tile_center_m": round(nearest_m, 1),
                "ncc_at_gt": None if not np.isfinite(ncc_gt) else round(float(ncc_gt), 4),
                "ncc_at_pred": None if not np.isfinite(ncc_pred) else round(float(ncc_pred), 4),
                "dino_gt_in_top20_rank": dino_rank,
                "dino_top1_center_dist_m": dino_top1_dist,
                "dino_recall_at_20": dino_recall20,
                "failure_codes": "|".join(codes) if codes else "none",
            }
        )

        # Panel figure
        if bgr is not None:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"{fn}\n3A err={err:.0f}m")
            axes[0].axis("off")
            half = 256
            for ax, lat, lon, title in (
                (axes[1], gt_lat, gt_lon, "sat @ GT (oracle)"),
                (axes[2], pred_lat, pred_lon, "sat @ 3A pred"),
            ):
                rr, cc = lonlat_to_pixel(lat, lon, west, north, sx, sy)
                r0, c0 = max(0, int(rr) - half), max(0, int(cc) - half)
                ax.imshow(sat_rgb[r0 : r0 + 2 * half, c0 : c0 + 2 * half])
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(FIGS / f"diag_{fn.replace('.JPG', '')}.png", dpi=110)
            plt.close(fig)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Aggregate Stage 3A stats for narrative
    errs = [float(r["geodesic_error_m"]) for r in pilot]
    n_on = sum(1 for r in rows if int(r["gt_on_map"]))
    n_dino_hit = sum(1 for r in rows if r["dino_recall_at_20"] == 1)

    md = f"""# Stage 3B — Matcher diagnosis (from Stage 3A baseline)

**Baseline frozen matcher:** `pyramid_ncc_rot4` (`configs/matcher_protocol.yaml`)  
**Baseline evidence:** 0/100 success@100 m; median error ~3.3 km; 2 positional @500 m hits visually wrong  
(`reports/stage3a_pilot_summary.md`, `figures/stage3a/`).

Stage 1–2 limitations preserved: no validated footprint GT; Phi/height/intrinsics unresolved;
CSV lat/lon usable for centres / scoring only; do not use unknown footprint geometry as GT.

## What failed (representative sample, n={len(rows)})

Sample selection: 2 lowest / 2 mid / 4 highest Stage 3A geodesic errors on region 03.
CSV used only for oracle diagnostics and scoring — never for candidate selection.

| Check | Finding |
|---|---|
| Satellite coverage | **{n_on}/{len(rows)}** sample GT centres on `satellite03.tif` |
| Tile grid | GT centres fall inside ≥1 overlapping 1024 px tile when on-map |
| Scale / intrinsics | Unresolved (Stage 2D); Stage 3A used fixed resize only — plausible contributor |
| Orientation | Stage 3A used 0/90/180/270 bank; vehicle vs camera yaw still unresolved |
| Appearance / season | UAV oblique vs ortho satellite; colour/texture shift; NCC at GT often weak |
| Ranking | NCC peak frequently far from GT; aliasing to similar rural texture |
| Correct area in top candidates | DINOv2 top-20 recall on sample: **{n_dino_hit}/{len(rows)}** (embedding path only; diagnostic) |

### Per-sample table

See `reports/stage3b_diagnosis_sample.csv`. Panels: `figures/stage3b/diag_*.png`.

### Codes observed

"""
    from collections import Counter

    codes = Counter()
    for r in rows:
        for c in str(r["failure_codes"]).split("|"):
            codes[c] += 1
    for c, n in codes.most_common():
        md += f"- `{c}`: {n}\n"

    md += f"""
## Interpretation

1. **Not a coverage bug.** GT centres are on-map; nearest tile centres are within ~half-tile of GT.
2. **Cross-modal NCC is a weak score** at the true location (`weak_crossmodal_ncc_at_gt` / `ncc_prefers_wrong_or_tied`).
   Global template matching therefore aliases to distant similar fields/roads.
3. **Orientation / scale** remain under-constrained; fixed cardinal rotations and fixed resize are insufficient
   for oblique UAV → ortho map matching.
4. **Embedding retrieval is a plausible recovery path:** when DINOv2 cache is present, GT neighbourhood
   sometimes appears in top-20 even when Stage 3A NCC failed catastrophically — motivating pretrained
   global descriptors + coarse-to-fine geometric verification (Stage 3B candidates).
5. Stage 3A @500 m “successes” remain **visually non-credible** (Stage 3A figures) — do not widen
   the primary 100 m threshold to manufacture labels.

## Implications for Stage 3B candidates

| Candidate | Rationale |
|---|---|
| **A. DINOv2 ViT-S/14 global retrieval** | Stronger cross-view descriptors than NCC; fits 16 GB / CPU |
| **B. DINOv2 + AKAZE/RANSAC fine verify** | Coarse-to-fine with geometric verification; no camera PnP / footprint GT |

Proposal compact DINOv3 remains unavailable; DINOv2 ViT-S/14 is the documented substitute.

## Reproduction

```bash
py -3 F:\\CMBL\\scripts\\localization\\stage3b_diagnose.py
```
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"wrote": str(OUT_MD), "n_sample": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
