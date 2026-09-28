#!/usr/bin/env python3
"""SUPERSEDED 2026-09-28 (deviation D1): early Stage 3A/3B development pilot, predates
the frozen windowed pipeline and the DINOv2 patch verifier. Do not run this for label
generation. Use scripts/label_windowed.py with configs/matcher_stage4_frozen.yaml.

Development-only localization pilot: baseline vs DINO retrieval vs DINO+AKAZE."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from dino_embedder import DinoGlobalEmbedder, load_embedding_cache  # noqa: E402
from fine_verify import akaze_rerank  # noqa: E402
from geo_utils import geodesic_m  # noqa: E402
from leakage import assert_dev_only, dev_regions_from_config  # noqa: E402
from project_paths import (  # noqa: E402
    CACHE_EMB,
    DATASET_ROOT,
    DINO_CONFIG,
    MATCHER_PROTOCOL,
    RESULTS_PILOT,
    SAT_META,
)
from query_select import load_manifest_region, select_queries  # noqa: E402
from retrieval import gt_tile_id, top_k_similar  # noqa: E402
from tile_index import TileSpec, build_tiles, georef_from_metadata, load_sat_memmap  # noqa: E402

PILOT_SEED = 20260923
DEV_SEED = 42
THRESHOLDS = (25, 50, 100, 200)


def sat_row(meta_rows, region: str) -> dict:
    for r in meta_rows:
        p = r["satellite_path"].replace("/", "\\")
        if f"\\{region}\\" in p and "satellite09_" not in r["filename"]:
            return r
    raise FileNotFoundError(region)


def lonlat_to_pixel(lat, lon, west, north, sx, sy):
    col = (lon - west) / sx
    row = (north - lat) / sy
    return row, col


def success_flags(err: float) -> dict:
    return {f"success_{t}m": int(err <= t) for t in THRESHOLDS}


def load_baseline_csv() -> dict[str, dict]:
    p = ROOT / "reports" / "stage3a_pilot_results.csv"
    if not p.is_file():
        return {}
    out = {}
    with p.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["filename"]] = r
    return out


def dev_tune(
    queries: list[dict],
    embedder: DinoGlobalEmbedder,
    embs: np.ndarray,
    tiles,
    sat,
    georef,
) -> dict:
    grid = []
    for image_size in (224,):  # 336 skipped for CPU time; 224 matches cache
        for top_k in (5, 10, 20):
            for fine in (False, True):
                succ = 0
                errs = []
                recalls = []
                for q in queries:
                    path = DATASET_ROOT / "03" / "drone" / q["filename"]
                    bgr = cv2.imread(str(path))
                    qv = embedder.embed_rgb_arrays([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)])[0]
                    idx, scores = top_k_similar(qv, embs, top_k)
                    gt_lat, gt_lon = float(q["lat"]), float(q["lon"])
                    row, col = lonlat_to_pixel(gt_lat, gt_lon, *georef)
                    gt_id = gt_tile_id(tiles, row, col)
                    recalls.append(int(gt_id in [tiles[int(i)].tile_id for i in idx] if gt_id is not None else 0))
                    if fine:
                        fv = akaze_rerank(
                            bgr,
                            sat,
                            tiles,
                            [int(i) for i in idx],
                            georef,
                            {"query_max_long_edge": 768, "min_inliers": 12},
                        )
                        plat, plon = fv["pred_lat"], fv["pred_lon"]
                    else:
                        t = tiles[int(idx[0])]
                        plat, plon = t.center_lat, t.center_lon
                    err = geodesic_m(gt_lon, gt_lat, plon, plat)
                    errs.append(err)
                    if err <= 100:
                        succ += 1
                grid.append(
                    {
                        "image_size": image_size,
                        "top_k": top_k,
                        "fine_match": fine,
                        "success_100m": succ,
                        "n": len(queries),
                        "median_err": float(np.median(errs)),
                        "recall_at_k": float(np.mean(recalls)) if recalls else 0,
                    }
                )
    grid.sort(key=lambda x: (-x["success_100m"], x["median_err"], -x["recall_at_k"]))
    return grid[0], grid


def run_mode_dino(
    queries: list[dict],
    embedder: DinoGlobalEmbedder,
    embs: np.ndarray,
    tiles,
    sat,
    georef,
    frozen: dict,
    mode: str,
) -> list[dict]:
    top_k = int(frozen["retrieval"]["top_k"])
    use_fine = mode == "dino_fine"
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        path = DATASET_ROOT / "03" / "drone" / q["filename"]
        bgr = cv2.imread(str(path))
        fail = ""
        if bgr is None:
            fail = "read_error"
            continue
        qv = embedder.embed_rgb_arrays([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)])[0]
        idx, scores = top_k_similar(qv, embs, top_k)
        gt_lat, gt_lon = float(q["lat"]), float(q["lon"])
        row, col = lonlat_to_pixel(gt_lat, gt_lon, *georef)
        gt_tid = gt_tile_id(tiles, row, col)
        rank_gt = None
        for rnk, i in enumerate(idx):
            if gt_tid is not None and tiles[int(i)].tile_id == gt_tid:
                rank_gt = rnk + 1
                break
        inliers = 0
        if use_fine:
            fv = akaze_rerank(
                bgr,
                sat,
                tiles,
                [int(i) for i in idx],
                georef,
                frozen["fine_match"],
            )
            plat, plon = fv["pred_lat"], fv["pred_lon"]
            ref_id = fv["tile_id"]
            inliers = fv["inliers"]
            reason = "dino+akaze"
        else:
            t = tiles[int(idx[0])]
            plat, plon = t.center_lat, t.center_lon
            ref_id = t.tile_id
            reason = "dino_top1"
        err = geodesic_m(gt_lon, gt_lat, plon, plat)
        rt = time.perf_counter() - t0
        rec = {
            "query_id": q["num"],
            "region": "03",
            "filename": q["filename"],
            "gt_lat": gt_lat,
            "gt_lon": gt_lon,
            "pred_lat": plat,
            "pred_lon": plon,
            "reference_id": ref_id,
            "localization_error_m": round(err, 3),
            **success_flags(err),
            "global_top1_similarity": round(float(scores[0]), 5),
            "top_k": top_k,
            "top_k_indices": "|".join(str(int(i)) for i in idx),
            "top_k_scores": "|".join(f"{s:.4f}" for s in scores),
            "gt_tile_id": gt_tid if gt_tid is not None else "",
            "gt_in_topk_rank": rank_gt if rank_gt is not None else "",
            "fine_inliers": inliers,
            "selection_reason": reason,
            "runtime_s": round(rt, 3),
            "failure_reason": fail,
            "matcher_mode": mode,
        }
        rows.append(rec)
    return rows


def failure_analysis(rows: list[dict], mode: str) -> list[dict]:
    """Heuristic codes for a sample of failures/successes (dev region only)."""
    out = []
    by_err = sorted(rows, key=lambda r: r["localization_error_m"])
    samples = by_err[:5] + by_err[-10:]
    seen = set()
    for r in samples:
        if r["filename"] in seen:
            continue
        seen.add(r["filename"])
        err = r["localization_error_m"]
        rank = r.get("gt_in_topk_rank")
        code = "other"
        if rank == "" or rank is None:
            code = "retrieval_miss_or_georef"
        elif rank and int(rank) > 1:
            code = "retrieval_hit_but_rerank_fail"
        elif err <= 100:
            code = "success_verify"
        elif err > 500:
            code = "retrieval_failure_or_ambiguous_texture"
        out.append(
            {
                "filename": r["filename"],
                "mode": mode,
                "error_m": err,
                "gt_in_topk_rank": rank,
                "failure_code": code,
                "notes": "Heuristic; dev region only.",
            }
        )
    return out


def summarize(rows: list[dict], mode: str) -> dict:
    errs = [r["localization_error_m"] for r in rows]
    n = len(rows)
    s100 = sum(r["success_100m"] for r in rows)
    return {
        "mode": mode,
        "n": n,
        "success_100m": s100,
        "failure_100m": n - s100,
        "success_rate_100m": s100 / n if n else 0,
        "median_error_m": float(np.median(errs)) if errs else None,
        "mean_error_m": float(np.mean(errs)) if errs else None,
        "p90_error_m": float(np.percentile(errs, 90)) if errs else None,
        **{f"success_{t}m": sum(r[f"success_{t}m"] for r in rows) for t in THRESHOLDS},
        "mean_runtime_s": float(np.mean([float(r["runtime_s"]) for r in rows if str(r.get("runtime_s", "")).strip()]))
        if rows
        else 0,
    }


def write_frozen_config(best: dict, embedder: DinoGlobalEmbedder, dev_grid: list) -> None:
    cfg = {
        "schema_version": "dinov2_retrieval_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "equivalent_note": "DINOv3 slot filled by frozen DINOv2 ViT-S/14 (torch.hub).",
        "model": {k: str(v) for k, v in embedder.meta().__dict__.items()},
        "retrieval": {
            "metric": "cosine",
            "top_k": best["top_k"],
            "l2_normalize": True,
        },
        "fine_match": {
            "enabled": best["fine_match"],
            "method": "akaze_ransac_homography",
            "query_max_long_edge": 768,
            "ratio_test": 0.75,
            "min_inliers": 12,
            "ransac_reproj_threshold_px": 3.0,
        },
            "dev_tuning": {
            "regions_only": list(dev_regions_from_config(yaml.safe_load(MATCHER_PROTOCOL.open(encoding="utf-8")))),
            "n_dev_queries": 24,
            "seed": DEV_SEED,
            "grid_results": dev_grid,
            "selected": best,
        },
        "success_threshold_primary_m": 100,
        "random_seed": PILOT_SEED,
    }
    DINO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with DINO_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embed", action="store_true", help="Require existing embedding cache")
    ap.add_argument("--skip-dev-tune", action="store_true", help="Load frozen config if present")
    args = ap.parse_args()

    with MATCHER_PROTOCOL.open(encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    region = proto["leakage_and_splits"]["pilot_region"]
    assert_dev_only(region, proto, "pilot")

    RESULTS_PILOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_EMB / "region03_dinov2_vits14_224.npz"
    if not cache_path.is_file() and not args.skip_embed:
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts" / "extract_reference_embeddings.py"), "--region", "03"]
        )
    embs, tile_ids, cache_meta = load_embedding_cache(cache_path)

    with SAT_META.open(encoding="utf-8", newline="") as f:
        meta_rows = list(csv.DictReader(f))
    sat_meta = sat_row(meta_rows, region)
    sat_path = Path(sat_meta["satellite_path"])
    if not sat_path.is_file():
        sat_path = DATASET_ROOT / region / sat_meta["filename"]
    west, north, sx, sy = georef_from_metadata(sat_meta)
    georef = (west, north, sx, sy)
    spec = TileSpec(int(proto["satellite_tiling"]["tile_size_px"]), int(proto["satellite_tiling"]["stride_px"]))
    sat, (h, w) = load_sat_memmap(sat_path)
    tiles = build_tiles(h, w, west, north, sx, sy, spec)
    assert len(tiles) == len(embs)

    manifest = load_manifest_region(region)
    dev_q = select_queries(manifest, 24, DEV_SEED)
    pilot_q = select_queries(manifest, 100, PILOT_SEED)

    embedder = DinoGlobalEmbedder(image_size=224)
    if args.skip_dev_tune and DINO_CONFIG.is_file():
        frozen = yaml.safe_load(DINO_CONFIG.read_text(encoding="utf-8"))
        dev_grid = frozen.get("dev_tuning", {}).get("grid_results", [])
        best = frozen.get("dev_tuning", {}).get("selected", {})
    else:
        best, dev_grid = dev_tune(dev_q, embedder, embs, tiles, sat, georef)
        frozen = write_frozen_config(best, embedder, dev_grid)

    modes = ["dino_global", "dino_fine"]
    all_rows = {}
    summaries = {}
    for mode in modes:
        if mode == "dino_global" and not frozen["fine_match"]["enabled"] and best["fine_match"]:
            pass
        rows = run_mode_dino(pilot_q, embedder, embs, tiles, sat, georef, frozen, mode)
        all_rows[mode] = rows
        summaries[mode] = summarize(rows, mode)
        fa = failure_analysis(rows, mode)
        with (RESULTS_PILOT / f"failure_analysis_{mode}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fa[0].keys()) if fa else ["filename"])
            w.writeheader()
            w.writerows(fa)

    # Baseline from Stage 3A (frozen CPU experiment — not re-tuned)
    baseline = load_baseline_csv()
    base_rows = []
    for q in pilot_q:
        b = baseline.get(q["filename"])
        if not b:
            continue
        err = float(b["geodesic_error_m"])
        base_rows.append(
            {
                "query_id": q["num"],
                "region": "03",
                "filename": q["filename"],
                "localization_error_m": err,
                **success_flags(err),
                "matcher_mode": "baseline_pyramid_ncc_rot4",
                "runtime_s": b.get("runtime_s", ""),
            }
        )
    if base_rows:
        summaries["baseline_pyramid_ncc_rot4"] = summarize(base_rows, "baseline_pyramid_ncc_rot4")

    # Write per-query CSV (primary: best frozen mode)
    primary_mode = "dino_fine" if frozen["fine_match"]["enabled"] else "dino_global"
    pq = RESULTS_PILOT / "per_query_results.csv"
    fields = list(all_rows[primary_mode][0].keys()) if all_rows[primary_mode] else []
    with pq.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows[primary_mode])

    with (RESULTS_PILOT / "per_query_results_dino_global.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows["dino_global"])
    with (RESULTS_PILOT / "per_query_results_dino_fine.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows["dino_fine"])

    reg = [{"region": "03", **summaries[primary_mode]}]
    with (RESULTS_PILOT / "region_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reg[0].keys()))
        w.writeheader()
        w.writerows(reg)

    # Gate
    s = summaries[primary_mode]
    mixed = 0 < s["success_100m"] < s["n"]
    gate = "PASS" if mixed else "FAIL"
    summary = {
        "gate": gate,
        "primary_mode": primary_mode,
        "frozen_config": str(DINO_CONFIG),
        "summaries": summaries,
        "dev_tuning_best": best,
        "leakage": {
            "dev_regions_only": list(dev_regions_from_config(proto)),
            "pilot_region": region,
            "held_out_not_used": True,
        },
    }
    (RESULTS_PILOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Manifest
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_hash = "unknown"
    import torch

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "config_hash": hashlib.sha256(DINO_CONFIG.read_bytes()).hexdigest()[:16],
        "frozen_config_path": str(DINO_CONFIG),
        "torch_version": torch.__version__,
        "device": str(embedder.device),
        "pilot_seed": PILOT_SEED,
        "dev_seed": DEV_SEED,
    }
    (RESULTS_PILOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    gate_md = RESULTS_PILOT / "gate_report.md"
    gate_md.write_text(
        "\n".join(
            [
                "# Label-generation gate (development pilot)",
                "",
                f"**Decision: {gate}**",
                "",
                f"Primary frozen mode: `{primary_mode}`",
                "",
                "## Summaries @100m",
                "",
                json.dumps({k: v for k, v in summaries.items()}, indent=2),
                "",
                "## Evidence",
                "",
                f"- Mixed success/failure @100m: **{mixed}**",
                f"- Dev tuning used **24** queries on region **03** only.",
                "",
                "No localizability predictor training. No full 6,774 run.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
