"""Frozen Stage 3B matcher: DINOv2 ViT-S/14 retrieval + AKAZE/RANSAC.

CSV lat/lon are never used here. Callers score afterwards.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from dino_embedder import DinoGlobalEmbedder
from fine_verify import akaze_rerank
from region_map import Raster, read_tile_rgb_region
from tile_index import Tile

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "configs" / "matcher_stage3b_frozen.yaml"


def load_frozen() -> dict:
    return yaml.safe_load(FROZEN.read_text(encoding="utf-8"))


def query_embeddings(embedder: DinoGlobalEmbedder, bgr: np.ndarray, rotations: tuple[int, ...]) -> np.ndarray:
    vecs = []
    images = []
    for rot in rotations:
        if rot == 0:
            im = bgr
        else:
            M = cv2.getRotationMatrix2D((bgr.shape[1] / 2, bgr.shape[0] / 2), rot, 1.0)
            im = cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REFLECT)
        images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    stacked = embedder.embed_rgb_arrays(images, batch_size=len(images))
    for v in stacked:
        vecs.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(vecs, axis=0)


def top_k_multi_rot(embs: np.ndarray, qvecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.max(embs @ qvecs.T, axis=1)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def localize_query(
    bgr: np.ndarray | None,
    embedder: DinoGlobalEmbedder,
    embs: np.ndarray,
    tiles: list[Tile],
    rasters: list[Raster],
    frozen: dict,
) -> dict:
    t0 = time.perf_counter()
    matcher_fail = ""
    if bgr is None:
        return {
            "pred_lat": "",
            "pred_lon": "",
            "predicted_tile_id": "",
            "predicted_raster_id": "",
            "top_score": "",
            "score_margin": "",
            "top_k": frozen["retrieval"]["top_k"],
            "fine_inliers": 0,
            "used_homography": 0,
            "geo_verify_status": "not_run",
            "matcher_failure_reason": "read_error",
            "runtime_s": round(time.perf_counter() - t0, 3),
            "search_used_gt_position": False,
            "matcher_id": frozen["matcher_selected"],
        }
    if not tiles or embs.size == 0:
        return {
            "pred_lat": "",
            "pred_lon": "",
            "predicted_tile_id": "",
            "predicted_raster_id": "",
            "top_score": "",
            "score_margin": "",
            "top_k": frozen["retrieval"]["top_k"],
            "fine_inliers": 0,
            "used_homography": 0,
            "geo_verify_status": "no_tiles",
            "matcher_failure_reason": "empty_tile_index",
            "runtime_s": round(time.perf_counter() - t0, 3),
            "search_used_gt_position": False,
            "matcher_id": frozen["matcher_selected"],
        }

    rots = tuple(int(x) for x in frozen["preprocessing"]["query_rotations_deg"])
    top_k = int(frozen["retrieval"]["top_k"])
    qvecs = query_embeddings(embedder, bgr, rots)
    idx, scores = top_k_multi_rot(embs, qvecs, top_k)
    sats = [r.sat for r in rasters]
    georef = (rasters[0].west, rasters[0].north, rasters[0].sx, rasters[0].sy)
    fv = akaze_rerank(bgr, sats, tiles, [int(i) for i in idx], georef, frozen["fine_match"])
    plat, plon = fv["pred_lat"], fv["pred_lon"]
    used_h = bool(fv["used_homography"])
    inliers = int(fv["inliers"])
    if used_h:
        geo_status = "homography"
    elif inliers > 0:
        geo_status = "inliers_below_min_tile_center"
    else:
        geo_status = "tile_center_fallback"

    chosen = next((t for t in tiles if t.tile_id == fv["tile_id"]), tiles[int(idx[0])])
    return {
        "pred_lat": plat,
        "pred_lon": plon,
        "predicted_tile_id": chosen.tile_id,
        "predicted_raster_id": chosen.raster_id,
        "top_score": round(float(scores[0]), 5),
        "score_margin": round(float(scores[0] - scores[1]), 5) if len(scores) > 1 else 0.0,
        "top_k": top_k,
        "fine_inliers": inliers,
        "used_homography": int(used_h),
        "geo_verify_status": geo_status,
        "matcher_failure_reason": matcher_fail,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "search_used_gt_position": False,
        "matcher_id": frozen["matcher_selected"],
    }
