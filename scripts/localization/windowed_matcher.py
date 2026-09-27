"""Windowed variant of the frozen dinov2_c2f matcher (Stage 4/5 labelling).

Identical backbone, tiling, rotations, top-k and AKAZE/RANSAC verification as
frozen_matcher.localize_query; the only difference is that retrieval candidates are
restricted to tiles whose centre lies within radius R of a prior centre.
The prior centre is supplied by the caller and is the only position input.
"""
from __future__ import annotations

import time

import numpy as np

from fine_verify import akaze_rerank, query_features
from frozen_matcher import query_embeddings

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


class QueryContext:
    """Per-frame work reused across every radius/offset attempt of that frame."""

    def __init__(self, bgr: np.ndarray, embedder, frozen: dict):
        rots = tuple(int(x) for x in frozen["preprocessing"]["query_rotations_deg"])
        self.bgr = bgr
        self.qvecs = query_embeddings(embedder, bgr, rots)
        self.qfeat = query_features(bgr, frozen["fine_match"])


class RegionIndex:
    def __init__(self, tiles, rasters, embs: np.ndarray):
        self.tiles = tiles
        self.rasters = rasters
        self.embs = embs
        self.lat = np.array([t.center_lat for t in tiles], dtype=np.float64)
        self.lon = np.array([t.center_lon for t in tiles], dtype=np.float64)
        self.tile_cache: dict = {}

    def window(self, lat0: float, lon0: float, radius_m: float) -> np.ndarray:
        dy = (self.lat - lat0) * M_PER_DEG_LAT
        dx = (self.lon - lon0) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
        return np.flatnonzero(dx * dx + dy * dy <= radius_m * radius_m)

    def clear_cache(self) -> None:
        self.tile_cache.clear()


def localize_windowed(ctx: QueryContext, idx: RegionIndex, lat0: float, lon0: float, radius_m: float, frozen: dict) -> dict:
    t0 = time.perf_counter()
    win = idx.window(lat0, lon0, radius_m)
    base = {
        "n_window_tiles": int(len(win)),
        "matcher_id": frozen["matcher_selected"],
    }
    if len(win) == 0:
        return {
            **base,
            "pred_lat": "",
            "pred_lon": "",
            "predicted_tile_id": "",
            "predicted_raster_id": "",
            "top_score": "",
            "score_margin": "",
            "n_candidates": 0,
            "fine_inliers": 0,
            "used_homography": 0,
            "geo_verify_status": "empty_window",
            "runtime_s": round(time.perf_counter() - t0, 4),
        }
    scores_all = np.max(idx.embs[win] @ ctx.qvecs.T, axis=1)
    k = min(int(frozen["retrieval"]["top_k"]), len(win))
    order = np.argsort(-scores_all, kind="stable")[:k]
    cand = [int(win[i]) for i in order]
    scores = scores_all[order]

    r0 = idx.rasters[0]
    fv = akaze_rerank(
        ctx.bgr,
        [r.sat for r in idx.rasters],
        idx.tiles,
        cand,
        (r0.west, r0.north, r0.sx, r0.sy),
        frozen["fine_match"],
        qfeat=ctx.qfeat,
        tile_cache=idx.tile_cache,
    )
    used_h = bool(fv["used_homography"])
    inl = int(fv["inliers"])
    status = "homography" if used_h else ("inliers_below_min_tile_center" if inl > 0 else "tile_center_fallback")
    chosen = next((t for t in idx.tiles if t.tile_id == fv["tile_id"]), idx.tiles[cand[0]])
    return {
        **base,
        "pred_lat": fv["pred_lat"],
        "pred_lon": fv["pred_lon"],
        "predicted_tile_id": chosen.tile_id,
        "predicted_raster_id": chosen.raster_id,
        "top_score": round(float(scores[0]), 5),
        "score_margin": round(float(scores[0] - scores[1]), 5) if len(scores) > 1 else "",
        "n_candidates": len(cand),
        "fine_inliers": inl,
        "used_homography": int(used_h),
        "geo_verify_status": status,
        "runtime_s": round(time.perf_counter() - t0, 4),
    }
