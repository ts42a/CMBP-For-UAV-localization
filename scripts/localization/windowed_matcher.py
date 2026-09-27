"""Windowed variant of the frozen matcher (Stage 4/5 labelling).

Same backbone, tiling, rotations and top-k as frozen_matcher.localize_query; retrieval
candidates are restricted to tiles whose centre lies within radius R of a prior centre.
The fine verifier is chosen by fine_match.method: akaze_ransac_homography (Stage 3B
frozen) or dinov2_patch_similarity_ransac. The prior centre is the only position input.
"""
from __future__ import annotations

import time

import numpy as np

from fine_verify import akaze_rerank, query_features
from frozen_matcher import query_embeddings
from patch_verify import PatchVerifier

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0
PATCH_METHOD = "dinov2_patch_similarity_ransac"
AKAZE_METHOD = "akaze_ransac_homography"


def is_patch(frozen: dict) -> bool:
    m = frozen["fine_match"]["method"]
    if m not in (PATCH_METHOD, AKAZE_METHOD):
        raise ValueError(f"unknown fine_match.method {m}")
    return m == PATCH_METHOD


class QueryContext:
    """Per-frame work reused across every radius/offset attempt of that frame."""

    def __init__(self, bgr: np.ndarray, embedder, frozen: dict, verifier: PatchVerifier | None = None):
        rots = tuple(int(x) for x in frozen["preprocessing"]["query_rotations_deg"])
        self.bgr = bgr
        self.qvecs = query_embeddings(embedder, bgr, rots)
        if is_patch(frozen):
            self.qfeat = None
            self.patch = verifier.query_feats(bgr)
        else:
            self.qfeat = query_features(bgr, frozen["fine_match"])
            self.patch = None


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


def _empty(n_window: int, frozen: dict, t0: float) -> dict:
    return {
        "n_window_tiles": n_window, "matcher_id": frozen["matcher_selected"],
        "pred_lat": "", "pred_lon": "", "predicted_tile_id": "", "predicted_raster_id": "",
        "top_score": "", "score_margin": "", "n_candidates": 0, "fine_inliers": 0,
        "used_homography": 0, "geo_verify_status": "empty_window",
        "runtime_s": round(time.perf_counter() - t0, 4),
    }


def localize_candidates(ctx: QueryContext, idx: RegionIndex, win: np.ndarray, frozen: dict,
                        verifier: PatchVerifier | None = None, t0: float | None = None) -> dict:
    t0 = time.perf_counter() if t0 is None else t0
    if len(win) == 0:
        return _empty(0, frozen, t0)
    scores_all = np.max(idx.embs[win] @ ctx.qvecs.T, axis=1)
    k = min(int(frozen["retrieval"]["top_k"]), len(win))
    order = np.argsort(-scores_all, kind="stable")[:k]
    cand = [int(win[i]) for i in order]
    scores = scores_all[order]
    out = {
        "n_window_tiles": int(len(win)),
        "matcher_id": frozen["matcher_selected"],
        "top_score": round(float(scores[0]), 5),
        "score_margin": round(float(scores[0] - scores[1]), 5) if len(scores) > 1 else "",
        "n_candidates": len(cand),
    }

    if is_patch(frozen):
        fm = frozen["fine_match"]
        best = verifier.verify(ctx.patch, idx.rasters, idx.tiles, cand, [float(s) for s in fm["context_scales"]], idx.tile_cache)
        if best:
            t = idx.tiles[best["tile_index"]]
            out.update({
                "pred_lat": best["pred_lat"], "pred_lon": best["pred_lon"],
                "predicted_tile_id": t.tile_id, "predicted_raster_id": t.raster_id,
                "fine_inliers": best["inliers"], "used_homography": 1, "geo_verify_status": "patch_similarity",
                "patch_n_mutual": best["n_mutual"], "patch_query_rot_deg": best["query_rot_deg"],
                "patch_sim_scale": best["sim_scale"], "patch_sim_angle_deg": best["sim_angle_deg"],
                "patch_context_scale": best["context_scale"],
            })
        else:
            t = idx.tiles[cand[0]]
            out.update({
                "pred_lat": t.center_lat, "pred_lon": t.center_lon,
                "predicted_tile_id": t.tile_id, "predicted_raster_id": t.raster_id,
                "fine_inliers": 0, "used_homography": 0, "geo_verify_status": "tile_center_fallback",
            })
    else:
        r0 = idx.rasters[0]
        fv = akaze_rerank(ctx.bgr, [r.sat for r in idx.rasters], idx.tiles, cand,
                          (r0.west, r0.north, r0.sx, r0.sy), frozen["fine_match"],
                          qfeat=ctx.qfeat, tile_cache=idx.tile_cache)
        used_h = bool(fv["used_homography"])
        inl = int(fv["inliers"])
        chosen = next((t for t in idx.tiles if t.tile_id == fv["tile_id"]), idx.tiles[cand[0]])
        out.update({
            "pred_lat": fv["pred_lat"], "pred_lon": fv["pred_lon"],
            "predicted_tile_id": chosen.tile_id, "predicted_raster_id": chosen.raster_id,
            "fine_inliers": inl, "used_homography": int(used_h),
            "geo_verify_status": "homography" if used_h else ("inliers_below_min_tile_center" if inl > 0 else "tile_center_fallback"),
        })
    out["runtime_s"] = round(time.perf_counter() - t0, 4)
    return out


def localize_windowed(ctx: QueryContext, idx: RegionIndex, lat0: float, lon0: float, radius_m: float,
                      frozen: dict, verifier: PatchVerifier | None = None) -> dict:
    t0 = time.perf_counter()
    return localize_candidates(ctx, idx, idx.window(lat0, lon0, radius_m), frozen, verifier, t0)


def localize_fullmap(ctx: QueryContext, idx: RegionIndex, frozen: dict, verifier: PatchVerifier | None = None) -> dict:
    return localize_candidates(ctx, idx, np.arange(len(idx.tiles)), frozen, verifier)
