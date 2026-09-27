"""AKAZE + RANSAC homography reranking on retrieval candidates (no camera PnP)."""
from __future__ import annotations

import cv2
import numpy as np

from geo_utils import pixel_to_lonlat
from tile_index import read_tile_rgb


def _sat_arr(sat, tile):
    if isinstance(sat, (list, tuple)):
        return sat[int(getattr(tile, "raster_id", 0))]
    return sat


def _georef_of(tile, georef):
    sx = float(getattr(tile, "sx", 0) or 0)
    if sx:
        return tile.west, tile.north, tile.sx, tile.sy
    return georef


def akaze_rerank(
    query_bgr: np.ndarray,
    sat,
    tiles,
    cand_indices: list[int],
    georef: tuple[float, float, float, float],
    params: dict,
) -> dict:
    """
    Homography is used only as a 2D image-plane alignment prior (oblique UAV vs ortho map).
    No intrinsics / PnP — not claiming metric footprint GT.
    """
    west, north, sx, sy = georef
    max_long = int(params.get("query_max_long_edge", 768))
    h, w = query_bgr.shape[:2]
    sc = min(1.0, max_long / max(h, w))
    if sc < 1.0:
        query_bgr = cv2.resize(query_bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    qg = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    kp_q, des_q = akaze.detectAndCompute(qg, None)
    ratio = float(params.get("ratio_test", 0.75))
    min_inl = int(params.get("min_inliers", 12))
    ransac_th = float(params.get("ransac_reproj_threshold_px", 3.0))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    best = {
        "tile_id": cand_indices[0] if cand_indices else -1,
        "inliers": 0,
        "pred_lat": None,
        "pred_lon": None,
        "used_homography": False,
    }

    if des_q is None:
        t = tiles[cand_indices[0]]
        best.update({"pred_lat": t.center_lat, "pred_lon": t.center_lon})
        return best

    for ti in cand_indices:
        t = tiles[int(ti)]
        pg = cv2.cvtColor(read_tile_rgb(_sat_arr(sat, t), t), cv2.COLOR_RGB2GRAY)
        kp_t, des_t = akaze.detectAndCompute(pg, None)
        if des_t is None:
            continue
        pairs = bf.knnMatch(des_q, des_t, k=2)
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
        if len(good) < 8:
            continue
        src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_th)
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers > best["inliers"]:
            pred_lat, pred_lon = t.center_lat, t.center_lon
            used_h = False
            if H is not None and inliers >= min_inl:
                qh, qw = qg.shape[:2]
                pt = np.array([[[qw / 2, qh / 2]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(pt, H)[0, 0]
                fr = t.row0 + mapped[1]
                fc = t.col0 + mapped[0]
                tw, tn, tsx, tsy = _georef_of(t, georef)
                pred_lat, pred_lon = pixel_to_lonlat(fr, fc, tw, tn, tsx, tsy)
                used_h = True
            best = {
                "tile_id": t.tile_id,
                "inliers": inliers,
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "used_homography": used_h,
            }
    if best["pred_lat"] is None:
        t = tiles[cand_indices[0]]
        best.update({"pred_lat": t.center_lat, "pred_lon": t.center_lon})
    return best
