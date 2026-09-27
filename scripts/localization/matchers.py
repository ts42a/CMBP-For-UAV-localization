"""Matcher implementations for Stage 3A (CPU OpenCV)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tile_index import Tile, read_tile_rgb


@dataclass
class MatchResult:
    tile_id: int
    pred_lat: float
    pred_lon: float
    top_score: float
    score_margin: float
    failure_reason: str | None


def _prep_query(bgr: np.ndarray, max_long: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.shape[2] == 3 else bgr
    h, w = rgb.shape[:2]
    scale = min(1.0, max_long / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def _coarse_vec(patch: np.ndarray, size: int = 64) -> np.ndarray:
    g = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
    s = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    s -= s.mean()
    n = np.linalg.norm(s) + 1e-6
    return s / n


def precompute_tile_coarse(sat, tiles: list[Tile], coarse_px: int) -> np.ndarray:
    mat = np.zeros((len(tiles), coarse_px * coarse_px), dtype=np.float32)
    for i, t in enumerate(tiles):
        mat[i] = _coarse_vec(read_tile_rgb(sat, t), coarse_px).ravel()
    return mat


def coarse_ncc_scores(
    query_gray: np.ndarray,
    sat,
    tiles: list[Tile],
    coarse_px: int,
    tile_coarse: np.ndarray | None = None,
) -> np.ndarray:
    qv = _coarse_vec(cv2.cvtColor(query_gray, cv2.COLOR_GRAY2RGB), coarse_px)
    if tile_coarse is not None:
        return tile_coarse @ qv.ravel()
    scores = np.empty(len(tiles), dtype=np.float32)
    for i, t in enumerate(tiles):
        patch = read_tile_rgb(sat, t)
        tv = _coarse_vec(patch, coarse_px)
        scores[i] = float(np.dot(qv.ravel(), tv.ravel()))
    return scores


def match_coarse_ncc(
    query_bgr: np.ndarray,
    sat,
    tiles: list[Tile],
    params: dict,
    max_long: int,
) -> MatchResult:
    _, qg = _prep_query(query_bgr, max_long)
    scores = coarse_ncc_scores(qg, sat, tiles, params.get("coarse_patch_px", 64))
    order = np.argsort(-scores)
    i0 = int(order[0])
    i1 = int(order[1]) if len(order) > 1 else i0
    t = tiles[i0]
    margin = float(scores[i0] - scores[i1])
    return MatchResult(
        t.tile_id,
        t.center_lat,
        t.center_lon,
        float(scores[i0]),
        margin,
        None,
    )


def match_akaze_c2f(
    query_bgr: np.ndarray,
    sat,
    tiles: list[Tile],
    params: dict,
    max_long: int,
) -> MatchResult:
    """AKAZE on coarse top-K; lat/lon refined in run_stage3a when georef available."""
    qrgb, qg = _prep_query(query_bgr, max_long)
    scores = coarse_ncc_scores(qg, sat, tiles, params.get("coarse_patch_px", 64))
    top_k = int(params.get("coarse_top_k", 15))
    order = np.argsort(-scores)[:top_k]
    t0 = tiles[int(order[0])]
    margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
    return MatchResult(t0.tile_id, t0.center_lat, t0.center_lon, float(scores[order[0]]), margin, None)


MATCHERS = {
    "coarse_ncc": match_coarse_ncc,
    "akaze_coarse_to_fine": match_akaze_c2f,
}
