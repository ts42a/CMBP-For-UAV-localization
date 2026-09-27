"""Full-map pyramid NCC localizer (CPU, no GT search mask)."""
from __future__ import annotations

import cv2
import numpy as np

from geo_utils import pixel_to_lonlat


def pyramid_ncc_predict(
    query_bgr: np.ndarray,
    sat_rgb: np.ndarray,
    georef: tuple[float, float, float, float],
    max_long: int = 768,
    sat_long: int = 4096,
    rotations: tuple[int, ...] = (0, 90, 180, 270),
) -> tuple[float, float, float, float, str | None]:
    """
    Returns pred_lat, pred_lon, top_score, score_margin, failure_reason.
    """
    west, north, sx, sy = georef
    q = query_bgr
    h, w = q.shape[:2]
    qs = min(1.0, max_long / max(h, w))
    if qs < 1.0:
        q = cv2.resize(q, (int(w * qs), int(h * qs)), interpolation=cv2.INTER_AREA)

    sat = sat_rgb
    sh, sw = sat.shape[:2]
    ss = min(1.0, sat_long / max(sh, sw))
    if ss < 1.0:
        sat = cv2.resize(sat, (int(sw * ss), int(sh * ss)), interpolation=cv2.INTER_AREA)

    best_score = -1.0
    second_score = -1.0
    best_loc = (0, 0)
    best_rot = 0

    for rot in rotations:
        if rot == 0:
            qr = q
        else:
            M = cv2.getRotationMatrix2D((q.shape[1] / 2, q.shape[0] / 2), rot, 1.0)
            qr = cv2.warpAffine(q, M, (q.shape[1], q.shape[0]), borderMode=cv2.BORDER_REFLECT)
        qg = cv2.cvtColor(qr, cv2.COLOR_BGR2GRAY)
        sg = cv2.cvtColor(sat, cv2.COLOR_RGB2GRAY)
        if qg.shape[0] >= sg.shape[0] or qg.shape[1] >= sg.shape[1]:
            continue
        res = cv2.matchTemplate(sg, qg, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            second_score = best_score
            best_score = float(max_val)
            best_loc = max_loc
            best_rot = rot
        elif max_val > second_score:
            second_score = float(max_val)

    if best_score < 0:
        return 0.0, 0.0, 0.0, 0.0, "template_too_large"

    # Map peak to full-resolution pixel (template top-left + centre)
    scale_back = 1.0 / ss
    row = (best_loc[1] + q.shape[0] / 2) * scale_back
    col = (best_loc[0] + q.shape[1] / 2) * scale_back
    pred_lat, pred_lon = pixel_to_lonlat(row, col, west, north, sx, sy)
    margin = best_score - max(second_score, -1.0)
    return pred_lat, pred_lon, best_score, margin, None
