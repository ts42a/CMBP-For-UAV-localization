"""DINOv2 patch-token verification: mutual nearest neighbours + RANSAC similarity.

Uses the same frozen DINOv2 ViT-S/14 weights as retrieval. For each retrieval candidate
tile, a context crop (context_scale x tile size, centred on the tile) is resampled to
sat_crop_px and its patch tokens are matched against the query's patch tokens at 8
rotations. The query centre is mapped through the best similarity fit.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from geo_utils import pixel_to_lonlat

PATCH = 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class PatchVerifier:
    def __init__(self, model, params: dict):
        self.model = model
        self.device = next(model.parameters()).device
        self.p = params
        self.q_long = int(params["query_long_edge"])
        self.sat_px = int(params["sat_crop_px"])
        self.rots = [int(a) for a in params["query_rotations_deg"]]
        self.thr = float(params["ransac_reproj_threshold_px"])
        self.min_fit = int(params["min_matches_fit"])
        self.smin, self.smax = (float(x) for x in params["scale_range"])
        self.cov = float(params["min_valid_coverage"])

    @torch.inference_mode()
    def tokens(self, rgb: np.ndarray) -> np.ndarray:
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        out = self.model.forward_features(t)["x_norm_patchtokens"][0]
        return F.normalize(out, dim=-1).cpu().numpy()

    def _valid(self, mask: np.ndarray) -> np.ndarray:
        gh, gw = mask.shape[0] // PATCH, mask.shape[1] // PATCH
        cov = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        return (cov >= self.cov).reshape(-1)

    @staticmethod
    def _centres(h: int, w: int) -> np.ndarray:
        ys, xs = np.mgrid[0 : h // PATCH, 0 : w // PATCH]
        return np.stack([(xs + 0.5) * PATCH, (ys + 0.5) * PATCH], -1).reshape(-1, 2).astype(np.float32)

    def query_feats(self, bgr: np.ndarray) -> dict:
        h, w = bgr.shape[:2]
        sc = self.q_long / max(h, w)
        bw, bh = int(round(w * sc)), int(round(h * sc))
        base = cv2.cvtColor(cv2.resize(bgr, (bw, bh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        ones = np.ones((bh, bw), np.float32)
        rots = []
        for ang in self.rots:
            M = cv2.getRotationMatrix2D((bw / 2, bh / 2), ang, 1.0)
            c, s = abs(M[0, 0]), abs(M[0, 1])
            nw = int(np.ceil((bh * s + bw * c) / PATCH)) * PATCH
            nh = int(np.ceil((bh * c + bw * s) / PATCH)) * PATCH
            M[0, 2] += nw / 2 - bw / 2
            M[1, 2] += nh / 2 - bh / 2
            img = cv2.warpAffine(base, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
            mask = cv2.warpAffine(ones, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
            valid = self._valid(mask)
            tok = self.tokens(img)[valid]
            pts = self._centres(nh, nw)[valid]
            Minv = cv2.invertAffineTransform(M)
            pts_base = pts @ Minv[:, :2].T + Minv[:, 2]
            rots.append((ang, tok, pts_base.astype(np.float32)))
        return {"base_size": (bw, bh), "rots": rots}

    def sat_feats(self, raster, tile, scale: float) -> dict:
        half = int(round(scale * (tile.row1 - tile.row0) / 2))
        cr, cc = int(round(tile.center_row)), int(round(tile.center_col))
        r0, c0 = cr - half, cc - half
        size = 2 * half
        crop = np.zeros((size, size, 3), np.uint8)
        valid = np.zeros((size, size), np.float32)
        a0, b0 = max(0, r0), max(0, c0)
        a1, b1 = min(raster.height, r0 + size), min(raster.width, c0 + size)
        if a1 > a0 and b1 > b0:
            crop[a0 - r0 : a1 - r0, b0 - c0 : b1 - c0] = np.asarray(raster.sat[a0:a1, b0:b1, :3])
            valid[a0 - r0 : a1 - r0, b0 - c0 : b1 - c0] = 1.0
        n = self.sat_px
        img = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA)
        vmask = cv2.resize(valid, (n, n), interpolation=cv2.INTER_AREA)
        v = self._valid(vmask)
        return {
            "tok": self.tokens(img)[v],
            "pts": self._centres(n, n)[v],
            "r0": r0,
            "c0": c0,
            "px_per_crop_px": size / n,
        }

    def match(self, qf: dict, sf: dict) -> dict | None:
        """Best plausible similarity fit over query rotations; None if no plausible fit."""
        bw, bh = qf["base_size"]
        qc = np.array([bw / 2, bh / 2], np.float32)
        best = None
        if len(sf["tok"]) == 0:
            return None
        for ang, qt, qp in qf["rots"]:
            if len(qt) == 0:
                continue
            sim = qt @ sf["tok"].T
            a = sim.argmax(1)
            b = sim.argmax(0)
            mutual = np.flatnonzero(b[a] == np.arange(len(a)))
            if len(mutual) < self.min_fit:
                continue
            src = qp[mutual]
            dst = sf["pts"][a[mutual]]
            M, mask = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.thr,
                maxIters=2000, confidence=0.999, refineIters=10,
            )
            if M is None or mask is None:
                continue
            inl = int(mask.sum())
            scale = float(np.hypot(M[0, 0], M[1, 0]))
            mc = M[:, :2] @ qc + M[:, 2]
            n = self.sat_px
            plausible = self.smin <= scale <= self.smax and 0 <= mc[0] < n and 0 <= mc[1] < n
            if plausible and (best is None or inl > best["inliers"]):
                best = {
                    "inliers": inl,
                    "n_mutual": int(len(mutual)),
                    "query_rot_deg": ang,
                    "sim_scale": round(scale, 4),
                    "sim_angle_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2),
                    "crop_xy": (float(mc[0]), float(mc[1])),
                }
        return best

    def verify(self, qf: dict, rasters, tiles, cand: list[int], scales: list[float], cache: dict) -> dict:
        best = None
        for ci in cand:
            t = tiles[ci]
            ras = rasters[t.raster_id]
            for s in scales:
                key = (t.raster_id, t.tile_id, float(s))
                if key not in cache:
                    cache[key] = self.sat_feats(ras, t, s)
                sf = cache[key]
                mkey = ("match", id(qf)) + key
                if mkey not in cache:
                    cache[mkey] = self.match(qf, sf)
                m = cache[mkey]
                if m is not None and (best is None or m["inliers"] > best["inliers"]):
                    row = sf["r0"] + m["crop_xy"][1] * sf["px_per_crop_px"]
                    col = sf["c0"] + m["crop_xy"][0] * sf["px_per_crop_px"]
                    lat, lon = pixel_to_lonlat(row, col, ras.west, ras.north, ras.sx, ras.sy)
                    best = {**m, "tile_index": ci, "context_scale": s, "pred_lat": float(lat), "pred_lon": float(lon)}
        return best or {}
