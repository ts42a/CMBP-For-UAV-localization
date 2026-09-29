#!/usr/bin/env python3
"""H1 synthetic UAV query cut from the satellite map (configs/h1_protocol_v1.yaml).

synth_query() returns a BGR image the frozen localizer can consume exactly like a real frame:
a rotated, slightly tilted rectangle of the region's satellite map centred on the frame's CSV
position, sized from height above ground, with seeded appearance changes. Nothing here reads a
localizer outcome.

  py -3 scripts/h1/synthetic_query.py --region 03 --n 6       # real | synthetic panels for review
Writes figures/h1_examples/<region>_<filename>.jpg.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))
H1_PROTOCOL = ROOT / "configs" / "h1_protocol_v1.yaml"
AGL_CSV = ROOT / "datasets" / "manifests" / "frame_agl.csv"
M_LAT, M_LON_EQ = 110_574.0, 111_320.0


@lru_cache(maxsize=1)
def protocol() -> dict:
    return yaml.safe_load(H1_PROTOCOL.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def agl_table() -> dict:
    import csv
    with AGL_CSV.open(encoding="utf-8", newline="") as f:
        return {(r["region"], r["filename"]): float(r["agl_m"]) for r in csv.DictReader(f)}


def frame_rng(region: str, filename: str) -> np.random.Generator:
    seed = int(protocol()["generator"]["seed"])
    h = hashlib.sha256(f"{seed}|{region}|{filename}|h1".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def draw_params(region: str, filename: str) -> dict:
    """All random choices for one frame, drawn in a fixed order from the frame's own stream."""
    g = protocol()["generator"]
    geo, a = g["geometry"], g["appearance"]
    rng = frame_rng(region, filename)
    u = lambda lo_hi: float(rng.uniform(float(lo_hi[0]), float(lo_hi[1])))  # noqa: E731
    return {
        "scale": float(np.exp(rng.normal(0.0, float(g["footprint"]["scale_jitter_log_sd"])))),
        "rotation_deg": u(geo["rotation_deg"]),
        "tilt_deg": u(geo["tilt_deg"]),
        "tilt_dir_deg": u((0.0, 360.0)),
        "brightness": u(a["brightness_shift"]),
        "contrast": u(a["contrast"]),
        "gamma": u(a["gamma"]),
        "saturation": u(a["saturation"]),
        "hue_deg": u(a["hue_shift_deg"]),
        "blur_sigma": u(a["gaussian_blur_sigma_px"]),
        "noise_sd": u(a["gaussian_noise_sd"]),
        "jpeg_q": int(round(u(a["jpeg_quality"]))),
    }


def camera_key(image_size: tuple[int, int]) -> str:
    w, h = max(image_size), min(image_size)
    return f"{w}x{h}"


def raster_for(rasters, lat: float, lon: float):
    for r in rasters:
        col, row = (lon - r.west) / r.sx, (r.north - lat) / r.sy
        if 0 <= col < r.width and 0 <= row < r.height:
            return r, row, col
    r = rasters[0]
    return r, (r.north - lat) / r.sy, (lon - r.west) / r.sx


def synth_query(rasters, region: str, filename: str, lat: float, lon: float, image_size: tuple[int, int]) -> tuple[np.ndarray, dict]:
    g = protocol()["generator"]
    p = draw_params(region, filename)
    cam = camera_key(image_size)
    k = float(g["footprint"]["k_by_camera"][cam])
    agl = agl_table()[(region, filename)]
    out_w, out_h = (int(x) for x in g["output_px"])
    long_m = k * max(agl, 10.0) * p["scale"]

    ras, row, col = raster_for(rasters, lat, lon)
    gsd_y = ras.sy * M_LAT
    gsd_x = ras.sx * M_LON_EQ * math.cos(math.radians(lat))
    gsd = 0.5 * (gsd_x + gsd_y)
    m_per_out_px = long_m / out_w
    s = m_per_out_px / gsd                        # sat px per output px
    half = int(math.ceil(0.5 * math.hypot(out_w, out_h) * s * 1.15)) + 4
    r0, c0 = int(round(row)) - half, int(round(col)) - half
    size = 2 * half
    win = np.zeros((size, size, 3), np.uint8)
    valid = np.zeros((size, size), np.uint8)
    a0, b0 = max(0, r0), max(0, c0)
    a1, b1 = min(ras.height, r0 + size), min(ras.width, c0 + size)
    if a1 > a0 and b1 > b0:
        win[a0 - r0 : a1 - r0, b0 - c0 : b1 - c0] = np.asarray(ras.sat[a0:a1, b0:b1, :3])
        valid[a0 - r0 : a1 - r0, b0 - c0 : b1 - c0] = 255
    win = cv2.cvtColor(win, cv2.COLOR_RGB2BGR)

    # output px -> window px: rotate about the output centre, scale by s, translate to the frame centre
    th = math.radians(p["rotation_deg"])
    cx_w, cy_w = col - c0, row - r0
    A = np.array([[s * math.cos(th), -s * math.sin(th)], [s * math.sin(th), s * math.cos(th)]])
    t = np.array([cx_w, cy_w]) - A @ np.array([out_w / 2, out_h / 2])
    M = np.hstack([A, t[:, None]])
    img = cv2.warpAffine(win, M, (out_w, out_h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA, borderValue=0)
    cov = cv2.warpAffine(valid, M, (out_w, out_h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST, borderValue=0)

    # keystone tilt: shrink the far edge by tan(tilt) in a random direction
    if p["tilt_deg"] > 0:
        f = math.tan(math.radians(p["tilt_deg"]))
        d = math.radians(p["tilt_dir_deg"])
        src = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
        cxy = np.float32([out_w / 2, out_h / 2])
        dirv = np.float32([math.cos(d), math.sin(d)])
        dst = []
        for q in src:
            along = float(np.dot(q - cxy, dirv)) / (0.5 * math.hypot(out_w, out_h))
            dst.append(cxy + (q - cxy) * (1 - 0.5 * f * along))
        H = cv2.getPerspectiveTransform(src, np.float32(dst))
        img = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0)
        cov = cv2.warpPerspective(cov, H, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)

    # appearance
    x = img.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0] + p["hue_deg"]) % 360.0
    hsv[..., 1] = np.clip(hsv[..., 1] * p["saturation"], 0, 1)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    x = np.clip((x - 0.5) * p["contrast"] + 0.5 + p["brightness"], 0, 1) ** p["gamma"]
    if p["blur_sigma"] > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), p["blur_sigma"])
    x = x * 255.0 + frame_rng(region, filename + "|noise").normal(0.0, p["noise_sd"], x.shape)
    img = np.clip(x, 0, 255).astype(np.uint8)
    img[cov == 0] = 0
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_q"]])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    meta = {**p, "camera": cam, "k": k, "agl_m": agl, "footprint_long_m": round(long_m, 1),
            "coverage": round(float((cov > 0).mean()), 4)}
    return img, meta


def main() -> int:
    import csv
    from PIL import Image
    from project_paths import DATASET_ROOT, MANIFEST
    from region_map import load_region
    from tile_index import TileSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="figures/h1_examples")
    args = ap.parse_args()
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["region_folder"] == args.region]
    pick = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, args.n).round()]
    rasters, _ = load_region(args.region, TileSpec(1024, 512))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for r in pick:
        path = DATASET_ROOT / args.region / "drone" / r["filename"]
        real = cv2.imread(str(path))
        syn, meta = synth_query(rasters, args.region, r["filename"], float(r["lat"]), float(r["lon"]), Image.open(path).size)
        h = 500
        a = cv2.resize(real, (int(real.shape[1] * h / real.shape[0]), h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(syn, (int(syn.shape[1] * h / syn.shape[0]), h), interpolation=cv2.INTER_AREA)
        panel = np.hstack([a, np.full((h, 10, 3), 255, np.uint8), b])
        cv2.putText(panel, f"real {r['filename']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(panel, f"synthetic: {meta['footprint_long_m']:.0f} m, rot {meta['rotation_deg']:.0f}, cov {meta['coverage']:.2f}",
                    (a.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"{args.region}_{Path(r['filename']).stem}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(r["filename"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
