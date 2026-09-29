#!/usr/bin/env python3
"""Height above ground per frame: CSV height minus SRTM terrain elevation (label-free audit).

The CSV `height` is above sea level, not above ground (regions 05 and 11 fly at ~2,300-2,600 m),
so it cannot describe how far the camera is from the ground. This audit samples SRTM 1-arc-second
terrain (public AWS "skadi" HGT tiles, EGM96 orthometric metres) bilinearly at every frame's CSV
position and writes AGL = height - terrain. No localizer outcome is read.

  py -3 scripts/audit/agl_height.py
Writes datasets/manifests/frame_agl.csv (one row per manifest frame) and reports/agl_height_audit.{md,json}.
Tiles are cached in datasets/cache/dem/ (git-ignored).

Caveats (in the report): the CSV height datum (ellipsoidal vs mean sea level) is undocumented;
over these regions the geoid-ellipsoid separation is of the order of tens of metres, so AGL carries
that much systematic uncertainty per region. SRTM is a surface model (canopy/buildings included),
with ~5-10 m vertical error.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "datasets" / "manifests" / "manifest.csv"
DEM_DIR = ROOT / "datasets" / "cache" / "dem"
OUT_CSV = ROOT / "datasets" / "manifests" / "frame_agl.csv"
URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{lat:02d}/{ns}{lat:02d}{ew}{lon:03d}.hgt.gz"


def tile_name(lat: float, lon: float) -> tuple[str, str]:
    la, lo = math.floor(lat), math.floor(lon)
    ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
    name = f"{ns}{abs(la):02d}{ew}{abs(lo):03d}"
    return name, URL.format(ns=ns, lat=abs(la), ew=ew, lon=abs(lo))


def load_tile(name: str, url: str) -> np.ndarray:
    path = DEM_DIR / f"{name}.hgt.gz"
    if not path.is_file():
        DEM_DIR.mkdir(parents=True, exist_ok=True)
        print(f"download {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    raw = gzip.decompress(path.read_bytes())
    n = int(round(math.sqrt(len(raw) // 2)))
    if n * n * 2 != len(raw):
        raise RuntimeError(f"{name}: unexpected HGT size {len(raw)}")
    z = np.frombuffer(raw, dtype=">i2").reshape(n, n).astype(np.float64)
    z[z == -32768] = np.nan  # voids
    return z


def sample(z: np.ndarray, lat: float, lon: float) -> float:
    """Bilinear sample; HGT row 0 is the tile's north edge, samples on a (n-1)-per-degree grid."""
    n = z.shape[0]
    fr = (math.floor(lat) + 1 - lat) * (n - 1)
    fc = (lon - math.floor(lon)) * (n - 1)
    r0, c0 = min(int(fr), n - 2), min(int(fc), n - 2)
    dr, dc = fr - r0, fc - c0
    q = z[r0 : r0 + 2, c0 : c0 + 2]
    return float(q[0, 0] * (1 - dr) * (1 - dc) + q[0, 1] * (1 - dr) * dc + q[1, 0] * dr * (1 - dc) + q[1, 1] * dr * dc)


def main() -> int:
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        frames = [r for r in csv.DictReader(f) if r["metadata_valid_coords"] == "True"]
    tiles: dict[str, np.ndarray] = {}
    rows = []
    for r in frames:
        lat, lon, h = float(r["lat"]), float(r["lon"]), float(r["height"])
        name, url = tile_name(lat, lon)
        if name not in tiles:
            tiles[name] = load_tile(name, url)
        terr = sample(tiles[name], lat, lon)
        rows.append({"region": r["region_folder"], "filename": r["filename"], "num": r["num"],
                     "height_csv_m": h, "terrain_srtm_m": round(terr, 2),
                     "agl_m": round(h - terr, 2) if not math.isnan(terr) else ""})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for x in rows:
        by[x["region"]].append(x)
    summ = {}
    for reg, xs in sorted(by.items()):
        agl = np.array([x["agl_m"] for x in xs if x["agl_m"] != ""], float)
        ter = np.array([x["terrain_srtm_m"] for x in xs], float)
        hc = np.array([x["height_csv_m"] for x in xs], float)
        summ[reg] = {"n": len(xs), "voids": int(np.isnan(ter).sum()),
                     "height_csv_median": round(float(np.median(hc)), 1),
                     "terrain_median": round(float(np.nanmedian(ter)), 1),
                     "terrain_p05_p95": [round(float(np.nanpercentile(ter, 5)), 1), round(float(np.nanpercentile(ter, 95)), 1)],
                     "agl_median": round(float(np.median(agl)), 1),
                     "agl_p05_p95": [round(float(np.percentile(agl, 5)), 1), round(float(np.percentile(agl, 95)), 1)],
                     "agl_min": round(float(agl.min()), 1)}
    (ROOT / "reports" / "agl_height_audit.json").write_text(json.dumps({"tiles": sorted(tiles), "regions": summ}, indent=2), encoding="utf-8")

    md = ["# Height above ground (AGL) audit — label-free", "",
          "`scripts/audit/agl_height.py`. AGL = CSV `height` − SRTM 1″ terrain (AWS skadi tiles, EGM96 metres), "
          "bilinear at each frame's CSV position. No localizer outcome is read. Per-frame values: "
          "`datasets/manifests/frame_agl.csv`.", "",
          "| region | frames | CSV height median | terrain median (p5–p95) | **AGL median** (p5–p95) | AGL min | SRTM voids |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for reg, s in summ.items():
        md.append(f"| {reg} | {s['n']} | {s['height_csv_median']} | {s['terrain_median']} ({s['terrain_p05_p95'][0]}–{s['terrain_p05_p95'][1]}) | "
                  f"**{s['agl_median']}** ({s['agl_p05_p95'][0]}–{s['agl_p05_p95'][1]}) | {s['agl_min']} | {s['voids']} |")
    md += ["", "## Caveats", "",
           "- The CSV height datum (ellipsoidal or mean sea level) is undocumented. Geoid–ellipsoid separation over these "
           "regions is of the order of tens of metres, so each region's AGL carries that much systematic offset.",
           "- SRTM is a surface model (canopy and buildings included), ~5–10 m vertical error; it dates from 2000.",
           "- AGL is not a footprint: no focal length or sensor size is published (Stage 2A/2D), so footprint = k × AGL "
           "needs one scale constant k per camera.",
           "- Negative or near-zero AGL would mean a datum mismatch or a wrong CSV height; see the AGL min column."]
    (ROOT / "reports" / "agl_height_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
