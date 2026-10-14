#!/usr/bin/env python3
"""Stage 6-7 map-only features per windowed attempt (configs/predictor_protocol_v1.yaml).

Inputs per attempt: the prior centre, the radius and the region's satellite map. Ground truth,
the UAV image and every localizer output are excluded; only the label column comes from the
outcomes. DINO features reuse the cached tile embeddings, so no model is run.

  py -3 scripts/predictor/map_features.py --outcomes results/stage5/pilot/outcomes.csv --out results/stage6_7/pilot
Writes <out>/features.csv, <out>/embeddings.npz, <out>/run_manifest.json and
figures/<fig-name>/distinct_vs_radius.png.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))
from dino_embedder import load_embedding_cache  # noqa: E402
from project_paths import CACHE_EMB  # noqa: E402
from region_map import load_region, read_tile_rgb_region  # noqa: E402
from tile_index import TileSpec  # noqa: E402
from windowed_matcher import M_PER_DEG_LAT, M_PER_DEG_LON_EQ, RegionIndex  # noqa: E402
from integrity import read_csv, validate_outcomes, encoded_keys

PROTOCOL = ROOT / "configs" / "predictor_protocol_v1.yaml"
TEX = ("entropy", "edge_density", "grad_mean", "lap_var", "valid_frac")
KEEP = ("region", "group_7", "filename", "num", "variant", "radius_m", "offset_k", "center_lat", "center_lon")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def tile_texture(rgb: np.ndarray, side: int = 256) -> dict:
    valid = cv2.resize(rgb.any(axis=2).astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST).astype(bool)
    grey = cv2.resize(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY), (side, side), interpolation=cv2.INTER_AREA)
    n = int(valid.sum())
    if n == 0:
        return {k: 0.0 for k in TEX}
    hist = np.bincount(grey[valid], minlength=256).astype(np.float64) / n
    nz = hist[hist > 0]
    g = grey.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "entropy": float(-(nz * np.log2(nz)).sum()),
        "edge_density": float((cv2.Canny(grey, 50, 150) > 0)[valid].mean()),
        "grad_mean": float(np.hypot(gx, gy)[valid].mean()),
        "lap_var": float(cv2.Laplacian(g, cv2.CV_32F, ksize=3)[valid].var()),
        "valid_frac": n / valid.size,
    }


def dist_m(idx: RegionIndex, lat0: float, lon0: float) -> np.ndarray:
    dy = (idx.lat - lat0) * M_PER_DEG_LAT
    dx = (idx.lon - lon0) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
    return np.hypot(dx, dy)



class MapFeatureService:
    """Map-only features at arbitrary estimated centres. Bounded per-region texture cache."""
    def __init__(self,idx,extent_add=450,exclusion_m=400,cache_limit=2048):
        from collections import OrderedDict
        self.idx=idx;self.extent_add=extent_add;self.exclusion_m=exclusion_m
        self.texture_cache=OrderedDict();self.cache_limit=cache_limit

    def texture(self,i):
        if i not in self.texture_cache:
            self.texture_cache[i]=tile_texture(read_tile_rgb_region(self.idx.rasters,self.idx.tiles[i]))
            if len(self.texture_cache)>self.cache_limit:self.texture_cache.popitem(last=False)
        else:self.texture_cache.move_to_end(i)
        return self.texture_cache[i]

    def extract(self,lat,lon,radius):
        if not np.isfinite([lat,lon,radius]).all() or radius<=0 or abs(lat)>90 or abs(lon)>180:
            raise ValueError('invalid feature query centre/radius')
        idx=self.idx;embs=idx.embs
        win=idx.window(lat,lon,radius);near=int(np.argmin(dist_m(idx,lat,lon)))
        members=list(win) if len(win) else [near]
        row={'log_radius':float(np.log(radius)),'n_window_tiles':len(win)}
        near_t=self.texture(near);win_t=[self.texture(int(i)) for i in members]
        for k in TEX:
            row[f'tex_near_{k}']=near_t[k]
            row[f'tex_win_{k}']=float(np.mean([t[k] for t in win_t]))
        ext=idx.window(lat,lon,radius+self.extent_add)
        ref_d=dist_m(idx,idx.lat[near],idx.lon[near])
        comp=[int(i) for i in ext if ref_d[i]>self.exclusion_m]
        if comp:
            cos=np.sort(embs[comp]@embs[near])[::-1]
            row.update(distinct_max=float(1-cos[0]),distinct_top5=float(1-cos[:5].mean()),no_competitor=0,n_competitors=len(comp))
        else:row.update(distinct_max=1.,distinct_top5=1.,no_competitor=1,n_competitors=0)
        return row,embs[near],embs[members].mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default="results/stage5/pilot/outcomes.csv")
    ap.add_argument("--out", default="results/stage6_7/pilot")
    ap.add_argument("--fig-name", default="stage7_pilot")
    args = ap.parse_args()

    proto = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    excluded = set(proto["inputs"]["excluded_regions"])
    d = proto["distinctiveness"]
    ext_add, excl_m = float(d["search_extent_m"].split("+")[1]), float(d["exclusion_m"])
    spec = TileSpec(int(proto["tiling"]["tile_size_px"]), int(proto["tiling"]["stride_px"]))

    outcomes_path = ROOT / args.outcomes
    rows = read_csv(outcomes_path)
    prereg = yaml.safe_load((ROOT/proto['inputs']['prereg']).read_text())
    validate_outcomes(rows, expected={'accept_rule':prereg['accept_rule']['selected'],
        'config_sha256':sha256(ROOT/prereg['inputs']['frozen_matcher'])}, tolerance=prereg['tolerances_m']['primary'])
    source_manifest = json.loads((outcomes_path.parent/'run_manifest.json').read_text())
    if source_manifest.get('prereg_sha256') != rows[0]['prereg_sha256']:
        raise ValueError('outcome rows differ from producer preregistration identity')
    rows = [r for r in rows if r['region'] not in excluded]
    if not rows:
        raise ValueError('no feature rows remain after protocol exclusions')
    rows.sort(key=lambda r: (r["region"], int(r["num"]), int(r["radius_m"]), int(r["offset_k"])))

    out_rows, emb_near, emb_win = [], [], []
    for region in sorted({r["region"] for r in rows}):
        rasters, tiles = load_region(region, spec)
        embs, _, _ = load_embedding_cache(CACHE_EMB / f"region{region}_dinov2_vits14_224.npz", tiles=tiles, rasters=rasters)
        if len(embs) != len(tiles):
            raise RuntimeError(f"region {region}: {len(embs)} embeddings != {len(tiles)} tiles")
        idx = RegionIndex(tiles, rasters, embs)
        service = MapFeatureService(idx, ext_add, excl_m)
        for r in (r for r in rows if r["region"] == region):
            features, near, window = service.extract(float(r["center_lat"]),float(r["center_lon"]),float(r["radius_m"]))
            if features['n_window_tiles'] != int(r['n_window_tiles']):
                raise RuntimeError('feature/localizer window mismatch')
            row={k:r[k] for k in KEEP}
            row['label']=int(r['outcome_primary']==proto['label']['positive'].split('==')[1].strip())
            row.update(features)
            out_rows.append(row);emb_near.append(near);emb_win.append(window)
        print(f"region {region}: {sum(r['region'] == region for r in out_rows)} rows, {len(service.texture_cache)} tiles textured",flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_sha = sha256(PROTOCOL)
    for row in out_rows:
        row["protocol_sha256"] = proto_sha
    with (out_dir / "features.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    np.savez_compressed(out_dir / "embeddings.npz", near=np.asarray(emb_near, np.float32),
                        window=np.asarray(emb_win, np.float32), attempt_keys=np.asarray(encoded_keys(out_rows)))
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_scripts_or_configs": bool(git("status", "--porcelain", "--", "scripts", "configs")),
        "protocol_sha256": proto_sha,
        "prereg_sha256": sha256(ROOT / proto["inputs"]["prereg"]),
        "outcomes_path": args.outcomes,
        "outcomes_sha256": sha256(outcomes_path),
        "excluded_regions": sorted(excluded),
        "n_rows": len(out_rows),
        "embeddings_row_order": "same as features.csv",
        "features_csv_sha256": sha256(out_dir/'features.csv'),
        "embeddings_npz_sha256": sha256(out_dir/'embeddings.npz'),
        "outcome_manifest_sha256": sha256(outcomes_path.parent/'run_manifest.json'),
        "cache_identities": {r: {'data':sha256(CACHE_EMB/f'region{r}_dinov2_vits14_224.npz'),
                                  'metadata':sha256(CACHE_EMB/f'region{r}_dinov2_vits14_224.meta.json')}
                             for r in sorted({x['region'] for x in rows})},
    }, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    radii = sorted({int(r["radius_m"]) for r in out_rows})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title in ((axes[0], "distinct_max", "distinct_max"), (axes[1], "n_competitors", "competitor tiles")):
        ax.boxplot([[r[key] for r in out_rows if int(r["radius_m"]) == R] for R in radii], tick_labels=[str(R) for R in radii])
        ax.set_xlabel("search radius (m)")
        ax.set_title(f"{title} by radius")
    fig.tight_layout()
    fig_dir = ROOT / "figures" / args.fig_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "distinct_vs_radius.png", dpi=120)
    for R in radii:
        v = np.array([r["distinct_max"] for r in out_rows if int(r["radius_m"]) == R])
        print(f"R={R}: distinct_max median {np.median(v):.4f}, no_competitor {sum(r['no_competitor'] for r in out_rows if int(r['radius_m']) == R)}")
    print(f"done: {len(out_rows)} rows -> {out_dir / 'features.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
