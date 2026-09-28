#!/usr/bin/env python3
"""Stage 10: does storing the tile-embedding cache in FP16 change retrieval?

For every benchmark attempt (configs/stage10_benchmark.yaml), rebuild the window around the
recorded prior centre and compare the top-k retrieval candidates computed from the FP32 cache
with those from an FP16 round-trip of the same cache. FP16 is acceptable only if every ordered
top-k list is identical (the verifier then sees exactly the same candidates).

  py -3 scripts/protocol/check_fp16_cache.py
Writes reports/stage10_fp16_cache.json.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "localization"))
from dino_embedder import DinoGlobalEmbedder, load_embedding_cache  # noqa: E402
from frozen_matcher import query_embeddings  # noqa: E402
from project_paths import CACHE_EMB, DATASET_ROOT  # noqa: E402
from query_select import load_manifest_region, select_queries  # noqa: E402
from region_map import load_region  # noqa: E402
from tile_index import TileSpec  # noqa: E402
from windowed_matcher import RegionIndex  # noqa: E402

BENCH_REGIONS = ["01", "02", "03", "04", "05", "06", "08", "09", "10", "11"]


def npz_bytes(arr: np.ndarray) -> int:
    buf = io.BytesIO()
    np.savez_compressed(buf, embeddings=arr)
    return buf.getbuffer().nbytes


def main() -> int:
    frozen = yaml.safe_load((ROOT / "configs" / "matcher_stage4_frozen.yaml").read_text(encoding="utf-8"))
    top_k = int(frozen["retrieval"]["top_k"])
    rots = tuple(int(x) for x in frozen["preprocessing"]["query_rotations_deg"])
    spec = TileSpec(int(frozen["retrieval"]["tiling"]["tile_size_px"]), int(frozen["retrieval"]["tiling"]["stride_px"]))
    with (ROOT / "results" / "stage5" / "pilot" / "outcomes.csv").open(encoding="utf-8", newline="") as f:
        pilot = list(csv.DictReader(f))
    embedder = DinoGlobalEmbedder(image_size=int(frozen["model"]["image_size"]), device="cpu")

    n_att = n_same_order = n_same_set = 0
    size32 = size16 = 0
    max_abs = 0.0
    diffs = []
    for region in BENCH_REGIONS:
        frames = {q["filename"] for q in select_queries(load_manifest_region(region), 2, seed=20260927)}
        rasters, tiles = load_region(region, spec)
        e32, _, _ = load_embedding_cache(CACHE_EMB / f"region{region}_dinov2_vits14_224.npz")
        e16 = e32.astype(np.float16).astype(np.float32)
        max_abs = max(max_abs, float(np.abs(e32 - e16).max()))
        size32 += npz_bytes(e32)
        size16 += npz_bytes(e32.astype(np.float16))
        idx = RegionIndex(tiles, rasters, e32)
        for fn in sorted(frames):
            q = cv2.imread(str(DATASET_ROOT / region / "drone" / fn))
            qv = query_embeddings(embedder, q, rots)
            for r in (r for r in pilot if r["region"] == region and r["filename"] == fn):
                win = idx.window(float(r["center_lat"]), float(r["center_lon"]), float(r["radius_m"]))
                n_att += 1
                if len(win) == 0:
                    n_same_order += 1
                    n_same_set += 1
                    continue
                k = min(top_k, len(win))
                s32 = np.max(e32[win] @ qv.T, axis=1)
                s16 = np.max(e16[win] @ qv.T, axis=1)
                c32 = [int(win[i]) for i in np.argsort(-s32, kind="stable")[:k]]
                c16 = [int(win[i]) for i in np.argsort(-s16, kind="stable")[:k]]
                n_same_order += c32 == c16
                n_same_set += set(c32) == set(c16)
                if c32 != c16:
                    diffs.append({"region": region, "filename": fn, "radius_m": r["radius_m"], "offset_k": r["offset_k"],
                                  "top1_same": c32[0] == c16[0], "same_set": set(c32) == set(c16)})
        print(f"region {region}: done", flush=True)

    out = {
        "attempts": n_att, "identical_ordered_topk": n_same_order, "identical_topk_set": n_same_set,
        "max_abs_embedding_change": max_abs,
        "cache_bytes_fp32_npz": size32, "cache_bytes_fp16_npz": size16,
        "fp16_accepted": n_same_order == n_att, "differences": diffs,
    }
    (ROOT / "reports" / "stage10_fp16_cache.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "differences"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
