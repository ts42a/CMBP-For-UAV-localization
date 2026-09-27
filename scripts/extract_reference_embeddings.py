#!/usr/bin/env python3
"""Extract and cache satellite tile embeddings (development regions only)."""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "localization"))

from dino_embedder import DinoGlobalEmbedder, save_embedding_cache  # noqa: E402
from leakage import assert_dev_only  # noqa: E402
from project_paths import CACHE_EMB, DATASET_ROOT, MATCHER_PROTOCOL, SAT_META  # noqa: E402
from tile_index import TileSpec, build_tiles, georef_from_metadata, load_sat_memmap, read_tile_rgb  # noqa: E402

import yaml  # noqa: E402


def sat_row(meta_rows, region: str) -> dict:
    for r in meta_rows:
        p = r["satellite_path"].replace("/", "\\")
        if f"\\{region}\\" in p and "satellite09_" not in r["filename"]:
            return r
    raise FileNotFoundError(region)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="03")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with MATCHER_PROTOCOL.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert_dev_only(args.region, cfg, "embedding extraction")

    cache_name = f"region{args.region}_dinov2_vits14_{args.image_size}.npz"
    out = CACHE_EMB / cache_name
    if out.exists() and not args.force:
        print(f"Cache exists: {out}")
        return 0

    with SAT_META.open(encoding="utf-8", newline="") as f:
        meta_rows = list(csv.DictReader(f))
    sat_meta = sat_row(meta_rows, args.region)
    sat_path = __import__("pathlib").Path(sat_meta["satellite_path"])
    if not sat_path.is_file():
        sat_path = DATASET_ROOT / args.region / sat_meta["filename"]

    west, north, sx, sy = georef_from_metadata(sat_meta)
    spec = TileSpec(
        int(cfg["satellite_tiling"]["tile_size_px"]),
        int(cfg["satellite_tiling"]["stride_px"]),
    )
    sat, (h, w) = load_sat_memmap(sat_path)
    tiles = build_tiles(h, w, west, north, sx, sy, spec)
    print(f"Tiles: {len(tiles)}", flush=True)

    embedder = DinoGlobalEmbedder(image_size=args.image_size)
    rgbs = []
    for i, t in enumerate(tiles):
        if i and i % 200 == 0:
            print(f"  read {i}/{len(tiles)}", flush=True)
        rgbs.append(read_tile_rgb(sat, t))
    embs = embedder.embed_rgb_arrays(rgbs, batch_size=args.batch_size)
    tile_ids = np.array([t.tile_id for t in tiles], dtype=np.int32)
    save_embedding_cache(
        out,
        embs,
        tile_ids,
        embedder.meta(),
        extra={"region": args.region, "n_tiles": len(tiles), "georef": [west, north, sx, sy]},
    )
    print(f"Wrote {out} shape {embs.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
