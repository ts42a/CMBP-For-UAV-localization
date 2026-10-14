#!/usr/bin/env python3
"""Extract satellite tile embeddings for ALL regions after Stage 3C gate PASS.

Does not use held-out outcomes to choose a matcher. Frozen backbone only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "localization"))

from dino_embedder import DinoGlobalEmbedder, save_embedding_cache, load_embedding_cache  # noqa: E402
from project_paths import CACHE_EMB  # noqa: E402
from region_map import load_region, read_tile_rgb_region, tile_to_dict  # noqa: E402
from tile_index import TileSpec  # noqa: E402
from integrity import sha256

FROZEN = ROOT / "configs" / "matcher_stage3b_frozen.yaml"
GATE = ROOT / "reports" / "stage3c_preflight_gate.json"
REGIONS = [f"{i:02d}" for i in range(1, 12)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    if not args.skip_gate:
        if not GATE.is_file() or json.loads(GATE.read_text(encoding="utf-8")).get("gate") != "PASS":
            print("Stage 3C preflight gate is not PASS. Run stage3c_gate.py first.", file=sys.stderr)
            return 2

    frozen = yaml.safe_load(FROZEN.read_text(encoding="utf-8"))
    spec = TileSpec(
        int(frozen["retrieval"]["tiling"]["tile_size_px"]),
        int(frozen["retrieval"]["tiling"]["stride_px"]),
    )
    image_size = int(frozen["model"]["image_size"])
    wanted = REGIONS if args.regions == "all" else [x.strip() for x in args.regions.split(",")]

    embedder = None
    for region in wanted:
        out = CACHE_EMB / f"region{region}_dinov2_vits14_{image_size}.npz"
        rasters, tiles = load_region(region, spec)
        print(f"Region {region}: {len(rasters)} raster(s), {len(tiles)} tiles", flush=True)
        if out.exists() and not args.force:
            import numpy as np

            load_embedding_cache(out, tiles=tiles, rasters=rasters)
            print(f"  cache identity verified {out.name}", flush=True)
            continue
        if embedder is None:
            embedder = DinoGlobalEmbedder(image_size=image_size)
        import numpy as np

        vecs = []
        batch_rgb: list = []
        for i, t in enumerate(tiles):
            batch_rgb.append(read_tile_rgb_region(rasters, t))
            if len(batch_rgb) == args.batch_size or i + 1 == len(tiles):
                vecs.append(embedder.embed_rgb_arrays(batch_rgb, batch_size=args.batch_size))
                batch_rgb = []
            if (i + 1) % 200 == 0:
                print(f"  embedded {i+1}/{len(tiles)}", flush=True)
        embs = np.vstack(vecs) if vecs else np.zeros((0, 384), np.float32)
        save_embedding_cache(
            out,
            embs,
            np.array([t.tile_id for t in tiles], dtype=np.int32),
            embedder.meta(),
            extra={
                "region": region,
                "n_tiles": len(tiles),
                "n_rasters": len(rasters),
                "rasters": [r.filename for r in rasters],
                "tiles": [tile_to_dict(t) for t in tiles],
                "rasters_sha256": {r.filename:sha256(r.path) for r in rasters},
            },
        )
        print(f"  wrote {out} {embs.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
