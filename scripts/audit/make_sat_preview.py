"""Windowed satellite TIFF tile previews (never load full raster into RAM).

tifffile page.segments() yields (decoded_array, index_tuple, shape_tuple).
Decoded tiles are typically (1, tile_h, tile_w, bands) uint8.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root

import numpy as np
import tifffile
from PIL import Image

SAT = ROOT / "datasets/raw/UAV_VisLoc_dataset/03/satellite03.tif"
FIGS = ROOT / "figures/sample_audit"
FIGS.mkdir(parents=True, exist_ok=True)


def to_rgb(tile) -> np.ndarray | None:
    arr = np.asarray(tile)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr[..., :3]
    if arr.ndim == 3 and arr.shape[0] >= 3:
        return np.transpose(arr[:3], (1, 2, 0))
    return None


with tifffile.TiffFile(SAT) as tif:
    page = tif.pages[0]
    print(
        "tiled",
        page.is_tiled,
        "tile",
        page.tilewidth,
        page.tilelength,
        "shape",
        page.shape,
    )
    segs = list(page.segments())
    print("n_segments", len(segs))

    for label, idx in [("ul", 0), ("mid", len(segs) // 2)]:
        decoded = segs[idx][0]
        rgb = to_rgb(decoded)
        print(label, "decoded_shape", getattr(decoded, "shape", None), "rgb", None if rgb is None else rgb.shape)
        if rgb is None:
            continue
        out = FIGS / f"satellite03_{label}_tile_preview.jpg"
        Image.fromarray(rgb.astype(np.uint8)).resize((512, 512)).save(out, quality=85)
        print("wrote", out)

print("written", sorted(p.name for p in FIGS.glob("satellite*.jpg")))
