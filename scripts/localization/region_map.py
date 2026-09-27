"""Load a region's satellite search space, including region-09 four-tile union."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from tile_index import Tile, TileSpec, build_tiles, georef_from_metadata, load_sat_memmap

from project_paths import DATASET_ROOT, SAT_META


@dataclass
class Raster:
    raster_id: int
    path: Path
    filename: str
    west: float
    north: float
    sx: float
    sy: float
    height: int
    width: int
    sat: object  # memmap


def sat_meta_rows() -> list[dict]:
    with SAT_META.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def region_sat_rows(region: str) -> list[dict]:
    rows = []
    for r in sat_meta_rows():
        p = r["satellite_path"].replace("/", "\\")
        if f"\\{region}\\" in p:
            rows.append(r)
    if not rows:
        raise FileNotFoundError(f"No satellite metadata for region {region}")
    if region != "09":
        rows = [x for x in rows if "satellite09_" not in x["filename"]]
        if not rows:
            raise FileNotFoundError(f"No canonical satellite for region {region}")
    else:
        rows = sorted(rows, key=lambda x: x["filename"])
        if len(rows) != 4:
            raise RuntimeError(f"Region 09 expected 4 tiles, got {len(rows)}")
    return rows


def _retag(tiles: list[Tile], raster_id: int, tid0: int, west, north, sx, sy) -> list[Tile]:
    out = []
    for i, t in enumerate(tiles):
        out.append(
            Tile(
                tid0 + i,
                t.row0,
                t.col0,
                t.row1,
                t.col1,
                t.center_row,
                t.center_col,
                t.center_lat,
                t.center_lon,
                raster_id=raster_id,
                west=west,
                north=north,
                sx=sx,
                sy=sy,
            )
        )
    return out


def build_tiles_adaptive(
    h: int,
    w: int,
    west: float,
    north: float,
    sx: float,
    sy: float,
    spec: TileSpec,
    raster_id: int,
    tid0: int,
) -> list[Tile]:
    """Frozen 1024/512 grid when the raster is large enough; shrink only if a side is < tile_size."""
    if h >= spec.tile_size and w >= spec.tile_size:
        return _retag(build_tiles(h, w, west, north, sx, sy, spec), raster_id, tid0, west, north, sx, sy)

    ts_r, ts_c = min(spec.tile_size, h), min(spec.tile_size, w)
    if ts_r < 16 or ts_c < 16:
        return []
    st_r = min(spec.stride, max(1, ts_r // 2))
    st_c = min(spec.stride, max(1, ts_c // 2))
    tiles: list[Tile] = []
    tid = tid0
    for r0 in range(0, max(1, h - ts_r + 1), st_r):
        for c0 in range(0, max(1, w - ts_c + 1), st_c):
            r1, c1 = r0 + ts_r, c0 + ts_c
            cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
            tiles.append(
                Tile(
                    tid, r0, c0, r1, c1, cr, cc,
                    north - cr * sy, west + cc * sx,
                    raster_id=raster_id, west=west, north=north, sx=sx, sy=sy,
                )
            )
            tid += 1
    return tiles


def load_region(region: str, spec: TileSpec) -> tuple[list[Raster], list[Tile]]:
    rasters: list[Raster] = []
    tiles: list[Tile] = []
    for i, row in enumerate(region_sat_rows(region)):
        path = Path(row["satellite_path"])
        if not path.is_file():
            path = DATASET_ROOT / region / row["filename"]
        west, north, sx, sy = georef_from_metadata(row)
        sat, (h, w) = load_sat_memmap(path)
        rasters.append(
            Raster(i, path, row["filename"], west, north, sx, sy, h, w, sat)
        )
        part = build_tiles_adaptive(h, w, west, north, sx, sy, spec, i, len(tiles))
        tiles.extend(part)
    return rasters, tiles


def tile_to_dict(t: Tile) -> dict:
    return asdict(t)


def tiles_from_dicts(rows: list[dict]) -> list[Tile]:
    out = []
    for r in rows:
        out.append(
            Tile(
                int(r["tile_id"]),
                int(r["row0"]),
                int(r["col0"]),
                int(r["row1"]),
                int(r["col1"]),
                float(r["center_row"]),
                float(r["center_col"]),
                float(r["center_lat"]),
                float(r["center_lon"]),
                raster_id=int(r.get("raster_id", 0)),
                west=float(r.get("west", 0)),
                north=float(r.get("north", 0)),
                sx=float(r.get("sx", 0)),
                sy=float(r.get("sy", 0)),
            )
        )
    return out


def read_tile_rgb_region(rasters: list[Raster], tile: Tile):
    import numpy as np

    sat = rasters[tile.raster_id].sat
    return np.asarray(sat[tile.row0 : tile.row1, tile.col0 : tile.col1, :3])
