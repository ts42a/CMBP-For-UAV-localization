"""Satellite tile grid for a single-region map (Stage 3A)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True)
class TileSpec:
    tile_size: int
    stride: int


@dataclass
class Tile:
    tile_id: int
    row0: int
    col0: int
    row1: int
    col1: int
    center_row: float
    center_col: float
    center_lat: float
    center_lon: float
    raster_id: int = 0
    west: float = 0.0
    north: float = 0.0
    sx: float = 0.0
    sy: float = 0.0


def load_sat_memmap(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray(out="memmap")
    h, w = arr.shape[:2]
    return arr, (h, w)


def georef_from_metadata(row: dict) -> tuple[float, float, float, float]:
    west = float(row["bounds_west"])
    north = float(row["bounds_north"])
    ps = row["pixel_scale"].strip().strip("()")
    parts = [p.strip() for p in ps.split(",")]
    sx, sy = float(parts[0]), float(parts[1])
    return west, north, sx, sy


def build_tiles(
    h: int,
    w: int,
    west: float,
    north: float,
    sx: float,
    sy: float,
    spec: TileSpec,
) -> list[Tile]:
    tiles: list[Tile] = []
    tid = 0
    ts, st = spec.tile_size, spec.stride
    for r0 in range(0, max(1, h - ts + 1), st):
        for c0 in range(0, max(1, w - ts + 1), st):
            r1, c1 = min(h, r0 + ts), min(w, c0 + ts)
            if r1 - r0 < ts // 2 or c1 - c0 < ts // 2:
                continue
            cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
            lat = north - cr * sy
            lon = west + cc * sx
            tiles.append(
                Tile(
                    tid, r0, c0, r1, c1, cr, cc, lat, lon,
                    raster_id=0, west=west, north=north, sx=sx, sy=sy,
                )
            )
            tid += 1
    return tiles


def read_tile_rgb(sat: np.ndarray, tile: Tile) -> np.ndarray:
    return np.asarray(sat[tile.row0 : tile.row1, tile.col0 : tile.col1, :3])
