"""WGS84 geodesic helpers for Stage 3A."""
from __future__ import annotations

from pyproj import Geod

GEOD = Geod(ellps="WGS84")


def geodesic_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    _, _, dist = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(abs(dist))


def pixel_to_lonlat(
    row: float,
    col: float,
    west: float,
    north: float,
    sx: float,
    sy: float,
) -> tuple[float, float]:
    lon = west + col * sx
    lat = north - row * sy
    return lat, lon
