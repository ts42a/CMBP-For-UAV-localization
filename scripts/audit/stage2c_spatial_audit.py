#!/usr/bin/env python3
"""Stage 2C — spatial audit and independent-region grouping."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Geod
from shapely.geometry import Point, box
from shapely.ops import transform
from pyproj import Transformer

ROOT = Path(r"F:\CMBL")
EXTRACT = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figures"
DATASET = ROOT / "datasets" / "manifests"
GEOD = Geod(ellps="WGS84")
DOCUMENTED_IMAGES = 6742
THRESHOLDS_M = [0, 1000, 5000, 10000, 25000, 50000]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_range_csv() -> dict[str, dict]:
    p = next(EXTRACT.glob("satellite*coordinates*range*.csv"))
    out = {}
    with p.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            key = r["mapname"].replace("satellite", "").replace(".tif", "").zfill(2)
            if key.isdigit():
                rid = key
            else:
                rid = r["mapname"]
            w, e = float(r["LT_lon_map"]), float(r["RB_lon_map"])
            n, s = float(r["LT_lat_map"]), float(r["RB_lat_map"])
            out[rid] = {
                "west": min(w, e),
                "east": max(w, e),
                "south": min(s, n),
                "north": max(s, n),
                "region_label": r.get("region", ""),
                "mapname": r["mapname"],
            }
    return out


def load_tile_bounds_from_metadata() -> dict[str, list[dict]]:
    tiles: dict[str, list[dict]] = defaultdict(list)
    with (DATASET / "satellite_metadata.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fn = r["filename"]
            if not fn.startswith("satellite"):
                continue
            rid = fn.replace("satellite", "").split("_")[0].split(".")[0].zfill(2)
            if not rid.isdigit():
                continue
            tiles[rid].append(
                {
                    "filename": fn,
                    "west": float(r["bounds_west"]),
                    "east": float(r["bounds_east"]),
                    "south": float(r["bounds_south"]),
                    "north": float(r["bounds_north"]),
                }
            )
    return tiles


def combined_footprint(tile_list: list[dict]) -> dict:
    return {
        "west": min(t["west"] for t in tile_list),
        "east": max(t["east"] for t in tile_list),
        "south": min(t["south"] for t in tile_list),
        "north": max(t["north"] for t in tile_list),
        "source": "union_of_tiles",
        "n_tiles": len(tile_list),
    }


def bbox_geodesic_area_m2(w: float, s: float, e: float, n: float) -> float:
    if e <= w or n <= s:
        return 0.0
    lons = [w, e, e, w, w]
    lats = [s, s, n, n, s]
    try:
        area, _ = GEOD.geometry_area_perimeter(lons, lats)
        return abs(area)
    except Exception:
        _, _, width = GEOD.inv(w, (s + n) / 2, e, (s + n) / 2)
        _, _, height = GEOD.inv((w + e) / 2, s, (w + e) / 2, n)
        return abs(width) * abs(height)


def bbox_intersection(w1, s1, e1, n1, w2, s2, e2, n2):
    iw, ie = max(w1, w2), min(e1, e2)
    is_, in_ = max(s1, s2), min(n1, n2)
    if iw >= ie or is_ >= in_:
        return None
    return iw, is_, ie, in_


def local_transformer(lon: float, lat: float) -> Transformer:
    return Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m",
        always_xy=True,
    )


def min_bbox_distance_m(b1, b2) -> float:
    """Minimum distance between two WGS84 axis-aligned boxes (metres)."""
    w1, s1, e1, n1 = b1
    w2, s2, e2, n2 = b2
    inter = bbox_intersection(w1, s1, e1, n1, w2, s2, e2, n2)
    if inter:
        return 0.0
    lon = (w1 + e1 + w2 + e2) / 4
    lat = (s1 + n1 + s2 + n2) / 4
    tr = local_transformer(lon, lat)
    poly1 = transform(tr.transform, box(w1, s1, e1, n1))
    poly2 = transform(tr.transform, box(w2, s2, e2, n2))
    return float(poly1.distance(poly2))


def load_tracks() -> dict[str, np.ndarray]:
    tracks = {}
    for rid in [f"{i:02d}" for i in range(1, 12)]:
        csv_path = EXTRACT / rid / f"{rid}.csv"
        pts = []
        with csv_path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    pts.append((float(r["lon"]), float(r["lat"])))
                except (KeyError, ValueError):
                    continue
        tracks[rid] = np.array(pts, dtype=float) if pts else np.zeros((0, 2))
    return tracks


def track_centroid(pts: np.ndarray) -> tuple[float, float]:
    if len(pts) == 0:
        return float("nan"), float("nan")
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def min_track_distance_m(ta: np.ndarray, tb: np.ndarray, step_a=1, step_b=1) -> float:
    if len(ta) == 0 or len(tb) == 0:
        return float("nan")
    ta_s = ta[:: max(1, len(ta) // 200 or 1)]
    tb_s = tb[:: max(1, len(tb) // 200 or 1)]
    best = float("inf")
    for lon1, lat1 in ta_s:
        for lon2, lat2 in tb_s:
            _, _, d = GEOD.inv(lon1, lat1, lon2, lat2)
            best = min(best, abs(d))
    return best


def geodesic_centroid_distance_m(c1, c2) -> float:
    if any(math.isnan(x) for x in (*c1, *c2)):
        return float("nan")
    _, _, d = GEOD.inv(c1[0], c1[1], c2[0], c2[1])
    return abs(d)


def fraction_track_in_map(pts: np.ndarray, mp: dict) -> float:
    if len(pts) == 0:
        return float("nan")
    inside = 0
    for lon, lat in pts:
        if mp["west"] <= lon <= mp["east"] and mp["south"] <= lat <= mp["north"]:
            inside += 1
    return inside / len(pts)


def union_find_groups(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        unite(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [sorted(g) for g in groups.values()]


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    range_maps = load_range_csv()
    tile_bounds = load_tile_bounds_from_metadata()
    tracks = load_tracks()

    regions = [f"{i:02d}" for i in range(1, 12)]
    maps = {}
    for rid in regions:
        rm = range_maps.get(rid, {})
        if rid == "09" and tile_bounds.get("09"):
            union = combined_footprint(tile_bounds["09"])
            maps[rid] = {
                **union,
                "region_label": rm.get("region_label", "Huzhou-3"),
                "range_csv_footprint": {
                    k: rm[k] for k in ("west", "east", "south", "north") if k in rm
                },
                "tile_files": [t["filename"] for t in tile_bounds["09"]],
            }
        else:
            maps[rid] = {
                "west": rm["west"],
                "east": rm["east"],
                "south": rm["south"],
                "north": rm["north"],
                "region_label": rm.get("region_label", ""),
                "source": "range_csv_geotiff_agree",
            }

    # per-region image counts (+32 note)
    image_counts = {}
    for rid in regions:
        with (EXTRACT / rid / f"{rid}.csv").open(encoding="utf-8", newline="") as f:
            image_counts[rid] = sum(1 for _ in csv.DictReader(f))

    pairwise = []
    overlap_rows = []
    idx = {r: i for i, r in enumerate(regions)}

    for i, ra in enumerate(regions):
        ma = maps[ra]
        ba = (ma["west"], ma["south"], ma["east"], ma["north"])
        for rb in regions[i + 1 :]:
            mb = maps[rb]
            bb = (mb["west"], mb["south"], mb["east"], mb["north"])
            inter = bbox_intersection(*ba, *bb)
            area_a = bbox_geodesic_area_m2(*ba)
            area_b = bbox_geodesic_area_m2(*bb)
            if inter:
                area_i = bbox_geodesic_area_m2(*inter)
                min_bd = 0.0
            else:
                area_i = 0.0
                min_bd = min_bbox_distance_m(ba, bb)
            overlap_min = area_i / min(area_a, area_b) if min(area_a, area_b) > 0 else 0
            overlap_union = area_i / (area_a + area_b - area_i) if (area_a + area_b - area_i) > 0 else 0

            ta, tb = tracks[ra], tracks[rb]
            min_tr = min_track_distance_m(ta, tb)
            ca, cb = track_centroid(ta), track_centroid(tb)
            rep_tr = geodesic_centroid_distance_m(ca, cb)
            frac_a_in_b = fraction_track_in_map(ta, mb)
            frac_b_in_a = fraction_track_in_map(tb, ma)

            pairwise.append(
                {
                    "region_a": ra,
                    "region_b": rb,
                    "label_a": ma.get("region_label", ""),
                    "label_b": mb.get("region_label", ""),
                    "map_intersection_area_m2": round(area_i, 2),
                    "map_overlap_fraction_of_smaller": round(overlap_min, 6),
                    "map_overlap_fraction_of_union": round(overlap_union, 6),
                    "map_min_boundary_distance_m": round(min_bd, 2),
                    "flight_min_point_distance_m": round(min_tr, 2),
                    "flight_centroid_distance_m": round(rep_tr, 2),
                    "fraction_track_a_in_map_b": round(frac_a_in_b, 6),
                    "fraction_track_b_in_map_a": round(frac_b_in_a, 6),
                    "maps_intersect": area_i > 0,
                }
            )
            overlap_rows.append(
                {
                    "region_a": ra,
                    "region_b": rb,
                    "intersection_m2": round(area_i, 2),
                    "area_a_m2": round(area_a, 2),
                    "area_b_m2": round(area_b, 2),
                    "overlap_vs_smaller": round(overlap_min, 6),
                    "overlap_vs_union": round(overlap_union, 6),
                }
            )

    # sensitivity grouping
    sensitivity = []
    group_records = []
    for thr in THRESHOLDS_M:
        edges = []
        cross_links = []
        for row in pairwise:
            ra, rb = row["region_a"], row["region_b"]
            ia, ib = idx[ra], idx[rb]
            if row["maps_intersect"] or row["map_min_boundary_distance_m"] <= thr:
                edges.append((ia, ib))
            if row["fraction_track_a_in_map_b"] >= 0.05 or row["fraction_track_b_in_map_a"] >= 0.05:
                cross_links.append((ra, rb, row["fraction_track_a_in_map_b"], row["fraction_track_b_in_map_a"]))
                if thr >= 5000:  # terrain-sharing link at moderate threshold only
                    edges.append((ia, ib))
        groups = union_find_groups(len(regions), edges)
        sensitivity.append(
            {
                "threshold_map_edge_or_overlap_m": thr,
                "n_groups": len(groups),
                "groups": "|".join("+".join(regions[i] for i in g) for g in groups),
                "merge_rule": "union if map intersection OR min boundary distance <= threshold; "
                "if threshold>=5000m also merge if >=5% flight points in other map",
            }
        )
        if thr == 5000:
            for gi, g in enumerate(groups):
                group_records.append(
                    {
                        "group_id": gi + 1,
                        "member_regions": "+".join(regions[i] for i in g),
                        "member_labels": "+".join(maps[regions[i]].get("region_label", "") for i in g),
                        "threshold_m": thr,
                        "rule": sensitivity[-1]["merge_rule"],
                    }
                )

    # plot
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.cm.tab10
    for i, rid in enumerate(regions):
        m = maps[rid]
        w, s, e, n = m["west"], m["south"], m["east"], m["north"]
        rect = plt.Rectangle((w, s), e - w, n - s, fill=False, edgecolor=cmap(i), linewidth=1.5, label=f"{rid} {m.get('region_label','')}")
        ax.add_patch(rect)
        if rid == "09":
            for t in tile_bounds.get("09", []):
                ax.plot(
                    [t["west"], t["east"], t["east"], t["west"], t["west"]],
                    [t["south"], t["south"], t["north"], t["north"], t["south"]],
                    color=cmap(i),
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
        pts = tracks[rid]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=1, c=[cmap(i)], alpha=0.35)
        ax.text((w + e) / 2, n, rid, ha="center", va="bottom", fontsize=8, color=cmap(i))
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("UAV-VisLoc: map footprints (solid; region 09 tiles dashed) and flight tracks")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = FIGS / "regions_and_tracks.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # write CSVs
    def wcsv(path, rows):
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    wcsv(REPORTS / "pairwise_region_distances.csv", pairwise)
    wcsv(REPORTS / "map_overlap.csv", overlap_rows)
    ind_rows = []
    for s in sensitivity:
        ind_rows.append(
            {
                "record_type": "sensitivity",
                "group_id": "",
                "threshold_m": s["threshold_map_edge_or_overlap_m"],
                "member_regions": s["groups"],
                "member_labels": "",
                "n_groups": s["n_groups"],
                "rule": s["merge_rule"],
            }
        )
    for gr in group_records:
        ind_rows.append(
            {
                "record_type": "primary_group",
                "group_id": gr["group_id"],
                "threshold_m": gr["threshold_m"],
                "member_regions": gr["member_regions"],
                "member_labels": gr["member_labels"],
                "n_groups": "",
                "rule": gr["rule"],
            }
        )
    wcsv(REPORTS / "independent_region_groups.csv", ind_rows)

    summary = {
        "generated_at": utc_now(),
        "n_regions": 11,
        "image_count_total": sum(image_counts.values()),
        "documented_images": DOCUMENTED_IMAGES,
        "image_delta": sum(image_counts.values()) - DOCUMENTED_IMAGES,
        "image_counts_by_region": image_counts,
        "region_09_tiles": maps["09"].get("tile_files", []),
        "region_09_union_equals_range_csv": maps["09"].get("range_csv_footprint"),
        "sensitivity": sensitivity,
        "primary_grouping_threshold_m": 5000,
        "n_groups_at_5000m": next(s["n_groups"] for s in sensitivity if s["threshold_map_edge_or_overlap_m"] == 5000),
    }
    (REPORTS / "stage2c_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
