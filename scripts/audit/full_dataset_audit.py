#!/usr/bin/env python3
"""Stage 2B full-release inventory and stats for UAV-VisLoc.

Streaming / incremental processing suitable for ~16 GB RAM.
Does not load full satellite GeoTIFFs into memory.
"""
from __future__ import annotations

import csv
import hashlib
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(r"F:\CMBL")
RAW = ROOT / "datasets" / "raw"
FULL = ROOT / "datasets" / "full"
FULL_EXTRACT = RAW / "UAV_VisLoc_dataset"
SAMPLE = ROOT / "datasets" / "sample"
REPORTS = ROOT / "reports"
MANIFESTS = ROOT / "datasets" / "manifests"
ARCHIVE = RAW / "UAV_VisLoc_dataset.zip"
DOCUMENTED_UAV = 6742
DOCUMENTED_MAPS = 11

for p in (FULL, REPORTS, MANIFESTS):
    p.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest().upper()


def find_extract_root() -> Path | None:
    """Locate extracted GitHub-layout tree (01..11 under datasets/full or datasets/raw/UAV_VisLoc_dataset)."""
    for base in (FULL, FULL_EXTRACT):
        if not base.is_dir():
            continue
        candidates = [base]
        for child in base.iterdir():
            if child.is_dir():
                candidates.append(child)
        for c in candidates:
            if (c / "01").is_dir() and (c / "11").is_dir():
                return c
        for c in candidates:
            nums = [p.name for p in c.iterdir() if p.is_dir() and p.name.isdigit()]
            if len(nums) >= 5:
                return c
    return None


def list_range_csv(root: Path) -> Path | None:
    hits = list(root.glob("satellite*coordinates*range*.csv"))
    return hits[0] if hits else None


def inspect_satellite(path: Path, range_row: dict | None) -> dict:
    out = {
        "satellite_path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size if path.exists() else "",
        "exists": path.exists(),
        "shape_h": "",
        "shape_w": "",
        "bands": "",
        "dtype": "",
        "tiled": "",
        "tile_w": "",
        "tile_h": "",
        "nodata": "",
        "crs_epsg": "",
        "geo_ascii": "",
        "gt_raster_type": "",
        "pixel_is_area": "",
        "tiepoint": "",
        "pixel_scale": "",
        "bounds_west": "",
        "bounds_east": "",
        "bounds_north": "",
        "bounds_south": "",
        "csv_LT_lat": "",
        "csv_LT_lon": "",
        "csv_RB_lat": "",
        "csv_RB_lon": "",
        "csv_vs_tif_delta_max": "",
        "georef_status": "",
        "error": "",
    }
    if not path.exists():
        out["georef_status"] = "missing_file"
        return out
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            shape = page.shape
            if len(shape) == 3:
                h, w, bands = shape
            elif len(shape) == 2:
                h, w, bands = shape[0], shape[1], 1
            else:
                h, w, bands = shape[0], shape[1] if len(shape) > 1 else 1, 1
            out.update(
                {
                    "shape_h": h,
                    "shape_w": w,
                    "bands": bands,
                    "dtype": str(page.dtype),
                    "tiled": page.is_tiled,
                    "tile_w": page.tilewidth,
                    "tile_h": page.tilelength,
                }
            )
            tags = page.tags
            if "GDAL_NODATA" in tags:
                out["nodata"] = tags["GDAL_NODATA"].value
            if "ModelTiepointTag" in tags:
                out["tiepoint"] = str(tags["ModelTiepointTag"].value)
            if "ModelPixelScaleTag" in tags:
                out["pixel_scale"] = str(tags["ModelPixelScaleTag"].value)
            if "GeoAsciiParamsTag" in tags:
                out["geo_ascii"] = str(tags["GeoAsciiParamsTag"].value)
            if "GeoKeyDirectoryTag" in tags:
                keys = tags["GeoKeyDirectoryTag"].value
                # GeoKey directory: [version, rev, numkeys, ...] then triples
                try:
                    arr = list(keys)
                    i = 4
                    while i + 3 < len(arr):
                        key, loc, count, val = arr[i : i + 4]
                        if key == 2048:  # GeographicTypeGeoKey
                            out["crs_epsg"] = val
                        if key == 1025:  # GTRasterTypeGeoKey
                            out["gt_raster_type"] = val
                            out["pixel_is_area"] = val == 1
                        i += 4
                except Exception as e:
                    out["error"] = f"geokey_parse:{e}"
            # bounds from tiepoint + scale + dims (PixelIsArea edge)
            if "ModelTiepointTag" in tags and "ModelPixelScaleTag" in tags:
                tp = tags["ModelTiepointTag"].value
                sc = tags["ModelPixelScaleTag"].value
                # tp: I,J,K, X,Y,Z — usually 0,0,0, lon, lat, 0
                west = float(tp[3])
                north = float(tp[4])
                sx, sy = float(sc[0]), float(sc[1])
                east = west + sx * w
                south = north - sy * h
                out.update(
                    {
                        "bounds_west": west,
                        "bounds_east": east,
                        "bounds_north": north,
                        "bounds_south": south,
                        "georef_status": "embedded_geotiff",
                    }
                )
            else:
                out["georef_status"] = "no_model_tags"
        if range_row:
            out["csv_LT_lat"] = range_row.get("LT_lat_map", "")
            out["csv_LT_lon"] = range_row.get("LT_lon_map", "")
            out["csv_RB_lat"] = range_row.get("RB_lat_map", "")
            out["csv_RB_lon"] = range_row.get("RB_lon_map", "")
            try:
                deltas = [
                    abs(float(out["bounds_west"]) - float(out["csv_LT_lon"])),
                    abs(float(out["bounds_north"]) - float(out["csv_LT_lat"])),
                    abs(float(out["bounds_east"]) - float(out["csv_RB_lon"])),
                    abs(float(out["bounds_south"]) - float(out["csv_RB_lat"])),
                ]
                out["csv_vs_tif_delta_max"] = max(deltas)
            except Exception:
                out["csv_vs_tif_delta_max"] = "compare_failed"
    except Exception as e:
        out["error"] = str(e)
        out["georef_status"] = "read_error"
    return out


def quantiles(vals: list[float], qs=(0.5, 0.9, 0.95, 0.99)) -> dict:
    if not vals:
        return {f"q{int(q*100)}": "" for q in qs}
    s = sorted(vals)
    out = {}
    for q in qs:
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        out[f"q{int(q*100)}"] = s[idx]
    return out


def parse_ts(s: str):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def inventory_region(region_id: str, region_dir: Path, range_by_map: dict, issues: list) -> dict:
    drone_dir = region_dir / "drone"
    csv_path = region_dir / f"{region_id}.csv"
    sat_path = region_dir / f"satellite{region_id}.tif"
    jpgs = {}
    if drone_dir.is_dir():
        for p in drone_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}:
                jpgs[p.name.upper()] = p

    rows = []
    headers = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            rows = list(reader)
    else:
        issues.append(
            {
                "severity": "definite",
                "reason_code": "missing_region_csv",
                "region": region_id,
                "item": str(csv_path),
                "evidence": "expected XX.csv absent",
            }
        )

    # mapname alias: try 03.tif and satellite03.tif
    range_row = range_by_map.get(f"{region_id}.tif") or range_by_map.get(f"satellite{region_id}.tif")
    region_label = (range_row or {}).get("region", "")

    # correspondence
    csv_names = [r.get("filename", "").strip() for r in rows]
    csv_set = {n.upper() for n in csv_names if n}
    jpg_set = set(jpgs)
    missing = sorted(csv_set - jpg_set)
    orphans = sorted(jpg_set - csv_set)
    dup_counts = Counter(n.upper() for n in csv_names if n)
    dups = [k for k, v in dup_counts.items() if v > 1]
    for name in missing:
        issues.append(
            {
                "severity": "definite",
                "reason_code": "missing_image_for_csv_row",
                "region": region_id,
                "item": name,
                "evidence": f"listed in {csv_path.name} but no matching file under drone/",
            }
        )
    for name in orphans:
        issues.append(
            {
                "severity": "definite",
                "reason_code": "orphan_image",
                "region": region_id,
                "item": name,
                "evidence": "JPG present without CSV row",
            }
        )
    for name in dups:
        issues.append(
            {
                "severity": "definite",
                "reason_code": "duplicate_csv_filename",
                "region": region_id,
                "item": name,
                "evidence": f"filename appears {dup_counts[name]} times in CSV",
            }
        )

    # numeric validity
    lats, lons, heights, omegas, kappas, phi1s, phi2s = [], [], [], [], [], [], []
    dates = []
    invalid = 0
    usable_meta = 0
    manifest_rows = []
    for r in rows:
        fn = (r.get("filename") or "").strip()
        img = jpgs.get(fn.upper())
        exists = img is not None
        try:
            lat = float(r["lat"])
            lon = float(r["lon"])
            height = float(r["height"])
            omega = float(r.get("Omega", "nan"))
            kappa = float(r.get("Kappa", "nan"))
            phi1 = float(r.get("Phi1", "nan"))
            phi2 = float(r.get("Phi2", "nan"))
            ok_coords = math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            if not ok_coords:
                invalid += 1
                issues.append(
                    {
                        "severity": "definite",
                        "reason_code": "invalid_coordinates",
                        "region": region_id,
                        "item": fn,
                        "evidence": f"lat={r.get('lat')} lon={r.get('lon')}",
                    }
                )
            else:
                lats.append(lat)
                lons.append(lon)
                if exists:
                    usable_meta += 1
            if math.isfinite(height):
                heights.append(height)
            else:
                issues.append(
                    {
                        "severity": "investigate",
                        "reason_code": "nonfinite_height",
                        "region": region_id,
                        "item": fn,
                        "evidence": r.get("height"),
                    }
                )
            for name, val, bucket in [
                ("Omega", omega, omegas),
                ("Kappa", kappa, kappas),
                ("Phi1", phi1, phi1s),
                ("Phi2", phi2, phi2s),
            ]:
                if math.isfinite(val):
                    bucket.append(val)
            dates.append(r.get("date", ""))
        except Exception as e:
            invalid += 1
            issues.append(
                {
                    "severity": "definite",
                    "reason_code": "unparseable_metadata_row",
                    "region": region_id,
                    "item": fn,
                    "evidence": str(e),
                }
            )
            lat = lon = height = omega = kappa = phi1 = phi2 = ""
        manifest_rows.append(
            {
                "region_folder": region_id,
                "map_id": region_id,
                "flight_id": region_id,  # README: one flight per map folder; relationship recorded explicitly
                "num": r.get("num", ""),
                "filename": fn,
                "image_path": str(img) if img else "",
                "image_exists": exists,
                "date": r.get("date", ""),
                "lat": r.get("lat", ""),
                "lon": r.get("lon", ""),
                "height": r.get("height", ""),
                "Omega": r.get("Omega", ""),
                "Kappa": r.get("Kappa", ""),
                "Phi1": r.get("Phi1", ""),
                "Phi2": r.get("Phi2", ""),
                "region_label_from_range_csv": region_label,
                "metadata_valid_coords": bool(lats and fn and exists) if False else "",  # filled below
            }
        )

    # fix metadata_valid_coords properly
    for i, r in enumerate(rows):
        try:
            lat = float(r["lat"])
            lon = float(r["lon"])
            ok = math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            exists = jpgs.get((r.get("filename") or "").strip().upper()) is not None
            manifest_rows[i]["metadata_valid_coords"] = ok and exists
            # note: valid coords != verified localization GT
        except Exception:
            manifest_rows[i]["metadata_valid_coords"] = False

    # temporal / spacing
    ordered = []
    for r in rows:
        ts = parse_ts(r.get("date", ""))
        try:
            lat = float(r["lat"])
            lon = float(r["lon"])
        except Exception:
            continue
        if ts is None or not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        ordered.append((ts, lat, lon, r.get("filename", "")))
    ordered.sort(key=lambda x: x[0])
    dt_s, dist_m = [], []
    for a, b in zip(ordered, ordered[1:]):
        dt_s.append((b[0] - a[0]).total_seconds())
        dist_m.append(haversine_m(a[1], a[2], b[1], b[2]))

    def summ(vals):
        if not vals:
            return {"min": "", "median": "", "max": "", "n": 0}
        return {
            "min": min(vals),
            "median": statistics.median(vals),
            "max": max(vals),
            "n": len(vals),
        }

    q_dist = quantiles(dist_m)
    # large gaps: > q99 or > 5x median
    med = q_dist.get("q50") or (statistics.median(dist_m) if dist_m else None)
    large_gaps = []
    if dist_m and med:
        thr = max(float(q_dist.get("q99") or 0), 5 * float(med))
        for i, d in enumerate(dist_m):
            if d >= thr and d > 0:
                large_gaps.append((ordered[i][3], ordered[i + 1][3], d))

    sat_paths = sorted(region_dir.glob("satellite*.tif"))
    sat_meta_list = []
    canonical = region_dir / f"satellite{region_id}.tif"
    if not sat_paths:
        issues.append(
            {
                "severity": "definite",
                "reason_code": "missing_satellite_tif",
                "region": region_id,
                "item": f"satellite{region_id}.tif",
                "evidence": "no satellite*.tif under region folder",
            }
        )
    elif not canonical.exists() and len(sat_paths) > 1:
        issues.append(
            {
                "severity": "investigate",
                "reason_code": "multi_tile_satellite_map",
                "region": region_id,
                "item": ",".join(p.name for p in sat_paths),
                "evidence": f"canonical {canonical.name} absent; {len(sat_paths)} tile GeoTIFFs present",
            }
        )
    for sp in sat_paths:
        sm = inspect_satellite(sp, range_row)
        sat_meta_list.append(sm)
        if sm.get("georef_status") not in {"embedded_geotiff"}:
            issues.append(
                {
                    "severity": "investigate",
                    "reason_code": "satellite_georef_incomplete",
                    "region": region_id,
                    "item": sp.name,
                    "evidence": sm.get("georef_status") or sm.get("error"),
                }
            )
        elif sm.get("csv_vs_tif_delta_max") not in ("", None) and isinstance(
            sm.get("csv_vs_tif_delta_max"), (int, float)
        ):
            if float(sm["csv_vs_tif_delta_max"]) > 1e-6:
                issues.append(
                    {
                        "severity": "investigate",
                        "reason_code": "satellite_bounds_csv_mismatch",
                        "region": region_id,
                        "item": sp.name,
                        "evidence": f"max_abs_delta={sm['csv_vs_tif_delta_max']} (tile may be chip not full map)",
                    }
                )
    sat_meta = sat_meta_list[0] if sat_meta_list else inspect_satellite(canonical, range_row)

    # platform/camera unknown from files (sample had no EXIF); record unknown unless docs map region
    platform = "unknown"
    camera = "unknown"

    flight_stats = {
        "flight_id": region_id,
        "region_folder": region_id,
        "map_id": region_id,
        "region_label": region_label,
        "relationship_note": "README/Fig.2: one satellite map encompasses images from a single drone flight; folder id shared across map/flight/CSV",
        "platform": platform,
        "camera": camera,
        "platform_camera_evidence": "not stated per-region in release files; paper: mix of fixed-wing and multi-rotor; sample JPGs had no EXIF",
        "n_csv_rows": len(rows),
        "n_jpg": len(jpgs),
        "n_usable_metadata_coords": usable_meta,
        "n_missing_images": len(missing),
        "n_orphan_images": len(orphans),
        "n_duplicate_filenames": len(dups),
        "n_invalid_coord_rows": invalid,
        "date_min": min((d for d in dates if d), default=""),
        "date_max": max((d for d in dates if d), default=""),
        "date_source": "CSV column date",
        "timezone": "unresolved",
        "height_col": "height",
        "height_ref": "unresolved (AGL/AMSL/ellipsoid not stated)",
        "height_min": summ(heights)["min"],
        "height_median": summ(heights)["median"],
        "height_max": summ(heights)["max"],
        "Omega_min": summ(omegas)["min"],
        "Omega_median": summ(omegas)["median"],
        "Omega_max": summ(omegas)["max"],
        "Kappa_min": summ(kappas)["min"],
        "Kappa_median": summ(kappas)["median"],
        "Kappa_max": summ(kappas)["max"],
        "Phi1_min": summ(phi1s)["min"],
        "Phi1_median": summ(phi1s)["median"],
        "Phi1_max": summ(phi1s)["max"],
        "Phi2_min": summ(phi2s)["min"],
        "Phi2_median": summ(phi2s)["median"],
        "Phi2_max": summ(phi2s)["max"],
        "phi_convention_note": "Paper footnote: Phi is drone heading; 0=due east; north positive; south negative; range [-180,180]. CSV has Phi1/Phi2; Phi1 preferred per README. Units degrees per paper.",
        "lat_min": min(lats) if lats else "",
        "lat_max": max(lats) if lats else "",
        "lon_min": min(lons) if lons else "",
        "lon_max": max(lons) if lons else "",
        "temporal_dt_median_s": summ(dt_s)["median"],
        "temporal_dt_max_s": summ(dt_s)["max"],
        "ground_spacing_median_m": q_dist.get("q50", ""),
        "ground_spacing_q90_m": q_dist.get("q90", ""),
        "ground_spacing_q95_m": q_dist.get("q95", ""),
        "ground_spacing_q99_m": q_dist.get("q99", ""),
        "ground_spacing_max_m": max(dist_m) if dist_m else "",
        "n_large_ground_gaps": len(large_gaps),
        "large_gap_examples": "; ".join(f"{a}->{b}:{d:.1f}m" for a, b, d in large_gaps[:5]),
        "satellite_file": "|".join(p.name for p in sat_paths) if sat_paths else "",
        "csv_headers": "|".join(headers),
    }

    region_stats = {
        "region_folder": region_id,
        "map_id": region_id,
        "flight_id": region_id,
        "region_label": region_label,
        "n_jpg": len(jpgs),
        "n_csv_rows": len(rows),
        "n_usable_metadata_coords": usable_meta,
        "satellite_exists": bool(sat_paths),
        "satellite_georef_status": "|".join(sm.get("georef_status", "") for sm in sat_meta_list),
        "satellite_crs_epsg": sat_meta.get("crs_epsg", "") if sat_meta_list else "",
        "satellite_bounds_delta_max": "|".join(str(sm.get("csv_vs_tif_delta_max", "")) for sm in sat_meta_list),
        "lat_min": flight_stats["lat_min"],
        "lat_max": flight_stats["lat_max"],
        "lon_min": flight_stats["lon_min"],
        "lon_max": flight_stats["lon_max"],
        "height_min": flight_stats["height_min"],
        "height_median": flight_stats["height_median"],
        "height_max": flight_stats["height_max"],
        "date_min": flight_stats["date_min"],
        "date_max": flight_stats["date_max"],
    }

    rel = {
        "region_folder": region_id,
        "map_file": "|".join(p.name for p in sat_paths) if sat_paths else "",
        "flight_id": region_id,
        "pose_csv": csv_path.name if csv_path.exists() else "",
        "range_csv_mapname": (range_row or {}).get("mapname", ""),
        "region_label": region_label,
        "relationship": "folder_id_shared: README states each satellite map covers UAV images from a single drone flight",
        "evidence": "GitHub README Fig.2 caption; sample README_example.txt; paper §II",
        "n_images": len(jpgs),
        "n_csv_rows": len(rows),
        "join_status": "ok_1to1" if not missing and not orphans and not dups and len(rows) == len(jpgs) else "mismatch",
    }

    return {
        "manifest_rows": manifest_rows,
        "flight_stats": flight_stats,
        "region_stats": region_stats,
        "relationship": rel,
        "sat_meta": sat_meta,
        "sat_meta_list": sat_meta_list,
        "n_jpg": len(jpgs),
        "file_entries": [],
    }


def walk_inventory(root: Path) -> list[dict]:
    rows = []
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            rows.append({"relative_path": rel, "extension": p.suffix.lower(), "size_bytes": p.stat().st_size})
    return rows


def compare_sample_region03(full_root: Path, issues: list) -> list[dict]:
    """Hash-compare sample vs full region 03 for CSV and a few images; size-compare satellite."""
    out = []
    s_dir = SAMPLE / "03"
    f_dir = full_root / "03"
    pairs = [
        ("03.csv", s_dir / "03.csv", f_dir / "03.csv"),
        ("03_0001.JPG", s_dir / "drone" / "03_0001.JPG", f_dir / "drone" / "03_0001.JPG"),
        ("03_0385.JPG", s_dir / "drone" / "03_0385.JPG", f_dir / "drone" / "03_0385.JPG"),
        ("03_0768.JPG", s_dir / "drone" / "03_0768.JPG", f_dir / "drone" / "03_0768.JPG"),
    ]
    for label, a, b in pairs:
        row = {"item": label, "sample_exists": a.exists(), "full_exists": b.exists()}
        if a.exists() and b.exists():
            sa, sb = a.stat().st_size, b.stat().st_size
            row["sample_size"] = sa
            row["full_size"] = sb
            if sa == sb and sa < 50_000_000:
                ha, hb = sha256_file(a), sha256_file(b)
                row["sample_sha256"] = ha
                row["full_sha256"] = hb
                row["identical"] = ha == hb
            else:
                row["identical"] = sa == sb
                row["note"] = "size_compare_only" if sa >= 50_000_000 else ""
        else:
            row["identical"] = False
        out.append(row)
        if a.exists() and b.exists() and row.get("identical") is False:
            issues.append(
                {
                    "severity": "investigate",
                    "reason_code": "sample_vs_full_region03_differs",
                    "region": "03",
                    "item": label,
                    "evidence": str(row),
                }
            )
    # satellite: size + optional partial hash of first 16MB
    sa, sb = s_dir / "satellite03.tif", f_dir / "satellite03.tif"
    row = {"item": "satellite03.tif", "sample_exists": sa.exists(), "full_exists": sb.exists()}
    if sa.exists() and sb.exists():
        row["sample_size"] = sa.stat().st_size
        row["full_size"] = sb.stat().st_size
        row["identical"] = row["sample_size"] == row["full_size"]
        # quick prefix hash
        def prefix_hash(p: Path, n=16 * 1024 * 1024):
            h = hashlib.sha256()
            with p.open("rb") as f:
                h.update(f.read(n))
            return h.hexdigest().upper()

        row["sample_prefix16mb_sha256"] = prefix_hash(sa)
        row["full_prefix16mb_sha256"] = prefix_hash(sb)
        row["prefix_identical"] = row["sample_prefix16mb_sha256"] == row["full_prefix16mb_sha256"]
        if not row["identical"] or not row["prefix_identical"]:
            issues.append(
                {
                    "severity": "investigate",
                    "reason_code": "sample_vs_full_satellite03_differs",
                    "region": "03",
                    "item": "satellite03.tif",
                    "evidence": str(row),
                }
            )
    out.append(row)
    return out


def flatten_issues(issues: list[dict]) -> list[dict]:
    out = []
    for i in issues:
        out.append(
            {
                "severity": i.get("severity", ""),
                "category": i.get("category", i.get("reason_code", "")),
                "reason_code": i.get("reason_code", ""),
                "region": i.get("region", ""),
                "item": i.get("item", ""),
                "detail": i.get("detail", i.get("evidence", "")),
                "evidence": i.get("evidence", ""),
            }
        )
    return out


def verify_github_release_layout(root: Path, issues: list) -> dict:
    """Distinguish IntelliSensing GitHub layout from Zenodo tile-release."""
    markers = list(root.rglob("UAV-image")) + list(root.rglob("satellite-tile"))
    zips = [p for p in root.rglob("*.zip") if p.parent == root]
    ok = (
        (root / "README_dataset.txt").exists()
        and not markers
        and len([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]) == 11
    )
    if markers:
        issues.append(
            {
                "severity": "definite",
                "reason_code": "zenodo_layout_detected",
                "region": "ALL",
                "item": "folder layout",
                "evidence": "UAV-image or satellite-tile paths found",
            }
        )
    return {
        "release_class": "github_intellisensing" if ok else "unknown_or_mixed",
        "readme_present": (root / "README_dataset.txt").exists(),
        "zenodo_markers": len(markers),
        "region_folders": len([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    import shutil

    root = find_extract_root()
    if root is None:
        print("EXTRACT_NOT_READY", FULL, FULL_EXTRACT)
        return 3
    print("ROOT", root)

    archive_info = {"path": str(ARCHIVE), "present": ARCHIVE.exists()}
    if ARCHIVE.exists():
        archive_info["size_bytes"] = ARCHIVE.stat().st_size
        with zipfile.ZipFile(ARCHIVE, "r") as zf:
            archive_info["zip_testzip"] = zf.testzip()
            archive_info["zip_entries"] = len(zf.namelist())
        print("Computing archive SHA256...")
        archive_info["sha256_local"] = sha256_file(ARCHIVE)
        (RAW / "UAV_VisLoc_dataset.zip.sha256.txt").write_text(
            archive_info["sha256_local"] + "\n", encoding="utf-8"
        )
    else:
        archive_info["note"] = "local integrity SHA256 skipped — zip not present; using manual extract only"
    range_path = list_range_csv(root)
    range_by_map = {}
    range_rows = []
    if range_path:
        with range_path.open(encoding="utf-8", newline="") as f:
            range_rows = list(csv.DictReader(f))
            for r in range_rows:
                range_by_map[r.get("mapname", "").strip()] = r
    else:
        print("NO_RANGE_CSV")

    issues = []
    layout = verify_github_release_layout(root, issues)
    manifest_all = []
    flight_stats = []
    region_stats = []
    relationships = []
    sat_metas = []
    total_jpg = 0

    region_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    for rd in region_dirs:
        rid = rd.name
        print("REGION", rid, flush=True)
        res = inventory_region(rid, rd, range_by_map, issues)
        manifest_all.extend(res["manifest_rows"])
        flight_stats.append(res["flight_stats"])
        region_stats.append(res["region_stats"])
        relationships.append(res["relationship"])
        sat_metas.extend(res.get("sat_meta_list") or [res["sat_meta"]])
        total_jpg += res["n_jpg"]

    file_inv = walk_inventory(root)
    sample_cmp = compare_sample_region03(root, issues)

    # capture date comparison: flight CSV dates vs satellite (unknown acquisition)
    capture_rows = []
    for rs in region_stats:
        capture_rows.append(
            {
                "region_folder": rs["region_folder"],
                "flight_date_min": rs["date_min"],
                "flight_date_max": rs["date_max"],
                "flight_date_source": "pose CSV date column (capture metadata per README)",
                "flight_timezone": "unresolved",
                "satellite_acquisition_date": "not documented in README/sample/release CSV",
                "satellite_date_source": "none found in release files; paper does not provide per-map acquisition dates",
                "file_mtime_note": "filesystem mtime is download/extract time — not used as acquisition date",
                "comparison": "cannot compare flight vs satellite acquisition dates — satellite acquisition date absent",
            }
        )

    # counts check
    n_maps = len(region_dirs)
    if total_jpg != DOCUMENTED_UAV:
        issues.append(
            {
                "severity": "investigate" if abs(total_jpg - DOCUMENTED_UAV) < 50 else "definite",
                "reason_code": "uav_count_mismatch_vs_docs",
                "region": "ALL",
                "item": "jpg_count",
                "evidence": f"actual={total_jpg} documented={DOCUMENTED_UAV}",
            }
        )
    if n_maps != DOCUMENTED_MAPS:
        issues.append(
            {
                "severity": "definite",
                "reason_code": "map_count_mismatch_vs_docs",
                "region": "ALL",
                "item": "region_folders",
                "evidence": f"actual={n_maps} documented={DOCUMENTED_MAPS}",
            }
        )

    write_csv(MANIFESTS / "manifest.csv", manifest_all)
    write_csv(MANIFESTS / "full_file_inventory.csv", file_inv)
    write_csv(MANIFESTS / "flight_map_relationships.csv", relationships)
    write_csv(MANIFESTS / "satellite_metadata.csv", sat_metas)
    write_csv(REPORTS / "flight_stats.csv", flight_stats)
    write_csv(REPORTS / "region_stats.csv", region_stats)
    write_csv(REPORTS / "full_frame_issues.csv", flatten_issues(issues))
    write_csv(REPORTS / "capture_date_comparison.csv", capture_rows)
    write_csv(REPORTS / "sample_vs_full_region03.csv", sample_cmp)

    free = shutil.disk_usage("F:\\").free
    summary = {
        "generated_at": utc_now(),
        "extract_root": str(root),
        "release_layout": layout,
        "archive": archive_info,
        "n_jpg": total_jpg,
        "n_maps": n_maps,
        "n_satellite_rasters": len(sat_metas),
        "documented_uav": DOCUMENTED_UAV,
        "documented_maps": DOCUMENTED_MAPS,
        "n_issues": len(issues),
        "range_csv": str(range_path) if range_path else "",
        "n_range_rows": len(range_rows),
        "F_free_bytes": free,
        "manifest_rows": len(manifest_all),
    }
    (REPORTS / "full_audit_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    print("SUMMARY", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
