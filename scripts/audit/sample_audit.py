#!/usr/bin/env python3
"""Stage 2A sample audit for UAV-VisLoc GitHub sample (region 03).

Windowed TIFF reads only; never load full satellite into RAM.
Writes manifests and issue CSV under datasets/ and reports/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"F:\CMBL")
SAMPLE = ROOT / "datasets" / "sample"
RAW = ROOT / "datasets" / "raw"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figures" / "sample_audit"
MANIFEST_DIR = ROOT / "datasets"

for p in (REPORTS, FIGS, MANIFEST_DIR / "manifests"):
    p.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        # sniff delimiter
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


def missingness(rows: list[dict], col: str) -> dict:
    vals = [r.get(col, "") for r in rows]
    empty = sum(1 for v in vals if v is None or str(v).strip() == "")
    return {
        "n": len(vals),
        "missing": empty,
        "missing_frac": (empty / len(vals)) if vals else None,
        "example_nonempty": next((str(v) for v in vals if str(v).strip()), ""),
    }


def inventory_files(base: Path) -> list[dict]:
    rows = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rel = p.relative_to(base).as_posix()
            rows.append(
                {
                    "relative_path": rel,
                    "abs_path": str(p),
                    "extension": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
                    "parent": p.parent.relative_to(base).as_posix() if p.parent != base else ".",
                }
            )
    return rows


def inspect_exif(path: Path) -> dict:
    out = {"file": path.name, "ok": False}
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
    except Exception as e:
        out["error"] = f"Pillow unavailable: {e}"
        return out

    try:
        with Image.open(path) as im:
            out["ok"] = True
            out["pil_format"] = im.format
            out["pil_mode"] = im.mode
            out["width"], out["height"] = im.size
            exif = im.getexif()
            if not exif:
                out["exif_present"] = False
                return out
            out["exif_present"] = True
            tagged = {}
            for k, v in exif.items():
                name = TAGS.get(k, str(k))
                if name == "GPSInfo" and isinstance(v, dict):
                    gps = {}
                    for gk, gv in v.items():
                        gps[GPSTAGS.get(gk, str(gk))] = str(gv)
                    tagged["GPSInfo"] = gps
                else:
                    try:
                        tagged[name] = v if isinstance(v, (int, float, str)) else str(v)
                    except Exception:
                        tagged[name] = repr(v)
            # Common fields
            for key in (
                "Make",
                "Model",
                "DateTime",
                "DateTimeOriginal",
                "FocalLength",
                "FocalLengthIn35mmFilm",
                "Orientation",
                "ExifImageWidth",
                "ExifImageHeight",
                "LensModel",
            ):
                if key in tagged:
                    out[key] = tagged[key]
            if "GPSInfo" in tagged:
                out["GPSInfo"] = tagged["GPSInfo"]
            # keep compact dump of remaining keys of interest
            out["exif_keys"] = sorted(tagged.keys())
    except Exception as e:
        out["error"] = str(e)
    return out


def inspect_geotiff_windowed(path: Path) -> dict:
    """Inspect TIFF metadata without loading full raster into RAM."""
    out = {"file": path.name, "ok": False, "size_bytes": path.stat().st_size}
    try:
        import tifffile
        import numpy as np
    except Exception as e:
        out["error"] = f"tifffile unavailable: {e}"
        return out

    try:
        with tifffile.TiffFile(path) as tif:
            out["ok"] = True
            out["n_pages"] = len(tif.pages)
            page = tif.pages[0]
            out["shape"] = list(page.shape) if page.shape else None
            out["dtype"] = str(page.dtype)
            out["axes"] = getattr(page, "axes", None)
            out["photometric"] = str(getattr(page, "photometric", None))
            out["compression"] = str(getattr(page, "compression", None))
            tags = {tag.name: tag.value for tag in page.tags.values()}
            # Geo keys of interest
            for name in (
                "ImageWidth",
                "ImageLength",
                "BitsPerSample",
                "SamplesPerPixel",
                "SampleFormat",
                "PlanarConfiguration",
                "TileWidth",
                "TileLength",
                "ModelPixelScaleTag",
                "ModelTiepointTag",
                "GeoKeyDirectoryTag",
                "GeoAsciiParamsTag",
                "GeoDoubleParamsTag",
                "GDAL_NODATA",
            ):
                if name in tags:
                    val = tags[name]
                    out[name] = val if isinstance(val, (int, float, str, tuple, list)) else str(val)

            # Parse geokeys if present
            if "GeoKeyDirectoryTag" in tags:
                gdk = tags["GeoKeyDirectoryTag"]
                # structure: [version, rev, minor, numkeys, key,loc,count,value/offset ...]
                try:
                    gdk = list(gdk)
                    nkeys = int(gdk[3])
                    geokeys = {}
                    for i in range(nkeys):
                        key = int(gdk[4 + 4 * i])
                        loc = int(gdk[5 + 4 * i])
                        count = int(gdk[6 + 4 * i])
                        val = gdk[7 + 4 * i]
                        geokeys[key] = {"location": loc, "count": count, "value_or_offset": val}
                    out["geokey_count"] = nkeys
                    # Common keys: 1024 ModelType, 2048 GeographicType, 3072 ProjectedCSType
                    interesting = {
                        1024: "GTModelTypeGeoKey",
                        1025: "GTRasterTypeGeoKey",
                        2048: "GeographicTypeGeoKey",
                        3072: "ProjectedCSTypeGeoKey",
                    }
                    for code, label in interesting.items():
                        if code in geokeys:
                            out[label] = geokeys[code]

                    if "GeoAsciiParamsTag" in tags:
                        out["GeoAsciiParamsTag"] = str(tags["GeoAsciiParamsTag"])[:500]
                except Exception as e:
                    out["geokey_parse_error"] = str(e)

            # Bounds from ModelTiepoint + ModelPixelScale if available
            if "ModelTiepointTag" in tags and "ModelPixelScaleTag" in tags:
                tp = list(tags["ModelTiepointTag"])
                sc = list(tags["ModelPixelScaleTag"])
                # tiepoint: I,J,K,X,Y,Z ; scale: ScaleX, ScaleY, ScaleZ
                if len(tp) >= 6 and len(sc) >= 2:
                    i, j, k, x, y, z = tp[:6]
                    sx, sy = float(sc[0]), float(sc[1])
                    w = int(tags.get("ImageWidth", page.shape[-1] if page.shape else 0))
                    h = int(tags.get("ImageLength", page.shape[-2] if page.shape else 0))
                    # Assuming north-up: origin at (x,y) for pixel (i,j); typically i=j=0
                    # Pixel-edge vs centre still ambiguous without GTRasterType
                    out["model_tiepoint"] = {"i": i, "j": j, "x": x, "y": y, "z": z}
                    out["model_pixel_scale"] = {"sx": sx, "sy": sy, "sz": float(sc[2]) if len(sc) > 2 else None}
                    # Approximate bounds treating (x,y) as upper-left of pixel (i,j)
                    # and sy positive meaning north-up with decreasing Y southward often stored as positive ScaleY
                    # GeoTIFF convention: ModelPixelScale Y is often positive while map Y decreases south
                    west = float(x) - float(i) * sx
                    north = float(y) + float(j) * sy  # if y decreases with j, many files use positive sy and north - j*sy
                    # Prefer common north-up: Y_map = y - j*sy when sy>0
                    north_alt = float(y) - float(j) * sy
                    east = west + w * sx
                    south = north_alt - h * sy
                    out["approx_bounds_assuming_ul_origin_y_decreases_south"] = {
                        "west": west,
                        "east": east,
                        "north": north_alt,
                        "south": south,
                        "note": "Approximate; pixel-edge vs centre and Y sign need GTRasterType confirmation",
                    }

            # Windowed readability check only — never page.asarray() on full image
            try:
                out["is_tiled"] = bool(getattr(page, "is_tiled", False))
                if out["is_tiled"]:
                    seg0 = next(page.segments())
                    tile = seg0[1]
                    if hasattr(tile, "shape"):
                        out["first_tile_shape"] = list(tile.shape)
                        out["first_tile_dtype"] = str(tile.dtype)
                        out["first_tile_minmax"] = [float(np.min(tile)), float(np.max(tile))]
                else:
                    mm = tifffile.memmap(path)
                    out["memmap_shape"] = list(mm.shape)
                    out["memmap_dtype"] = str(mm.dtype)
                    if mm.ndim == 3 and mm.shape[-1] in (1, 3, 4):
                        win = mm[:64, :64, :]
                    elif mm.ndim == 3 and mm.shape[0] in (1, 3, 4):
                        win = mm[:, :64, :64]
                    elif mm.ndim == 2:
                        win = mm[:64, :64]
                    else:
                        win = mm.reshape(-1)[:4096]
                    out["window_64_minmax"] = [float(np.min(win)), float(np.max(win))]
                    del mm
            except Exception as e:
                out["window_read_error"] = str(e)

            # Overviews
            out["is_bigtiff"] = bool(getattr(tif, "is_bigtiff", False))
            if hasattr(tif, "series") and tif.series:
                out["series0_shape"] = list(tif.series[0].shape)
                out["series0_dtype"] = str(tif.series[0].dtype)
                out["series0_axes"] = tif.series[0].axes
    except Exception as e:
        out["error"] = str(e)
    return out


def make_previews(drone_paths: list[Path], sat_path: Path | None) -> list[str]:
    saved = []
    try:
        from PIL import Image
        import numpy as np
    except Exception as e:
        return [f"preview_error:{e}"]

    FIGS.mkdir(parents=True, exist_ok=True)
    for p in drone_paths[:6]:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((512, 512))
                dest = FIGS / f"drone_preview_{p.stem}.jpg"
                im.save(dest, quality=85)
                saved.append(str(dest))
        except Exception as e:
            saved.append(f"fail:{p.name}:{e}")

    if sat_path and sat_path.exists():
        try:
            import tifffile

            mm = tifffile.memmap(sat_path)
            # take a small window and downsample
            if mm.ndim == 3 and mm.shape[-1] in (3, 4):
                win = np.array(mm[0:2048:8, 0:2048:8, :3])
            elif mm.ndim == 3 and mm.shape[0] in (3, 4):
                win = np.array(mm[:3, 0:2048:8, 0:2048:8]).transpose(1, 2, 0)
            elif mm.ndim == 2:
                win = np.array(mm[0:2048:8, 0:2048:8])
                win = np.stack([win, win, win], axis=-1)
            else:
                win = np.array(mm.reshape(mm.shape[-2], mm.shape[-1], -1)[0:2048:8, 0:2048:8, :3])
            # normalize to 8-bit
            win = win.astype(np.float32)
            for c in range(min(3, win.shape[-1])):
                ch = win[..., c]
                lo, hi = np.percentile(ch, [2, 98])
                if hi <= lo:
                    hi = lo + 1
                win[..., c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
            img = Image.fromarray(win.astype(np.uint8)[..., :3])
            dest = FIGS / "satellite03_ul_preview.jpg"
            img.save(dest, quality=85)
            saved.append(str(dest))
            del mm
        except Exception as e:
            saved.append(f"sat_preview_fail:{e}")
    return saved


def main() -> int:
    issues: list[dict] = []
    summary: dict = {"generated_at": utc_now(), "sample_root": str(SAMPLE)}

    # Locate files (handle space in satellite_ coordinates_range.csv)
    csv_candidates = list(SAMPLE.rglob("*.csv")) + list(SAMPLE.glob("*.csv"))
    # Also odd names
    csv_candidates += [p for p in SAMPLE.iterdir() if p.is_file() and "csv" in p.name.lower()]
    csv_candidates = sorted(set(csv_candidates))

    tif_candidates = list(SAMPLE.rglob("*.tif")) + list(SAMPLE.rglob("*.tiff"))
    jpg_candidates = sorted(SAMPLE.rglob("*.JPG")) + sorted(SAMPLE.rglob("*.jpg"))

    summary["n_jpg"] = len(jpg_candidates)
    summary["n_tif"] = len(tif_candidates)
    summary["csv_files"] = [str(p.relative_to(SAMPLE)) for p in csv_candidates]

    # File inventory
    inv = inventory_files(SAMPLE)
    inv_path = MANIFEST_DIR / "manifests" / "sample_file_inventory.csv"
    with inv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "abs_path", "extension", "size_bytes", "parent"])
        w.writeheader()
        w.writerows(inv)

    # Parse flight CSV (03.csv)
    flight_csv = next((p for p in csv_candidates if p.name.lower() in ("03.csv",) or p.name.endswith("03.csv")), None)
    range_csv = next(
        (p for p in csv_candidates if "satellite" in p.name.lower() and "range" in p.name.lower()),
        None,
    )

    data_dict_rows: list[dict] = []
    manifest_rows: list[dict] = []

    if flight_csv is None:
        issues.append(
            {
                "severity": "definite",
                "category": "missing_file",
                "item": "03.csv",
                "detail": "Flight CSV not found under sample",
            }
        )
    else:
        headers, rows = read_csv_rows(flight_csv)
        summary["flight_csv"] = str(flight_csv.relative_to(SAMPLE))
        summary["flight_csv_headers"] = headers
        summary["flight_csv_nrows"] = len(rows)
        # write header examples
        for col in headers:
            miss = missingness(rows, col)
            examples = []
            for r in rows[:5]:
                examples.append(str(r.get(col, "")))
            # infer type
            non_empty = [str(r.get(col, "")).strip() for r in rows if str(r.get(col, "")).strip()]
            dtype = "string"
            if non_empty:
                try:
                    [float(x) for x in non_empty[:20]]
                    dtype = "float_or_numeric_string"
                except ValueError:
                    dtype = "string"
            data_dict_rows.append(
                {
                    "file": flight_csv.name,
                    "column": col,
                    "data_type": dtype,
                    "example_values": " | ".join(examples),
                    "missing_count": miss["missing"],
                    "missing_frac": miss["missing_frac"],
                    "documented_meaning_units": "See README: filename latitude longitude height ··· (units not fully specified in README)",
                    "evidence_source": "GitHub README §2.1 schema comment; observed headers in sample",
                    "inferred_meaning": "",
                    "inference_confidence": "",
                }
            )

        # Image correspondence
        csv_names = [str(r.get(headers[0], "")).strip() for r in rows]
        # normalize
        jpg_names = {p.name for p in jpg_candidates}
        jpg_stems = {p.stem for p in jpg_candidates}
        missing_images = []
        for name in csv_names:
            base = Path(name).name
            if base not in jpg_names and Path(base).stem not in jpg_stems and name not in jpg_names:
                # try with .JPG
                if not (base.endswith(".JPG") or base.endswith(".jpg")):
                    if f"{base}.JPG" not in jpg_names and f"{base}.jpg" not in jpg_names:
                        missing_images.append(name)
                else:
                    missing_images.append(name)
        orphan_images = []
        csv_basenames = set()
        for name in csv_names:
            b = Path(name).name
            csv_basenames.add(b)
            csv_basenames.add(Path(b).stem)
        for p in jpg_candidates:
            if p.name not in csv_basenames and p.stem not in csv_basenames:
                orphan_images.append(p.name)

        if missing_images:
            issues.append(
                {
                    "severity": "definite",
                    "category": "missing_image",
                    "item": "csv_without_image",
                    "detail": f"count={len(missing_images)}; examples={missing_images[:5]}",
                }
            )
        if orphan_images:
            issues.append(
                {
                    "severity": "definite",
                    "category": "orphan_image",
                    "item": "image_without_csv_row",
                    "detail": f"count={len(orphan_images)}; examples={orphan_images[:5]}",
                }
            )

        # Duplicate filenames in CSV
        c = Counter(csv_names)
        dups = [k for k, v in c.items() if v > 1]
        if dups:
            issues.append(
                {
                    "severity": "definite",
                    "category": "duplicate_csv_row",
                    "item": "filename",
                    "detail": f"count={len(dups)}; examples={dups[:5]}",
                }
            )

        # Build per-image manifest
        # Map header names flexibly
        def col_like(options):
            for h in headers:
                hl = h.strip().lower()
                for o in options:
                    if o in hl:
                        return h
            return None

        fcol = headers[0]
        lat_c = col_like(["lat"])
        lon_c = col_like(["lon", "lng"])
        h_c = col_like(["height", "alt", "altitude"])
        date_c = col_like(["date", "time"])
        phi_c = col_like(["phi", "heading", "yaw", "angle"])

        summary["mapped_columns"] = {
            "filename": fcol,
            "lat": lat_c,
            "lon": lon_c,
            "height": h_c,
            "date": date_c,
            "heading": phi_c,
        }

        # Update inferred meanings for known columns
        infer_map = {
            lat_c: ("Centre latitude of drone image (WGS84 assumed; CRS not stated in CSV)", "medium"),
            lon_c: ("Centre longitude of drone image (WGS84 assumed; CRS not stated in CSV)", "medium"),
            h_c: ("Shooting height — AGL vs AMSL vs ellipsoid NOT documented in README", "low"),
            date_c: ("Shooting date string — timezone not documented", "medium"),
            phi_c: ("Drone heading angle Phi — units/zero/convention not documented in README", "low"),
            fcol: ("Image filename corresponding to drone/*.JPG", "high"),
        }
        for row in data_dict_rows:
            col = row["column"]
            if col in infer_map:
                row["inferred_meaning"], row["inference_confidence"] = infer_map[col]

        for r in rows:
            name = str(r.get(fcol, "")).strip()
            base = Path(name).name
            if not base.lower().endswith((".jpg", ".jpeg")):
                # try as-is
                img_path = SAMPLE / "03" / "drone" / base
                if not img_path.exists():
                    img_path = SAMPLE / "03" / "drone" / f"{base}.JPG"
            else:
                img_path = SAMPLE / "03" / "drone" / base
            manifest_rows.append(
                {
                    "region": "03",
                    "filename": base,
                    "image_path": str(img_path) if img_path.exists() else "",
                    "image_exists": img_path.exists(),
                    "lat": r.get(lat_c, "") if lat_c else "",
                    "lon": r.get(lon_c, "") if lon_c else "",
                    "height": r.get(h_c, "") if h_c else "",
                    "date": r.get(date_c, "") if date_c else "",
                    "heading_phi": r.get(phi_c, "") if phi_c else "",
                    "extra_json": json.dumps({h: r.get(h, "") for h in headers}, ensure_ascii=False),
                }
            )

    if range_csv is None:
        issues.append(
            {
                "severity": "definite",
                "category": "missing_file",
                "item": "satellite_coordinates_range.csv",
                "detail": "Root range CSV not found (note: zip may use 'satellite_ coordinates_range.csv' with a space)",
            }
        )
    else:
        headers, rows = read_csv_rows(range_csv)
        summary["range_csv"] = str(range_csv.relative_to(SAMPLE))
        summary["range_csv_headers"] = headers
        summary["range_csv_nrows"] = len(rows)
        summary["range_csv_rows"] = rows
        for col in headers:
            miss = missingness(rows, col)
            examples = [str(r.get(col, "")) for r in rows[:5]]
            data_dict_rows.append(
                {
                    "file": range_csv.name,
                    "column": col,
                    "data_type": "string_or_numeric",
                    "example_values": " | ".join(examples),
                    "missing_count": miss["missing"],
                    "missing_frac": miss["missing_frac"],
                    "documented_meaning_units": "README: format as filename latitude longitude (range of satellite maps)",
                    "evidence_source": "GitHub README §2.1; observed sample file",
                    "inferred_meaning": "Likely corner or bbox coordinates for each satellite map — corner order NOT documented",
                    "inference_confidence": "low",
                }
            )

    # EXIF on representative frames (first, mid, last + a few)
    exif_reports = []
    if jpg_candidates:
        idxs = sorted(set([0, len(jpg_candidates) // 4, len(jpg_candidates) // 2, (3 * len(jpg_candidates)) // 4, len(jpg_candidates) - 1, 1, 2]))
        picks = [jpg_candidates[i] for i in idxs if 0 <= i < len(jpg_candidates)]
        for p in picks:
            ex = inspect_exif(p)
            exif_reports.append(ex)
            if not ex.get("ok"):
                issues.append(
                    {
                        "severity": "definite",
                        "category": "unreadable_image",
                        "item": p.name,
                        "detail": ex.get("error", "unknown"),
                    }
                )
            elif not ex.get("exif_present"):
                issues.append(
                    {
                        "severity": "unresolved",
                        "category": "missing_exif",
                        "item": p.name,
                        "detail": "No EXIF block readable by Pillow",
                    }
                )
        # Check for camera model / focal length availability
        models = {e.get("Model") for e in exif_reports if e.get("Model")}
        focals = {e.get("FocalLength") for e in exif_reports if e.get("FocalLength") is not None}
        summary["exif_models"] = list(models)
        summary["exif_focals"] = [str(x) for x in focals]
        summary["exif_reports"] = exif_reports

    # Satellite TIFF
    sat = tif_candidates[0] if tif_candidates else None
    sat_meta = None
    if sat is None:
        issues.append(
            {
                "severity": "definite",
                "category": "missing_file",
                "item": "satellite03.tif",
                "detail": "No TIFF found in sample",
            }
        )
    else:
        sat_meta = inspect_geotiff_windowed(sat)
        summary["satellite_meta"] = sat_meta
        if not sat_meta.get("ok"):
            issues.append(
                {
                    "severity": "definite",
                    "category": "unreadable_tiff",
                    "item": sat.name,
                    "detail": sat_meta.get("error", ""),
                }
            )
        else:
            if "ModelTiepointTag" not in sat_meta and "model_tiepoint" not in sat_meta:
                # keys may be under different names
                if "ModelTiepointTag" not in str(sat_meta):
                    issues.append(
                        {
                            "severity": "unresolved",
                            "category": "georef",
                            "item": sat.name,
                            "detail": "ModelTiepoint/Scale may be missing or named differently — see satellite_meta dump",
                        }
                    )

    # Previews
    preview_paths = make_previews(jpg_candidates, sat)
    summary["previews"] = preview_paths

    # Write outputs
    # data dictionary
    dd_path = MANIFEST_DIR / "manifests" / "data_dictionary.csv"
    with dd_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "file",
            "column",
            "data_type",
            "example_values",
            "missing_count",
            "missing_frac",
            "documented_meaning_units",
            "evidence_source",
            "inferred_meaning",
            "inference_confidence",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data_dict_rows)

    # manifest
    man_path = MANIFEST_DIR / "manifests" / "manifest.csv"
    with man_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "region",
            "filename",
            "image_path",
            "image_exists",
            "lat",
            "lon",
            "height",
            "date",
            "heading_phi",
            "extra_json",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest_rows)

    # issues
    iss_path = REPORTS / "sample_frame_issues.csv"
    with iss_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["severity", "category", "item", "detail"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(issues)

    # JSON summary for report writing
    # Strip huge arrays
    summary_path = REPORTS / "sample_audit_raw_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Copy manifests also to datasets/manifests
    for src, name in (
        (dd_path, "data_dictionary.csv"),
        (man_path, "manifest.csv"),
        (inv_path, "sample_file_inventory.csv"),
    ):
        (MANIFEST_DIR / "manifests" / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps({"ok": True, "n_issues": len(issues), "n_manifest": len(manifest_rows), "summary_path": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
