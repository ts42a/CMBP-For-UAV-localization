#!/usr/bin/env python3
"""Focused re-audit: fix column mapping, EXIF/XMP probes, satellite tile preview."""
from __future__ import annotations

import csv
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"F:\CMBL")
SAMPLE = ROOT / "datasets" / "sample"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figures" / "sample_audit"
MANIFESTS = ROOT / "datasets" / "manifests"
for p in (REPORTS, FIGS, MANIFESTS):
    p.mkdir(parents=True, exist_ok=True)

FLIGHT_CSV = SAMPLE / "03" / "03.csv"
# exact name with possible space
RANGE_CSV = next(SAMPLE.glob("satellite*coordinates*range*.csv"))
SAT = SAMPLE / "03" / "satellite03.tif"
DRONE = SAMPLE / "03" / "drone"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def unique_jpgs():
    # Windows case-insensitive: use set of resolved paths
    files = {}
    for p in DRONE.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg"} and p.is_file():
            files[p.name.upper()] = p
    return [files[k] for k in sorted(files)]


def probe_jpeg_metadata(path: Path) -> dict:
    out = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "exif_app1": False,
        "xmp": False,
        "com": False,
        "pil_size": None,
        "pil_exif_keys": [],
        "interesting": {},
    }
    raw = path.read_bytes()[: 256 * 1024]  # first 256KB enough for markers usually
    # JPEG markers
    if raw.startswith(b"\xff\xd8"):
        i = 2
        while i < len(raw) - 4:
            if raw[i] != 0xFF:
                break
            marker = raw[i + 1]
            if marker in (0xD8, 0xD9):
                i += 2
                continue
            if marker == 0xDA:  # SOS
                break
            seglen = struct.unpack(">H", raw[i + 2 : i + 4])[0]
            seg = raw[i + 4 : i + 2 + seglen]
            if marker == 0xE1:
                if seg.startswith(b"Exif\x00\x00") or seg.startswith(b"Exif\x00\x00"):
                    out["exif_app1"] = True
                if b"http://ns.adobe.com/xap/1.0/" in seg or seg.startswith(b"http://ns.adobe.com/xap/1.0/"):
                    out["xmp"] = True
                if b"<x:xmpmeta" in seg or b"drone-dji" in seg.lower() or b"Camera:" in seg:
                    out["xmp"] = True
                    # extract a few fields
                    text = seg.decode("utf-8", errors="ignore")
                    for key in (
                        "drone-dji:GpsLatitude",
                        "drone-dji:GpsLongitude",
                        "drone-dji:AbsoluteAltitude",
                        "drone-dji:RelativeAltitude",
                        "drone-dji:GimbalYawDegree",
                        "drone-dji:FlightYawDegree",
                        "drone-dji:FocalLength",
                        "tiff:Model",
                        "tiff:Make",
                    ):
                        m = re.search(re.escape(key) + r'[="\s]+([^<"\s]+)', text)
                        if m:
                            out["interesting"][key] = m.group(1)
            if marker == 0xFE:
                out["com"] = True
            i += 2 + seglen
            if i > len(raw) - 4:
                break

    with Image.open(path) as im:
        out["pil_size"] = list(im.size)
        out["pil_mode"] = im.mode
        exif = im.getexif()
        if exif:
            out["pil_exif_keys"] = [str(k) for k in exif.keys()]
            # try IFD
            try:
                from PIL.ExifTags import Base

                for tag in ("Make", "Model", "DateTime", "FocalLength", "FocalLengthIn35mmFilm", "Orientation"):
                    try:
                        val = exif.get_ifd(0x8769).get(getattr(Base, tag, None)) if False else None
                    except Exception:
                        val = None
            except Exception:
                pass
            tagged = {k: exif.get(k) for k in list(exif.keys())[:30]}
            out["pil_exif_sample"] = {str(k): str(v)[:80] for k, v in tagged.items()}
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    jpgs = unique_jpgs()
    issues = []

    with FLIGHT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames)
        rows = list(reader)

    with RANGE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        range_rows = list(csv.DictReader(f))
        range_headers = list(range_rows[0].keys()) if range_rows else []

    # correspondence using filename column
    csv_files = [r["filename"].strip() for r in rows]
    jpg_names = {p.name for p in jpgs}
    missing_images = [n for n in csv_files if n not in jpg_names]
    orphan_images = [p.name for p in jpgs if p.name not in set(csv_files)]
    dup_files = [n for n, c in __import__("collections").Counter(csv_files).items() if c > 1]

    if missing_images:
        issues.append({"severity": "definite", "category": "missing_image", "item": "csv_without_image", "detail": f"count={len(missing_images)}; examples={missing_images[:5]}"})
    if orphan_images:
        issues.append({"severity": "definite", "category": "orphan_image", "item": "image_without_csv_row", "detail": f"count={len(orphan_images)}; examples={orphan_images[:5]}"})
    if dup_files:
        issues.append({"severity": "definite", "category": "duplicate_csv_row", "item": "filename", "detail": f"count={len(dup_files)}; examples={dup_files[:5]}"})
    if len(rows) != len(jpgs):
        issues.append({"severity": "unresolved", "category": "count_mismatch", "item": "nrows_vs_jpgs", "detail": f"csv_rows={len(rows)} jpgs={len(jpgs)}"})

    # Data dictionary from README_example.txt
    readme_doc = {
        "num": ("Index of drone image", "unitless", "README_example.txt"),
        "filename": ("Drone image filename e.g. 03_0001.JPG", "filename", "README_example.txt"),
        "date": ("When the drone image was taken", "timestamp string; timezone not stated", "README_example.txt"),
        "lat": ("Latitude of the centre of the drone image", "degrees; CRS implied WGS84 via satellite geokeys but not stated in CSV", "README_example.txt"),
        "lon": ("Longitude of the centre of the drone image", "degrees", "README_example.txt"),
        "height": ("Height at which the drone image was taken", "numeric; AGL vs AMSL vs ellipsoid NOT stated", "README_example.txt"),
        "Omega": ("Pitch of the drone when capturing", "degrees presumed; not stated", "README_example.txt"),
        "Kappa": ("Roll of the drone when capturing", "degrees presumed; not stated", "README_example.txt"),
        "Phi1": ("Yaw (higher confidence than Phi2)", "degrees presumed; zero/convention not stated", "README_example.txt"),
        "Phi2": ("Yaw (lower confidence than Phi1)", "degrees presumed; zero/convention not stated", "README_example.txt"),
        "mapname": ("Satellite map name", "filename stem/name", "README_example.txt"),
        "LT_lat_map": ("Latitude of left-top corner of satellite map", "degrees", "README_example.txt"),
        "LT_lon_map": ("Longitude of left-top corner of satellite map", "degrees", "README_example.txt"),
        "RB_lat_map": ("Latitude of right-bottom corner of satellite map", "degrees", "README_example.txt"),
        "RB_lon_map": ("Longitude of right-bottom corner of satellite map", "degrees; README has typo saying RB_lat twice", "README_example.txt"),
        "region": ("Area label where satellite and drone maps are located", "string place label; not proof of independence", "README_example.txt"),
    }

    dd = []
    for col in headers:
        vals = [r.get(col, "") for r in rows]
        missing = sum(1 for v in vals if str(v).strip() == "")
        examples = " | ".join(str(v) for v in vals[:3])
        meaning, units, evid = readme_doc.get(col, ("not documented in sample README", "not reported", "observed header only"))
        dd.append(
            {
                "file": "03.csv",
                "column": col,
                "data_type": "int" if col == "num" else ("float" if col in {"lat", "lon", "height", "Omega", "Kappa", "Phi1", "Phi2"} else "string"),
                "example_values": examples,
                "missing_count": missing,
                "missing_frac": f"{missing/len(vals):.6f}" if vals else "",
                "documented_meaning_units": f"{meaning} [{units}]",
                "evidence_source": evid,
                "inferred_meaning": "" if col in readme_doc else "Header present; meaning not in README_example.txt",
                "inference_confidence": "high" if col in readme_doc else "low",
            }
        )
    for col in range_headers:
        vals = [r.get(col, "") for r in range_rows]
        missing = sum(1 for v in vals if str(v).strip() == "")
        examples = " | ".join(str(v) for v in vals[:3])
        meaning, units, evid = readme_doc.get(col, ("not documented", "not reported", "observed"))
        dd.append(
            {
                "file": RANGE_CSV.name,
                "column": col,
                "data_type": "string" if col in {"mapname", "region"} else "float",
                "example_values": examples,
                "missing_count": missing,
                "missing_frac": f"{missing/len(vals):.6f}" if vals else "",
                "documented_meaning_units": f"{meaning} [{units}]",
                "evidence_source": evid,
                "inferred_meaning": "mapname '03.tif' vs file 'satellite03.tif' naming mismatch — unresolved alias rule",
                "inference_confidence": "medium" if col == "mapname" else "high",
            }
        )

    # Manifest
    man = []
    for r in rows:
        fn = r["filename"].strip()
        p = DRONE / fn
        man.append(
            {
                "region": "03",
                "num": r["num"],
                "filename": fn,
                "image_path": str(p) if p.exists() else "",
                "image_exists": p.exists(),
                "date": r["date"],
                "lat": r["lat"],
                "lon": r["lon"],
                "height": r["height"],
                "Omega": r["Omega"],
                "Kappa": r["Kappa"],
                "Phi1": r["Phi1"],
                "Phi2": r["Phi2"],
                "region_label_from_range_csv": range_rows[0]["region"] if range_rows else "",
            }
        )

    # File inventory
    inv = []
    for p in sorted(SAMPLE.rglob("*")):
        if p.is_file():
            inv.append(
                {
                    "relative_path": p.relative_to(SAMPLE).as_posix(),
                    "extension": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
                }
            )

    # EXIF/XMP probes
    idxs = [0, 1, 2, len(jpgs) // 2, len(jpgs) - 1]
    meta_reports = []
    for i in idxs:
        rep = probe_jpeg_metadata(jpgs[i])
        meta_reports.append(rep)
        if not rep["exif_app1"] and not rep["xmp"]:
            issues.append({"severity": "definite", "category": "no_embedded_exif_xmp_in_first_256kb", "item": rep["file"], "detail": "No APP1 Exif/XMP found in first 256KB; camera intrinsics not available from file"})
        elif not rep["exif_app1"] and rep["xmp"]:
            issues.append({"severity": "unresolved", "category": "xmp_only", "item": rep["file"], "detail": f"XMP present; fields={rep['interesting']}"})

    # Satellite tile preview via first tile
    sat_info = {}
    try:
        import tifffile
        from PIL import Image as PILImage

        with tifffile.TiffFile(SAT) as tif:
            page = tif.pages[0]
            sat_info = {
                "shape": list(page.shape),
                "dtype": str(page.dtype),
                "tiled": bool(page.is_tiled),
                "tile": [page.tilewidth, page.tilelength] if page.is_tiled else None,
                "ModelTiepointTag": list(page.tags["ModelTiepointTag"].value) if "ModelTiepointTag" in page.tags else None,
                "ModelPixelScaleTag": list(page.tags["ModelPixelScaleTag"].value) if "ModelPixelScaleTag" in page.tags else None,
                "GeoAsciiParamsTag": str(page.tags["GeoAsciiParamsTag"].value) if "GeoAsciiParamsTag" in page.tags else None,
                "GeographicTypeGeoKey_EPSG": 4326,
                "GTRasterTypeGeoKey": 1,  # PixelIsArea from prior run
            }
            seg = next(page.segments())
            tile = np.asarray(seg[1])
            # tile may be (th,tw,3)
            if tile.ndim == 2:
                tile = np.stack([tile] * 3, axis=-1)
            # take up to 512
            preview = tile[:512, :512, :3].astype(np.uint8)
            dest = FIGS / "satellite03_first_tile_preview.jpg"
            PILImage.fromarray(preview).save(dest, quality=85)
            sat_info["preview"] = str(dest)
            sat_info["first_tile_minmax"] = [int(tile.min()), int(tile.max())]
    except Exception as e:
        sat_info["error"] = str(e)
        issues.append({"severity": "definite", "category": "sat_preview", "item": SAT.name, "detail": str(e)})

    # Compare CSV bounds to geotiff
    if range_rows and sat_info.get("ModelTiepointTag"):
        tp = sat_info["ModelTiepointTag"]
        sc = sat_info["ModelPixelScaleTag"]
        w, h = 35092, 24308
        west, north = float(tp[3]), float(tp[4])
        sx, sy = float(sc[0]), float(sc[1])
        east = west + w * sx
        south = north - h * sy
        rr = range_rows[0]
        sat_info["bounds_from_geotiff"] = {"west": west, "east": east, "north": north, "south": south}
        sat_info["bounds_from_csv"] = {
            "LT_lon": float(rr["LT_lon_map"]),
            "LT_lat": float(rr["LT_lat_map"]),
            "RB_lon": float(rr["RB_lon_map"]),
            "RB_lat": float(rr["RB_lat_map"]),
        }
        sat_info["csv_vs_geotiff_delta"] = {
            "LT_lon": float(rr["LT_lon_map"]) - west,
            "LT_lat": float(rr["LT_lat_map"]) - north,
            "RB_lon": float(rr["RB_lon_map"]) - east,
            "RB_lat": float(rr["RB_lat_map"]) - south,
        }
        # Pixel size in metres approx
        import math

        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(north))
        sat_info["approx_gsd_m"] = {"x": sx * lon_m, "y": sy * lat_m}

    # mapname mismatch note
    if range_rows and range_rows[0]["mapname"] != "satellite03.tif":
        issues.append(
            {
                "severity": "unresolved",
                "category": "naming",
                "item": "mapname",
                "detail": f"range CSV mapname={range_rows[0]['mapname']} but file is satellite03.tif",
            }
        )

    # height / heading interpretation issues
    issues.append(
        {
            "severity": "unresolved",
            "category": "units",
            "item": "height",
            "detail": "README says 'at what height' but does not state AGL vs AMSL vs ellipsoid; do not treat as HAGL without further evidence",
        }
    )
    issues.append(
        {
            "severity": "unresolved",
            "category": "units",
            "item": "Phi1_Phi2",
            "detail": "Yaw fields documented; units/zero/north-clockwise convention not stated; Phi1 preferred over Phi2; vehicle yaw vs camera yaw not stated",
        }
    )

    # Write outputs
    write_csv(
        MANIFESTS / "data_dictionary.csv",
        dd,
        [
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
        ],
    )
    write_csv(
        MANIFESTS / "manifest.csv",
        man,
        [
            "region",
            "num",
            "filename",
            "image_path",
            "image_exists",
            "date",
            "lat",
            "lon",
            "height",
            "Omega",
            "Kappa",
            "Phi1",
            "Phi2",
            "region_label_from_range_csv",
        ],
    )
    write_csv(MANIFESTS / "sample_file_inventory.csv", inv, ["relative_path", "extension", "size_bytes"])
    write_csv(REPORTS / "sample_frame_issues.csv", issues, ["severity", "category", "item", "detail"])

    summary = {
        "generated_at": utc_now(),
        "n_jpg": len(jpgs),
        "n_csv_rows": len(rows),
        "missing_images": len(missing_images),
        "orphan_images": len(orphan_images),
        "range_csv_name": RANGE_CSV.name,
        "range_rows": range_rows,
        "jpeg_meta_reports": meta_reports,
        "satellite": sat_info,
        "headers": headers,
    }
    (REPORTS / "sample_audit_raw_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"ok": True, "n_jpg": len(jpgs), "n_issues": len(issues), "sat_preview": sat_info.get("preview"), "delta": sat_info.get("csv_vs_geotiff_delta")}, indent=2))


if __name__ == "__main__":
    main()
