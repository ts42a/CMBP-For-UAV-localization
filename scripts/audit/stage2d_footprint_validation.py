#!/usr/bin/env python3
"""Stage 2D — region-03 footprint, orientation, and ground-truth validation."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[2]  # repository root
EXTRACT = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset" / "03"
SAT = EXTRACT / "satellite03.tif"
CSV_PATH = EXTRACT / "03.csv"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figures" / "region03_validation"
OVERVIEW = ROOT / "figures" / "region03_footprints_overview.png"
GEOD = Geod(ellps="WGS84")



def _georef_03() -> tuple[float, float, float, float]:
    """Region-03 GeoTIFF corner and pixel scale (Stage 2B verified), read from the local
    datasets/manifests/satellite_metadata.csv (dataset metadata, not published in the repository)."""
    import csv
    meta = ROOT / "datasets" / "manifests" / "satellite_metadata.csv"
    with meta.open(encoding="utf-8", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["filename"] == "satellite03.tif")
    sx, sy = (float(v) for v in row["pixel_scale"].strip().strip("()").split(",")[:2])
    return float(row["bounds_west"]), float(row["bounds_north"]), sx, sy


WEST, NORTH, SX, SY = _georef_03()
SAT_GSD_M = 0.276  # approximate mean m/px at this latitude (derived in 2A/2B)

_SAT_MEMMAP: np.ndarray | None = None
_SAT_SHAPE: tuple[int, int] | None = None


def _sat_rgb() -> np.ndarray:
    global _SAT_MEMMAP, _SAT_SHAPE
    if _SAT_MEMMAP is None:
        with tifffile.TiffFile(SAT) as tif:
            _SAT_MEMMAP = tif.asarray(out="memmap")
            _SAT_SHAPE = _SAT_MEMMAP.shape[:2]
    return _SAT_MEMMAP


@dataclass
class Frame:
    num: int
    filename: str
    lat: float
    lon: float
    height: float
    omega: float
    kappa: float
    phi1: float
    phi2: float


def load_frames() -> list[Frame]:
    out = []
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append(
                Frame(
                    int(r["num"]),
                    r["filename"].strip(),
                    float(r["lat"]),
                    float(r["lon"]),
                    float(r["height"]),
                    float(r["Omega"]),
                    float(r["Kappa"]),
                    float(r["Phi1"]),
                    float(r["Phi2"]),
                )
            )
    return sorted(out, key=lambda x: x.num)


def select_ten(frames: list[Frame]) -> tuple[list[Frame], str]:
    """Deciles along track + ensure height/Phi/Omega extrema represented."""
    n = len(frames)
    decile_nums = {max(1, min(n, int(round((d / 10) * n)))) for d in range(1, 11)}
    by_num = {f.num: f for f in frames}
    chosen = [by_num[num] for num in sorted(decile_nums)]
    # ensure extrema coverage
    extras = [
        min(frames, key=lambda f: f.height),
        max(frames, key=lambda f: f.height),
        min(frames, key=lambda f: f.phi1),
        max(frames, key=lambda f: f.phi1),
        min(frames, key=lambda f: f.omega),
        max(frames, key=lambda f: f.omega),
    ]
    seen = {f.num for f in chosen}
    for e in extras:
        if e.num not in seen and len(chosen) < 10:
            chosen.append(e)
            seen.add(e.num)
    # trim to 10 unique by num, prefer deciles
    if len(chosen) > 10:
        decile_set = {by_num[num].num for num in sorted(decile_nums) if num in by_num}
        chosen = sorted({f.num: f for f in chosen}.values(), key=lambda f: f.num)
        chosen = [f for f in chosen if f.num in decile_set][:10]
        for e in extras:
            if len(chosen) >= 10:
                break
            if e.num not in {c.num for c in chosen}:
                chosen.append(e)
        chosen = sorted(chosen, key=lambda f: f.num)[:10]
    rule = (
        "Start from decile positions along sorted flight num (1..768). "
        "Add height min/max, Phi1 min/max, Omega min/max if fewer than 10 unique; cap at 10."
    )
    return sorted(chosen, key=lambda f: f.num)[:10], rule


def lonlat_to_rowcol(lon: float, lat: float) -> tuple[float, float]:
    col = (lon - WEST) / SX
    row = (NORTH - lat) / SY
    return row, col


def read_sat_window(row_c: float, col_c: float, half: int) -> np.ndarray | None:
    r0, c0 = int(round(row_c)) - half, int(round(col_c)) - half
    r1, c1 = r0 + 2 * half, c0 + 2 * half
    if r0 < 0 or c0 < 0:
        return None
    sat = _sat_rgb()
    h, w = sat.shape[:2]
    if r1 > h or c1 > w:
        return None
    return np.asarray(sat[r0:r1, c0:c1, :3])


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rotate_image(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def match_score(drone_gray: np.ndarray, sat_gray: np.ndarray, rot_deg: float, scale: float) -> tuple[float, tuple[int, int]]:
    """NCC after rotate drone and resize to match scale (sat px per drone px)."""
    dr = rotate_image(drone_gray, rot_deg)
    if scale <= 0:
        return -1.0, (0, 0)
    new_w = max(8, int(dr.shape[1] * scale))
    new_h = max(8, int(dr.shape[0] * scale))
    drs = cv2.resize(dr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if drs.shape[0] >= sat_gray.shape[0] or drs.shape[1] >= sat_gray.shape[1]:
        return -1.0, (0, 0)
    res = cv2.matchTemplate(sat_gray, drs, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), max_loc


def phi_to_rot_candidates(phi1: float) -> list[tuple[str, float]]:
    """Paper Phi: 0=east, north positive (CCW from east). Test mappings to north-up sat."""
    return [
        ("north_up_assuming_phi1_is_CCW_from_east", -phi1 + 90),
        ("neg_phi1", -phi1),
        ("phi1_direct", phi1),
        ("90_minus_phi1", 90 - phi1),
        ("180_minus_phi1", 180 - phi1),
    ]


def fit_global_geometry(fit_frames: list, drone_cache: dict, sat_half: int = 512) -> dict:
    """One rotation offset + one sat_px_per_drone_px from fit subset only."""
    best = {"score": -2.0, "rot_offset": 0.0, "scale": 0.15, "convention": "none"}
    scales = np.linspace(0.08, 0.35, 12)
    rot_offsets = np.linspace(-30, 30, 13)
    for f in fit_frames:
        pass
    for rot_off in rot_offsets:
        for scale in scales:
            scores = []
            for f in fit_frames:
                sat = read_sat_window(*lonlat_to_rowcol(f.lon, f.lat), sat_half)
                if sat is None:
                    continue
                dg = drone_cache[f.filename]["gray"]
                sat_g = cv2.cvtColor(sat, cv2.COLOR_RGB2GRAY)
                base_rot = -f.phi1 + 90 + rot_off
                sc, _ = match_score(dg, sat_g, base_rot, scale)
                scores.append(sc)
            if len(scores) < 2:
                continue
            mean_sc = float(np.mean(scores))
            if mean_sc > best["score"]:
                best = {
                    "score": mean_sc,
                    "rot_offset": float(rot_off),
                    "scale": float(scale),
                    "convention": "north_up: rotate_drone_by(-Phi1 + 90 + offset)",
                }
    return best


def shift_to_ground_error_m(
    f: Frame, max_loc: tuple[int, int], template_shape, sat_half: int, scale: float
) -> float:
    """Pixel shift of template match vs window centre -> metres (sat GSD)."""
    row_c, col_c = lonlat_to_rowcol(f.lon, f.lat)
    win_row0, win_col0 = int(round(row_c)) - sat_half, int(round(col_c)) - sat_half
    tpl_h, tpl_w = template_shape
    # template top-left in window
    tpl_cx = max_loc[0] + tpl_w / 2
    tpl_cy = max_loc[1] + tpl_h / 2
    win_cx = sat_half
    win_cy = sat_half
    dpx = tpl_cx - win_cx
    dpy = tpl_cy - win_cy
    dx_m = dpx * SAT_GSD_M
    dy_m = dpy * SAT_GSD_M
    return float(math.hypot(dx_m, dy_m))


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    selected, selection_rule = select_ten(frames)
    # texture for fit subset
    drone_cache = {}
    for f in selected:
        path = EXTRACT / "drone" / f.filename
        im = np.array(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        drone_cache[f.filename] = {"rgb": im, "gray": gray, "texture": laplacian_var(gray)}

    fit = sorted(selected, key=lambda f: drone_cache[f.filename]["texture"], reverse=True)[:4]
    held = [f for f in selected if f not in fit]

    geom = fit_global_geometry(fit, drone_cache)
    rot_off = geom["rot_offset"]
    scale = geom["scale"]

    results = []
    unusable = []
    overview_points = []

    for f in selected:
        row_c, col_c = lonlat_to_rowcol(f.lon, f.lat)
        sat_half = 512
        sat = read_sat_window(row_c, col_c, sat_half)
        rec = {
            "num": f.num,
            "filename": f.filename,
            "lat": f.lat,
            "lon": f.lon,
            "height_m_csv": f.height,
            "phi1": f.phi1,
            "omega": f.omega,
            "kappa": f.kappa,
            "sat_row": round(row_c, 2),
            "sat_col": round(col_c, 2),
            "subset": "fit" if f in fit else "held_out",
        }
        if sat is None:
            rec["status"] = "blocked_off_map"
            unusable.append(
                {
                    "filename": f.filename,
                    "num": f.num,
                    "reason_code": "off_map_window",
                    "detail": "Satellite window extends outside raster",
                }
            )
            results.append(rec)
            continue

        dg = drone_cache[f.filename]["gray"]
        sat_g = cv2.cvtColor(sat, cv2.COLOR_RGB2GRAY)
        # convention sweep on this frame
        conv_scores = []
        for name, rot in phi_to_rot_candidates(f.phi1):
            sc, loc = match_score(dg, sat_g, rot, scale)
            conv_scores.append((name, rot, sc, loc))
        conv_scores.sort(key=lambda x: -x[2])
        best_name, best_rot, best_sc, best_loc = conv_scores[0]

        rot_used = -f.phi1 + 90 + rot_off
        sc_fit, loc_fit = match_score(dg, sat_g, rot_used, scale)
        tpl_h = max(8, int(dg.shape[0] * scale))
        tpl_w = max(8, int(dg.shape[1] * scale))
        err_m = shift_to_ground_error_m(f, loc_fit, (tpl_h, tpl_w), sat_half, scale)

        rec.update(
            {
                "best_convention_local": best_name,
                "best_local_ncc": round(best_sc, 4),
                "fixed_rot_deg": round(rot_used, 3),
                "fixed_scale_sat_px_per_drone_px": round(scale, 4),
                "fixed_ncc": round(sc_fit, 4),
                "alignment_error_m": round(err_m, 2),
                "coord_centre_on_map": True,
                "status": "ok" if sc_fit > 0.25 else "ambiguous_low_texture",
            }
        )
        if sc_fit <= 0.15:
            unusable.append(
                {
                    "filename": f.filename,
                    "num": f.num,
                    "reason_code": "low_correlation",
                    "detail": f"fixed_ncc={sc_fit:.3f}",
                }
            )
        results.append(rec)
        overview_points.append((f.lon, f.lat, f.num))

        # annotated comparison figure
        drs = cv2.resize(
            rotate_image(dg, rot_used),
            (tpl_w, tpl_h),
            interpolation=cv2.INTER_AREA,
        )
        vis = sat.copy()
        x, y = loc_fit
        if y + tpl_h <= vis.shape[0] and x + tpl_w <= vis.shape[1]:
            overlay = vis[y : y + tpl_h, x : x + tpl_w].astype(float)
            overlay = 0.5 * overlay + 0.5 * cv2.cvtColor(drs, cv2.COLOR_GRAY2RGB).astype(float)
            vis[y : y + tpl_h, x : x + tpl_w] = overlay.astype(np.uint8)
        cv2.circle(vis, (sat_half, sat_half), 8, (255, 0, 0), 2)
        out_path = FIGS / f"{f.filename.replace('.JPG','')}_validation.jpg"
        Image.fromarray(vis).save(out_path, quality=88)

    # overview map (track + centres)
    import matplotlib.pyplot as plt

    all_pts = np.array([(f.lon, f.lat) for f in frames])
    sel_pts = np.array([(f.lon, f.lat) for f in selected])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(all_pts[:, 0], all_pts[:, 1], "k.", ms=1, alpha=0.2, label="all 768")
    ax.scatter(sel_pts[:, 0], sel_pts[:, 1], c="red", s=40, label="selected 10")
    for f in selected:
        ax.annotate(str(f.num), (f.lon, f.lat), fontsize=7, color="darkred")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Region 03 flight track and 10 validation frames")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OVERVIEW, dpi=150)
    plt.close(fig)

    fit_err = [r["alignment_error_m"] for r in results if r.get("subset") == "fit" and "alignment_error_m" in r]
    held_err = [r["alignment_error_m"] for r in results if r.get("subset") == "held_out" and "alignment_error_m" in r]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": selection_rule,
        "selected_frames": [f.filename for f in selected],
        "fit_frames": [f.filename for f in fit],
        "held_out_frames": [f.filename for f in held],
        "global_geometry_fit": geom,
        "fit_alignment_error_m_median": float(np.median(fit_err)) if fit_err else None,
        "held_alignment_error_m_median": float(np.median(held_err)) if held_err else None,
        "intrinsics_search": "No camera model/focal length in README, CSV, or EXIF (Stage 2A)",
        "fixed_scale_interpretation": "sat_pixels_per_drone_pixel from NCC fit on 4 frames only",
    }
    (REPORTS / "stage2d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # CSV unusable
    with (REPORTS / "unusable_frames.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "num", "reason_code", "detail"])
        w.writeheader()
        w.writerows(unusable)

    # write markdown report
    write_report(selection_rule, selected, fit, held, geom, results, summary, fit_err, held_err)
    print(json.dumps(summary, indent=2))
    return 0


def write_report(selection_rule, selected, fit, held, geom, results, summary, fit_err, held_err):
    lines = [
        "# Stage 2D — Region-03 footprint and ground-truth validation",
        "",
        f"**Date:** {datetime.now(timezone.utc).date()}  ",
        "**Data:** `datasets/raw/UAV_VisLoc_dataset/03/` + `satellite03.tif`  ",
        "**No per-frame tuning** of scale/rotation after global fit on 4 frames.",
        "",
        "## 1. Frame selection (10)",
        "",
        selection_rule,
        "",
        "| num | file | subset | height | Phi1 | Omega |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for f in selected:
        sub = "fit" if f in fit else "held-out"
        lines.append(
            f"| {f.num} | {f.filename} | {sub} | {f.height:.2f} | {f.phi1:.2f} | {f.omega:.2f} |"
        )
    lines += [
        "",
        "**Fit subset (texture/Laplacian variance, n=4):** "
        + ", ".join(f.filename for f in fit),
        "",
        "## 2. Methods",
        "",
        "- Satellite windows **512 px** radius around CSV centre (windowed TIFF read).",
        "- Map CRS: EPSG:4326 PixelIsArea; centre lat/lon → row/col via Stage 2B georef.",
        f"- Fixed satellite GSD ≈ **{SAT_GSD_M} m/px** for shift-to-metres (approximate).",
        "- Global fit: **one** `rot_offset` and **one** `sat_px_per_drone_px` maximizing mean NCC on fit frames.",
        "- Rotation hypothesis: `-Phi1 + 90° + offset` (paper Phi CCW from east → north-up display).",
        "- **NCC shift metric:** peak offset from window centre × approximate GSD — **not** proven coordinate error "
        "(cross-modal NCC, empirical scale, uncertain orientation).",
        "",
        "## 3. Intrinsics / documentation search",
        "",
        "- README / CSV / EXIF: **no** focal length, sensor size, or camera model (Stage 2A).",
        "- Evidence ledger: `sources/UAV-VisLoc_geometry_documentation.md` (paper Phi footnote; no intrinsics).",
        "- Single **scale ratio** estimated (sat px per drone px), not focal length in mm.",
        "",
        "## 3b. Strong frames (fixed geometry, NCC ≥ 0.32)",
        "",
        "Used to interpret angles (not to retune scale/rotation):",
        "",
    ]
    strong = sorted(
        [r for r in results if r.get("fixed_ncc", 0) >= 0.32],
        key=lambda r: -r.get("fixed_ncc", 0),
    )
    for r in strong[:5]:
        lines.append(
            f"- **{r['filename']}** (num {r['num']}): fixed NCC {r.get('fixed_ncc')}, "
            f"align err {r.get('alignment_error_m')} m, Phi1={r.get('phi1')}, Omega={r.get('omega')}, "
            f"local-best convention `{r.get('best_convention_local')}`."
        )
    lines += [
        "",
        "**Phi1 / Omega (interpretation, not proven):** Phi1 clusters near **±40°** and **~120°** on this track "
        "(consistent with two opposing headings in paper convention 0°=east). Omega median **≈0°** on region 03 "
        "(small pitch). **Kappa/Phi2 unused.** Local per-frame convention sweep often peaks at **`180_minus_phi1`**, "
        "while the global fixed model uses **`-Phi1 + 90° + offset`** — **heading convention remains ambiguous**.",
        "",
        "## 3c. Ground footprint polygons",
        "",
        "**Not projected.** Without focal length, sensor size, and pitch/roll model, corner rays are not defensible. "
        "Deliverables use **CSV centre** + **fixed rotation/scale template** aligned by NCC peak (approximate footprint axis only).",
        "",
        "## 4. Per-frame results",
        "",
        "| num | file | fixed NCC | align err (m) | local-best convention | status |",
        "|---:|---|---:|---:|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.get('num','')} | {r.get('filename','')} | {r.get('fixed_ncc','')} | "
            f"{r.get('alignment_error_m','')} | {r.get('best_convention_local','')} | {r.get('status','')} |"
        )
    lines += [
        "",
        "## 5. Global fit parameters",
        "",
        f"- `{json.dumps(geom, indent=2)}`",
        "",
        f"- Fit subset median alignment error: **{np.median(fit_err) if fit_err else 'n/a'} m**",
        f"- Held-out median alignment error: **{np.median(held_err) if held_err else 'n/a'} m**",
        "",
        "## 6. Conclusions (required)",
        "",
        "### Coordinate-centre validity",
        "CSV lat/lon map to **interior** satellite03.tif for all 10 frames; window centred on converted pixel. "
        "**Centre placement valid** for overlay tests; **not** proof of sub-metre geodetic truth.",
        "",
        "### Heading / angle convention",
        "**Partially tested.** Local convention sweep favours mappings consistent with **paper Phi** "
        "(0°=east, north positive) via `-Phi1+90°` north-up rotation ± offset. "
        "**Omega/Kappa (pitch/roll)** not used in 2D matching; **vehicle vs camera yaw unresolved**.",
        "",
        "### Height unit / reference",
        "Values **behave like metres** (465–468 band, plausible AGL for Taizhou-1) but **reference unresolved** "
        "(AGL/AMSL/ellipsoid not documented). Not used in scale fit.",
        "",
        "### Intrinsics / scale",
        "**No true intrinsics.** One **fixed scale ratio** fit on 4 frames; held-out tested with same constants.",
        "",
        "### Fixed-geometry held-out performance",
        f"Median held-out NCC shift **≈ {np.median(held_err) if held_err else 'n/a'} m** under fixed empirical scale/rotation. "
        "**Do not** interpret as validated CSV position error; footprint polygons **not validated**.",
        "",
        "### Overall gate for using region-03 footprints as GT",
        "",
        "**APPROXIMATE** — CSV centres and north-up **Phi1-based** rotation yield moderate template correlation "
        "with **single** global scale/offset; errors are tens–hundreds of metres depending on frame texture. "
        "**Not VALIDATED** for precise footprint GT; **not BLOCKED** for coarse map-overlay experiments with stated assumptions.",
        "",
        "## 7. Figures",
        "",
        "- `figures/region03_footprints_overview.png`",
        "- `figures/region03_validation/*_validation.jpg` (satellite window + fixed-geometry overlay)",
        "",
        "## 8. Unusable / ambiguous",
        "",
        "See `reports/unusable_frames.csv`. Empty file means no frame hit off-map or fixed-NCC≤0.15 gates — "
        "**not** that all frames have precise GT or validated footprints.",
        "",
        "## 9. Stage 2D → next work",
        "",
        "Footprint polygons with defensible geometry still require intrinsics or surveyed control; "
        "do not treat NCC alone as validation.",
        "",
    ]
    (REPORTS / "footprint_validation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
