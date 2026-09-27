# Stage 3A — Matcher protocol (frozen after pilot)

**Frozen matcher:** `pyramid_ncc_rot4`  
**Frozen at (UTC):** 2026-09-22T22:44:01.897745+00:00  
**Config:** `configs/matcher_protocol.yaml`  
**Full-label gate:** **NOT READY** (see `reports/stage3a_pilot_summary.md`)

---

## 1. Reproducible localization task

| Role | Definition |
|---|---|
| **Queries** | Real UAV JPG images from UAV-VisLoc (GitHub IntelliSensing release; **6,774** frames). |
| **Reference DB** | Satellite raster(s) for the **query’s region map**, tiled into an overlapping searchable index. |
| **Predicted location** | Comes **only** from the matcher (NCC peak → WGS84 via georef, or tile centre / homography for other candidates). |
| **CSV lat/lon** | Used **only** for geodesic scoring after prediction. **Never** used to crop, mask, or rank the search area. |
| **Success** | Geodesic position error (WGS84, `pyproj` Geod) ≤ threshold (metres). |

Pilot scope: region **03** only (matcher **development** region). This is a function check, **not** held-out-region performance.

---

## 2. Leakage prevention (Stage 2C)

Source: `reports/independent_region_groups.csv`, `reports/region_independence.md`.

| Rule | Protocol |
|---|---|
| Overlap merge | Regions **08+09** always one coupled unit. |
| Matcher tuning | Dev subset + pilot on region **03** only. |
| Evaluation (future full run) | Hold out **entire group units**; never tune on evaluation units. |
| Reporting plan A | **10-group** overlap-only (`threshold_m=0`): `01\|02\|03\|04\|05\|06\|07\|08+09\|10\|11` |
| Reporting plan B | **7-group** 5 km sensitivity: `01+02\|03+04\|05\|06\|07+10\|08+09\|11` |

Under the 7-group definition, **03+04** are one unit — future holdouts that use 5 km grouping must exclude **both** if 03 was used for tuning.

Spatial groups reduce geographic leakage; they do **not** prove visual independence (Stage 1–2 limitation).

---

## 3. Intended matcher (proposal) vs Stage 3A candidates

**Proposal (research material, not instructions):** *“Start with a compact DINOv3 matcher; switch to a documented alternative if access is delayed.”* Freeze backbone, resolution, rotation/scale policy before final evaluation.

**Hardware (this workspace):** Dell XPS-class, **16 GB RAM**, Intel **Iris Xe**, CPU-first path.

| Candidate | Role in 3A | Why considered |
|---|---|---|
| compact DINOv3 | Proposal default | **Not executed** in 3A (weights/access + VRAM/deps). Documented deferred path. |
| `pyramid_ncc_rot4` | **Selected / frozen** | Full-map downsampled NCC + 0/90/180/270° query bank; fits CPU/RAM. |
| `akaze_coarse_to_fine` | Dev-only | Tile coarse NCC → AKAZE + RANSAC when inliers suffice. |
| `coarse_ncc` | Dev-only | Tile-level 64×64 NCC; winner tile centre. |

### Dev-only comparison (n=24, region 03, seed 42)

| Matcher | Median error (m) | Success@50 / @100 / @250 / @500 m |
|---|---:|---|
| **pyramid_ncc_rot4** | ~3460 | 0 / 0 / 0 / 4.2% |
| akaze_coarse_to_fine | ~4125 | 0 / 0 / 0 / 0 |
| coarse_ncc | ~3887 | 0 / 0 / 0 / 0 |

**Selection rule:** maximize success@100 m, then minimize median error → **`pyramid_ncc_rot4`**.

---

## 4. Satellite tiling, resolution, preprocessing

| Parameter | Value | Rationale |
|---|---|---|
| Tile size | **1024** px | Large enough for local texture; fits Iris Xe / 16 GB when windowed. |
| Stride | **512** px | **~50%** overlap so centres are not unique grid artefacts. |
| Tile centre | Pixel centre → WGS84 via TIFF georef | Same centre convention as Stage 2B/2D. |
| Region 09 | Union of four tiles (future) | No single `satellite09.tif` (Stage 1–2). |
| Query prep | RGB, max long edge **768** px | Memory-safe; no EXIF intrinsics. |
| Rotation | Frozen matcher: fixed **0/90/180/270°** bank | Phi1 vehicle vs camera yaw **unresolved** (Stage 2D); CSV heading **not** used (would leak pose into search). |
| Scale | Fixed resize only | Height datum + intrinsics unresolved; no height-based GSD scaling. |
| Footprint GT | **Not used** | Stage 2D: footprint polygons **APPROXIMATE** / not validated. |

`pyramid_ncc_rot4` additionally downsamples the **entire** regional satellite to max long edge **4096** px and runs template matching over the full map (no GT mask). Tile index (**3082** tiles on region 03) records searchable DB size for tile-based candidates and leakage audits.

---

## 5. Success thresholds (geodesic metres)

| Threshold | Justification |
|---:|---|
| **50 m** | Strict; ~0.18× satellite GSD (~0.27 m/px on region 03). |
| **100 m** | Primary; ~1× median along-track spacing region 03 (~95 m, `reports/flight_stats.csv`). |
| **250 m** | Moderate coarse fix. |
| **500 m** | Lenient coarse localization / upper provisional search-radius class. |

Stage 2D NCC shift (~97–105 m) is **not** treated as proven CSV geolocation error and is **not** used as a success threshold.

---

## 6. Limitations preserved (Stage 1–2)

From `reports/stage1_stage2_final.md`: height datum unknown; no intrinsics/EXIF; Phi/Omega/Kappa semantics unresolved; footprint polygons not validated; 6,774 vs README 6,742; region 09 multi-tile; spatial groups ≠ visual independence; MLSNet unread.

---

## 7. Reproduction

```bash
# Full Stage 3A (dev select + 100-query pilot; rewrites config freeze timestamp)
py -3 F:\CMBL\scripts\localization\run_stage3a.py

# Regenerate inspection figures from frozen pilot CSV only
py -3 F:\CMBL\scripts\localization\stage3a_inspect_figures.py
```

Modules: `scripts/localization/{run_stage3a,pyramid_ncc,matchers,tile_index,geo_utils,stage3a_inspect_figures}.py`.
