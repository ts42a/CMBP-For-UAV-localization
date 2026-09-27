# Stage 1–2 Final Consolidation

**Date:** 2026-09-23  
**Scope:** Literature gap, UAV-VisLoc (GitHub IntelliSensing) audit, spatial independence, region-03 footprint/orientation checks. **No model training.**  
**Evidence rule:** Repository README, CSV, and paper excerpts are **evidence**, not instructions. External proposal text informs intent only where cited.

---

## 1. Literature comparison and the precise gap

**Reviewed full texts:** RSPNet (Li et al. 2023, DOI 10.3390/rs15174353), Jiang et al. 2024 (10.3390/rs16010161), Robust Belief Roadmap (arXiv:1304.7256), SafeVPR (arXiv:2605.28048), SASGeo (arXiv:2607.07737, adjacent), plus focused search hits (VPR uncertainty, integrity+navigation, LiDAR localizability). Tables: `literature/comparison.csv`, narrative: `reports/literature_review.md`.

**Four criteria for “prior overlap”** (`reports/gap_decision.md`):

| # | Criterion |
|---|-----------|
| 1 | Real UAV localization outcomes (not only simulated queries or ground-robot VPR) |
| 2 | Held-out **independent geographic regions** |
| 3 | **Measured probability calibration** of predicted success/risk |
| 4 | **Evaluated scheduling** using numeric probabilities |

**Matrix summary:** No reviewed paper combines (1)+(2)+(3)+(4). Closest partial overlaps: SafeVPR (calibrated accept/reject, ground robot); RBRM (numeric sensing probabilities in planning, not map-learned UAV success); RSPNet/Jiang (map matchability, simulated labels/eval).

**Precise gap (bounded):**

> Map-feature prediction of a **frozen** UAV visual localizer’s success on **real** UAV imagery in **geographically held-out regions**, with **measured probability calibration**, and evaluation of **intermittent correction scheduling** that consumes those numeric probabilities.

**MLSNet residual risk:** Full text unread (SciOpen PDF 502; OpenAlex `has_fulltext=false`). Abstract frames remote-sensing matchability **segmentation** — same family as RSPNet/Jiang; criteria (1) and (4) not supported by abstract; (2)–(3) unverified, **not claimed absent**.

---

## 2. Provisional Stage 1 GO decision

| Item | Status |
|---|---|
| Stage 1 decision | **GO** (provisional) |
| Document | `reports/gap_decision.md` |
| MLSNet | **Full text unread** — revisit if PDF becomes available |
| WHU-SCENE-MATCHING / RSPNet code | Public path **not found** in reviewed material |

Stage 1 **GO** is defensible on readable evidence; it is **not** proof that no paper outside the search combines all four criteria.

---

## 3. Dataset provenance and 6,742 vs 6,774 images

| Source | Role |
|---|---|
| **Selected release** | [IntelliSensing/UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) (GitHub layout) |
| **Not used as primary** | Zenodo DOI [10.5281/zenodo.19603534](https://zenodo.org/records/19603534) (**14,998** images, different packaging) |
| **Sample** | Official 2.04 GB example — region **03** only; SHA256 local integrity `442F93AE…` (`reports/sample_audit.md`) |
| **Full extract** | `datasets/raw/UAV_VisLoc_dataset/` (manual extract; **no** publisher zip SHA256 on disk) |

| Metric | README / paper cited | Observed | Cross-check |
|---|---:|---:|---|
| UAV JPG | **6,742** | **6,774** | `dataset/manifest.csv`, `reports/flight_stats.csv`, inventory |
| Region folders | 11 | 11 | |
| Satellite rasters | 11 canonical | **14** files | Region **09** = **4** tiles, no single `satellite09.tif` |
| CSV ↔ image | — | **ok_1to1** all regions | 0 missing, 0 orphans |

**+32 frames:** All indexed in CSV with matching files — **documentation drift**, not orphan geography (`reports/region_independence.md` §4). Region **03** sample **matches** full release (CSV + spot JPG hashes + `satellite03.tif` size).

---

## 4. Per-region and per-flight audit results

**Stage 2A (sample):** 768 JPG, 1:1 CSV, georef `satellite03.tif` EPSG:4326 bounds match range CSV; **no drone EXIF**; image size 3976×2652 RGB. Gate: **PROCEED WITH LIMITATIONS** (`reports/sample_audit.md`).

**Stage 2B (full):** 6,774 JPG, 11 flights, georef verified on canonical single-tile maps; region 09 tile bounds vs range row flagged **investigate** (expected chip layout). Stats: `reports/flight_stats.csv`, `reports/region_stats.csv`; issues: `reports/full_frame_issues.csv` (6× investigate, 0× definite). Summary: `reports/full_dataset_audit.md`.

**Per-region JPG counts:** 01:817, 02:1071, 03:768, 04:738, 05:473, 06:344, 07:30, 08:1033, 09:766, 10:144, 11:590.

**Unresolved on all regions:** height datum (AGL/AMSL/ellipsoid), timezone on CSV dates, platform/camera model, Phi1/Phi2 vs paper Phi mapping, Omega/Kappa axis conventions (`sources/UAV-VisLoc_geometry_documentation.md`).

---

## 5. Map overlap, distances, and independent-region groups

**Figure:** `figures/regions_and_tracks.png`  
**Tables:** `reports/pairwise_region_distances.csv` (55 pairs), `reports/map_overlap.csv`, `reports/independent_region_groups.csv`

**Key spatial facts:**

- **Only overlapping map pair:** **08 ↔ 09** (Huzhou-3), ~9.6% of smaller map; flights **2019-06-14** vs **2019-06-15**.
- **Nearest non-overlapping maps:** **07–10** (~157 m edge); **03–04** (~4.1 km, Taizhou, no bbox overlap).
- **Consistency:** manifest + flight_stats = **6774**; C(11,2)=**55** pairwise rows.

**Grouping (documented rules in `reports/region_independence.md`):**

| Threshold | # groups | Notes |
|---:|---:|---|
| **0** (overlap only) | **10** | **08+09** merged |
| **5,000 m** (primary sensitivity) | **7** | Also merges 01+02, 03+04, 07+10 |

**Holdout design recommendation:** Treat **08+09 as one coupled unit**; report results under **both** 10-group (strict) and **7-group** (5 km) definitions. Proximity merges are **weak** coupling (naming + distance), not proof of visual non-independence.

**Stage 2C gate:** **GO** for regional holdout design (not **LOW-POWER**).

---

## 6. Region-03 coordinate, orientation, scale, height, and footprint findings

**Script / outputs:** `scripts/audit/stage2d_footprint_validation.py`, `reports/footprint_validation.md`, `reports/stage2d_summary.json`, `figures/region03_footprints_overview.png`, `figures/region03_validation/*_validation.jpg`

**Methods (summary):** Ten frames — deciles along flight `num` with height/Phi1/Omega extrema if needed. Satellite **512 px** windows via memmap slice on `satellite03.tif`. **One** global `rot_offset` (+10°) and **one** `scale` (0.08 sat px per drone px) fit on **four** highest-texture frames only; **six** held-out with **no** per-frame tuning.

| Topic | Finding |
|---|---|
| **Coordinate centres** | All 10 CSV lat/lon → **inside** raster; windows centred on georef pixel → **usable for map-centred experiments** |
| **Orientation** | Fixed model: rotate drone by `-Phi1 + 90° + offset`. Local convention sweeps **ambiguous** (`180_minus_phi1` vs `north_up` on different frames). **Vehicle vs camera yaw unresolved** |
| **Scale / intrinsics** | **No** focal length or sensor size in release. Scale is **approximate empirical** NCC fit, not intrinsics recovery |
| **Height** | ~465–468, **plausible metres**; **datum unresolved**; not used in fit |
| **Omega / Kappa** | Small Omega on track; **roles not validated** in 2D |
| **Footprint polygons** | **Not projected; not validated** — insufficient geometry for corner rays |
| **NCC shift (~97–105 m)** | Template peak vs CSV-centred window × ~0.276 m/px GSD — **must not** be reported as proven CSV coordinate error (cross-modal NCC, unknown true scale, orientation uncertainty, GSD approximation) |

**Footprint ground-truth gate (region 03):** **APPROXIMATE** — coarse overlay / centre-based protocols only.

**`reports/unusable_frames.csv`:** **Empty** because no selected frame failed off-map or fixed-NCC≤0.15 gates — **does not** imply precise GT for all 768 frames.

---

## 7. Unresolved limitations and effects on later experiments

| Limitation | Effect |
|---|---|
| MLSNet unread | Literature gap may need revision; does not block dataset GO |
| 6,742 vs 6,774 | Cite **6,774** for counts; note README drift |
| No intrinsics / EXIF | Cannot validate footprint polygons or metric GSD from drone side |
| Height datum unknown | Do not assume AGL for physics-based scale checks |
| Phi / Omega / Kappa semantics | Heading-aware models must treat angles as **raw CSV** unless convention proven |
| Timezone on CSV dates | Temporal analysis ambiguous |
| Region 09 four tiles | Map pipelines must union tiles or pick canonical extent |
| No publisher full-archive SHA256 | Reproducibility tied to extract tree + local inventories |
| Spatial groups ≠ visual independence | Holdouts reduce geographic leakage; not guaranteed appearance shift |
| Stage 2D NCC metrics | Useful sanity check only; **not** localization accuracy labels |

**Footprint-based labels or scheduling** should be treated as **approximate** unless camera geometry is **independently** recovered (survey, intrinsics, or dedicated calibration flight).

---

## 8. Reproduction instructions

**Environment:** Python 3 with `numpy`, `opencv-python`, `tifffile`, `Pillow`, `pyproj`, `shapely`, `matplotlib` (2C/2D).

| Stage | Script | Main outputs |
|---|---|---|
| 2A | `scripts/audit/sample_audit.py` | `reports/sample_audit.md`, `dataset/manifest.csv` (sample), `figures/sample_audit/` |
| 2B | `scripts/audit/full_dataset_audit.py` | `reports/full_dataset_audit.md`, `dataset/manifest.csv` (6774), `reports/flight_stats.csv` |
| 2C | `scripts/audit/stage2c_spatial_audit.py` | `figures/regions_and_tracks.png`, `reports/independent_region_groups.csv`, `reports/region_independence.md` |
| 2D | `scripts/audit/stage2d_footprint_validation.py` | `reports/footprint_validation.md`, `figures/region03_*` |

Mirrors under `audit_scripts/`. Literature artifacts: `literature/comparison.csv`, `literature/evidence_ledger.csv`, `reports/gap_decision.md`.

**Data paths:** Full release `F:\CMBL\datasets\raw\UAV_VisLoc_dataset\`; sample mirror `datasets/sample/`.

---

## 9. Evidence-backed decision table

| Decision domain | Verdict | Primary evidence |
|---|---|---|
| **Literature gap** | **GO** (provisional) | `reports/gap_decision.md`, `literature/comparison.csv`; MLSNet unread |
| **Region independence (holdout design)** | **GO** | `reports/region_independence.md`; 10 strict / 7 at 5 km |
| **CSV coordinate centres (map overlay)** | **Usable** | 2B georef Δ=0; 2D all 10 frames on-map |
| **Precise footprint ground truth** | **APPROXIMATE** | 2D: no polygons, empirical scale, NCC not geodetic error |
| **Overall dataset for proposed study** | **GO WITH LIMITATIONS** | 6774 consistent release; geometry/metadata gaps above |

---

## Final decisions (explicit)

| Question | Answer |
|---|---|
| Literature | **GO** (provisional; MLSNet unread) |
| Region independence | **GO** (not LOW-POWER) |
| Coordinate centres | **Usable** (not sub-metre validated) |
| Precise footprint GT | **APPROXIMATE** |
| Overall dataset | **GO WITH LIMITATIONS** |

**Recommended experimental scope (next stage):**

- Use **GitHub UAV-VisLoc** (6,774 frames, 11 regions) with **CSV lat/lon** as pose labels for map-centred localization / success prediction.
- Regional holdouts: default **merge 08+09**; sensitivity under **10-group** (overlap-only) and **7-group** (5 km) partitions.
- Frozen matcher + real UAV outcomes on held-out regions; calibration metrics (ECE/Brier etc.) and scheduling evaluation per proposal — **without** claiming validated footprint polygons.
- Optional: recover camera geometry in a **separate** calibration sub-study before any footprint-GT claims.

**Claims that must NOT appear in the paper (without new evidence):**

1. MLSNet or any prior **fully** closes the four-criterion gap (MLSNet unread).
2. README **6,742** is the actual image count (use **6,774**).
3. Stage 2D **~97 m** shift is **CSV geolocation error** or surveyed accuracy.
4. **Validated** ground-truth **footprint polygons** from release metadata alone.
5. Phi1/Omega/Kappa are **proven** compass vehicle attitudes with resolved height datum.
6. **Visual independence** of holdout regions from map distance/overlap rules alone.
7. Zenodo 14,998-image release is the **same** experiment as the GitHub extract used here.
8. Empty `unusable_frames.csv` means **all** frames have precise ground truth.

---

*End of Stage 1–2 consolidation.*
