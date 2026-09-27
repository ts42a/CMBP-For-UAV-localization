# Stage 2B — Full UAV-VisLoc release audit (final)

**Audit date:** 2026-09-23  
**Extract root:** `F:\CMBL\datasets\raw\UAV_VisLoc_dataset`  
**Stage-1 caveat:** Provisional GO (MLSNet unread). **Geometry not validated.**

---

## 1. Archive and extract location

| Item | Finding |
|---|---|
| Official zip `datasets/raw/UAV_VisLoc_dataset.zip` | **Not present** (manual extract only; prior `.part` retained separately) |
| Extracted tree | `datasets/raw/UAV_VisLoc_dataset/` — folders **01–11**, root `satellite_ coordinates_range.csv`, `README_dataset.txt` |
| Re-download / re-extract | **Not performed** (per instruction) |
| Archive SHA256 | **Not computed** (zip absent) |

---

## 2. Release identity (not Zenodo)

| Check | Result |
|---|---|
| Layout | `NN/drone/`, `NN/NN.csv`, `satelliteNN.tif` (GitHub README schema) |
| Zenodo markers (`UAV-image/`, `satellite-tile/`) | **0** |
| Region folders | **11** |
| Range CSV rows | **11** (site labels match paper TABLE II, e.g. Changjiang-20, Taizhou-1) |
| Classification | **`github_intellisensing`** |

---

## 3. Counts and cross-output totals

| Metric | Documented | Actual | Cross-check |
|---|---:|---:|---|
| UAV JPG | 6,742 | **6,774** | = `manifest.csv` rows = sum `flight_stats.n_jpg` = `full_file_inventory` `.jpg` entries |
| Map / region folders | 11 | **11** | = `region_stats` rows = `flight_map_relationships` rows |
| Range CSV rows | 11 | **11** | |
| Satellite GeoTIFFs | 11 (canonical) | **14 files** | 09 split into 4 tiles |
| Manifest rows | — | **6,774** | matches JPG count |

**Documented vs actual images:** **+32** frames (`investigate` — all have CSV rows and files; README count may be stale).

**CSV ↔ image (all regions):** **0** missing images, **0** orphans, **0** duplicate filenames, **0** invalid coord rows; all `join_status=ok_1to1`.

---

## 4. Region 03 vs audited sample

| Item | Match |
|---|---|
| `03.csv` | SHA256 **identical** sample vs full |
| `03_0001.JPG`, `03_0385.JPG`, `03_0768.JPG` | SHA256 **identical** |
| `satellite03.tif` | Size **identical** (2,577,636,150 B) |

See `reports/sample_vs_full_region03.csv`.

---

## 5. Satellite rasters

- **13** regions: single `satelliteNN.tif` each — **EPSG:4326**, PixelIsArea, **embedded_geotiff**, bounds vs range CSV **Δ=0** (within float tolerance).
- **Region 09:** **4** tiles (`satellite09_01-01` … `02-02`); canonical `satellite09.tif` **absent**; tile bounds **do not** match full-map range row (expected for chips) — `investigate`.
- Details: `dataset/satellite_metadata.csv` (14 rows).

---

## 6. Flight / region statistics

- Per-region height (column `height`, reference **unresolved**), Omega/Kappa/Phi1/Phi2 (raw stats only), lat/lon extents, temporal spacing, haversine ground spacing: `reports/flight_stats.csv`, `reports/region_stats.csv`.
- Platform/camera: **unknown** (no per-region EXIF audit in this pass; Stage 2A: no EXIF in sample).

---

## 7. Issues (`reports/full_frame_issues.csv`)

| Severity | Count | Summary |
|---|---:|---|
| investigate | 6 | +32 vs 6742 doc count; region 09 multi-tile + 4× bounds vs range CSV |
| definite | 0 | No missing/orphan/corrupt join failures |

**Unresolved (unchanged unless new evidence):** height datum, Phi/Omega semantics vs paper Phi, intrinsics, timezone, satellite acquisition dates not in release CSVs.

**Geometry documentation:** `sources/UAV-VisLoc_geometry_documentation.md`  
**Capture dates:** `reports/capture_date_comparison.csv` (CSV flight dates + paper TABLE II months).

---

## 8. Disk space

F: **≈ 36.0 GB** free after audit (`38625972224` B).

---

## Stage 2C gate

### **READY** for Stage 2C (spatial / track mapping and later grouping prep)

**Rationale:** Full GitHub-layout release is present, internally consistent (6,774 frames with 1:1 metadata, 11 regions, georef verified on canonical maps), and sample region 03 matches the prior audit.

**Carry-forward limitations (not blockers for 2C inventory/spatial prep):**
1. README documents **6,742** images; release contains **6,774** (+32).
2. No complete **zip** on disk → no publisher/local archive SHA256.
3. Region **09** uses **four** satellite tiles instead of one `satellite09.tif`.
4. Unresolved height / angles / intrinsics / timezone / satellite CSV dates (see §7).

**Not started (per scope):** footprint validation, independent-region grouping, model training, Zenodo analysis.

---

## Deliverables

| Path | Rows / status |
|---|---|
| `dataset/manifest.csv` | 6,774 |
| `dataset/full_file_inventory.csv` | 6,801 files |
| `dataset/flight_map_relationships.csv` | 11 |
| `dataset/satellite_metadata.csv` | 14 |
| `dataset/data_dictionary.csv` | updated |
| `reports/flight_stats.csv` | 11 |
| `reports/region_stats.csv` | 11 |
| `reports/full_frame_issues.csv` | 6 |
| `reports/capture_date_comparison.csv` | 11 |
| `reports/sample_vs_full_region03.csv` | 5 |
| `reports/full_audit_summary.json` | machine summary |

Mirrors under `datasets/manifests/`. Script: `scripts/audit/full_dataset_audit.py`.
