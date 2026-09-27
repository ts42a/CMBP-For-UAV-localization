# Source Inventory

Updated: 2026-09-23 (Stages 1–2 consolidated). Stage 1 entries preserved.

## Local files (key)

| Local path | Role | Notes |
|---|---|---|
| `sources/Calibrated_Map-Based_Prediction_of_UAV_Localization_Success_in_Unseen_Regions.pdf` | Proposal | SHA256 `1BD96DEFCDB6E155…` |
| `sources/RSPNet_Li2023_rs15174353_webfetch.txt` | RSPNet full text (HTML mirror) | Access 2026-09-22; MDPI PDF blocked |
| `sources/Jiang2024_rs16010161_webfetch.txt` | Jiang 2024 full text (HTML mirror) | Access 2026-09-22 |
| `sources/RobustBeliefRoadmap_1304.7256.pdf` / `.html` | RBRM full text | arXiv:1304.7256 |
| `sources/SafeVPR_2605.28048.pdf` / `.html` | SafeVPR full text | arXiv:2605.28048 |
| `sources/SASGeo_2607.07737.pdf` / `.html` | SASGeo full text | arXiv:2607.07737 |
| `sources/MLSNet_SciOpen.html` | MLSNet landing | Abstract only in practice |
| `sources/MLSNet_openalex.json` | OA status | `has_fulltext=false`, `pdf_url=null` |
| `sources/MLSNet_full_text_api.json` | SciOpen API | Empty `contentList` |
| `sources/ACCESS_LOG_2026-09-22.txt` | Access attempts | PDF 403/502 notes |
| `sources/WHU_data_page.html` | WHU dataset listing | WHU-SCENE-MATCHING not found by name |
| `sources/search_*.txt` | Search candidate extracts | VPR uncertainty; VPR integrity+nav; LiDAR localizability |
| `literature/comparison.csv` | Stage-1 comparison table | |
| `literature/evidence_ledger.csv` | Evidence ledger | |
| `literature/search_log_localizability_2026-09-22.md` | Search log | |
| `literature/notes/*.md` | Per-paper notes | |
| `reports/literature_review.md` | Narrative review | |
| `reports/gap_decision.md` | GO / NARROW / UNRESOLVED | **GO** (provisional: MLSNet unread) |
| `reports/release_provenance.md` | GitHub vs Zenodo release evidence | Stage 2A |
| `reports/sample_audit.md` | Sample audit + gate decision | **PROCEED WITH LIMITATIONS** |
| `reports/full_dataset_audit.md` | Full release audit | 6,774 JPG; **READY** for 2C+ |
| `reports/stage1_stage2_final.md` | Stages 1–2 consolidation | **GO WITH LIMITATIONS** dataset |
| `reports/stage1_stage2_decision.json` | Consolidated gate JSON | Literature GO provisional |
| `reports/footprint_validation.md` | Stage 2D region-03 validation | **APPROXIMATE** GT gate |
| `reports/stage2d_summary.json` | Stage 2D machine summary | NCC + global geometry |
| `reports/unusable_frames.csv` | Stage 2D unusable frames | Reason codes |
| `figures/region03_footprints_overview.png` | Stage 2D track + 10 frames | |
| `figures/region03_validation/` | Per-frame sat/drone overlays | Fixed global geometry |
| `reports/flight_stats.csv` / `region_stats.csv` | Per-region stats | Region **03** only until full extract |
| `datasets/manifests/full_file_inventory.csv` | raw + sample inventory | Pending full tree |
| `sources/UAV-VisLoc_geometry_documentation.md` | Height/Phi/intrinsics evidence | arXiv + README |
| `sources/UAV-VisLoc_arxiv_table2_extract.txt` | Paper TABLE II | Capture dates per site |
| `reports/sample_audit_raw_summary.json` | Machine summary from audit scripts | |
| `reports/sample_frame_issues.csv` | Definite / unresolved issues | |
| `reports/hardware_snapshot.json` | Hardware (setup) | F: free rechecked in sample audit (~69.5 GB after sample) |
| `datasets/manifests/manifest.csv` | Per-frame sample manifest | |
| `datasets/manifests/sample_file_inventory.csv` | Relative path inventory | |
| `datasets/manifests/data_dictionary.csv` | Column dictionary | |
| `datasets/raw/UAV-VisLoc_sample_gdrive.zip` | Official sample archive | SHA256 `442F93AE…` (local integrity) |
| `datasets/raw/UAV-VisLoc_sample_download_record.txt` | Download provenance | Drive ID `16tY7tPZiNIoyAhknvyXnp0jAfccIcHtL` |
| `datasets/sample/` | Extracted sample (region 03) | 768 JPG + GeoTIFF + CSVs |
| `figures/sample_audit/` | Drone + satellite tile previews | Not footprint overlays |
| `scripts/audit/` | Audit scripts (2A–2D) | `audit_scripts/` (old duplicate) removed 2026-09-24 |

## Named paper status (Stage 1)

### RSPNet (Li et al. 2023)
- **Full text:** reviewed (HTML mirror).
- **Code:** not reported in reviewed MDPI end matter; **public repository not found** for matchability RSPNet.
- **WHU-SCENE-MATCHING:** described in paper; **public download path not found** on WHU data page (2026-09-22). Distinct from PeihaoChen/RSPNet (video).

### Jiang et al. 2024
- **Full text:** reviewed.
- **Data:** “Data sharing is not applicable.”
- **Code:** not reported.

### Robust Belief Roadmap
- **Full text:** reviewed (arXiv PDF+HTML).

### SafeVPR
- **Full text:** reviewed.
- **Code:** https://github.com/Hasar12139/SafeVPR (URL verified in PDF; clone not audited).

### MLSNet
- **Full text:** **unavailable** after SciOpen PDF 502, ScienceDirect failure, OpenAlex no repository PDF.
- **Do not** assert absence of calibration/held-out results.
- Unrelated genomics MLSNet GitHub excluded.

### SASGeo (not in proposal Sources)
- **Full text:** reviewed.
- **Placement:** adjacent literature (see `gap_decision.md` / notes).
- **Code URL:** https://gitlab.com/emilab-group/sasgeo (HTTP 200; contents not audited).

### UAV-VisLoc
- **Selected release:** GitHub IntelliSensing — **6,742** images / **11** maps / **16.4 GB** full / **2.04 GB** sample (proposal-linked).
- **Sample downloaded:** yes — region `03` only; see `reports/release_provenance.md`, `reports/sample_audit.md`.
- **Full release:** Extracted at `datasets/raw/UAV_VisLoc_dataset/` (11 regions, 6,774 JPG). Official zip **not** in `datasets/raw/` (manual extract). Old `.part` may remain.
- **Stage 2B audit:** `reports/full_dataset_audit.md`
- **Stage 2C:** `reports/region_independence.md`, `figures/regions_and_tracks.png`, pairwise/group CSVs — **GO**
- **Stage 2D:** `reports/footprint_validation.md` — CSV centres on-map; empirical scale/rotation; NCC shift **not** proven coordinate error → **APPROXIMATE** footprint GT
- **Consolidation:** `reports/stage1_stage2_final.md` — literature **GO** (provisional); independence **GO**; dataset **GO WITH LIMITATIONS**; hold **08+09** as one unit; 10- vs 7-group sensitivity
- **Zenodo** DOI 10.5281/zenodo.19603534 (**14,998** images): related but **distinct**; kept separate; **not** downloaded.
- **License (GitHub):** not stated on repo (`license: null`). Zenodo record: Academic Research Use Only (does not govern the GitHub sample used here).

## Hardware (updated free-space note)
Dell XPS 9320; ~15.6 GB RAM; Iris Xe; prefer **F:** for dataset I/O.  
Setup snapshot: F: ~74.7 GB free. **After sample archive+extract (2026-09-22):** F: ≈ **69.5 GB** free. Full-release next step needs fresh ≥50 GB free plus archive/extract capacity.
