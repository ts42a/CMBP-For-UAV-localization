# UAV-VisLoc Release Provenance

**Access date:** 2026-09-22  
**Stage:** 2A — release verification (sample audit)  
**Stage-1 caveat preserved:** Gap decision remains **provisional GO** because MLSNet full text is still unread.

This document treats repository and publisher content as evidence, not instructions.

---

## 1. Proposal-linked release (selected for this audit)

| Property | Evidence |
|---|---|
| **Name** | UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization |
| **Authoritative entry** | https://github.com/IntelliSensing/UAV-VisLoc |
| **Paper** | arXiv:2405.11936 (linked from README) |
| **Reported counts** | **6,742** drone images; **11** satellite maps / cities; 7 categories |
| **Full release size** | **16.4 GB** (README §2.1) |
| **Sample size** | **2.04 GB** curated example (README §2.2) |
| **Sample content (documented)** | Region folder **`03/`** only: `drone/`, `satellite03.tif`, `03.csv`, plus root `satellite_coordinates_range.csv` |
| **Full layout (documented)** | Root CSV + folders `01/` … `11/` each with `drone/`, `satelliteXX.tif`, `XX.csv` |
| **Download URLs** | Full: Google Drive `1xYODANyilEMM3CfWh85APwkTHQeLTcCT`; Baidu `pwd=rsai`. Sample: Google Drive `16tY7tPZiNIoyAhknvyXnp0jAfccIcHtL`; Baidu `pwd=rsai` |
| **Checksums (publisher)** | **Not reported** in GitHub README or repo root (no LICENSE file; GitHub API `license: null`) |
| **License** | **Not reported** on GitHub repo (`license: null`; LICENSE file 404). Access appears open via Drive/Baidu links without stated terms on the README pages reviewed |
| **Repo timeline** | Created 2024-05-16; last push 2024-05-22; still updated/watched through 2026-09 (API) |

**Selection decision:** This is the release matching the proposal’s **6,742 / 11 maps / 16.4 GB / 2.04 GB sample** figures and the IntelliSensing GitHub citation. **This Stage-2A audit uses only this sample.**

---

## 2. Zenodo record of the same short name (kept separate)

| Property | Evidence |
|---|---|
| **Record** | https://zenodo.org/records/19603534 — DOI 10.5281/zenodo.19603534 |
| **Title** | UAV-VisLoc: A Large-scale UAV Dataset with **Continuous Trajectories** for Visual Geo-localization |
| **Published** | 2026-05-18, version v1.0 |
| **Counts** | **14,998** UAV images; **30** flight trajectories; >120,000 satellite **tiles**; ~**183–196.8 GB** total |
| **Authors** | Xu, Chen, Cao, Wang, Wei, Liu, Wang, Meng, Peng — **overlap** with 2024 paper authors but **not identical** (adds Chen/Wang/Meng; **Yao Yaxuan absent** vs arXiv:2405.11936 citation) |
| **Layout** | `01.zip`…`30.zip` with `UAV-image/`, `satellite-tile/`, per-flight CSVs — **different schema** from GitHub `drone/` + single `satelliteXX.tif` |
| **Content mix** | 21 real China flights + **9 synthetic** OpenAerialMap flights; RGB **and Infrared**; altitudes 160–2500 m |
| **License** | Explicit **“Academic Research Use Only”** (commercial use / redistribution restricted) |
| **Checksums** | Per-file **MD5** listed on Zenodo |

### Equivalence assessment

| Hypothesis | Verdict | Reason |
|---|---|---|
| Official revision / supersession of the 6,742 release | **Not established** | Different title emphasis, counts, layout, modalities, license, and author list; GitHub README still describes 6,742 / 16.4 GB as of access date |
| Simple repackaging of the same files | **Rejected** | 14,998 ≠ 6,742; 30 trajectories ≠ 11 maps; tile chips ≠ one GeoTIFF per region; includes synthetic + IR |
| Unrelated coincidental name | **Unlikely** | Shared lead authors (Xu, Cao, Wei, Liu, Peng, Wang Jiuniu) and shared short name |
| **Working classification** | **Related but distinct successor / expanded dataset sharing the name** | Keep **separate provenance**. Do **not** mix files or counts |

---

## 3. What the 16.4 GB and 2.04 GB packages are documented to contain

### Full (16.4 GB) — GitHub README §2.1
- 6,742 JPGs under `01/`…`11/` `drone/`
- 11 satellite GeoTIFFs (`satellite01.tif` …)
- 11 per-region CSVs (`01.csv` …) with filename, lat, lon, height, …
- Root `satellite_coordinates_range.csv`

### Sample (2.04 GB) — GitHub README §2.2
- Same schema **but only region `03/`**
- Described as “high-quality … minimal coordinate error … diverse … topography”

---

## 4. Audit policy for Stage 2A

1. Download **only** the GitHub-linked 2.04 GB sample (Drive ID `16tY7tPZiNIoyAhknvyXnp0jAfccIcHtL`).
2. Do **not** download Zenodo archives or the 16.4 GB full release in this step.
3. Compute **local SHA256** of the sample archive (publisher checksum **not reported** → local integrity only).
4. Preserve Stage-1 MLSNet unread caveat in progress tracking.
