# Dataset manifests

Derived, label-free tables about the UAV-VisLoc release. **The CSVs are not in the public
repository** (git-ignored): they copy the dataset's per-frame annotations, which are not cleared for
redistribution. Rebuild them locally from the dataset with the scripts below (top-level README, Data).

| file | contents | made by |
|---|---|---|
| `manifest.csv` | one row per drone frame: region, file, CSV position, height (above sea level), attitude, date | Stage 2 audit (`scripts/audit/full_dataset_audit.py`) |
| `satellite_metadata.csv` | one row per satellite GeoTIFF: size, CRS, bounds, pixel scale | Stage 2 audit |
| `flight_map_relationships.csv` | flight ↔ map pairing | Stage 2 audit |
| `data_dictionary.csv` | column meanings for the manifests | Stage 2 audit |
| `full_file_inventory.csv`, `sample_file_inventory.csv` | file listings of the full release and the curated sample | Stage 2 audit |
| `frame_agl.csv` | height above ground per frame = CSV height − SRTM terrain | `scripts/audit/agl_height.py` |
| `frame_quality.csv` | label-free frame-quality flags (track jumps, speed outliers, low AGL, turns) | `scripts/audit/frame_quality.py` |

`manifest.csv` stores the absolute image paths of the machine that built it; the labeller falls back
to `datasets/raw/UAV_VisLoc_dataset/<region>/drone/<file>` when that path does not exist.
