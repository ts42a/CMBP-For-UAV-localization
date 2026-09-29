# Audit scripts

Stage 1–2 dataset and ground-truth audits, plus the label-free audits added later. None of them
reads a localizer outcome.

| script | purpose |
|---|---|
| `download_full_release.py`, `resume_drive_range.py`, `extract_full_release.py` | fetch and unpack the UAV-VisLoc full release into `datasets/raw/` |
| `sample_audit.py`, `sample_audit_fix.py`, `full_dataset_audit.py` | file, CSV and GeoTIFF audits → `datasets/manifests/`, `reports/*audit*` |
| `stage2b_pipeline.py`, `stage2b_blocked_outputs.py`, `stage2c_spatial_audit.py` | georeferencing, map overlap and independent region groups |
| `stage2d_footprint_validation.py`, `make_sat_preview.py` | region-03 footprint and ground-truth checks |
| `parse_arxiv_table2.py` | extract the dataset paper's Table 2 (needs the local `sources/` copy) |
| `agl_height.py` | height above ground per frame from SRTM terrain |
| `frame_quality.py` | label-free frame-quality flags |
