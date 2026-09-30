# Calibrated map-based prediction of UAV localization success in unseen regions

Can satellite-map features predict whether a fixed, zero-shot visual localizer will succeed on real
UAV imagery in regions it has never seen, do those probabilities stay calibrated across regions,
and does scheduling localization attempts with them keep a UAV localized better along a route?

The study uses the [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) dataset (11 flights,
one satellite map each), a frozen DINOv2 retrieval + patch-verification localizer (nothing trained
on UAV-VisLoc), real localization outcomes as labels, and leave-one-region-group-out evaluation.
Every threshold and decision rule is fixed in a pre-registration before the data it applies to is
generated; every later change is logged as a dated deviation.

| Question | Evidence |
|---|---|
| RQ1 — do map features predict success in unseen regions? | real localizer outcomes, geographic hold-outs, simple baselines |
| RQ2 — do the probabilities stay calibrated across regions? | calibration fitted on source regions, reliability on untouched test regions |
| RQ3 — do they improve localization-update scheduling? | paired recorded-route replay at matched attempt budgets |

## Status

Pilot scale only; no full-scale result exists yet.

| Stage | State |
|---|---|
| 1–2 Literature gap, dataset and ground-truth audit | done (`reports/gap_decision.md`, `reports/full_dataset_audit.md`) |
| 3–4 Splits, pre-registration, localizer freeze | done (`configs/splits_v1.json`, `configs/preregistration_v1.yaml`, deviation D1) |
| 5 Multi-region pilot labels (110 frames × 9 attempts) | done, gate GO (`reports/stage5_pilot_decision.md`) |
| 6–8 Predictors and calibration, pilot function check | done (`reports/stage6_8_pilot_v3_summary.md`); protocol v4 in force, v5 proposed |
| 10 Kaggle GPU benchmark | kit ready and tested locally (`kaggle/README.md`); GPU run pending |
| 9 Route replay, H1 synthetic labels | protocols proposed (`configs/stage9_protocol_v1.yaml`, `configs/h1_protocol_v1.yaml`) |
| 11–12 Full-scale runs, write-up | not started |

## Repository layout

```
configs/      frozen protocols: pre-registration + deviations D1-D6, splits, matcher, predictor v1-v5, H1, Stage 9, Stage 10
scripts/
  label_windowed.py       frozen windowed localizer -> outcome rows (resumable, order-independent seeding)
  localization/           retrieval, tiling, DINOv2 patch verifier, geo utilities
  predictor/              map features, predictors + calibration evaluator, label-noise bound
  protocol/               splits check, verify, Kaggle packaging/driver, Stage 10 compare, cost projection, archiving
  analysis/, audit/       summaries and label-free audits (AGL, frame quality, track offsets, dataset audits)
  h1/                     synthetic-query generator for H1
kaggle/       Kaggle kernel + runner for the Stage 10 benchmark
datasets/manifests/       frame and satellite manifests, per-frame height above ground and quality flags
reports/      one report per decision or run (Markdown + JSON)
results/      pilot-scale outputs (full-scale outputs are archived outside git)
figures/      pilot figures and visual-review panels
literature/   literature notes, comparison table, evidence ledger
```

## Setup

Python 3.14. One project environment with pinned versions (exactly one OpenCV package):

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements-localization.txt
# Linux / macOS
.venv/bin/python -m pip install -r requirements-localization.txt
```

Run every command below with that interpreter.

### Data (not in this repository)

UAV-VisLoc states no licence, so it is never committed. Download the full release from the
dataset's page (`scripts/audit/download_full_release.py` fetches the Google Drive archive with
resume) and unpack it so that the region folders sit here:

```
datasets/raw/UAV_VisLoc_dataset/01 ... 11/   # each: drone/*.JPG, <RR>.csv, satellite<RR>.tif
```

Its satellite maps come from Google Earth Pro, so Google Earth/Maps terms and attribution apply to any
crop of them. Nothing derived from the dataset that could substitute for it is published here:

- figures showing dataset imagery (drone frames, satellite crops, visual-review panels);
- per-frame metadata: the manifests in `datasets/manifests/*.csv`, and every table carrying frame
  positions, heights, attitudes or map bounds (e.g. `results/**/outcomes.csv`, `features.csv`, tile
  embeddings, some Stage 1-3 reports).

Only code, protocols, charts, aggregate reports and model outputs keyed by file name are published.
After downloading the dataset, rebuild the local metadata:

```bash
python scripts/audit/full_dataset_audit.py   # manifest.csv, satellite_metadata.csv, inventories (also rewrites the Stage 2 audit reports)
python scripts/audit/agl_height.py           # frame_agl.csv (downloads SRTM tiles)
python scripts/audit/frame_quality.py        # frame_quality.csv
```

A rebuilt `manifest.csv` stores your own absolute image paths, so its sha256 differs from the one
recorded in `configs/splits_v1.json`; the split assignment itself does not depend on it.

### Caches (rebuilt locally, git-ignored)

- DINOv2 ViT-S/14 weights: downloaded by `torch.hub` on first use (internet needed once).
- Satellite tile embeddings: `python scripts/extract_all_reference_embeddings.py` → `datasets/cache/embeddings/`.
- SRTM terrain for height above ground: `python scripts/audit/agl_height.py` downloads the public tiles it needs.

## Check the pipeline

```bash
python scripts/protocol/verify.py            # 7 checks, ~10 min on a CPU
python scripts/protocol/verify.py --quick    # skips the three slow checks
```

`verify` confirms the pinned environment, that frozen files still match the hashes the committed
results were produced with, the splits, byte-for-byte reproduction of the predictor pilot, the
protocol-v4/v5 rules, and a 2-frame golden reproduction of the localizer labels. Run it before any
freeze or full run. Checks whose inputs are local-only (the dataset and the metadata derived from it)
report SKIP with the missing files listed; on a fresh public clone only `env` can run until you
have the dataset and have rebuilt the pilot outputs.

## Main commands

```bash
# localizer outcomes for the pilot set (resumable)
python scripts/label_windowed.py --set pilot --out results/stage5/pilot --resume
# the same attempts with H1 synthetic queries (proposed protocol)
python scripts/label_windowed.py --set pilot --query-source synthetic --out <dir> --resume
# map-only features and predictor evaluation
python scripts/predictor/map_features.py --outcomes results/stage5/pilot/outcomes.csv --out results/stage6_7/pilot
python scripts/predictor/evaluate_predictors.py --protocol configs/predictor_protocol_v3.yaml \
    --features results/stage6_7/pilot --out results/stage8/pilot_v3 --name stage6_8_pilot_v3
# Stage 10 on Kaggle: see kaggle/README.md
```

Full-scale outputs go under `results/full/` or `figures/full/` and are archived with
`scripts/protocol/archive_outputs.py`, which commits only a checksum manifest.

## Not in this repository

- the dataset, caches and build outputs (see `.gitignore`);
- figures that show dataset imagery, and per-frame dataset metadata (see Data above);
- third-party papers and web pages collected for the literature review (`sources/`, copyright);
  `literature/` keeps the notes and the evidence ledger that cite them;
- internal planning and progress notes.

## Licence

Not yet specified.
