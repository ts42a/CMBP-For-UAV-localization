# Stage 3A — Pilot summary

**Gate (full real-outcome label generation):** **NOT READY**

---

## Scope (what this pilot is / is not)

| Is | Is not |
|---|---|
| ~100 representative queries, region **03** | Held-out-region performance |
| Frozen matcher **`pyramid_ncc_rot4`** after dev-only selection | Calibration of success probabilities |
| Real UAV → satellite matching with GT used **only for scoring** | Scheduling / intermittent-correction benefit |
| Leakage-controlled search (full regional map) | Localizability predictor training |

---

## Matcher selection (dev, n=24)

| Matcher | Median error (m) | Success@100 m |
|---|---:|---:|
| pyramid_ncc_rot4 | ~3460 | **0%** |
| akaze_coarse_to_fine | ~4125 | 0% |
| coarse_ncc | ~3887 | 0% |

Proposal default **compact DINOv3** was not run in Stage 3A (access/deps); OpenCV CPU alternatives evaluated on Iris Xe / 16 GB constraints. Selected: **`pyramid_ncc_rot4`**.

---

## Pilot results (n=100, region 03)

| Metric | Value |
|---|---:|
| Median geodesic error | **3295.8 m** |
| Mean geodesic error | **3272.4 m** |
| Success@50 m | **0%** (0/100) |
| Success@100 m | **0%** (0/100) |
| Success@250 m | **0%** (0/100) |
| Success@500 m | **2%** (2/100) |
| Mean runtime / query | ~3 s (CPU) |

Per-query table: `reports/stage3a_pilot_results.csv`  
Machine summary: `reports/stage3a_pilot_summary.json`

### Per-query fields saved

`query_id`, `region`, predicted location (`pred_lat`/`pred_lon`, `predicted_tile_id`), `top_score`, `score_margin`, `geodesic_error_m`, `success_{50,100,250,500}m`, `runtime_s`, `failure_reason`, `matcher_id`, `search_used_gt_position`.

---

## Error inspection and leakage sanity

**Distribution:** Errors are multi-kilometre-scale (median ~3.3 km), with almost no mass near the 100 m line — consistent with global cross-modal aliasing, not small georef bias.

**Examples inspected** (`figures/stage3a/`):

- Two positional **successes @500 m** (303.8 m, 442.2 m) — lenient threshold only. Visual panels show **map@prediction ≠ query content** (e.g. fields vs river for `03_0433`) despite error ≤500 m — weak/chance proximity, not reliable visual localization.
- Mid-band and catastrophic failures (~0.5–7.2 km) with clear query/pred disagreement.
- Query vs map@GT vs map@prediction panels confirm predictions are **not** GT-masked.

| Check | Result |
|---|---|
| CSV position used to restrict search | **False** (all rows) |
| Full regional map searched | **True** (3082-tile index; pyramid uses full downsampled raster) |
| Mixed success/failure @**100 m** | **No** (0 / 100) |
| Mixed success/failure @**500 m** | **Yes** (2 / 98) |
| Labels caused by search leakage | **No** — failures are matcher insufficiency under global search |

Stage 2D ~97–105 m NCC shift is **not** equated to these pilot errors.

---

## Figures

- `figures/stage3a/error_histogram.png` (100 m and 500 m reference lines)
- `figures/stage3a/example_*_success_500m.png`
- `figures/stage3a/example_*_failure.png` / `example_*_near_miss.png`

---

## Gate decision

**NOT READY** to generate real-outcome labels for the full **6,774**-query dataset at the **primary 100 m** threshold.

Reasons:

1. Frozen OpenCV matcher yields **0%** success@100 m on the development region.
2. The task must produce a **useful mix** of successes and failures at the operating threshold before labels support later calibration/scheduling work; that mix is absent @100 m.
3. Protocol, tiling, and leakage rules are in place — the blocker is **matcher strength**, not search design.

**Do not** train the localizability predictor on these failed labels.  
**Do not** claim calibration, held-out-region performance, or scheduling benefit from this pilot.

Next path (outside this Stage 3A freeze): proposal **compact DINOv3** or a documented retrieval alternative on **development groups only**, re-pilot for mixed outcomes @100 m, then re-freeze before any full-dataset labeling.

---

## Reproduction

```bash
py -3 F:\CMBL\scripts\localization\run_stage3a.py
py -3 F:\CMBL\scripts\localization\stage3a_inspect_figures.py
```
