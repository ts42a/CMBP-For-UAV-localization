# stage6_8_pilot_v2 — Stages 6-8 predictor summary

**Function check on the Stage 5 pilot labels (10 frames per region), not a geographic result.** Protocol `configs/predictor_protocol_v2.yaml` (sha256 `0318a65bf89b…`), fixed hyperparameters, region 07 excluded (deviation D2). Rows 900, prevalence 0.491.

Variants: `raw` (uncalibrated), `cal_v2_fe` (primary: fixed-effects Platt, slope from within-group variation), `cal_v1_registered` (the v1 pooled Platt/isotonic, kept for comparison).

## Headline: macro over held-out groups

| model | variant | macro AUROC | macro AUROC excl. G2 | macro PR-AUC | macro Brier | macro log loss | macro ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| M0_constant | raw | 0.500 | 0.500 | 0.510 | 0.285 | 0.763 | 0.354 |
| M1_texture | raw | 0.552 | 0.536 | 0.554 | 0.320 | 0.875 | 0.388 |
| M1_texture | cal_v1_registered | 0.445 | 0.461 | 0.513 | 0.301 | 0.802 | 0.370 |
| M1_texture | cal_v2_fe | 0.552 | 0.536 | 0.554 | 0.291 | 0.777 | 0.359 |
| M2_dino | raw | 0.651 | 0.658 | 0.624 | 0.287 | 0.940 | 0.331 |
| M2_dino | cal_v1_registered | 0.421 | 0.428 | 0.519 | 0.328 | 0.877 | 0.396 |
| M2_dino | cal_v2_fe | 0.651 | 0.658 | 0.624 | 0.283 | 0.779 | 0.357 |
| M3_dino_distinct | raw | 0.639 | 0.643 | 0.601 | 0.293 | 0.961 | 0.343 |
| M3_dino_distinct | cal_v1_registered | 0.423 | 0.431 | 0.518 | 0.329 | 0.878 | 0.396 |
| M3_dino_distinct | cal_v2_fe | 0.639 | 0.643 | 0.601 | 0.285 | 0.779 | 0.357 |

### H3 check: does source-fitted calibration improve held-out probability quality?

| model | macro Brier raw → fe (v1) | macro log loss raw → fe (v1) | fe better on both |
|---|---|---|---|
| M1_texture | 0.320 → 0.291 (0.301) | 0.875 → 0.777 (0.802) | yes |
| M2_dino | 0.287 → 0.283 (0.328) | 0.940 → 0.779 (0.877) | yes |
| M3_dino_distinct | 0.293 → 0.285 (0.329) | 0.961 → 0.779 (0.878) | yes |

## Pooled probability quality over all held-out rows (95% CI: region-group bootstrap)

| model | variant | Brier (CI) | log loss (CI) | ECE |
|---|---|---|---|---:|
| M0_constant | raw | 0.285 (0.262–0.315) | 0.763 (0.718–0.825) | 0.323 |
| M1_texture | raw | 0.317 (0.267–0.380) | 0.868 (0.746–1.022) | 0.264 |
| M1_texture | cal_v1_registered | 0.298 (0.272–0.327) | 0.799 (0.748–0.853) | 0.307 |
| M1_texture | cal_v2_fe | 0.289 (0.256–0.326) | 0.774 (0.706–0.849) | 0.243 |
| M2_dino | raw | 0.291 (0.182–0.418) | 0.940 (0.577–1.406) | 0.174 |
| M2_dino | cal_v1_registered | 0.318 (0.265–0.393) | 0.850 (0.723–1.040) | 0.313 |
| M2_dino | cal_v2_fe | 0.278 (0.232–0.344) | 0.764 (0.658–0.926) | 0.118 |
| M3_dino_distinct | raw | 0.295 (0.185–0.427) | 0.955 (0.578–1.453) | 0.192 |
| M3_dino_distinct | cal_v1_registered | 0.320 (0.266–0.399) | 0.852 (0.724–1.048) | 0.317 |
| M3_dino_distinct | cal_v2_fe | 0.279 (0.239–0.341) | 0.764 (0.674–0.911) | 0.115 |

## Secondary — pooled AUROC (unreliable under leave-one-group-out)

Each fold's predictions carry the other groups' success rate, which moves opposite to the held-out group's own rate, so pooled ranking mixes in a between-group artefact (the constant model scores well below 0.5). Read the macro AUROC above instead.

| model | variant | pooled AUROC (CI) |
|---|---|---|
| M0_constant | raw | 0.155 (0.070–0.327) |
| M1_texture | raw | 0.427 (0.271–0.590) |
| M1_texture | cal_v1_registered | 0.199 (0.091–0.408) |
| M1_texture | cal_v2_fe | 0.248 (0.130–0.519) |
| M2_dino | raw | 0.597 (0.389–0.827) |
| M2_dino | cal_v1_registered | 0.228 (0.076–0.529) |
| M2_dino | cal_v2_fe | 0.508 (0.324–0.701) |
| M3_dino_distinct | raw | 0.587 (0.377–0.804) |
| M3_dino_distinct | cal_v1_registered | 0.212 (0.072–0.492) |
| M3_dino_distinct | cal_v2_fe | 0.488 (0.299–0.684) |

Shuffle control (M1, labels permuted within training groups): macro AUROC 0.428, range over 20 permutations 0.396–0.463 (per fold: G1 0.508, G2 0.576, G3 0.454, G4 0.229, G5 —, G6 0.486, G7 0.315).

## Per held-out group

| fold | regions | model | variant | n | prev. | AUROC | PR-AUC | Brier | log loss | ECE | calibrator |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| G1 | 01,02 | M0_constant | raw | 180 | 0.38 | 0.500 | 0.378 | 0.255 | 0.703 | 0.142 |  |
| G1 | 01,02 | M1_texture | raw | 180 | 0.38 | 0.689 | 0.494 | 0.327 | 0.895 | 0.343 |  |
| G1 | 01,02 | M1_texture | cal_v1_registered | 180 | 0.38 | 0.689 | 0.494 | 0.255 | 0.704 | 0.143 | platt |
| G1 | 01,02 | M1_texture | cal_v2_fe | 180 | 0.38 | 0.689 | 0.494 | 0.256 | 0.706 | 0.160 | fe_platt slope=0.076 |
| G1 | 01,02 | M2_dino | raw | 180 | 0.38 | 0.663 | 0.486 | 0.347 | 1.100 | 0.359 |  |
| G1 | 01,02 | M2_dino | cal_v1_registered | 180 | 0.38 | 0.663 | 0.486 | 0.259 | 0.713 | 0.193 | platt |
| G1 | 01,02 | M2_dino | cal_v2_fe | 180 | 0.38 | 0.663 | 0.486 | 0.282 | 0.770 | 0.261 | fe_platt slope=0.355 |
| G1 | 01,02 | M3_dino_distinct | raw | 180 | 0.38 | 0.659 | 0.484 | 0.350 | 1.109 | 0.363 |  |
| G1 | 01,02 | M3_dino_distinct | cal_v1_registered | 180 | 0.38 | 0.659 | 0.484 | 0.257 | 0.707 | 0.175 | platt |
| G1 | 01,02 | M3_dino_distinct | cal_v2_fe | 180 | 0.38 | 0.659 | 0.484 | 0.280 | 0.765 | 0.255 | fe_platt slope=0.346 |
| G2 (dev) | 03,04 | M0_constant | raw | 180 | 0.88 | 0.500 | 0.883 | 0.343 | 0.883 | 0.490 |  |
| G2 (dev) | 03,04 | M1_texture | raw | 180 | 0.88 | 0.635 | 0.924 | 0.308 | 0.810 | 0.448 |  |
| G2 (dev) | 03,04 | M1_texture | cal_v1_registered | 180 | 0.88 | 0.365 | 0.842 | 0.346 | 0.889 | 0.490 | platt |
| G2 (dev) | 03,04 | M1_texture | cal_v2_fe | 180 | 0.88 | 0.635 | 0.924 | 0.346 | 0.889 | 0.493 | fe_platt slope=0.298 |
| G2 (dev) | 03,04 | M2_dino | raw | 180 | 0.88 | 0.615 | 0.934 | 0.215 | 0.628 | 0.289 |  |
| G2 (dev) | 03,04 | M2_dino | cal_v1_registered | 180 | 0.88 | 0.385 | 0.876 | 0.372 | 0.944 | 0.517 | platt |
| G2 (dev) | 03,04 | M2_dino | cal_v2_fe | 180 | 0.88 | 0.615 | 0.934 | 0.249 | 0.692 | 0.381 | fe_platt slope=0.271 |
| G2 (dev) | 03,04 | M3_dino_distinct | raw | 180 | 0.88 | 0.619 | 0.935 | 0.212 | 0.620 | 0.285 |  |
| G2 (dev) | 03,04 | M3_dino_distinct | cal_v1_registered | 180 | 0.88 | 0.381 | 0.876 | 0.380 | 0.961 | 0.525 | platt |
| G2 (dev) | 03,04 | M3_dino_distinct | cal_v2_fe | 180 | 0.88 | 0.619 | 0.935 | 0.254 | 0.701 | 0.388 | fe_platt slope=0.252 |
| G3 | 05 | M0_constant | raw | 90 | 0.27 | 0.500 | 0.267 | 0.258 | 0.709 | 0.249 |  |
| G3 | 05 | M1_texture | raw | 90 | 0.27 | 0.593 | 0.312 | 0.210 | 0.603 | 0.108 |  |
| G3 | 05 | M1_texture | cal_v1_registered | 90 | 0.27 | 0.407 | 0.223 | 0.284 | 0.762 | 0.288 | platt |
| G3 | 05 | M1_texture | cal_v2_fe | 90 | 0.27 | 0.593 | 0.312 | 0.241 | 0.676 | 0.220 | fe_platt slope=0.154 |
| G3 | 05 | M2_dino | raw | 90 | 0.27 | 0.824 | 0.545 | 0.153 | 0.431 | 0.084 |  |
| G3 | 05 | M2_dino | cal_v1_registered | 90 | 0.27 | 0.176 | 0.174 | 0.270 | 0.733 | 0.259 | platt |
| G3 | 05 | M2_dino | cal_v2_fe | 90 | 0.27 | 0.824 | 0.545 | 0.185 | 0.557 | 0.173 | fe_platt slope=0.211 |
| G3 | 05 | M3_dino_distinct | raw | 90 | 0.27 | 0.780 | 0.432 | 0.172 | 0.479 | 0.155 |  |
| G3 | 05 | M3_dino_distinct | cal_v1_registered | 90 | 0.27 | 0.220 | 0.182 | 0.278 | 0.749 | 0.268 | platt |
| G3 | 05 | M3_dino_distinct | cal_v2_fe | 90 | 0.27 | 0.780 | 0.432 | 0.199 | 0.587 | 0.179 | fe_platt slope=0.167 |
| G4 | 06 | M0_constant | raw | 90 | 0.06 | 0.500 | 0.056 | 0.287 | 0.767 | 0.484 |  |
| G4 | 06 | M1_texture | raw | 90 | 0.06 | 0.579 | 0.100 | 0.193 | 0.574 | 0.361 |  |
| G4 | 06 | M1_texture | cal_v1_registered | 90 | 0.06 | 0.421 | 0.065 | 0.336 | 0.868 | 0.530 | platt |
| G4 | 06 | M1_texture | cal_v2_fe | 90 | 0.06 | 0.579 | 0.100 | 0.255 | 0.704 | 0.450 | fe_platt slope=0.219 |
| G4 | 06 | M2_dino | raw | 90 | 0.06 | 0.751 | 0.246 | 0.116 | 0.380 | 0.209 |  |
| G4 | 06 | M2_dino | cal_v1_registered | 90 | 0.06 | 0.249 | 0.045 | 0.290 | 0.774 | 0.487 | platt |
| G4 | 06 | M2_dino | cal_v2_fe | 90 | 0.06 | 0.751 | 0.246 | 0.211 | 0.613 | 0.399 | fe_platt slope=0.306 |
| G4 | 06 | M3_dino_distinct | raw | 90 | 0.06 | 0.751 | 0.237 | 0.121 | 0.393 | 0.218 |  |
| G4 | 06 | M3_dino_distinct | cal_v1_registered | 90 | 0.06 | 0.249 | 0.046 | 0.294 | 0.781 | 0.491 | platt |
| G4 | 06 | M3_dino_distinct | cal_v2_fe | 90 | 0.06 | 0.751 | 0.237 | 0.222 | 0.636 | 0.413 | fe_platt slope=0.269 |
| G5 | 10 | M0_constant | raw | 90 | 0.00 | — | — | 0.298 | 0.789 | 0.546 |  |
| G5 | 10 | M1_texture | raw | 90 | 0.00 | — | — | 0.407 | 1.032 | 0.630 |  |
| G5 | 10 | M1_texture | cal_v1_registered | 90 | 0.00 | — | — | 0.292 | 0.777 | 0.540 | platt |
| G5 | 10 | M1_texture | cal_v2_fe | 90 | 0.00 | — | — | 0.317 | 0.828 | 0.562 | fe_platt slope=0.183 |
| G5 | 10 | M2_dino | raw | 90 | 0.00 | — | — | 0.118 | 0.406 | 0.318 |  |
| G5 | 10 | M2_dino | cal_v1_registered | 90 | 0.00 | — | — | 0.278 | 0.750 | 0.527 | platt |
| G5 | 10 | M2_dino | cal_v2_fe | 90 | 0.00 | — | — | 0.262 | 0.718 | 0.511 | fe_platt slope=0.209 |
| G5 | 10 | M3_dino_distinct | raw | 90 | 0.00 | — | — | 0.124 | 0.419 | 0.325 |  |
| G5 | 10 | M3_dino_distinct | cal_v1_registered | 90 | 0.00 | — | — | 0.282 | 0.756 | 0.530 | platt |
| G5 | 10 | M3_dino_distinct | cal_v2_fe | 90 | 0.00 | — | — | 0.265 | 0.723 | 0.514 | fe_platt slope=0.198 |
| G6 | 08,09 | M0_constant | raw | 180 | 0.59 | 0.500 | 0.589 | 0.257 | 0.707 | 0.122 |  |
| G6 | 08,09 | M1_texture | raw | 180 | 0.59 | 0.514 | 0.634 | 0.297 | 0.856 | 0.235 |  |
| G6 | 08,09 | M1_texture | cal_v1_registered | 180 | 0.59 | 0.486 | 0.593 | 0.274 | 0.784 | 0.132 | platt |
| G6 | 08,09 | M1_texture | cal_v2_fe | 180 | 0.59 | 0.514 | 0.634 | 0.256 | 0.705 | 0.132 | fe_platt slope=0.156 |
| G6 | 08,09 | M2_dino | raw | 180 | 0.59 | 0.547 | 0.634 | 0.340 | 1.093 | 0.271 |  |
| G6 | 08,09 | M2_dino | cal_v1_registered | 180 | 0.59 | 0.547 | 0.634 | 0.256 | 0.705 | 0.113 | platt |
| G6 | 08,09 | M2_dino | cal_v2_fe | 180 | 0.59 | 0.547 | 0.634 | 0.264 | 0.731 | 0.125 | fe_platt slope=0.334 |
| G6 | 08,09 | M3_dino_distinct | raw | 180 | 0.59 | 0.542 | 0.630 | 0.341 | 1.096 | 0.271 |  |
| G6 | 08,09 | M3_dino_distinct | cal_v1_registered | 180 | 0.59 | 0.542 | 0.630 | 0.256 | 0.705 | 0.113 | platt |
| G6 | 08,09 | M3_dino_distinct | cal_v2_fe | 180 | 0.59 | 0.542 | 0.630 | 0.263 | 0.726 | 0.115 | fe_platt slope=0.305 |
| G7 | 11 | M0_constant | raw | 90 | 0.89 | 0.500 | 0.889 | 0.294 | 0.782 | 0.442 |  |
| G7 | 11 | M1_texture | raw | 90 | 0.89 | 0.304 | 0.860 | 0.501 | 1.355 | 0.589 |  |
| G7 | 11 | M1_texture | cal_v1_registered | 90 | 0.89 | 0.304 | 0.860 | 0.319 | 0.833 | 0.464 | platt |
| G7 | 11 | M1_texture | cal_v2_fe | 90 | 0.89 | 0.304 | 0.860 | 0.363 | 0.934 | 0.498 | fe_platt slope=0.402 |
| G7 | 11 | M2_dino | raw | 90 | 0.89 | 0.506 | 0.899 | 0.723 | 2.540 | 0.783 |  |
| G7 | 11 | M2_dino | cal_v1_registered | 90 | 0.89 | 0.506 | 0.899 | 0.568 | 1.519 | 0.678 | platt |
| G7 | 11 | M2_dino | cal_v2_fe | 90 | 0.89 | 0.506 | 0.899 | 0.528 | 1.371 | 0.649 | fe_platt slope=0.411 |
| G7 | 11 | M3_dino_distinct | raw | 90 | 0.89 | 0.484 | 0.892 | 0.731 | 2.610 | 0.787 |  |
| G7 | 11 | M3_dino_distinct | cal_v1_registered | 90 | 0.89 | 0.484 | 0.892 | 0.560 | 1.488 | 0.672 | platt |
| G7 | 11 | M3_dino_distinct | cal_v2_fe | 90 | 0.89 | 0.484 | 0.892 | 0.510 | 1.311 | 0.635 | fe_platt slope=0.367 |

AUROC and PR-AUC are undefined (—) for a held-out group with one outcome class. (dev) marks the fold containing localizer-dev region 03.

Reliability diagrams: `figures/stage6_8_pilot_v2/reliability_<model>.png`. Per-row predictions and per-fold metrics: `results/stage8/pilot_v2/`.
