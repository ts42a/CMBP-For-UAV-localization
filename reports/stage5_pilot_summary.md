# stage5_pilot — windowed outcome summary

Rows 990, frames 110, regions 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11. Accept rule `['patch_20']`; primary tolerance 100 m.

## Checks (gating — frozen in preregistration_v1.yaml stage5_go)

```json
{
  "prevalence_in_band": true,
  "prevalence_band": [
    0.15,
    0.85
  ],
  "accepted_correct_nonincreasing_with_offset": true,
  "regions_in_band": "6/11",
  "stage5_go": true
}
```

## Informational only — not part of the frozen gate

```json
{
  "all_three_outcomes_present": true,
  "accepted_correct_nonincreasing_with_radius": false,
  "note": "plan prose also expects the radius trend; only the offset trend and prevalence band are frozen in preregistration_v1.yaml stage5_go."
}
```

### Pooled (all regions, incl. localizer-dev region 03)

| | n | acc. correct | rejected | acc. incorrect | acc.-correct rate | precision |
|---|---:|---:|---:|---:|---:|---:|
| all | 990 | 442 | 539 | 9 | 0.4465 | 0.98 |

### Pooled excluding localizer-dev region (informational)

| | n | acc. correct | rejected | acc. incorrect | acc.-correct rate | precision |
|---|---:|---:|---:|---:|---:|---:|
| all | 900 | 357 | 534 | 9 | 0.3967 | 0.9754 |

Secondary tolerances (accepted-correct rate): {'25m': 0.204, '50m': 0.3909}

### By radius (m)

| | n | acc. correct | rejected | acc. incorrect | acc.-correct rate | precision |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 330 | 150 | 180 | 0 | 0.4545 | 1.0 |
| 1000 | 330 | 131 | 193 | 6 | 0.397 | 0.9562 |
| 300 | 330 | 161 | 166 | 3 | 0.4879 | 0.9817 |

### By prior offset / radius

| | n | acc. correct | rejected | acc. incorrect | acc.-correct rate | precision |
|---|---:|---:|---:|---:|---:|---:|
| 0-0.5R | 369 | 193 | 172 | 4 | 0.523 | 0.9797 |
| 0.5-1R | 476 | 209 | 263 | 4 | 0.4391 | 0.9812 |
| 1-1.5R | 139 | 39 | 99 | 1 | 0.2806 | 0.975 |
| 1.5-99R | 6 | 1 | 5 | 0 | 0.1667 | 1.0 |

### By truth inside window

| | n | acc. correct | rejected | acc. incorrect | acc.-correct rate | precision |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 145 | 40 | 104 | 1 | 0.2759 | 0.9756 |
| 1 | 845 | 402 | 435 | 8 | 0.4757 | 0.9805 |

### By region

| | n | acc. correct | rejected | acc. incorrect | acc.-correct rate | precision |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 90 | 38 | 52 | 0 | 0.4222 | 1.0 |
| 02 | 90 | 30 | 60 | 0 | 0.3333 | 1.0 |
| 03 | 90 | 85 | 5 | 0 | 0.9444 | 1.0 |
| 04 | 90 | 74 | 16 | 0 | 0.8222 | 1.0 |
| 05 | 90 | 24 | 63 | 3 | 0.2667 | 0.8889 |
| 06 | 90 | 5 | 79 | 6 | 0.0556 | 0.4545 |
| 07 | 90 | 0 | 90 | 0 | 0.0 | None |
| 08 | 90 | 55 | 35 | 0 | 0.6111 | 1.0 |
| 09 | 90 | 51 | 39 | 0 | 0.5667 | 1.0 |
| 10 | 90 | 0 | 90 | 0 | 0.0 | None |
| 11 | 90 | 80 | 10 | 0 | 0.8889 | 1.0 |

### By 7-group

| | n | acc. correct | rejected | acc. incorrect | acc.-correct rate | precision |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 180 | 68 | 112 | 0 | 0.3778 | 1.0 |
| G2 | 180 | 159 | 21 | 0 | 0.8833 | 1.0 |
| G3 | 90 | 24 | 63 | 3 | 0.2667 | 0.8889 |
| G4 | 90 | 5 | 79 | 6 | 0.0556 | 0.4545 |
| G5 | 180 | 0 | 180 | 0 | 0.0 | None |
| G6 | 180 | 106 | 74 | 0 | 0.5889 | 1.0 |
| G7 | 90 | 80 | 10 | 0 | 0.8889 | 1.0 |

Gallery: `figures/stage5_pilot/` (17 panels; every accepted-incorrect row included).
