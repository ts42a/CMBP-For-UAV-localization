# Accept-rule selection (region 03 tune set only)

Data: `results/stage4/tune/outcomes.csv` — 216 windowed attempts, 24 frames. Primary tolerance 100 m.
Criterion: highest precision among rules with >= 5 accepted-correct; ties -> more accepted-correct.

| Rule | Attempts | Accepted | Acc. correct | Acc. incorrect | Precision | Acc.-correct rate |
|---|---:|---:|---:|---:|---:|---:|
| any_output | 216 | 215 | 108 | 107 | 0.5023 | 0.5 |
| homography_12 | 216 | 7 | 7 | 0 | 1.0 | 0.0324 |
| homography_20 | 216 | 0 | 0 | 0 |  | 0.0 |
| homography_30 | 216 | 0 | 0 | 0 |  | 0.0 |

**Selected: `homography_12`**

Per-radius breakdown: `reports/accept_rule_selection.csv`.
