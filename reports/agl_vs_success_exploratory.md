# Pilot success vs height above ground — EXPLORATORY, post hoc

**Not a registered analysis.** Looked at after the pilot outcomes were known, to decide whether a flight-context diagnostic (M4) is worth registering. Two functional forms were tried, in the order below; both are reported. AGL from `reports/agl_height_audit.md` (CSV height − SRTM terrain). Region 07 excluded (D2). 10 frames per region, 9 attempts per frame.

| region | success (accepted-correct rate) | AGL median (m) | AGL / region 03 |
|---|---:|---:|---:|
| 10 | 0.000 | 293 | 0.63 |
| 06 | 0.056 | 308 | 0.67 |
| 05 | 0.267 | 375 | 0.81 |
| 01 | 0.422 | 388 | 0.84 |
| 02 | 0.333 | 388 | 0.84 |
| 03 | 0.944 | 462 | 1.00 |
| 04 | 0.822 | 541 | 1.17 |
| 09 | 0.567 | 542 | 1.17 |
| 08 | 0.611 | 549 | 1.19 |
| 11 | 0.889 | 778 | 1.68 |

| test (Spearman) | rho | p | n |
|---|---:|---:|---:|
| (1) region: success vs -abs(log(AGL/AGL03)) | 0.54 | 0.108 | 10 |
| (1) frame: success vs -abs(log(AGL/AGL03)) | 0.37 | 0.000122 | 100 |
| (2) region: success vs AGL | 0.83 | 0.00294 | 10 |
| (2) frame: success vs AGL | 0.57 | 4.17e-10 | 100 |

Within-region frame-level rho (AGL varies little inside a flight): 01 -0.31, 02 -0.69, 03 -0.29, 04 -0.14, 05 0.46, 06 0.29, 08 -0.32, 09 -0.23, 11 -0.16

## Reading

- Across regions, success rises with height above ground (form 2); the region-03-centred form (1) is weaker. Higher flights give a larger ground footprint, closer to the verifier's 600–900 m context crops.
- Inside a flight AGL barely varies and the relation is mixed, so AGL speaks to the **between-region base rate** (40% of label variance), which a map-only predictor cannot see.
- Ten regions, post hoc, and confounded with terrain type (flat eastern plains fly higher above ground here). This motivates a diagnostic, not a claim.
