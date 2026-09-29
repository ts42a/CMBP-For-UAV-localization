# Effective reach of the frozen localizer — dev region 03

`scripts/analysis/reach_summary.py`. Region 03 tune queries (24 frames, localizer-dev region only), nominal radius **100 m**, prior centre displaced east of truth by the offset below; frozen matcher and accept rule `patch_20`. Outputs: `results/stage9_prep/reach_r03/off<m>/`.

| prior offset (m) | attempts | window tiles (median) | accepted-correct | accepted-incorrect | rejected | success rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 2 | 24 | 0 | 0 | 1.000 |
| 250 | 24 | 2 | 4 | 0 | 20 | 0.167 |
| 500 | 24 | 2 | 0 | 0 | 24 | 0.000 |
| 750 | 24 | 2 | 0 | 0 | 24 | 0.000 |
| 1000 | 24 | 2 | 0 | 0 | 24 | 0.000 |
| 1500 | 24 | 2 | 0 | 0 | 24 | 0.000 |
| 2000 | 24 | 1 | 0 | 0 | 24 | 0.000 |

## Reading

- The reach beyond the nominal radius is short: success is 100% with the prior on the truth, drops sharply by 250 m and is 0% from 500 m. The verifier only accepts when the ~500 m query footprint lies inside a 600–900 m crop centred on a window tile, which tolerates roughly 100–200 m between that tile centre and the truth.
- Effective reach ≈ R + ~250 m. An earlier estimate in the project evaluation (R + 455 m, from the crop size alone) was wrong and is corrected here.
- The flat pilot success-by-radius pattern is a design effect (prior σ = R/2 scales the offset with R, and a larger window still contains tiles near the truth), not long reach.
- Dev region only; other regions' reach also depends on height above ground (footprint) and is not measured here, to avoid using test-region outcomes.
