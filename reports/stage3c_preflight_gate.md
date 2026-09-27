# Stage 3C preflight gate

**Decision: READY**

- Untouched val CSV: `stage3b_val_dinov2_c2f.csv` n=30
- Recomputed success@100 m: **9** / failure: **21**
- Search used CSV position: **0** (must be 0)
- Predictions that are exact CSV copies: **0** (must be 0)
- Location from tile match or homography: **30/30**
- Among @100 m successes: homography=2, tile-centre=7
- Human scene review is **not** the 100 m criterion; geodesic error vs scoring columns is.

Gate passed. Full-map labeling may proceed with the frozen matcher.
