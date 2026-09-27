# Label-generation gate (development pilot)

**Decision: PASS**

Primary frozen mode: `dino_fine`

## Summaries @100m

{
  "dino_global": {
    "mode": "dino_global",
    "n": 100,
    "success_100m": 4,
    "failure_100m": 96,
    "success_rate_100m": 0.04,
    "median_error_m": 2254.8295,
    "mean_error_m": 2285.20945,
    "p90_error_m": 4403.490100000001,
    "success_25m": 0,
    "success_50m": 0,
    "success_200m": 12,
    "mean_runtime_s": 0.37439999999999996
  },
  "dino_fine": {
    "mode": "dino_fine",
    "n": 100,
    "success_100m": 8,
    "failure_100m": 92,
    "success_rate_100m": 0.08,
    "median_error_m": 2013.4875000000002,
    "mean_error_m": 2003.6786799999995,
    "p90_error_m": 4274.135800000001,
    "success_25m": 4,
    "success_50m": 5,
    "success_200m": 24,
    "mean_runtime_s": 4.3983799999999995
  },
  "baseline_pyramid_ncc_rot4": {
    "mode": "baseline_pyramid_ncc_rot4",
    "n": 100,
    "success_100m": 0,
    "failure_100m": 100,
    "success_rate_100m": 0.0,
    "median_error_m": 3295.831,
    "mean_error_m": 3272.3798899999992,
    "p90_error_m": 5795.532,
    "success_25m": 0,
    "success_50m": 0,
    "success_200m": 0,
    "mean_runtime_s": 2.2382899999999997
  }
}

## Evidence

- Mixed success/failure @100m: **True**
- Dev tuning used **24** queries on region **03** only.

No localizability predictor training. No full 6,774 run.
