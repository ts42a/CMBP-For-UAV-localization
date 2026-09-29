# stage10_local_mock — Stage 10 benchmark vs PC reference

Run `<temporary copy outside the repo>/CMBL/results/stage10/local_mock` (CPU; bundle `428c362`) vs `results/stage5/pilot/outcomes.csv` on the same attempt keys. Criteria frozen in `configs/stage10_benchmark.yaml`.

**Agreement: PASS**

```json
{
  "n_attempts": 18,
  "n_frames": 2,
  "expected_frames": 2,
  "expected_attempts": 18,
  "no_duplicate_keys": true,
  "all_keys_in_reference": true,
  "config_sha256_matches_reference": true,
  "complete": true,
  "outcome_disagreements": 0,
  "outcome_agreement_frac": 1.0,
  "outcome_criterion_pass": true,
  "both_accepted": 13,
  "position_within_5m": 13,
  "position_max_diff_m": 0.0,
  "position_criterion_pass": true,
  "agreement_pass": true
}
```

## Timing (wall seconds)

| | run | PC reference |
|---|---:|---:|
| attempt mean | 8.331 | 9.984 |
| attempt median, R=100 m | 0.554 | 0.789 |
| attempt median, R=300 m | 6.516 | 8.435 |
| attempt median, R=1000 m | 16.064 | 18.55 |
| frame setup mean | 4.748 | 5.974 |
| frame total mean (setup + 9 attempts) | 79.72 | 95.83 |

Runner: {"import_torch_s": 1.941, "model_load_s": 2.56, "label_run_s": 164.465, "gpu_peak_mem_mib_labelling": null, "embedding_bench": {"region": "10", "tiles": 88, "map_km2": 2.259, "seconds": 4.509, "s_per_tile": 0.0512, "s_per_km2": 1.996, "max_abs_diff_vs_pc_cache": 0.0, "min_cosine_vs_pc_cache": 0.9999998211860657}, "total_runner_s": 177.686}

## Disagreements

None.
