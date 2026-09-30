# stage10_benchmark_kaggle_cpu — Stage 10 benchmark vs PC reference

Run `results/stage10/kaggle_cpu_bench` (CPU; bundle `473ace0`) vs `results/stage5/pilot/outcomes.csv` on the same attempt keys. Criteria frozen in `configs/stage10_benchmark.yaml`.

**Agreement: PASS**

```json
{
  "n_attempts": 180,
  "n_frames": 20,
  "expected_frames": 20,
  "expected_attempts": 180,
  "no_duplicate_keys": true,
  "all_keys_in_reference": true,
  "config_sha256_matches_reference": true,
  "complete": true,
  "outcome_disagreements": 0,
  "outcome_agreement_frac": 1.0,
  "outcome_criterion_pass": true,
  "both_accepted": 105,
  "position_within_5m": 105,
  "position_max_diff_m": 0.0,
  "position_criterion_pass": true,
  "agreement_pass": true
}
```

## Timing (wall seconds)

| | run | PC reference |
|---|---:|---:|
| attempt mean | 10.375 | 6.524 |
| attempt median, R=100 m | 1.802 | 1.114 |
| attempt median, R=300 m | 12.486 | 7.797 |
| attempt median, R=1000 m | 18.105 | 10.407 |
| frame setup mean | 8.129 | 4.838 |
| frame total mean (setup + 9 attempts) | 101.5 | 63.56 |

Runner: {"import_torch_s": 1.912, "model_load_s": 33.5, "label_run_s": 2195.366, "gpu_peak_mem_mib_labelling": null, "embedding_bench": {"region": "10", "tiles": 88, "map_km2": 2.259, "seconds": 12.797, "s_per_tile": 0.1454, "s_per_km2": 5.665, "max_abs_diff_vs_pc_cache": 6.92903995513916e-07, "min_cosine_vs_pc_cache": 0.9999998807907104}, "total_runner_s": 2277.917}

## Disagreements

None.
