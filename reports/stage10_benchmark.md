# stage10_benchmark — Stage 10 benchmark vs PC reference

Run `F:\CMBL\results\stage10\kaggle_bench` (Tesla T4; bundle `473ace0`) vs `results/stage5/pilot/outcomes.csv` on the same attempt keys. Criteria frozen in `configs/stage10_benchmark.yaml`.

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
| attempt mean | 1.949 | 6.524 |
| attempt median, R=100 m | 0.29 | 1.114 |
| attempt median, R=300 m | 2.319 | 7.797 |
| attempt median, R=1000 m | 3.267 | 10.407 |
| frame setup mean | 1.22 | 4.838 |
| frame total mean (setup + 9 attempts) | 18.76 | 63.56 |

Runner: {"import_torch_s": 1.75, "model_load_s": 33.825, "label_run_s": 534.935, "gpu_peak_mem_mib_labelling": 119.95751953125, "embedding_bench": {"region": "10", "tiles": 88, "map_km2": 2.259, "seconds": 1.679, "s_per_tile": 0.0191, "s_per_km2": 0.743, "max_abs_diff_vs_pc_cache": 2.2687017917633057e-06, "min_cosine_vs_pc_cache": 0.9999997615814209}, "total_runner_s": 606.387}

## Disagreements

None.
