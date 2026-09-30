# Stage 10 decision — Kaggle benchmark

Date 2026-09-30. Rules frozen in `configs/stage10_benchmark.yaml` (2026-09-28) before any Kaggle output existed.

**Decision: GO on Kaggle GPU for the full-scale real labels (Stage 11). H1 synthetic labels and the
Stage 9 replay do not fit the frozen cost rule together with them; their compute is allocated when
`configs/h1_protocol_v1.yaml` and `configs/stage9_protocol_v1.yaml` are approved.**

## Agreement with the PC labels (the gate)

| run | hardware | attempts | outcome disagreements | both accepted | within 5 m | max position diff | result |
|---|---|---:|---:|---:|---:|---:|---|
| `results/stage10/kaggle_bench` | Kaggle, Tesla T4, torch 2.10.0+cu128, TF32 off | 180 | 0 | 105 | 105 | 0.0 m | **PASS** |
| `results/stage10/kaggle_cpu_bench` | Kaggle CPU (4 cores; GPU not granted, see incidents) | 180 | 0 | 105 | 105 | 0.0 m | PASS |

Criteria: ≥ 97 % identical outcomes and ≥ 95 % of jointly accepted fixes within 5 m, same config
sha256 as the PC rows. Both runs meet them with no disagreement at all
(`reports/stage10_benchmark.md`, `reports/stage10_benchmark_kaggle_cpu.md`). GPU labels are
therefore equivalent to the PC labels the localizer was selected on; `on_fail` does not apply.

## Measurements

| item | value |
|---|---|
| data transfer (private dataset upload, 24.0 GB, 11 region tars) | 7,758 s at 3.09 MB/s + 189 s Kaggle processing |
| code bundle / cache bundle upload | 0.7 MB in 10 s / 111 MB in 38 s |
| environment setup (import torch + DINOv2 load from the uploaded cache) | 1.8 s + 33.8 s |
| per attempt, T4 (mean; median at R = 100 / 300 / 1000 m) | 1.95 s (0.29 / 2.32 / 3.27 s) |
| per frame, T4 (setup + 9 attempts) | 18.8 s, vs 63.6 s for the same frames on the PC CPU reference |
| per frame, Kaggle CPU | 101.5 s |
| GPU peak memory while labelling | 120 MiB |
| map feature extraction, region 10 (88 tiles, 2.26 km²) | 0.019 s/tile, 0.74 s/km²; max abs diff to the PC cache 2.3e-6 |
| FP16 tile cache | rejected: ordered top-20 identical for 172 of 180 windows, not all (`reports/stage10_fp16_cache.json`); FP32 stays (25.6 MB) |
| weekly GPU quota, measured on the account | 30 h |

## Cost projection (`reports/stage10_cost_sheet_kaggle_gpu.md`, × 1.5 safety)

| workload | GPU h | vs rule (3 × 30 h = 90 h) |
|---|---:|---|
| full real labels, 6,744 frames × 9 attempts | 52.8 | fits |
| + H1 synthetic labels (matched count) | 105.5 | exceeds |
| + Stage 9 replay (assumed design, 379,350 attempts) | 417.5 | exceeds |

The frozen rule (`stay_on_kaggle_if: projected GPU hours <= 3 x the weekly GPU quota`) is applied
per workload in the order the study needs them. Full real labels fit, so they run on Kaggle GPU.
H1 and replay are not started under this decision; before their protocols are approved, choose
among: a smaller replay design fixed in Stage 9 (its counts are still assumptions), running replay
on the PC CPU in parallel, or rented GPU time. That choice is recorded with the Stage 9 freeze.

## Incidents (no effect on the gate)

1. The kernel read `/kaggle/input` as a writable torch hub; `torch.hub` writes `trusted_list` and
   failed. Fixed in `kaggle/stage10_kernel.py` (writable hub view in `/tmp`) before the benchmark ran.
2. The first push ran on CPU because the account could not be given a GPU (Kaggle substitutes CPU
   silently). Those rows were kept as a Kaggle CPU run, not labelled GPU; the kernel now stops when
   no GPU is attached. The GPU run followed once the account had GPU access.

## Next

Stage 11 labels: `configs/stage11_labels_v1.yaml`, run with
`py -3 scripts/protocol/kaggle_stage10.py labels`.
