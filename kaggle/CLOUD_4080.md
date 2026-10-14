# RTX 4080 cloud benchmark handoff

The author confirmed on 2026-10-14 that the 4080 will be a cloud computer.
No cloud host or completed 4080 run is currently connected to this workspace.
No cloud instance has been rented or started by this task.

## Prepare on the PC

Use the committed repair revision recorded in the remediation report and package
it explicitly (replace `<repair-commit>` with that exact commit):

```powershell
.\.venv\Scripts\python.exe scripts/protocol/package_kaggle.py --ref <repair-commit> --out build/cloud4080
```

The package command produces a code ZIP, cache ZIP and bundle manifest with SHA-256
hashes. Transfer those privately to the cloud host, together with the necessary
UAV-VisLoc region folders, in accordance with `DATA_STATEMENT.md`. The benchmark
uses only two pilot frames per benchmark region, but the verified reference maps,
cache metadata and manifest must be available. Do not expose these bundles publicly.

## Run on the cloud host

Use a compatible CUDA torch environment; keep its installed versions in the
runner's `env.json`. Do not copy the CPU virtual environment from Windows to Linux.
Install the non-torch dependencies appropriate to the recorded runtime, and
preserve the pinned DINOv2 source and checkpoint from the cache bundle. This
handoff does not claim an unmeasured environment is equivalent to the T4.

Extract the code to `/workspace/CMBL` and the cache to `/workspace/cache` in a
fresh controlled directory. Verify transferred ZIP SHA-256 values against
`bundle_manifest.json`. The torch cache must be writable because torch hub writes
its trust metadata. Then, from the code directory:

```bash
python scripts/protocol/cloud_4080_benchmark.py \
  --data-root /workspace/UAV_VisLoc_dataset \
  --emb-root /workspace/cache/embeddings \
  --torch-home /workspace/cache/torch_home
```

The wrapper refuses a non-4080 or CPU runtime. The runner fixes float32/TF32-off
numerics and executes the registered 20 frames × 9 attempts, without a reduced
`--max-frames` option. Its output includes actual Python/torch/GPU details and model
pins. If an old output already exists with incompatible provenance, retain it and
use a fresh checkout; do not silently merge runs.

## Return and evaluate

Return the **entire** `results/stage10/rtx4080_bench` folder to the same path on the
PC. Preserve any previous folder rather than overwriting it. Then run:

```powershell
.\.venv\Scripts\python.exe scripts/protocol/launch_gate.py
```

It requires the exact 180 T4 attempt keys, identical outcome labels and at least
95% of jointly accepted positions within 5 m. Missing runtime evidence fails.
Only after PASS may the seven missing Stage 11 jobs be launched. The separate
2,000-attempt replay fidelity check is still required before hybrid replay use.
