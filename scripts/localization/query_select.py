"""Query selection (same rules as Stage 3A)."""
from __future__ import annotations

import csv

import numpy as np

from project_paths import MANIFEST


def load_manifest_region(region: str) -> list[dict]:
    rows = []
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["region_folder"] == region:
                rows.append(r)
    return sorted(rows, key=lambda x: int(x["num"]))


def select_queries(manifest: list[dict], n: int, seed: int) -> list[dict]:
    by_num = {int(r["num"]): r for r in manifest}
    nmax = max(by_num)
    deciles = {max(1, min(nmax, int(round((d / 10) * nmax)))) for d in range(1, 11)}
    chosen = [by_num[k] for k in sorted(deciles) if k in by_num]
    seen = {int(r["num"]) for r in chosen}
    if len(chosen) < n:
        step = max(1, nmax // (n - len(chosen) + 1))
        for num in range(1, nmax + 1, step):
            if num not in seen:
                chosen.append(by_num[num])
                seen.add(num)
            if len(chosen) >= n:
                break
    rng = np.random.default_rng(seed)
    if len(chosen) > n:
        idx = rng.choice(len(chosen), size=n, replace=False)
        chosen = [chosen[i] for i in sorted(idx)]
    return sorted(chosen[:n], key=lambda r: int(r["num"]))
