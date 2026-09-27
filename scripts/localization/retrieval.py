"""Cosine retrieval over L2-normalized embeddings."""
from __future__ import annotations

import numpy as np


def top_k_similar(query: np.ndarray, refs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """query (D,), refs (N,D) L2-normalized -> indices (k,), scores (k,)."""
    if refs.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    q = query / (np.linalg.norm(query) + 1e-8)
    scores = refs @ q
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def gt_tile_id(tiles, row: float, col: float) -> int | None:
    for t in tiles:
        if t.row0 <= row < t.row1 and t.col0 <= col < t.col1:
            return t.tile_id
    return None
