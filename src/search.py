from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .embedding import encode_images
from .indexer import load_index


@dataclass
class SearchResult:
    score: float
    metadata: dict


def search_by_image(
    image_path: str,
    top_k: int = 10,
    index_dir: str | None = None,
) -> List[SearchResult]:
    """Search most similar frames for a given query image."""
    index, metas = load_index(index_dir)
    query_embed = encode_images([image_path])
    if query_embed.shape[0] == 0:
        return []

    D, I = index.search(query_embed, top_k)
    distances = D[0]  # FAISS returns squared L2 for IndexFlatL2 / IndexIVFPQ
    indices = I[0]

    results: List[SearchResult] = []
    for dist, idx in zip(distances, indices):
        if idx < 0 or idx >= len(metas):
            continue
        # Embeddings are L2-normalized: sq_dist = 2 - 2*cos(θ), so cos(θ) = 1 - sq_dist/2.
        # Use cosine similarity as score: 1.0 = identical, 0 = orthogonal, negative = opposite.
        score = float(1.0 - dist / 2.0)
        results.append(SearchResult(score=score, metadata=metas[int(idx)]))
    return results

