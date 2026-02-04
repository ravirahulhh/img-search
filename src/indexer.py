from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from .config import IndexConfig, ensure_directories, index as index_cfg, paths
from .embedding import encode_images, get_embedding_dim


@dataclass
class IndexPaths:
    index_path: str
    metadata_path: str


def _default_index_paths(index_dir: str | None = None) -> IndexPaths:
    base = index_dir or paths.index_dir
    os.makedirs(base, exist_ok=True)
    return IndexPaths(
        index_path=os.path.join(base, "frames_index.faiss"),
        metadata_path=os.path.join(base, "index_meta.jsonl"),
    )


def _load_frames_metadata(frames_metadata_path: str) -> Tuple[List[str], List[dict]]:
    image_paths: List[str] = []
    metas: List[dict] = []
    with open(frames_metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            meta = json.loads(line)
            image_paths.append(meta["frame_path"])
            metas.append(meta)
    return image_paths, metas


def build_pq_index(
    frames_metadata_path: str | None = None,
    index_dir: str | None = None,
    cfg: IndexConfig = index_cfg,
) -> IndexPaths:
    """Build a FAISS IVFPQ index from frame metadata and save it to disk."""
    ensure_directories()

    frames_metadata_path = frames_metadata_path or paths.frames_metadata_path
    paths_obj = _default_index_paths(index_dir)

    image_paths, metas = _load_frames_metadata(frames_metadata_path)
    if not image_paths:
        raise ValueError("No frame metadata found; did you run extract-frames?")

    dim = get_embedding_dim()
    # Compute embeddings in batches and collect them in memory (PoC scale).
    embeds = encode_images(image_paths)
    if embeds.shape[1] != dim:
        raise RuntimeError(f"Unexpected embedding dim {embeds.shape[1]} != {dim}")

    xb = embeds.astype("float32")
    n, d = xb.shape

    # IVFPQ needs ~9984+ points for PQ codebook (256 centroids × 39); otherwise use flat index.
    MIN_POINTS_FOR_IVFPQ = 10_000
    if n < MIN_POINTS_FOR_IVFPQ:
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        print(f"Using exact search (IndexFlatL2) for {n} vectors; use IVFPQ when you have ≥{MIN_POINTS_FOR_IVFPQ}.")
    else:
        train_size = min(100_000, n)
        nlist_actual = min(cfg.nlist, train_size, max(1, train_size // 39))
        nprobe_actual = min(cfg.nprobe, nlist_actual)
        if nlist_actual < cfg.nlist:
            print(
                f"Note: nlist set to {nlist_actual} for {n} vectors (FAISS needs ~39 points per centroid; cfg nlist={cfg.nlist})"
            )
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(
            quantizer,
            d,
            nlist_actual,
            cfg.m,
            cfg.nbits,
        )
        train_samples = xb[np.random.choice(n, size=train_size, replace=False)]
        index.train(train_samples)
        index.add(xb)
        index.nprobe = nprobe_actual

    # Save index and metadata.
    faiss.write_index(index, paths_obj.index_path)
    with open(paths_obj.metadata_path, "w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    return paths_obj


def load_index(
    index_dir: str | None = None,
    cfg: IndexConfig = index_cfg,
) -> Tuple[faiss.Index, List[dict]]:
    """Load FAISS index and corresponding metadata from disk."""
    paths_obj = _default_index_paths(index_dir)
    if not os.path.exists(paths_obj.index_path):
        raise FileNotFoundError(f"Index file not found: {paths_obj.index_path}")
    if not os.path.exists(paths_obj.metadata_path):
        raise FileNotFoundError(f"Index metadata not found: {paths_obj.metadata_path}")

    index = faiss.read_index(paths_obj.index_path)
    if hasattr(index, "nlist") and hasattr(index, "nprobe"):
        index.nprobe = min(cfg.nprobe, index.nlist)

    metas: List[dict] = []
    with open(paths_obj.metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            metas.append(json.loads(line))

    if index.ntotal != len(metas):
        # Not fatal for PoC, but warn loudly.
        print(
            f"Warning: index size ({index.ntotal}) != metadata count ({len(metas)}). "
            "Results may be misaligned."
        )

    return index, metas

