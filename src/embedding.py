from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

import numpy as np
import open_clip
from PIL import Image
import torch

from .config import ModelConfig, model


@lru_cache(maxsize=1)
def _load_model(cfg: ModelConfig = model):
    """Load OpenCLIP model and preprocess pipeline (cached)."""
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained
    )
    device = torch.device(cfg.device)
    clip_model.to(device)
    clip_model.eval()
    return clip_model, preprocess, device


def _load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def encode_images(
    image_paths: Iterable[str],
    batch_size: int | None = None,
    cfg: ModelConfig = model,
) -> np.ndarray:
    """Encode a list of image paths into an array of embeddings (N x D)."""
    clip_model, preprocess, device = _load_model(cfg)
    bs = batch_size or cfg.batch_size

    paths_list: List[str] = list(image_paths)
    if not paths_list:
        return np.zeros((0, clip_model.visual.output_dim), dtype=np.float32)

    all_embeds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(paths_list), bs):
            batch_paths = paths_list[i : i + bs]
            images = [_load_image(p) for p in batch_paths]
            inputs = torch.stack([preprocess(img) for img in images]).to(device)
            feats = clip_model.encode_image(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeds, axis=0)


def get_embedding_dim(cfg: ModelConfig = model) -> int:
    clip_model, _, _ = _load_model(cfg)
    return int(clip_model.visual.output_dim)

