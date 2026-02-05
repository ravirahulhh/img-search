from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoImageProcessor

from .config import ModelConfig, model


@lru_cache(maxsize=1)
def _load_model(cfg: ModelConfig = model):
    device = torch.device(cfg.device)
    # Load SigLIP2 (or compatible SigLIP) model and image processor from HuggingFace.
    vision_model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    vision_model.to(device)
    vision_model.eval()
    return vision_model, image_processor, device


def _infer_embedding_dim(vision_model: torch.nn.Module) -> int:
    """Best-effort inference of image embedding dimension from SigLIP/SigLIP2 config."""
    cfg = getattr(vision_model, "config", None)
    if cfg is None:
        raise AttributeError("Model has no config; cannot infer embedding dimension.")

    # Prefer explicit projection dimension used for image/text alignment.
    for attr in (
        "projection_dim",
        "image_embed_dim",
    ):
        dim = getattr(cfg, attr, None)
        if isinstance(dim, int) and dim > 0:
            return dim

    # Try nested vision config (common for multi-modal models).
    vision_cfg = getattr(cfg, "vision_config", None)
    if vision_cfg is not None:
        for attr in (
            "hidden_size",
            "embed_dim",
            "width",
        ):
            dim = getattr(vision_cfg, attr, None)
            if isinstance(dim, int) and dim > 0:
                return dim

    # Fallback to a few common attribute names on the root config.
    for attr in (
        "hidden_size",
        "embed_dim",
        "width",
    ):
        dim = getattr(cfg, attr, None)
        if isinstance(dim, int) and dim > 0:
            return dim

    raise AttributeError(
        "Could not infer embedding dimension from model config; "
        "please check the SigLIP/SigLIP2 config fields."
    )


def _load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def encode_images(
    image_paths: Iterable[str],
    batch_size: int | None = None,
    cfg: ModelConfig = model,
) -> np.ndarray:
    """Encode a list of image paths into an array of embeddings (N x D)."""
    vision_model, image_processor, device = _load_model(cfg)
    bs = batch_size or cfg.batch_size

    paths_list: List[str] = list(image_paths)
    if not paths_list:
        # Use model config to determine embedding dimension.
        vision_model, _, _ = _load_model(cfg)
        dim = _infer_embedding_dim(vision_model)
        return np.zeros((0, int(dim)), dtype=np.float32)

    all_embeds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(paths_list), bs):
            batch_paths = paths_list[i : i + bs]
            images = [_load_image(p) for p in batch_paths]
            inputs = image_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # SigLIP2 exposes get_image_features for normalized image embeddings.
            if hasattr(vision_model, "get_image_features"):
                feats = vision_model.get_image_features(**inputs)
            else:
                outputs = vision_model(**inputs)
                # Fallback: pool last hidden state if projection head is not available.
                last_hidden = outputs.last_hidden_state  # (B, seq_len, hidden_size)
                feats = last_hidden[:, 0]  # CLS token
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeds, axis=0)


def get_embedding_dim(cfg: ModelConfig = model) -> int:
    vision_model, _, _ = _load_model(cfg)
    dim = _infer_embedding_dim(vision_model)
    return int(dim)

