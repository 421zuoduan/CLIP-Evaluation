"""
Model package exposing utilities for loading and running CLIP ViT-L/14.

Typical usage::

    from model import load_clip_model

    model, processor = load_clip_model(device="cuda")
    feats = encode_images(model, processor, pil_images, device="cuda")
"""

from .clip import (
    DEFAULT_MODEL_NAME,
    load_clip_model,
    encode_images,
    encode_texts,
    normalize_features,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "load_clip_model",
    "encode_images",
    "encode_texts",
    "normalize_features",
]
