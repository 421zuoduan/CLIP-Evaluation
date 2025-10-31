from __future__ import annotations

import os
from typing import List, Sequence

import torch
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL_NAME = "openai/clip-vit-large-patch14"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def _ensure_hf_mirror() -> None:
    """
    Configure Hugging Face to use a mainland China friendly mirror by default.
    Users can override the endpoint via environment variables before importing
    this module if they prefer a different mirror.
    """
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def load_clip_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> tuple[CLIPModel, CLIPProcessor]:
    """
    Load CLIP ViT-L/14 weights and processor.

    Args:
        model_name: Hugging Face model identifier.
        device: Target device (defaults to CUDA if available, else CPU).
        dtype: Optional dtype string or torch.dtype (e.g. "fp16").

    Returns:
        Tuple of (CLIPModel, CLIPProcessor).
    """
    _ensure_hf_mirror()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    resolved_dtype: torch.dtype | None
    if dtype is None:
        resolved_dtype = torch.float16 if device.type == "cuda" else None
    elif isinstance(dtype, str):
        lower = dtype.lower()
        if lower == "fp16":
            resolved_dtype = torch.float16
        elif lower == "bf16":
            resolved_dtype = torch.bfloat16
        else:
            resolved_dtype = torch.float32
    else:
        resolved_dtype = dtype

    model = CLIPModel.from_pretrained(model_name)
    if resolved_dtype is not None:
        model = model.to(dtype=resolved_dtype)
    model = model.to(device)
    model.eval()

    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(p=2, dim=-1, keepdim=True)


def _infer_autocast_dtype(model: CLIPModel) -> torch.dtype:
    return next(model.parameters()).dtype


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompts: Sequence[str],
    *,
    device: str | torch.device,
    autocast: bool = True,
) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)
    text_inputs = processor(text=list(prompts), padding=True, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    dtype = _infer_autocast_dtype(model)
    if autocast and device.type == "cuda":
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            text_features = model.get_text_features(**text_inputs)
    else:
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
    return normalize_features(text_features)


def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: Sequence,
    *,
    device: str | torch.device,
    batch_size: int = 32,
    autocast: bool = True,
) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)

    outputs: List[torch.Tensor] = []
    length = len(images)
    for start in range(0, length, batch_size):
        batch = images[start : start + batch_size]
        image_inputs = processor(images=batch, return_tensors="pt").to(device)
        if autocast and device.type == "cuda":
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=_infer_autocast_dtype(model)):
                image_features = model.get_image_features(**image_inputs)
        else:
            with torch.no_grad():
                image_features = model.get_image_features(**image_inputs)
        outputs.append(image_features)

    image_embeddings = torch.cat(outputs, dim=0)
    return normalize_features(image_embeddings)
