"""
Download package for fetching benchmark datasets used in CLIP ViT-L/14 evaluation.

The public API exposes helpers and a registry so tooling (e.g. main.py) can
programmatically ensure all required assets exist under the local data/ tree.
"""

from .registry import DATASET_SPECS, DatasetSpec  # noqa: F401
from .utils import download_dataset  # noqa: F401

__all__ = ["DATASET_SPECS", "DatasetSpec", "download_dataset"]
