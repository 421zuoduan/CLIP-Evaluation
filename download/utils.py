from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable
import shutil

from .registry import DATASET_SPECS, DatasetSpec


@dataclass
class DownloadResult:
    key: str
    status: str
    message: str = ""


def _dataset_present(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _import_class(path: str):
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _download_torchvision(spec: DatasetSpec, root: Path, force: bool, verbose: bool = True) -> DownloadResult:
    dataset_root = root / spec.target_subdir
    if dataset_root.exists() and force:
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    if not force and _dataset_present(dataset_root):
        return DownloadResult(spec.key, "skipped", "already present")

    cls = _import_class(spec.identifier)
    splits = spec.splits or tuple(str(i) for i in range(len(spec.kwargs_per_split)))

    try:
        did_download = False
        for idx, kwargs in enumerate(spec.kwargs_per_split or ({},)):
            split_name = splits[idx] if idx < len(splits) else str(idx)
            if verbose:
                print(f"[torchvision] downloading {spec.key} split={split_name}")
            cls(root=str(dataset_root), download=True, **kwargs)
            did_download = True
        status = "downloaded" if did_download else "skipped"
        message = "" if did_download else "no action required"
        return DownloadResult(spec.key, status, message)
    except Exception as exc:
        return DownloadResult(spec.key, "failed", str(exc))


def _download_datasets(spec: DatasetSpec, root: Path, force: bool, verbose: bool = True) -> DownloadResult:
    try:
        import datasets as hf_datasets
    except ImportError as exc:
        return DownloadResult(spec.key, "failed", f"datasets library not available: {exc}")

    dataset_root = root / spec.target_subdir
    cache_dir = dataset_root / "__hf_cache__"
    dataset_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def save_split(split_name: str, ds_obj) -> bool:
        split_dir = dataset_root / split_name
        if split_dir.exists():
            if force:
                shutil.rmtree(split_dir)
            elif _dataset_present(split_dir):
                if verbose:
                    print(f"[datasets] skip existing split {spec.key}:{split_name}")
                return False
        if verbose:
            print(f"[datasets] saving {spec.key}:{split_name} to {split_dir}")
        ds_obj.save_to_disk(str(split_dir))
        return True

    try:
        did_download = False
        if spec.splits:
            for split_name in spec.splits:
                if verbose:
                    print(f"[datasets] downloading {spec.key} split={split_name}")
                ds = hf_datasets.load_dataset(
                    path=spec.identifier,
                    name=spec.hf_config,
                    split=split_name,
                    cache_dir=str(cache_dir),
                )
                if save_split(split_name, ds):
                    did_download = True
        else:
            if verbose:
                print(f"[datasets] downloading {spec.key} (all splits)")
            ds_all = hf_datasets.load_dataset(
                path=spec.identifier,
                name=spec.hf_config,
                cache_dir=str(cache_dir),
            )
            if hasattr(ds_all, "items"):
                for split_name, split_ds in ds_all.items():
                    if save_split(str(split_name), split_ds):
                        did_download = True
            else:
                if save_split("data", ds_all):
                    did_download = True
        status = "downloaded" if did_download else "skipped"
        message = "" if did_download else "no action required"
        return DownloadResult(spec.key, status, message)
    except Exception as exc:
        return DownloadResult(spec.key, "failed", str(exc))


def download_dataset(
    key: str,
    data_root: Path | str = Path("data"),
    *,
    force: bool = False,
    verbose: bool = True,
) -> DownloadResult:
    """
    Download a dataset defined in :mod:`download.registry` into ``data_root``.

    Args:
        key: Dataset key from the registry (case-insensitive).
        data_root: Target directory holding dataset folders.
        force: Re-download even if target folder already contains files.
        verbose: Print progress information.
    """
    key_norm = key.lower()
    spec = DATASET_SPECS.get(key_norm)
    if not spec:
        available = ", ".join(sorted(DATASET_SPECS))
        return DownloadResult(key_norm, "failed", f"unknown dataset key. available: {available}")

    data_root_path = Path(data_root)
    data_root_path.mkdir(parents=True, exist_ok=True)

    if spec.backend == "torchvision":
        return _download_torchvision(spec, data_root_path, force, verbose=verbose)
    if spec.backend == "datasets":
        return _download_datasets(spec, data_root_path, force, verbose=verbose)

    return DownloadResult(spec.key, "failed", f"unsupported backend {spec.backend}")


def download_many(
    keys: Iterable[str],
    data_root: Path | str = Path("data"),
    *,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, DownloadResult]:
    results: Dict[str, DownloadResult] = {}
    for key in keys:
        result = download_dataset(key, data_root=data_root, force=force, verbose=verbose)
        results[result.key] = result
        if verbose and result.message:
            print(f" -> {result.status.upper()}: {result.message}")
    return results
