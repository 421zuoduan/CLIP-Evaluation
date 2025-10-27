from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .registry import DATASET_SPECS
from .utils import download_many


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets required for CLIP ViT-L/14 evaluation.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated list of dataset keys to download (default: all registered datasets).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Destination folder for downloaded datasets (default: ./data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the target directory already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset keys and exit.",
    )
    return parser.parse_args()


def _select_datasets(selection: str) -> Sequence[str]:
    if selection.lower() == "all":
        return list(DATASET_SPECS.keys())
    keys = []
    for item in selection.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset key '{key}'. Use --list to inspect available options.")
        keys.append(key)
    return keys


def main() -> None:
    args = _parse_args()

    if args.list:
        print("Available datasets:")
        for key, spec in sorted(DATASET_SPECS.items()):
            print(f" - {key:15s} ({spec.backend}) -> {spec.description}")
        return

    try:
        keys = _select_datasets(args.datasets)
    except ValueError as exc:
        print(exc)
        return

    verbose = not args.quiet
    results = download_many(keys, data_root=args.data_root, force=args.force, verbose=verbose)

    failures = [r for r in results.values() if r.status == "failed"]
    skipped = [r for r in results.values() if r.status == "skipped"]
    downloaded = [r for r in results.values() if r.status == "downloaded"]

    print("\nSummary:")
    print(f" downloaded: {len(downloaded)}")
    print(f" skipped:    {len(skipped)}")
    print(f" failed:     {len(failures)}")

    if failures:
        print("\nFailures:")
        for fail in failures:
            print(f" - {fail.key}: {fail.message}")


if __name__ == "__main__":
    main()
