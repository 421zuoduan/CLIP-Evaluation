from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple


Backend = Literal["torchvision", "datasets"]


@dataclass(frozen=True)
class DatasetSpec:
    """
    Specification describing how to fetch and store a dataset locally.

    For torchvision-backed datasets, ``identifier`` is the import path for the
    dataset class (e.g. ``torchvision.datasets.CIFAR10``) and ``kwargs_per_split``
    contains one dict per split instantiation.

    For Hugging Face datasets, ``identifier`` is the dataset id passed to
    ``datasets.load_dataset`` and ``hf_config`` optionally selects a named
    configuration. ``splits`` enumerates the splits to download; when empty the
    downloader will fetch all available splits.
    """

    key: str
    backend: Backend
    identifier: str
    target_subdir: str
    splits: Tuple[str, ...] = ()
    kwargs_per_split: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    hf_config: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.backend == "torchvision" and not self.kwargs_per_split:
            object.__setattr__(self, "kwargs_per_split", ({},))
        if self.backend == "torchvision" and self.splits and len(self.splits) != len(self.kwargs_per_split):
            raise ValueError(f"{self.key}: splits and kwargs_per_split length mismatch")


DATASET_SPECS: Dict[str, DatasetSpec] = {
    # ImageNet variants (Hugging Face datasets)
    "imagenet1k": DatasetSpec(
        key="imagenet1k",
        backend="datasets",
        identifier="imagenet-1k",
        target_subdir="imagenet1k",
        splits=("train", "validation"),
        description="ImageNet-1K classification benchmark.",
    ),
    "imagenet_v2": DatasetSpec(
        key="imagenet_v2",
        backend="datasets",
        identifier="imagenet_v2",
        target_subdir="imagenet_v2",
        splits=("test",),
        description="ImageNet-V2 matched-frequency evaluation split.",
    ),
    "imagenet_adv": DatasetSpec(
        key="imagenet_adv",
        backend="datasets",
        identifier="imagenet-a",
        target_subdir="imagenet_adv",
        splits=("test",),
        description="Adversarial ImageNet-A evaluation.",
    ),
    "imagenet_ren": DatasetSpec(
        key="imagenet_ren",
        backend="datasets",
        identifier="imagenet-r",
        target_subdir="imagenet_ren",
        splits=("test",),
        description="ImageNet-R (renditions) robustness benchmark.",
    ),
    "imagenet_ske": DatasetSpec(
        key="imagenet_ske",
        backend="datasets",
        identifier="imagenet-sketch",
        target_subdir="imagenet_ske",
        splits=("test",),
        description="ImageNet-Sketch robustness benchmark.",
    ),
    "objectnet": DatasetSpec(
        key="objectnet",
        backend="datasets",
        identifier="objectnet",
        target_subdir="objectnet",
        description="ObjectNet out-of-distribution benchmark.",
    ),
    # Torchvision classic benchmarks
    "cifar10": DatasetSpec(
        key="cifar10",
        backend="torchvision",
        identifier="torchvision.datasets.CIFAR10",
        target_subdir="cifar10",
        splits=("train", "test"),
        kwargs_per_split=(
            {"train": True},
            {"train": False},
        ),
        description="CIFAR-10 image classification dataset.",
    ),
    "cifar100": DatasetSpec(
        key="cifar100",
        backend="torchvision",
        identifier="torchvision.datasets.CIFAR100",
        target_subdir="cifar100",
        splits=("train", "test"),
        kwargs_per_split=(
            {"train": True},
            {"train": False},
        ),
        description="CIFAR-100 image classification dataset.",
    ),
    "mnist": DatasetSpec(
        key="mnist",
        backend="torchvision",
        identifier="torchvision.datasets.MNIST",
        target_subdir="mnist",
        splits=("train", "test"),
        kwargs_per_split=(
            {"train": True},
            {"train": False},
        ),
        description="Handwritten digit recognition dataset.",
    ),
    "caltech101": DatasetSpec(
        key="caltech101",
        backend="torchvision",
        identifier="torchvision.datasets.Caltech101",
        target_subdir="caltech101",
        description="Caltech-101 object classification dataset.",
    ),
    "sun397": DatasetSpec(
        key="sun397",
        backend="torchvision",
        identifier="torchvision.datasets.SUN397",
        target_subdir="sun397",
        description="SUN397 scene classification dataset.",
    ),
    "fgvc_aircraft": DatasetSpec(
        key="fgvc_aircraft",
        backend="torchvision",
        identifier="torchvision.datasets.FGVCAircraft",
        target_subdir="fgvc_aircraft",
        splits=("train", "val", "test"),
        kwargs_per_split=(
            {"split": "train"},
            {"split": "val"},
            {"split": "test"},
        ),
        description="Fine-grained FGVC Aircraft dataset.",
    ),
    "stanford_cars": DatasetSpec(
        key="stanford_cars",
        backend="torchvision",
        identifier="torchvision.datasets.StanfordCars",
        target_subdir="stanford_cars",
        splits=("train", "test"),
        kwargs_per_split=(
            {"split": "train"},
            {"split": "test"},
        ),
        description="Stanford Cars fine-grained dataset.",
    ),
    "dtd": DatasetSpec(
        key="dtd",
        backend="torchvision",
        identifier="torchvision.datasets.DTD",
        target_subdir="dtd",
        splits=("train", "val", "test"),
        kwargs_per_split=(
            {"split": "train"},
            {"split": "val"},
            {"split": "test"},
        ),
        description="Describable Textures Dataset.",
    ),
    "flowers102": DatasetSpec(
        key="flowers102",
        backend="torchvision",
        identifier="torchvision.datasets.Flowers102",
        target_subdir="flowers102",
        splits=("train", "val", "test"),
        kwargs_per_split=(
            {"split": "train"},
            {"split": "val"},
            {"split": "test"},
        ),
        description="Oxford 102 category flower dataset.",
    ),
    "food101": DatasetSpec(
        key="food101",
        backend="torchvision",
        identifier="torchvision.datasets.Food101",
        target_subdir="food101",
        description="Food-101 large-scale food classification dataset.",
    ),
    "gtsrb": DatasetSpec(
        key="gtsrb",
        backend="torchvision",
        identifier="torchvision.datasets.GTSRB",
        target_subdir="gtsrb",
        splits=("train", "test"),
        kwargs_per_split=(
            {"split": "train"},
            {"split": "test"},
        ),
        description="German Traffic Sign Recognition Benchmark.",
    ),
    "pets": DatasetSpec(
        key="pets",
        backend="torchvision",
        identifier="torchvision.datasets.OxfordIIITPet",
        target_subdir="pets",
        splits=("trainval", "test"),
        kwargs_per_split=(
            {"split": "trainval"},
            {"split": "test"},
        ),
        description="Oxford-IIIT Pet dataset.",
    ),
    "stl10": DatasetSpec(
        key="stl10",
        backend="torchvision",
        identifier="torchvision.datasets.STL10",
        target_subdir="stl10",
        splits=("train", "test", "unlabeled"),
        kwargs_per_split=(
            {"split": "train"},
            {"split": "test"},
            {"split": "unlabeled"},
        ),
        description="STL-10 dataset with train/test/unlabeled splits.",
    ),
    "voc2007": DatasetSpec(
        key="voc2007",
        backend="torchvision",
        identifier="torchvision.datasets.VOCDetection",
        target_subdir="voc2007",
        splits=("train", "val", "test"),
        kwargs_per_split=(
            {"image_set": "train", "year": "2007"},
            {"image_set": "val", "year": "2007"},
            {"image_set": "test", "year": "2007"},
        ),
        description="PASCAL VOC 2007 detection dataset.",
    ),
    # Hugging Face datasets for remaining benchmarks
    "country211": DatasetSpec(
        key="country211",
        backend="datasets",
        identifier="country211",
        target_subdir="country211",
        description="Country211 geographic classification dataset.",
    ),
    "birdsnap": DatasetSpec(
        key="birdsnap",
        backend="datasets",
        identifier="birdsnap",
        target_subdir="birdsnap",
        description="Birdsnap fine-grained bird species dataset.",
    ),
    "eurosat": DatasetSpec(
        key="eurosat",
        backend="datasets",
        identifier="eurosat",
        target_subdir="eurosat",
        description="EuroSAT remote sensing dataset.",
    ),
    "fer2013": DatasetSpec(
        key="fer2013",
        backend="datasets",
        identifier="fer2013",
        target_subdir="fer2013",
        description="FER2013 facial expression recognition dataset.",
    ),
    "pcam": DatasetSpec(
        key="pcam",
        backend="datasets",
        identifier="patchcamelyon",
        target_subdir="pcam",
        description="PatchCamelyon histopathology dataset.",
    ),
    "rendered_sst2": DatasetSpec(
        key="rendered_sst2",
        backend="datasets",
        identifier="rendered-sst2",
        target_subdir="rendered_sst2",
        description="Rendered SST-2 sentiment images dataset.",
    ),
    "resisc45": DatasetSpec(
        key="resisc45",
        backend="datasets",
        identifier="resisc45",
        target_subdir="resisc45",
        description="RESISC45 remote sensing dataset.",
    ),
}
