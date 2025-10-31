#!/usr/bin/env python3
"""
CLIP ViT-L/14 零样本评估脚本

该脚本用于评估 CLIP ViT-L/14 模型在多个视觉数据集上的零样本分类性能。
支持自动检测 GPU、批处理推理、进度显示和结果保存。

使用示例:
    python main.py --datasets cifar10,imagenet1k --data-root ./data --model-root ./model_weights --output-dir ./results
    python main.py --datasets all --batch-size 64 --num-workers 4
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets, transforms
from tqdm import tqdm
from PIL import Image
import pickle

# 数据集类名映射
DATASET_CLASSES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ],
    "cifar100": [f"class_{i}" for i in range(100)],  # 简化处理，实际应使用真实类名
    "mnist": [str(i) for i in range(10)],
    "imagenet1k": [f"imagenet_class_{i}" for i in range(1000)],  # 将在加载时更新为实际类别名称
    "caltech101": [f"caltech101_class_{i}" for i in range(101)],  # 简化处理
    "stanford_cars": [f"car_model_{i}" for i in range(196)],  # 简化处理
    "flowers102": [f"flower_{i}" for i in range(102)],  # 简化处理
    "food101": [f"food_{i}" for i in range(101)],  # 简化处理
    "pets": [f"pet_{i}" for i in range(37)],  # 简化处理
    "sun397": [f"scene_{i}" for i in range(397)],  # 简化处理
    "dtd": [f"texture_{i}" for i in range(47)],  # 简化处理
    "gtsrb": [f"traffic_sign_{i}" for i in range(43)],  # 简化处理
    "stl10": [f"stl10_class_{i}" for i in range(10)],  # 简化处理
    "voc2007": [f"voc_object_{i}" for i in range(20)],  # 简化处理
    "country211": [f"country_{i}" for i in range(211)],  # 简化处理
    "eurosat": [f"land_use_{i}" for i in range(10)],  # 简化处理
    "fer2013": [f"emotion_{i}" for i in range(7)],  # 简化处理
    "resisc45": [f"scene_{i}" for i in range(45)],  # 简化处理
    "rendered_sst2": ["negative", "positive"],
    "imagenet_v2": [f"imagenet_v2_class_{i}" for i in range(1000)],  # 简化处理
    "imagenet_adv": [f"imagenet_adv_class_{i}" for i in range(1000)],  # 简化处理
    "imagenet_ren": [f"imagenet_ren_class_{i}" for i in range(200)],  # 简化处理
    "imagenet_ske": [f"imagenet_ske_class_{i}" for i in range(1000)],  # 简化处理
    "objectnet": [f"object_{i}" for i in range(313)],  # 简化处理
    "fgvc_aircraft": [f"aircraft_{i}" for i in range(100)],  # 简化处理
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CLIP ViT-L/14 零样本评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        default="cifar10,cifar100,mnist",
        help="要评估的数据集列表，用逗号分隔，或使用 'all' 评估所有支持的数据集"
    )
    
    parser.add_argument(
        "--model-root",
        type=str,
        default="./model_weights",
        help="CLIP 模型根目录"
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="数据集根目录"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="结果输出目录"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="数据加载器的工作进程数"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备 (auto/cuda/cpu)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="数据类型 (auto/fp16/fp32)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制每个数据集的样本数量（用于快速测试）"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="计算 top-k 准确率"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """设置计算设备"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ 使用 GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("⚠️  未检测到 GPU，使用 CPU")
    else:
        device = torch.device(device_arg)
        print(f"✅ 使用设备: {device}")
    
    return device


def setup_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    """设置数据类型"""
    if dtype_arg == "auto":
        if device.type == "cuda":
            dtype = torch.float16
            print("✅ 使用 FP16 精度")
        else:
            dtype = torch.float32
            print("✅ 使用 FP32 精度")
    else:
        if dtype_arg == "fp16":
            dtype = torch.float16
        elif dtype_arg == "fp32":
            dtype = torch.float32
        else:
            raise ValueError(f"不支持的数据类型: {dtype_arg}")
        print(f"✅ 使用 {dtype_arg.upper()} 精度")
    
    return dtype


def create_directories(output_dir: str) -> None:
    """创建必要的目录"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✅ 结果将保存到: {output_dir}")


def get_available_datasets(data_root: str) -> List[str]:
    """获取 data 文件夹中可用的数据集列表"""
    data_path = Path(data_root)
    if not data_path.exists():
        print(f"⚠️  数据根目录不存在: {data_root}")
        return []
    
    # 获取所有子目录
    subdirs = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    # 过滤出支持的数据集
    available_datasets = []
    for subdir in subdirs:
        if subdir in DATASET_CLASSES:
            available_datasets.append(subdir)
        else:
            print(f"⚠️  跳过不支持的数据集: {subdir}")
    
    return sorted(available_datasets)


class CIFAR10Dataset(Dataset):
    """CIFAR-10 数据集加载器"""
    def __init__(self, data_path: str, train: bool = False, transform=None):
        self.transform = transform
        self.data_path = data_path
        
        # 加载 CIFAR-10 数据
        cifar_path = os.path.join(data_path, "cifar-10-batches-py")
        if train:
            # 加载训练数据
            self.data = []
            self.labels = []
            for i in range(1, 6):
                batch_path = os.path.join(cifar_path, f"data_batch_{i}")
                with open(batch_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    self.data.append(batch[b'data'])
                    self.labels.extend(batch[b'labels'])
            self.data = np.concatenate(self.data)
        else:
            # 加载测试数据
            batch_path = os.path.join(cifar_path, "test_batch")
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data = batch[b'data']
                self.labels = batch[b'labels']
        
        # 重塑数据为 (N, 3, 32, 32)
        self.data = self.data.reshape(len(self.data), 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)  # 转换为 (N, 32, 32, 3)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # 转换为 PIL Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class CIFAR100Dataset(Dataset):
    """CIFAR-100 数据集加载器"""
    def __init__(self, data_path: str, train: bool = False, transform=None):
        self.transform = transform
        self.data_path = data_path
        
        # 加载 CIFAR-100 数据
        cifar_path = os.path.join(data_path, "cifar-100-python")
        if train:
            batch_path = os.path.join(cifar_path, "train")
        else:
            batch_path = os.path.join(cifar_path, "test")
            
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            self.data = batch[b'data']
            self.labels = batch[b'fine_labels']
        
        # 重塑数据为 (N, 3, 32, 32)
        self.data = self.data.reshape(len(self.data), 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)  # 转换为 (N, 32, 32, 3)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # 转换为 PIL Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class MNISTRawFallbackDataset(Dataset):
    """当 torchvision 处理过的 MNIST 文件缺失时的后备加载器"""

    def __init__(self, raw_root: str, split: str = "test", transform=None):
        self.transform = transform
        if split == "train":
            prefix = "train"
        elif split == "test":
            prefix = "t10k"
        else:
            raise ValueError(f"MNISTRawFallbackDataset 不支持的 split: {split}")

        images_path = os.path.join(raw_root, f"{prefix}-images-idx3-ubyte")
        labels_path = os.path.join(raw_root, f"{prefix}-labels-idx1-ubyte")

        self.data = self._load_images(images_path)
        self.labels = self._load_labels(labels_path)

        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError("MNIST 图像与标签数量不一致")

    @staticmethod
    def _resolve_path(base_path: str) -> str:
        if os.path.exists(base_path):
            return base_path
        gz_path = base_path + ".gz"
        if os.path.exists(gz_path):
            return gz_path
        raise FileNotFoundError(f"未找到 MNIST 文件: {base_path}(.gz)")

    @classmethod
    def _load_images(cls, base_path: str) -> np.ndarray:
        import gzip
        import struct

        path = cls._resolve_path(base_path)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"无效的 MNIST 图像文件魔数: {magic}")
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return data

    @classmethod
    def _load_labels(cls, base_path: str) -> np.ndarray:
        import gzip
        import struct

        path = cls._resolve_path(base_path)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"无效的 MNIST 标签文件魔数: {magic}")
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.data[idx], mode="L").convert("RGB")
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


class Caltech101Dataset(Dataset):
    """Caltech-101 数据集加载器"""
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # Caltech-101 特殊路径结构
        image_folder_path = os.path.join(data_path, "caltech101", "101_ObjectCategories")
        if not os.path.exists(image_folder_path):
            raise FileNotFoundError(f"Caltech-101 图像文件夹不存在: {image_folder_path}")
        
        print(f"🔍 扫描 Caltech-101 图像文件夹: {image_folder_path}")
        
        # 查找所有类别文件夹
        classes = []
        for item in os.listdir(image_folder_path):
            item_path = os.path.join(image_folder_path, item)
            if os.path.isdir(item_path):
                classes.append(item)
        
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"📁 找到 {len(classes)} 个类别: {classes[:5]}{'...' if len(classes) > 5 else ''}")
        
        # 收集所有图像文件
        for class_name in classes:
            class_dir = os.path.join(image_folder_path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            for img_name in img_files:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
        
        print(f"🖼️  找到 {len(self.samples)} 个图像文件")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class StanfordCarsDataset(Dataset):
    """Stanford Cars 数据集加载器"""
    def __init__(self, data_path: str, transform=None, limit=None):
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        # Stanford Cars 数据集路径
        cars_root = os.path.join(data_path, "stanford_cars")
        if not os.path.exists(cars_root):
            # 尝试其他可能的路径
            possible_paths = [
                os.path.join(data_path, "stanford_cars", "stanford_cars"),
                data_path
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    cars_root = path
                    break
            else:
                raise FileNotFoundError(f"Stanford Cars数据集目录不存在: {cars_root}")
        
        print(f"🔍 扫描 Stanford Cars 数据集: {cars_root}")
        
        # 加载类别名称
        meta_path = os.path.join(cars_root, "devkit", "cars_meta.mat")
        if os.path.exists(meta_path):
            try:
                import scipy.io
                meta_data = scipy.io.loadmat(meta_path)
                # 获取实际的类别名称
                self.class_names = [name[0] for name in meta_data['class_names'][0]]
                print(f"✅ 从元数据文件加载了 {len(self.class_names)} 个类别")
            except Exception as e:
                print(f"⚠️  无法加载类别元数据文件: {e}")
                self.class_names = [f"Car Model {i+1}" for i in range(196)]
        else:
            print(f"⚠️  类别元数据文件不存在: {meta_path}")
            self.class_names = [f"Car Model {i+1}" for i in range(196)]
        
        # 尝试加载测试集的标注文件
        test_annos_path = os.path.join(cars_root, "cars_test_annos_withlabels.mat")
        test_dir = os.path.join(cars_root, "cars_test")
        
        if os.path.exists(test_annos_path) and os.path.exists(test_dir):
            try:
                import scipy.io
                # 加载标注文件
                annotations = scipy.io.loadmat(test_annos_path)
                test_annotations = annotations['annotations'][0]
                
                # 处理每个测试图像
                for anno in test_annotations:
                    # 获取图像文件名和标签
                    img_filename = anno[-1][0]  # 图像文件名在最后一个位置
                    label = int(anno[-2][0] - 1)  # MATLAB索引从1开始，转换为0开始
                    
                    img_path = os.path.join(test_dir, img_filename)
                    if os.path.exists(img_path):
                        self.samples.append((img_path, label))
                
                print(f"📁 从标注文件加载了 {len(self.samples)} 个测试图像")
            except Exception as e:
                print(f"⚠️  无法加载标注文件: {e}")
                # 回退到简单的文件夹扫描
                self._fallback_load(cars_root, limit)
        else:
            # 回退到简单的文件夹扫描
            self._fallback_load(cars_root, limit)
        
        print(f"🖼️  找到 {len(self.samples)} 个图像文件")
        print(f"🏷️  类别数量: {len(self.class_names)}")
        
        # 限制样本数量
        if limit is not None and limit < len(self.samples):
            self.samples = self.samples[:limit]
            print(f"⚠️  限制样本数量为: {limit}")
    
    def _fallback_load(self, cars_root, limit=None):
        """回退加载方法，当无法加载标注文件时使用"""
        print("🔄 使用回退方法加载数据...")
        
        # 查找所有图像文件
        for subdir in ["cars_test", "cars_train"]:
            subdir_path = os.path.join(cars_root, subdir)
            if os.path.exists(subdir_path):
                img_files = [f for f in os.listdir(subdir_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_file in img_files:
                    img_path = os.path.join(subdir_path, img_file)
                    # 使用简单的类别标签
                    self.samples.append((img_path, 0 if subdir == "cars_test" else 1))
        
        # 使用简单的类别名称
        self.class_names = ["Test Cars", "Train Cars"]
        
        # 限制样本数量
        if limit is not None and limit < len(self.samples):
            self.samples = self.samples[:limit]
            print(f"⚠️  限制样本数量为: {limit}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 确保标签在有效范围内
        if label >= len(self.class_names):
            label = 0
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @property
    def class_to_idx(self):
        """为了兼容性，提供class_to_idx属性"""
        return {name: i for i, name in enumerate(self.class_names)}


class Flowers102Dataset(Dataset):
    """Flowers-102 数据集加载器"""
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        # Flowers-102 数据集结构
        # 图像在 ./data/flowers102/flowers-102/jpg/ 目录下
        # 标签在 ./data/flowers102/flowers-102/imagelabels.mat 文件中
        
        # 查找数据集目录
        dataset_dir = None
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path) and "flowers-102" in item.lower():
                dataset_dir = item_path
                break
        
        if dataset_dir is None:
            raise FileNotFoundError(f"未找到Flowers-102数据集目录在 {data_path}")
        
        # 图像目录
        images_dir = os.path.join(dataset_dir, "jpg")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Flowers-102图像目录不存在: {images_dir}")
        
        # 标签文件
        labels_file = os.path.join(dataset_dir, "imagelabels.mat")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Flowers-102标签文件不存在: {labels_file}")
        
        # 加载标签
        import scipy.io
        labels_data = scipy.io.loadmat(labels_file)
        # imagelabels.mat包含一个名为'labels'的数组，形状为(1, N)，其中N是图像数量
        labels = labels_data['labels'][0]  # 获取标签数组
        
        # Flowers-102类别名称（102种花）
        self.class_names = [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
            "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle",
            "snapdragon", "colts' foot", "king protea", "spear thistle", "yellow iris",
            "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
            "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
            "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
            "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
            "ruby-lipped cattleya", "cape flower", "masterwort", "siam tulip", "lenten rose",
            "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
            "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
            "petunia", "wild pansy", "primula", "sunflower", "pelargonium",
            "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
            "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "french marigold",
            "bromelia", "blanket flower", "trumpet creeper", "camellia", "mallow",
            "mexican petunia", "bougainvillea", "water lily", "rose", "thorn apple",
            "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
            "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
            "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
            "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
            "camellia", "mallow", "mexican petunia", "bougainvillea", "water lily",
            "rose", "thorn apple", "morning glory", "passion flower", "lotus"
        ]
        
        # 获取所有图像文件并排序
        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        img_files.sort()  # 确保图像文件按名称排序，与标签对应
        
        # 创建样本列表
        for i, img_file in enumerate(img_files):
            if i < len(labels):
                # Flowers-102的标签从1开始，需要转换为0开始的索引
                label_idx = int(labels[i]) - 1
                if 0 <= label_idx < len(self.class_names):
                    img_path = os.path.join(images_dir, img_file)
                    self.samples.append((img_path, label_idx))
        
        print(f"✅ 成功加载 {len(self.samples)} 个样本")
        print(f"✅ 加载了 {len(self.class_names)} 个类别")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_classes(self):
        return self.class_names


class Food101Dataset(Dataset):
    """Food-101 数据集加载器"""
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        # Food-101 数据集路径
        food101_path = os.path.join(data_path, "food-101")
        if not os.path.exists(food101_path):
            # 尝试其他可能的路径
            possible_paths = [
                os.path.join(data_path, "food101"),
                data_path
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    food101_path = path
                    break
            else:
                raise FileNotFoundError(f"无法找到 Food-101 数据集目录")
        
        print(f"🔍 扫描 Food-101 数据集: {food101_path}")
        
        # 加载类别名称
        classes_file = os.path.join(food101_path, "meta", "classes.txt")
        if not os.path.exists(classes_file):
            raise FileNotFoundError(f"Food-101 类别文件不存在: {classes_file}")
        
        with open(classes_file, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        print(f"📁 加载了 {len(self.class_names)} 个类别")
        print(f"📝 前5个类别: {self.class_names[:5]}")
        
        # 创建类别到索引的映射
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # 加载测试集列表
        test_file = os.path.join(food101_path, "meta", "test.txt")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Food-101 测试集文件不存在: {test_file}")
        
        with open(test_file, 'r') as f:
            test_lines = [line.strip() for line in f.readlines()]
        
        # 图像目录
        images_dir = os.path.join(food101_path, "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Food-101 图像目录不存在: {images_dir}")
        
        # 处理测试集
        for line in test_lines:
            # 格式: 类别名/图像ID
            if '/' in line:
                class_name, img_id = line.split('/', 1)
                if class_name in class_to_idx:
                    class_idx = class_to_idx[class_name]
                    img_path = os.path.join(images_dir, class_name, f"{img_id}.jpg")
                    if os.path.exists(img_path):
                        self.samples.append((img_path, class_idx))
        
        print(f"🖼️  找到 {len(self.samples)} 个测试图像文件")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_classes(self):
        return self.class_names


class PetsDataset(Dataset):
    """Oxford-IIIT Pets 数据集加载器"""
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        # Oxford-IIIT Pets 数据集结构
        # 图像在 ./data/pets/oxford-iiit-pet/images/ 目录下
        # 标注在 ./data/pets/oxford-iiit-pet/annotations/xmls/ 目录下
        
        # 查找数据集目录
        dataset_dir = None
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path) and "oxford-iiit-pet" in item.lower():
                dataset_dir = item_path
                break
        
        if dataset_dir is None:
            raise FileNotFoundError(f"未找到Oxford-IIIT Pets数据集目录在 {data_path}")
        
        # 图像目录
        images_dir = os.path.join(dataset_dir, "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Oxford-IIIT Pets图像目录不存在: {images_dir}")
        
        # 标注目录
        annotations_dir = os.path.join(dataset_dir, "annotations", "xmls")
        if not os.path.exists(annotations_dir):
            raise FileNotFoundError(f"Oxford-IIIT Pets标注目录不存在: {annotations_dir}")
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 从图像文件名提取类别名称
        # Oxford-IIIT Pets的图像文件名格式为: breed_XXX.jpg
        breed_names = set()
        for img_file in image_files:
            # 提取品种名称（下划线前的部分）
            breed_name = "_".join(img_file.split("_")[:-1])
            breed_names.add(breed_name)
        
        # 排序并创建类别映射
        self.class_names = sorted(list(breed_names))
        class_to_idx = {breed: i for i, breed in enumerate(self.class_names)}
        
        print(f"✅ 从图像文件名提取了 {len(self.class_names)} 个类别")
        
        # 创建样本列表
        for img_file in image_files:
            # 提取品种名称
            breed_name = "_".join(img_file.split("_")[:-1])
            
            if breed_name in class_to_idx:
                img_path = os.path.join(images_dir, img_file)
                class_idx = class_to_idx[breed_name]
                self.samples.append((img_path, class_idx))
        
        print(f"✅ 成功加载 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    @property
    def class_to_idx(self):
        """为了兼容性，提供class_to_idx属性"""
        return {name: i for i, name in enumerate(self.class_names)}


class DTDataset(Dataset):
    """Describable Textures Dataset (DTD) 数据集加载器"""
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # DTD 可能的路径结构
        possible_paths = [
            os.path.join(data_path, "test"),
            os.path.join(data_path, "dtd"),
            os.path.join(data_path, "images"),
            data_path
        ]
        
        image_folder_path = None
        for path in possible_paths:
            if os.path.exists(path):
                # 检查是否包含子目录
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    image_folder_path = path
                    break
        
        if image_folder_path is None:
            raise FileNotFoundError(f"无法找到 DTD 的图像文件夹")
        
        print(f"🔍 扫描 DTD 图像文件夹: {image_folder_path}")
        
        # 查找所有类别文件夹
        classes = []
        for item in os.listdir(image_folder_path):
            item_path = os.path.join(image_folder_path, item)
            if os.path.isdir(item_path):
                classes.append(item)
        
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"📁 找到 {len(classes)} 个类别: {classes[:5]}{'...' if len(classes) > 5 else ''}")
        
        # 收集所有图像文件
        for class_name in classes:
            class_dir = os.path.join(image_folder_path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            for img_name in img_files:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
        
        print(f"🖼️  找到 {len(self.samples)} 个图像文件")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class GTSRBDataset(Dataset):
    """German Traffic Sign Recognition Benchmark (GTSRB) 数据集加载器"""
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # GTSRB 可能的路径结构
        possible_paths = [
            os.path.join(data_path, "test"),
            os.path.join(data_path, "gtsrb"),
            os.path.join(data_path, "images"),
            data_path
        ]
        
        image_folder_path = None
        for path in possible_paths:
            if os.path.exists(path):
                # 检查是否包含子目录
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    image_folder_path = path
                    break
        
        if image_folder_path is None:
            raise FileNotFoundError(f"无法找到 GTSRB 的图像文件夹")
        
        print(f"🔍 扫描 GTSRB 图像文件夹: {image_folder_path}")
        
        # 查找所有类别文件夹
        classes = []
        for item in os.listdir(image_folder_path):
            item_path = os.path.join(image_folder_path, item)
            if os.path.isdir(item_path):
                classes.append(item)
        
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"📁 找到 {len(classes)} 个类别: {classes[:5]}{'...' if len(classes) > 5 else ''}")
        
        # 收集所有图像文件
        for class_name in classes:
            class_dir = os.path.join(image_folder_path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            for img_name in img_files:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
        
        print(f"🖼️  找到 {len(self.samples)} 个图像文件")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class STL10Dataset(Dataset):
    """STL-10 数据集加载器"""
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # STL-10 可能的路径结构
        possible_paths = [
            os.path.join(data_path, "test"),
            os.path.join(data_path, "stl10"),
            os.path.join(data_path, "images"),
            data_path
        ]
        
        image_folder_path = None
        for path in possible_paths:
            if os.path.exists(path):
                # 检查是否包含子目录
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    image_folder_path = path
                    break
        
        if image_folder_path is None:
            raise FileNotFoundError(f"无法找到 STL-10 的图像文件夹")
        
        print(f"🔍 扫描 STL-10 图像文件夹: {image_folder_path}")
        
        # 查找所有类别文件夹
        classes = []
        for item in os.listdir(image_folder_path):
            item_path = os.path.join(image_folder_path, item)
            if os.path.isdir(item_path):
                classes.append(item)
        
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"📁 找到 {len(classes)} 个类别: {classes[:5]}{'...' if len(classes) > 5 else ''}")
        
        # 收集所有图像文件
        for class_name in classes:
            class_dir = os.path.join(image_folder_path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            for img_name in img_files:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
        
        print(f"🖼️  找到 {len(self.samples)} 个图像文件")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class SUN397Dataset(Dataset):
    """SUN397 数据集加载器"""
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # SUN397 可能的路径结构
        possible_paths = [
            os.path.join(data_path, "test"),
            os.path.join(data_path, "sun397"),
            os.path.join(data_path, "images"),
            data_path
        ]
        
        image_folder_path = None
        for path in possible_paths:
            if os.path.exists(path):
                # 检查是否包含子目录
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    image_folder_path = path
                    break
        
        if image_folder_path is None:
            raise FileNotFoundError(f"无法找到 SUN397 的图像文件夹")
        
        print(f"🔍 扫描 SUN397 图像文件夹: {image_folder_path}")
        
        # 查找所有类别文件夹
        classes = []
        for item in os.listdir(image_folder_path):
            item_path = os.path.join(image_folder_path, item)
            if os.path.isdir(item_path):
                classes.append(item)
        
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"📁 找到 {len(classes)} 个类别: {classes[:5]}{'...' if len(classes) > 5 else ''}")
        
        # 收集所有图像文件
        for class_name in classes:
            class_dir = os.path.join(image_folder_path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            for img_name in img_files:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
        
        print(f"🖼️  找到 {len(self.samples)} 个图像文件")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class FGVCAircraftDataset(Dataset):
    """FGVC Aircraft 数据集加载器"""
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        # FGVC Aircraft数据集结构
        # data/fgvc_aircraft/fgvc-aircraft-2013b/data/images_variant_test.txt
        # data/fgvc_aircraft/fgvc-aircraft-2013b/data/images/
        
        # 查找数据集目录
        dataset_dir = None
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path) and "fgvc-aircraft" in item.lower():
                dataset_dir = item_path
                break
        
        if dataset_dir is None:
            raise FileNotFoundError(f"未找到FGVC Aircraft数据集目录在 {data_path}")
        
        # 加载类别名称
        variants_file = os.path.join(dataset_dir, "data", "variants.txt")
        if os.path.exists(variants_file):
            with open(variants_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✅ 从 {variants_file} 加载了 {len(self.class_names)} 个类别")
        
        # 加载测试集标签
        test_file = os.path.join(dataset_dir, "data", "images_variant_test.txt")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"未找到测试集标签文件: {test_file}")
        
        # 读取测试集标签
        with open(test_file, 'r') as f:
            lines = f.readlines()
        
        # 解析每行：图像ID 类别名称
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_id = parts[0]
                variant = " ".join(parts[1:])  # 处理类别名称中可能包含空格的情况
                
                # 查找类别索引
                if variant in self.class_names:
                    class_idx = self.class_names.index(variant)
                    
                    # 构建图像路径
                    image_path = os.path.join(dataset_dir, "data", "images", f"{image_id}.jpg")
                    
                    if os.path.exists(image_path):
                        self.samples.append((image_path, class_idx))
        
        print(f"✅ 成功加载 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @property
    def class_to_idx(self):
        """为了兼容性，提供class_to_idx属性"""
        return {name: i for i, name in enumerate(self.class_names)}


def load_dataset(dataset_name: str, data_root: str, limit: int = None) -> Tuple[Any, List[str]]:
    """从本地 data 文件夹加载数据集"""
    # 标准图像预处理
    # CLIP ViT-L/14模型期望输入图像为224x224，这是它在预训练时使用的尺寸
    # 大多数数据集需要resize到这个尺寸以匹配模型的期望输入
    # 注意：MNIST使用单独的transform处理，因为它需要特殊的resize操作
    standard_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    
    dataset_path = os.path.join(data_root, dataset_name)
    
    try:
        if dataset_name == "imagenet1k":
            # ImageNet1k数据集需要特殊处理
            class ImageNet1kDataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    self.class_names = []
                    
                    # 加载meta.mat文件获取类别信息
                    meta_path = os.path.join(data_path, "ILSVRC2012_devkit_t12", "data", "meta.mat")
                    if not os.path.exists(meta_path):
                        raise FileNotFoundError(f"ImageNet1k meta文件不存在: {meta_path}")
                    
                    import scipy.io
                    meta = scipy.io.loadmat(meta_path)
                    synsets = meta['synsets']
                    
                    # 获取所有类别信息
                    for i in range(synsets.shape[0]):
                        synset = synsets[i][0]
                        # 获取WNID和类别名称
                        wnid = str(synset[1][0])
                        class_name = str(synset[2][0])
                        self.class_names.append(class_name)
                    
                    # 加载验证集标签
                    gt_path = os.path.join(data_path, "ILSVRC2012_devkit_t12", "data", "ILSVRC2012_validation_ground_truth.txt")
                    if not os.path.exists(gt_path):
                        raise FileNotFoundError(f"ImageNet1k验证集标签文件不存在: {gt_path}")
                    
                    with open(gt_path, 'r') as f:
                        labels = [int(line.strip()) for line in f.readlines()]
                    
                    # 获取所有验证图像
                    img_files = [f for f in os.listdir(data_path) if f.startswith("ILSVRC2012_val_") and f.endswith(".JPEG")]
                    img_files.sort()
                    
                    # 创建样本列表
                    for i, img_file in enumerate(img_files):
                        if i < len(labels):
                            label_idx = labels[i] - 1  # MATLAB索引从1开始，转换为0开始
                            if 0 <= label_idx < len(self.class_names):
                                img_path = os.path.join(data_path, img_file)
                                self.samples.append((img_path, label_idx))
                    
                    print(f"✅ 成功加载 {len(self.samples)} 个ImageNet1k验证样本")
                    print(f"✅ 加载了 {len(self.class_names)} 个类别")
                    
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    
                    # 加载图像
                    img = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = ImageNet1kDataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "cifar10":
            dataset = CIFAR10Dataset(dataset_path, train=False, transform=standard_transform)
            # 使用实际的CIFAR-10类别名称
            class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
            
        elif dataset_name == "cifar100":
            dataset = CIFAR100Dataset(dataset_path, train=False, transform=standard_transform)
            # 从CIFAR-100元数据文件加载实际的类别名称
            cifar_path = os.path.join(dataset_path, "cifar-100-python")
            meta_file = os.path.join(cifar_path, "meta")
            
            try:
                with open(meta_file, 'rb') as f:
                    meta = pickle.load(f, encoding='bytes')
                    # 获取细粒度类别名称
                    class_names = [label.decode('utf-8') for label in meta[b'fine_label_names']]
                    print(f"✅ 从元数据文件加载了 {len(class_names)} 个CIFAR-100类别")
            except Exception as e:
                print(f"⚠️  无法加载CIFAR-100元数据文件，使用默认类别名称: {e}")
                # 使用默认的CIFAR-100类别名称
                class_names = [
                    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
                    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
                    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
                    "cloud", "cockroach", "couch", "crab", "crocodile", "cruise_ship", "cup",
                    "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
                    "house", "kangaroo", "computer_keyboard", "lamp", "lawn_mower", "leopard", "lion",
                    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
                    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
                    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
                    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
                    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
                    "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger",
                    "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree",
                    "wolf", "woman", "worm", "zookeeper"
                ]
            
        elif dataset_name == "mnist":
            # MNIST需要特殊的resize处理，因为原始图像很小(28x28)
            mnist_transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                   (0.26862954, 0.26130258, 0.27577711))
            ])
            try:
                dataset = tv_datasets.MNIST(
                    root=dataset_path,
                    train=False,
                    transform=mnist_transform,
                    download=False,
                )
            except RuntimeError as exc:
                raw_candidates = [
                    os.path.join(dataset_path, "MNIST", "raw"),
                    os.path.join(dataset_path, "raw"),
                ]
                for raw_root in raw_candidates:
                    if os.path.exists(raw_root):
                        dataset = MNISTRawFallbackDataset(raw_root, split="test", transform=mnist_transform)
                        break
                else:
                    raise RuntimeError(
                        "未找到处理后的 MNIST 文件，且原始数据缺失。请先下载 MNIST 数据集。"
                    ) from exc
            class_names = DATASET_CLASSES["mnist"]
            
        elif dataset_name == "caltech101":
            dataset = Caltech101Dataset(dataset_path, transform=standard_transform)
            class_names = list(dataset.class_to_idx.keys())
            
        elif dataset_name == "stanford_cars":
            dataset = StanfordCarsDataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.class_names
            
        elif dataset_name == "flowers102":
            dataset = Flowers102Dataset(dataset_path, transform=standard_transform)
            class_names = dataset.get_classes()
            
        elif dataset_name == "food101":
            dataset = Food101Dataset(dataset_path, transform=standard_transform)
            class_names = dataset.get_classes()
            
        elif dataset_name == "pets":
            dataset = PetsDataset(dataset_path, transform=standard_transform)
            class_names = dataset.class_names
            
        elif dataset_name == "dtd":
            # DTD数据集的图像在dtd子目录的images文件夹中
            dtd_path = os.path.join(dataset_path, "dtd", "images")
            dataset = DTDataset(dtd_path, transform=standard_transform)
            class_names = list(dataset.class_to_idx.keys())
            
        elif dataset_name == "gtsrb":
            # GTSRB数据集需要特殊处理，图像和标签在不同的位置
            class GTSRBDataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    
                    # GTSRB类别名称
                    self.class_names = [
                        "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
                        "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
                        "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
                        "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
                        "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution",
                        "Dangerous curve to the left", "Dangerous curve to the right", "Double curve", "Bumpy road",
                        "Slippery road", "Road narrows on the right", "Road work", "Traffic signals",
                        "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
                        "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
                        "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
                        "Keep left", "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
                    ]
                    
                    # 图像路径
                    images_path = os.path.join(data_path, "gtsrb", "GTSRB", "Final_Test", "Images")
                    # 标签文件路径
                    labels_path = os.path.join(data_path, "gtsrb", "GT-final_test.csv")
                    
                    # 读取标签文件
                    import pandas as pd
                    df = pd.read_csv(labels_path, sep=';')
                    
                    # 创建样本列表
                    for idx, row in df.iterrows():
                        if limit is not None and len(self.samples) >= limit:
                            break
                        
                        filename = row['Filename']
                        class_id = int(row['ClassId'])
                        
                        # 确保类别ID在有效范围内
                        if class_id < len(self.class_names):
                            img_path = os.path.join(images_path, filename)
                            if os.path.exists(img_path):
                                self.samples.append((img_path, class_id))
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    
                    # 加载图像
                    img = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = GTSRBDataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "stl10":
            # STL10数据集需要特殊处理，使用二进制文件
            class STL10Dataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    
                    # STL10类别名称
                    class_names_file = os.path.join(data_path, "stl10_binary", "class_names.txt")
                    with open(class_names_file, 'r') as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                    
                    # 读取二进制数据文件
                    test_x_file = os.path.join(data_path, "stl10_binary", "test_X.bin")
                    test_y_file = os.path.join(data_path, "stl10_binary", "test_y.bin")
                    
                    # 读取图像数据
                    with open(test_x_file, 'rb') as f:
                        # STL10图像是96x96x3的RGB图像
                        # 每个图像有96*96*3 = 27648字节
                        raw_data = f.read()
                        num_images = len(raw_data) // (96 * 96 * 3)
                        
                        # 将原始数据转换为图像数组
                        import numpy as np
                        images = np.frombuffer(raw_data, dtype=np.uint8)
                        images = images.reshape(num_images, 3, 96, 96)  # 通道优先
                        images = images.transpose(0, 2, 3, 1)  # 转换为图像优先
                    
                    # 读取标签
                    with open(test_y_file, 'rb') as f:
                        labels = np.frombuffer(f.read(), dtype=np.uint8)
                    
                    # 创建样本列表
                    for i in range(min(num_images, len(labels))):
                        if limit is not None and len(self.samples) >= limit:
                            break
                        
                        # STL-10标签从1开始，需要转换为0开始的索引
                        label_idx = int(labels[i] - 1)  # 确保转换为Python int类型
                        if 0 <= label_idx < len(self.class_names):
                            self.samples.append((images[i], label_idx))
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    image_array, label = self.samples[idx]
                    
                    # 从numpy数组创建图像
                    img = Image.fromarray(image_array, mode='RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = STL10Dataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "sun397":
            dataset = SUN397Dataset(dataset_path, transform=standard_transform)
            class_names = list(dataset.class_to_idx.keys())
            
        elif dataset_name == "fer2013":
            # FER2013 数据集加载器
            class FER2013Dataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    self.class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
                    
                    # 加载CSV文件
                    csv_path = os.path.join(data_path, "fer2013.csv")
                    if not os.path.exists(csv_path):
                        raise FileNotFoundError(f"FER2013 CSV文件不存在: {csv_path}")
                    
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    
                    # 只使用测试集
                    df = df[df['Usage'] == 'PublicTest']
                    
                    # 处理每一行数据
                    for idx, row in df.iterrows():
                        if limit is not None and len(self.samples) >= limit:
                            break
                        
                        emotion = int(row['emotion'])
                        pixels = row['pixels']
                        
                        # 将像素字符串转换为numpy数组
                        pixel_array = np.array([int(p) for p in pixels.split()]).reshape(48, 48)
                        
                        # 确保emotion在有效范围内
                        emotion = min(emotion, len(self.class_names) - 1)
                        
                        self.samples.append((pixel_array, emotion))
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    pixel_array, label = self.samples[idx]
                    
                    # 从像素数组创建图像
                    img = Image.fromarray(pixel_array.astype(np.uint8), mode='L').convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
            
            dataset = FER2013Dataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.class_names
            
        elif dataset_name == "fgvc_aircraft":
            dataset = FGVCAircraftDataset(dataset_path, transform=standard_transform)
            class_names = dataset.class_names
            
        elif dataset_name == "resisc45":
            # Resisc45数据集需要特殊处理，使用Parquet文件
            class Resisc45Dataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    
                    # Resisc45类别名称
                    self.class_names = [
                        "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
                        "bridge", "chaparral", "church", "circular_farmland", "cloud",
                        "commercial_area", "dense_residential", "desert", "forest", "freeway",
                        "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
                        "island", "lake", "meadow", "medium_residential", "mobile_home_park",
                        "mountain", "overpass", "palace", "parking_lot", "railway",
                        "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
                        "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
                        "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"
                    ]
                    
                    # 测试集Parquet文件路径
                    test_file = os.path.join(data_path, "data", "test-00000-of-00001.parquet")
                    
                    if not os.path.exists(test_file):
                        raise FileNotFoundError(f"Resisc45测试集文件不存在: {test_file}")
                    
                    # 读取Parquet文件
                    import pandas as pd
                    df = pd.read_parquet(test_file)
                    
                    # 创建样本列表
                    for idx, row in df.iterrows():
                        if limit is not None and len(self.samples) >= limit:
                            break
                        
                        image = row['image']
                        label = int(row['label'])
                        
                        # 确保标签在有效范围内
                        if 0 <= label < len(self.class_names):
                            self.samples.append((image, label))
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    image_data, label = self.samples[idx]
                    
                    # 处理字典格式的图像数据
                    if isinstance(image_data, dict):
                        # 从字典中提取图像数据
                        if 'bytes' in image_data:
                            # 从字节数据创建图像
                            from io import BytesIO
                            image = Image.open(BytesIO(image_data['bytes']))
                        elif 'path' in image_data:
                            # 从路径加载图像
                            image = Image.open(image_data['path'])
                        else:
                            raise ValueError(f"无法识别的图像数据格式: {image_data.keys()}")
                    else:
                        # 确保图像是PIL Image
                        if not isinstance(image_data, Image.Image):
                            image = Image.fromarray(image_data)
                        else:
                            image = image_data
                    
                    # 转换为RGB
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    return image, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = Resisc45Dataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "imagenet_v2":
            # ImageNet V2数据集需要特殊处理
            class ImageNetV2Dataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    self.class_names = []
                    
                    # ImageNet V2数据集路径
                    v2_path = os.path.join(data_path, "imagenetv2-matched-frequency-format-val")
                    if not os.path.exists(v2_path):
                        raise FileNotFoundError(f"ImageNet V2数据集路径不存在: {v2_path}")
                    
                    # 获取所有类别文件夹
                    class_dirs = [d for d in os.listdir(v2_path)
                                 if os.path.isdir(os.path.join(v2_path, d))]
                    class_dirs.sort(key=int)  # 按数字排序
                    
                    # 创建类别名称列表，使用简单的编号
                    num_classes = len(class_dirs)
                    self.class_names = [f"imagenet_v2_class_{i}" for i in range(num_classes)]
                    
                    # 创建样本列表
                    for class_dir in class_dirs:
                        # ImageNet V2的类别文件夹名称是数字，直接作为索引
                        class_idx = int(class_dir)  # 已经是0开始的索引
                        if class_idx < 0 or class_idx >= len(self.class_names):
                            continue  # 跳过超出范围的类别
                        
                        class_path = os.path.join(v2_path, class_dir)
                        img_files = [f for f in os.listdir(class_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                        
                        for img_file in img_files:
                            img_path = os.path.join(class_path, img_file)
                            self.samples.append((img_path, class_idx))
                    
                    print(f"✅ 成功加载 {len(self.samples)} 个ImageNet V2样本")
                    print(f"✅ 加载了 {len(self.class_names)} 个类别")
                    
                    # 限制样本数量
                    if limit is not None and limit < len(self.samples):
                        self.samples = self.samples[:limit]
                        print(f"⚠️  限制样本数量为: {limit}")
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    
                    # 加载图像
                    img = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = ImageNetV2Dataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "imagenet_ske":
            # ImageNet Sketch数据集需要特殊处理
            class ImageNetSkeDataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    self.class_names = []
                    
                    # 尝试加载类别信息
                    classes_file = os.path.join(data_path, "classes.py")
                    if os.path.exists(classes_file):
                        # 执行classes.py文件以获取类别信息
                        import sys
                        sys.path.append(os.path.dirname(classes_file))
                        import classes as imagenet_classes
                        self.class_names = list(imagenet_classes.IMAGENET2012_CLASSES.values())
                        # 创建一个从WNID到索引的映射
                        wnid_to_idx = {wnid: i for i, wnid in enumerate(imagenet_classes.IMAGENET2012_CLASSES.keys())}
                    else:
                        # 如果找不到classes.py，尝试使用ImageNet1k的meta文件
                        meta_path = os.path.join(data_root, "imagenet1k", "ILSVRC2012_devkit_t12", "data", "meta.mat")
                        if os.path.exists(meta_path):
                            import scipy.io
                            meta = scipy.io.loadmat(meta_path)
                            synsets = meta['synsets']
                            
                            # 获取所有类别信息
                            for i in range(synsets.shape[0]):
                                synset = synsets[i][0]
                                # 获取WNID和类别名称
                                wnid = str(synset[1][0])
                                class_name = str(synset[2][0])
                                self.class_names.append(class_name)
                            
                            # 创建一个从WNID到索引的映射
                            wnid_to_idx = {}
                            for i in range(synsets.shape[0]):
                                synset = synsets[i][0]
                                wnid = str(synset[1][0])
                                wnid_to_idx[wnid] = i
                        else:
                            # 如果都找不到，使用简化的类别名称
                            self.class_names = [f"imagenet_class_{i}" for i in range(1000)]
                            wnid_to_idx = {f"n{i:08d}": i for i in range(1000)}

                    sketch_path = os.path.join(data_path, "imagenet-sketch")
                    if not os.path.isdir(sketch_path):
                        alt_path = os.path.join(data_path, "data", "sketch")
                        if os.path.isdir(alt_path):
                            sketch_path = alt_path
                        else:
                            raise FileNotFoundError(f"ImageNet Sketch数据集路径不存在: {sketch_path}")

                    class_dirs = [d for d in os.listdir(sketch_path)
                                  if os.path.isdir(os.path.join(sketch_path, d))]
                    class_dirs.sort()

                    # 创建样本列表
                    for class_dir in class_dirs:
                        if class_dir not in wnid_to_idx:
                            continue  # 跳过不在类别列表中的文件夹
                        
                        class_idx = wnid_to_idx[class_dir]
                        class_path = os.path.join(sketch_path, class_dir)
                        img_files = [f for f in os.listdir(class_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                        
                        for img_file in img_files:
                            img_path = os.path.join(class_path, img_file)
                            self.samples.append((img_path, class_idx))
                    
                    print(f"✅ 成功加载 {len(self.samples)} 个ImageNet Sketch样本")
                    print(f"✅ 加载了 {len(self.class_names)} 个类别")
                    
                    # 限制样本数量
                    if limit is not None and limit < len(self.samples):
                        self.samples = self.samples[:limit]
                        print(f"⚠️  限制样本数量为: {limit}")
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    
                    # 加载图像
                    img = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = ImageNetSkeDataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "imagenet_ren":
            # ImageNet Renditions数据集需要特殊处理
            class ImageNetRenDataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    self.class_names = []
                    
                    # 解析ImageNet1k的类别元数据，构建WNID到可读名称的映射
                    meta_path = os.path.join(data_root, "imagenet1k", "ILSVRC2012_devkit_t12", "data", "meta.mat")
                    wnid_to_name: Dict[str, str] = {}
                    if os.path.exists(meta_path):
                        import scipy.io
                        meta = scipy.io.loadmat(meta_path)
                        synsets = meta["synsets"]
                        for i in range(synsets.shape[0]):
                            synset = synsets[i][0]
                            wnid = str(synset[1][0])
                            class_name = str(synset[2][0])
                            wnid_to_name[wnid] = class_name

                    ren_path = os.path.join(data_path, "imagenet-r")
                    if not os.path.isdir(ren_path):
                        alt_path = os.path.join(data_path, "data", "imagenet-r")
                        if os.path.isdir(alt_path):
                            ren_path = alt_path
                        else:
                            raise FileNotFoundError(f"ImageNet Renditions数据集路径不存在: {ren_path}")

                    class_dirs = [d for d in os.listdir(ren_path)
                                  if os.path.isdir(os.path.join(ren_path, d))]
                    class_dirs.sort()

                    wnid_to_idx = {}
                    self.class_names = []
                    for idx, class_dir in enumerate(class_dirs):
                        wnid_to_idx[class_dir] = idx
                        self.class_names.append(wnid_to_name.get(class_dir, class_dir))

                    # 创建样本列表
                    for class_dir in class_dirs:
                        class_idx = wnid_to_idx[class_dir]
                        class_path = os.path.join(ren_path, class_dir)
                        img_files = [f for f in os.listdir(class_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                        
                        for img_file in img_files:
                            img_path = os.path.join(class_path, img_file)
                            self.samples.append((img_path, class_idx))
                    
                    print(f"✅ 成功加载 {len(self.samples)} 个ImageNet Renditions样本")
                    print(f"✅ 加载了 {len(self.class_names)} 个类别")
                    
                    # 限制样本数量
                    if limit is not None and limit < len(self.samples):
                        self.samples = self.samples[:limit]
                        print(f"⚠️  限制样本数量为: {limit}")
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    
                    # 加载图像
                    img = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = ImageNetRenDataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "imagenet_adv":
            # ImageNet-A (Adversarial)数据集需要特殊处理
            class ImageNetAdvDataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    self.class_names = []
                    
                    label_to_idx: Dict[str, int] = {}
                    unique_labels: List[str] = []

                    # 读取samples.json文件
                    samples_file = os.path.join(data_path, "samples.json")
                    if not os.path.exists(samples_file):
                        raise FileNotFoundError(f"ImageNet-A samples.json文件不存在: {samples_file}")
                    
                    import json
                    with open(samples_file, 'r') as f:
                        data = json.load(f)
                        samples = data.get('samples', [])
                        
                        # 处理每个样本
                        for sample in samples:
                            # 获取图像路径
                            filepath = sample.get('filepath', '')
                            if not filepath:
                                continue
                                
                            # 获取真实标签
                            ground_truth = sample.get('ground_truth', {})
                            label_str = ground_truth.get('label', '')
                            label_clean = label_str.strip()
                            if not label_clean:
                                continue

                            normalized_label = label_clean.lower()
                            class_idx = label_to_idx.get(normalized_label)
                            if class_idx is None:
                                class_idx = len(unique_labels)
                                label_to_idx[normalized_label] = class_idx
                                unique_labels.append(label_clean)

                            # 构建完整的图像路径
                            img_path = os.path.join(data_path, filepath)
                            if os.path.exists(img_path):
                                self.samples.append((img_path, class_idx))
                    
                    self.class_names = unique_labels
                    
                    print(f"✅ 成功加载 {len(self.samples)} 个ImageNet-A样本")
                    print(f"✅ 加载了 {len(self.class_names)} 个类别")
                    
                    # 限制样本数量
                    if limit is not None and limit < len(self.samples):
                        self.samples = self.samples[:limit]
                        print(f"⚠️  限制样本数量为: {limit}")
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    
                    # 加载图像
                    img = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = ImageNetAdvDataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "voc2007":
            # PASCAL VOC 2007数据集需要特殊处理，使用Parquet格式
            class VOC2007Dataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    self.class_names = [
                        "aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow",
                        "diningtable", "dog", "horse", "motorbike", "person",
                        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
                    ]
                    
                    # VOC2007数据集路径
                    parquet_file = os.path.join(data_path, "data", "train-00000-of-00001.parquet")
                    if not os.path.exists(parquet_file):
                        raise FileNotFoundError(f"PASCAL VOC 2007 Parquet文件不存在: {parquet_file}")
                    
                    print(f"🔄 从Parquet文件加载PASCAL VOC 2007数据集: {parquet_file}")
                    
                    # 读取Parquet文件
                    import pandas as pd
                    df = pd.read_parquet(parquet_file)
                    
                    print(f"📊 数据集大小: {len(df)}")
                    print(f"📋 数据集列: {list(df.columns)}")
                    
                    # 处理每个样本
                    for idx, row in df.iterrows():
                        if limit is not None and len(self.samples) >= limit:
                            break
                        
                        try:
                            # 获取图像数据
                            image_data = row.get('image')
                            if image_data is None:
                                continue
                            
                            # 获取标签
                            label = row.get('label')
                            if label is None:
                                continue
                            
                            # 确保标签在有效范围内
                            if isinstance(label, (int, np.integer)) and 0 <= label < len(self.class_names):
                                self.samples.append((image_data, int(label)))
                            elif isinstance(label, str) and label in self.class_names:
                                class_idx = self.class_names.index(label)
                                self.samples.append((image_data, class_idx))
                            else:
                                # 尝试将标签转换为整数
                                try:
                                    label_int = int(label)
                                    if 0 <= label_int < len(self.class_names):
                                        self.samples.append((image_data, label_int))
                                except (ValueError, TypeError):
                                    print(f"⚠️  无法处理的标签: {label} (类型: {type(label)})")
                                    continue
                        except Exception as e:
                            print(f"⚠️  处理样本 {idx} 时出错: {e}")
                            continue
                    
                    print(f"✅ 成功加载 {len(self.samples)} 个PASCAL VOC 2007样本")
                    print(f"✅ 加载了 {len(self.class_names)} 个类别")
                    
                    # 限制样本数量
                    if limit is not None and limit < len(self.samples):
                        self.samples = self.samples[:limit]
                        print(f"⚠️  限制样本数量为: {limit}")
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    image_data, label = self.samples[idx]
                    
                    # 处理不同类型的图像数据
                    if isinstance(image_data, dict):
                        # 从字典中提取图像数据
                        if 'bytes' in image_data:
                            # 从字节数据创建图像
                            from io import BytesIO
                            image = Image.open(BytesIO(image_data['bytes']))
                        elif 'path' in image_data:
                            # 从路径加载图像
                            image = Image.open(image_data['path'])
                        else:
                            raise ValueError(f"无法识别的图像数据格式: {image_data.keys()}")
                    else:
                        # 确保图像是PIL Image
                        if not isinstance(image_data, Image.Image):
                            image = Image.fromarray(image_data)
                        else:
                            image = image_data
                    
                    # 转换为RGB
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    return image, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = VOC2007Dataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "rendered_sst2":
            # Rendered SST2数据集需要特殊处理
            class RenderedSST2Dataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    self.class_names = ["negative", "positive"]
                    
                    # Rendered SST2数据集路径
                    rendered_sst2_path = os.path.join(data_path, "rendered-sst2")
                    if not os.path.exists(rendered_sst2_path):
                        alt_path = os.path.join(data_path, "Rendered-SST2")
                        if os.path.exists(alt_path):
                            rendered_sst2_path = alt_path
                        elif os.path.exists(os.path.join(data_path, "test")):
                            rendered_sst2_path = data_path
                        else:
                            raise FileNotFoundError(f"Rendered SST2数据集路径不存在: {rendered_sst2_path}")
                    
                    # 测试集路径
                    test_path = os.path.join(rendered_sst2_path, "test")
                    if not os.path.exists(test_path):
                        raise FileNotFoundError(f"Rendered SST2测试集路径不存在: {test_path}")
                    
                    # 处理negative类别
                    negative_path = os.path.join(test_path, "negative")
                    if os.path.exists(negative_path):
                        img_files = [f for f in os.listdir(negative_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                        for img_file in img_files:
                            img_path = os.path.join(negative_path, img_file)
                            self.samples.append((img_path, 0))  # negative的标签为0
                    
                    # 处理positive类别
                    positive_path = os.path.join(test_path, "positive")
                    if os.path.exists(positive_path):
                        img_files = [f for f in os.listdir(positive_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                        for img_file in img_files:
                            img_path = os.path.join(positive_path, img_file)
                            self.samples.append((img_path, 1))  # positive的标签为1
                    
                    print(f"✅ 成功加载 {len(self.samples)} 个Rendered SST2样本")
                    print(f"✅ 加载了 {len(self.class_names)} 个类别")
                    
                    if limit is not None:
                        print(f"⚠️  limit 参数在 Rendered SST2 中仅用于日志，实际截断将由评估阶段处理。")
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    
                    # 加载图像
                    img = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = RenderedSST2Dataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        elif dataset_name == "country211":
            # Country-211数据集需要特殊处理
            class Country211Dataset(Dataset):
                def __init__(self, data_path: str, transform=None, limit=None):
                    self.transform = transform
                    self.samples = []
                    self.class_names = []
                    
                    # Country-211数据集路径
                    country211_path = os.path.join(data_path, "country211", "test")
                    if not os.path.exists(country211_path):
                        raise FileNotFoundError(f"Country-211数据集路径不存在: {country211_path}")
                    
                    print(f"🔍 扫描 Country-211 数据集: {country211_path}")
                    
                    # 获取所有国家代码目录
                    country_dirs = [d for d in os.listdir(country211_path)
                                  if os.path.isdir(os.path.join(country211_path, d))]
                    country_dirs.sort()
                    
                    # 创建国家代码到国家名称的映射
                    # 这里使用国家代码作为类别名称，因为数据集没有提供完整的国家名称
                    self.class_names = country_dirs
                    
                    # 创建国家代码到索引的映射
                    country_to_idx = {country: i for i, country in enumerate(country_dirs)}
                    
                    # 收集所有图像文件
                    for country_code in country_dirs:
                        country_idx = country_to_idx[country_code]
                        country_path = os.path.join(country211_path, country_code)
                        
                        img_files = [f for f in os.listdir(country_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                        
                        for img_file in img_files:
                            img_path = os.path.join(country_path, img_file)
                            self.samples.append((img_path, country_idx))
                    
                    print(f"✅ 成功加载 {len(self.samples)} 个Country-211样本")
                    print(f"✅ 加载了 {len(self.class_names)} 个国家/地区")
                    
                    # 限制样本数量
                    if limit is not None and limit < len(self.samples):
                        self.samples = self.samples[:limit]
                        print(f"⚠️  限制样本数量为: {limit}")
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    
                    # 加载图像
                    img = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    return img, label
                
                def get_classes(self):
                    return self.class_names
            
            dataset = Country211Dataset(dataset_path, transform=standard_transform, limit=limit)
            class_names = dataset.get_classes()
            
        else:
            # 尝试作为通用图像文件夹数据集加载
            if os.path.exists(dataset_path):
                # 检查是否有子目录结构（每个子目录代表一个类别）
                subdirs = [d for d in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, d))]
                
                if subdirs:
                    # 使用一个通用的图像文件夹数据集
                    class GenericImageFolderDataset(Dataset):
                        def __init__(self, root_path: str, transform=None):
                            self.transform = transform
                            self.samples = []
                            self.class_to_idx = {}
                            
                            # 查找所有类别文件夹
                            classes = []
                            for item in os.listdir(root_path):
                                item_path = os.path.join(root_path, item)
                                if os.path.isdir(item_path):
                                    classes.append(item)
                            
                            classes.sort()
                            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
                            
                            # 收集所有图像文件
                            for class_name in classes:
                                class_dir = os.path.join(root_path, class_name)
                                class_idx = self.class_to_idx[class_name]
                                
                                for img_name in os.listdir(class_dir):
                                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                        img_path = os.path.join(class_dir, img_name)
                                        self.samples.append((img_path, class_idx))
                        
                        def __len__(self):
                            return len(self.samples)
                        
                        def __getitem__(self, idx):
                            img_path, label = self.samples[idx]
                            
                            # 加载图像
                            img = Image.open(img_path).convert('RGB')
                            
                            if self.transform:
                                img = self.transform(img)
                            
                            return img, label
                    
                    dataset = GenericImageFolderDataset(dataset_path, transform=standard_transform)
                    class_names = list(dataset.class_to_idx.keys())
                else:
                    raise FileNotFoundError(f"数据集 {dataset_name} 没有有效的子目录结构")
            else:
                raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
                
        return dataset, class_names
        
    except Exception as e:
        print(f"❌ 加载数据集 {dataset_name} 失败: {str(e)}")
        print(f"📁 尝试的路径: {dataset_path}")
        raise


def load_clip_model(model_root: str, device: torch.device, dtype: torch.dtype):
    """从本地加载 CLIP 模型"""
    try:
        from transformers import CLIPModel, CLIPProcessor
        
        model_path = os.path.join(model_root, "clip")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CLIP 模型路径不存在: {model_path}")
        
        print(f"🤖 从 {model_path} 加载 CLIP 模型...")
        
        model = CLIPModel.from_pretrained(model_path)
        if dtype is not None:
            model = model.to(dtype=dtype)
        model = model.to(device)
        model.eval()
        
        processor = CLIPProcessor.from_pretrained(model_path)
        
        print("✅ CLIP 模型加载成功")
        return model, processor
        
    except Exception as e:
        print(f"❌ 加载 CLIP 模型失败: {str(e)}")
        raise


def create_text_prompts(class_names: List[str], dataset_name: Optional[str] = None) -> List[str]:
    """创建文本提示"""
    prompts = []
    if dataset_name == "mnist":
        for class_name in class_names:
            prompts.append(f"the number {class_name}")
    else:
        for class_name in class_names:
            formatted_name = class_name.replace('_', ' ').strip()
            prompts.append(f"a photo of a {formatted_name}")
    return prompts


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 5,
    num_classes: int = None
) -> Dict[str, float]:
    """计算评估指标"""
    # Top-1 准确率
    top1_acc = (predictions[:, 0] == targets).float().mean().item()
    
    # Top-k 准确率
    topk_acc = 0.0
    if top_k > 1:
        correct = torch.zeros_like(targets, dtype=torch.bool)
        # 确保k不超过类别数量和预测维度
        max_k = min(top_k, predictions.size(1), num_classes)
        for k in range(max_k):
            correct |= (predictions[:, k] == targets)
        topk_acc = correct.float().mean().item()
    
    # 计算混淆矩阵（仅适用于小数据集）
    if num_classes is None:
        num_classes = max(targets.max().item() + 1, predictions.size(1))
    # 确保索引不越界
    t = targets.long()
    p = predictions[:, 0].long()
    # 限制预测值在有效范围内
    p = torch.clamp(p, 0, num_classes - 1)
    t = torch.clamp(t, 0, num_classes - 1)
    
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for ti, pi in zip(t, p):
        confusion_matrix[ti, pi] += 1
    
    return {
        "top1_accuracy": top1_acc,
        f"top{top_k}_accuracy": topk_acc,
        "confusion_matrix": confusion_matrix.tolist()
    }


def evaluate_dataset(
    model,
    processor,
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    limit: int = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """评估单个数据集"""
    print(f"\n🔄 开始评估数据集: {dataset_name}")
    
    try:
        # 加载数据集
        dataset, class_names = load_dataset(dataset_name, data_root, limit)
        
        # 限制样本数量
        if limit is not None and limit < len(dataset):
            indices = torch.randperm(len(dataset))[:limit]
            dataset = torch.utils.data.Subset(dataset, indices)
            print(f"⚠️  限制样本数量为: {limit}")
        
        print(f"📊 数据集大小: {len(dataset)}")
        print(f"🏷️  类别数量: {len(class_names)}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda"
        )
        
        # 编码文本特征
        print("📝 编码文本特征...")
        text_prompts = create_text_prompts(class_names, dataset_name=dataset_name)
        with torch.no_grad():
            text_inputs = processor(text=text_prompts, padding=True, return_tensors="pt").to(device)
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp().to(device)

        # 推理
        print("🔍 开始推理...")
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        to_pil = transforms.ToPILImage() if dataset_name == "mnist" else None
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"评估 {dataset_name}")):
            with torch.no_grad():
                # 编码图像特征
                if dataset_name == "mnist":
                    if isinstance(images, torch.Tensor):
                        images_pil = [to_pil(img.cpu()) for img in images]
                    else:
                        images_pil = images
                    image_inputs = processor(images=images_pil, return_tensors="pt")
                    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                    image_features = model.get_image_features(**image_inputs)
                else:
                    # 其他数据集的图像已经预处理过了，直接传入模型
                    image_features = model.get_image_features(pixel_values=images.to(device))
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # 计算相似度
                logits = (image_features @ text_features.T) * logit_scale
                similarities = logits.softmax(dim=-1)
                
                # 获取预测结果，确保k不超过类别数量
                actual_top_k = min(top_k, similarities.size(-1))
                _, predictions = similarities.topk(actual_top_k, dim=-1)
                
                all_predictions.append(predictions)
                all_targets.append(targets.to(device))
        
        inference_time = time.time() - start_time
        
        # 合并结果
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算指标
        metrics = calculate_metrics(all_predictions, all_targets, top_k, len(class_names))
        
        # 准备结果
        result = {
            "dataset": dataset_name,
            "num_samples": len(dataset),
            "num_classes": len(class_names),
            "inference_time": inference_time,
            "samples_per_second": len(dataset) / inference_time,
            **metrics
        }
        
        print(f"✅ {dataset_name} 评估完成")
        print(f"📊 Top-1 准确率: {metrics['top1_accuracy']:.4f}")
        print(f"📊 Top-{top_k} 准确率: {metrics[f'top{top_k}_accuracy']:.4f}")
        print(f"⏱️  推理时间: {inference_time:.2f} 秒")
        print(f"🚀 处理速度: {len(dataset) / inference_time:.2f} 样本/秒")
        
        return result
        
    except Exception as e:
        print(f"❌ 评估数据集 {dataset_name} 时出错: {str(e)}")
        return {
            "dataset": dataset_name,
            "error": str(e),
            "status": "failed"
        }


def save_results(results: List[Dict[str, Any]], output_dir: str) -> None:
    """保存评估结果"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"clip_evaluation_{timestamp}.json")
    
    # 添加摘要信息
    summary = {
        "timestamp": timestamp,
        "total_datasets": len(results),
        "successful_evaluations": len([r for r in results if "error" not in r]),
        "failed_evaluations": len([r for r in results if "error" in r]),
        "results": results
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到: {results_file}")
    
    # 打印摘要
    print("\n📋 评估摘要:")
    print(f"总数据集数: {summary['total_datasets']}")
    print(f"成功评估: {summary['successful_evaluations']}")
    print(f"失败评估: {summary['failed_evaluations']}")
    
    if summary['successful_evaluations'] > 0:
        print("\n🏆 最佳结果:")
        successful_results = [r for r in results if "error" not in r]
        successful_results.sort(key=lambda x: x.get("top1_accuracy", 0), reverse=True)
        
        for i, result in enumerate(successful_results[:5]):
            print(f"{i+1}. {result['dataset']}: {result.get('top1_accuracy', 0):.4f}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = setup_device(args.device)
    
    # 设置数据类型
    dtype = setup_dtype(args.dtype, device)
    
    # 创建输出目录
    create_directories(args.output_dir)
    
    # 解析数据集列表
    if args.datasets.lower() == "all":
        dataset_names = list(DATASET_CLASSES.keys())
    else:
        dataset_names = [name.strip() for name in args.datasets.split(",")]
    
    print(f"🎯 将评估以下数据集: {', '.join(dataset_names)}")
    
    # 加载 CLIP 模型
    print("\n🤖 加载 CLIP 模型...")
    try:
        model, processor = load_clip_model(args.model_root, device, dtype)
    except Exception as e:
        print(f"❌ 加载 CLIP 模型失败: {str(e)}")
        return
    
    # 评估所有数据集
    results = []
    for dataset_name in dataset_names:
        result = evaluate_dataset(
            model=model,
            processor=processor,
            dataset_name=dataset_name,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            limit=args.limit,
            top_k=args.top_k
        )
        results.append(result)
    
    # 保存结果
    save_results(results, args.output_dir)
    
    print("\n🎉 评估完成!")


if __name__ == "__main__":
    main()
