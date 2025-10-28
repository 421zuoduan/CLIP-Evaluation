# CLIP ViT-L/14 基准套件

本仓库用于评测 `openai/clip-vit-large-patch14` 在多种分类与鲁棒性数据集上的表现，包括 ImageNet 系列、CIFAR、Caltech-101、DTD、FER2013 等。脚本会统一下载数据并为后续实验提供标准化入口。

## 环境准备
- Python 3.10
- 推荐安装支持 CUDA 的 PyTorch（如需 GPU 评测）

安装依赖（可按需替换镜像源）：

```bash
pip install -r requirements.txt
```

## 数据集下载

所有数据默认存放在 `data/<dataset-name>`。下载脚本自动区分 torchvision 与 Hugging Face datasets。

下载全部基准：

```bash
python -m download.download_all --data-root ./data
```

仅下载部分数据集（逗号分隔 key）：

```bash
python -m download.download_all --datasets cifar10,imagenet1k,caltech101,dtd,fer2013 --data-root ./data
```

查看可用数据集 key：

```bash
python -m download.download_all --list
```

强制重新下载（忽略已有文件）：

```bash
python -m download.download_all --force
```

## 运行评测

数据下载完成后，执行主入口（需先实现 `main.py` 中的评测逻辑）：

```bash
python main.py --datasets all --data-root ./data --device cuda
```

限制评测范围或抽样验证：

```bash
python main.py --datasets cifar10,imagenet_v2,caltech101,dtd,fer2013 --limit 128
```

> 提示：脚本默认设置 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HUB_ENABLE_HF_TRANSFER=1`，确保在国内网络环境下通过 Hugging Face 镜像节点下载模型与数据集。若需要切换镜像，可在运行命令前自行导出环境变量覆盖默认值。

## 项目结构

- `download_single/`：各数据集的下载脚本。
- `model/`：CLIP 模型封装。
- `main.py`：负责触发数据下载（可选）并运行零样本评测。
- `data/`：数据缓存目录（已在 `.gitignore` 中排除）。

## 支持的数据集

目前支持以下数据集的评测：
- ImageNet 系列：imagenet1k, imagenet_v2, imagenet_adv, imagenet_ren, imagenet_ske
- CIFAR 系列：cifar10, cifar100
- 其他数据集：caltech101, country211, dtd, fer2013, fgvc_aircraft, flowers102, food101, gtsrb, mnist, pets, rendered_sst2, resisc45, stanford_cars, stl10, voc2007

贡献者协作指南详见 `AGENTS.md`。
