# CLIP ViT-L/14 基准套件

本仓库用于评测 `openai/clip-vit-large-patch14` 在多种分类与鲁棒性数据集上的表现，包括 ImageNet 系列、CIFAR、ObjectNet、Country211 等。脚本会统一下载数据并为后续实验提供标准化入口。

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
python -m download.download_all --datasets cifar10,imagenet1k --data-root ./data
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
python main.py --datasets cifar10,imagenet_v2 --limit 128
```

## 项目结构

- `download/`：数据集注册表与下载 CLI。
- `model/`：CLIP 模型封装（自行添加实现）。
- `main.py`：负责触发数据下载（可选）并运行零样本评测。
- `data/`：数据缓存目录（已在 `.gitignore` 中排除）。

贡献者协作指南详见 `AGENTS.md`。
