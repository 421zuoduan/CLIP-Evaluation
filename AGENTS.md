# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the single entry point: it must load the CLIP ViT-L/14 model under `model/`, orchestrate dataset preparation, and run evaluations for every supported benchmark.
- Place reusable helpers in `model/` (e.g., `model/clip.py` that wraps `openai/clip-vit-large-patch14`). Add `__init__.py` so imports stay explicit.
- All downloaded data lives under `data/` with per-dataset subfolders (`data/imagenet1k`, `data/cifar10`, …) to avoid collisions.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs core dependencies for PyTorch, Hugging Face, and data utilities.
- `python main.py --datasets all --data-root ./data` downloads any missing datasets and evaluates CLIP ViT-L/14 sequentially.
- `python main.py --datasets cifar10,imagenet1k --device cuda` limits the run to specific benchmarks while forcing GPU usage.
- `python -m model.clip --download-only --data-root ./data` (example helper) should initialize weights without launching evaluation; add similar utilities when debugging.

## Dataset Acquisition Workflow
- Use `torchvision.datasets` for ImageNet variants, CIFAR-10/100, MNIST, STL10, VOC2007, Caltech-101, SUN397, FGVC Aircraft, Stanford Cars, Birdsnap, DTD, Flowers-102, Food-101, GTSRB, Pets, Resisc45.
- Use the Hugging Face `datasets` library for ObjectNet, Country-211, Eurosat, FER2013, PCam, Rendered SST2, ImageNet-V2/Adv/Ren/Ske, and benchmarks lacking a torchvision wrapper.
- Ensure each loader writes to `data/<dataset>` and exposes a consistent `(image, label)` interface so `main.py` can reuse evaluation utilities.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, type hints, and snake_case identifiers (`load_imagenet_split`, `run_eval_loop`).
- Prefer dataclasses or typed dictionaries to describe dataset metadata (classes, resolution, normalization).
- Keep shell snippets `bash`-compatible and uppercase environment variables (`HF_ENDPOINT`, `TORCH_HOME`).

## Testing Guidelines
- Add smoke evaluations before large runs: `python main.py --datasets cifar10 --limit 64` should finish without grabbing the entire dataset.
- When changing download logic, clear the target subdirectory and rerun to verify fresh installation succeeds.
- Track accuracy deltas in a `results/` log or PR comment; regressions >0.5% on any dataset require investigation.

## Commit & Pull Request Guidelines
- Use imperative commit titles (`Refactor dataset registry`, `Add ObjectNet loader`) and mention affected datasets in the body.
- PR descriptions must list datasets tested, commands executed, and any mirrors or environment overrides used.
- Include tables summarizing new accuracy numbers when touching evaluation code or preprocessing defaults.*** End Patch
