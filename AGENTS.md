# Repository Guidelines

## Project Structure & Module Organization
- `evaluate_clip.py` holds the zero-shot evaluation pipeline, including dataset fallback logic and prompt construction. Keep new inference code here or in adjacent modules under the root to maintain a single entry point.
- Shell entrypoints (`run.sh`, `run_llava.sh`) orchestrate installs, environment variables, and evaluation presets; prefer extending them instead of duplicating logic.
- `download.sh` and `download_datasets.py` manage dataset retrieval; place additional data utilities beside them.
- Use `data/` (git-ignored) for local datasets and cache artifacts. Avoid tracking large binaries in the repository.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs the default CUDA-friendly stack.
- `bash run.sh [caltech101|sun397|all]` runs end-to-end evaluation with automatic batch tuning on GPU 0.
- `bash run_llava.sh …` mirrors `run.sh` but constrains dependencies to the pre-existing `llava` Conda environment.
- `python evaluate_clip.py --dataset caltech101 --data-root ./data --no-download` lets you reproduce results locally without touching the download logic.
- `bash download.sh [dataset]` or `python download_datasets.py --datasets all` fetches torchvision-managed datasets into `data/`.

## Coding Style & Naming Conventions
- Python follows PEP 8 with 4-space indentation, descriptive snake_case names, and explicit type hints (see `set_hf_mirror`, `_scan_caltech101`).
- Keep imports lazy when they pull heavy dependencies; prefer helper functions that bail gracefully when `torchvision` is absent.
- Shell scripts assume `bash`; guard environment assumptions, echo actionable errors, and keep variables UPPER_SNAKE_CASE.

## Testing Guidelines
- No formal test suite exists; validate changes by running `python evaluate_clip.py` on a constrained subset (e.g., `--dataset caltech101 --batch-size 8`) and confirm accuracy deltas.
- When modifying download logic, test with mirror URLs via `CALTECH101_URL` / `SUN397_URL` to ensure patching still works.
- Record GPU/CPU, dataset root, and key flags in your test notes so reviewers can reproduce.

## Commit & Pull Request Guidelines
- Use imperative, concise commit titles (e.g., `Add SUN397 folder fallback scanner`) and describe user-facing effects in the body when needed.
- For PRs, link to the motivating issue or experiment, list runnable commands plus observed metrics, and call out any new environment variables.
- Include before/after accuracy or throughput numbers when touching model or batching code, and mention mirror defaults if they change.

## Dataset & Mirror Tips
- Default Hugging Face access relies on `HF_ENDPOINT=https://hf-mirror.com` and `HF_HUB_ENABLE_HF_TRANSFER=1`; keep these documented in new tooling.
