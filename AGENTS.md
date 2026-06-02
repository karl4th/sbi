# Repository Guidelines

## Project Structure & Module Organization

Core library code lives in `src/sbi/`: `core/` contains the transformer and config dataclasses, `memory/` contains episodic, Hebbian, and compression logic, `search/` contains fingerprinting and retrieval, and `system.py` wires the full SBI system together. Dataset utilities live under `data/`, with bAbI loading/tokenization in `data/babi/` and synthetic task generation in `data/tasks/`. Training and evaluation entry points are in `training/`; YAML experiment settings are in `configs/`; exploratory workflow is in `notebooks/`. Put future automated tests in `tests/`. Generated checkpoints and result artifacts belong in `experiments/`.

## Build, Test, and Development Commands

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Train the control model:

```bash
python training/train_baseline.py --config configs/baseline.yaml
```

Train the SBI model:

```bash
python training/train_sbi.py --config configs/sbi_small.yaml
```

Compare baseline and SBI checkpoints:

```bash
python training/evaluate.py
```

For import checks, use `PYTHONPATH=src python -c "import sbi"`. Training scripts already add `src/` and the repository root to `sys.path`.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, type hints for public helpers, and concise docstrings where behavior is not obvious. Follow existing naming: `snake_case` for functions, variables, and modules; `PascalCase` for classes such as `SBISystem`; uppercase constants such as `PAD_ID`. Keep tensor shape handling explicit.

## Testing Guidelines

No formal test runner is currently configured. Add tests under `tests/` using `pytest` conventions (`test_*.py`, `test_<behavior>()`) when changing shared components in `src/sbi/` or `data/`. For training changes, include a smoke check that instantiates the model or dataset with a small config. Before opening a PR, run the relevant training/evaluation command on the smallest practical scope and note hardware used.

## Commit & Pull Request Guidelines

Recent commits use imperative, descriptive subjects such as `Enhance episodic memory structure...` and `Refactor bAbI task generator...`. Keep subjects specific to the behavior changed; avoid vague messages like `update code`. Pull requests should include the purpose, main files touched, commands run, observed metrics or failures, and any checkpoint/result artifacts produced. Link related issues when available and include notebook screenshots only when notebook output or plots changed.

## Security & Configuration Tips

Do not commit downloaded datasets, checkpoints, cache files, or large experiment outputs. Keep configuration in `configs/*.yaml` and document any non-default path, GPU requirement, or external download needed to reproduce a run.
