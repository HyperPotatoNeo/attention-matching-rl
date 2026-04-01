---
name: installation
description: How to install prime-rl and its optional dependencies. Use when setting up the project, installing extras like deep-gemm for FP8 models, or troubleshooting dependency issues.
---

# Installation

## Basic install

```bash
uv sync
```

This installs all core dependencies defined in `pyproject.toml`.

## All extras at once

The recommended way to install for most users:

```bash
uv sync --all-extras
```

This installs all optional extras (flash-attn, flash-attn-cute, etc.) in one go.

## FP8 inference with deep-gemm

For certain models like GLM-5-FP8, you need `deep-gemm`. Install it via the `fp8-inference` dependency group:

```bash
uv sync --group fp8-inference
```

This installs the pre-built `deep-gemm` wheel. No CUDA build step is needed.

## Dev dependencies

```bash
uv sync --group dev
```

Installs pytest, ruff, pre-commit, and other development tools.

## BabyAI / BALROG / TextWorld setup

The BabyAI and TextWorld evals depend on `balrog`, `textworld`, and `balrog-bench`.
`balrog` cannot go in `pyproject.toml` because it pulls `google-genai` which conflicts
with `verifiers` → `prime-sandboxes`. Instead, `textworld` is tracked in the `balrog`
dependency group and `balrog` itself is pip-installed with `--no-deps`.

**One-command install:**

```bash
./scripts/install_balrog.sh              # deps only
./scripts/install_balrog.sh --with-data  # deps + TextWorld game files
```

**Manual install (equivalent):**

```bash
uv sync --group balrog                   # installs textworld
uv pip install git+https://github.com/DavidePaglieri/BALROG.git@a5fa0e7 --no-deps
```

`balrog-bench` is vendored at `src/balrog_bench.py` — installed automatically via the
`src/` layout, no extra steps needed.

> **Warning:** Do NOT `pip install balrog` or `pip install balrog-bench` from PyPI — the
> PyPI `balrog` package is Paylogic's ACL library (unrelated), not the BALROG benchmark.

Verify:
```bash
uv run python -c "from balrog.environments import make_env; print('balrog OK')"
uv run python -c "import balrog_bench; print('balrog_bench OK')"
uv run python -c "import textworld; print('textworld OK')"
```

## Key files

- `pyproject.toml` — all dependencies, extras, and dependency groups
- `uv.lock` — pinned lockfile (update with `uv sync --all-extras`)
