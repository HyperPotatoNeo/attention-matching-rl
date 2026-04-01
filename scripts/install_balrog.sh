#!/bin/bash
# Install BALROG + TextWorld dependencies for BabyAI and TextWorld evals.
#
# Usage:
#   ./scripts/install_balrog.sh              # install deps only
#   ./scripts/install_balrog.sh --with-data  # install deps + download TextWorld game files
#
# Why not in pyproject.toml?
#   balrog pulls google-genai which conflicts with verifiers -> prime-sandboxes.
#   We install balrog with --no-deps and track its safe transitive deps (textworld)
#   in the [dependency-groups] balrog group.
#
# WARNING: Do NOT `pip install balrog` from PyPI — that's Paylogic's ACL library.

set -euo pipefail

BALROG_GIT="git+https://github.com/DavidePaglieri/BALROG.git@a5fa0e7"
BALROG_DIR="${BALROG_DIR:-/tmp/balrog}"

echo "==> Installing textworld (dependency group)..."
uv sync --group balrog

echo "==> Installing balrog from GitHub (--no-deps to avoid google-genai conflict)..."
uv pip install "$BALROG_GIT" --no-deps

echo "==> Verifying imports..."
uv run python -c "from balrog.environments import make_env; print('  balrog OK')"
uv run python -c "import balrog_bench; print('  balrog_bench OK')"
uv run python -c "import textworld; print('  textworld OK')"

if [[ "${1:-}" == "--with-data" ]]; then
    echo "==> Downloading TextWorld game files to $BALROG_DIR..."
    uv run python -c "
import os
os.environ.setdefault('BALROG_DIR', '$BALROG_DIR')
from balrog_bench import setup_balrog_data
path = setup_balrog_data()
print(f'  Game files at: {path}')
"
    echo "==> Done. Set BALROG_DIR=$BALROG_DIR when running TextWorld evals."
else
    echo ""
    echo "Skipping TextWorld game files. Run with --with-data to download them,"
    echo "or they'll be downloaded automatically on first TextWorld eval run."
fi

echo ""
echo "Installation complete. You can now run:"
echo "  python scripts/eval_balrog_babyai.py --mode baseline --n 10"
echo "  python scripts/eval_balrog_textworld.py --mode baseline --n 10  (needs --with-data or BALROG_DIR)"
