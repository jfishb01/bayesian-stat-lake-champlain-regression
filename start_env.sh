#!/usr/bin/env bash
set -e

export PATH="$HOME/.local/bin:$PATH:/opt/homebrew/bin/"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.14
fi

echo "Syncing dependencies..."
uv sync

echo "Installing project..."
uv pip install -e .

uv pip install --upgrade cmdstanpy
