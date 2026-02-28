#!/usr/bin/env bash
# setup.sh — create the virtual environment and install all dependencies.
# Run this once before running the test suite for the first time, or after
# updating requirements.txt.
#
# Usage:
#   ./setup.sh

set -euo pipefail

VENV_DIR="venv"
PYTHON="${PYTHON:-python3}"

# ── Virtual environment ───────────────────────────────────────────────────────

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists, skipping creation."
fi

PIP="$VENV_DIR/bin/pip"

echo "Upgrading pip..."
"$PIP" install --upgrade pip --quiet

# ── Dependencies ──────────────────────────────────────────────────────────────

echo "Installing dependencies (requirements.txt)..."
"$PIP" install -r requirements.txt --quiet

# ── Config check ─────────────────────────────────────────────────────────────

echo ""
if [ ! -f "config.yaml" ]; then
    echo "WARNING: config.yaml not found."
    echo "  Copy the example and fill in your gateway URLs and API keys:"
    echo ""
    echo "    cp config.yaml.example config.yaml"
    echo "    # then edit config.yaml"
    echo ""
else
    echo "config.yaml found."
fi

echo "Setup complete. Run the tests with:"
echo ""
echo "    ./run.sh              # all enabled providers and proxies"
echo "    ./run.sh -m openai    # a specific provider"
echo ""
