#!/usr/bin/env bash
# run.sh — run the LLM SDK test suite using the local virtual environment.
#
# Usage:
#   ./run.sh                          # run all enabled providers and proxies
#   ./run.sh -m openai                # OpenAI tests only
#   ./run.sh -m "anthropic or gemini" # multiple providers
#   ./run.sh -m langchain             # all LangChain tests
#   ./run.sh -m "provider and chat"   # filter by type
#   ./run.sh tests/providers/test_openai_provider.py  # a single file
#
# Any extra arguments are passed directly to pytest.
#
# Provider markers:
#   openai, anthropic, gemini, mistral, azure_openai, azure_foundry
#
# Type markers:
#   provider, proxy, langchain, chat, streaming

set -euo pipefail

VENV_DIR="venv"

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if [ ! -f "$VENV_DIR/bin/pytest" ]; then
    echo "ERROR: Virtual environment not set up."
    echo "  Run: ./setup.sh"
    exit 1
fi

if [ ! -f "config.yaml" ]; then
    echo "ERROR: config.yaml not found."
    echo "  Copy and fill in: cp config.yaml.example config.yaml"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────

exec "$VENV_DIR/bin/pytest" "$@"
