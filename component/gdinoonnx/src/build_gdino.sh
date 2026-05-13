#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/app/assets/models/wrappers/gdinoonnx}"
OUTPUT_DIR="${GDINO_TRT_OUTPUT_DIR:-/app/assets/models/compiled/gdinoonnx}"
TRT_PROMPT="${GDINO_TRT_PROMPT:-car .}"
TRT_HEIGHT="${GDINO_TRT_HEIGHT:-800}"
TRT_WIDTH="${GDINO_TRT_WIDTH:-1200}"
TRT_MAX_TEXT_LEN="${GDINO_TRT_MAX_TEXT_LEN:-32}"
TRT_PRECISION="${GDINO_TRT_PRECISION:-fp16}"
PYTHON_BIN="${PYTHON_BIN:-/app/.venv/bin/python}"

/app/src/boot.sh

"$PYTHON_BIN" /app/src/build.py \
    --repo-dir "$REPO_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --prompt "$TRT_PROMPT" \
    --height "$TRT_HEIGHT" \
    --width "$TRT_WIDTH" \
    --max-text-len "$TRT_MAX_TEXT_LEN" \
    --precision "$TRT_PRECISION"
