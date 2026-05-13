#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/app/assets/models/wrappers/gdinoonnx}"
REPO_URL="${REPO_URL:-https://github.com/wingdzero/GroundingDINO-TensorRT-and-ONNX-Inference.git}"
REPO_REF="${REPO_REF:-master}"
PATHS_FILE="/app/src/upstream_paths.txt"
CHECKPOINT_PATH="$REPO_DIR/weights/groundingdino_swint_ogc.pth"
RESOLVE_CHECKPOINT_SCRIPT="${RESOLVE_CHECKPOINT_SCRIPT:-/app/src/resolve_checkpoint.py}"

case "$REPO_DIR" in
    /app/src|/app/src/*)
        echo "REPO_DIR must point to runtime assets, not tracked source: $REPO_DIR"
        exit 1
        ;;
esac

mkdir -p /app/assets/models/wrappers

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning GroundingDINO wrapper repository metadata..."
    git clone --no-checkout "$REPO_URL" "$REPO_DIR"
else
    echo "GroundingDINO wrapper repository already exists."
fi

ORIGIN_URL="$(git -C "$REPO_DIR" config --get remote.origin.url || true)"
if [ "$ORIGIN_URL" != "$REPO_URL" ]; then
    echo "Unexpected wrapper origin URL: $ORIGIN_URL"
    echo "Expected: $REPO_URL"
    exit 1
fi

echo "Fetching latest wrapper files from $REPO_REF..."
git -C "$REPO_DIR" fetch --depth 1 origin "$REPO_REF"
mapfile -t CHECKOUT_PATHS < <(grep -vE '^\s*(#|$)' "$PATHS_FILE")
git -C "$REPO_DIR" checkout -f FETCH_HEAD -- "${CHECKOUT_PATHS[@]}"
mkdir -p "$REPO_DIR/weights"

if git lfs version >/dev/null 2>&1; then
    echo "Fetching GroundingDINO checkpoint with Git LFS..."
    if ! git -C "$REPO_DIR" lfs pull --include="weights/groundingdino_swint_ogc.pth"; then
        echo "Git LFS fetch failed. Falling back to alternate checkpoint sources."
    fi
fi

if [ ! -f "$RESOLVE_CHECKPOINT_SCRIPT" ]; then
    echo "Checkpoint resolver script not found at $RESOLVE_CHECKPOINT_SCRIPT"
    exit 1
fi

python "$RESOLVE_CHECKPOINT_SCRIPT" --checkpoint-path "$CHECKPOINT_PATH"

if [ -f "$CHECKPOINT_PATH" ] && grep -q '^version https://git-lfs.github.com/spec/v1' "$CHECKPOINT_PATH"; then
    echo "GroundingDINO checkpoint is still a Git LFS pointer at $CHECKPOINT_PATH"
    echo "Provide a real checkpoint file or set GDINO_CHECKPOINT_SOURCE / GDINO_CHECKPOINT_URL."
    exit 1
fi

if ! python -c "import onnx, onnxscript, tensorrt, torch" >/dev/null 2>&1; then
    echo "Missing gdinoonnx Python dependencies in /app/.venv. Rebuild the image."
    exit 1
fi

echo "GroundingDINO wrapper is ready."
echo "Build command: python /app/src/build.py --prompt \"car .\""
echo "Build wrapper: bash /app/src/build_gdino.sh"
