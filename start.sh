#!/bin/bash
# =============================================================================
# start.sh — Container startup script for IDM-VTON RunPod Serverless
# =============================================================================
#
# This script runs every time the container starts (including warm restarts).
# Its job is to:
#   1. Download the IDM-VTON model weights from HuggingFace (if not cached)
#   2. Download the DensePose checkpoint (if not cached)
#   3. Download Human Parsing & OpenPose model checkpoints (if not cached)
#   4. Start the RunPod handler
#
# =============================================================================

set -e
set -o pipefail

# Error handler: trap any non-zero exit and log failure location
failure_handler() {
    local exit_code=$?
    local line_no=$1
    echo "==========================================" >&2
    echo "[start.sh ERROR] Script execution failed at line $line_no with exit code $exit_code" >&2
    echo "==========================================" >&2
}
trap 'failure_handler $LINENO' ERR

echo "=========================================="
echo "[start.sh] start.sh: beginning execution"
echo "=========================================="

echo "[start.sh] Environment & Diagnostics:"
echo "  - Current Working Directory: $(pwd)"
echo "  - Python Version: $(python3 --version 2>&1)"
echo "  - HF_HOME: ${HF_HOME:-/workspace/hf_cache}"

# --------------------------------------------------------------------------
# 1. Download IDM-VTON model weights
# --------------------------------------------------------------------------
# huggingface_hub's snapshot_download respects HF_HOME cache — if the files
# are already there it skips the download instantly.

echo "[start.sh] Checking / downloading IDM-VTON weights from HuggingFace..."
python3 -u -c "
from huggingface_hub import snapshot_download
import os, sys

model_id = 'yisol/IDM-VTON'
cache_dir = os.environ.get('HUGGINGFACE_HUB_CACHE', '/workspace/hf_cache')

print(f'  Downloading {model_id} to {cache_dir} (skipped if cached)...', flush=True)
try:
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        repo_type='model',
        ignore_patterns=['*.md', '*.txt', 'example/*'],
    )
    print('  IDM-VTON weights ready.', flush=True)
except Exception as e:
    print(f'  ERROR downloading HF weights: {e}', file=sys.stderr, flush=True)
    sys.exit(1)
"

# --------------------------------------------------------------------------
# 2. Download DensePose checkpoint
# --------------------------------------------------------------------------
# IDM-VTON's apply_net.py looks for this file at:
#   /workspace/IDM-VTON/ckpt/densepose/model_final_162be9.pkl

DENSEPOSE_CKPT_DIR="/workspace/IDM-VTON/ckpt/densepose"
DENSEPOSE_CKPT="$DENSEPOSE_CKPT_DIR/model_final_162be9.pkl"

mkdir -p "$DENSEPOSE_CKPT_DIR"

if [ ! -f "$DENSEPOSE_CKPT" ]; then
    echo "[start.sh] Downloading DensePose checkpoint (~900 MB)..."
    wget -q --show-progress \
        -O "$DENSEPOSE_CKPT" \
        "https://dl.fbaipublicfiles.com/detectron2/DensePose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl" || {
            echo "[start.sh ERROR] Failed to download DensePose checkpoint!" >&2
            exit 1
        }
    echo "[start.sh] DensePose checkpoint downloaded."
else
    echo "[start.sh] DensePose checkpoint already exists — skipping download."
fi

# --------------------------------------------------------------------------
# 3. Download human parsing model checkpoints
# --------------------------------------------------------------------------
PARSING_CKPT_DIR="/workspace/IDM-VTON/ckpt/humanparsing"
mkdir -p "$PARSING_CKPT_DIR"

if [ ! -f "$PARSING_CKPT_DIR/parsing_atr.onnx" ]; then
    echo "[start.sh] Downloading human parsing model (parsing_atr.onnx)..."
    wget -q --show-progress \
        -O "$PARSING_CKPT_DIR/parsing_atr.onnx" \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx" || {
            echo "[start.sh ERROR] Failed to download parsing_atr.onnx!" >&2
            exit 1
        }
    echo "[start.sh] parsing_atr.onnx downloaded."
fi

if [ ! -f "$PARSING_CKPT_DIR/parsing_lip.onnx" ]; then
    echo "[start.sh] Downloading human parsing model (parsing_lip.onnx)..."
    wget -q --show-progress \
        -O "$PARSING_CKPT_DIR/parsing_lip.onnx" \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx" || {
            echo "[start.sh ERROR] Failed to download parsing_lip.onnx!" >&2
            exit 1
        }
    echo "[start.sh] parsing_lip.onnx downloaded."
fi

# --------------------------------------------------------------------------
# 4. Download OpenPose model checkpoint
# --------------------------------------------------------------------------
OPENPOSE_CKPT_DIR="/workspace/IDM-VTON/ckpt/openpose/ckpts"
mkdir -p "$OPENPOSE_CKPT_DIR"

if [ ! -f "$OPENPOSE_CKPT_DIR/body_pose_model.pth" ]; then
    echo "[start.sh] Downloading OpenPose body pose model..."
    wget -q --show-progress \
        -O "$OPENPOSE_CKPT_DIR/body_pose_model.pth" \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth" || {
            echo "[start.sh ERROR] Failed to download body_pose_model.pth!" >&2
            exit 1
        }
    echo "[start.sh] OpenPose model downloaded."
fi

echo ""
echo "=========================================="
echo "[start.sh] All checkpoints ready."
echo "[start.sh] start.sh: launching handler.py"
echo "=========================================="

# --------------------------------------------------------------------------
# Launch the RunPod handler
# --------------------------------------------------------------------------
python3 -u /workspace/IDM-VTON/handler.py || {
    echo "==========================================" >&2
    echo "[start.sh ERROR] handler.py crashed or exited unexpectedly with exit code $?" >&2
    echo "==========================================" >&2
    exit 1
}
