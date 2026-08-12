#!/bin/bash
# =============================================================================
# start.sh — Container startup script for IDM-VTON RunPod Serverless
# =============================================================================
#
# This script runs every time the container starts (including warm restarts).
# Its job is to:
#   1. Download the IDM-VTON model weights from HuggingFace (if not cached)
#   2. Download the DensePose checkpoint (if not cached)
#   3. Start the RunPod handler
#
# WHY DOWNLOAD HERE INSTEAD OF BAKING INTO THE IMAGE?
#   The IDM-VTON weights are ~7 GB and the DensePose model is ~900 MB.
#   Baking them would create a ~15+ GB Docker image that:
#     - Takes forever to push to a registry
#     - Takes forever to pull to a RunPod worker
#     - Costs money to store in the registry
#   Downloading at startup is slower on the very first cold start but:
#     - If you mount a RunPod Network Volume, the cache persists
#     - Subsequent cold starts skip the download (files already exist)
#
# RECOMMENDATION: In RunPod, set Container Disk to 30 GB and mount a
# Network Volume of 50 GB to /workspace/hf_cache — this makes cold starts
# fast after the first one.
#
# =============================================================================

set -e  # Exit immediately if any command fails

echo "=========================================="
echo " IDM-VTON RunPod Serverless — Starting Up"
echo "=========================================="

# --------------------------------------------------------------------------
# Optional: Hugging Face token (for gated models, not needed for IDM-VTON)
# --------------------------------------------------------------------------
# If you need to download private or gated models, set HF_TOKEN as a
# RunPod secret (Environment Variable in the endpoint settings), and
# uncomment the line below.
# huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# --------------------------------------------------------------------------
# Download IDM-VTON model weights
# --------------------------------------------------------------------------
# huggingface_hub's snapshot_download respects HF_HOME cache — if the files
# are already there it skips the download instantly.

echo "[start.sh] Checking / downloading IDM-VTON weights from HuggingFace..."
python3 -c "
from huggingface_hub import snapshot_download
import os

model_id = 'yisol/IDM-VTON'
cache_dir = os.environ.get('HUGGINGFACE_HUB_CACHE', '/workspace/hf_cache')

print(f'  Downloading {model_id} to {cache_dir} (skipped if cached)...')
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    repo_type='model',
    ignore_patterns=['*.md', '*.txt', 'example/*'],  # skip docs & examples
)
print('  IDM-VTON weights ready.')
"

# --------------------------------------------------------------------------
# Download DensePose checkpoint
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
        "https://dl.fbaipublicfiles.com/detectron2/DensePose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    echo "[start.sh] DensePose checkpoint downloaded."
else
    echo "[start.sh] DensePose checkpoint already exists — skipping download."
fi

# --------------------------------------------------------------------------
# Download human parsing model checkpoints
# --------------------------------------------------------------------------
# The Parsing model (SCHP / ATR) downloads its own weights on first run
# from a hardcoded URL inside preprocess/humanparsing/. If it fails in
# headless mode, pre-download them here.

PARSING_CKPT_DIR="/workspace/IDM-VTON/ckpt/humanparsing"
mkdir -p "$PARSING_CKPT_DIR"

if [ ! -f "$PARSING_CKPT_DIR/parsing_atr.onnx" ]; then
    echo "[start.sh] Downloading human parsing model (parsing_atr.onnx)..."
    wget -q --show-progress \
        -O "$PARSING_CKPT_DIR/parsing_atr.onnx" \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx"
    echo "[start.sh] parsing_atr.onnx downloaded."
fi

if [ ! -f "$PARSING_CKPT_DIR/parsing_lip.onnx" ]; then
    echo "[start.sh] Downloading human parsing model (parsing_lip.onnx)..."
    wget -q --show-progress \
        -O "$PARSING_CKPT_DIR/parsing_lip.onnx" \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx"
    echo "[start.sh] parsing_lip.onnx downloaded."
fi

# --------------------------------------------------------------------------
# Download OpenPose model checkpoint
# --------------------------------------------------------------------------

OPENPOSE_CKPT_DIR="/workspace/IDM-VTON/ckpt/openpose/ckpts"
mkdir -p "$OPENPOSE_CKPT_DIR"

if [ ! -f "$OPENPOSE_CKPT_DIR/body_pose_model.pth" ]; then
    echo "[start.sh] Downloading OpenPose body pose model..."
    wget -q --show-progress \
        -O "$OPENPOSE_CKPT_DIR/body_pose_model.pth" \
        "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth"
    echo "[start.sh] OpenPose model downloaded."
fi

echo ""
echo "[start.sh] All checkpoints ready. Launching handler..."
echo "=========================================="

# --------------------------------------------------------------------------
# Launch the RunPod handler
# The handler.py does its own cold-start model loading before calling
# runpod.serverless.start(), so this is the final blocking command.
# --------------------------------------------------------------------------
exec python3 /workspace/IDM-VTON/handler.py
