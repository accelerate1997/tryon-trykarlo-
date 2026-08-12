# =============================================================================
# Dockerfile for IDM-VTON RunPod Serverless
# =============================================================================
#
# DESIGN DECISIONS:
#
# 1. Base image: pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
#    - Python 3.10 is included — numpy==1.24.4 and scipy==1.10.1 install
#      cleanly here (they break on Python 3.12).
#    - CUDA 12.1 + cuDNN 8 gives us full compatibility with the GPU drivers
#      used by RunPod's A4000/A5000/L40S worker machines.
#    - The 'devel' variant (vs 'runtime') includes the CUDA compiler (nvcc)
#      which is REQUIRED to build Detectron2 from source.
#
# 2. Model weights: NOT baked into the image.
#    - The IDM-VTON weights (~7 GB) + DensePose checkpoint (~900 MB) would
#      make the image enormous and slow to pull.
#    - Instead, start.sh downloads them from HuggingFace at container startup.
#    - TRADEOFF: First cold start takes ~3-5 min while weights download.
#      Subsequent warm starts are fast because RunPod caches the network volume.
#      If you set "Container Disk" to >=30 GB and enable "Network Storage",
#      the weights persist across worker restarts.
#
# 3. Detectron2: built from source.
#    - Pre-built wheels almost never match your exact CUDA/PyTorch version.
#    - Building from source inside the container guarantees compatibility
#      and compiles the CUDA extensions for your specific GPU architecture.
#
# =============================================================================

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# --------------------------------------------------------------------------
# System-level dependencies
# --------------------------------------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Python environment setup
# --------------------------------------------------------------------------

WORKDIR /workspace

# Upgrade pip first to avoid old-pip quirks
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --------------------------------------------------------------------------
# Install PyTorch-dependent libraries
# (Use the torch already in the base image — don't reinstall it)
# --------------------------------------------------------------------------

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# --------------------------------------------------------------------------
# Install Detectron2 from source
#
# WHY FROM SOURCE:
#   Detectron2 has CUDA operator extensions (like ROI Align, DensePose
#   renderers) that must be compiled against your EXACT CUDA version.
#   pip wheels only cover a narrow set of CUDA versions, so building from
#   source is the only reliable approach here.
#
# This step takes ~10-15 minutes on first build but is cached by Docker
# layer caching — only re-runs if requirements.txt changes.
# --------------------------------------------------------------------------

RUN git clone https://github.com/facebookresearch/detectron2.git /workspace/detectron2
RUN cd /workspace/detectron2 && pip install --no-cache-dir -e .

# DensePose is a Detectron2 "project" — it lives in the projects/ subdirectory.
# We need its Python modules on the path, which handler.py handles at runtime.
# The checkpoint (model_final_162be9.pkl) is downloaded by start.sh.

# --------------------------------------------------------------------------
# Clone IDM-VTON source code from HuggingFace
# This contains the custom UNet variants, pipeline, and preprocessing code.
# --------------------------------------------------------------------------

RUN git clone https://huggingface.co/spaces/yisol/IDM-VTON /workspace/IDM-VTON

# Copy our RunPod handler into the IDM-VTON directory so that relative
# imports (apply_net, configs/, ckpt/) resolve correctly.
COPY handler.py /workspace/IDM-VTON/handler.py

# --------------------------------------------------------------------------
# Startup script — downloads model weights at container start
# --------------------------------------------------------------------------

COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# --------------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------------

# Tells HuggingFace where to cache downloaded models.
# We point it to /workspace so it persists if you mount a network volume.
ENV HF_HOME=/workspace/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/workspace/hf_cache

# Memory fragmentation prevention (also set in handler.py, belt-and-suspenders)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# --------------------------------------------------------------------------
# Working directory and entrypoint
# --------------------------------------------------------------------------

WORKDIR /workspace/IDM-VTON

# The startup script downloads weights and then launches the handler
CMD ["/workspace/start.sh"]
