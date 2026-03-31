# On-the-fly NVS - Docker Image
# Requires NVIDIA GPU with CUDA 12.x driver support (nvidia-smi should show CUDA 12.x)
# Build: docker build -t on-the-fly-nvs .
# Submodules must be initialized before building:
#   git submodule update --init --recursive

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Suppress xformers/triton warnings in headless mode
    XFORMERS_FORCE_DISABLE_TRITON=1 \
    # HuggingFace / torch hub cache dirs (mountable as volumes)
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    # OpenCV runtime deps
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ── PyTorch + CUDA 12.8 ────────────────────────────────────────────────────────
# Install before submodule builds so torch is available during compilation
RUN pip install --no-cache-dir \
    torch torchvision xformers \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir cupy-cuda12x

# ── Python dependencies (non-submodule) ───────────────────────────────────────
RUN pip install --no-cache-dir \
    plyfile \
    tqdm \
    opencv-python \
    lpips \
    websockets

# ── Copy source (submodules must be initialized locally first) ─────────────────
COPY . .

# ── Build custom CUDA extensions ───────────────────────────────────────────────
# MAX_JOBS limits parallel compilation to avoid OOM during build
ENV MAX_JOBS=4
RUN pip install --no-cache-dir submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir submodules/fused-ssim
RUN pip install --no-cache-dir submodules/simple-knn
RUN pip install --no-cache-dir submodules/graphdecoviewer

# ── Cache volume mount points ──────────────────────────────────────────────────
# Mount these as volumes to persist model downloads across container restarts:
#   /cache/huggingface  - Depth-Anything-V2 models (~400MB per model)
#   /cache/torch        - XFeat (torch hub)
#   /app/results        - reconstruction outputs
RUN mkdir -p /cache/huggingface /cache/torch /app/results /app/data

EXPOSE 6009 8000

# Default: stream mode with web viewer
# Override STREAM_URL at runtime:
#   docker run -e STREAM_URL=http://192.168.1.x:8080/video ...
ENV STREAM_URL=""

CMD ["sh", "-c", \
    "python train.py \
        -s ${STREAM_URL} \
        --downsampling 1.5 \
        --viewer_mode web \
        -m /app/results/$(date +%Y%m%d_%H%M%S)"]
