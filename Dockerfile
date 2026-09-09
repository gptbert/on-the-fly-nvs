# syntax=docker/dockerfile:1
# On-the-fly NVS - Docker Image
# Requires NVIDIA GPU with CUDA 12.x driver support (nvidia-smi should show CUDA 12.x)
# Build: docker build -t on-the-fly-nvs .
# Submodules are auto-cloned at build time if not locally initialized.

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XFORMERS_FORCE_DISABLE_TRITON=1 \
    MODELS_DIR=/app/models \
    HF_HOME=/app/models/huggingface \
    TORCH_HOME=/app/models/torch \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    CCCL_IGNORE_DEPRECATED_CPP_DIALECT=1

RUN sed -i -E \
    -e 's|https?://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
    -e 's|https?://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
    /etc/apt/sources.list

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
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
    openssh-server \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6

RUN --mount=type=cache,target=/root/.cache/pip \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY constraints-runtime.txt /opt/constraints-runtime.txt
ARG PIP_PACKAGE_INDEX=https://repo.huaweicloud.com/repository/pypi/simple
ENV PIP_CONSTRAINT=/opt/constraints-runtime.txt \
    PIP_INDEX_URL=${PIP_PACKAGE_INDEX}

# Fetch only GPU wheels from PyTorch; resolve their dependencies via PIP_INDEX_URL.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 xformers==0.0.35 \
    --index-url https://download.pytorch.org/whl/cu128 && \
    pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 xformers==0.0.35

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install cupy-cuda12x

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    plyfile \
    tqdm \
    opencv-python \
    lpips \
    websockets

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-r3.txt && \
    pip install --no-deps -r requirements-r3-source.txt

RUN git config --global --add safe.directory /app && \
    if [ ! -f submodules/fused-ssim/setup.py ]; then \
        git clone https://github.com/rahul-goel/fused-ssim submodules/fused-ssim && \
        git -C submodules/fused-ssim checkout 8bdb59feb7b9a41b1fab625907cb21f5417deaac; \
    fi && \
    if [ ! -f submodules/graphdecoviewer/pyproject.toml ]; then \
        git clone https://github.com/graphdeco-inria/graphdecoviewer.git submodules/graphdecoviewer && \
        git -C submodules/graphdecoviewer checkout ae889ccdf47df76d039b455acf6f443077eb0f06; \
    fi && \
    if [ ! -f submodules/Depth-Anything-V2/README.md ]; then \
        git clone https://github.com/DepthAnything/Depth-Anything-V2.git submodules/Depth-Anything-V2 && \
        git -C submodules/Depth-Anything-V2 checkout e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1; \
    fi

ENV MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation submodules/diff-gaussian-rasterization
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation submodules/fused-ssim
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation submodules/simple-knn
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install submodules/graphdecoviewer

RUN mkdir -p /app/models /app/results /app/data

RUN mkdir -p /run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    sed -i \
        -e 's|#PermitRootLogin.*|PermitRootLogin prohibit-password|' \
        -e 's|#PasswordAuthentication.*|PasswordAuthentication no|' \
        -e 's|#PubkeyAuthentication.*|PubkeyAuthentication yes|' \
        /etc/ssh/sshd_config

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 22 6009 8000

ENV STREAM_URL="" \
    GEOMETRY_PROVIDER="r3" \
    R3_CHECKPOINT="r3" \
    R3_RESOLUTION="504" \
    R3_BANK_SIZE="8" \
    MAX_ACTIVE_KEYFRAMES="40" \
    DEPTH_MODEL="vitb" \
    DOWNSAMPLING="1.5"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sh", "-c", \
    "python train.py \
        -s ${STREAM_URL} \
        --downsampling ${DOWNSAMPLING} \
        --geometry_provider ${GEOMETRY_PROVIDER} \
        --r3_checkpoint ${R3_CHECKPOINT} \
        --r3_resolution ${R3_RESOLUTION} \
        --r3_bank_size ${R3_BANK_SIZE} \
        --max_active_keyframes ${MAX_ACTIVE_KEYFRAMES} \
        --viewer_mode web \
        -m /app/results/$(date +%Y%m%d_%H%M%S)"]
