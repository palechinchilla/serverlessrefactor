# syntax=docker/dockerfile:1.7
# =============================================================================
# RunPod Serverless ComfyUI worker
#   Target hardware : RTX 5090 (Blackwell, SM 12.0)
#   Stack pin       : CUDA 13.2.1 / cuDNN runtime / PyTorch 2.11.0+cu130 /
#                     Triton 3.6 / SageAttention 2.2.0 (prebuilt Blackwell wheel)
#   Install path    : comfy-cli (no manual ComfyUI clone, no Manager)
#   Drivers         : provided by the RunPod host — DO NOT install nvidia-* here
#
# Layers are ordered for cache friendliness: anything you iterate on (custom
# nodes + baked models + handler) goes AFTER the heavy/stable layers, so editing
# the handler does not invalidate the torch / sage / ComfyUI layers.
# =============================================================================

FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04

# Deterministic, fail-fast shell behaviour. -o pipefail makes piped RUNs fail
# if any stage in the pipe fails; -e exits on any non-zero return.
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# Build/runtime hygiene.
#   PYTHONUNBUFFERED      : flush stdout/stderr immediately (CloudWatch / RunPod logs)
#   PYTHONDONTWRITEBYTECODE: no .pyc clutter inside the image
#   PIP_NO_CACHE_DIR      : keep image small; uv handles caching itself
#   UV_LINK_MODE=copy     : avoid hardlink errors across overlay layers
#   UV_NO_CACHE           : avoid persisting uv's resolver cache in the image
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
    COMFY_PRELOAD_COMFY_KITCHEN=0

# -----------------------------------------------------------------------------
# System packages — kept minimal: Python 3.12 + libs ComfyUI / OpenCV / video pull in.
#   libgl1, libglib2.0-0  : OpenCV + image preview deps used by some custom nodes
#   ffmpeg                : video I/O for Wan / LTX SaveVideo nodes
#   libc-bin              : provides ldconfig, used by the NVIDIA base image's
#                           entrypoint to verify driver presence. Without it,
#                           every cold-start log opens with a misleading
#                           "WARNING: The NVIDIA Driver was not detected" line.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        git wget curl ca-certificates \
        libgl1 libglib2.0-0 ffmpeg \
        libc-bin \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# uv — single static binary, replaces pip everywhere below.
# 10–30× faster resolves, deterministic, plays well with venvs.
# -----------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# -----------------------------------------------------------------------------
# Workspace + dedicated venv.
# Activating the venv via ENV (VIRTUAL_ENV + PATH) is enough — no `source`
# needed. uv pip will install into $VIRTUAL_ENV automatically.
#
# `--seed` installs pip + setuptools + wheel INTO the venv. Required because
# `comfy install` runs `python -m pip install --upgrade pip uv` early in its
# bootstrap; without seed packages, the venv has no pip and that step fails.
# -----------------------------------------------------------------------------
WORKDIR /workspace
# PATH must include /sbin and /usr/sbin (system admin tools live there). The
# NVIDIA base image's entrypoint script calls `ldconfig` to verify the driver,
# and ldconfig is at /usr/sbin/ldconfig. Dropping those dirs from PATH was what
# caused the spurious "NVIDIA Driver was not detected" warning at every boot,
# even after libc-bin was installed.
ENV COMFY_HOME=/workspace/ComfyUI \
    VIRTUAL_ENV=/workspace/venv \
    PATH=/workspace/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN uv venv --seed --python 3.12 "$VIRTUAL_ENV"

# -----------------------------------------------------------------------------
# 1) Torch stack — pinned to cu130 wheels. MUST come before comfy-cli so that
# `comfy install --skip-torch-or-directml` doesn't try to (re)install torch.
# Triton 3.6.x is pulled transitively as a torch 2.11 dep.
# -----------------------------------------------------------------------------
RUN uv pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0+cu130 torchvision torchaudio

# -----------------------------------------------------------------------------
# 2) SageAttention 2.2.0 — prebuilt Blackwell cp312 wheel.
# Compiled against the same torch+CUDA we just installed; ComfyUI will detect
# it at launch when --use-sage-attention is passed.
# -----------------------------------------------------------------------------
RUN wget -q -O /tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl \
        "https://github.com/palechinchilla/SageAttention-2.2.0-Blackwell-/raw/refs/heads/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl" && \
    uv pip install --no-cache-dir /tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl && \
    rm /tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl

# -----------------------------------------------------------------------------
# 3) comfy-cli + ComfyUI itself.
# Flag placement matters: `--skip-prompt` and `--workspace` are GLOBAL options
# on the `comfy` app callback, so they MUST come before the `install` subcommand.
# `--skip-manager`, `--skip-torch-or-directml`, `--fast-deps`, `--nvidia` are
# install-subcommand options and follow it.
#
#   --skip-prompt              : non-interactive (Dockerfile-safe)
#   --workspace                : where ComfyUI is cloned
#   --skip-manager             : no ComfyUI-Manager (you bake nodes manually)
#   --skip-torch-or-directml   : keep our pinned torch from step 1
#   --fast-deps                : uv-backed dep resolution
#   --nvidia                   : platform context (CUDA-flavored install layout)
# -----------------------------------------------------------------------------
RUN uv pip install --no-cache-dir comfy-cli && \
    comfy --skip-prompt --workspace "$COMFY_HOME" install \
          --skip-manager \
          --skip-torch-or-directml \
          --fast-deps \
          --nvidia

# -----------------------------------------------------------------------------
# 4) Worker-only deps (small, fast layer — you'll edit this rarely).
# -----------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# -----------------------------------------------------------------------------
# 4b) Pre-compile bytecode for the venv + ComfyUI tree.
# Populates __pycache__ so cold-start first-imports skip the compile step
# (~50–150 ms saved per cold start). `-j 0` parallelises across all build
# cores. `|| true` because some 3rd-party packages ship intentionally-broken
# Python files (test fixtures, conditional imports) that compileall flags but
# nothing at runtime ever imports — failing the build on those would be wrong.
# -----------------------------------------------------------------------------
RUN python -m compileall -q -j 0 /workspace/venv /workspace/ComfyUI || true

# -----------------------------------------------------------------------------
# 5) extra_model_paths.yaml — mount /runpod-volume/models/* on top of baked
# model dirs. If no network volume is attached, ComfyUI ignores the missing path.
# -----------------------------------------------------------------------------
COPY extra_paths.yaml ${COMFY_HOME}/extra_model_paths.yaml

# =============================================================================
# USER-EXTENSION LAYER — custom nodes + (eventually) baked models.
# One RUN per node so editing/adding/removing a single node only invalidates
# that layer and everything below. `--depth=1` keeps clones small (no git
# history). `--no-cache-dir` matches the rest of the Dockerfile's uv hygiene.
# The `if [ -f requirements.txt ]` guard tolerates nodes that ship without one.
# =============================================================================

# KJNodes — utility nodes incl. SageAttention wrapper used by Wan workflows
RUN git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git \
        ${COMFY_HOME}/custom_nodes/ComfyUI-KJNodes && \
    if [ -f ${COMFY_HOME}/custom_nodes/ComfyUI-KJNodes/requirements.txt ]; then \
        uv pip install --no-cache-dir -r ${COMFY_HOME}/custom_nodes/ComfyUI-KJNodes/requirements.txt; \
    fi

# WanVideoWrapper — Wan 2.x video diffusion nodes
RUN git clone --depth=1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
        ${COMFY_HOME}/custom_nodes/ComfyUI-WanVideoWrapper && \
    if [ -f ${COMFY_HOME}/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt ]; then \
        uv pip install --no-cache-dir -r ${COMFY_HOME}/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt; \
    fi

# Custom-Scripts — pythongosssss UX/QoL nodes
RUN git clone --depth=1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git \
        ${COMFY_HOME}/custom_nodes/ComfyUI-Custom-Scripts && \
    if [ -f ${COMFY_HOME}/custom_nodes/ComfyUI-Custom-Scripts/requirements.txt ]; then \
        uv pip install --no-cache-dir -r ${COMFY_HOME}/custom_nodes/ComfyUI-Custom-Scripts/requirements.txt; \
    fi

# Easy-Use — yolain workflow simplifiers
RUN git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use.git \
        ${COMFY_HOME}/custom_nodes/ComfyUI-Easy-Use && \
    if [ -f ${COMFY_HOME}/custom_nodes/ComfyUI-Easy-Use/requirements.txt ]; then \
        uv pip install --no-cache-dir -r ${COMFY_HOME}/custom_nodes/ComfyUI-Easy-Use/requirements.txt; \
    fi

# VideoHelperSuite — VHS_VideoCombine + load/save video nodes (LTX/Wan output)
RUN git clone --depth=1 https://github.com/kosinkadink/ComfyUI-VideoHelperSuite.git \
        ${COMFY_HOME}/custom_nodes/ComfyUI-VideoHelperSuite && \
    if [ -f ${COMFY_HOME}/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt ]; then \
        uv pip install --no-cache-dir -r ${COMFY_HOME}/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; \
    fi

# Frame-Interpolation + RIFE47 weight bake. The wget runs in the same layer
# as the clone so a missing-weights image never gets cached.
RUN git clone --depth=1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
        ${COMFY_HOME}/custom_nodes/ComfyUI-Frame-Interpolation && \
    if [ -f ${COMFY_HOME}/custom_nodes/ComfyUI-Frame-Interpolation/requirements.txt ]; then \
        uv pip install --no-cache-dir -r ${COMFY_HOME}/custom_nodes/ComfyUI-Frame-Interpolation/requirements.txt; \
    fi && \
    mkdir -p ${COMFY_HOME}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife && \
    wget -q -O ${COMFY_HOME}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife47.pth \
        https://huggingface.co/wavespeed/misc/resolve/main/rife/rife47.pth

# =============================================================================
# Baked model weights — Wan 2.2 I2V stack (text encoder + 4-step Lightning
# LoRAs + Wan 2.1 VAE). The Wan 2.2 diffusion checkpoint itself is NOT baked
# here; load it from /runpod-volume/models/diffusion_models/ via extra_paths.yaml,
# or add another RUN below if you want it in-image.
# One RUN per file (or LoRA pair) so changing one URL only re-downloads that.
# =============================================================================

# Model directories — created once so each download RUN can write directly.
RUN mkdir -p ${COMFY_HOME}/models/checkpoints \
             ${COMFY_HOME}/models/vae \
             ${COMFY_HOME}/models/unet \
             ${COMFY_HOME}/models/clip \
             ${COMFY_HOME}/models/text_encoders \
             ${COMFY_HOME}/models/diffusion_models \
             ${COMFY_HOME}/models/model_patches \
             ${COMFY_HOME}/models/loras

# UMT5-XXL text encoder for Wan (FP8-scaled). Largest of the four (~6 GB);
# put it earliest so smaller-item URL edits don't bust this layer.
RUN wget -q -O ${COMFY_HOME}/models/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors \
        https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors

# Wan 2.2 Lightning 4-step LoRAs (HIGH + LOW noise), paired — they're always
# loaded together, so one RUN keeps cache invalidation atomic.
RUN wget -q -O ${COMFY_HOME}/models/loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors \
        https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors && \
    wget -q -O ${COMFY_HOME}/models/loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors \
        https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors

# Wan 2.1 VAE — used by Wan 2.2 I2V workflows.
RUN wget -q -O ${COMFY_HOME}/models/vae/wan_2.1_vae.safetensors \
        https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors

# -----------------------------------------------------------------------------
# 6) Handler last — small layer, frequent edits, doesn't bust anything heavier.
# launch_comfy.py is a thin wrapper that sets allocator-related environment
# before handing off to ComfyUI's main.py.
# -----------------------------------------------------------------------------
COPY scripts /workspace/scripts
COPY handler.py /workspace/handler.py
COPY launch_comfy.py /workspace/launch_comfy.py
COPY test_input.json /workspace/test_input.json

# -----------------------------------------------------------------------------
# Runtime tunables — override at deploy time via RunPod env vars.
#   COMFY_HOST/PORT          : ComfyUI bind address (loopback by default)
#   COMFY_EXTRA_ARGS         : extra flags appended to ComfyUI's main.py
#                              (e.g. "--reserve-vram 1.5 --fp8_e4m3fn-text-enc")
#   MIN_VRAM_GB              : fitness gate; pod self-terminates if below
#   REQUIRED_SM_MAJOR/MINOR  : required compute capability (12.0 = Blackwell)
#   SERVER_READY_TIMEOUT_S   : how long to wait for ComfyUI to come up
# -----------------------------------------------------------------------------
ENV COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    COMFY_EXTRA_ARGS="" \
    MIN_VRAM_GB=24 \
    REQUIRED_SM_MAJOR=12 \
    REQUIRED_SM_MINOR=0 \
    SERVER_READY_TIMEOUT_S=120

# Handler is PID 1 — no shell wrapper, no start.sh, no SSH.
# `-u` keeps stdout unbuffered for clean RunPod logs.
CMD ["python", "-u", "/workspace/handler.py"]
