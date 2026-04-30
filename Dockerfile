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
    UV_NO_CACHE=1

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
ENV COMFY_HOME=/workspace/ComfyUI \
    VIRTUAL_ENV=/workspace/venv \
    PATH=/workspace/venv/bin:/usr/local/bin:/usr/bin:/bin

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
# USER-EXTENSION LAYER — bake your custom nodes and models BELOW this line.
# Cache-busting your weights only happens when you edit this section, not when
# you tweak handler.py.
#
# Examples (uncomment + adapt):
#
#   RUN cd ${COMFY_HOME}/custom_nodes && \
#       git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
#       git clone --depth=1 https://github.com/kijai/ComfyUI-WanVideoWrapper && \
#       git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo
#
#   # If a custom node has its own requirements.txt:
#   RUN find ${COMFY_HOME}/custom_nodes -maxdepth 2 -name requirements.txt \
#         -exec uv pip install --no-cache-dir -r {} \;
#
#   # Bake LTX 2.3 / Wan 2.2 weights:
#   RUN mkdir -p ${COMFY_HOME}/models/diffusion_models && \
#       wget -O ${COMFY_HOME}/models/diffusion_models/wan22.safetensors \
#            https://huggingface.co/.../wan22.safetensors
# =============================================================================

# -----------------------------------------------------------------------------
# 6) Handler last — small layer, frequent edits, doesn't bust anything heavier.
# launch_comfy.py is a thin wrapper that pre-configures comfy-kitchen (enables
# the Triton backend for NVFP4/MXFP8 on Blackwell) and comfy-aimdo (sets log
# level to INFO) before handing off to ComfyUI's main.py.
# -----------------------------------------------------------------------------
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
