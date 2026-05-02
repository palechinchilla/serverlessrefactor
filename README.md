# RunPod Serverless ComfyUI Worker

This image runs ComfyUI behind a RunPod Serverless handler. It is pinned for a
Blackwell GPU target and keeps ComfyUI bound to loopback inside the worker.

## Current Log Diagnosis

The attached log shows the GPU fitness check passing, then ComfyUI exits during
startup:

```text
WARNING: Potential Error in code: Torch already imported
RuntimeError: Allocator backend parsed at runtime != allocator backend parsed at load time, cudaMallocAsync != native
```

That means the driver and GPU are visible, but PyTorch is being imported before
ComfyUI finishes setting its CUDA allocator mode. The worker now makes
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` explicit before either the
handler or ComfyUI child can import `torch`, and `comfy_kitchen` preloading is
disabled by default to avoid importing torch too early.

## Build

```bash
docker build -t comfy-cli-serverless:local .
```

## Smoke Check

Run this before testing workflows:

```bash
docker run --rm --gpus all comfy-cli-serverless:local \
  python -u /workspace/scripts/smoke_check.py
```

Or with Compose on a GPU-capable Docker host:

```bash
docker compose -f docker-compose.smoke.yml up --build --abort-on-container-exit
```

The smoke check should report `cuda_available: true` and
`cuda_alloc_smoke_test: ok`.

## RunPod Environment

Recommended defaults:

```text
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
COMFY_PRELOAD_COMFY_KITCHEN=0
COMFY_EXTRA_ARGS=
MIN_VRAM_GB=24
REQUIRED_SM_MAJOR=12
REQUIRED_SM_MINOR=0
WS_CONNECT_TIMEOUT_S=30
WS_RECV_TIMEOUT_S=0
```

If you deliberately set `PYTORCH_CUDA_ALLOC_CONF=backend:native`, also add
`--disable-cuda-malloc` to `COMFY_EXTRA_ARGS` so ComfyUI does not switch the
allocator after torch has loaded.

`WS_RECV_TIMEOUT_S=0` leaves the ComfyUI websocket receive open during long
quiet decode, interpolation, or save stages. Keep `WS_CONNECT_TIMEOUT_S` finite
so an unreachable ComfyUI server still fails promptly.

## Log Triage

For a downloaded RunPod log:

```bash
python scripts/triage_logs.py "C:\Users\Admin\Downloads\logs(120).txt"
```

The important first check is whether the fitness check passes. If it does, GPU
visibility is probably fine and the next failures are usually ComfyUI startup,
custom node imports, missing model paths, or workflow validation errors.
