"""RunPod Serverless ComfyUI worker — Blackwell / CUDA 13 / PyTorch 2.11.

Pipeline per cold start:
    fitness_check()  -> launch_comfyui()  -> wait_for_server()  -> runpod.serverless.start()

Per job:
    validate -> upload_images -> queue_prompt -> wait_for_completion -> collect_outputs
"""

from __future__ import annotations

import atexit
import base64
import binascii
import json
import logging
import mimetypes
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from typing import Any
from urllib.parse import urlencode

import requests
import websocket  # websocket-client

# ─── Config ──────────────────────────────────────────────────────────────────

COMFY_HOME = os.environ.get("COMFY_HOME", "/workspace/ComfyUI")
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_EXTRA_ARGS = os.environ.get("COMFY_EXTRA_ARGS", "")

MIN_VRAM_GB = float(os.environ.get("MIN_VRAM_GB", "24"))
REQUIRED_SM_MAJOR = int(os.environ.get("REQUIRED_SM_MAJOR", "12"))
REQUIRED_SM_MINOR = int(os.environ.get("REQUIRED_SM_MINOR", "0"))

SERVER_READY_TIMEOUT_S = float(os.environ.get("SERVER_READY_TIMEOUT_S", "120"))
SERVER_POLL_INTERVAL_S = 0.2
WS_CONNECT_TIMEOUT_S = float(os.environ.get("WS_CONNECT_TIMEOUT_S", "30"))
WS_RECV_TIMEOUT_S = float(os.environ.get("WS_RECV_TIMEOUT_S", "0"))

COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_WS = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"

# ComfyUI enables cudaMallocAsync during its early startup. The value must be
# present before either this parent process or the ComfyUI child imports torch;
# otherwise PyTorch can load with one allocator backend and later see another.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("comfy-worker")


# ─── Fitness check ───────────────────────────────────────────────────────────
# Runs before ComfyUI launches. On any failure: log structured reason and
# os._exit(1) so RunPod releases the pod immediately. No GPU work is wasted.

def fitness_check() -> None:
    t0 = time.perf_counter()
    try:
        import torch  # heavy import, but unavoidable to verify CUDA path
    except Exception as e:
        _fail(f"torch import failed: {e!r}")

    if not torch.cuda.is_available():
        _fail("torch.cuda.is_available() == False (no CUDA runtime / driver mismatch)")

    n = torch.cuda.device_count()
    if n < 1:
        _fail("torch.cuda.device_count() == 0")

    cap = torch.cuda.get_device_capability(0)
    required = (REQUIRED_SM_MAJOR, REQUIRED_SM_MINOR)
    if cap < required:
        _fail(f"compute capability {cap} < required {required} (expected Blackwell SM{REQUIRED_SM_MAJOR}.{REQUIRED_SM_MINOR})")

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    if total_gb < MIN_VRAM_GB:
        _fail(f"VRAM {total_gb:.1f}GB < required {MIN_VRAM_GB}GB on device '{props.name}'")

    try:
        # Smoke-test driver/runtime alignment with a 1-element alloc.
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except Exception as e:
        _fail(f"CUDA alloc smoke-test failed: {e!r}")

    dt_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "fitness OK: device='%s' cap=%s vram=%.1fGB torch=%s cuda=%s allocator=%s (took %.0fms)",
        props.name,
        cap,
        total_gb,
        torch.__version__,
        torch.version.cuda,
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        dt_ms,
    )


def _fail(reason: str) -> None:
    log.error("Fitness check failed: %s", reason)
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(1)


# ─── ComfyUI lifecycle ───────────────────────────────────────────────────────

_comfy_proc: subprocess.Popen | None = None


def launch_comfyui() -> None:
    global _comfy_proc
    if _comfy_proc is not None:
        return

    # /workspace/launch_comfy.py is a thin wrapper that sets allocator-related
    # environment before exec()ing main.py with our argv.
    args = [
        sys.executable, "/workspace/launch_comfy.py",
        "--listen", COMFY_HOST,
        "--port", str(COMFY_PORT),
        "--disable-auto-launch",
        "--disable-metadata",
        "--use-sage-attention",
    ]
    if COMFY_EXTRA_ARGS.strip():
        args.extend(shlex.split(COMFY_EXTRA_ARGS))

    log.info("Launching ComfyUI: %s (cwd=%s)", " ".join(args), COMFY_HOME)
    _comfy_proc = subprocess.Popen(
        args,
        cwd=COMFY_HOME,
        # stdout/stderr inherited from PID 1 → RunPod logs.
        # New session keeps signals from double-delivering through the parent.
        start_new_session=True,
    )
    atexit.register(_terminate_comfyui)
    # On SIGTERM/SIGINT: sys.exit(0) triggers atexit which kills the child.
    # A bare lambda that just terminates the child would leave the parent
    # blocked in ws.recv() / runpod's event loop, defeating graceful shutdown.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))


def _terminate_comfyui() -> None:
    global _comfy_proc
    if _comfy_proc is None or _comfy_proc.poll() is not None:
        return
    log.info("Terminating ComfyUI (pid=%d)", _comfy_proc.pid)
    try:
        _comfy_proc.terminate()
        _comfy_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _comfy_proc.kill()
    except Exception as e:
        log.warning("ComfyUI shutdown error: %r", e)


def wait_for_server() -> None:
    deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
    last_err: str = ""
    while time.monotonic() < deadline:
        # Detect early subprocess crash so we surface ComfyUI's own error,
        # not an opaque "server never came up" timeout.
        if _comfy_proc is not None and _comfy_proc.poll() is not None:
            _fail(f"ComfyUI exited during startup with code {_comfy_proc.returncode}")
        try:
            r = requests.get(f"{COMFY_BASE}/system_stats", timeout=2)
            if r.status_code == 200:
                log.info("ComfyUI ready at %s", COMFY_BASE)
                return
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = repr(e)
        time.sleep(SERVER_POLL_INTERVAL_S)
    _fail(f"ComfyUI did not become ready within {SERVER_READY_TIMEOUT_S}s (last: {last_err})")


# ─── Job execution ───────────────────────────────────────────────────────────

class WorkflowError(RuntimeError):
    pass


def validate_input(job_input: Any) -> tuple[dict, list[dict]]:
    if not isinstance(job_input, dict):
        raise ValueError("job 'input' must be an object")

    workflow = job_input.get("workflow")
    if isinstance(workflow, str):
        try:
            workflow = json.loads(workflow)
        except json.JSONDecodeError as e:
            raise ValueError(f"'workflow' is not valid JSON: {e.msg}") from e
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("'workflow' is required and must be a non-empty object")

    images = job_input.get("images") or []
    if not isinstance(images, list):
        raise ValueError("'images' must be a list when provided")
    for i, img in enumerate(images):
        if not isinstance(img, dict) or "name" not in img or "image" not in img:
            raise ValueError(f"images[{i}] must be {{'name': str, 'image': base64-or-data-uri str}}")
        if not isinstance(img["name"], str) or not isinstance(img["image"], str):
            raise ValueError(f"images[{i}]: 'name' and 'image' must be strings")

    return workflow, images


def _decode_b64_image(s: str) -> bytes:
    # Accept raw base64 or a data: URI.
    if s.startswith("data:"):
        comma = s.find(",")
        if comma < 0:
            raise ValueError("malformed data URI: missing ','")
        s = s[comma + 1:]
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid base64 image: {e}") from e


def upload_images(images: list[dict]) -> None:
    for img in images:
        data = _decode_b64_image(img["image"])
        files = {"image": (img["name"], data, "application/octet-stream")}
        payload = {"overwrite": "true"}
        r = requests.post(f"{COMFY_BASE}/upload/image", files=files, data=payload, timeout=30)
        if r.status_code != 200:
            raise WorkflowError(f"upload of {img['name']!r} failed: HTTP {r.status_code} {r.text[:300]}")


def queue_prompt(workflow: dict, client_id: str) -> str:
    body = {"prompt": workflow, "client_id": client_id}
    r = requests.post(f"{COMFY_BASE}/prompt", json=body, timeout=30)
    if r.status_code != 200:
        # ComfyUI returns structured per-node validation errors here — pass them through.
        try:
            details = r.json()
        except ValueError:
            details = {"raw": r.text[:1000]}
        raise WorkflowError(f"/prompt rejected (HTTP {r.status_code}): {json.dumps(details)[:1500]}")
    prompt_id = r.json().get("prompt_id")
    if not prompt_id:
        raise WorkflowError(f"/prompt response missing prompt_id: {r.text[:300]}")
    return prompt_id


def wait_for_completion(prompt_id: str, client_id: str) -> None:
    url = f"{COMFY_WS}?{urlencode({'clientId': client_id})}"
    ws = websocket.create_connection(url, timeout=WS_CONNECT_TIMEOUT_S)
    ws.settimeout(WS_RECV_TIMEOUT_S if WS_RECV_TIMEOUT_S > 0 else None)
    try:
        while True:
            msg = ws.recv()
            if not isinstance(msg, str):
                # Binary frames are progress previews; ignore.
                continue
            try:
                evt = json.loads(msg)
            except json.JSONDecodeError:
                continue

            etype = evt.get("type")
            data = evt.get("data") or {}
            if data.get("prompt_id") not in (None, prompt_id):
                continue

            if etype == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
                return  # done
            if etype == "execution_error":
                raise WorkflowError(
                    f"node {data.get('node_id')} ({data.get('node_type')}) raised: "
                    f"{data.get('exception_type')}: {data.get('exception_message')}"
                )
            if etype == "execution_interrupted":
                raise WorkflowError(f"execution interrupted at node {data.get('node_id')}")
    finally:
        try:
            ws.close()
        except Exception:
            pass


# ComfyUI emits outputs under varying keys depending on the SaveX node used.
# These are the keys we walk. Anything else we encounter is reported as 'other'.
_OUTPUT_KEYS = ("images", "gifs", "videos", "audio")


def _classify(filename: str, parent_key: str) -> tuple[str, str]:
    """Return (kind, mime). kind ∈ {image,video,gif,audio,other}."""
    ext = os.path.splitext(filename)[1].lower()
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        return "image", mime
    if ext in (".mp4", ".webm", ".mov", ".mkv", ".avi"):
        return "video", mime
    if ext == ".gif":
        return "gif", mime
    if ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        return "audio", mime
    # Fall back to the parent key as a hint (VHS_VideoCombine emits mp4 under "gifs").
    if parent_key == "gifs":
        return "video" if ext in (".mp4", ".webm") else "gif", mime
    return "other", mime


def collect_outputs(prompt_id: str) -> list[dict]:
    r = requests.get(f"{COMFY_BASE}/history/{prompt_id}", timeout=30)
    r.raise_for_status()
    history = r.json().get(prompt_id, {})
    outputs = history.get("outputs", {})
    if not outputs:
        raise WorkflowError("workflow finished but produced no outputs")

    results: list[dict] = []
    for node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        for key, entries in node_out.items():
            if key not in _OUTPUT_KEYS or not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict) or "filename" not in entry:
                    continue
                if entry.get("type") == "temp":
                    continue
                params = {
                    "filename": entry["filename"],
                    "subfolder": entry.get("subfolder", ""),
                    "type": entry.get("type", "output"),
                }
                vr = requests.get(f"{COMFY_BASE}/view", params=params, timeout=120)
                vr.raise_for_status()
                kind, mime = _classify(entry["filename"], key)
                results.append({
                    "node_id": node_id,
                    "filename": entry["filename"],
                    "type": kind,
                    "mime": mime,
                    "data": base64.b64encode(vr.content).decode("ascii"),
                })

    if not results:
        raise WorkflowError("workflow finished but no persistable outputs were collected")
    return results


def handler(job: dict) -> dict:
    job_id = job.get("id", "?")
    log.info("job %s: starting", job_id)
    try:
        workflow, images = validate_input(job.get("input") or {})
    except ValueError as e:
        return {"error": f"invalid input: {e}"}

    client_id = uuid.uuid4().hex
    try:
        if images:
            upload_images(images)
        prompt_id = queue_prompt(workflow, client_id)
        log.info("job %s: queued prompt_id=%s", job_id, prompt_id)
        wait_for_completion(prompt_id, client_id)
        outputs = collect_outputs(prompt_id)
    except WorkflowError as e:
        log.warning("job %s: workflow error: %s", job_id, e)
        return {"error": str(e)}
    except Exception as e:
        log.exception("job %s: unexpected error", job_id)
        return {"error": f"unexpected: {e!r}"}

    log.info("job %s: returning %d outputs", job_id, len(outputs))
    return {"outputs": outputs}


# ─── Bootstrap ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fitness_check()
    launch_comfyui()
    wait_for_server()

    import runpod  # imported late so a fitness failure can't be masked by SDK init
    runpod.serverless.start({"handler": handler})
