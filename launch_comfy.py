"""ComfyUI launcher with deterministic allocator startup.

Runs `python main.py` indirectly so we can set process environment before
ComfyUI imports torch.

Optional pre-config steps are wrapped in try/except so a future package rename
or API change downgrades to a stderr warning, not a boot failure.
"""

import os
import sys


# ComfyUI defaults to cudaMallocAsync. Set it before any optional accelerator
# package can import torch, or PyTorch may observe two different backends.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")


# Python uses the script's own directory as sys.path[0]. Since this file lives
# at /workspace/ and ComfyUI's `comfy/` package is at /workspace/ComfyUI/comfy,
# `import comfy` would fail when we exec main.py. handler.py launches us with
# cwd=COMFY_HOME, so prepending cwd to sys.path mirrors what `python main.py`
# would have given us natively.
sys.path.insert(0, os.getcwd())


# Keep accelerator preloads opt-in. Importing comfy_kitchen here may import
# torch before ComfyUI's startup has finished, which is exactly the ordering
# ComfyUI warns about. If you need to experiment with this again, set
# COMFY_PRELOAD_COMFY_KITCHEN=1 and keep PYTORCH_CUDA_ALLOC_CONF explicit.
if os.environ.get("COMFY_PRELOAD_COMFY_KITCHEN", "").lower() in {"1", "true", "yes", "on"}:
    try:
        import comfy_kitchen as _ck

        _ck.enable_backend("triton")
    except Exception as e:
        print(f"[launch_comfy] comfy_kitchen triton enable skipped: {e!r}", file=sys.stderr)


# NOTE: comfy-aimdo log-level configuration intentionally omitted. The 0.3.0
# `control` module's actual API surface couldn't be verified from PyPI/GitHub
# (raw repo path 404'd), and the previous guess (`set_log_level_info`) was
# wrong. The package's C-side DEBUG line at boot is cosmetic; revisit when
# we can introspect the installed package directly.


# Hand off to ComfyUI's main as if it had been invoked directly.
sys.argv[0] = "main.py"
with open("main.py", "rb") as _f:
    _src = _f.read()
exec(compile(_src, "main.py", "exec"), {"__name__": "__main__", "__file__": "main.py"})
