"""ComfyUI launcher that pre-configures accelerator packages.

Runs `python main.py` indirectly so we can configure package-level state
(e.g. comfy-kitchen's Triton backend) before ComfyUI imports those packages.

Each pre-config step is wrapped in try/except so a future package rename or
API change downgrades to a stderr warning, not a boot failure.
"""

import os
import sys


# Python uses the script's own directory as sys.path[0]. Since this file lives
# at /workspace/ and ComfyUI's `comfy/` package is at /workspace/ComfyUI/comfy,
# `import comfy` would fail when we exec main.py. handler.py launches us with
# cwd=COMFY_HOME, so prepending cwd to sys.path mirrors what `python main.py`
# would have given us natively.
sys.path.insert(0, os.getcwd())


# Enable the Triton backend in comfy-kitchen so Wan/LTX video diffusion gets
# NVFP4 / MXFP8 quantize-dequantize on Blackwell SM12.0. The package
# auto-registers backends at import time but ships triton disabled by default;
# enable_backend("triton") flips that flag before ComfyUI consults the registry.
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
