"""ComfyUI launcher that pre-configures accelerator packages.

comfy-kitchen and comfy-aimdo expose only Python-API config knobs (verified
against their GitHub repos), so we have to import + configure them before
ComfyUI itself does. Running this once at process start is enough; the
package-level state persists for the lifetime of the interpreter.

Each pre-config step is wrapped in try/except so a future package rename or
API change downgrades to a stderr warning, not a boot failure.
"""

import sys


# Quiet the C-side DEBUG hooks from comfy-aimdo.
try:
    from comfy_aimdo import control as _aimdo

    _aimdo.set_log_level_info()
except Exception as e:
    print(f"[launch_comfy] aimdo log-level config skipped: {e!r}", file=sys.stderr)


# Enable the Triton backend in comfy-kitchen so Wan/LTX video diffusion gets
# NVFP4 / MXFP8 quantize-dequantize on Blackwell SM12.0.
try:
    import comfy_kitchen as _ck

    _ck.enable_backend("triton")
except Exception as e:
    print(f"[launch_comfy] comfy_kitchen triton enable skipped: {e!r}", file=sys.stderr)


# Hand off to ComfyUI's main as if it had been invoked directly. cwd is set
# by the parent (handler.py) to /workspace/ComfyUI, so main.py resolves.
sys.argv[0] = "main.py"
with open("main.py", "rb") as _f:
    _src = _f.read()
exec(compile(_src, "main.py", "exec"), {"__name__": "__main__", "__file__": "main.py"})
