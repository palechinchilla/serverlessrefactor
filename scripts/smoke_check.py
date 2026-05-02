"""Container smoke check for the RunPod ComfyUI image.

Run inside the built image to verify CUDA, torch, and allocator startup before
debugging ComfyUI workflows.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import sysconfig
import traceback
from pathlib import Path


def _check_python_h(report: dict[str, object]) -> bool:
    include_dir = sysconfig.get_paths().get("include") or ""
    python_h = Path(include_dir) / "Python.h" if include_dir else None
    available = bool(python_h and python_h.exists())
    report["python_include_dir"] = include_dir
    report["python_h_path"] = str(python_h) if python_h else ""
    report["python_h_available"] = available
    return available


def _run_torch_compile_smoke(torch) -> None:
    @torch.compile(backend="inductor")
    def tiny_inductor_fn(x):
        return torch.sin(x) + (x * 2)

    x = torch.randn(8, device="cuda")
    y = tiny_inductor_fn(x)
    expected = torch.sin(x) + (x * 2)
    torch.cuda.synchronize()
    if not torch.allclose(y, expected):
        raise RuntimeError("torch.compile smoke output mismatch")


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }

    try:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
        _check_python_h(report)
        if torch.cuda.is_available() and torch.cuda.device_count():
            props = torch.cuda.get_device_properties(0)
            report.update(
                {
                    "device_name": props.name,
                    "device_capability": torch.cuda.get_device_capability(0),
                    "device_vram_gb": round(props.total_memory / (1024**3), 2),
                }
            )
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            report["cuda_alloc_smoke_test"] = "ok"
            report["torch_compile_smoke_test"] = "failed"
            if not report["python_h_available"]:
                raise RuntimeError(f"Python.h not found at {report['python_h_path']}")
            _run_torch_compile_smoke(torch)
            report["torch_compile_smoke_test"] = "ok"
        else:
            report["cuda_alloc_smoke_test"] = "skipped"
            report["torch_compile_smoke_test"] = "skipped"
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("cuda_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
