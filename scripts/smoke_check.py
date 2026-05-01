"""Container smoke check for the RunPod ComfyUI image.

Run inside the built image to verify CUDA, torch, and allocator startup before
debugging ComfyUI workflows.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import traceback


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
        else:
            report["cuda_alloc_smoke_test"] = "skipped"
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("cuda_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
