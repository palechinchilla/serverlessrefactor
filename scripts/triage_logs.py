"""Summarize common RunPod/ComfyUI startup failures from a text log."""

from __future__ import annotations

import argparse
from pathlib import Path


PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "allocator_mismatch",
        "Allocator backend parsed at runtime != allocator backend parsed at load time",
    ),
    ("torch_imported_early", "Torch already imported"),
    ("comfy_startup_exit", "ComfyUI exited during startup"),
    ("traceback", "Traceback (most recent call last):"),
    ("runtime_error", "RuntimeError:"),
    ("server_timeout", "ComfyUI did not become ready"),
    ("cuda_unavailable", "torch.cuda.is_available() == False"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    args = parser.parse_args()

    counts = {name: 0 for name, _ in PATTERNS}
    first_hits: dict[str, str] = {}

    with args.log_file.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            for name, needle in PATTERNS:
                if needle in line:
                    counts[name] += 1
                    first_hits.setdefault(name, f"{line_no}: {line.strip()}")

    print("RunPod ComfyUI log triage")
    print(f"log: {args.log_file}")
    for name, count in counts.items():
        if count:
            print(f"- {name}: {count}")
            print(f"  first: {first_hits[name]}")

    if counts["allocator_mismatch"]:
        print()
        print("Likely cause: torch loaded before ComfyUI settled PYTORCH_CUDA_ALLOC_CONF.")
        print("Check launch-time imports and keep the allocator env explicit before torch imports.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
