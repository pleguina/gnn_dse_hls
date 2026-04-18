#!/usr/bin/env python3
"""Quick readiness check for OMTF GPU training on this host."""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def check_nvidia() -> bool:
    rc, out, err = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_mode",
        "--format=csv,noheader",
    ])
    if rc != 0:
        print("[FAIL] nvidia-smi unavailable")
        if err:
            print(err)
        return False

    print("GPU inventory:")
    print(out)

    prohibited = [line for line in out.splitlines() if "Prohibited" in line]
    if prohibited:
        print("[FAIL] Compute mode is Prohibited on at least one GPU.")
        return False

    print("[OK] Compute mode allows CUDA jobs.")
    return True


def check_torch() -> bool:
    try:
        import torch
    except Exception as e:
        print(f"[FAIL] torch import failed: {e}")
        return False

    print(f"torch: {torch.__version__}")

    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() is False")
        return False

    arch = torch.cuda.get_arch_list()
    print(f"CUDA arch list: {arch}")
    if "sm_60" not in arch:
        print("[FAIL] Installed torch build does not include sm_60 (Tesla P100).")
        return False

    try:
        _ = torch.empty(1, device="cuda")
    except Exception as e:
        print(f"[FAIL] CUDA allocation failed: {e}")
        return False

    print("[OK] CUDA allocation succeeded.")
    return True


def main() -> int:
    print("== OMTF GPU readiness check ==")
    ok_nvidia = check_nvidia()
    ok_torch = check_torch()

    if ok_nvidia and ok_torch:
        print("READY: GPU training should work on this host.")
        return 0

    print("NOT READY: fix issues above before starting GPU training.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
