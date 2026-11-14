#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that the CUDA libraries rawatta loads at run time are present and work.

rawatta dlopens libnvrtc.so.12 (NVRTC, which loads libnvrtc-builtins on first compile) and
libcublasLt.so.12 by soname, the same way this script does. Neither check needs a GPU or the
driver, so the image build runs it and fails if a library is missing. libcuda.so.1 comes from
the host driver at run time (`docker run --gpus ...`); `rawatta info` checks that part.

    python3 docker/check_cuda_libs.py
"""
from __future__ import annotations

import argparse
import ctypes
import sys

# The NVRTC targets the engine compiles for (rawatta_nn::nvrtc_arch): Ampere, Ada, Hopper, Blackwell.
ARCHES = ("compute_80", "compute_86", "compute_87", "compute_89", "compute_90", "compute_100", "compute_120")
KERNEL = b'extern "C" __global__ void rawatta_probe(float* x) { x[threadIdx.x] *= 2.0f; }'


def nvrtc_check(lib: ctypes.CDLL) -> str:
    major, minor = ctypes.c_int(), ctypes.c_int()
    if lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) != 0:
        raise RuntimeError("nvrtcVersion failed")
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    for arch in ARCHES:
        prog = ctypes.c_void_p()
        rc = lib.nvrtcCreateProgram(ctypes.byref(prog), KERNEL, b"rawatta_probe.cu", 0, None, None)
        if rc != 0:
            raise RuntimeError(f"nvrtcCreateProgram: {lib.nvrtcGetErrorString(rc).decode()}")
        opts = (ctypes.c_char_p * 2)(f"--gpu-architecture={arch}".encode(), b"--std=c++17")
        rc = lib.nvrtcCompileProgram(prog, 2, opts)
        size = ctypes.c_size_t()
        if rc == 0:
            rc = lib.nvrtcGetPTXSize(prog, ctypes.byref(size))
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
        if rc != 0 or size.value == 0:
            raise RuntimeError(f"NVRTC compile for {arch}: {lib.nvrtcGetErrorString(rc).decode()}")
    return f"{major.value}.{minor.value}"


def cublaslt_check(lib: ctypes.CDLL) -> str:
    lib.cublasLtGetVersion.restype = ctypes.c_size_t
    v = lib.cublasLtGetVersion()
    if v == 0:
        raise RuntimeError("cublasLtGetVersion returned 0")
    return f"{v // 10000}.{v // 100 % 100}.{v % 100}"


def main() -> int:
    argparse.ArgumentParser(
        prog="check_cuda_libs.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    ok = True
    for soname, check in (("libnvrtc.so.12", nvrtc_check), ("libcublasLt.so.12", cublaslt_check)):
        try:
            version = check(ctypes.CDLL(soname))
        except (OSError, AttributeError, RuntimeError) as e:
            print(f"{soname}: FAILED: {e}", file=sys.stderr)
            ok = False
            continue
        extra = f", compiled for {', '.join(ARCHES)}" if check is nvrtc_check else ""
        print(f"{soname}: ok (version {version}{extra})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
