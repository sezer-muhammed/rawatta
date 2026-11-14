#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert the official VGGT-Omega checkpoint into the safetensors files the
rawatta engine loads.

    python3 tools/convert_weights.py --input vggt_omega_1b_512.pt \
        [--output vggt_omega_1b_512.safetensors] [--fp8 [PATH]] [--use-torch]

PyTorch is NOT needed. The checkpoint is a torch zip archive (`<prefix>/data.pkl`
plus one raw file per storage under `<prefix>/data/<key>`). It is read with a
restricted unpickler that knows only the handful of globals a tensor state dict
uses (`collections.OrderedDict`, `torch._utils._rebuild_tensor_v2/_v3`,
`torch._utils._rebuild_parameter*`, `torch._tensor._rebuild_from_type_v2`, the
`torch.*Storage` types and dtypes) and refuses everything else, so no code from
the file is ever run; a malformed pickle is reported as an error, not a traceback.
Storages are memory-mapped with numpy; BF16 is handled as
uint16 bit patterns. `--use-torch` loads the checkpoint with
`torch.load(weights_only=True)` instead and uses torch for the BF16/FP8 casts.
Both paths write byte-identical files, which are also byte-identical to the
original converter of the authors' research codebase (tested in
tools/tests/test_convert_weights.py).

Output layout (format `rawatta-vggt-1`, read by crates/vggt-omega/src/weights.rs):

- `aggregator.*` first: the device region, uploaded to the GPU in one copy.
  2-D weights (Linear; the patch conv reshaped to [1024, 768] in (c, ky, kx)
  order) and their biases in BF16; LayerNorm parameters, LayerScale gammas,
  tokens and RoPE periods in F32.
- `camera_head.*`, `dense_head.*` after it, all F32: the heads read them on
  the host and upload their own layouts.
- Every tensor starts at a 256-byte offset of the data section; gaps are
  filled with `__pad.<n>` U8 tensors so the file stays standard safetensors.
- `qkv.bias_mask` buffers are folded in: the effective bias is
  `bias * bias_mask`. All aggregator masks of the released checkpoint are
  zero, so those biases are dropped; the camera head keeps [b_q, 0, b_v].
- Dropped: `mask_token`, `text_alignment_head.*`, every `bias_mask`.
- `__metadata__`: `format = rawatta-vggt-1`, `source` (input file name),
  `device_bytes` (length of the device region from the start of the data
  section).

`--fp8` writes a second file (`format = rawatta-vggt-1-fp8`) with FP8 E4M3 copies
of the aggregator Linear weights (`<name>` F8_E4M3, per-output-channel
`<name>_scale` F32 = amax / 448). Without a path it is written next to the
output as `<stem>_fp8.safetensors`, the name the engine looks for. The
RAWATTA settings do not use it (FP8 GEMMs were rejected by the calibration
check); it is only needed for `+fp8` research configs.

The weights are under the FAIR Noncommercial Research License: research use
only; never redistribute the converted files or bake them into an image.
"""
from __future__ import annotations

import _compat_pickle
import argparse
import codecs
import json
import math
import os
import pickle
import struct
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ALIGN = 256
DEVICE_PREFIX = "aggregator."
HOST_PREFIXES = ("camera_head.", "dense_head.")
PATCH_WEIGHT = "aggregator.patch_embed.patch_embed.proj.weight"
FP8_MAX = 448.0
DEVICE_END = "__device_end__"
MAX_COMPRESSED_STORAGE = 1 << 31  # bytes
MAX_EXPANDED_NUMEL = 1 << 28  # elements of a view with more elements than its storage

# --------------------------------------------------------------------------
# Torch-free checkpoint reader
# --------------------------------------------------------------------------

# Legacy typed-storage class name -> (numpy dtype char code without byte order, our dtype tag).
_STORAGE_TYPES = {
    "FloatStorage": ("f4", "F32"),
    "DoubleStorage": ("f8", "F64"),
    "HalfStorage": ("f2", "F16"),
    "BFloat16Storage": ("u2", "BF16"),
    "LongStorage": ("i8", "I64"),
    "IntStorage": ("i4", "I32"),
    "ShortStorage": ("i2", "I16"),
    "CharStorage": ("i1", "I8"),
    "ByteStorage": ("u1", "U8"),
    "BoolStorage": ("b1", "BOOL"),
    "UntypedStorage": ("u1", "U8"),
}
# torch dtype attribute name -> same tag (for _rebuild_tensor_v3's explicit dtype).
_DTYPES = {
    "float32": "F32", "float": "F32", "float64": "F64", "double": "F64",
    "float16": "F16", "half": "F16", "bfloat16": "BF16", "int64": "I64", "long": "I64",
    "int32": "I32", "int": "I32", "int16": "I16", "short": "I16", "int8": "I8",
    "uint8": "U8", "bool": "BOOL",
}
_TAG_NP = {"F32": "f4", "F64": "f8", "F16": "f2", "BF16": "u2", "I64": "i8", "I32": "i4",
           "I16": "i2", "I8": "i1", "U8": "u1", "BOOL": "b1"}


class CheckpointError(Exception):
    """The checkpoint is not a tensor state dict this reader understands."""


class _StorageType:
    def __init__(self, tag: str):
        self.tag = tag


class _DType:
    def __init__(self, tag: str):
        self.tag = tag


class _Marker:
    """A class reference (torch.Tensor / Parameter / Size / device) that needs no behaviour."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args):  # torch.Size((…)) / torch.device("cpu")
        if self.name == "torch.Size":
            return tuple(args[0]) if args else ()
        return args[0] if len(args) == 1 else args


class Storage:
    """One `data/<key>` record: raw little/big-endian bytes, memory-mapped when stored uncompressed."""

    def __init__(self, raw: np.ndarray, tag: str, byteorder: str):
        self.raw = raw  # 1-D uint8
        self.tag = tag
        self.byteorder = byteorder

    def view(self, tag: str) -> np.ndarray:
        dt = np.dtype(("<" if self.byteorder == "little" else ">") + _TAG_NP[tag]) if _TAG_NP[tag] not in (
            "u1", "i1", "b1") else np.dtype(_TAG_NP[tag])
        n = self.raw.size // dt.itemsize
        return self.raw[: n * dt.itemsize].view(dt)


class TensorRef:
    """A lazily materialised tensor: storage + offset + size + stride (in elements)."""

    def __init__(self, storage: Storage, offset: int, size, stride, tag: str):
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (offset, *size, *stride)):
            raise CheckpointError("tensor offset, size or stride is not an integer")
        if len(size) != len(stride) or offset < 0 or any(x < 0 for x in (*size, *stride)):
            raise CheckpointError("bad tensor offset, size or stride")
        self.storage = storage
        self.offset = int(offset)
        self.shape = tuple(int(s) for s in size)
        self.stride = tuple(int(s) for s in stride)
        self.tag = tag

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def array(self) -> np.ndarray:
        """Contiguous native-endian copy (BF16 as uint16 bit patterns)."""
        base = self.storage.view(self.tag)
        n = math.prod(self.shape)
        if n == 0:
            return np.zeros(self.shape, dtype=base.dtype.newbyteorder("="))
        extent = self.offset + sum((s - 1) * st for s, st in zip(self.shape, self.stride)) + 1
        if extent > base.size:
            raise CheckpointError("tensor view outside its storage")
        if n > max(base.size, MAX_EXPANDED_NUMEL):  # bounds the copy below (expanded views only up to a limit)
            raise CheckpointError("tensor has far more elements than its storage")
        isz = base.dtype.itemsize
        v = np.lib.stride_tricks.as_strided(base[self.offset:], shape=self.shape,
                                            strides=[st * isz for st in self.stride], writeable=False)
        return np.ascontiguousarray(v).astype(base.dtype.newbyteorder("="), copy=False)

    def float32(self) -> np.ndarray:
        """Exactly `tensor.float()`: BF16/F16 widen exactly, F64/ints round to nearest."""
        a = self.array()
        if self.tag == "BF16":
            return (a.astype(np.uint32) << np.uint32(16)).view(np.float32)
        return a.astype(np.float32)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None,
                       metadata=None):
    if not isinstance(storage, tuple) or storage[0] != "__storage__":
        raise CheckpointError("tensor without a storage")
    return TensorRef(storage[1], storage_offset, size, stride, storage[2])


def _rebuild_tensor_v3(storage, storage_offset, size, stride, requires_grad, backward_hooks, dtype,
                       metadata=None):
    if not isinstance(dtype, _DType):
        raise CheckpointError("bad dtype in _rebuild_tensor_v3")
    if not isinstance(storage, tuple) or storage[0] != "__storage__":
        raise CheckpointError("tensor without a storage")
    return TensorRef(storage[1], storage_offset, size, stride, dtype.tag)


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None, state=None):
    return data


def _rebuild_from_type_v2(func, new_type, args, state):
    return func(*args)


def _codecs_encode(obj, encoding="latin1"):
    # protocol-2 pickles store bytes as _codecs.encode(<str>, "latin1"); nothing else is needed
    if not isinstance(obj, str) or encoding not in ("latin1", "latin-1", "iso-8859-1"):
        raise CheckpointError("unsupported _codecs.encode call in the checkpoint")
    return codecs.encode(obj, encoding)


_ALLOWED: dict[tuple[str, str], object] = {
    ("collections", "OrderedDict"): OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_tensor_v3"): _rebuild_tensor_v3,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("torch._utils", "_rebuild_parameter_with_state"): _rebuild_parameter,
    ("torch._tensor", "_rebuild_from_type_v2"): _rebuild_from_type_v2,
    ("torch", "Tensor"): _Marker("torch.Tensor"),
    ("torch.nn.parameter", "Parameter"): _Marker("torch.nn.Parameter"),
    ("torch", "Size"): _Marker("torch.Size"),
    ("torch", "device"): _Marker("torch.device"),
    ("_codecs", "encode"): _codecs_encode,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
}
for _n, (_c, _t) in _STORAGE_TYPES.items():
    _ALLOWED[("torch", _n)] = _StorageType(_t)
_ALLOWED[("torch.storage", "UntypedStorage")] = _StorageType("U8")
for _n, _t in _DTYPES.items():
    _ALLOWED[("torch", _n)] = _DType(_t)


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, load_storage: Callable[[str, str], Storage]):
        super().__init__(f)
        self._load_storage = load_storage

    def find_class(self, module, name):
        # torch.save writes protocol 2: Python-2 names (`__builtin__`) may appear.
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        elif module in _compat_pickle.IMPORT_MAPPING:
            module = _compat_pickle.IMPORT_MAPPING[module]
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise CheckpointError(
                f"checkpoint references {module}.{name}, which the torch-free reader does not allow; "
                "re-run with --use-torch (needs PyTorch)") from None

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage"):
            raise CheckpointError(f"unsupported persistent id {pid!r:.80}")
        _, stype, key, _location, _numel = pid[:5]
        if not isinstance(stype, _StorageType):
            raise CheckpointError("unsupported storage type")
        return ("__storage__", self._load_storage(str(key), stype.tag), stype.tag)


def _member_data_offset(path: Path, info: zipfile.ZipInfo) -> int:
    """Byte offset of a member's data in the archive (local header + name + extra)."""
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        hdr = f.read(30)
    if hdr[:4] != b"PK\x03\x04":
        raise CheckpointError(f"bad local header for {info.filename}")
    n, m = struct.unpack("<HH", hdr[26:30])
    return info.header_offset + 30 + n + m


def load_torch_file(path: str | os.PathLike):
    """Reads any torch zip file whose pickle uses only tensors, containers, numbers and
    strings, without torch. Tensors come back as TensorRef (memory-mapped, lazy)."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"{path}: no such file")
    if not zipfile.is_zipfile(path):
        raise CheckpointError(f"{path} is not a torch zip checkpoint (legacy format?); use --use-torch")
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkls = [n for n in names if n.endswith("/data.pkl") or n == "data.pkl"]
    if len(pkls) != 1:
        raise CheckpointError(f"expected one data.pkl in {path}, found {len(pkls)}")
    prefix = pkls[0][: -len("data.pkl")]
    byteorder = "little"
    if prefix + "byteorder" in names:
        byteorder = zf.read(prefix + "byteorder").decode().strip()
        if byteorder not in ("little", "big"):
            raise CheckpointError(f"bad byteorder {byteorder!r}")
    infos = {i.filename: i for i in zf.infolist()}
    cache: dict[str, Storage] = {}
    whole: list[np.ndarray] = []  # one read-only mapping of the archive, sliced per storage

    def load_storage(key: str, tag: str) -> Storage:
        if key in cache:
            return cache[key]
        name = f"{prefix}data/{key}"
        info = infos.get(name)
        if info is None:
            raise CheckpointError(f"missing storage record {name}")
        if info.compress_type == zipfile.ZIP_STORED and info.file_size > 0:
            if not whole:
                whole.append(np.memmap(path, dtype=np.uint8, mode="r"))
            start = _member_data_offset(path, info)
            if start + info.file_size > whole[0].size:
                raise CheckpointError(f"storage record {name} past the end of the file")
            raw = whole[0][start:start + info.file_size]
        else:
            if info.file_size > MAX_COMPRESSED_STORAGE:  # torch.save never compresses storages
                raise CheckpointError(f"compressed storage record {name} too large")
            raw = np.frombuffer(zf.read(name), dtype=np.uint8)
        cache[key] = Storage(raw, tag, byteorder)
        return cache[key]

    try:
        with zf.open(pkls[0]) as f:
            return _Unpickler(f, load_storage).load()
    except CheckpointError:
        raise
    except Exception as e:  # noqa: BLE001  a malformed or hostile pickle: report, never a traceback
        raise CheckpointError(f"malformed checkpoint ({type(e).__name__}: {str(e)[:200]})") from None


def load_checkpoint(path: str | os.PathLike) -> "OrderedDict[str, TensorRef]":
    """Reads a torch zip checkpoint without torch. Returns name -> TensorRef in file order.

    A top-level dict with a `model` or `state_dict` entry is unwrapped, as the
    official loader does."""
    obj = load_torch_file(path)
    if isinstance(obj, dict) and ("model" in obj or "state_dict" in obj):
        obj = obj.get("model", obj.get("state_dict"))
    if not isinstance(obj, dict):
        raise CheckpointError("checkpoint is not a state dict")
    out: OrderedDict[str, TensorRef] = OrderedDict()
    for k, v in obj.items():
        if not isinstance(v, TensorRef):
            raise CheckpointError(f"state dict entry {k!r} is not a tensor")
        out[str(k)] = v
    return out


# --------------------------------------------------------------------------
# Numeric casts (bit-exact with torch's CPU casts for finite values)
# --------------------------------------------------------------------------

def f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns, round to nearest even (c10::BFloat16); NaN -> 0x7FC0."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    out = ((u + (((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF))) >> np.uint32(16)).astype(np.uint16)
    nan = np.isnan(a)
    if nan.any():
        out[nan] = 0x7FC0
    return out


def f32_to_e4m3fn_bits(a: np.ndarray) -> np.ndarray:
    """float32 -> FP8 E4M3 (fn: no inf, 0x7F/0xFF = NaN) bit patterns, as c10::Float8_e4m3fn.

    |x| >= 480 (and inf/NaN) -> NaN; subnormals (|x| < 2^-6) are rounded by adding
    2^14 in float32; normals round to nearest even on the 3 mantissa bits."""
    f = np.ascontiguousarray(a, dtype=np.float32)
    bits = f.view(np.uint32)
    sign = bits & np.uint32(0x80000000)
    absb = bits ^ sign
    out = np.zeros(f.shape, dtype=np.uint8)
    big = absb >= np.uint32(1087 << 20)
    small = absb < np.uint32(121 << 23)
    norm = ~(big | small)
    if small.any():
        s = (absb[small].view(np.float32) + np.float32(2.0 ** 14)).view(np.uint32) - np.uint32(141 << 23)
        out[small] = s.astype(np.uint8)
    if norm.any():
        nb = absb[norm]
        odd = (nb >> np.uint32(20)) & np.uint32(1)
        nb = nb - np.uint32(120 << 23) + np.uint32(0x7FFFF) + odd
        out[norm] = (nb >> np.uint32(20)).astype(np.uint8)
    out[big] = 0x7F
    out |= (sign >> np.uint32(24)).astype(np.uint8)
    return out


class NumpyCodec:
    name = "numpy"

    @staticmethod
    def bf16(a: np.ndarray) -> bytes:
        return f32_to_bf16_bits(a).astype("<u2", copy=False).tobytes()

    @staticmethod
    def fp8(a: np.ndarray) -> bytes:
        return f32_to_e4m3fn_bits(a).tobytes()


class TorchCodec:
    name = "torch"

    def __init__(self):
        import torch  # noqa: PLC0415

        self.torch = torch

    def bf16(self, a: np.ndarray) -> bytes:
        t = self.torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return t.to(self.torch.bfloat16).view(self.torch.int16).numpy().tobytes()

    def fp8(self, a: np.ndarray) -> bytes:
        t = self.torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return t.to(self.torch.float8_e4m3fn).view(self.torch.uint8).numpy().tobytes()


def f32_bytes(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a, dtype=np.float32).astype("<f4", copy=False).tobytes()


# --------------------------------------------------------------------------
# Torch loader (--use-torch)
# --------------------------------------------------------------------------

class _TorchRef:
    def __init__(self, t):
        self.t = t
        self.shape = tuple(t.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def float32(self) -> np.ndarray:
        return self.t.detach().float().contiguous().numpy()


def load_checkpoint_torch(path: str | os.PathLike) -> "OrderedDict[str, _TorchRef]":
    import torch  # noqa: PLC0415

    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # torch < 2.1 has no mmap
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and ("model" in state or "state_dict" in state):
        state = state.get("model", state.get("state_dict"))
    return OrderedDict((k, _TorchRef(v)) for k, v in state.items())


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

class Eff:
    """One effective tensor: final shape and a producer of its float32 values."""

    def __init__(self, shape: tuple, produce: Callable[[], np.ndarray]):
        self.shape = tuple(int(s) for s in shape)
        self._produce = produce

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1

    def f32(self) -> np.ndarray:
        a = self._produce()
        return np.ascontiguousarray(a, dtype=np.float32).reshape(self.shape)


def effective(state) -> "OrderedDict[str, Eff]":
    """The tensors to write, in checkpoint order (masks folded, dropped entries removed)."""
    out: OrderedDict[str, Eff] = OrderedDict()
    for name, t in state.items():
        if name.endswith(".bias_mask") or name.endswith("mask_token"):
            continue
        if name.startswith("text_alignment_head."):
            continue
        if not (name.startswith(DEVICE_PREFIX) or name.startswith(HOST_PREFIXES)):
            raise SystemExit(f"unexpected tensor {name}")
        shape = tuple(t.shape)
        mask = state.get(name + "_mask") if name.endswith(".qkv.bias") else None
        if mask is not None:
            v = t.float32() * mask.float32()
            if name.startswith(DEVICE_PREFIX) and not np.any(v != 0):
                continue
            shape = v.shape
            out[name] = Eff(shape, lambda v=v: v)
            continue
        if name == PATCH_WEIGHT:
            shape = (shape[0], int(np.prod(shape[1:], dtype=np.int64)))
        out[name] = Eff(shape, t.float32)
    return out


def device_dtype(name: str, t: Eff, all_: dict) -> str:
    if not name.startswith(DEVICE_PREFIX):
        return "F32"
    if name.endswith(".weight") and t.ndim == 2:
        return "BF16"
    if name.endswith(".bias"):
        w = all_.get(name[: -len(".bias")] + ".weight")
        if w is not None and w.ndim == 2:
            return "BF16"
    return "F32"


_SIZES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}

# An entry: (name, dtype, shape, producer of bytes). DEVICE_END marks the end of the device region.
Entry = tuple


def write_safetensors(path: Path, entries: Iterable[Entry], meta: dict, quiet: bool = False) -> int:
    """Streams `entries` into a 256-byte aligned safetensors file (header first, data after).

    Same layout as the original converter: a `__pad.<n>` U8 tensor before any tensor
    (or the device end) that is not 256-aligned; header JSON padded with spaces so
    the data section starts 8-aligned; written to `<path>.tmp` and renamed."""
    entries = list(entries)
    header: dict[str, object] = {}
    plan: list[tuple[int, object]] = []  # (gap bytes, producer or None)
    off = 0
    pad = 0
    count = 0
    for name, dtype, shape, produce in entries:
        gap = (-off) % ALIGN
        if gap:
            header[f"__pad.{pad}"] = {"dtype": "U8", "shape": [gap], "data_offsets": [off, off + gap]}
            plan.append((gap, None))
            off += gap
            pad += 1
        if name == DEVICE_END:
            meta["device_bytes"] = str(off)
            continue
        n = _SIZES[dtype] * (int(np.prod(shape, dtype=np.int64)) if shape else 1)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        plan.append((n, produce))
        off += n
        count += 1
    header["__metadata__"] = meta
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-(8 + len(h))) % 8)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for n, produce in plan:
            if produce is None:
                f.write(b"\0" * n)
                continue
            data = produce()
            if len(data) != n:
                raise RuntimeError(f"internal: {len(data)} bytes produced, {n} planned")
            f.write(data)
    tmp.rename(path)
    if not quiet:
        print(f"wrote {path}: {count} tensors, {off / 1e9:.3f} GB")
    return off


def convert(input_path: str | os.PathLike, output: str | os.PathLike, fp8: str | os.PathLike | None = None,
            use_torch: bool = False, quiet: bool = False) -> None:
    input_path = Path(input_path)
    if use_torch:
        state = load_checkpoint_torch(input_path)
        codec = TorchCodec()
    else:
        state = load_checkpoint(input_path)
        codec = NumpyCodec()
    eff = effective(state)
    dev = [(n, t) for n, t in eff.items() if n.startswith(DEVICE_PREFIX)]
    host = [(n, t) for n, t in eff.items() if not n.startswith(DEVICE_PREFIX)]
    entries: list[Entry] = []
    for n, t in dev:
        d = device_dtype(n, t, eff)
        enc = codec.bf16 if d == "BF16" else f32_bytes
        entries.append((n, d, list(t.shape), lambda t=t, enc=enc: enc(t.f32())))
    entries.append((DEVICE_END, "", [], None))
    for n, t in host:
        entries.append((n, "F32", list(t.shape), lambda t=t: f32_bytes(t.f32())))
    write_safetensors(Path(output), entries, {"format": "rawatta-vggt-1", "source": input_path.name}, quiet)

    if fp8:
        cache: dict[str, tuple] = {}

        def quant(n: str, t: Eff):
            if n not in cache:
                w = t.f32()
                scale = np.maximum(np.abs(w).max(axis=1) / np.float32(FP8_MAX), np.float32(1e-12))
                scale = scale.astype(np.float32, copy=False)
                cache.clear()
                cache[n] = (codec.fp8(w / scale[:, None]), f32_bytes(scale))
            return cache[n]

        f8: list[Entry] = []
        for n, t in dev:
            if not (n.endswith(".weight") and t.ndim == 2) or "patch_embed.patch_embed" in n:
                continue
            f8.append((n, "F8_E4M3", list(t.shape), lambda n=n, t=t: quant(n, t)[0]))
            f8.append((n + "_scale", "F32", [t.shape[0]], lambda n=n, t=t: quant(n, t)[1]))
        f8.append((DEVICE_END, "", [], None))
        write_safetensors(Path(fp8), f8, {"format": "rawatta-vggt-1-fp8", "source": input_path.name}, quiet)


# --------------------------------------------------------------------------
# Reading back (for checks)
# --------------------------------------------------------------------------

def read_header(path: str | os.PathLike) -> tuple[int, dict]:
    """(data section start, header dict) of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100 << 20:
            raise ValueError("header too large")
        return 8 + n, json.loads(f.read(n))


def check_layout(path: str | os.PathLike) -> dict:
    """Validates what crates/vggt-omega/src/weights.rs requires; returns a summary."""
    start, h = read_header(path)
    size = os.path.getsize(path)
    meta = h.get("__metadata__", {})
    dev = int(meta.get("device_bytes", 0))
    tensors = {k: v for k, v in h.items() if k != "__metadata__"}
    end = 0
    for k, v in tensors.items():
        a, b = v["data_offsets"]
        numel = int(np.prod(v["shape"], dtype=np.int64)) if v["shape"] else 1
        if b - a != numel * _SIZES[v["dtype"]]:
            raise ValueError(f"{k}: data_offsets do not match dtype/shape")
        if not k.startswith("__pad") and b <= dev and a % ALIGN:
            raise ValueError(f"{k}: device tensor not {ALIGN}-aligned")
        end = max(end, b)
    if start + end > size or start + dev > size:
        raise ValueError("data past the end of the file")
    return {"format": meta.get("format"), "device_bytes": dev,
            "tensors": sum(1 for k in tensors if not k.startswith("__pad")), "data_bytes": end}


def default_output() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(cache) / "rawatta" / "vggt_omega_1b_512.safetensors"


def fp8_companion(path: Path) -> Path:
    return path.with_name(f"{path.stem}_fp8.safetensors")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="official checkpoint vggt_omega_1b_512.pt")
    p.add_argument("--output", default=None,
                   help=f"safetensors to write (default: {default_output()})")
    p.add_argument("--fp8", nargs="?", const="auto", default=None,
                   help="also write the FP8 E4M3 Linear weights (default path: <output stem>_fp8.safetensors)")
    p.add_argument("--use-torch", action="store_true",
                   help="load with torch.load(weights_only=True) and cast with torch (needs PyTorch)")
    p.add_argument("--check", action="store_true", help="validate the written files' layout afterwards")
    a = p.parse_args(argv)
    out = Path(a.output) if a.output else default_output()
    out.parent.mkdir(parents=True, exist_ok=True)
    fp8 = None
    if a.fp8:
        fp8 = fp8_companion(out) if a.fp8 == "auto" else Path(a.fp8)
    try:
        convert(a.input, out, fp8, use_torch=a.use_torch)
    except (CheckpointError, zipfile.BadZipFile) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"error: {e} (--use-torch needs PyTorch)", file=sys.stderr)
        return 1
    if a.check:
        for f in (out, fp8):
            if f:
                print(f"check {f}: {check_layout(f)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
