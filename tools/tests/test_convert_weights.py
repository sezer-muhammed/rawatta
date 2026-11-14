# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/convert_weights.py.

Torch-free tests always run: known FP8/BF16 bit patterns, a torch-format zip
checkpoint written by hand (pure-Python pickle opcodes) read back without torch,
refusal of foreign pickle globals, and the output layout.

With PyTorch installed, the casts are compared with torch on random and special
values, the reader with torch.load, and the converter's output, run in a
subprocess in which `import torch` fails, is compared byte for byte with the
original converter (tests/reference_convert_vggt.py) on synthetic
checkpoints that mimic the real key names and dtypes (tests/synthetic_checkpoint.py).

    python3 -m pytest tools/tests -q
"""
from __future__ import annotations

import io
import pickle
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(HERE))

import convert_weights as cw  # noqa: E402

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="PyTorch not installed")


# --------------------------------------------------------------------------
# casts
# --------------------------------------------------------------------------

def test_fp8_known_values():
    vals = np.array([0.0, -0.0, 1.0, -1.0, 448.0, -448.0, 2.0 ** -6, 2.0 ** -9, 2.0 ** -10, 3 * 2.0 ** -10,
                     1.0625, 1.1875, 464.0, 465.0, 480.0, 1e9, np.inf, np.nan, 0.5, 2.0 ** -11], np.float32)
    want = [0x00, 0x80, 0x38, 0xB8, 0x7E, 0xFE, 0x08, 0x01, 0x00, 0x02,
            0x38, 0x3A, 0x7E, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x30, 0x00]
    assert cw.f32_to_e4m3fn_bits(vals).tolist() == want


def test_bf16_known_values():
    bits = np.array([0x3F800000, 0x3F808000, 0x3F818000, 0x3F80FFFF, 0x7F7FFFFF, 0x00000001, 0x80000000,
                     0x7FC00001], np.uint32)
    want = [0x3F80, 0x3F80, 0x3F82, 0x3F81, 0x7F80, 0x0000, 0x8000, 0x7FC0]
    assert cw.f32_to_bf16_bits(bits.view(np.float32)).tolist() == want


def _special_f32(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = [rng.standard_normal(n).astype(np.float32) * s for s in (1e-3, 1.0, 100.0)]
    raw = rng.integers(0, 2 ** 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
    raw = raw[np.isfinite(raw)]
    grid = np.arange(-600, 600, 0.0625, dtype=np.float32)  # every fp8 tie and the 448..480 edge
    tiny = (np.arange(0, 2 ** 12, dtype=np.float32) * np.float32(2.0 ** -18))  # fp8 subnormal range
    sub = np.arange(1, 2 ** 16, 7, dtype=np.uint32).view(np.float32)  # float32 subnormals
    ties = (np.arange(0, 2 ** 16, dtype=np.uint32) << 16 | 0x8000).view(np.float32)
    ties = ties[np.isfinite(ties)]
    return np.concatenate(a + [raw, grid, -grid, tiny, -tiny, sub, ties])


@needs_torch
def test_casts_match_torch():
    import torch

    x = _special_f32(200_000)
    t = torch.from_numpy(x)
    assert np.array_equal(cw.f32_to_bf16_bits(x), t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
    f8 = x[np.abs(x) < 1e30]
    assert np.array_equal(cw.f32_to_e4m3fn_bits(f8),
                          torch.from_numpy(f8).to(torch.float8_e4m3fn).view(torch.uint8).numpy())


# --------------------------------------------------------------------------
# a torch-format checkpoint written without torch
# --------------------------------------------------------------------------

class G:  # a pickle GLOBAL
    def __init__(self, module, name):
        self.module, self.name = module, name


class R:  # REDUCE(callable, args)
    def __init__(self, func, args):
        self.func, self.args = func, args


class PID:  # BINPERSID
    def __init__(self, obj):
        self.obj = obj


class OD:  # collections.OrderedDict with items
    def __init__(self, items):
        self.items = items


def _enc(o, b: bytearray) -> None:
    if o is None:
        b += b"N"
    elif o is True:
        b += b"\x88"
    elif o is False:
        b += b"\x89"
    elif isinstance(o, int):
        if -(2 ** 31) <= o < 2 ** 31:
            b += b"J" + struct.pack("<i", o)
        else:
            raw = o.to_bytes((o.bit_length() + 8) // 8, "little", signed=True)
            b += b"\x8a" + bytes([len(raw)]) + raw
    elif isinstance(o, str):
        e = o.encode()
        b += b"X" + struct.pack("<I", len(e)) + e
    elif isinstance(o, tuple):
        if not o:
            b += b")"
        else:
            b += b"("
            for x in o:
                _enc(x, b)
            b += b"t"
    elif isinstance(o, G):
        b += b"c" + o.module.encode() + b"\n" + o.name.encode() + b"\n"
    elif isinstance(o, R):
        _enc(o.func, b)
        _enc(o.args, b)
        b += b"R"
    elif isinstance(o, PID):
        _enc(o.obj, b)
        b += b"Q"
    elif isinstance(o, OD):
        _enc(G("collections", "OrderedDict"), b)
        b += b")R"
        if o.items:
            b += b"("
            for k, v in o.items:
                _enc(k, b)
                _enc(v, b)
            b += b"u"
    else:
        raise TypeError(type(o))


def _tensor(key, stype, numel, offset, size, stride):
    pid = PID(("storage", G("torch", stype), key, "cpu", numel))
    return R(G("torch._utils", "_rebuild_tensor_v2"), (pid, offset, tuple(size), tuple(stride), False, OD([])))


def write_handmade(path: Path, entries, storages: dict[str, bytes], wrap: str | None = None) -> None:
    body = OD(entries)
    b = bytearray(b"\x80\x02")
    if wrap:
        b += b"}("
        _enc(wrap, b)
        _enc(body, b)
        b += b"u"
    else:
        _enc(body, b)
    b += b"."
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", bytes(b))
        z.writestr("archive/byteorder", "little")
        for k, v in storages.items():
            z.writestr(f"archive/data/{k}", v)
        z.writestr("archive/version", "3\n")


def _bf16(a):
    return cw.f32_to_bf16_bits(np.asarray(a, np.float32))


def _handmade_state(tmp_path: Path, wrap=None):
    rng = np.random.default_rng(1)
    D = 8
    qkv = rng.standard_normal((3 * D, D)).astype(np.float32)
    big = np.zeros((D + 1, 3 * D + 2), np.float32)  # qkv stored transposed inside a larger buffer
    big[1:, 2:] = qkv.T
    proj = rng.standard_normal((D, D)).astype(np.float32)
    patch = rng.standard_normal((D, 3, 2, 2)).astype(np.float32)
    vec = rng.standard_normal(4 * D).astype(np.float32)
    bias = rng.standard_normal(3 * D).astype(np.float32)
    mask = np.ones(3 * D, np.float32)
    mask[D:2 * D] = 0
    zero_mask = np.zeros(3 * D, np.float32)
    head = rng.standard_normal((5, 3 * D)).astype(np.float32)
    storages = {
        "0": big.tobytes(), "1": _bf16(proj).tobytes(), "2": patch.tobytes(), "3": vec.tobytes(),
        "4": bias.tobytes(), "5": mask.tobytes(), "6": zero_mask.tobytes(), "7": head.tobytes(),
    }
    e = [
        ("aggregator.patch_embed.patch_embed.proj.weight", _tensor("2", "FloatStorage", patch.size, 0, patch.shape,
                                                                   (12, 4, 2, 1))),
        ("aggregator.patch_embed.patch_embed.proj.bias", _tensor("3", "FloatStorage", vec.size, 0, (D,), (1,))),
        ("aggregator.frame_blocks.0.attn.qkv.weight", _tensor("0", "FloatStorage", big.size, 3 * D + 2 + 2,
                                                              (3 * D, D), (1, 3 * D + 2))),
        ("aggregator.frame_blocks.0.attn.qkv.bias", _tensor("4", "FloatStorage", bias.size, 0, (3 * D,), (1,))),
        ("aggregator.frame_blocks.0.attn.qkv.bias_mask", _tensor("6", "FloatStorage", 3 * D, 0, (3 * D,), (1,))),
        ("aggregator.frame_blocks.0.attn.proj.weight", _tensor("1", "BFloat16Storage", proj.size, 0, (D, D), (D, 1))),
        ("aggregator.frame_blocks.0.attn.proj.bias", _tensor("3", "FloatStorage", vec.size, D, (D,), (1,))),
        ("aggregator.frame_blocks.0.ls1.gamma", _tensor("3", "FloatStorage", vec.size, 2 * D, (D,), (1,))),
        ("aggregator.patch_embed.mask_token", _tensor("3", "FloatStorage", vec.size, 0, (1, D), (D, 1))),
        ("camera_head.trunk.0.attn.qkv.weight", _tensor("7", "FloatStorage", head.size, 0, head.shape, (3 * D, 1))),
        ("camera_head.trunk.0.attn.qkv.bias", _tensor("4", "FloatStorage", bias.size, 0, (3 * D,), (1,))),
        ("camera_head.trunk.0.attn.qkv.bias_mask", _tensor("5", "FloatStorage", 3 * D, 0, (3 * D,), (1,))),
        ("text_alignment_head.w", _tensor("3", "FloatStorage", vec.size, 0, (D,), (1,))),
    ]
    p = tmp_path / "hand.pt"
    write_handmade(p, e, storages, wrap)
    ref = {"qkv": qkv, "proj_bits": _bf16(proj), "patch": patch, "vec": vec, "bias": bias, "mask": mask,
           "head": head, "D": D}
    return p, ref


@pytest.mark.parametrize("wrap", [None, "model", "state_dict"])
def test_reader_handmade(tmp_path, wrap):
    p, ref = _handmade_state(tmp_path, wrap)
    sd = cw.load_checkpoint(p)
    D = ref["D"]
    assert list(sd)[0] == "aggregator.patch_embed.patch_embed.proj.weight"
    assert np.array_equal(sd["aggregator.frame_blocks.0.attn.qkv.weight"].float32(), ref["qkv"])
    assert sd["aggregator.frame_blocks.0.attn.proj.weight"].tag == "BF16"
    assert np.array_equal(sd["aggregator.frame_blocks.0.attn.proj.weight"].array(), ref["proj_bits"])
    assert np.array_equal(sd["aggregator.frame_blocks.0.ls1.gamma"].float32(), ref["vec"][2 * D:3 * D])
    assert np.array_equal(sd["aggregator.patch_embed.patch_embed.proj.weight"].float32(), ref["patch"])


@needs_torch
def test_handmade_matches_torch_load(tmp_path):
    import torch

    p, _ = _handmade_state(tmp_path)
    ours = cw.load_checkpoint(p)
    theirs = torch.load(p, map_location="cpu", weights_only=True)
    assert list(ours) == list(theirs)
    for k, v in theirs.items():
        assert np.array_equal(ours[k].float32(), v.float().numpy()), k


def test_convert_handmade_layout(tmp_path):
    p, ref = _handmade_state(tmp_path)
    out, f8 = tmp_path / "w.safetensors", tmp_path / "w_fp8.safetensors"
    cw.convert(p, out, f8, quiet=True)
    s = cw.check_layout(out)
    assert s["format"] == "rawatta-vggt-1"
    start, h = cw.read_header(out)
    names = [k for k in h if not k.startswith("__")]
    # dropped: masks, mask_token, text head, the all-zero-masked aggregator qkv bias
    assert names == ["aggregator.patch_embed.patch_embed.proj.weight", "aggregator.patch_embed.patch_embed.proj.bias",
                     "aggregator.frame_blocks.0.attn.qkv.weight", "aggregator.frame_blocks.0.attn.proj.weight",
                     "aggregator.frame_blocks.0.attn.proj.bias", "aggregator.frame_blocks.0.ls1.gamma",
                     "camera_head.trunk.0.attn.qkv.weight", "camera_head.trunk.0.attn.qkv.bias"]
    dt = {k: h[k]["dtype"] for k in names}
    assert dt["aggregator.patch_embed.patch_embed.proj.weight"] == "BF16"
    assert h["aggregator.patch_embed.patch_embed.proj.weight"]["shape"] == [ref["D"], 12]
    assert dt["aggregator.patch_embed.patch_embed.proj.bias"] == "BF16"
    assert dt["aggregator.frame_blocks.0.ls1.gamma"] == "F32"
    assert dt["camera_head.trunk.0.attn.qkv.weight"] == "F32"
    dev = int(h["__metadata__"]["device_bytes"])
    assert all(h[k]["data_offsets"][1] <= dev for k in names if k.startswith("aggregator."))
    assert all(h[k]["data_offsets"][0] >= dev for k in names if k.startswith("camera_head."))
    raw = out.read_bytes()
    a, b = h["camera_head.trunk.0.attn.qkv.bias"]["data_offsets"]
    got = np.frombuffer(raw[start + a:start + b], "<f4")
    assert np.array_equal(got, ref["bias"] * ref["mask"])  # [b_q, 0, b_v]
    s8 = cw.check_layout(f8)
    assert s8["format"] == "rawatta-vggt-1-fp8" and s8["tensors"] == 4  # qkv + proj weights and their scales
    _, h8 = cw.read_header(f8)
    assert "aggregator.patch_embed.patch_embed.proj.weight" not in h8
    assert h8["aggregator.frame_blocks.0.attn.qkv.weight_scale"]["shape"] == [3 * ref["D"]]


def test_refuses_foreign_globals(tmp_path):
    b = bytearray(b"\x80\x02")
    _enc(R(G("os", "system"), ("echo pwned",)), b)
    b += b"."
    p = tmp_path / "evil.pt"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("archive/data.pkl", bytes(b))
    with pytest.raises(cw.CheckpointError, match="os.system"):
        cw.load_checkpoint(p)


def _pickle_file(tmp_path: Path, obj, storages: dict[str, bytes] | None = None) -> Path:
    b = bytearray(b"\x80\x02")
    _enc(obj, b)
    b += b"."
    p = tmp_path / "bad.pt"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("archive/data.pkl", bytes(b))
        for k, v in (storages or {}).items():
            z.writestr(f"archive/data/{k}", v)
    return p


@pytest.mark.parametrize("obj,match", [
    (R(G("torch", "FloatStorage"), ()), "malformed checkpoint"),  # calls an allowed non-callable
    (R(G("builtins", "bytearray"), (1 << 40,)), "builtins.bytearray"),  # no longer allowed
    (R(G("_codecs", "encode"), ("x", "zlib")), "_codecs.encode"),
    (R(G("torch", "Size"), (5,)), "malformed checkpoint"),  # torch.Size(5): not iterable
    (OD([("w", _tensor("0", "FloatStorage", 1, 0, ("a",), (1,)))]), "not an integer"),
])
def test_malformed_pickle_is_checkpoint_error(tmp_path, obj, match):
    p = _pickle_file(tmp_path, obj, {"0": b"\0" * 4})
    with pytest.raises(cw.CheckpointError, match=match):
        cw.load_torch_file(p)


def test_expanded_view_is_bounded(tmp_path):
    # one float32 element viewed as 2^40 elements (stride 0): refused before any allocation
    p = _pickle_file(tmp_path, OD([("w", _tensor("0", "FloatStorage", 1, 0, (1 << 20, 1 << 20), (0, 0)))]),
                     {"0": b"\0" * 4})
    sd = cw.load_checkpoint(p)
    with pytest.raises(cw.CheckpointError, match="more elements"):
        sd["w"].array()


def test_cli_reports_malformed_file_without_traceback(tmp_path):
    p = _pickle_file(tmp_path, R(G("torch", "FloatStorage"), ()))
    r = subprocess.run([sys.executable, str(TOOLS / "convert_weights.py"), "--input", str(p), "--output",
                        str(tmp_path / "o.safetensors")], capture_output=True, text=True)
    assert r.returncode == 1 and r.stderr.startswith("error: malformed checkpoint") and "Traceback" not in r.stderr
    r = subprocess.run([sys.executable, str(TOOLS / "convert_weights.py"), "--input", str(tmp_path / "missing.pt"),
                        "--output", str(tmp_path / "o.safetensors")], capture_output=True, text=True)
    assert r.returncode == 1 and "no such file" in r.stderr and "Traceback" not in r.stderr


def test_refuses_non_zip(tmp_path):
    p = tmp_path / "legacy.pt"
    p.write_bytes(pickle.dumps({"a": 1}))
    with pytest.raises(cw.CheckpointError):
        cw.load_checkpoint(p)


def test_cli_help():
    r = subprocess.run([sys.executable, str(TOOLS / "convert_weights.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--use-torch" in r.stdout


# --------------------------------------------------------------------------
# byte identity with the original converter on synthetic checkpoints
# --------------------------------------------------------------------------

NO_TORCH = ("import sys, runpy; sys.modules['torch'] = None; "
            "sys.argv = ['convert_weights.py'] + sys.argv[1:]; "
            f"runpy.run_path({str(TOOLS / 'convert_weights.py')!r}, run_name='__main__')")


def _run(args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


@needs_torch
@pytest.mark.parametrize("dtype,wrap,views,agg_mask,extra", [
    ("f32", "none", False, "zero", False),
    ("bf16", "model", True, "zero", True),
    ("mixed", "state_dict", True, "pattern", False),
])
def test_byte_identical_to_original(tmp_path, dtype, wrap, views, agg_mask, extra):
    import synthetic_checkpoint as sc

    ck = tmp_path / "synth.pt"
    sc.save(ck, scale=32, dtype=dtype, wrap=wrap, views=views, agg_mask=agg_mask, extra=extra)
    ref, ref8 = tmp_path / "ref.safetensors", tmp_path / "ref_fp8.safetensors"
    _run([str(HERE / "reference_convert_vggt.py"), "--input", str(ck), "--output", str(ref), "--fp8", str(ref8)])
    ours, ours8 = tmp_path / "x" / "synth.safetensors", tmp_path / "x" / "synth_fp8.safetensors"
    ours.parent.mkdir()
    # torch blocked in the converter's process: `import torch` raises ImportError there
    r = _run(["-c", NO_TORCH, "--input", str(ck), "--output", str(ours), "--fp8", "--check"])
    assert "check" in r.stdout
    assert ours.read_bytes() == ref.read_bytes()
    assert ours8.read_bytes() == ref8.read_bytes()
    t, t8 = tmp_path / "t.safetensors", tmp_path / "t_fp8.safetensors"
    _run([str(TOOLS / "convert_weights.py"), "--use-torch", "--input", str(ck), "--output", str(t), "--fp8", str(t8)])
    assert t.read_bytes() == ref.read_bytes()
    assert t8.read_bytes() == ref8.read_bytes()


@needs_torch
def test_reader_matches_torch_load_synthetic(tmp_path):
    import torch

    import synthetic_checkpoint as sc

    ck = tmp_path / "synth.pt"
    sc.save(ck, scale=64, dtype="mixed", views=True, extra=True)
    ours = cw.load_checkpoint(ck)
    theirs = torch.load(ck, map_location="cpu", weights_only=True)
    assert list(ours) == list(theirs)
    for k, v in theirs.items():
        assert ours[k].shape == tuple(v.shape), k
        assert np.array_equal(ours[k].float32(), v.float().numpy(), equal_nan=True), k


def test_synthetic_keys_count():
    import synthetic_checkpoint as sc

    ks = sc.key_shapes(1)
    assert len(ks) == 1411 and len({k for k, _ in ks}) == 1411
    assert dict(ks)["aggregator.patch_embed.patch_embed.proj.weight"] == (1024, 3, 16, 16)


@needs_torch
def test_load_torch_file_reads_re10k_like_chunks(tmp_path):
    """eval/data/re10k.py reads pixelSplat chunks (a list of dicts of tensors) with load_torch_file."""
    import torch

    chunk = [{"key": "abc", "url": "u", "timestamps": torch.arange(5, dtype=torch.int64) * 1000,
              "cameras": torch.randn(5, 18), "images": [torch.randint(0, 255, (7 + i,), dtype=torch.uint8)
                                                         for i in range(5)]}]
    p = tmp_path / "chunk.torch"
    torch.save(chunk, p)
    got = cw.load_torch_file(p)
    assert got[0]["key"] == "abc" and got[0]["timestamps"].array().tolist() == [0, 1000, 2000, 3000, 4000]
    assert np.array_equal(got[0]["cameras"].array(), chunk[0]["cameras"].numpy())
    assert [x.array().tobytes() for x in got[0]["images"]] == [x.numpy().tobytes() for x in chunk[0]["images"]]
