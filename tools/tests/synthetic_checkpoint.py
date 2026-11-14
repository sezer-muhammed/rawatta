#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a synthetic VGGT-Omega checkpoint for converter tests (needs PyTorch).

The key names, key order, ranks and dtypes follow the official model's state
dict (1411 tensors, float32, `qkv.bias_mask` buffers zero in the aggregator and
[1, 0, 1] in the camera head, as in the released checkpoint); every width is
divided by --scale so the file stays small. --scale 1 gives the real shapes
(checked against `VGGTOmega().state_dict()` of the pinned official code).
Values are random, with special cases mixed in (bf16 rounding ties, float32
subnormals, -0.0, all-zero rows, wide dynamic range) so that every branch of the
BF16 and FP8 casts is exercised.

    python3 tools/tests/synthetic_checkpoint.py --out /tmp/synth.pt [--scale 32]
        [--dtype f32|bf16|mixed] [--wrap none|model|state_dict] [--views]
        [--agg-mask zero|pattern] [--extra] [--seed 0]

--views stores some tensors as strided views into shared storages (as torch.save
does for slices), --extra adds `text_alignment_head.*` tensors (dropped by the
converter), --agg-mask pattern gives the aggregator masks the [1, 0, 1] pattern
so their biases are kept.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict

import numpy as np


def key_shapes(scale: int = 1, depth: int = 24, cam_depth: int = 4) -> list[tuple[str, tuple[int, ...]]]:
    """(name, shape) of every VGGT-Omega state-dict tensor, in state-dict order."""
    c = lambda x: max(1, x // scale)  # noqa: E731
    D, H = c(1024), c(2048)
    hd = c(64)
    out: list[tuple[str, tuple[int, ...]]] = []

    def block(p: str, dim: int, qk: bool) -> None:
        out.extend([(f"{p}.norm1.weight", (dim,)), (f"{p}.norm1.bias", (dim,))])
        if qk:
            out.extend([(f"{p}.attn.q_norm.weight", (hd,)), (f"{p}.attn.q_norm.bias", (hd,)),
                        (f"{p}.attn.k_norm.weight", (hd,)), (f"{p}.attn.k_norm.bias", (hd,))])
        out.extend([
            (f"{p}.attn.qkv.weight", (3 * dim, dim)), (f"{p}.attn.qkv.bias", (3 * dim,)),
            (f"{p}.attn.qkv.bias_mask", (3 * dim,)),
            (f"{p}.attn.proj.weight", (dim, dim)), (f"{p}.attn.proj.bias", (dim,)),
            (f"{p}.ls1.gamma", (dim,)), (f"{p}.norm2.weight", (dim,)), (f"{p}.norm2.bias", (dim,)),
            (f"{p}.mlp.fc1.weight", (4 * dim, dim)), (f"{p}.mlp.fc1.bias", (4 * dim,)),
            (f"{p}.mlp.fc2.weight", (dim, 4 * dim)), (f"{p}.mlp.fc2.bias", (dim,)),
            (f"{p}.ls2.gamma", (dim,)),
        ])

    a = "aggregator"
    out += [(f"{a}.camera_token", (1, 2, 1, D)), (f"{a}.register_token", (1, 2, 16, D)),
            (f"{a}.patch_embed.cls_token", (1, 1, D)), (f"{a}.patch_embed.storage_tokens", (1, 4, D)),
            (f"{a}.patch_embed.mask_token", (1, D)),
            (f"{a}.patch_embed.patch_embed.proj.weight", (D, 3, 16, 16)),
            (f"{a}.patch_embed.patch_embed.proj.bias", (D,)), (f"{a}.patch_embed.rope_embed.periods", (16,))]
    for i in range(depth):
        block(f"{a}.patch_embed.blocks.{i}", D, False)
    out += [(f"{a}.patch_embed.norm.weight", (D,)), (f"{a}.patch_embed.norm.bias", (D,)),
            (f"{a}.rope_embed.periods", (16,))]
    for i in range(depth):
        block(f"{a}.frame_blocks.{i}", D, True)
    for i in range(depth):
        block(f"{a}.inter_frame_blocks.{i}", D, True)
    out += [("camera_head.token_norm.weight", (H,)), ("camera_head.token_norm.bias", (H,))]
    for i in range(cam_depth):
        block(f"camera_head.trunk.{i}", H, False)
    out += [("camera_head.trunk_norm.weight", (H,)), ("camera_head.trunk_norm.bias", (H,)),
            ("camera_head.camera_branch.0.weight", (c(1024), H)), ("camera_head.camera_branch.0.bias", (c(1024),)),
            ("camera_head.camera_branch.2.weight", (9, c(1024))), ("camera_head.camera_branch.2.bias", (9,))]
    F = c(256)
    chans = [c(256), c(512), c(1024), c(1024)]
    out += [("dense_head.norm.weight", (H,)), ("dense_head.norm.bias", (H,))]
    for i, ch in enumerate(chans):
        out += [(f"dense_head.projects.{i}.weight", (ch, H, 1, 1)), (f"dense_head.projects.{i}.bias", (ch,))]
    out += [("dense_head.resize_layers.0.weight", (chans[0], chans[0], 4, 4)),
            ("dense_head.resize_layers.0.bias", (chans[0],)),
            ("dense_head.resize_layers.1.weight", (chans[1], chans[1], 2, 2)),
            ("dense_head.resize_layers.1.bias", (chans[1],)),
            ("dense_head.resize_layers.3.weight", (chans[3], chans[3], 3, 3)),
            ("dense_head.resize_layers.3.bias", (chans[3],))]
    for i, ch in enumerate(chans):
        out.append((f"dense_head.scratch.layer{i + 1}_rn.weight", (F, ch, 3, 3)))
    for r in (1, 2, 3, 4):
        p = f"dense_head.scratch.refinenet{r}"
        out += [(f"{p}.out_conv.weight", (F, F, 1, 1)), (f"{p}.out_conv.bias", (F,))]
        for u in ((1, 2) if r < 4 else (2,)):
            for k in (1, 2):
                out += [(f"{p}.resConfUnit{u}.conv{k}.weight", (F, F, 3, 3)), (f"{p}.resConfUnit{u}.conv{k}.bias", (F,))]
    out += [("dense_head.proj.weight", (16, F, 1, 1)), ("dense_head.proj.bias", (16,)),
            ("dense_head.proj_conf.weight", (16, F, 1, 1)), ("dense_head.proj_conf.bias", (16,))]
    return out


def build(scale=32, dtype="f32", views=False, agg_mask="zero", extra=False, seed=0):
    import torch  # noqa: PLC0415

    g = torch.Generator().manual_seed(seed)
    dts = {"f32": [torch.float32], "bf16": [torch.bfloat16],
           "mixed": [torch.float32, torch.bfloat16, torch.float16, torch.float64]}[dtype]
    sd: OrderedDict = OrderedDict()
    shared = None
    for idx, (name, shape) in enumerate(key_shapes(scale)):
        dt = dts[idx % len(dts)]
        if name.endswith(".bias_mask"):
            n = shape[0]
            m = torch.ones(n)
            m[n // 3: 2 * n // 3] = 0
            if name.startswith("aggregator.") and agg_mask == "zero":
                m.zero_()
            sd[name] = m.to(dt)
            continue
        t = torch.randn(*shape, generator=g) * (0.02 if len(shape) >= 2 else 1.0)
        flat = t.view(-1)
        n = flat.numel()
        if n >= 64:
            # bf16 rounding ties (low 16 bits exactly 0x8000), odd and even upper halves
            ties = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0x40498000], dtype=np.uint32)
            flat[:4] = torch.from_numpy(ties.view(np.float32).copy())
            flat[4] = -0.0
            flat[5] = 1e-40  # float32 subnormal
            flat[6] = -3e-39
            flat[7] = 3.0e38 if dt in (torch.float32, torch.float64) else 1e4
            flat[8] = 2.0 ** -10  # fp8 subnormal range after scaling
        if len(shape) == 2 and shape[0] >= 4:
            t[1].zero_()  # all-zero row: fp8 scale clamps to 1e-12
            t[2] *= 1e-6
        t = t.to(dt)
        if views and len(shape) == 2 and idx % 3 == 0:
            # a transposed, offset view into a larger storage
            big = torch.zeros(shape[1] + 1, shape[0] + 2, dtype=dt)
            big[1:, 2:] = t.t()
            t = big[1:, 2:].t()
            assert not t.is_contiguous()
        elif views and len(shape) == 1 and dt == torch.float32:
            # 1-D tensors sharing one storage at different offsets
            if shared is None or shared[1] + shape[0] > shared[0].numel():
                shared = [torch.zeros(1 << 16), 0]
            base, off = shared
            base[off: off + shape[0]] = t
            t = base[off: off + shape[0]]
            shared[1] = off + shape[0] + 3
        sd[name] = t
    if extra:
        sd["text_alignment_head.proj.weight"] = torch.randn(8, 16, generator=g)
        sd["text_alignment_head.proj.bias"] = torch.randn(8, generator=g)
    return sd


def save(path, scale=32, dtype="f32", wrap="none", views=False, agg_mask="zero", extra=False, seed=0):
    import torch  # noqa: PLC0415

    sd = build(scale, dtype, views, agg_mask, extra, seed)
    obj = sd if wrap == "none" else {wrap: sd, "epoch": 7}
    torch.save(obj, path)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=int, default=32, help="divide every width by this (1 = real shapes)")
    p.add_argument("--dtype", choices=("f32", "bf16", "mixed"), default="f32")
    p.add_argument("--wrap", choices=("none", "model", "state_dict"), default="none")
    p.add_argument("--views", action="store_true")
    p.add_argument("--agg-mask", choices=("zero", "pattern"), default="zero")
    p.add_argument("--extra", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    save(a.out, a.scale, a.dtype, a.wrap, a.views, a.agg_mask, a.extra, a.seed)
    print(f"wrote {a.out}: {len(key_shapes(a.scale))} tensors (scale {a.scale}, {a.dtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
