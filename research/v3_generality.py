#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""v3-generality: regime-aware schedule on VGGT-1B and π3 (PyTorch route).

Extends the E8 harness with π3 support and the one-pass regime rule.  Runs in the
research container via research/run.sh.  Model-agnostic: schedule, Sage and FP8
work on either model.
Research code, unsupported: kept as used for the paper (see research/README.md), with
the container names and host paths of the authors' research codebase replaced by options
and environment variables.
The replacement attention forward follows upstream facebookresearch/vggt's
Attention.forward; π3 (yyfz233/Pi3) is imported from sparse-vggt's pinned
submodule. Neither is vendored; see their licences.

Differences from vggt1b_sched.py:
  - π3 model support (36-block decoder, odd blocks are global, no reference frame)
  - merge without frame-0 exemption for π3 (π3 is permutation-equivariant)
  - camera-to-world → world-to-camera conversion for π3 pose output
  - configurable seed for multi-seed evaluation
  - rule-derived schedules as named configs

Subcommands:
  eval    --model vggt1b|pi3 --configs a;b --data-root --frames  -> results.jsonl
  search  greedy calibration search (same protocol as E8)
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TOOLS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_official", TOOLS / "run_official.py")
ro = importlib.util.module_from_spec(_spec)
sys.modules["run_official"] = ro
_spec.loader.exec_module(ro)

_orig_omega_module = ro.omega_module
def _patched_omega_module(name):
    mod = _orig_omega_module(name)
    if name == "datasets" and hasattr(mod, "_read_depth"):
        _orig_rd = mod._read_depth
        def _safe_rd(path):
            try:
                return _orig_rd(path)
            except FileNotFoundError:
                return np.zeros((1, 1), dtype=np.float32)
        mod._read_depth = _safe_rd
    return mod
ro.omega_module = _patched_omega_module

load_and_preprocess_images = pose_encoding_to_extri_intri = None


def _upstream():
    """Imports upstream vggt (sparse-vggt's pinned submodule) after argument parsing, so --help
    works without it."""
    global load_and_preprocess_images, pose_encoding_to_extri_intri
    if load_and_preprocess_images is not None:
        return
    ro.setup_paths("dense")
    from vggt.utils.load_fn import load_and_preprocess_images  # noqa: PLC0415
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: PLC0415

RESULTS = ro.RESULTS

VGGT1B_RULE_A = "L0:frame,L1:frame,L2:frame,L3:frame,L4:frame,L5:frame,L6:frame,L7:frame,L8:frame,L9:frame,L16:merge4,L17:merge4,L18:merge4,L19:merge4,L20:merge4,L22:merge4,L23:merge4"
VGGT1B_RULE_B = "L0:merge4,L1:merge4,L2:merge4,L3:merge4,L4:merge4,L5:merge4,L6:merge4,L7:merge4,L8:merge4,L9:merge4,L16:merge4,L17:merge4,L18:merge4,L19:merge4,L20:merge4,L22:merge4,L23:merge4"

PI3_RULE_A = "L0:frame,L1:frame,L2:frame,L3:frame,L4:frame,L5:frame,L6:frame,L7:frame,L8:frame,L9:frame,L10:frame,L17:merge4nx"
PI3_RULE_B = "L0:merge4nx,L1:merge4nx,L2:merge4nx,L3:merge4nx,L4:merge4nx,L5:merge4nx,L6:merge4nx,L7:merge4nx,L8:merge4nx,L9:merge4nx,L10:merge4nx,L17:merge4nx"

NAMED = {
    "vggt1b-rule-a": VGGT1B_RULE_A,
    "vggt1b-rule-b": VGGT1B_RULE_B,
    "pi3-rule-a": PI3_RULE_A,
    "pi3-rule-b": PI3_RULE_B,
}

CTX = {"S": 1, "P": 1, "H": 1, "W": 1, "start": 5, "keep0": True}
FP8_BLOCKS_1B = range(16, 24)
FP8_BLOCKS_PI3 = range(12, 18)


def sage2(q, k, v):
    from spas_sage_attn.core import block_sparse_sage2_attn_cuda
    B, H, N, _ = q.shape
    if N < 128:
        return F.scaled_dot_product_attention(q, k, v)
    mask = torch.ones(B, H, math.ceil(N / 128), math.ceil(k.shape[2] / 64),
                      dtype=torch.bool, device=q.device)
    return block_sparse_sage2_attn_cuda(q, k, v, mask_id=mask, pvthreshd=1e6,
                                        output_dtype=q.dtype)


def merge_kv(k, v, s, keep0=True):
    """Merge KV: specials kept, s×s mean-pooled patches, +log(count) bias.
    keep0=True: frame 0 stays dense (VGGT).  keep0=False: all frames merged (π3)."""
    B, h, _, d = k.shape
    S, P, Hh, Ww, st = CTX["S"], CTX["P"], CTX["H"], CTX["W"], CTX["start"]
    first = 1 if keep0 else 0

    def pool(x):
        x = x.view(B, h, S, P, d)[:, :, first:]
        sp, pt = x[:, :, :, :st], x[:, :, :, st:]
        pt = pt.reshape(B * h * (S - first), Hh, Ww, d).permute(0, 3, 1, 2)
        pt = F.avg_pool2d(pt.float(), s, s, ceil_mode=True).to(x.dtype)
        return torch.cat([sp, pt.permute(0, 2, 3, 1).reshape(B, h, S - first, -1, d)], 3)

    ones = torch.ones(1, 1, Hh, Ww, device=k.device)
    cnt = F.avg_pool2d(ones, s, s, ceil_mode=True, divisor_override=1).flatten()
    row = torch.cat([torch.zeros(st, device=k.device), cnt.log()])
    bias_merged = row.repeat(S - first)
    if keep0:
        bias = torch.cat([torch.zeros(P, device=k.device), bias_merged])
        km = torch.cat([k.view(B, h, S, P, d)[:, :, 0],
                        pool(k).reshape(B, h, -1, d)], 2)
        vm = torch.cat([v.view(B, h, S, P, d)[:, :, 0],
                        pool(v).reshape(B, h, -1, d)], 2)
    else:
        bias = bias_merged
        km = pool(k).reshape(B, h, -1, d)
        vm = pool(v).reshape(B, h, -1, d)
    return km, vm, bias


def make_forward_vggt1b(attn, mode, quant):
    def forward(x, pos=None):
        B, N, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        if attn.rope is not None:
            q, k = attn.rope(q, pos), attn.rope(k, pos)
        S, P = CTX["S"], CTX["P"]
        if mode == "frame" and S > 1:
            h, d = q.shape[1], q.shape[-1]
            r = lambda t: t.view(B, h, S, P, d).transpose(1, 2).reshape(B * S, h, P, d)  # noqa: E731
            o = F.scaled_dot_product_attention(r(q), r(k), r(v))
            o = o.view(B, S, h, P, d).transpose(1, 2).reshape(B, h, N, d)
        elif mode.startswith("merge") and S > 1:
            s = int(mode.replace("nx", "")[5:])
            keep0 = "nx" not in mode
            km, vm, b = merge_kv(k, v, s, keep0=keep0)
            o = F.scaled_dot_product_attention(
                q, km, vm,
                attn_mask=b.to(q.dtype).view(1, 1, 1, -1).expand(B, 1, N, -1))
        elif quant:
            o = sage2(q, k, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v)
        return attn.proj(o.transpose(1, 2).reshape(B, N, C))
    return forward


class Fp8Linear(torch.nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin
        w = lin.weight.float()
        s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        self.wq = ((w / s).to(torch.float8_e4m3fn).float() * s).to(lin.weight.dtype)

    def forward(self, x):
        xf = x.float()
        s = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
        xq = ((xf / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)
        return F.linear(xq, self.wq.to(x.dtype),
                        None if self.lin.bias is None else self.lin.bias.to(x.dtype))


def parse_config(name, model_name):
    """Parse config: `<schedule>@<bf16|sage>[+fp8]` or named config."""
    if name in NAMED:
        name = NAMED[name] + "@sage"
    fp8 = name.endswith("+fp8")
    base = name[:-4] if fp8 else name
    sched_s, _, prec = base.partition("@")
    prec = prec or "bf16"
    assert prec in ("bf16", "sage"), name
    sched = {}
    if sched_s != "dense":
        for part in sched_s.split(","):
            li, m = part.split(":")
            sched[int(li.lstrip("L"))] = m
    return sched, prec == "sage", fp8


def sched_str(sched):
    return ",".join(f"L{k}:{sched[k]}" for k in sorted(sched)) or "dense"


# ---- VGGT-1B model ----
class VGGT1BModel:
    def __init__(self, ckpt, depth=True):
        from vggt.models.vggt import VGGT
        m = VGGT(enable_track=False, enable_point=False, enable_depth=depth)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        keys = set(m.state_dict())
        missing, _ = m.load_state_dict({k: v for k, v in sd.items() if k in keys},
                                       strict=False)
        assert not missing, missing[:5]
        del sd
        m.aggregator.to(torch.bfloat16)
        self.m = m.eval().cuda()
        ag = self.m.aggregator
        from types import MethodType
        from sparse_vggt.models.vggt import sparse_vggt_aggregator_forward
        ag.forward = MethodType(sparse_vggt_aggregator_forward, ag)
        self.blocks = ag.global_blocks
        self.n_global = len(self.blocks)
        self.fp8_range = FP8_BLOCKS_1B
        self.orig = {}
        for i in self.fp8_range:
            for b in (ag.frame_blocks[i], ag.global_blocks[i]):
                for path in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                    parent, attr = path.split(".")
                    lin = getattr(getattr(b, parent), attr)
                    self.orig[(id(b), path)] = (getattr(b, parent), attr, lin,
                                                 Fp8Linear(lin))

    def configure(self, sched, quant, fp8):
        CTX["keep0"] = True
        for i, b in enumerate(self.blocks):
            b.attn.forward = make_forward_vggt1b(b.attn, sched.get(i, "dense"), quant)
        for parent, attr, lin, f8 in self.orig.values():
            setattr(parent, attr, f8 if fp8 else lin)

    @torch.inference_mode()
    def __call__(self, images):
        S, _, Hi, Wi = images.shape
        p = self.m.aggregator.patch_embed.patch_size
        p = p[0] if isinstance(p, tuple) else p
        CTX.update(S=S, H=Hi // p, W=Wi // p,
                   start=self.m.aggregator.patch_start_idx)
        CTX["P"] = CTX["start"] + CTX["H"] * CTX["W"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m(images)

    def extract_poses(self, images, pred):
        ext, _ = pose_encoding_to_extri_intri(pred["pose_enc"].float(),
                                              images.shape[-2:])
        return ro.metrics_mod.pad_extrinsics(ext[0].double().cpu().numpy())


# ---- π3 model ----
class Pi3Model:
    def __init__(self, ckpt, depth=True):
        PO = os.environ.get("PI3_SRC", str(ro.SRC / "sparse" / "external" / "Pi3"))
        sys.path.insert(0, PO)
        from pi3.models.pi3 import Pi3
        import pi3.models.layers.attention as PA
        from safetensors.torch import load_file
        m = Pi3()
        m.load_state_dict(load_file(ckpt))
        m = m.eval().cuda().to(torch.bfloat16)
        for head in (m.point_head, m.conf_head, m.camera_head):
            head.to(torch.float32)
        self.m = m
        self.PA = PA
        self.global_ids = list(range(1, len(m.decoder), 2))
        self.n_global = len(self.global_ids)
        self.fp8_range = FP8_BLOCKS_PI3
        self.orig_sdpa = PA.scaled_dot_product_attention
        self.blocks = [m.decoder[i] for i in self.global_ids]
        self.orig = {}
        for gi, di in enumerate(self.global_ids):
            if gi in self.fp8_range:
                b = m.decoder[di]
                for path in ("attn.qkv", "attn.proj"):
                    parent, attr = path.split(".")
                    lin = getattr(getattr(b, parent), attr)
                    self.orig[(id(b), path)] = (getattr(b, parent), attr, lin,
                                                 Fp8Linear(lin))

    def configure(self, sched, quant, fp8):
        CTX["keep0"] = False
        current_layer = {"v": None}

        def patched_sdpa(q, k, v, *a, **kw):
            layer = current_layer["v"]
            if layer is not None and q.dim() == 4:
                S, P = CTX["S"], CTX["P"]
                if q.shape[2] == S * P and k.shape[2] == S * P:
                    mode = sched.get(layer, "dense")
                    B, h, N, d = q.shape
                    if mode == "frame" and S > 1:
                        r = lambda t: t.view(B, h, S, P, d).transpose(1, 2).reshape(B * S, h, P, d)  # noqa: E731
                        o = self.orig_sdpa(r(q), r(k), r(v))
                        return o.view(B, S, h, P, d).transpose(1, 2).reshape(B, h, N, d)
                    elif mode.startswith("merge") and S > 1:
                        s = int(mode.replace("nx", "")[5:])
                        km, vm, b = merge_kv(k, v, s, keep0=False)
                        from torch.nn.attention import SDPBackend, sdpa_kernel
                        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                            return self.orig_sdpa(
                                q, km, vm,
                                attn_mask=b.to(q.dtype).view(1, 1, 1, -1).expand(B, 1, N, -1))
                    elif quant:
                        return sage2(q, k, v)
            return self.orig_sdpa(q, k, v, *a, **kw)

        self.PA.scaled_dot_product_attention = patched_sdpa

        for gi, di in enumerate(self.global_ids):
            def pre(m, a, kw=None, _gi=gi):
                current_layer["v"] = _gi
            def post(m, a, o, _gi=gi):
                current_layer["v"] = None
            b = self.m.decoder[di]
            b._v3_hooks = [b.register_forward_pre_hook(pre),
                           b.register_forward_hook(post)]

        for parent, attr, lin, f8 in self.orig.values():
            setattr(parent, attr, f8 if fp8 else lin)

    @torch.inference_mode()
    def __call__(self, images):
        S, _, Hi, Wi = images.shape
        p = self.m.patch_size
        CTX.update(S=S, H=Hi // p, W=Wi // p,
                   start=self.m.patch_start_idx)
        CTX["P"] = CTX["start"] + CTX["H"] * CTX["W"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m(images[None])

    def extract_poses(self, images, pred):
        c2w = pred["camera_poses"][0].double().cpu()
        R = c2w[:, :3, :3]
        t = c2w[:, :3, 3:]
        w2c = torch.zeros_like(c2w)
        w2c[:, :3, :3] = R.transpose(-1, -2)
        w2c[:, :3, 3:] = -(R.transpose(-1, -2) @ t)
        w2c[:, 3, 3] = 1.0
        return ro.metrics_mod.pad_extrinsics(w2c.numpy())


def build_model(model_name, ckpt, depth=True):
    if model_name == "vggt1b":
        return VGGT1BModel(ckpt, depth)
    elif model_name == "pi3":
        return Pi3Model(ckpt, depth)
    raise ValueError(model_name)


def layer_flops(mode, S, P, HW, start=5, D=1024, keep0=True):
    N = S * P
    if mode == "dense":
        kv = N
    elif mode == "frame":
        kv = P
    else:
        s = int(mode.replace("nx", "")[5:])
        h, w = HW
        first = 1 if (keep0 and "nx" not in mode) else 0
        pooled = (S - first) * (start + math.ceil(h / s) * math.ceil(w / s))
        kv = first * P + pooled
    return 4 * N * kv * D


def flop_ratio(sched, n_global, S=50, HW=(25, 37), keep0=True):
    P = 5 + HW[0] * HW[1]
    dense = n_global * layer_flops("dense", S, P, HW, keep0=keep0)
    return sum(layer_flops(sched.get(i, "dense"), S, P, HW, keep0=keep0)
               for i in range(n_global)) / dense


# ---- evaluation ----
def predict(model, sc):
    images = load_and_preprocess_images(sc.image_paths).cuda()
    pred = model(images)
    ext = model.extract_poses(images, pred)
    return images, pred, ext


def cmd_eval(args):
    metrics = ro.metrics_mod
    scenes = ro.load_scenes(Path(args.data_root), args.frames, args.max_scenes)
    model = build_model(args.model, args.checkpoint, depth=not args.no_depth)
    keep0 = args.model == "vggt1b"

    for cfg in args.configs.split(";"):
        sched, quant, fp8 = parse_config(cfg.strip(), args.model)
        model.configure(sched, quant, fp8)
        util_before = ro.gpu_util()
        torch.cuda.reset_peak_memory_stats()
        rots, trans, per_scene, lat_all = [], [], {}, []
        for sc in scenes:
            t0 = time.time()
            images, pred, ext = predict(model, sc)
            row = {"n_frames": len(sc.image_paths),
                   "first_call_s": round(time.time() - t0, 3)}
            r, t = metrics.pose_errors(ext, sc.extrinsics)
            rots.append(r)
            trans.append(t)
            row.update(metrics.pose_metrics(r, t))
            if sc.depth is not None and "depth" in pred:
                pd = pred["depth"].float().cpu().numpy()[0][..., 0]
                tgt = metrics.crop_depth(sc.depth, ro.vggt_crop_box(sc.image_paths[0]))
                row.update(metrics.depth_metrics(
                    metrics.resize_depth(pd, tgt.shape[-2:]), tgt))
            del pred
            if args.timing:
                ts = ro.time_forward(model, images)
                row["latency_ms"] = float(np.median(ts))
                lat_all.append(ts)
            per_scene[sc.name] = row
            del images
            torch.cuda.empty_cache()
        r, t = np.concatenate(rots), np.concatenate(trans)
        m = metrics.pose_metrics(r, t)
        dr = [v for v in per_scene.values() if "delta125" in v]
        if dr:
            m["delta125"] = float(np.mean([v["delta125"] for v in dr])) * 100.0
            m["AbsRel"] = float(np.mean([v["AbsRel"] for v in dr]))
        m["pose_pairs"] = int(r.size)
        med = [float(np.median(x)) for x in lat_all]
        lat = {"median": float(np.median(med)) if med else None,
               "k": 5 if med else 0, "warmup": 2,
               "per_scene_median": dict(zip(per_scene, med))}
        nf = sorted({v["n_frames"] for v in per_scene.values()})
        HW = None
        for sc in scenes:
            im = load_and_preprocess_images(sc.image_paths[:1])
            ps = 14
            HW = (im.shape[-2] // ps, im.shape[-1] // ps)
            break
        rec = {
            "exp": args.exp, "task": "v3-generality",
            "impl": "pytorch-v3gen",
            "method": f"{args.model}-rule",
            "model": args.model.upper().replace("VGGT1B", "VGGT-1B").replace("PI3", "Pi3"),
            "config": cfg.strip(),
            "schedule": sched_str(sched),
            "precision": ("sage2(spas_sage_attn) dense-global" if quant else "bf16")
                         + (" +fp8-sim" if fp8 else "")
                         + "; bf16 weights + bf16 autocast",
            "flop_ratio_attn_global_n50": round(
                flop_ratio(sched, model.n_global, keep0=keep0, HW=HW or (25, 37)), 4),
            "dataset": args.dataset or "/".join(Path(args.data_root).parts[-2:]),
            "frames": args.frames,
            "n_frames": nf[0] if len(nf) == 1 else nf,
            "seed": args.seed,
            "scenes": list(per_scene), "metrics": m, "per_scene": per_scene,
            "latency_ms": lat,
            "peak_mem_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu": {"lock_held": bool(args.lock_held),
                    "util_before": util_before,
                    "util_after": ro.gpu_util(),
                    "name": torch.cuda.get_device_name(0)},
            "checkpoint": Path(args.checkpoint).name,
            "torch": torch.__version__,
            "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(cfg.strip(), json.dumps({"metrics": m, "latency_ms": lat["median"],
                                        "peak_GiB": rec["peak_mem_bytes"] / 2**30}),
              flush=True)
    return 0


def cmd_search(args):
    """Greedy calibration search for π3 (same protocol as E8)."""
    MODES = ["frame", "merge4nx"] if args.model == "pi3" else ["frame", "merge4", "merge2"]
    BUDGETS = {"safe": (0.5, 0.3), "balanced": (1.5, 0.5), "fast": (3.0, 1.0)}
    CAL = {
        "outdoor": ("mipnerf360", {s: 50 for s in
                    ("bicycle", "garden", "stump", "kitchen")}),
        "object": ("co3d", {s: 10 for s in (
            "banana_434_61617_120791", "banana_47_2694_7604",
            "banana_595_90690_181239", "banana_606_94897_190816",
            "banana_612_97867_196978", "donut_198_21296_42378",
            "donut_370_40631_80640", "donut_404_53454_104227",
            "donut_434_61642_120955", "donut_616_99283_198865")}),
        "indoor": ("re10k", {s: 10 for s in (
            "004e9db3337e8206", "008cd8c450342e49", "01842c6b21e1d679",
            "0223924f43297881", "02b59cd60efb924e", "040de715f9303ba5",
            "05e2f342142bc50d", "05ef56b2656c9318", "0c4a239e265ae1c3",
            "0e3951bf1db22064")}),
    }
    st_path = Path(args.state)
    st = json.loads(st_path.read_text()) if st_path.exists() else {"evals": {}}
    t_end = time.time() + args.minutes * 60
    metrics = ro.metrics_mod
    model = build_model(args.model, args.checkpoint, depth=False)

    def _tmp_map(fm):
        p = Path(tempfile.gettempdir()) / f"v3map-{abs(hash(json.dumps(fm, sort_keys=True)))}.json"
        p.write_text(json.dumps(fm))
        return str(p)

    doms = {d: ro.load_scenes(Path(args.data) / root, _tmp_map(fm), None)
            for d, (root, fm) in CAL.items()}
    imgs = {d: [(load_and_preprocess_images(sc.image_paths), sc.extrinsics)
                for sc in v] for d, v in doms.items()}

    def score(sched):
        key = sched_str(sched)
        if key in st["evals"]:
            return st["evals"][key]
        if time.time() > t_end:
            st_path.write_text(json.dumps(st, indent=1))
            print("[search] time slice used up; re-run to continue", flush=True)
            sys.exit(3)
        model.configure(sched, False, False)
        out = {}
        for d, lst in imgs.items():
            a3, a30 = [], []
            for im, gt in lst:
                pred = model(im.cuda())
                ext = model.extract_poses(im.cuda(), pred)
                pm = metrics.pose_metrics(*metrics.pose_errors(ext, gt))
                a3.append(pm["AUC@3"])
                a30.append(pm["AUC@30"])
                del pred
            out[d] = {"AUC@3": float(np.mean(a3)), "AUC@30": float(np.mean(a30))}
        st["evals"][key] = out
        st_path.write_text(json.dumps(st, indent=1))
        print(f"[eval] {key}: {json.dumps(out)}", flush=True)
        return out

    def drops(r, base):
        return (max(base[d]["AUC@3"] - r[d]["AUC@3"] for d in base),
                max(base[d]["AUC@30"] - r[d]["AUC@30"] for d in base))

    n_global = model.n_global
    base = score({})
    sens = {}
    for L in range(n_global):
        for m in MODES[:2]:
            sens[f"{L}:{m}"] = drops(score({L: m}), base)
    order = sorted(range(n_global),
                   key=lambda L: np.mean([
                       base[d]["AUC@3"] - score({L: MODES[0]})[d]["AUC@3"]
                       for d in base]))
    res = {}
    for bname, (b3, b30) in BUDGETS.items():
        sched = {}
        for L in order:
            best_sl = min(sens.get(f"{L}:{m}", (999, 999))[0] for m in MODES[:2])
            if best_sl > b3 + 1.0:
                continue
            for m in MODES:
                trial = {**sched, L: m}
                w3, w30 = drops(score(trial), base)
                if w3 <= b3 and w30 <= b30:
                    sched = trial
                    break
        res[bname] = {"config_str": sched_str(sched),
                      "flop_ratio": round(flop_ratio(sched, n_global, keep0=False), 4)}
        print(f"[search] {bname}: {sched_str(sched)}  "
              f"FLOPs {flop_ratio(sched, n_global, keep0=False):.1%}", flush=True)
    st["sensitivity"] = sens
    st["order"] = order
    st["schedules"] = res
    st["done"] = True
    st_path.write_text(json.dumps(st, indent=1))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("eval", "search"))
    p.add_argument("--model", choices=("vggt1b", "pi3"), required=True)
    p.add_argument("--checkpoint", required=True, help="VGGT-1B model.pt or π3 model.safetensors")
    p.add_argument("--data", default=os.environ.get("RAWATTA_DATA", "data"), help="dataset root (search)")
    p.add_argument("--configs", default="dense@bf16")
    p.add_argument("--data-root")
    p.add_argument("--frames")
    p.add_argument("--dataset")
    p.add_argument("--max-scenes", type=int)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--exp", default="v3-generality")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state", default="runs/research/v3gen/search_state.json")
    p.add_argument("--minutes", type=float, default=13.0)
    p.add_argument("--lock-held", type=int,
                   default=int(os.environ.get("PO_LOCK_HELD", "0")))
    args = p.parse_args()
    _upstream()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ro.metrics_mod = ro.omega_module("metrics")
    Path(args.state).parent.mkdir(parents=True, exist_ok=True)
    return cmd_eval(args) if args.cmd == "eval" else cmd_search(args)


if __name__ == "__main__":
    raise SystemExit(main())
