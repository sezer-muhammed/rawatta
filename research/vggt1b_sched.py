#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E8 (generality): the per-layer global-attention schedule, SageAttention2
quantisation and FP8 GEMMs applied to upstream VGGT-1B (facebook/VGGT-1B,
14-px patches) in PyTorch, scored with the official VGGT-Omega evaluator.

Runs in the research container via research/run.sh (data loading and the record
schema are reused from research/run_official.py).
Research code, unsupported: kept as used for the paper (see research/README.md), with
the container names and host paths of the authors' research codebase replaced by options
and environment variables.
The replacement attention forward follows upstream facebookresearch/vggt's
Attention.forward (that package is imported at run time from sparse-vggt's pinned
submodule, not vendored; see its licence).

The engine is not ported to 14-px patches; this is the PyTorch route (experiment E8):
  schedule  per global block: dense | frame | merge2 | merge4. frame = attention
            within each frame (AVGGT-style); mergeS = keys/values of every frame
            but frame 0 have their patch grid SxS mean-pooled (special tokens
            kept) with a +log(count) logit bias (GPA-VGGT-style), as the authors'
            earlier VGGT-Omega PyTorch emulation did (lost; see README).  bf16 SDPA (attn_mask carries the bias).
  sage      the dense global layers run the SageAttention2 kernel from the
            SpargeAttn package (Apache-2.0, built for sm_89 by setup.sh):
            `block_sparse_sage2_attn_cuda` with an all-ones block map and the
            PV threshold disabled = dense INT8 QK (per-warp) / FP8 PV / FP32
            accumulation.  Frame and merged layers stay bf16 (the engine runs
            merged layers quantised; the kernel has no logit bias).
  fp8       simulated: per-token activation x per-output-channel weight FP8
            E4M3 quantise-dequantise on qkv/proj/fc1/fc2 of frame and global
            blocks 16-23 (the engine's P2_FP8_BLOCKS).  Accuracy only.

Subcommands:
  eval    --configs a,b --data-root --frames  -> one results.jsonl record per config
  search  greedy calibration search (search_v2.py's domains, budgets and order),
          resumable state in --state, stops after --minutes (re-run to continue)
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
OFFICIAL = len(sys.argv) > 1 and sys.argv[1] == "official"
VGGT = load_and_preprocess_images = pose_encoding_to_extri_intri = None


def _upstream():
    """Imports upstream vggt (sparse-vggt's pinned submodule) after argument parsing, so --help
    works without it. `official` lets run_official.py pick its own `vggt` package (GPA ships one)."""
    global VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri
    ro.setup_paths("dense")
    from vggt.models.vggt import VGGT  # noqa: PLC0415
    from vggt.utils.load_fn import load_and_preprocess_images  # noqa: PLC0415
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: PLC0415

RESULTS = ro.RESULTS
_omega_module = ro.omega_module


def _omega_module_posefallback(name):
    """official datasets.Dataset.load needs depths/; CO3Dv2 and RE10K subsets are
    pose-only (no depths/), so load frames + cameras.npz there and leave depth None."""
    mod = _omega_module(name)
    if name == "datasets":
        orig = mod.Dataset.load

        def load(self, scene, indices):
            root = Path(self.root) / scene
            if (root / "depths").is_dir():
                return orig(self, scene, indices)
            names = [ln.strip() for ln in (root / "frames.txt").read_text().splitlines() if ln.strip()]
            ext = np.load(root / "cameras.npz")["extrinsics"]
            from types import SimpleNamespace
            return SimpleNamespace(image_paths=[str(root / "images" / names[i]) for i in indices],
                                   extrinsics=ext[indices], depth=None)
        mod.Dataset.load = load
    return mod


ro.omega_module = _omega_module_posefallback
FP8_BLOCKS = range(16, 24)
CTX = {"S": 1, "P": 1, "H": 1, "W": 1, "start": 5}

# search_v2.py: modes in the order tried, budgets, calibration domains
MODES = ["frame", "merge4", "merge2"]
BUDGETS = {"safe": (0.5, 0.3), "balanced": (1.5, 0.5), "fast": (3.0, 1.0)}
CAL = {  # roots relative to --data
    "outdoor": ("mipnerf360", {s: 50 for s in ("bicycle", "garden", "stump", "kitchen")}),
    "object": ("co3d", {s: 10 for s in (
        "banana_434_61617_120791", "banana_47_2694_7604", "banana_595_90690_181239",
        "banana_606_94897_190816", "banana_612_97867_196978", "donut_198_21296_42378",
        "donut_370_40631_80640", "donut_404_53454_104227", "donut_434_61642_120955",
        "donut_616_99283_198865")}),
    "indoor": ("re10k", {s: 10 for s in (
        "004e9db3337e8206", "008cd8c450342e49", "01842c6b21e1d679", "0223924f43297881",
        "02b59cd60efb924e", "040de715f9303ba5", "05e2f342142bc50d", "05ef56b2656c9318",
        "0c4a239e265ae1c3", "0e3951bf1db22064")}),
}


# ----------------------------------------------------------------------------
# attention variants
# ----------------------------------------------------------------------------
def sage2(q, k, v):
    from spas_sage_attn.core import block_sparse_sage2_attn_cuda
    B, H, N, _ = q.shape
    if N < 128:
        return F.scaled_dot_product_attention(q, k, v)
    mask = torch.ones(B, H, math.ceil(N / 128), math.ceil(k.shape[2] / 64), dtype=torch.bool, device=q.device)
    return block_sparse_sage2_attn_cuda(q, k, v, mask_id=mask, pvthreshd=1e6, output_dtype=q.dtype)


def merge_kv(k, v, s):
    """[B,h,S*P,d] -> frame 0 dense, other frames: specials + s x s pooled patches; log-count bias."""
    B, h, _, d = k.shape
    S, P, Hh, Ww, st = CTX["S"], CTX["P"], CTX["H"], CTX["W"], CTX["start"]

    def pool(x):
        x = x.view(B, h, S, P, d)[:, :, 1:]
        sp, pt = x[:, :, :, :st], x[:, :, :, st:]
        pt = pt.reshape(B * h * (S - 1), Hh, Ww, d).permute(0, 3, 1, 2)
        pt = F.avg_pool2d(pt.float(), s, s, ceil_mode=True).to(x.dtype)
        return torch.cat([sp, pt.permute(0, 2, 3, 1).reshape(B, h, S - 1, -1, d)], 3)

    ones = torch.ones(1, 1, Hh, Ww, device=k.device)
    cnt = F.avg_pool2d(ones, s, s, ceil_mode=True, divisor_override=1).flatten()
    row = torch.cat([torch.zeros(st, device=k.device), cnt.log()])
    bias = torch.cat([torch.zeros(P, device=k.device), row.repeat(S - 1)])
    km = torch.cat([k.view(B, h, S, P, d)[:, :, 0], pool(k).reshape(B, h, -1, d)], 2)
    vm = torch.cat([v.view(B, h, S, P, d)[:, :, 0], pool(v).reshape(B, h, -1, d)], 2)
    return km, vm, bias


def make_forward(attn, mode, quant):
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
            km, vm, b = merge_kv(k, v, int(mode[5:]))
            o = F.scaled_dot_product_attention(q, km, vm, attn_mask=b.to(q.dtype).view(1, 1, 1, -1).expand(B, 1, N, -1))
        elif quant:
            o = sage2(q, k, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v)
        return attn.proj(o.transpose(1, 2).reshape(B, N, C))
    return forward


class Fp8Linear(torch.nn.Module):
    """Quantise-dequantise wrapper: per-token activations, per-out-channel weights, E4M3."""

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
        return F.linear(xq, self.wq.to(x.dtype), None if self.lin.bias is None else self.lin.bias.to(x.dtype))


def parse_config(name):
    """`<schedule>@<bf16|sage>[+fp8]`; schedule = dense | L<i>:<mode>,..."""
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


class Model:
    def __init__(self, ckpt, depth=True):
        m = VGGT(enable_track=False, enable_point=False, enable_depth=depth)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        keys = set(m.state_dict())
        missing, _ = m.load_state_dict({k: v for k, v in sd.items() if k in keys}, strict=False)
        assert not missing, missing[:5]
        del sd
        # bf16 trunk weights (fp32 weights + 24 kept trunk outputs do not fit 50 frames in 12 GB)
        m.aggregator.to(torch.bfloat16)  # heads stay fp32 (they disable autocast)
        self.m = m.eval().cuda()
        ag = self.m.aggregator
        # keep only the trunk outputs the heads read (4, 11, 17, 23); same values, less memory.
        # sparse-vggt's aggregator forward does exactly this (called from the mounted repo, not vendored)
        from types import MethodType
        from sparse_vggt.models.vggt import sparse_vggt_aggregator_forward
        ag.forward = MethodType(sparse_vggt_aggregator_forward, ag)
        self.blocks = ag.global_blocks
        self.orig = {}
        for i in FP8_BLOCKS:
            for b in (ag.frame_blocks[i], ag.global_blocks[i]):
                for path in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                    parent, attr = path.split(".")
                    lin = getattr(getattr(b, parent), attr)
                    self.orig[(id(b), path)] = (getattr(b, parent), attr, lin, Fp8Linear(lin))

    def configure(self, sched, quant, fp8):
        for i, b in enumerate(self.blocks):
            b.attn.forward = make_forward(b.attn, sched.get(i, "dense"), quant)
        for parent, attr, lin, f8 in self.orig.values():
            setattr(parent, attr, f8 if fp8 else lin)

    @torch.inference_mode()
    def __call__(self, images):
        S, _, Hi, Wi = images.shape
        p = self.m.aggregator.patch_embed.patch_size if hasattr(self.m.aggregator.patch_embed, "patch_size") else 14
        p = p[0] if isinstance(p, tuple) else p
        CTX.update(S=S, H=Hi // p, W=Wi // p, start=self.m.aggregator.patch_start_idx)
        CTX["P"] = CTX["start"] + CTX["H"] * CTX["W"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m(images)


def layer_flops(mode, S, P, HW, start=5, D=1024):
    N = S * P
    if mode == "dense":
        kv = N
    elif mode == "frame":
        kv = P
    else:
        s = int(mode[5:])
        h, w = HW
        kv = P + (S - 1) * (start + math.ceil(h / s) * math.ceil(w / s))
    return 4 * N * kv * D


def flop_ratio(sched, S=50, HW=(25, 37)):
    P = 5 + HW[0] * HW[1]
    dense = 24 * layer_flops("dense", S, P, HW)
    return sum(layer_flops(sched.get(i, "dense"), S, P, HW) for i in range(24)) / dense


# ----------------------------------------------------------------------------
def predict(model, sc):
    images = load_and_preprocess_images(sc.image_paths).cuda()
    pred = model(images)
    ext, _ = pose_encoding_to_extri_intri(pred["pose_enc"].float(), images.shape[-2:])
    return images, pred, ro.metrics_mod.pad_extrinsics(ext[0].double().cpu().numpy())


def cmd_eval(args):
    metrics = ro.metrics_mod
    scenes = ro.load_scenes(Path(args.data_root), args.frames, args.max_scenes)
    model = Model(args.checkpoint, depth=not args.no_depth)
    for cfg in args.configs.split(";" if ";" in args.configs else ","):  # ";" when schedules contain ","
        sched, quant, fp8 = parse_config(cfg)
        model.configure(sched, quant, fp8)
        util_before = ro.gpu_util()
        torch.cuda.reset_peak_memory_stats()
        rots, trans, per_scene, lat_all = [], [], {}, []
        for sc in scenes:
            t0 = time.time()
            images, pred, ext = predict(model, sc)
            row = {"n_frames": len(sc.image_paths), "first_call_s": round(time.time() - t0, 3)}
            r, t = metrics.pose_errors(ext, sc.extrinsics)
            rots.append(r)
            trans.append(t)
            row.update(metrics.pose_metrics(r, t))
            if sc.depth is not None and "depth" in pred:
                pd = pred["depth"].float().cpu().numpy()[0][..., 0]
                tgt = metrics.crop_depth(sc.depth, ro.vggt_crop_box(sc.image_paths[0]))
                row.update(metrics.depth_metrics(metrics.resize_depth(pd, tgt.shape[-2:]), tgt))
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
        lat = {"median": float(np.median(med)) if med else None, "k": 5 if med else 0, "warmup": 2,
               "per_scene_median": dict(zip(per_scene, med))}
        nf = sorted({v["n_frames"] for v in per_scene.values()})
        rec = {"exp": args.exp, "task": "paper-e8-gen", "impl": "pytorch-e8", "method": "vggt1b-sched",
               "model": "VGGT-1B", "config": cfg, "schedule": sched_str(sched),
               "precision": ("sage2(spas_sage_attn) dense-global" if quant else "bf16")
               + (" +fp8-sim blocks16-23" if fp8 else "") + "; bf16 weights + bf16 autocast",
               "flop_ratio_attn_global_n50": round(flop_ratio(sched), 4),
               "dataset": args.dataset or "/".join(Path(args.data_root).parts[-2:]),
               "frames": args.frames, "n_frames": nf[0] if len(nf) == 1 else nf,
               "scenes": list(per_scene), "metrics": m, "per_scene": per_scene, "latency_ms": lat,
               "peak_mem_bytes": int(torch.cuda.max_memory_allocated()),
               "gpu": {"lock_held": bool(args.lock_held), "util_before": util_before, "util_after": ro.gpu_util(),
                       "name": torch.cuda.get_device_name(0)},
               "checkpoint": Path(args.checkpoint).name, "torch": torch.__version__,
               "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(cfg, json.dumps({"metrics": m, "latency_ms": lat["median"],
                               "peak_GiB": rec["peak_mem_bytes"] / 2**30}), flush=True)
    return 0


# ----------------------------------------------------------------------------
def cmd_search(args):
    """search_v2.py's algorithm on VGGT-1B: single-layer sensitivity (frame) to order
    the layers, then per budget greedy accept of the first mode in MODES that keeps the
    worst-domain AUC drop within budget. bf16 attention (as v2). Resumable."""
    st_path = Path(args.state)
    st = json.loads(st_path.read_text()) if st_path.exists() else {"evals": {}}
    t_end = time.time() + args.minutes * 60
    model = Model(args.checkpoint, depth=False)
    doms = {d: ro.load_scenes(Path(args.data) / root, _tmp_map(fm), None) for d, (root, fm) in CAL.items()}
    imgs = {d: [(load_and_preprocess_images(sc.image_paths), sc.extrinsics) for sc in v] for d, v in doms.items()}

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
                ext, _ = pose_encoding_to_extri_intri(pred["pose_enc"].float(), im.shape[-2:])
                ext = ro.metrics_mod.pad_extrinsics(ext[0].double().cpu().numpy())
                pm = ro.metrics_mod.pose_metrics(*ro.metrics_mod.pose_errors(ext, gt))
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

    base = score({})
    sens = {}
    for L in range(24):
        for m in ("frame", "merge4"):
            sens[f"{L}:{m}"] = drops(score({L: m}), base)
    order = sorted(range(24), key=lambda L: np.mean([base[d]["AUC@3"] - score({L: "frame"})[d]["AUC@3"] for d in base]))
    res = {}
    for bname, (b3, b30) in BUDGETS.items():
        sched = {}
        for L in order:
            # prefilter (GPU-time budget; search_v2 tries every layer): skip a layer whose best
            # single-layer worst-domain AUC@3 drop already exceeds the budget by > 1 point
            if min(sens[f"{L}:frame"][0], sens[f"{L}:merge4"][0]) > b3 + 1.0:
                continue
            for m in MODES:
                trial = {**sched, L: m}
                w3, w30 = drops(score(trial), base)
                if w3 <= b3 and w30 <= b30:
                    sched = trial
                    break
        res[bname] = {"config_str": sched_str(sched), "flop_ratio": round(flop_ratio(sched), 4)}
        print(f"[search] {bname}: {sched_str(sched)}  FLOPs {flop_ratio(sched):.1%}", flush=True)
    st["sensitivity"] = sens
    st["order"] = order
    st["schedules"] = res
    st["done"] = True
    st_path.write_text(json.dumps(st, indent=1))
    return 0


def _tmp_map(fm):
    p = Path(tempfile.gettempdir()) / f"e8map-{abs(hash(json.dumps(fm, sort_keys=True)))}.json"
    p.write_text(json.dumps(fm))
    return str(p)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if OFFICIAL:  # run_official.py with the pose-only loader
        sys.argv = [str(TOOLS / "run_official.py")] + sys.argv[2:]
        return ro.main()
    p.add_argument("cmd", choices=("eval", "search", "official"))
    p.add_argument("--checkpoint", default=os.environ.get("VGGT1B_CKPT"), help="facebook/VGGT-1B model.pt")
    p.add_argument("--data", default=os.environ.get("RAWATTA_DATA", "data"), help="dataset root (search)")
    p.add_argument("--configs", default="dense@bf16")
    p.add_argument("--data-root")
    p.add_argument("--frames")
    p.add_argument("--dataset")
    p.add_argument("--max-scenes", type=int)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--exp", default="E8-vggt1b")
    p.add_argument("--state", default="runs/research/e8/search_vggt1b.json")
    p.add_argument("--minutes", type=float, default=13.0)
    p.add_argument("--lock-held", type=int, default=int(os.environ.get("PO_LOCK_HELD", "0")))
    args = p.parse_args()
    if not args.checkpoint:
        p.error("--checkpoint (or $VGGT1B_CKPT) is required")
    _upstream()
    torch.manual_seed(0)
    ro.metrics_mod = ro.omega_module("metrics")
    Path(args.state).parent.mkdir(parents=True, exist_ok=True)
    return cmd_eval(args) if args.cmd == "eval" else cmd_search(args)


if __name__ == "__main__":
    raise SystemExit(main())
