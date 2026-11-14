#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run the *official* PyTorch code of two VGGT accelerators (and dense VGGT) on
Omega-layout or COLMAP datasets and score them with the official VGGT-Omega
evaluator functions.

Research code, unsupported: kept as used for the paper (see research/README.md), with
the container names and host paths of the authors' research codebase replaced by options
and environment variables.

Runs in the research container (research/run.sh); the official sources are
cloned by research/setup.sh into $RESEARCH_SRC and imported, never vendored:

  --method dense   upstream facebookresearch/vggt (sparse-vggt's pinned submodule)
  --method gpa     noheat61/GPA-VGGT (its own `vggt` package, paper defaults)
  --method sparse  brianwang00001/sparse-vggt (+ SpargeAttn spas_sage_attn kernels)
  --method omega   official VGGT-Omega eval/model.py Predictor (the `pt-dense` row;
                   fp32 weights, the evaluator's own preprocessing and crop)

Metrics: pose AUC@3/@30 via the official eval/metrics.py (pose_errors,
pose_metrics over all pairs of all scenes, as evaluate.py does); depth
delta125 / AbsRel via metrics.depth_metrics after the same GT handling as
evaluate.py (GT resized to width 518 with nearest, cropped to the model's input
crop, prediction resized to GT).  COLMAP scenes (Mip-NeRF 360, T&T, DB) have no
GT depth, so they contribute pose only.

Latency: CUDA events around the model forward on pre-loaded images, 2 warm-ups,
median/IQR of 5.  Peak memory: torch.cuda.max_memory_allocated over the
measured forwards (reset after model load, so it includes the weights).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import omega_src  # noqa: E402

RESEARCH_HOME = Path(os.environ.get("RESEARCH_HOME", omega_src.cache_root() / "research"))
# clones of GPA-VGGT and sparse-vggt (+ submodules vggt, SpargeAttn, Pi3), made by research/setup.sh
SRC = Path(os.environ.get("RESEARCH_SRC", os.environ.get("PO_SRC", RESEARCH_HOME / "src")))
# the official VGGT-Omega eval/ directory (default: the pinned source eval/omega_src.py fetches)
OMEGA_EVAL = Path(os.environ["OMEGA_EVAL"]) if os.environ.get("OMEGA_EVAL") else None
RESULTS = Path(os.environ.get("RESEARCH_RESULTS", "runs/research/results.jsonl"))


# ----------------------------------------------------------------------------
# environment / path setup (the two repos both ship a top-level `vggt` package)
# ----------------------------------------------------------------------------
def setup_paths(method: str) -> None:
    if method == "gpa":
        sys.path.insert(0, str(SRC / "gpa"))
    else:
        sys.path.insert(0, str(SRC / "sparse" / "external" / "Pi3"))
        sys.path.insert(0, str(SRC / "sparse" / "external" / "vggt"))
        sys.path.insert(0, str(SRC / "sparse" / "src"))


def omega_module(name: str):
    """Load the official eval/<name>.py by file path (avoids name clashes)."""
    import importlib.util
    global OMEGA_EVAL
    if OMEGA_EVAL is None:
        OMEGA_EVAL = omega_src.ensure() / "eval"
    if str(OMEGA_EVAL) not in sys.path:
        sys.path.insert(0, str(OMEGA_EVAL))
        sys.path.insert(0, str(OMEGA_EVAL.parent))
    spec = importlib.util.spec_from_file_location(f"omega_eval_{name}", OMEGA_EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    if name == "datasets" and NO_DEPTH:  # pose-only: depth maps need not be on disk
        mod._read_depth = lambda _p: np.zeros((4, 4), np.float32)
    return mod


NO_DEPTH = False


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def gpu_util() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def _qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap_images_bin(path: Path) -> dict[str, np.ndarray]:
    """name -> 4x4 world-to-camera (OpenCV), from COLMAP images.bin."""
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _id, qw, qx, qy, qz, tx, ty, tz, _cam = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.seek(npts * 24, 1)
            m = np.eye(4)
            m[:3, :3] = _qvec2rotmat((qw, qx, qy, qz))
            m[:3, 3] = (tx, ty, tz)
            out[name.decode()] = m
    return out


class Scene:
    def __init__(self, name, image_paths, extrinsics, depth):
        self.name, self.image_paths, self.extrinsics, self.depth = name, image_paths, extrinsics, depth


def uniform(n_total: int, n: int) -> list[int]:
    if n >= n_total:
        return list(range(n_total))
    return [int(round(i)) for i in np.linspace(0, n_total - 1, n)]


def load_scenes(root: Path, frames_arg: str, max_scenes: int | None):
    """Omega layout (frames.txt, cameras.npz, images/, depths/) or COLMAP
    (sparse/0/images.bin + images*/).  --frames is either a JSON file
    {scene: [indices]} (indices into frames.txt / sorted COLMAP names) or
    `uniform:N` (N evenly spaced frames of whatever is present)."""
    ds_mod = omega_module("datasets")
    if frames_arg.startswith("uniform:"):
        n = int(frames_arg.split(":", 1)[1])
        frame_map = {d.name: n for d in sorted(root.iterdir()) if d.is_dir()}
    else:
        frame_map = json.loads(Path(frames_arg).read_text())
    scenes = []
    for name, sel in frame_map.items():
        sdir = root / name
        if not sdir.is_dir():
            print(f"[skip] {name}: not present under {root}")
            continue
        if (sdir / "frames.txt").is_file():
            names = [l.strip() for l in (sdir / "frames.txt").read_text().splitlines() if l.strip()]
            if isinstance(sel, int):  # only frames whose image exists locally
                avail = [i for i, nm in enumerate(names) if (sdir / "images" / nm).is_file()]
                sel = [avail[i] for i in uniform(len(avail), sel)]
            if not all((sdir / "images" / names[i]).is_file() for i in sel):
                print(f"[skip] {name}: selected frames missing locally")
                continue
            b = ds_mod.Dataset(str(root)).load(name, sel)
            scenes.append(Scene(name, b.image_paths, b.extrinsics, b.depth))
        else:
            sp = next((p for p in [sdir / "sparse/0/images.bin", sdir / "sparse/images.bin"] if p.is_file()), None)
            if sp is None:
                print(f"[skip] {name}: no frames.txt or COLMAP images.bin")
                continue
            poses = read_colmap_images_bin(sp)
            imdir = next(d for d in [sdir / "images_4", sdir / "images_2", sdir / "images", sdir / "images_8"] if d.is_dir())
            names = sorted(n for n in poses if (imdir / n).is_file())
            if isinstance(sel, int):
                sel = uniform(len(names), sel)
            pick = [names[i] for i in sel]
            scenes.append(Scene(name, [str(imdir / n) for n in pick],
                                np.stack([poses[n] for n in pick]), None))
        if max_scenes and len(scenes) >= max_scenes:
            break
    return scenes


def vggt_crop_box(path: str) -> tuple[float, float, float, float]:
    """Fractional crop applied by vggt load_and_preprocess_images(mode='crop')."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    nh = round(h * (518 / w) / 14) * 14
    if nh <= 518:
        return 0.0, 0.0, 1.0, 1.0
    top = (nh - 518) // 2
    return 0.0, top / nh, 1.0, (top + 518) / nh


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def build(args):
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    info = {}
    if args.method == "gpa":
        from vggt.models.vggt import VGGT
        cfg = dict(merging=args.gpa_merging, merge_end=23, merge_ratio=args.gpa_merge_ratio,
                   early_frame_split=args.gpa_early_frame_split, broadcast_pool=8,
                   core_blocks=(13, 14, 15, 16), core_merge_ratio=args.gpa_core_merge_ratio,
                   merge_metric_dim=128)
        model = VGGT(enable_track=False, enable_point=False, **cfg)
        info["config"] = cfg
    else:
        from vggt.models.vggt import VGGT
        model = VGGT(enable_track=False, enable_point=False)
    keys = set(model.state_dict())
    missing, unexpected = model.load_state_dict({k: v for k, v in sd.items() if k in keys}, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    del sd
    if args.method == "sparse":
        from sparse_vggt.models.vggt import sparse_aggregator_from_vggt
        model.aggregator, store = sparse_aggregator_from_vggt(
            model.aggregator, sparse_ratio=args.sparse_ratio, cdf_threshold=args.cdf_threshold,
            aux_output=True, aux_sparsity_only=True)
        info["config"] = dict(sparse_ratio=args.sparse_ratio, cdf_threshold=args.cdf_threshold,
                              pool_mode="avg", use_hilbert=False)
        info["aux"] = store
    model = model.eval().cuda()
    if args.precision == "fp16":  # GPA's eval scripts: fp16 weights + fp16 autocast
        model = model.half()
    return model, info


def make_forward(model, args):
    amp = torch.float16 if args.precision == "fp16" else torch.bfloat16

    def fwd(images):
        if args.method == "gpa":
            _, _, H, W = images.shape
            model.update_patch_dimensions(W // 14, H // 14)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            return model(images)
    return fwd


def time_forward(fwd, images, warmup=2, k=5):
    for _ in range(warmup):
        fwd(images)
    torch.cuda.synchronize()
    ts = []
    for _ in range(k):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fwd(images)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


# ----------------------------------------------------------------------------
def main_omega(args) -> int:
    """Official VGGT-Omega predictor; same record schema as the other methods."""
    metrics = omega_module("metrics")
    model_mod = omega_module("model")
    ckpt = args.checkpoint
    scenes = load_scenes(Path(args.data_root), args.frames, args.max_scenes)
    if NO_DEPTH:
        for sc in scenes:
            sc.depth = None
    util_before = gpu_util()
    pred = model_mod.Predictor(ckpt)
    torch.cuda.reset_peak_memory_stats()
    rots, trans, per_scene, lat_all = [], [], {}, []
    for sc in scenes:
        t0 = time.time()
        depth, ext, crop = pred.predict(sc.image_paths)
        row = {"n_frames": len(sc.image_paths), "first_call_s": round(time.time() - t0, 3)}
        r, t = metrics.pose_errors(ext, sc.extrinsics)
        rots.append(r)
        trans.append(t)
        row.update(metrics.pose_metrics(r, t))
        if sc.depth is not None:
            tgt = metrics.crop_depth(sc.depth, crop)
            row.update(metrics.depth_metrics(metrics.resize_depth(depth, tgt.shape[-2:]), tgt))
        if not args.no_timing:
            images = model_mod.load_and_preprocess_images(
                sc.image_paths, mode="max_size", image_resolution=512, patch_size=16).cuda()

            def fwd(x):
                with torch.inference_mode():
                    return pred.model(x)
            ts = time_forward(fwd, images)
            row["latency_ms"] = float(np.median(ts))
            lat_all.append(ts)
            del images
        per_scene[sc.name] = row
        print(sc.name, json.dumps(row))
        torch.cuda.empty_cache()
    peak = int(torch.cuda.max_memory_allocated())
    r, t = np.concatenate(rots), np.concatenate(trans)
    m = metrics.pose_metrics(r, t)
    depth_rows = [v for v in per_scene.values() if "delta125" in v]
    if depth_rows:
        m["delta125"] = float(np.mean([v["delta125"] for v in depth_rows])) * 100.0
        m["AbsRel"] = float(np.mean([v["AbsRel"] for v in depth_rows]))
    m["pose_pairs"] = int(r.size)
    med = [float(np.median(x)) for x in lat_all]
    lat = {"median": float(np.median(med)) if med else None, "k": 5 if med else 0, "warmup": 2,
           "per_scene_median": dict(zip(per_scene, med))}
    rec = {"exp": args.exp, "impl": "official-pytorch", "method": "omega", "config": "pt-dense",
           "schedule": "dense", "precision": "official eval/model.py (fp32 weights)",
           "dataset": args.dataset or "/".join(Path(args.data_root).parts[-2:]),
           "n_frames": sorted({v["n_frames"] for v in per_scene.values()}),
           "scenes": list(per_scene), "metrics": m, "per_scene": per_scene, "latency_ms": lat,
           "peak_mem_bytes": peak,
           "gpu": {"lock_held": bool(args.lock_held), "util_before": util_before, "util_after": gpu_util(),
                   "name": torch.cuda.get_device_name(0)},
           "checkpoint": Path(ckpt).name, "torch": torch.__version__,
           "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2) + "\n")
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({"metrics": m, "latency_ms": lat["median"], "peak_GiB": peak / 2**30}))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("gpa", "sparse", "dense", "omega"), required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--frames", required=True, help="frames JSON {scene: [idx]} or uniform:N")
    p.add_argument("--max-scenes", type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="dense/gpa/sparse: facebook/VGGT-1B model.pt; omega: vggt_omega_1b_512.pt")
    p.add_argument("--precision", choices=("fp16", "bf16"), default=None,
                   help="fp16 = fp16 weights + fp16 autocast (GPA eval scripts); "
                        "bf16 = fp32 weights + bf16 autocast (sparse-vggt README). "
                        "Default: fp16 for gpa, bf16 otherwise")
    p.add_argument("--exp", default="official-parity")
    p.add_argument("--dataset", default=None, help="label; default = data-root path tail")
    p.add_argument("--no-timing", action="store_true")
    p.add_argument("--lock-held", type=int, default=int(os.environ.get("PO_LOCK_HELD", "0")))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-depth", action="store_true", help="score poses only (no depth maps needed)")
    # GPA (defaults = repo defaults at the pinned commit)
    p.add_argument("--gpa-merging", type=int, default=9)
    p.add_argument("--gpa-merge-ratio", type=float, default=0.98)
    p.add_argument("--gpa-core-merge-ratio", type=float, default=0.2)
    p.add_argument("--gpa-early-frame-split", type=int, default=9)
    # sparse-vggt (defaults = README example config)
    p.add_argument("--sparse-ratio", type=float, default=0.1)
    p.add_argument("--cdf-threshold", type=float, default=0.97)
    args = p.parse_args()
    global NO_DEPTH
    NO_DEPTH = args.no_depth
    if args.precision is None:
        args.precision = "fp16" if args.method == "gpa" else "bf16"
    if args.cdf_threshold is not None and args.cdf_threshold < 0:
        args.cdf_threshold = None

    if args.method == "omega":
        return main_omega(args)
    setup_paths(args.method)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    metrics = omega_module("metrics")  # official eval/metrics.py
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    root = Path(args.data_root)
    scenes = load_scenes(root, args.frames, args.max_scenes)
    if not scenes:
        raise SystemExit("no scenes")
    util_before = gpu_util()
    model, info = build(args)
    fwd = make_forward(model, args)
    torch.cuda.reset_peak_memory_stats()

    rots, trans, per_scene, lat_all = [], [], {}, []
    for sc in scenes:
        images = load_and_preprocess_images(sc.image_paths).cuda()
        if args.precision == "fp16":
            images = images.half()
        t0 = time.time()
        pred = fwd(images)
        torch.cuda.synchronize()
        row = {"n_frames": len(sc.image_paths), "first_call_s": round(time.time() - t0, 3)}
        ext, _ = pose_encoding_to_extri_intri(pred["pose_enc"].float(), images.shape[-2:])
        ext = metrics.pad_extrinsics(ext[0].double().cpu().numpy())
        r, t = metrics.pose_errors(ext, sc.extrinsics)
        rots.append(r)
        trans.append(t)
        row.update(metrics.pose_metrics(r, t))
        if sc.depth is not None:
            pd = pred["depth"].float().cpu().numpy()[0][..., 0]
            tgt = metrics.crop_depth(sc.depth, vggt_crop_box(sc.image_paths[0]))
            pd = metrics.resize_depth(pd, tgt.shape[-2:])
            row.update(metrics.depth_metrics(pd, tgt))
        if "aux" in info:
            sp = [float(v["sparsity"]) for v in info["aux"].values() if v and "sparsity" in v]
            row["effective_sparsity"] = float(np.mean(sp)) if sp else None
        del pred
        if not args.no_timing:
            ts = time_forward(fwd, images)
            row["latency_ms"] = float(np.median(ts))
            lat_all.append(ts)
        per_scene[sc.name] = row
        print(sc.name, json.dumps(row))
        del images
        torch.cuda.empty_cache()
    peak = int(torch.cuda.max_memory_allocated())
    util_after = gpu_util()

    r, t = np.concatenate(rots), np.concatenate(trans)
    m = metrics.pose_metrics(r, t)
    depth_rows = [v for v in per_scene.values() if "delta125" in v]
    if depth_rows:  # evaluate.py: mean over scenes, delta in percent
        m["delta125"] = float(np.mean([v["delta125"] for v in depth_rows])) * 100.0
        m["AbsRel"] = float(np.mean([v["AbsRel"] for v in depth_rows]))
    if "aux" in info:
        m["effective_sparsity"] = float(np.mean([v["effective_sparsity"] for v in per_scene.values()]))
    m["pose_pairs"] = int(r.size)

    if lat_all:
        med = [float(np.median(x)) for x in lat_all]  # per-scene medians
        flat = np.concatenate(lat_all)
        q1, q3 = np.percentile(flat, [25, 75])
        lat = {"median": float(np.median(med)), "iqr": float(q3 - q1), "k": 5,
               "warmup": 2, "per_scene_median": dict(zip(per_scene, med))}
    else:
        lat = {"median": None, "iqr": None, "k": 0}

    n_frames = sorted({v["n_frames"] for v in per_scene.values()})
    repo = SRC / ("gpa" if args.method == "gpa" else "sparse")
    if args.method == "sparse":
        schedule = f"sparse-vggt sparse_ratio={args.sparse_ratio} cdf={args.cdf_threshold}"
    elif args.method == "gpa":
        schedule = "gpa " + json.dumps(info["config"], sort_keys=True)
    else:
        schedule = "dense"
    rec = {
        "exp": args.exp,
        "impl": "official-pytorch",
        "method": args.method,
        "schedule": schedule,
        "precision": {"fp16": "fp16-weights+fp16-autocast", "bf16": "fp32-weights+bf16-autocast"}[args.precision],
        "n_frames": n_frames[0] if len(n_frames) == 1 else n_frames,
        "dataset": args.dataset or "/".join(root.parts[-2:]),
        "scenes": list(per_scene),
        "metrics": m,
        "per_scene": per_scene,
        "latency_ms": lat,
        "peak_mem_bytes": peak,
        "gpu": {"lock_held": bool(args.lock_held), "util_before": util_before,
                "util_after": util_after, "name": torch.cuda.get_device_name(0)},
        "commit": {"repo": git_commit(repo),
                   "vggt_submodule": git_commit(SRC / "sparse" / "external" / "vggt") if args.method != "gpa" else None,
                   "spargeattn": git_commit(SRC / "sparse" / "external" / "SpargeAttn") if args.method == "sparse" else None},
        "checkpoint": Path(args.checkpoint).name,
        "torch": torch.__version__,
        "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2) + "\n")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({"metrics": m, "latency_ms": lat["median"], "peak_GiB": peak / 2**30}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
