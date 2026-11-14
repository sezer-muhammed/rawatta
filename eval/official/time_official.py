#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Accuracy and latency of the Official VGGT-Omega PyTorch code on the same frames as
eval/run_eval.py (the "Official VGGT-Ω" baseline column).

Runs inside the image built from eval/official/Dockerfile (PyTorch 2.9.1 + CUDA 12.8 +
the official repository at the pinned commit; see run_official.sh). The model is the
official `eval/model.py` Predictor, unchanged: fp32 weights, bf16 autocast inside the
model, the evaluator's own preprocessing (max side 512, patch 16) and crop.

    time_official.py --checkpoint /models/vggt_omega_1b_512.pt --data-root /data/eth3d \
        --frames <map.json | uniform:N> --out /out/official-eth3d.json [--no-timing] [--no-depth]

--frames: a frame map {scene: [indices]} (run_eval.py writes the one it used to
<run>/<dataset>-s<seed>/fmap.json), or uniform:N = N evenly spaced frames of every scene
(works for COLMAP scenes such as Mip-NeRF 360 kitchen, the paper's timing scene).

Metrics: official eval/metrics.py `pose_errors`/`pose_metrics` pooled over all pairs of all
scenes (AUC@3/@30) and `depth_metrics` after the same crop/resize as evaluate.py.
Latency: CUDA events around the model forward on pre-loaded images, --warmup warm-ups,
median of --reps (default 2 and 5); peak memory = torch.cuda.max_memory_allocated after
the model load. Timings are only valid on an otherwise idle GPU (hold the GPU lock).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import omega_src  # noqa: E402


def gpu_util() -> int | None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                      text=True, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def _qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap_images_bin(path: Path) -> dict[str, np.ndarray]:
    """name -> 4x4 world-to-camera (OpenCV) from COLMAP images.bin."""
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


def load_scenes(root: Path, frames_arg: str, max_scenes: int | None, no_depth: bool) -> list[Scene]:
    """Omega layout (frames.txt, cameras.npz, images/, depths/) or COLMAP (sparse/0/images.bin + images*/)."""
    ds_mod = omega_src.eval_module("datasets")
    if no_depth:  # pose only: depth maps need not be on disk
        ds_mod._read_depth = lambda _p: np.zeros((4, 4), np.float32)
    if frames_arg.startswith("uniform:"):
        n = int(frames_arg.split(":", 1)[1])
        frame_map = {d.name: n for d in sorted(root.iterdir()) if d.is_dir()}
    else:
        frame_map = json.loads(Path(frames_arg).read_text())
    scenes = []
    for name, sel in frame_map.items():
        sdir = root / name
        if not sdir.is_dir():
            print(f"[skip] {name}: not under {root}")
            continue
        if (sdir / "frames.txt").is_file():
            names = [ln.strip() for ln in (sdir / "frames.txt").read_text().splitlines() if ln.strip()]
            if isinstance(sel, int):
                avail = [i for i, nm in enumerate(names) if (sdir / "images" / nm).is_file()]
                sel = [avail[i] for i in uniform(len(avail), sel)]
            if not (sdir / "depths").is_dir() or no_depth:
                with np.load(sdir / "cameras.npz") as c:
                    ext = c["extrinsics"][sel]
                scenes.append(Scene(name, [str(sdir / "images" / names[i]) for i in sel], ext, None))
            else:
                b = ds_mod.Dataset(str(root)).load(name, sel)
                scenes.append(Scene(name, b.image_paths, b.extrinsics, b.depth))
        else:
            sp = next((p for p in (sdir / "sparse/0/images.bin", sdir / "sparse/images.bin") if p.is_file()), None)
            if sp is None:
                print(f"[skip] {name}: no frames.txt or COLMAP images.bin")
                continue
            poses = read_colmap_images_bin(sp)
            imdir = next(d for d in (sdir / "images_4", sdir / "images_2", sdir / "images", sdir / "images_8")
                         if d.is_dir())
            names = sorted(n for n in poses if (imdir / n).is_file())
            if isinstance(sel, int):
                sel = uniform(len(names), sel)
            pick = [names[i] for i in sel]
            scenes.append(Scene(name, [str(imdir / n) for n in pick], np.stack([poses[n] for n in pick]), None))
        if max_scenes and len(scenes) >= max_scenes:
            break
    return scenes


def time_forward(fwd, images, warmup=2, k=5):
    import torch  # noqa: PLC0415

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=os.environ.get("VGGT_OMEGA_CKPT", "/models/vggt_omega_1b_512.pt"))
    p.add_argument("--data-root", required=True)
    p.add_argument("--frames", required=True, help="frame map JSON {scene: [idx]} or uniform:N")
    p.add_argument("--out", required=True)
    p.add_argument("--max-scenes", type=int)
    p.add_argument("--no-timing", action="store_true")
    p.add_argument("--no-depth", action="store_true", help="score poses only")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--dataset", default=None, help="label for the record (default: data-root name)")
    p.add_argument("--lock-held", action="store_true", help="record that the caller holds the GPU lock")
    a = p.parse_args()

    import torch  # noqa: PLC0415

    metrics = omega_src.eval_module("metrics")
    model_mod = omega_src.eval_module("model")
    scenes = load_scenes(Path(a.data_root), a.frames, a.max_scenes, a.no_depth)
    if not scenes:
        raise SystemExit("no scenes")
    util_before = gpu_util()
    pred = model_mod.Predictor(a.checkpoint)
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
        if not a.no_timing:
            images = model_mod.load_and_preprocess_images(sc.image_paths, mode="max_size", image_resolution=512,
                                                          patch_size=16).cuda()

            def fwd(x):
                with torch.inference_mode():
                    return pred.model(x)

            ts = time_forward(fwd, images, a.warmup, a.reps)
            row["latency_ms"] = float(np.median(ts))
            row["times_ms"] = ts
            lat_all.append(ts)
            del images
        per_scene[sc.name] = row
        print(sc.name, json.dumps({k: v for k, v in row.items() if k != "times_ms"}), flush=True)
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
    rec = {
        "impl": "official-pytorch", "method": "Official VGGT-Omega", "config": "official",
        "precision": "official eval/model.py (fp32 weights, bf16 autocast in the model)",
        "dataset": a.dataset or Path(a.data_root).name, "frames": a.frames,
        "n_frames": sorted({v["n_frames"] for v in per_scene.values()}),
        "scenes": list(per_scene), "metrics": m, "per_scene": per_scene,
        "latency_ms": {"median": float(np.median(med)) if med else None, "k": a.reps if med else 0,
                       "warmup": a.warmup, "per_scene_median": dict(zip(per_scene, med)),
                       "clock": "CUDA events, model forward only"},
        "peak_mem_bytes": peak,
        "gpu": {"name": torch.cuda.get_device_name(0), "lock_held": a.lock_held, "util_before": util_before,
                "util_after": gpu_util()},
        "official_commit": omega_src.OMEGA_COMMIT, "checkpoint": Path(a.checkpoint).name,
        "torch": torch.__version__, "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({"metrics": m, "latency_ms": rec["latency_ms"]["median"], "peak_GiB": peak / 2 ** 30}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
