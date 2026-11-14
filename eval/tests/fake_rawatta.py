#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for `rawatta bench --job <job.json>` in tests (no GPU, no weights).

Writes the bench output layout (<out>/<config>/<sequence>/{extrinsics,intrinsics,depth,conf}.npy,
cell.json) from the dataset's own ground truth: configs whose name contains "exact" get the GT
poses (perfect; also "dense@bf16"), other configs get the GT rotated by a per-frame random angle of up to
--noise degrees (default 2). Depth = GT depth (omega layout with depths/) at 1/2 resolution.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def rot(axis, deg):
    a = np.deg2rad(deg)
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def perfect(cfg):
    """The exact engine (by name or grammar) returns the ground truth."""
    return "exact" in cfg or cfg == "dense@bf16"


def gt_for(images):
    sd = Path(images[0]).parent.parent
    if (sd / "frames.txt").is_file():
        names = (sd / "frames.txt").read_text().split()
        with np.load(sd / "cameras.npz") as c:
            ex, K = c["extrinsics"], c["intrinsics"]
        idx = [names.index(Path(p).name) for p in images]
        depth = None
        if (sd / "depths").is_dir():
            depth = np.stack([np.load(sd / "depths" / (Path(names[i]).stem + ".npz"))["depth"] for i in idx])
        return ex[idx], K[idx], depth
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import score  # noqa: PLC0415

    gt = score.read_colmap_images(sd / "sparse/0/images.bin")
    ex = np.stack([gt[Path(p).name] for p in images])
    return ex, np.tile(np.eye(3), (len(images), 1, 1)), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["bench"])
    p.add_argument("--job", required=True)
    p.add_argument("--noise", type=float, default=2.0)
    a = p.parse_args()
    job = json.loads(Path(a.job).read_text())
    rng = np.random.default_rng(0)
    for cfg in job["configs"]:
        for s in job["sequences"]:
            d = Path(job["out"]) / cfg.replace("/", "%") / s["name"]
            d.mkdir(parents=True, exist_ok=True)
            ex, K, depth = gt_for(s["images"])
            ex = ex.astype(np.float64).copy()
            if not perfect(cfg):
                for i in range(1, len(ex)):
                    ex[i, :3, :3] = rot(rng.standard_normal(3), rng.uniform(0, a.noise)) @ ex[i, :3, :3]
            np.save(d / "extrinsics.npy", ex[:, :3, :4].astype(np.float32))
            np.save(d / "intrinsics.npy", K.astype(np.float32))
            if job.get("depth", True) and depth is not None:
                np.save(d / "depth.npy", depth[:, ::2, ::2].astype(np.float32))
                np.save(d / "conf.npy", np.ones_like(depth[:, ::2, ::2], np.float32))
            timed = job.get("reps", 0) > 0
            (d / "cell.json").write_text(json.dumps({"config": cfg, "sequence": s["name"], "frames": len(ex),
                                                     "timed": timed, "median_ms": 100.0 if perfect(cfg) else 80.0}))
    print("fake bench done")


if __name__ == "__main__":
    main()
