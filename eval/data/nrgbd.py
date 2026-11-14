#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Neural RGB-D (Azinovic et al., CVPR 2022) synthetic scenes -> Omega eval layout.

Source: http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip
(project page https://dazinovic.github.io/neural-rgbd-surface-reconstruction/).
Only the needed members are read with HTTP range requests (no full download).

Per scene in the zip: images/img<i>.png (RGB 640x480), depth/depth<i>.png
(noise-free rendered depth, uint16 millimetres), depth_with_noise/,
depth_filtered/, focal.txt (one float, pixels), poses.txt (4x4 per frame,
stacked; camera-to-world, OpenGL camera axes x right / y up / z back, as
read by the authors' dataloader and by Spann3R/CUT3R's NRGBD eval).

Conventions written here:
  * depth = depth/depth<i>.png / 1000 -> metres; 0 stays invalid
    (thin_geometry also ships depth_gt/; we use depth/ for every scene, as the
    Spann3R/CUT3R evaluations do).
  * intrinsics fx = fy = focal.txt, cx = W/2, cy = H/2 (same as those evals).
  * extrinsics: c2w_cv = c2w_gl @ diag(1,-1,-1,1); w2c = inv(c2w_cv).
  * frames with a non-finite pose are kept in frames.txt / cameras.npz (NaN)
    but excluded from sampling.
Frame names "%06d.png" in numeric index order of img<i>.png.

Needs numpy + opencv (eval/requirements.txt):
  python3 eval/data/nrgbd.py --out $DATA/nrgbd --maps $DATA/_frames
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GL_TO_CV, c2w_to_w2c, frame_maps, union, write_depth, write_maps, write_scene  # noqa: E402
from remote_zip import open_remote  # noqa: E402

URL = "http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--maps", required=True)
    a = p.parse_args()
    out = Path(a.out)
    zf, raw = open_remote(URL)
    members = zf.namelist()
    scenes = sorted({m.split("/")[0] for m in members if m.endswith("/poses.txt")})
    counts, valid, allmaps = {}, {}, {}
    for scene in scenes:
        imgs = {}
        for m in members:
            r = re.fullmatch(rf"{scene}/images/img(\d+)\.png", m)
            if r:
                imgs[int(r.group(1))] = m
        idx = sorted(imgs)
        assert idx == list(range(len(idx))), scene
        n = len(idx)
        poses = np.loadtxt(zf.read(f"{scene}/poses.txt").decode().splitlines()).reshape(-1, 4, 4)
        assert len(poses) == n, (scene, len(poses), n)
        focal = float(zf.read(f"{scene}/focal.txt").decode().split()[0])
        ok = [i for i in range(n) if np.isfinite(poses[i]).all()]
        counts[scene], valid[scene] = n, ok
        maps = frame_maps({scene: n}, {scene: ok})
        need = union(maps, scene)
        h = w = None
        for i in sorted(need):
            data = zf.read(imgs[i])
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            (out / scene / "images").mkdir(parents=True, exist_ok=True)
            (out / scene / "images" / f"{i:06d}.png").write_bytes(data)
            d = cv2.imdecode(np.frombuffer(zf.read(f"{scene}/depth/depth{i}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
            assert d.dtype == np.uint16 and d.shape == (h, w), (scene, i, d.dtype, d.shape)
            write_depth(out / scene / "depths" / f"{i:06d}.npz", d.astype(np.float32) / 1000.0)
        K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
        ex = [c2w_to_w2c(P @ GL_TO_CV) if np.isfinite(P).all() else np.full((3, 4), np.nan) for P in poses]
        write_scene(out / scene, [f"{i:06d}.png" for i in range(n)], ex, [K] * n)
        print(f"{scene}: {n} frames ({n - len(ok)} invalid poses), stored {len(need)}, {w}x{h} f={focal}", flush=True)
    write_maps(frame_maps(counts, valid), Path(a.maps), "nrgbd")
    print(f"range-fetched {raw.fetched} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
