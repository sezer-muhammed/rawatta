#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microsoft 7-Scenes test sequences -> official VGGT-Omega eval layout.

Source: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
  per-scene zip  http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/<scene>.zip
  containing TestSplit.txt and deflated seq-XX.zip members.
Each outer zip is read with HTTP range requests (remote_zip.py); one test
seq-XX.zip member at a time is streamed to TMP (~0.5 GB), the needed frames
are extracted, and the temp file is deleted immediately.

Conventions (as in the Spann3R / CUT3R / pi3 7-Scenes evaluations):
  * RGB 640x480; depth = published Kinect depth PNG (uint16 mm, NOT registered
    to RGB), 65535 -> 0 (invalid), mm -> m. Stored at 640x480.
  * intrinsics fx = fy = 585, cx = 320, cy = 240 (dataset page states the
    depth camera focal ~585; Spann3R datasets/sevenscenes.py uses this K).
  * frame-XXXXXX.pose.txt = 4x4 camera-to-world -> inverted to w2c (OpenCV
    axes already: Kinect camera x right, y down, z forward).
Scene id = "<scene>_seq-XX"; frame names "%06d.png".

Needs numpy + opencv (eval/requirements.txt):
  python3 eval/data/sevenscenes.py --out $DATA/7scenes --tmp $DATA/tmp --maps $DATA/_frames
Calibration (train) split, disjoint from the test sequences: the first
TrainSplit.txt sequence of each scene:
  python3 eval/data/sevenscenes.py --split train --seeded50 --name 7scenes_train --out $DATA/7scenes_train --tmp ... --maps ...
--seeded50 also writes <name>_n50_s1/_s2 (50 random frames with a finite pose, seeds 1 and 2:
frame samples 1 and 2 of calibration/calibrate.py) and stores their frames.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2w_to_w2c, frame_maps, seeded_maps, union, write_depth, write_maps, write_scene  # noqa: E402
from remote_zip import open_remote  # noqa: E402

URL = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/{}.zip"
SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
K = np.array([[585.0, 0, 320.0], [0, 585.0, 240.0], [0, 0, 1]])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--split", choices=("test", "train"), default="test")
    p.add_argument("--name", default="7scenes", help="dataset name of the frame maps")
    p.add_argument("--seeded50", action="store_true",
                   help="also write the 50-frame seeded maps <name>_n50_s1/_s2 (calibration samples 1 and 2)")
    a = p.parse_args()
    out, tmp = Path(a.out), Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    counts, valid, pending = {}, {}, []
    for scene in SCENES:
        zf, raw = open_remote(URL.format(scene))
        split = zf.read(f"{scene}/{a.split.capitalize()}Split.txt").decode().split()
        if a.split == "train":
            split = split[:1]
        for s in split:
            seq = f"seq-{int(s.replace('sequence', '')):02d}"
            pending.append((scene, seq, zf))
    for scene, seq, zf in pending:
        sid = f"{scene}_{seq}"
        local = tmp / f"{sid}.zip"
        with zf.open(f"{scene}/{seq}.zip") as src, local.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
        with zipfile.ZipFile(local) as inner:
            names = sorted(n for n in inner.namelist() if n.endswith(".color.png"))
            n = len(names)
            counts[sid] = n
            ex = [c2w_to_w2c(np.loadtxt(inner.open(name[: -len(".color.png")] + ".pose.txt")).reshape(4, 4))
                  for name in names]
            frames = [f"{i:06d}.png" for i in range(n)]
            maps = frame_maps({sid: n})
            if a.seeded50:
                valid[sid] = [i for i, e in enumerate(ex) if np.isfinite(e).all()]
                maps.update(seeded_maps({sid: valid[sid]}))
            need = union(maps, sid)
            for i, name in enumerate(names):
                if i not in need:
                    continue
                stem = name[: -len(".color.png")]
                img_dir = out / sid / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                (img_dir / frames[i]).write_bytes(inner.read(name))
                d = cv2.imdecode(np.frombuffer(inner.read(stem + ".depth.png"), np.uint8), cv2.IMREAD_UNCHANGED)
                assert d.dtype == np.uint16 and d.shape == (480, 640), (name, d.dtype, d.shape)
                depth = d.astype(np.float32) / 1000.0
                depth[d == 65535] = 0
                write_depth(out / sid / "depths" / f"{i:06d}.npz", depth)
            write_scene(out / sid, frames, ex, [K] * n)
        local.unlink()
        print(f"{sid}: {n} frames, stored {len(need)}", flush=True)
    write_maps({**frame_maps(counts), **(seeded_maps(valid) if a.seeded50 else {})}, Path(a.maps), a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
