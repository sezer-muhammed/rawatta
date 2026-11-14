#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Long 7-Scenes test sequences for the memory-scaling comparison (N up to 500).

Same source, conventions and layout as sevenscenes.py, but for a few chosen
sequences it stores every frame needed by the evenly spaced N-frame maps
(N in --ns, np.linspace over all 1000 frames) and writes one frame map per N:
  <maps>/7scenes-long_N<n>.json   {scene: [indices]}

Needs numpy + opencv (eval/requirements.txt):
  python3 eval/data/sevenscenes_long.py --out $DATA/7scenes-long \
      --tmp $DATA/tmp --maps $DATA/_frames --seqs office:seq-02,chess:seq-03
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2w_to_w2c, sample_even, write_depth, write_scene  # noqa: E402
from remote_zip import open_remote  # noqa: E402
from sevenscenes import K, URL  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--seqs", default="office:seq-02,chess:seq-03")
    p.add_argument("--ns", default="10,50,100,200,500")
    a = p.parse_args()
    out, tmp = Path(a.out), Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    ns = [int(x) for x in a.ns.split(",")]
    maps: dict[int, dict[str, list[int]]] = {n: {} for n in ns}
    for item in a.seqs.split(","):
        scene, seq = item.split(":")
        sid = f"{scene}_{seq}"
        zf, _ = open_remote(URL.format(scene))
        local = tmp / f"{sid}.zip"
        with zf.open(f"{scene}/{seq}.zip") as src, local.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
        with zipfile.ZipFile(local) as inner:
            names = sorted(x for x in inner.namelist() if x.endswith(".color.png"))
            for n in ns:
                maps[n][sid] = sample_even(len(names), n)
            need = set().union(*(m[sid] for m in maps.values()))
            frames, ex = [], []
            for i, name in enumerate(names):
                stem = name[: -len(".color.png")]
                ex.append(c2w_to_w2c(np.loadtxt(inner.open(stem + ".pose.txt")).reshape(4, 4)))
                frames.append(f"{i:06d}.png")
                if i not in need:
                    continue
                img_dir = out / sid / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                (img_dir / frames[-1]).write_bytes(inner.read(name))
                d = cv2.imdecode(np.frombuffer(inner.read(stem + ".depth.png"), np.uint8), cv2.IMREAD_UNCHANGED)
                assert d.dtype == np.uint16 and d.shape == (480, 640), (name, d.dtype, d.shape)
                depth = d.astype(np.float32) / 1000.0
                depth[d == 65535] = 0
                write_depth(out / sid / "depths" / f"{i:06d}.npz", depth)
            write_scene(out / sid, frames, ex, [K] * len(frames))
        local.unlink()
        print(f"{sid}: {len(names)} frames, stored {len(need)}", flush=True)
    Path(a.maps).mkdir(parents=True, exist_ok=True)
    for n, m in maps.items():
        (Path(a.maps) / f"7scenes-long_N{n}.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
