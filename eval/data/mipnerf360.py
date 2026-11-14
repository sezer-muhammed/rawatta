#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mip-NeRF 360 scenes (COLMAP ground truth) for calibration and timing.

Source: the official release http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
(project page https://jonbarron.info/mipnerf360/). Only the members of the chosen
scenes that the harness reads are range-read from the zip (remote_zip.py), not the
whole ~12 GB archive:
  <scene>/sparse/0/{cameras,images,points3D}.bin   COLMAP model = pose ground truth
  <scene>/images_4/*.JPG                            quarter-resolution frames

Layout written (what eval/run_eval.py --layout colmap and calibration/calibrate.py read):
  <out>/<scene>/sparse/0/*.bin, <out>/<scene>/images_4/*.JPG

Default scenes: bicycle, garden, stump, kitchen = the paper's outdoor calibration domain
(kitchen is also the timing scene). Held-out scenes the paper evaluated in an earlier
protocol: bonsai, counter, room, flowers, treehill. Stdlib only.

  python3 eval/data/mipnerf360.py --out $DATA/mipnerf360 [--scenes bicycle,garden,stump,kitchen]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import open_remote  # noqa: E402

URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
CAL = ("bicycle", "garden", "stump", "kitchen")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", default=",".join(CAL))
    p.add_argument("--images", default="images_4", help="image folder to fetch (images_4, images_2, images)")
    p.add_argument("--url", default=URL)
    a = p.parse_args()
    out = Path(a.out)
    zf, raw = open_remote(a.url)
    names = zf.namelist()
    for scene in a.scenes.split(","):
        want = [n for n in names if not n.endswith("/") and (
            n.startswith(f"{scene}/sparse/0/") or n.startswith(f"{scene}/{a.images}/"))]
        if not any(n.endswith("images.bin") for n in want):
            print(f"{scene}: no COLMAP model in the archive", file=sys.stderr)
            return 1
        got = 0
        for n in want:
            dest = out / n
            if dest.is_file() and dest.stat().st_size == zf.getinfo(n).file_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(n) as src, dest.open("wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            got += 1
        print(f"{scene}: {len(want)} files ({got} fetched)", flush=True)
    print(f"range-fetched {raw.fetched} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
