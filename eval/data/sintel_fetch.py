#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Range-read the Sintel members the official VGGT-Omega prepare.py needs.

Official `eval/prepare.py sintel --source local --raw-root RAW` reads
  RAW/training/clean/<scene>/frame_*.png       (glob = frame index space;
                                               only sampled ones are decoded)
  RAW/training/camdata_left/<scene>/*.cam      (all frames)
  RAW/training/depth/<scene>/*.dpt             (sampled frames)
Instead of downloading the 5.6 GB + 1.6 GB official zips, this pulls only those
members over HTTP range requests from the official URLs. Unsampled frames get
a zero-byte placeholder PNG (prepare.py only globs them; never decodes them).

Usage: sintel_fetch.py --frames sintel.json --raw RAW
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from remote_zip import open_remote

IMAGES = "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"
DEPTH = "http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip"


def extract(zf, info, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--raw", required=True)
    a = p.parse_args()
    frames = json.loads(Path(a.frames).read_text())
    raw = Path(a.raw)
    zi, ri = open_remote(IMAGES)
    zd, rd = open_remote(DEPTH)
    for scene, sel in frames.items():
        sel = set(sel)
        imgs = sorted(
            (i for i in zi.infolist() if i.filename.startswith(f"training/clean/{scene}/frame_") and i.filename.endswith(".png")),
            key=lambda i: i.filename,
        )
        for idx, info in enumerate(imgs):
            dest = raw / info.filename
            if idx in sel:
                extract(zi, info, dest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.touch()
        for info in zd.infolist():
            n = info.filename
            if n.startswith(f"training/camdata_left/{scene}/") and n.endswith(".cam"):
                extract(zd, info, raw / n)
            elif n.startswith(f"training/depth/{scene}/") and n.endswith(".dpt"):
                if int(Path(n).stem.split("_")[1]) - 1 in sel:
                    extract(zd, info, raw / n)
        print(f"{scene}: {len(imgs)} frames, {len(sel)} sampled", flush=True)
    print(f"range-fetched {ri.fetched + rd.fetched} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
