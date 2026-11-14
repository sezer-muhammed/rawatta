#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch only the ETH3D files the official VGGT-Omega frame map needs.

The official `eval/prepare.py eth3d` (run unchanged by eth3d.py with
`--source local --raw-root <raw>`) only reads the COLMAP calibration text
files plus the image / raw depth of the *sampled* frames. Downloading the whole
HF snapshot (all images + ~97 MB raw depth per frame) does not fit the disk, so
this script mirrors the needed subset of the pinned HF mirror into <raw>,
in the directory layout prepare.py expects:

  <raw>/<scene>/dslr_calibration_jpg/{cameras,images}.txt
  <raw>/<scene>/dslr_calibration_undistorted/{cameras,images}.txt
  <raw>/<scene>/images/dslr_images_undistorted/<stem>.JPG
  <raw>/<scene>/ground_truth_depth/dslr_images/<stem>.JPG   (raw float32)

Frame indexing (re-derived, not copied, from prepare.py's documented logic):
undistorted images sorted by stem, keeping those with a matching
`dslr_images/<stem>.JPG` in the distorted calibration. prepare.py itself then
fails loudly if a sampled file is missing, which doubles as a check.

Usage: eth3d_fetch.py --frames eth3d.json --raw RAW --scene SCENE
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO = "Livioni/eth3d_omnivggt"
REVISION = "64154bdca815ef7712f5f2621e231bac7dc99d59"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"


def fetch(rel: str, dest: Path) -> None:
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{BASE}/{rel}", headers={"User-Agent": "rawatta-eval-data"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.rename(dest)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


def image_names(path: Path) -> list[str]:
    names, lines = [], [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[0::2]:
        names.append(line.split()[9])
    return names


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--scene", required=True)
    a = p.parse_args()
    scene = a.scene
    root = Path(a.raw) / scene
    for sub in ("dslr_calibration_jpg", "dslr_calibration_undistorted"):
        for f in ("cameras.txt", "images.txt"):
            fetch(f"{scene}/{sub}/{f}", root / sub / f)
    distorted = set(image_names(root / "dslr_calibration_jpg" / "images.txt"))
    stems = sorted(Path(n).stem for n in image_names(root / "dslr_calibration_undistorted" / "images.txt"))
    stems = [s for s in stems if f"dslr_images/{s}.JPG" in distorted]
    selected = json.loads(Path(a.frames).read_text())[scene]
    for i in selected:
        s = stems[i]
        fetch(f"{scene}/images/dslr_images_undistorted/{s}.JPG", root / "images/dslr_images_undistorted" / f"{s}.JPG")
        fetch(f"{scene}/ground_truth_depth/dslr_images/{s}.JPG", root / "ground_truth_depth/dslr_images" / f"{s}.JPG")
    print(f"{scene}: {len(stems)} frames, fetched {len(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
