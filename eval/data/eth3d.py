#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ETH3D (13 scenes, VGGT-Omega's official 10-frame map) -> official eval layout.

Port of the paper's official-evaluator data script for ETH3D, without containers. Scene by scene:
  1. eth3d_fetch.py mirrors only the files the official frame map needs
     (calibration text files + sampled undistorted JPEGs + their raw depth)
     from the Hugging Face mirror pinned by the official prepare.py,
  2. the official `eval/prepare.py eth3d --source local` converts them
     (pinned source, fetched at run time; not copied),
  3. shrink_depth.py stores the depth maps pre-resized (bit-identical through
     the official loader),
  4. the raw files are deleted.
Needs numpy + opencv. ~1.8 GB final, a few GB of raw files per scene at a time.

  python3 eval/data/eth3d.py --out $DATA/eth3d [--tmp $DATA/tmp] [--scenes a,b]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import omega_src  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", default=None, help="raw download area (default: <out>/../tmp-eth3d)")
    p.add_argument("--scenes", default=None, help="comma-separated subset (default: all 13)")
    a = p.parse_args()
    out = Path(a.out).resolve()
    tmp = Path(a.tmp).resolve() if a.tmp else out.parent / "tmp-eth3d"
    src = omega_src.ensure()
    frames = src / "eval" / "frames" / "eth3d.json"
    scenes = a.scenes.split(",") if a.scenes else list(json.loads(frames.read_text()))
    out.mkdir(parents=True, exist_ok=True)
    for s in scenes:
        if (out / s / "cameras.npz").is_file():
            print(f"{s}: present")
            continue
        raw = tmp / "eth3d_raw"
        subprocess.run([sys.executable, str(HERE / "eth3d_fetch.py"), "--frames", str(frames), "--raw", str(raw),
                        "--scene", s], check=True)
        subprocess.run([sys.executable, str(src / "eval" / "prepare.py"), "eth3d", "--output", str(out),
                        "--source", "local", "--raw-root", str(raw), "--scenes", s], check=True, cwd=src)
        if (out / "source.json").is_file():
            (out / "source.json").replace(out / f"source.{s}.json")
        subprocess.run([sys.executable, str(HERE / "shrink_depth.py"), str(out), s], check=True)
        shutil.rmtree(raw / s, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
