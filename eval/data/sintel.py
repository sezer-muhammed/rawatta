#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MPI Sintel (VGGT-Omega's official 10-frame map) -> official eval layout.

Port of the paper's official-evaluator data script for Sintel, without containers:
  1. sintel_fetch.py range-reads only the needed members of the two official
     Sintel zips (images + camera/depth training data) from the official URLs,
  2. the official `eval/prepare.py sintel --source local` converts them
     (pinned source, fetched at run time; not copied),
  3. shrink_depth.py stores the depth maps pre-resized,
  4. the raw files are deleted.
Needs numpy + opencv. Sintel is under a non-commercial research licence that
forbids redistribution (eval/DATASETS.md); keep the prepared data local.

  python3 eval/data/sintel.py --out $DATA/sintel [--tmp $DATA/tmp]
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--tmp", default=None, help="raw download area (default: <out>/../tmp-sintel)")
    a = p.parse_args()
    out = Path(a.out).resolve()
    tmp = Path(a.tmp).resolve() if a.tmp else out.parent / "tmp-sintel"
    src = omega_src.ensure()
    raw = tmp / "sintel_raw"
    subprocess.run([sys.executable, str(HERE / "sintel_fetch.py"), "--frames", str(src / "eval/frames/sintel.json"),
                    "--raw", str(raw)], check=True)
    subprocess.run([sys.executable, str(src / "eval" / "prepare.py"), "sintel", "--output", str(out),
                    "--source", "local", "--raw-root", str(raw)], check=True, cwd=src)
    subprocess.run([sys.executable, str(HERE / "shrink_depth.py"), str(out)], check=True)
    shutil.rmtree(raw, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
