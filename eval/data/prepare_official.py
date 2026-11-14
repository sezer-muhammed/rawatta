#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the official VGGT-Omega `eval/prepare.py` (ETH3D, Sintel) from the pinned source.

The official script is fetched at run time into the cache (eval/omega_src.py), never
copied here. All arguments after `--` go to it unchanged, e.g.

  python3 eval/data/prepare_official.py -- eth3d --output $DATA/eth3d
  python3 eval/data/prepare_official.py -- sintel --output $DATA/sintel --source official

It downloads the full official archives (ETH3D ~ tens of GB). eth3d.py / sintel.py
fetch only the sampled frames and then call this same official script with
`--source local`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import omega_src  # noqa: E402


def run(args: list[str]) -> int:
    src = omega_src.ensure()
    return subprocess.run([sys.executable, str(src / "eval" / "prepare.py"), *args], cwd=src).returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the official prepare.py")
    a = p.parse_args()
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    if not args:
        p.error("no arguments for prepare.py (e.g. -- eth3d --output DIR)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
