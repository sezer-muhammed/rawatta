#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Store depth maps pre-resized the way the official loader resizes them.

Official eval/datasets.py always applies `_resize_depth` (width 518, height
rounded to a multiple of 14, INTER_NEAREST) after reading depths/<stem>.npz.
Full-resolution ETH3D depth is ~10-40 MB/frame compressed; the resized map
is ~100x smaller. For every depth file this script:

  1. loads it and applies the *official* `_resize_depth` (imported at run time
     from the pinned official source, see eval/omega_src.py; not copied),
  2. writes the result to a temporary npz,
  3. reloads it through the official `_read_depth` + `_resize_depth` and
     requires np.array_equal (bit identity, same dtype/shape) with step 1,
  4. only then replaces the original file.

Any mismatch aborts and leaves the original in place. Needs numpy + opencv.

  python3 eval/data/shrink_depth.py <dataset_root> [scene ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import omega_src  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("scenes", nargs="*")
    a = p.parse_args()
    ds = omega_src.eval_module("datasets")
    root = Path(a.root)
    scenes = a.scenes or sorted(p.name for p in root.iterdir() if (p / "depths").is_dir())
    n = 0
    for scene in scenes:
        for path in sorted((root / scene / "depths").glob("*.npz")):
            if path.name.endswith(".tmp.npz"):
                continue
            expected = ds._resize_depth(ds._read_depth(path.with_suffix("")))
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(tmp, depth=expected)
            got = ds._resize_depth(ds._read_depth(tmp.with_name(path.stem + ".tmp")))
            if got.dtype != expected.dtype or got.shape != expected.shape or not np.array_equal(got, expected):
                tmp.unlink()
                raise SystemExit(f"loader output differs for {path}; original kept")
            tmp.replace(path)
            n += 1
        print(f"{scene}: shrunk+verified")
    print(f"{n} depth maps verified bit-identical through official loader")
    return 0


if __name__ == "__main__":
    sys.exit(main())
