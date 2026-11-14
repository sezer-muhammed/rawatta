# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the paper evaluation-data scripts (own code, numpy only).

Output layout = official VGGT-Omega eval layout (eval/datasets.py):
  <root>/<scene>/frames.txt        all frame names of the sequence
  <root>/<scene>/cameras.npz       extrinsics (M,3,4) w2c OpenCV float64,
                                   intrinsics (M,3,3) float64 (pixels)
  <root>/<scene>/images/<name>     sampled frames only
  <root>/<scene>/depths/<stem>.npz key "depth", float32 metres, 0 invalid
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
N_RANDOM = 10
N_EVEN = 50


def sample_random(n: int, seed: int, k: int = N_RANDOM) -> list[int]:
    """k distinct indices uniformly at random, numpy default_rng(seed), sorted."""
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=min(k, n), replace=False))


def sample_even(n: int, k: int = N_EVEN) -> list[int]:
    """k evenly spaced indices over [0, n-1] (np.linspace, rounded)."""
    return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, min(k, n))))


def frame_maps(counts: dict[str, int], valid: dict[str, list[int]] | None = None) -> dict[str, dict[str, list[int]]]:
    """{map_suffix: {scene: indices}} for seeds 0,1,2 and the 50-frame subset.

    If valid[scene] is given (frames with finite GT pose), sampling runs over
    positions in that list and maps back to frame indices; otherwise over 0..n-1.
    """
    valid = valid or {}
    def pick(sc, n, f):
        v = valid.get(sc)
        return f(n) if v is None else sorted(v[i] for i in f(len(v)))
    maps = {f"_s{s}": {sc: pick(sc, n, lambda m, s=s: sample_random(m, s)) for sc, n in counts.items()} for s in SEEDS}
    maps["_n50"] = {sc: pick(sc, n, sample_even) for sc, n in counts.items()}
    return maps


def seeded_maps(valid: dict[str, list[int]], seeds=(1, 2), k: int = N_EVEN) -> dict[str, dict[str, list[int]]]:
    """{"_n50_s<seed>": {scene: indices}}: k frames drawn at random (sample_random over the
    positions of the frames with a finite pose, mapped back to frame indices). These are frame
    samples 1 and 2 of the calibration check for the 50-frame train splits (7-Scenes, TUM)."""
    return {f"_n{k}_s{s}": {sc: sorted(v[i] for i in sample_random(len(v), s, k)) for sc, v in valid.items()}
            for s in seeds}


def union(maps: dict[str, dict[str, list[int]]], scene: str) -> set[int]:
    return set().union(*(m[scene] for m in maps.values()))


def write_maps(maps, out_dir: Path, dataset: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, m in maps.items():
        (out_dir / f"{dataset}{suffix}.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")


def write_scene(scene_dir: Path, names: list[str], extrinsics, intrinsics) -> None:
    ex = np.asarray(extrinsics, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    assert ex.shape == (len(names), 3, 4) and K.shape == (len(names), 3, 3)
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "frames.txt").write_text("\n".join(names) + "\n")
    np.savez_compressed(scene_dir / "cameras.npz", extrinsics=ex, intrinsics=K)


def write_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    d[d < 0] = 0
    np.savez_compressed(path, depth=d)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """(4,4) camera-to-world -> (3,4) world-to-camera, same camera axes."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    return np.concatenate([R.T, (-R.T @ t)[:, None]], axis=1)


# OpenGL camera (x right, y up, z backward) -> OpenCV (x right, y down, z forward):
# right-multiply c2w by diag(1,-1,-1,1).
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
