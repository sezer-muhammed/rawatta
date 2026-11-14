# SPDX-License-Identifier: Apache-2.0
"""Tests for eval/score.py and eval/run_eval.py.

Pure-numpy parts always run. Parts that call the official VGGT-Omega evaluator need
torch, opencv, Pillow and the pinned official source (eval/omega_src.py fetches it, or
point RAWATTA_OMEGA_SRC at a checkout); they are skipped when any is missing. The
pipeline test runs eval/run_eval.py end to end with tests/fake_rawatta.py standing in for
`rawatta bench` (no GPU, no weights).

    python3 -m pytest eval/tests -q
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))

import run_eval  # noqa: E402
import score  # noqa: E402


def _official_available() -> bool:
    try:
        import cv2  # noqa: F401
        import PIL  # noqa: F401
        import torch  # noqa: F401

        import omega_src

        omega_src.ensure(quiet=True)
        return True
    except Exception:  # noqa: BLE001
        return False


OFFICIAL = _official_available()
needs_official = pytest.mark.skipif(not OFFICIAL, reason="official evaluator / torch / opencv / Pillow unavailable")


def rand_rot(rng):
    q = rng.standard_normal(4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# --------------------------------------------------------------------------
# numpy helpers
# --------------------------------------------------------------------------

def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 3))
    R, s, t = rand_rot(rng), 2.5, np.array([1.0, -2.0, 0.5])
    b = s * a @ R.T + t
    s2, R2, t2 = score.umeyama(a, b)
    assert np.isclose(s2, s) and np.allclose(R2, R) and np.allclose(t2, t)


def test_unproject_identity_camera():
    depth = np.full((1, 2, 3), 2.0)
    K = np.array([[[1.0, 0, 1.5], [0, 1.0, 1.0], [0, 0, 1]]])
    w2c = np.eye(4)[None]
    p = score.unproject(depth, K, w2c)
    assert np.allclose(p[0, :, :, 2], 2.0)
    assert np.allclose(p[0, 0, 0], [(0.5 - 1.5) * 2, (0.5 - 1.0) * 2, 2.0])


def test_scale_K():
    K = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
    K2 = score.scale_K(K, 0.5, 0.25, 3, 2)
    assert np.allclose(K2, [[50, 0, 22], [0, 25, 8], [0, 0, 1]])


def write_images_bin(path: Path, poses: dict[str, np.ndarray]) -> None:
    """COLMAP images.bin from name -> 3x4 w2c (rotation to quaternion via Shepperd)."""
    def quat(R):
        w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        x = np.copysign(x, R[2, 1] - R[1, 2])
        y = np.copysign(y, R[0, 2] - R[2, 0])
        z = np.copysign(z, R[1, 0] - R[0, 1])
        return w, x, y, z

    b = bytearray(struct.pack("<Q", len(poses)))
    for i, (name, P) in enumerate(poses.items()):
        b += struct.pack("<idddddddi", i + 1, *quat(P[:, :3]), *P[:, 3], 1)
        b += name.encode() + b"\0" + struct.pack("<Q", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(b))


def test_read_colmap_images_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    poses = {f"img{i}.jpg": np.concatenate([rand_rot(rng), rng.standard_normal((3, 1))], 1) for i in range(4)}
    write_images_bin(tmp_path / "images.bin", poses)
    got = score.read_colmap_images(tmp_path / "images.bin")
    assert set(got) == set(poses)
    for k in poses:
        assert np.allclose(got[k], poses[k], atol=1e-9)


# --------------------------------------------------------------------------
# run_eval: maps, splits, tables
# --------------------------------------------------------------------------

def test_paper_frame_maps():
    counts = {ds: len(run_eval.frame_map(ds, 0)) for ds in run_eval.PAPER}
    assert counts == {"eth3d": 13, "co3d": 40, "re10k": 190, "7scenes": 18, "nrgbd": 9, "tum_dynamic": 8}
    co3d = run_eval.frame_map("co3d", 1)
    assert not any(k.startswith(("banana", "donut")) for k in co3d)
    assert all(len(v) == 10 for v in co3d.values())
    assert run_eval.frame_map("eth3d", 0) == run_eval.frame_map("eth3d", 2)  # official frames for every seed
    assert len(run_eval.frame_map("re10k", 0, split="all")) == 200
    assert all(len(v) == 50 for v in run_eval.frame_map("7scenes", 0, "n50").values())


def test_colmap_pick_matches_lab_rule():
    assert run_eval.colmap_pick(10, 4, 0) == [0, 3, 6, 9]
    assert run_eval.colmap_pick(3, 50, 0) == [0, 1, 2]
    s = run_eval.colmap_pick(100, 5, 1)
    assert s == sorted(s) and len(set(s)) == 5 and s == run_eval.colmap_pick(100, 5, 1)


def _res(v3, v30):
    return {"AUC@3": v3, "AUC@30": v30}


def test_deltas_and_worst():
    res = {
        0: {"a": {"exact": _res(50, 90), "x": _res(49.5, 89.8)}, "b": {"exact": _res(30, 80), "x": _res(29, 80.1)}},
        1: {"a": {"exact": _res(52, 91), "x": _res(51, 90)}, "b": {"exact": _res(31, 81), "x": _res(31.2, 81)}},
    }
    dl = run_eval.deltas(res, "exact")
    assert dl[0]["x"]["b"] == pytest.approx((-1.0, 0.1))
    w = run_eval.worst(dl)
    assert w["x"]["worst3"] == pytest.approx([-1.0, -1.0])
    assert w["x"]["at"] == ["b", "a"]
    assert w["x"]["worst30"] == pytest.approx([-0.2, -1.0])
    assert w["exact"]["worst3"] == [0, 0]
    t = run_eval.format_table(res, ["exact", "x"], ["a", "b"], "exact")
    assert "| x | -1.00 ± 0.00 | -0.60 ± 0.57 |" in t


# --------------------------------------------------------------------------
# with the official evaluator
# --------------------------------------------------------------------------

def make_omega_dataset(root: Path, ds: str, scenes: dict[str, int], depth: bool, seed=0,
                       sample: dict[str, list[int]] | None = None) -> dict:
    """A tiny omega-layout dataset; images/depths only for the sampled frames (default: all)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    fmap = {}
    yy, xx = np.mgrid[0:48, 0:64].astype(np.float32)
    for sc, n in scenes.items():
        sd = root / ds / sc
        (sd / "images").mkdir(parents=True)
        names = [f"{i:06d}.png" for i in range(n)]
        ex = np.stack([np.concatenate([rand_rot(rng), rng.standard_normal((3, 1)) * 2], 1) for _ in range(n)])
        K = np.tile(np.array([[50.0, 0, 32], [0, 50, 24], [0, 0, 1]]), (n, 1, 1))
        (sd / "frames.txt").write_text("\n".join(names) + "\n")
        np.savez_compressed(sd / "cameras.npz", extrinsics=ex, intrinsics=K)
        idx = sample[sc] if sample else list(range(n))
        for i in idx:
            Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(sd / "images" / names[i])
            if depth:
                (sd / "depths").mkdir(exist_ok=True)
                d = (2 + xx / 64 + yy / 48 + rng.random()).astype(np.float32)  # smooth: survives resampling
                np.savez_compressed(sd / "depths" / f"{i:06d}.npz", depth=d)
        fmap[sc] = idx
    return fmap


@needs_official
def test_score_omega_perfect_and_noisy(tmp_path):
    fmap = make_omega_dataset(tmp_path / "data", "toy", {"s1": 6, "s2": 5}, depth=True)
    (tmp_path / "fmap.json").write_text(json.dumps(fmap))
    job = {"out": str(tmp_path / "cells"), "configs": ["exact", "noisy"], "depth": True,
           "sequences": [{"name": sc, "images": [str(tmp_path / "data/toy" / sc / "images" / f"{i:06d}.png")
                                                 for i in idx]} for sc, idx in fmap.items()]}
    (tmp_path / "job.json").write_text(json.dumps(job))
    subprocess.run([sys.executable, str(HERE / "fake_rawatta.py"), "bench", "--job", str(tmp_path / "job.json")],
                   check=True, capture_output=True)
    res = score.score_cells(tmp_path / "cells", "omega", tmp_path / "data/toy", fmap)
    ex, no = res["exact"]["metrics"], res["noisy"]["metrics"]
    assert ex["AUC@3"] == pytest.approx(100.0) and ex["AUC@30"] == pytest.approx(100.0)
    assert ex["pairs"] == 15 + 10
    assert no["AUC@3"] < ex["AUC@3"]
    assert ex["delta125"] > 95 and ex["AbsRel"] < 0.05


@needs_official
def test_run_eval_pipeline_with_fake_bench(tmp_path):
    data = tmp_path / "data"
    rng = np.random.default_rng(1)
    for ds, depth in (("7scenes", True), ("co3d", False)):
        fm = run_eval.frame_map(ds, 0)
        keep = dict(list(fm.items())[:2])
        make_omega_dataset(data, ds, {sc: max(idx) + 1 for sc, idx in keep.items()}, depth, seed=len(ds), sample=keep)
    from PIL import Image

    poses = {}
    sd = data / "mipnerf360" / "garden"
    (sd / "images_4").mkdir(parents=True)
    for i in range(12):
        nm = f"DSC{i:04d}.JPG"
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(sd / "images_4" / nm, format="JPEG")
        poses[nm] = np.concatenate([rand_rot(rng), rng.standard_normal((3, 1))], 1)
    write_images_bin(sd / "sparse/0/images.bin", poses)
    fake = [sys.executable, str(HERE / "fake_rawatta.py")]
    out = tmp_path / "run"
    wrapper = tmp_path / "rawatta"
    wrapper.write_text(f"#!/bin/sh\nexec {fake[0]} {fake[1]} \"$@\"\n")
    wrapper.chmod(0o755)
    r = subprocess.run([sys.executable, str(EVAL / "run_eval.py"), "--data", str(data), "--datasets",
                        "7scenes,co3d,mipnerf360", "--configs", "exact,rawatta", "--frames", "6", "--max-scenes", "2",
                        "--rawatta", str(wrapper), "--out", str(out), "--reps", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Worst-dataset change" in r.stdout
    res = json.loads((out / "results.json").read_text())
    m = res["results"]["0"]
    assert m["7scenes"]["exact"]["AUC@3"] == pytest.approx(100.0)
    assert m["mipnerf360"]["exact"]["AUC@3"] == pytest.approx(100.0)
    assert m["co3d"]["rawatta"]["AUC@3"] < 100.0
    assert res["latency_ms"]["exact"] == pytest.approx(100.0)
    assert not list(out.rglob("depth.npy"))  # cleaned up after scoring


@pytest.mark.parametrize("script", ["score.py", "run_eval.py", "omega_src.py", "official/time_official.py",
                                    "data/co3d.py", "data/re10k.py", "data/sevenscenes.py", "data/sevenscenes_long.py",
                                    "data/nrgbd.py", "data/tum_dynamic.py", "data/eth3d.py", "data/sintel.py",
                                    "data/eth3d_fetch.py", "data/sintel_fetch.py", "data/shrink_depth.py",
                                    "data/manifest.py", "data/mipnerf360.py", "data/prepare_official.py",
                                    "data/remote_zip.py"])
def test_help(script):
    pytest.importorskip("cv2") if script.startswith("data/") and script not in (
        "data/manifest.py", "data/mipnerf360.py", "data/remote_zip.py", "data/eth3d_fetch.py",
        "data/prepare_official.py", "data/eth3d.py", "data/sintel.py", "data/shrink_depth.py") else None
    r = subprocess.run([sys.executable, str(EVAL / script), "--help"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage" in r.stdout.lower()
