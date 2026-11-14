#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score `rawatta bench` predictions with the official VGGT-Omega evaluator functions.

    score.py --cells <bench out dir> --layout omega --root <dataset root> --map <frame map json> [--points]
    score.py --cells <bench out dir> --layout colmap --job <score job json>          # pose only

`--cells` is the `out` directory of a bench job: `<cells>/<config>/<sequence>/
{extrinsics,intrinsics,depth}.npy`. Every config directory found is scored.

Omega layout (ETH3D, Sintel, CO3Dv2, RE10K, 7-Scenes, NRGBD, TUM-dynamic): exactly
the per-scene logic of the official `eval/evaluate.py`, with the official functions
(`Dataset.load`, `model._source_crop`, `metrics.crop_depth`, `resize_depth`,
`depth_metrics`, `pose_errors`): AUC pooled over all frame pairs of all scenes,
depth averaged over scenes. Pose-only datasets (no `depths/`, e.g. CO3Dv2, RE10K)
load frames and cameras.npz directly. Extra numbers from the same pose errors:
AUC@5, RRA/RTA@5 and @15. The official code is fetched at run time at the pinned
commit (eval/omega_src.py); it is not part of this repository.

COLMAP layout (Mip-NeRF 360): the score job lists, per sequence, the images and
the scene directory whose `sparse/0/images.bin` holds the ground-truth poses.

`--points` adds point-map metrics (unproject predicted depth with the predicted
cameras and GT depth with the GT cameras on the GT grid, Umeyama Sim(3) on
corresponding pixels, then mean L2 = point error, accuracy, completeness and
Chamfer = (acc + comp) / 2 on <= 20k seeded points per scene, in GT units).

Needs numpy, torch (CPU is enough; the official metrics use it), opencv and
Pillow: `pip install -r eval/requirements.txt`. Prints one JSON object
{config: {"metrics": {...}, "per_scene": {...}}} on stdout.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omega_src  # noqa: E402

_M: dict = {}


def M():
    """The official eval/metrics.py."""
    if "m" not in _M:
        _M["m"] = omega_src.eval_module("metrics")
    return _M["m"]


def D():
    """The official eval/datasets.py."""
    if "d" not in _M:
        _M["d"] = omega_src.eval_module("datasets")
    return _M["d"]


def source_crop(paths):
    """The official preprocessing crop (eval/model.py `_source_crop`)."""
    if "model" not in _M:
        _M["model"] = omega_src.eval_module("model")
    return _M["model"]._source_crop(paths)


def auc(r, t, k):
    """Official AUC: mean of the cumulative histogram of max(rot, trans) error, 1-degree bins up to k."""
    return M()._auc(r, t, k)


def pose_extras(r, t):
    out = {"AUC@3": auc(r, t, 3), "AUC@5": auc(r, t, 5), "AUC@30": auc(r, t, 30)}
    for k in (5, 15):
        out[f"RRA@{k}"] = float(np.mean(r < k) * 100)
        out[f"RTA@{k}"] = float(np.mean(t < k) * 100)
    out["pairs"] = int(r.size)
    return out


def unproject(depth, K, w2c):
    """depth (N,H,W), K (N,3,3), w2c (N,3|4,4) -> world points (N,H,W,3); pixel centres at +0.5."""
    n, h, w = depth.shape
    v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    out = np.empty((n, h, w, 3))
    for i in range(n):
        x = (u - K[i, 0, 2]) / K[i, 0, 0] * depth[i]
        y = (v - K[i, 1, 2]) / K[i, 1, 1] * depth[i]
        cam = np.stack([x, y, depth[i]], -1)
        R, t = w2c[i][:3, :3], w2c[i][:3, 3]
        out[i] = (cam - t) @ R  # R^T (p - t)
    return out


def umeyama(a, b):
    """Sim(3) (s, R, t) minimising |s R a + t - b|^2 over corresponding rows."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    cov = B.T @ A / len(a)
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1
    R = U @ np.diag(d) @ Vt
    s = (S * d).sum() / (A ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def nn_mean(a, b, chunk=4096):
    """Mean distance from each row of a to its nearest row of b (torch cdist, GPU if present)."""
    import torch  # noqa: PLC0415

    ta, tb = torch.from_numpy(np.ascontiguousarray(a)).double(), torch.from_numpy(np.ascontiguousarray(b)).double()
    if torch.cuda.is_available():
        ta, tb = ta.cuda(), tb.cuda()
    best = [torch.cdist(c, tb).min(1).values for c in ta.split(chunk)]
    return float(torch.cat(best).mean())


def point_metrics(pp, gp, seed=0, cap=20000):
    s, R, t = umeyama(pp, gp)
    al = s * pp @ R.T + t
    pe = float(np.linalg.norm(al - gp, axis=1).mean())
    rng = np.random.default_rng(seed)
    if len(al) > cap:
        idx = rng.choice(len(al), cap, replace=False)
        al, gp = al[idx], gp[idx]
    acc, comp = nn_mean(al, gp), nn_mean(gp, al)
    return {"point_error": pe, "accuracy": acc, "completeness": comp, "chamfer": 0.5 * (acc + comp)}


def scale_K(K, sx, sy, ox=0.0, oy=0.0):
    K = K.copy()
    K[..., 0, :] *= sx
    K[..., 1, :] *= sy
    K[..., 0, 2] -= ox
    K[..., 1, 2] -= oy
    return K


def config_dirs(cells: Path) -> list[Path]:
    """Config directories of a bench output (a config name's `/` is stored as `%`)."""
    return sorted(p for p in cells.iterdir() if p.is_dir() and not p.name.startswith("_"))


def summarize(per: dict, rots: list, trs: list) -> dict:
    if not rots:
        return {"error": "no predictions", "per_scene": per}
    r, t = np.concatenate(rots), np.concatenate(trs)
    summary = pose_extras(r, t)
    for k in ("delta125", "AbsRel", "point_error", "accuracy", "completeness", "chamfer"):
        v = [row[k] for row in per.values() if k in row]
        if v:
            summary[k] = float(np.mean(v)) * (100.0 if k == "delta125" else 1.0)
    summary["scenes"] = sum(1 for v in per.values() if "missing" not in v)
    summary["missing"] = sum(1 for v in per.values() if "missing" in v)
    return {"metrics": summary, "per_scene": per}


def score_omega(cfg_dir: Path, root: Path, fmap: dict, points: bool) -> dict:
    metrics = M()
    ds = D().Dataset(str(root))
    rots, trs, per = [], [], {}
    for scene, idx in fmap.items():
        sd = cfg_dir / scene
        if not (sd / "extrinsics.npy").is_file():
            per[scene] = {"missing": True}
            continue
        if (root / scene / "depths").is_dir():
            batch = ds.load(scene, idx)
        else:  # pose-only datasets (CO3Dv2, RealEstate10K): same files, no depth
            names = (root / scene / "frames.txt").read_text().split()
            with np.load(root / scene / "cameras.npz") as c:
                batch = D().Batch(image_paths=[str(root / scene / "images" / names[i]) for i in idx], depth=None,
                                  extrinsics=c["extrinsics"][idx], frame_names=[names[i] for i in idx])
        crop = source_crop(batch.image_paths)
        ex = metrics.pad_extrinsics(np.load(sd / "extrinsics.npy"))
        r, t = metrics.pose_errors(ex, batch.extrinsics)
        rots.append(r)
        trs.append(t)
        row = {"rotation_mean": float(r.mean()), "translation_mean": float(t.mean()), **pose_extras(r, t)}
        if (sd / "depth.npy").is_file() and batch.depth is not None:
            pd = np.load(sd / "depth.npy")
            gd = metrics.crop_depth(batch.depth, crop)
            pr = metrics.resize_depth(pd, gd.shape[-2:])
            row.update(metrics.depth_metrics(pr, gd))
            if points:
                row.update(_points(root, scene, idx, batch, crop, sd, pd, gd, pr, ex))
        per[scene] = row
    return summarize(per, rots, trs)


def _points(root, scene, idx, batch, crop, sd, pd, gd, pr, ex):
    from PIL import Image  # noqa: PLC0415

    with np.load(root / scene / "cameras.npz") as c:
        Kg = c["intrinsics"][idx]
    with Image.open(batch.image_paths[0]) as im:
        W0, H0 = im.size
    full_w = 518  # the official loader's depth grid: width 518, height a multiple of 14
    full_h = round(H0 * full_w / W0 / 14) * 14
    left, top = crop[0], crop[1]
    Kg = scale_K(Kg, full_w / W0, full_h / H0, round(left * full_w), round(top * full_h))
    Kp = np.load(sd / "intrinsics.npy").astype(np.float64)
    Kp = scale_K(Kp, gd.shape[2] / pd.shape[2], gd.shape[1] / pd.shape[1])
    gp = unproject(gd, Kg, batch.extrinsics)
    pp = unproject(pr, Kp, ex)
    m = np.isfinite(gd) & (gd > 0.1) & (gd < 100)
    return point_metrics(pp[m], gp[m])


def read_colmap_images(path: Path) -> dict[str, np.ndarray]:
    """name -> (3,4) world-to-camera from a COLMAP images.bin."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            np2 = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * np2, 1)
            w, x, y, z = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            out[name.decode()] = np.concatenate([R, [[tx], [ty], [tz]]], 1)
    return out


def score_colmap(cfg_dir: Path, job: dict) -> dict:
    metrics = M()
    rots, trs, per = [], [], {}
    for s in job["sequences"]:
        sd = cfg_dir / s["name"]
        if not (sd / "extrinsics.npy").is_file():
            per[s["name"]] = {"missing": True}
            continue
        gt = read_colmap_images(Path(s["gt_scene"]) / "sparse/0/images.bin")
        G = metrics.pad_extrinsics(np.stack([gt[Path(p).name] for p in s["images"]]))
        r, t = metrics.pose_errors(metrics.pad_extrinsics(np.load(sd / "extrinsics.npy")), G)
        rots.append(r)
        trs.append(t)
        per[s["name"]] = pose_extras(r, t)
    return summarize(per, rots, trs)


def score_cells(cells: Path, layout: str, root: Path | None = None, fmap: dict | None = None,
                job: dict | None = None, points: bool = False, configs: list[str] | None = None) -> dict:
    out = {}
    for cfg_dir in config_dirs(cells):
        if configs is not None and cfg_dir.name not in configs:
            continue
        if layout == "omega":
            out[cfg_dir.name] = score_omega(cfg_dir, root, fmap, points)
        else:
            out[cfg_dir.name] = score_colmap(cfg_dir, job)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True, help="bench output directory (<config>/<sequence>/...)")
    ap.add_argument("--layout", choices=("omega", "colmap"), default="omega")
    ap.add_argument("--root", help="omega: dataset root")
    ap.add_argument("--map", help="omega: frame map JSON {scene: [frame indices]}")
    ap.add_argument("--job", help="colmap: score job JSON (sequences with images and gt_scene)")
    ap.add_argument("--points", action="store_true", help="also compute point-map metrics (needs depth)")
    ap.add_argument("--out", help="also write the JSON here")
    a = ap.parse_args()
    cells = Path(a.cells)
    if a.layout == "omega":
        if not (a.root and a.map):
            ap.error("--layout omega needs --root and --map")
        res = score_cells(cells, "omega", Path(a.root), json.loads(Path(a.map).read_text()), points=a.points)
    else:
        if not a.job:
            ap.error("--layout colmap needs --job")
        res = score_cells(cells, "colmap", job=json.loads(Path(a.job).read_text()))
    s = json.dumps(res)
    if a.out:
        Path(a.out).write_text(s + "\n")
    print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
