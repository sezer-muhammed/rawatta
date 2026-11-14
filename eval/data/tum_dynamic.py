#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TUM RGB-D fr3 dynamic sequences -> official VGGT-Omega eval layout.

Source: TUM RGB-D benchmark (CC BY 4.0), https://cvg.cit.tum.de/data/datasets/rgbd-dataset
  https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_<seq>.tgz
  seq in walking_{xyz,static,halfsphere,rpy}, sitting_{xyz,static,halfsphere,rpy}.
One tgz at a time is downloaded to TMP (<1 GB), the needed members are read with
tarfile.extractfile (never extracted to disk by path), and the tgz is deleted.

Conventions:
  * frames = rgb.txt entries with a depth frame within 20 ms (depth.txt) and a
    groundtruth pose within 20 ms (nearest stamp), the TUM associate.py rule.
  * intrinsics: fr3 ROS defaults fx 535.4, fy 539.2, cx 320.1, cy 247.6 (fr3
    images are distributed undistorted; dataset "file formats" page).
  * depth: uint16 PNG, registered to RGB for fr3, / 5000 -> metres, 0 invalid.
  * groundtruth.txt: tx ty tz qx qy qz qw = camera optical centre pose in the
    mocap world (camera-to-world, optical frame x right, y down, z forward)
    -> inverted to w2c, no axis change.
Frame names "<rgb timestamp with . -> _>.png" (the official loader's
Path.with_suffix would cut a dotted timestamp). Maps: seeds 0,1,2 x 10 + 50-frame even.
Verification: depth of each stored frame unprojected and reprojected into the
next stored frame; median |z_pred - z_obs| / z_obs (inside-image, valid) vs the
same with the c2w used un-inverted -> source_tum_dynamic.json.

Needs numpy + opencv (eval/requirements.txt):
  python3 eval/data/tum_dynamic.py --out $DATA/tum_dynamic --tmp $DATA/tmp --maps $DATA/_frames
Calibration (train) split, disjoint from the eval sequences: the fr3
`*_validation` recordings (separate takes with ground truth):
  python3 eval/data/tum_dynamic.py --seqs walking_xyz_validation,walking_halfsphere_validation,\
      walking_rpy_validation,sitting_xyz_validation --seeded50 --name tum_dynamic_train --out $DATA/tum_dynamic_train ...
--seeded50 also writes <name>_n50_s1/_s2 (50 random frames, seeds 1 and 2: frame samples 1 and 2
of calibration/calibrate.py) and stores their frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2w_to_w2c, frame_maps, seeded_maps, union, write_depth, write_maps, write_scene  # noqa: E402

URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_{}.tgz"
SEQS = [f"{a}_{b}" for a in ("walking", "sitting") for b in ("xyz", "static", "halfsphere", "rpy")]
K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])


def read_list(txt: str):
    rows = [l.split() for l in txt.splitlines() if l.strip() and not l.startswith("#")]
    return [(float(r[0]), r[1:]) for r in rows]


def nearest(stamps: np.ndarray, t: float):
    i = int(np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1))
    j = i - 1 if abs(stamps[i - 1] - t) <= abs(stamps[i] - t) else i
    return j, abs(stamps[j] - t)


def quat_to_c2w(v):
    tx, ty, tz, qx, qy, qz, qw = map(float, v)
    q = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [tx, ty, tz]
    return T


def reproj_err(d1, e1, d2, e2):
    h, w = d1.shape
    v, u = np.nonzero(d1 > 0)
    z = d1[v, u]
    Xc = np.linalg.inv(K) @ np.vstack([u * z, v * z, z])
    R1, t1 = e1[:, :3], e1[:, 3]
    Xw = R1.T @ (Xc - t1[:, None])
    X2 = e2[:, :3] @ Xw + e2[:, 3][:, None]
    p = K @ X2
    ok = p[2] > 0.1
    uu, vv = (p[0, ok] / p[2, ok]).round().astype(int), (p[1, ok] / p[2, ok]).round().astype(int)
    zz = X2[2, ok]
    inb = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    obs = d2[vv[inb], uu[inb]]
    m = obs > 0
    if m.sum() < 100:
        return float("nan")
    return float(np.median(np.abs(zz[inb][m] - obs[m]) / obs[m]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--seqs", help="comma-separated fr3 sequence names (default: the 8 eval sequences)")
    p.add_argument("--name", default="tum_dynamic", help="dataset name of the frame maps")
    p.add_argument("--seeded50", action="store_true",
                   help="also write the 50-frame seeded maps <name>_n50_s1/_s2 (calibration samples 1 and 2)")
    a = p.parse_args()
    out, tmp = Path(a.out), Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    counts, valid, verify = {}, {}, {}
    seqs = a.seqs.split(",") if a.seqs else SEQS
    for seq in seqs:
        local = tmp / f"{seq}.tgz"
        req = urllib.request.Request(URL.format(seq), headers={"User-Agent": "rawatta-eval-data"})
        with urllib.request.urlopen(req, timeout=300) as r, local.open("wb") as f:
            while b := r.read(1 << 22):
                f.write(b)
        with tarfile.open(local, "r:gz") as tf:
            members = {m.name.split("/", 1)[1]: m for m in tf.getmembers() if m.isfile() and "/" in m.name}
            txt = lambda n: tf.extractfile(members[n]).read().decode()  # noqa: E731
            rgb, dep, gt = read_list(txt("rgb.txt")), read_list(txt("depth.txt")), read_list(txt("groundtruth.txt"))
            ds, gs = np.array([t for t, _ in dep]), np.array([t for t, _ in gt])
            frames = []
            for t, (path,) in rgb:
                j, dd = nearest(ds, t)
                k, dg = nearest(gs, t)
                if dd < 0.02 and dg < 0.02:
                    frames.append((t, path, dep[j][1][0], gt[k][1]))
            n = len(frames)
            sid = seq
            names = [Path(f[1]).stem.replace(".", "_") + Path(f[1]).suffix for f in frames]
            exs = [c2w_to_w2c(quat_to_c2w(f[3])) for f in frames]
            maps = frame_maps({sid: n})
            if a.seeded50:
                valid[sid] = [i for i, e in enumerate(exs) if np.isfinite(e).all()]
                maps.update(seeded_maps({sid: valid[sid]}))
            need = sorted(union(maps, sid))
            img_dir = out / sid / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            depths = {}
            for i in need:
                _, path, dpath, _ = frames[i]
                (img_dir / names[i]).write_bytes(tf.extractfile(members[path]).read())
                d = cv2.imdecode(np.frombuffer(tf.extractfile(members[dpath]).read(), np.uint8), cv2.IMREAD_UNCHANGED)
                assert d.dtype == np.uint16 and d.shape == (480, 640), (dpath, d.dtype, d.shape)
                depths[i] = d.astype(np.float32) / 5000.0
                write_depth(out / sid / "depths" / f"{Path(names[i]).stem}.npz", depths[i])
        local.unlink()
        write_scene(out / sid, names, exs, [K] * n)
        counts[sid] = n
        ev = maps["_n50"][sid]
        good = [reproj_err(depths[i], exs[i], depths[j], exs[j]) for i, j in zip(ev, ev[1:])]
        bad = [reproj_err(depths[i], quat_to_c2w(frames[i][3])[:3], depths[j], quat_to_c2w(frames[j][3])[:3])
               for i, j in zip(ev, ev[1:])]
        verify[sid] = {"rel_depth_err_median_gt": round(float(np.nanmedian(good)), 4),
                       "rel_depth_err_median_uninverted": round(float(np.nanmedian(bad)), 4)}
        print(f"{sid}: {n} frames, stored {len(need)}, verify {verify[sid]}", flush=True)
    write_maps({**frame_maps(counts), **(seeded_maps(valid) if a.seeded50 else {})}, Path(a.maps), a.name)
    (out / "source_tum_dynamic.json").write_text(json.dumps({
        "source": "TUM RGB-D fr3 dynamic, " + URL, "sequences": seqs, "intrinsics": K.tolist(),
        "depth": "png/5000 m", "pose": "groundtruth c2w inverted, OpenCV axes",
        "verify_next_frame_depth_reprojection": verify}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
