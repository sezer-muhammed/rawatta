#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CO3Dv2 relative-pose test subset -> official VGGT-Omega eval layout.

Source: CO3Dv2 release (https://github.com/facebookresearch/co3d, links.json),
  https://dl.fbaipublicfiles.com/co3dv2_231130/<category>_000.zip  = metadata
      (set_lists/, sequence_annotations.jgz, frame_annotations.jgz)
  https://dl.fbaipublicfiles.com/co3dv2_231130/<category>_NNN.zip  = frames
      (~20 GB each; only the central directory, the masks and the sampled JPEGs
      of the chosen sequences are range-read via remote_zip.py, nothing else).

Protocol (VGGT / GPA-VGGT eval_co3d.py, which reads the `<cat>_test.jgz`
annotations of RelPose++ preprocess/preprocess_co3d.py):
  * categories = the 41 "seen" categories; per category the sequences of
    set_lists/set_lists_fewview_dev.json["test"] whose
    viewpoint_quality_score > 0.5; frames whose foreground mask (>0.4) is
    empty are dropped (RelPose++ bbox == []); sequences with < 64 frames
    (min_num_images) or T[0]+T[1]+T[2] > 1e5 on any frame are skipped.
  * 10 random frames per sequence.
Subset kept here (budget): 10 categories = random.Random(0).sample(SEEN, 10);
per category the first 5 usable sequences of random.Random(0).shuffle(sorted
eligible). VGGT draws 10 sequences per category with a global RNG that
depends on the category list, so this is not the same draw; frames are
sampled by common.frame_maps over the filtered, frame-number-ordered list
(seeds 0,1,2 x 10). The 50-frame even map is NOT materialised (CO3D protocol
is 10 frames; 50 x 75 frames would blow the disk budget) -> only _s0/_s1/_s2.

Camera conversion (PyTorch3D, intrinsics_format "ndc_isotropic"):
  PyTorch3D: X_cam = X_world @ R + T (row vectors), camera +X left, +Y up,
  +Z forward. OpenCV w2c: R_cv = diag(-1,-1,1) @ R.T, t_cv = diag(-1,-1,1) @ T
  (identical to pose_metrics.convert_pt3d_RT_to_opencv, checked numerically).
  NDC isotropic: s = min(H, W) / 2,  fx = f_ndc_x * s, fy = f_ndc_y * s,
  cx = W/2 - p_ndc_x * s, cy = H/2 - p_ndc_y * s (NDC +x left, +y up).
Verification: each sequence's pointcloud.ply is projected into every stored
frame; the fraction of in-image points landing on the foreground mask must
beat the same score for the un-flipped (wrong) conventions. Results in
<out>/source_co3d.json (verify block).

Needs numpy + opencv (eval/requirements.txt):
  python3 eval/data/co3d.py --out $DATA/co3d --maps $DATA/_frames
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import frame_maps, union, write_maps, write_scene  # noqa: E402
from remote_zip import open_remote  # noqa: E402

LINKS = "https://raw.githubusercontent.com/facebookresearch/co3d/main/co3d/links.json"
SEEN = ["apple", "backpack", "banana", "baseballbat", "baseballglove", "bench", "bicycle", "bottle",
        "bowl", "broccoli", "cake", "car", "carrot", "cellphone", "chair", "cup", "donut", "hairdryer",
        "handbag", "hydrant", "keyboard", "laptop", "microwave", "motorcycle", "mouse", "orange",
        "parkingmeter", "pizza", "plant", "stopsign", "teddybear", "toaster", "toilet", "toybus",
        "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass"]
N_CAT, N_SEQ, MIN_IMAGES, MIN_QUALITY = 10, 5, 64, 0.5
FLIP = np.diag([-1.0, -1.0, 1.0])


def pt3d_to_opencv(R, T, f_ndc, p_ndc, hw):
    R, T = np.asarray(R, np.float64), np.asarray(T, np.float64)
    ex = np.concatenate([FLIP @ R.T, (FLIP @ T)[:, None]], axis=1)
    h, w = hw
    s = min(h, w) / 2.0
    K = np.array([[f_ndc[0] * s, 0, w / 2.0 - p_ndc[0] * s], [0, f_ndc[1] * s, h / 2.0 - p_ndc[1] * s], [0, 0, 1]])
    return ex, K


def reference_convert(R, T):
    """pose_metrics.convert_pt3d_RT_to_opencv (GPA-VGGT / VGGT eval), re-typed for the check."""
    r, t = np.array(R, np.float64), np.array(T, np.float64)
    t[:2] *= -1
    r[:, :2] *= -1
    return np.hstack((r.T, t[:, None]))


def read_ply(data: bytes) -> np.ndarray:
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii", "replace").splitlines()
    assert any("binary_little_endian" in h for h in header), header[:3]
    n, props, in_vertex = 0, [], False
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
             "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
    for h in header:
        p = h.split()
        if p[:2] == ["element", "vertex"]:
            n, in_vertex = int(p[2]), True
        elif p and p[0] == "element":
            in_vertex = False
        elif in_vertex and p and p[0] == "property":
            props.append((p[2], types[p[1]]))
    v = np.frombuffer(data, dtype=np.dtype(props), count=n, offset=end)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def mask_score(pts, ex, K, mask):
    X = pts @ ex[:, :3].T + ex[:, 3]
    z = X[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = (X @ K.T)[:, :2] / zs[:, None]
    h, w = mask.shape
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if ok.sum() < 10:
        return 0.0
    u, v = uv[ok].astype(int).T
    return float((mask[v, u] > 102).sum() / len(pts))  # fraction of ALL points on foreground


class Archives:
    def __init__(self, urls):
        self.urls, self.open = urls, {}

    def find(self, prefix):
        for u in self.urls:
            if u not in self.open:
                self.open[u] = open_remote(u)[0]
                print(f"  listed {u.rsplit('/', 1)[1]}", flush=True)
            zf = self.open[u]
            if any(n.startswith(prefix) for n in zf.namelist()):
                return zf
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--links", default=LINKS, help="links.json URL or path")
    a = p.parse_args()
    out = Path(a.out)
    import urllib.request
    links = json.loads(Path(a.links).read_text() if Path(a.links).is_file()
                       else urllib.request.urlopen(a.links, timeout=60).read())["full"]
    cats = sorted(random.Random(0).sample(SEEN, N_CAT))
    print("categories", cats, flush=True)
    counts, chosen, verify = {}, {}, {}
    for cat in cats:
        meta_zf = open_remote(links[cat][0])[0]
        test = json.loads(meta_zf.read(f"{cat}/set_lists/set_lists_fewview_dev.json"))["test"]
        seqq = {s["sequence_name"]: s["viewpoint_quality_score"]
                for s in json.loads(gzip.decompress(meta_zf.read(f"{cat}/sequence_annotations.jgz")))}
        fa = {(f["sequence_name"], f["frame_number"]): f
              for f in json.loads(gzip.decompress(meta_zf.read(f"{cat}/frame_annotations.jgz")))}
        per_seq: dict[str, list] = {}
        for seq, fn, path in test:
            if seqq.get(seq, 0) > MIN_QUALITY:
                per_seq.setdefault(seq, []).append((fn, path))
        eligible = sorted(s for s, fr in per_seq.items() if len(fr) >= MIN_IMAGES and
                          all(sum(fa[(s, fn)]["viewpoint"]["T"]) <= 1e5 for fn, _ in fr))
        order = eligible[:]
        random.Random(0).shuffle(order)
        arch = Archives(links[cat][1:])
        chosen[cat] = []
        for seq in order:
            if len(chosen[cat]) == N_SEQ:
                break
            zf = arch.find(f"{cat}/{seq}/")
            if zf is None:
                print(f"  {seq}: not in any archive", flush=True)
                continue
            frames, masks = [], {}
            for fn, path in sorted(per_seq[seq]):
                mpath = path.replace("images", "masks").replace(".jpg", ".png")
                m = cv2.imdecode(np.frombuffer(zf.read(mpath), np.uint8), cv2.IMREAD_UNCHANGED)
                if m.ndim == 3:
                    m = m[..., 0]
                if m.dtype == np.uint16:
                    m = (m // 257).astype(np.uint8)
                if (m > 102).any():  # plt.imread(png8)/255 > 0.4
                    frames.append((fn, path))
                    masks[path] = m
            if len(frames) < MIN_IMAGES:
                print(f"  {seq}: {len(frames)} frames after mask filter, skipped", flush=True)
                continue
            sid = f"{cat}_{seq}"
            n = len(frames)
            maps = {k: v for k, v in frame_maps({sid: n}).items() if k != "_n50"}
            need = union(maps, sid)
            names, exs, Ks = [], [], []
            for fn, path in frames:
                vp = fa[(seq, fn)]["viewpoint"]
                assert vp["intrinsics_format"] == "ndc_isotropic"
                ex, K = pt3d_to_opencv(vp["R"], vp["T"], vp["focal_length"], vp["principal_point"],
                                       fa[(seq, fn)]["image"]["size"])
                assert np.allclose(ex, reference_convert(vp["R"], vp["T"]), atol=1e-12)
                names.append(Path(path).name), exs.append(ex), Ks.append(K)
            img_dir = out / sid / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            pts = read_ply(zf.read(f"{cat}/{seq}/pointcloud.ply"))
            sc = {"ours": [], "no_flip": [], "pp_unflipped": []}
            for i in sorted(need):
                fn, path = frames[i]
                blob = zf.read(path)
                img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                assert list(img.shape[:2]) == list(fa[(seq, fn)]["image"]["size"]), (path, img.shape)
                (img_dir / names[i]).write_bytes(blob)
                ex, K = exs[i], Ks[i]
                sc["ours"].append(mask_score(pts, ex, K, masks[path]))
                vp = fa[(seq, fn)]["viewpoint"]
                bad = np.concatenate([np.asarray(vp["R"]).T, np.asarray(vp["T"])[:, None]], 1)
                sc["no_flip"].append(mask_score(pts, bad, K, masks[path]))
                K2 = K.copy()
                h, w = img.shape[:2]
                K2[0, 2], K2[1, 2] = w - K[0, 2], h - K[1, 2]
                sc["pp_unflipped"].append(mask_score(pts, ex, K2, masks[path]))
            write_scene(out / sid, names, exs, Ks)
            counts[sid] = n
            chosen[cat].append(seq)
            verify[sid] = {k: round(float(np.mean(v)), 4) for k, v in sc.items()} | {"n_points": len(pts)}
            print(f"{sid}: {n} frames, stored {len(need)}, verify {verify[sid]}", flush=True)
    maps = {k: v for k, v in frame_maps(counts).items() if k != "_n50"}
    write_maps(maps, Path(a.maps), "co3d")
    (out / "source_co3d.json").write_text(json.dumps({
        "source": "CO3Dv2 co3dv2_231130 (facebookresearch/co3d links.json)",
        "split": "set_lists_fewview_dev.json[test], viewpoint_quality_score>0.5, non-empty mask, >=64 frames",
        "categories": cats, "sequences": chosen, "frame_maps": sorted(maps),
        "conversion": "R_cv=diag(-1,-1,1)R^T, t_cv=diag(-1,-1,1)T; ndc_isotropic s=min(H,W)/2, "
                      "f=f_ndc*s, c=(W/2,H/2)-p_ndc*s",
        "verify_pointcloud_on_mask_fraction": verify}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
