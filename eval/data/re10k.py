#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""RealEstate10K test subset -> official VGGT-Omega eval layout.

Source: the pixelSplat processed RealEstate10K release (360x640 JPEG frames +
cameras, `.torch` chunks), mirrored on Hugging Face as dataset
`lhmd/re10k_torch` (test/000000.torch .. test/000542.torch + test/index.json;
identical tree oid to Jathie/re10k_torch and BarbaraXXX/re10k_torch). The
original release ships poses + YouTube ids only; many videos are dead, so the
processed chunks are the reachable source with pixels.

Chunk format (pixelSplat src/dataset/dataset_re10k.py): a list of examples
{"key", "url", "timestamps" (N,), "cameras" (N,18), "images" [N uint8 JPEG
byte tensors]}. cameras row = fx fy cx cy (normalised by width/height), 0, 0,
then the 3x4 world-to-camera matrix row-major, OpenCV axes (RealEstate10K
convention; pixelSplat inverts it to c2w). Pixel K: fx*W, fy*H, cx*W, cy*H.
Chunks are read with a restricted unpickler (tensors and containers only; no pickle code runs).

Subset: chunks drawn with random.Random(0).shuffle(range(543)), downloaded one
at a time to TMP (~110 MB, deleted right after), every sequence of a chunk
taken (in key order) until 200 sequences with >= 10 frames. Frames: seeds 0,1,2
x 10 random (common.frame_maps); the 50-frame even map is not materialised
(10-frame protocol; disk budget). Frame names "<timestamp_us>.jpg".

Verification: SIFT matches between frame pairs of the stored images, median
Sampson distance (px) of the GT fundamental matrix vs. the same with the
extrinsics transposed (wrong w2c/c2w) -> source_re10k.json.

Needs numpy + opencv (eval/requirements.txt); no PyTorch: the chunks are read with the
torch-free reader of tools/convert_weights.py (restricted unpickler, tensors only):
  python3 eval/data/re10k.py --out $DATA/re10k --tmp $DATA/tmp --maps $DATA/_frames
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from common import frame_maps, union, write_maps, write_scene  # noqa: E402
from convert_weights import load_torch_file  # noqa: E402

REPO = "lhmd/re10k_torch"
N_CHUNKS, N_SEQ = 543, 200


def fetch(url: str, dst: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "rawatta-eval-data"})
    with urllib.request.urlopen(req, timeout=300) as r, dst.open("wb") as f:
        while b := r.read(1 << 22):
            f.write(b)
    return dst


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def fundamental(e1, K1, e2, K2):
    R1, t1, R2, t2 = e1[:, :3], e1[:, 3], e2[:, :3], e2[:, 3]
    R = R2 @ R1.T
    t = t2 - R @ t1
    return np.linalg.inv(K2).T @ skew(t) @ R @ np.linalg.inv(K1)


def sampson(F, p1, p2):
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    Fx1, Ftx2 = x1 @ F.T, x2 @ F
    num = np.sum(x2 * Fx1, 1) ** 2
    return np.sqrt(num / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2))


def epipolar_check(imgs, exs, Ks):
    sift = cv2.SIFT_create(2000)
    feats = [sift.detectAndCompute(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), None) for i in imgs]
    good, bad = [], []
    bf = cv2.BFMatcher()
    for a in range(len(imgs)):
        for b in range(a + 1, len(imgs)):
            (ka, da), (kb, db) = feats[a], feats[b]
            if da is None or db is None or len(da) < 20 or len(db) < 20:
                continue
            m = [x for x, y in bf.knnMatch(da, db, k=2) if x.distance < 0.7 * y.distance]
            if len(m) < 30:
                continue
            p1 = np.float64([ka[x.queryIdx].pt for x in m])
            p2 = np.float64([kb[x.trainIdx].pt for x in m])
            good.append(np.median(sampson(fundamental(exs[a], Ks[a], exs[b], Ks[b]), p1, p2)))
            wa = np.c_[exs[a][:, :3].T, exs[a][:, 3]]
            wb = np.c_[exs[b][:, :3].T, exs[b][:, 3]]
            bad.append(np.median(sampson(fundamental(wa, Ks[a], wb, Ks[b]), p1, p2)))
    return good, bad


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--max_chunks", type=int, default=40)
    a = p.parse_args()
    out, tmp = Path(a.out), Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    info = json.loads(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60).read())
    rev = info["sha"]
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/test/"
    order = list(range(N_CHUNKS))
    random.Random(0).shuffle(order)
    counts, used, goods, bads = {}, [], [], []
    for c in order[: a.max_chunks]:
        if len(counts) >= N_SEQ:
            break
        f = fetch(base + f"{c:06d}.torch", tmp / "chunk.torch")
        chunk = load_torch_file(f)
        chunk = [{k: (v.array() if hasattr(v, 'array') else [x.array() for x in v] if isinstance(v, list) else v)
                  for k, v in e.items()} for e in chunk]  # materialise before the file goes
        f.unlink()
        used.append(c)
        for ex in sorted(chunk, key=lambda e: e["key"]):
            if len(counts) >= N_SEQ:
                break
            n = int(ex["timestamps"].shape[0])
            if n < 10:
                continue
            sid = ex["key"]
            ts = [int(t) for t in ex["timestamps"]]
            cams = ex["cameras"].astype(np.float64)
            maps = {k: v for k, v in frame_maps({sid: n}).items() if k != "_n50"}
            need = sorted(union(maps, sid))
            img_dir = out / sid / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            hw = None
            imgs = []
            for i in need:
                blob = ex["images"][i].tobytes()
                img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                hw = hw or img.shape[:2]
                assert img.shape[:2] == hw
                (img_dir / f"{ts[i]}.jpg").write_bytes(blob)
                imgs.append(img)
            h, w = hw
            Ks = np.array([[[c_[0] * w, 0, c_[2] * w], [0, c_[1] * h, c_[3] * h], [0, 0, 1]] for c_ in cams])
            exs = cams[:, 6:].reshape(n, 3, 4)
            write_scene(out / sid, [f"{t}.jpg" for t in ts], exs, Ks)
            counts[sid] = n
            if len(counts) % 10 == 1:  # epipolar check on every 10th sequence, seed-0 frames
                s0 = maps["_s0"][sid]
                g, b = epipolar_check([imgs[need.index(i)] for i in s0], exs[s0], Ks[s0])
                goods += g
                bads += b
        del chunk
        print(f"chunk {c}: total {len(counts)} sequences", flush=True)
    maps = {k: v for k, v in frame_maps(counts).items() if k != "_n50"}
    write_maps(maps, Path(a.maps), "re10k")
    (out / "source_re10k.json").write_text(json.dumps({
        "source": f"hf://datasets/{REPO}@{rev} (pixelSplat RealEstate10K chunks, test split)",
        "chunks": used, "n_sequences": len(counts), "frame_maps": sorted(maps),
        "conversion": "cameras[6:18] = w2c 3x4 OpenCV (as is); K = fx*W, fy*H, cx*W, cy*H",
        "verify_sampson_px": {"pairs": len(goods), "gt_median": float(np.median(goods)) if goods else None,
                              "transposed_median": float(np.median(bads)) if bads else None}}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
