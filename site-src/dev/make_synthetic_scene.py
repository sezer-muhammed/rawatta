# SPDX-License-Identifier: Apache-2.0
"""Write a tiny synthetic example scene for testing the page's point-cloud viewer. NOT shipped with the site.

Writes <out-dir>/index.json and <out-dir>/synthetic/{points.ply, cameras.json} in the formats the page reads:
  points.ply    binary little-endian, x y z float + red green blue uchar (as `rawatta run` writes it)
  cameras.json  {"convention", "width", "height", "frames": [{"image", "w2c", "intrinsics", "fov_y", "fov_x",
                "pose_enc"}]}, the CameraDoc that `rawatta run` writes (crates/vggt-omega/src/output.rs): per frame
                the world-to-camera 3x4 (OpenCV: x right, y down, z forward; world = frame 0) and 3x3 intrinsics
  index.json    {"scenes": [...]} with every field examples/python/gallery_index.py writes, marked "synthetic": true
Pure standard library (no numpy), so it also runs in the Playwright container.
"""
import argparse, json, math, os, random, struct, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out_dir", help="folder to write index.json and synthetic/ into (created if missing)")
out = ap.parse_args().out_dir
scene_dir = os.path.join(out, "synthetic")
os.makedirs(scene_dir, exist_ok=True)
rng = random.Random(0)
pts = []  # (x, y, z, r, g, b) in frame-0 OpenCV coordinates
centre = (0.0, 0.0, 4.0)

# floor (y = +1 is below the cameras in OpenCV coordinates): checkerboard
for _ in range(14000):
    x, z = rng.uniform(-3, 3), rng.uniform(1, 7)
    dark = (math.floor(x * 2) + math.floor(z * 2)) % 2 == 0
    c = (70, 74, 82) if dark else (196, 200, 206)
    pts.append((x, 1.0, z) + c)
# torus knot in the middle, coloured by parameter
for i in range(12000):
    t = 2 * math.pi * i / 12000
    p, q = 2, 3
    r = 0.55 + 0.22 * math.cos(q * t)
    x, y, z = r * math.cos(p * t), -0.22 * math.sin(q * t) - 0.1, r * math.sin(p * t)
    jitter = [rng.gauss(0, 0.03) for _ in range(3)]
    h = i / 12000
    col = tuple(int(255 * (0.5 + 0.5 * math.cos(2 * math.pi * (h + k / 3)))) for k in range(3))
    pts.append((centre[0] + x + jitter[0], centre[1] + y + jitter[1], centre[2] + z + jitter[2]) + col)
# three spheres
for (cx, cy, cz, col) in ((-1.6, 0.6, 5.2, (66, 146, 198)), (1.7, 0.5, 4.8, (203, 24, 29)), (0.2, 0.55, 6.1, (254, 196, 79))):
    for _ in range(2500):
        u, v = rng.uniform(-1, 1), rng.uniform(0, 2 * math.pi)
        s = math.sqrt(1 - u * u)
        pts.append((cx + 0.4 * s * math.cos(v), cy + 0.4 * u, cz + 0.4 * s * math.sin(v)) + col)

with open(os.path.join(scene_dir, "points.ply"), "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\ncomment synthetic test scene (not real data)\n"
             f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
    for p in pts:
        f.write(struct.pack("<fffBBB", *p))


def look_at(cpos, target):
    """World-to-camera [R|t] (OpenCV) for a camera at cpos looking at target, with world 'down' = +y."""
    z = [target[i] - cpos[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in z)); z = [v / n for v in z]
    down = (0.0, 1.0, 0.0)
    x = [down[1] * z[2] - down[2] * z[1], down[2] * z[0] - down[0] * z[2], down[0] * z[1] - down[1] * z[0]]
    n = math.sqrt(sum(v * v for v in x)); x = [v / n for v in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    R = [x, y, z]
    t = [-sum(R[r][k] * cpos[k] for k in range(3)) for r in range(3)]
    return [[R[r][0], R[r][1], R[r][2], t[r]] for r in range(3)]


def quat_xyzw(R):
    """Unit quaternion (x, y, z, w) of a rotation matrix, the order of VGGT's pose encoding."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = ((R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, 0.25 * s)
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        q = (0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s)
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        q = ((R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s)
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        q = ((R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s, (R[1][0] - R[0][1]) / s)
    return q


W, H, F = 512, 336, 400.0
fov_y, fov_x = 2 * math.atan(H / 2 / F), 2 * math.atan(W / 2 / F)
frames = []
for i in range(8):
    a = -math.pi / 2 + i * (2 * math.pi / 3) / 7  # frame 0 at the origin, the rest on an arc around the centre
    cpos = (centre[0] + 4.0 * math.cos(a), -0.4 * (i > 0), centre[2] + 4.0 * math.sin(a))
    if i == 0:
        cpos = (0.0, 0.0, 0.0)
    E = [[round(v, 6) for v in row] for row in look_at(cpos, centre)]
    R = [row[:3] for row in E]
    pose_enc = [E[0][3], E[1][3], E[2][3], *quat_xyzw(R), fov_y, fov_x]
    # same keys and key order as FrameDoc in crates/vggt-omega/src/output.rs
    frames.append({"image": f"frame_{i:03d}.png", "w2c": E,
                   "intrinsics": [[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]],
                   "fov_y": round(fov_y, 6), "fov_x": round(fov_x, 6), "pose_enc": [round(v, 6) for v in pose_enc]})
json.dump({"convention": "opencv w2c (x right, y down, z forward); world = frame 0; model-scale units",
           "width": W, "height": H, "frames": frames},
          open(os.path.join(scene_dir, "cameras.json"), "w"), indent=2)
# every field of examples/python/gallery_index.py; the caption follows its wording and says the data is synthetic
json.dump({"scenes": [{
    "id": "synthetic", "title": "Synthetic test scene", "dir": "synthetic", "points": "points.ply",
    "cameras": "cameras.json", "thumb": "thumb.jpg",
    "caption": "8 synthetic views. SYNTHETIC test data, not a reconstruction or a measurement (site-src/dev/make_synthetic_scene.py).",
    "frames": len(frames), "config": "rawatta", "latency_ms": None, "latency_iqr_ms": None, "reps": None, "warmup": None,
    "gpu": None, "point_count": len(pts), "licence": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
    "source": "https://github.com/sezer-muhammed/rawatta/tree/main/site-src/dev", "author": None, "attribution": None,
    "changes": None, "kind": "synthetic", "synthetic": True}]},
    open(os.path.join(out, "index.json"), "w"), indent=2)
print(f"wrote {len(pts)} points and {len(frames)} cameras to {scene_dir}")
