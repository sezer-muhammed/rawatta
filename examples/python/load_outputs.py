#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Load the outputs of `rawatta run` with numpy only and print a summary.

  python3 load_outputs.py runs/buddha
  python3 load_outputs.py runs/buddha --plot cameras.png     # needs matplotlib (optional)

What it reads from the output folder (missing files are skipped):
  cameras.json                     per frame: world-to-camera 3x4 (OpenCV axes), 3x3 intrinsics, image name
  extrinsics.npy / intrinsics.npy  the same cameras as arrays, (N,3,4) and (N,3,3)
  depth.npy / conf.npy             (N,H,W) depth and confidence maps
  points.ply                       coloured point cloud
  colmap/                          COLMAP text model
  timing.json                      forward latency and memory of the run

World coordinates are those of frame 0 (its camera sits at the origin looking down +z).
Use the functions below from your own code, e.g. `from load_outputs import load_run`.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subsample_ply import read_ply  # noqa: E402
from timing_table import read_timing  # noqa: E402

W2C_KEYS = ("w2c", "extrinsics", "extrinsic", "world_to_camera", "cam_from_world", "Rt")
K_KEYS = ("intrinsics", "K", "intrinsic", "k")
NAME_KEYS = ("image", "image_name", "name", "file", "file_path")


def _pick(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_cameras_json(path):
    """Return (names, w2c (N,3,4), K (N,3,3)) from cameras.json."""
    with open(path) as f:
        data = json.load(f)
    frames = data
    if isinstance(data, dict):
        frames = _pick(data, ("frames", "cameras", "views", "images"))
        if frames is None:
            raise ValueError("%s: no frames/cameras list" % path)
    names, w2c, ks = [], [], []
    for i, fr in enumerate(frames):
        m = np.asarray(_pick(fr, W2C_KEYS), dtype=np.float64)
        if m.size == 16:
            m = m.reshape(4, 4)[:3]
        w2c.append(m.reshape(3, 4))
        ks.append(np.asarray(_pick(fr, K_KEYS), dtype=np.float64).reshape(3, 3))
        names.append(str(_pick(fr, NAME_KEYS) or "frame_%03d" % i))
    return names, np.stack(w2c), np.stack(ks)


def camera_centres(w2c):
    """Camera centres in world coordinates: C = -R^T t for each 3x4 [R|t]."""
    r, t = w2c[:, :, :3], w2c[:, :, 3]
    return -np.einsum("nji,nj->ni", r, t)


def viewing_directions(w2c):
    """Unit optical axes (camera +z) in world coordinates."""
    return w2c[:, 2, :3] / np.linalg.norm(w2c[:, 2, :3], axis=1, keepdims=True)


def load_run(out_dir):
    """Load everything that exists in a rawatta output folder into a dict of numpy arrays."""
    run = {"dir": out_dir}
    p = lambda name: os.path.join(out_dir, name)  # noqa: E731
    if os.path.exists(p("cameras.json")):
        run["names"], run["w2c"], run["K"] = load_cameras_json(p("cameras.json"))
    if os.path.exists(p("extrinsics.npy")):
        run["extrinsics"] = np.load(p("extrinsics.npy"))
        run["intrinsics"] = np.load(p("intrinsics.npy"))
        run.setdefault("w2c", run["extrinsics"].astype(np.float64))
        run.setdefault("K", run["intrinsics"].astype(np.float64))
    for name in ("depth", "conf"):
        if os.path.exists(p(name + ".npy")):
            run[name] = np.load(p(name + ".npy"), mmap_mode="r")
    if os.path.exists(p("points.ply")):
        run["points"], run["colours"] = read_ply(p("points.ply"))
    if os.path.exists(p("timing.json")):
        run["timing"] = read_timing(p("timing.json"))
        with open(p("timing.json")) as f:
            run["timing_raw"] = json.load(f)
    colmap = p("colmap")
    if os.path.isdir(colmap):
        run["colmap"] = {}
        for fn in ("cameras.txt", "images.txt", "points3D.txt"):
            if os.path.exists(os.path.join(colmap, fn)):
                with open(os.path.join(colmap, fn)) as f:
                    rows = [ln for ln in f if ln.strip() and not ln.startswith("#")]
                # images.txt has two lines per image (pose line + 2D points line, which may be empty).
                run["colmap"][fn] = rows
    return run


def summarise(run):
    lines = ["output folder: %s" % run["dir"]]
    if "w2c" in run:
        w2c, k = run["w2c"], run["K"]
        c = camera_centres(w2c)
        steps = np.linalg.norm(np.diff(c, axis=0), axis=1) if len(c) > 1 else np.zeros(0)
        lines.append("cameras: %d frames" % len(w2c))
        lines.append("  focal fx: min %.1f  median %.1f  max %.1f px" % (k[:, 0, 0].min(), np.median(k[:, 0, 0]), k[:, 0, 0].max()))
        lines.append("  principal point (median): (%.1f, %.1f) px" % (np.median(k[:, 0, 2]), np.median(k[:, 1, 2])))
        lines.append("  camera centres: extent %s, path length %.3f, mean step %.3f (scene units)" % (
            np.array2string(c.max(0) - c.min(0), precision=3), steps.sum(), steps.mean() if len(steps) else 0.0))
        lines.append("  frame 0 centre %s (should be ~0: world = frame 0)" % np.array2string(c[0] + 0.0, precision=4))
        r = w2c[:, :, :3]
        ortho = np.abs(np.einsum("nij,nkj->nik", r, r) - np.eye(3)).max()
        lines.append("  rotation orthogonality error (max): %.2e" % ortho)
        if "names" in run:
            lines.append("  images: %s ... %s" % (run["names"][0], run["names"][-1]))
        if "extrinsics" in run and run["extrinsics"].shape == w2c.shape:
            lines.append("  cameras.json vs extrinsics.npy max |diff|: %.2e" % np.abs(run["extrinsics"] - w2c).max())
    if "depth" in run:
        d = np.asarray(run["depth"], dtype=np.float32)
        ok = np.isfinite(d) & (d > 0)
        lines.append("depth: shape %s, valid %.1f%%, min %.3f  median %.3f  max %.3f" % (
            d.shape, 100.0 * ok.mean(), d[ok].min() if ok.any() else 0, np.median(d[ok]) if ok.any() else 0,
            d[ok].max() if ok.any() else 0))
    if "conf" in run:
        cf = np.asarray(run["conf"], dtype=np.float32)
        q = np.percentile(cf[np.isfinite(cf)], [5, 50, 95])
        lines.append("confidence: p5 %.3f  p50 %.3f  p95 %.3f" % tuple(q))
    if "points" in run:
        pts = run["points"]
        lines.append("points.ply: %d points, %s colours, bbox min %s max %s" % (
            len(pts), "with" if run["colours"] is not None else "no",
            np.array2string(pts.min(0), precision=3), np.array2string(pts.max(0), precision=3)))
    if "colmap" in run:
        cm = run["colmap"]
        lines.append("colmap/: %d camera(s), %d image line(s), %d 3D point(s)" % (
            len(cm.get("cameras.txt", [])), len(cm.get("images.txt", [])), len(cm.get("points3D.txt", []))))
    if "timing" in run:
        t = run["timing"]
        reps = t["reps"]
        iqr = t["iqr_ms"] if (reps or 2) >= 2 else None  # one timed pass has no spread
        lines.append("timing: config %s, %s frames, median %s ms (IQR %s) over %s timed pass(es), peak %s MiB, GPU %s%s" % (
            t["config"], t["frames"], _num(t["latency_ms"]), _num(iqr), "?" if reps is None else reps,
            _num(t["peak_mib"], "%.0f"), t["gpu"], "  [SYNTHETIC]" if t["synthetic"] else ""))
    return "\n".join(lines)


def _num(v, spec="%.1f"):
    return "-" if v is None else spec % v


def plot_cameras(run, path=None, show=False):
    """Plot camera centres and viewing directions (3D view + top view). Needs matplotlib."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("--plot needs matplotlib: pip install matplotlib")
    c = camera_centres(run["w2c"])
    v = viewing_directions(run["w2c"])
    scale = 0.1 * max(float(np.ptp(c, axis=0).max()), 1e-6)
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    if "points" in run and len(run["points"]):
        pts = run["points"]
        idx = np.random.default_rng(0).choice(len(pts), size=min(20000, len(pts)), replace=False)
        col = run["colours"][idx] / 255.0 if run["colours"] is not None else "0.6"
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], s=0.3, c=col, alpha=0.5)
    ax.plot(c[:, 0], c[:, 1], c[:, 2], "-", color="0.5", lw=0.8)
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=np.arange(len(c)), cmap="viridis", s=12)
    ax.quiver(c[:, 0], c[:, 1], c[:, 2], v[:, 0], v[:, 1], v[:, 2], length=scale, color="0.2", lw=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("camera centres (colour = frame index)")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(c[:, 0], c[:, 2], "-", color="0.5", lw=0.8)
    ax2.scatter(c[:, 0], c[:, 2], c=np.arange(len(c)), cmap="viridis", s=14)
    ax2.quiver(c[:, 0], c[:, 2], v[:, 0], v[:, 2], angles="xy", color="0.2", width=0.003)
    ax2.annotate("frame 0", (c[0, 0], c[0, 2]))
    ax2.set_xlabel("x")
    ax2.set_ylabel("z (frame 0 looks along +z)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title("top view (x-z)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        print("wrote %s" % path)
    if show:
        plt.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="folder written by `rawatta run --out`")
    ap.add_argument("--plot", metavar="PNG", help="save a plot of the camera centres (needs matplotlib)")
    ap.add_argument("--show", action="store_true", help="open the plot in a window")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.out_dir):
        sys.exit("not a folder: %s" % a.out_dir)
    run = load_run(a.out_dir)
    print(summarise(run))
    if a.plot or a.show:
        if "w2c" not in run:
            sys.exit("no cameras found to plot")
        plot_cameras(run, a.plot, a.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
