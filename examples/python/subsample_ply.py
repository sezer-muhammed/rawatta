#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Subsample a PLY point cloud with numpy only, keeping colours.

Reads ASCII, binary little-endian and binary big-endian PLY files and writes a
binary little-endian PLY with x, y, z (float32) and red, green, blue (uchar).
Other vertex properties (normals, confidence, ...) are dropped.

Two methods:
  voxel   (default) average the points (and colours) inside each cell of a
          regular grid. Without --voxel the cell size is searched so that the
          result has as many points as possible but at most --max-points.
  random  keep a uniform random subset (seeded, so the output is repeatable).

The output also respects --max-bytes (header included).

Examples:
  python3 subsample_ply.py points.ply small.ply                       # <=300k points, <=8 MB
  python3 subsample_ply.py points.ply small.ply --method random --max-points 100000
  python3 subsample_ply.py points.ply small.ply --voxel 0.01          # fixed 1 cm grid (scene units)
"""

import argparse
import os
import sys

import numpy as np

PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
COLOUR_NAMES = (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue"))


def _read_header(f):
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt, elements = None, []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("truncated PLY header")
        tok = line.decode("ascii", "replace").split()
        if not tok or tok[0] in ("comment", "obj_info"):
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append({"name": tok[1], "count": int(tok[2]), "props": []})
        elif tok[0] == "property":
            if not elements:
                raise ValueError("PLY property before any element")
            if tok[1] == "list":
                elements[-1]["props"].append((tok[4], None))
            else:
                elements[-1]["props"].append((tok[2], PLY_TYPES[tok[1]]))
        elif tok[0] == "end_header":
            break
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError("unsupported PLY format: %r" % fmt)
    return fmt, elements


def read_ply_vertices(path):
    """Return the vertex element of a PLY file as a numpy structured array."""
    with open(path, "rb") as f:
        fmt, elements = _read_header(f)
        endian = ">" if fmt == "binary_big_endian" else "<"
        for el in elements:
            has_list = any(t is None for _, t in el["props"])
            if el["name"] == "vertex":
                if has_list:
                    raise ValueError("list properties on vertices are not supported")
                if fmt == "ascii":
                    dtype = np.dtype([(n, t) for n, t in el["props"]])
                    rows = [f.readline() for _ in range(el["count"])]
                    flat = np.array(b" ".join(rows).split(), dtype=np.float64)
                    flat = flat.reshape(el["count"], len(el["props"]))
                    out = np.empty(el["count"], dtype=dtype)
                    for i, (n, _) in enumerate(el["props"]):
                        out[n] = flat[:, i]
                    return out
                dtype = np.dtype([(n, endian + t) for n, t in el["props"]])
                buf = f.read(dtype.itemsize * el["count"])
                if len(buf) != dtype.itemsize * el["count"]:
                    raise ValueError("truncated PLY vertex data")
                return np.frombuffer(buf, dtype=dtype).astype(dtype.newbyteorder("="))
            # Skip an element that comes before the vertices.
            if fmt == "ascii":
                for _ in range(el["count"]):
                    f.readline()
            elif has_list:
                raise ValueError("cannot skip binary list element %r before the vertices" % el["name"])
            else:
                f.seek(np.dtype([(n, t) for n, t in el["props"]]).itemsize * el["count"], os.SEEK_CUR)
    raise ValueError("no vertex element in %s" % path)


def points_and_colours(v):
    """Split a vertex array into xyz (N,3 float64) and rgb (N,3 uint8, or None)."""
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    for cn in COLOUR_NAMES:
        if all(c in names for c in cn):
            rgb = np.stack([v[c] for c in cn], axis=1)
            if rgb.dtype.kind == "f":
                scale = 255.0 if rgb.size and np.nanmax(rgb) <= 1.0 else 1.0
                rgb = np.clip(np.round(rgb * scale), 0, 255)
            elif rgb.dtype.itemsize > 1:  # 16-bit colours
                rgb = rgb.astype(np.float64) / 257.0
            return xyz, np.round(rgb).astype(np.uint8)
    return xyz, None


def read_ply(path):
    """Read a PLY file: returns (xyz float64 (N,3), rgb uint8 (N,3) or None)."""
    return points_and_colours(read_ply_vertices(path))


def _header(n, has_rgb, comments=()):
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += ["comment " + c for c in comments]
    lines += ["element vertex %d" % n, "property float x", "property float y", "property float z"]
    if has_rgb:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_ply(path, xyz, rgb=None, comments=()):
    """Write xyz (+ optional uint8 rgb) as a binary little-endian PLY. Returns the file size."""
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.empty(n, dtype=np.dtype(fields))
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    head = _header(n, rgb is not None, comments)
    with open(path, "wb") as f:
        f.write(head)
        f.write(data.tobytes())
    return len(head) + data.nbytes


def _voxel_keys(xyz, size):
    keys = np.floor((xyz - xyz.min(axis=0)) / size).astype(np.int64)
    dims = keys.max(axis=0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) < 2.0 ** 62:
        return (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    return np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)


def voxel_count(xyz, size):
    """Number of occupied cells of a grid with cell edge `size`."""
    return len(np.unique(_voxel_keys(xyz, size)))


def voxel_downsample(xyz, rgb, size):
    """Average points (and colours) per occupied grid cell."""
    _, inv, counts = np.unique(_voxel_keys(xyz, size), return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    m = len(counts)
    out = np.empty((m, 3))
    for c in range(3):
        out[:, c] = np.bincount(inv, weights=xyz[:, c], minlength=m) / counts
    out_rgb = None
    if rgb is not None:
        out_rgb = np.empty((m, 3), dtype=np.uint8)
        for c in range(3):
            mean = np.bincount(inv, weights=rgb[:, c].astype(np.float64), minlength=m) / counts
            out_rgb[:, c] = np.clip(np.round(mean), 0, 255).astype(np.uint8)
    return out, out_rgb


def find_voxel_size(xyz, max_points, iters=24):
    """Smallest cell size (by bisection in log space) whose cell count is <= max_points."""
    extent = float(np.max(xyz.max(axis=0) - xyz.min(axis=0)))
    if extent <= 0:
        return 1.0
    lo, hi = extent * 1e-6, extent * 1.01  # hi: a single cell, always <= max_points
    if voxel_count(xyz, lo) <= max_points:
        return lo
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if voxel_count(xyz, mid) <= max_points:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.002:
            break
    return hi


def subsample(xyz, rgb, max_points, method="voxel", voxel=None, seed=0):
    """Return (xyz, rgb, info) with at most max_points finite points."""
    keep = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[keep]
    rgb = rgb[keep] if rgb is not None else None
    info = {"input_finite": int(len(xyz)), "dropped_nonfinite": int((~keep).sum()), "method": method}
    rng = np.random.default_rng(seed)
    if method == "voxel" and len(xyz) and (voxel is not None or len(xyz) > max_points):
        size = voxel if voxel is not None else find_voxel_size(xyz, max_points)
        xyz, rgb = voxel_downsample(xyz, rgb, size)
        info["voxel_size"] = float(size)
    elif method not in ("voxel", "random"):
        raise ValueError("unknown method %r" % method)
    if len(xyz) > max_points:  # random method, or a fixed --voxel that left too many points
        idx = np.sort(rng.choice(len(xyz), size=max_points, replace=False))
        xyz = xyz[idx]
        rgb = rgb[idx] if rgb is not None else None
        info["random_cut"] = True
    return xyz, rgb, info


def point_budget(max_points, max_bytes, has_rgb, reserve=512):
    """Largest point count allowed by both limits (max_bytes <= 0 means no byte limit).

    `reserve` bytes are kept free for header comments.
    """
    if max_bytes is None or max_bytes <= 0:
        return max_points
    stride = 12 + (3 if has_rgb else 0)
    head = len(_header(max_points, has_rgb)) + reserve
    return max(0, min(max_points, (max_bytes - head) // stride))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input .ply")
    ap.add_argument("output", help="output .ply (binary little-endian, xyz + rgb)")
    ap.add_argument("--max-points", type=int, default=300_000, help="point limit (default 300000)")
    ap.add_argument("--max-bytes", type=int, default=8_000_000, help="file size limit, 0 = none (default 8000000)")
    ap.add_argument("--method", choices=("voxel", "random"), default="voxel")
    ap.add_argument("--voxel", type=float, default=None, help="fixed voxel size in scene units (voxel method)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    xyz, rgb = read_ply(a.input)
    n_in = len(xyz)
    budget = point_budget(a.max_points, a.max_bytes, rgb is not None)
    out_xyz, out_rgb, info = subsample(xyz, rgb, budget, a.method, a.voxel, a.seed)
    note = "subsampled by rawatta examples/python/subsample_ply.py: %s, %d of %d points" % (a.method, len(out_xyz), n_in)
    if "voxel_size" in info:
        note += ", voxel %.6g" % info["voxel_size"]
    size = write_ply(a.output, out_xyz, out_rgb, comments=(note,))
    if a.max_bytes > 0 and size > a.max_bytes:
        sys.exit("error: wrote %d bytes, above --max-bytes %d" % (size, a.max_bytes))
    if not a.quiet:
        print("%s: %d -> %d points (%s%s), %s colours, %.2f MB -> %s" % (
            a.input, n_in, len(out_xyz), a.method,
            ", voxel %.4g" % info["voxel_size"] if "voxel_size" in info else "",
            "with" if out_rgb is not None else "no", size / 1e6, a.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
