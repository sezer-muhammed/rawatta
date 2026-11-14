#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for subsample_ply.py on synthetic point clouds (numpy only).

Run:  python3 examples/python/test_subsample_ply.py
"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subsample_ply as sp  # noqa: E402


def write_custom_ply(path, fmt, columns, extra_elements_after=""):
    """Write a PLY with arbitrary vertex columns: columns = [(name, plytype, np.ndarray)]."""
    n = len(columns[0][2])
    head = ["ply", "format %s 1.0" % fmt, "comment synthetic test cloud", "element vertex %d" % n]
    head += ["property %s %s" % (t, name) for name, t, _ in columns]
    if extra_elements_after:
        head += ["element face 1", "property list uchar int vertex_indices"]
    head.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode())
        if fmt == "ascii":
            for i in range(n):
                f.write((" ".join(repr(c[2][i].item()) for c in columns) + "\n").encode())
            if extra_elements_after:
                f.write(b"3 0 1 2\n")
        else:
            e = ">" if fmt == "binary_big_endian" else "<"
            dt = np.dtype([(name, e + sp.PLY_TYPES[t]) for name, t, _ in columns])
            arr = np.empty(n, dtype=dt)
            for name, _, col in columns:
                arr[name] = col
            f.write(arr.tobytes())


class SubsamplePlyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        rng = np.random.default_rng(1)
        self.n = 200_000
        self.xyz = rng.uniform(-1, 1, size=(self.n, 3))
        self.xyz[:, 0] = np.arange(self.n) / self.n  # x encodes the point index (for exact checks)
        self.rgb = rng.integers(0, 256, size=(self.n, 3), dtype=np.uint8)

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, name):
        return os.path.join(self.d, name)

    def test_binary_round_trip(self):
        size = sp.write_ply(self.p("a.ply"), self.xyz, self.rgb)
        self.assertEqual(size, os.path.getsize(self.p("a.ply")))
        xyz, rgb = sp.read_ply(self.p("a.ply"))
        np.testing.assert_allclose(xyz, self.xyz.astype(np.float32), rtol=0, atol=0)
        np.testing.assert_array_equal(rgb, self.rgb)

    def test_no_colour_round_trip(self):
        sp.write_ply(self.p("a.ply"), self.xyz[:100], None)
        xyz, rgb = sp.read_ply(self.p("a.ply"))
        self.assertIsNone(rgb)
        self.assertEqual(xyz.shape, (100, 3))

    def test_ascii_with_extra_props_and_faces(self):
        n = 500
        cols = [("x", "float", self.xyz[:n, 0].astype(np.float32)),
                ("nx", "float", np.zeros(n, np.float32)),
                ("y", "float", self.xyz[:n, 1].astype(np.float32)),
                ("z", "float", self.xyz[:n, 2].astype(np.float32)),
                ("red", "uchar", self.rgb[:n, 0]), ("green", "uchar", self.rgb[:n, 1]),
                ("blue", "uchar", self.rgb[:n, 2]), ("confidence", "float", np.ones(n, np.float32))]
        write_custom_ply(self.p("ascii.ply"), "ascii", cols, extra_elements_after="face")
        xyz, rgb = sp.read_ply(self.p("ascii.ply"))
        np.testing.assert_allclose(xyz, self.xyz[:n].astype(np.float32), atol=1e-6)
        np.testing.assert_array_equal(rgb, self.rgb[:n])

    def test_big_endian_double_and_short_names(self):
        n = 1000
        cols = [("x", "double", self.xyz[:n, 0]), ("y", "double", self.xyz[:n, 1]), ("z", "double", self.xyz[:n, 2]),
                ("r", "uchar", self.rgb[:n, 0]), ("g", "uchar", self.rgb[:n, 1]), ("b", "uchar", self.rgb[:n, 2])]
        write_custom_ply(self.p("be.ply"), "binary_big_endian", cols)
        xyz, rgb = sp.read_ply(self.p("be.ply"))
        np.testing.assert_array_equal(xyz, self.xyz[:n])
        np.testing.assert_array_equal(rgb, self.rgb[:n])

    def test_float_colours_become_uint8(self):
        n = 64
        f = np.linspace(0, 1, n).astype(np.float32)
        cols = [("x", "float", f), ("y", "float", f), ("z", "float", f),
                ("red", "float", f), ("green", "float", f), ("blue", "float", f)]
        write_custom_ply(self.p("fc.ply"), "binary_little_endian", cols)
        _, rgb = sp.read_ply(self.p("fc.ply"))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(int(rgb[0, 0]), 0)
        self.assertEqual(int(rgb[-1, 0]), 255)

    def test_random_keeps_exact_points_and_colours(self):
        xyz, rgb, info = sp.subsample(self.xyz, self.rgb, 50_000, method="random", seed=3)
        self.assertEqual(len(xyz), 50_000)
        idx = np.round(xyz[:, 0] * self.n).astype(np.int64)
        self.assertEqual(len(np.unique(idx)), 50_000)  # no duplicates
        np.testing.assert_array_equal(xyz, self.xyz[idx])
        np.testing.assert_array_equal(rgb, self.rgb[idx])
        xyz2, _, _ = sp.subsample(self.xyz, self.rgb, 50_000, method="random", seed=3)
        np.testing.assert_array_equal(xyz, xyz2)  # deterministic for a fixed seed

    def test_voxel_auto_size_hits_budget(self):
        target = 30_000
        xyz, rgb, info = sp.subsample(self.xyz, self.rgb, target, method="voxel")
        self.assertLessEqual(len(xyz), target)
        self.assertGreaterEqual(len(xyz), int(0.5 * target))  # the search should not overshoot badly
        self.assertIn("voxel_size", info)
        self.assertTrue(np.all(xyz.min(axis=0) >= self.xyz.min(axis=0) - 1e-9))
        self.assertTrue(np.all(xyz.max(axis=0) <= self.xyz.max(axis=0) + 1e-9))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb.shape, xyz.shape)

    def test_voxel_averages_colours_per_cell(self):
        rng = np.random.default_rng(2)
        # Cell edges start at the cloud minimum; these clusters stay well inside one cell each.
        centres = np.array([[0.5, 0.5, 0.5], [3.7, 0.5, 0.5], [0.5, 3.7, 2.7]])
        colours = np.array([[255, 0, 0], [0, 255, 0], [10, 20, 30]], dtype=np.uint8)
        pts = np.concatenate([c + rng.uniform(-0.3, 0.3, size=(1000, 3)) for c in centres])
        cols = np.repeat(colours, 1000, axis=0)
        xyz, rgb, _ = sp.subsample(pts, cols, 100, method="voxel", voxel=1.0)
        self.assertEqual(len(xyz), 3)
        order = np.argsort(xyz[:, 0] * 10 + xyz[:, 1])
        exp = np.argsort(centres[:, 0] * 10 + centres[:, 1])
        np.testing.assert_allclose(xyz[order], centres[exp], atol=0.05)
        np.testing.assert_array_equal(rgb[order], colours[exp])

    def test_nonfinite_points_are_dropped(self):
        pts = self.xyz[:1000].copy()
        pts[10] = np.nan
        pts[20, 1] = np.inf
        xyz, rgb, info = sp.subsample(pts, self.rgb[:1000], 10_000, method="voxel")
        self.assertEqual(info["dropped_nonfinite"], 2)
        self.assertTrue(np.all(np.isfinite(xyz)))
        self.assertEqual(len(xyz), 998)  # under budget: unchanged apart from the dropped points

    def test_byte_budget(self):
        budget = sp.point_budget(300_000, 100_000, has_rgb=True)
        xyz, rgb, _ = sp.subsample(self.xyz, self.rgb, budget, method="random")
        size = sp.write_ply(self.p("b.ply"), xyz, rgb, comments=("x" * 200,))
        self.assertLessEqual(size, 100_000)
        self.assertGreater(size, 90_000)

    def test_cli_end_to_end(self):
        sp.write_ply(self.p("in.ply"), self.xyz, self.rgb)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subsample_ply.py")
        for method in ("voxel", "random"):
            out = self.p("out_%s.ply" % method)
            r = subprocess.run([sys.executable, script, self.p("in.ply"), out, "--max-points", "40000",
                                "--max-bytes", "500000", "--method", method],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertLessEqual(os.path.getsize(out), 500_000)
            xyz, rgb = sp.read_ply(out)
            self.assertLessEqual(len(xyz), 40_000)
            self.assertGreater(len(xyz), 10_000)
            self.assertIsNotNone(rgb)
            with open(out, "rb") as f:
                head = f.read(400)
            self.assertIn(b"format binary_little_endian 1.0", head)
            self.assertIn(b"property uchar red", head)

    def test_gallery_limits_on_a_large_cloud(self):
        # A cloud of roughly the size rawatta writes for ~30 frames: 2M points -> <=300k points and <=8 MB.
        rng = np.random.default_rng(5)
        big = rng.normal(size=(2_000_000, 3))
        cols = rng.integers(0, 256, size=(2_000_000, 3), dtype=np.uint8)
        sp.write_ply(self.p("big.ply"), big, cols)
        out = self.p("big_small.ply")
        self.assertEqual(sp.main([self.p("big.ply"), out, "--quiet"]), 0)
        xyz, rgb = sp.read_ply(out)
        self.assertLessEqual(len(xyz), 300_000)
        self.assertGreater(len(xyz), 150_000)
        self.assertLessEqual(os.path.getsize(out), 8_000_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
