#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for gallery_index.py: caption wording, photo count, scene order, refusals (numpy only).

Run:  python3 examples/python/test_gallery_index.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gallery_index as gi  # noqa: E402
from subsample_ply import write_ply  # noqa: E402

EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GalleryIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def scene(self, sid, photos, frames=None, config="rawatta", synthetic=False, reps=10):
        """A photo folder with `photos` files, a run folder whose timing.json used `frames` frames, a PLY."""
        d = os.path.join(self.tmp, sid)
        images, run, site = (os.path.join(d, x) for x in ("images", "run", "site"))
        for p in (images, run, site):
            os.makedirs(p)
        for i in range(photos):
            open(os.path.join(images, "frame_%03d.jpg" % i), "wb").close()
        open(os.path.join(images, "notes.txt"), "w").close()  # not a photo
        os.makedirs(os.path.join(images, "sub.jpg"))  # a folder, not a photo
        t = {"config": config, "frames": photos if frames is None else frames, "reps": reps, "warmup": 2,
             "median_ms": 123.45, "iqr_ms": 1.0, "gpu": "Test GPU"}
        if synthetic:
            t["synthetic"] = True
        with open(os.path.join(run, "timing.json"), "w") as f:
            json.dump(t, f)
        ply = os.path.join(site, "points.ply")
        write_ply(ply, np.zeros((5, 3)), np.zeros((5, 3), np.uint8))
        return images, run, ply

    def buddha_json(self):
        return os.path.join(EX, "buddha", "scene.json")

    def test_count_photos_reads_only_photos_directly_inside(self):
        images, _, _ = self.scene("a", 4)
        open(os.path.join(images, "UPPER.PNG"), "wb").close()
        open(os.path.join(images, "x.webp"), "wb").close()
        self.assertEqual(gi.count_photos(images), 6)

    def test_caption_names_model_setting_licence_url_and_folder_count(self):
        images, run, ply = self.scene("buddha", 7)
        e = gi.entry(self.buddha_json(), images, run, ply, os.path.join(self.tmp, "index.json"))
        c = e["caption"]
        self.assertTrue(c.startswith("Cameras and point cloud predicted by VGGT-Ω from 7 photos, "
                                     "accelerated with RAWATTA (--config rawatta)."), c)
        self.assertIn("Forward pass 123.5 ms on Test GPU (median of 10 timed runs).", c)
        self.assertIn("CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)", c)
        self.assertIn("doi:10.5281/zenodo.1203283", c)
        self.assertIn("; 7 of 67 photos, downscaled.", c)
        self.assertNotIn("34", c)  # the fetch.sh count must not leak into the caption
        self.assertEqual((e["photos"], e["frames"]), (7, 7))
        self.assertEqual(e["attribution"] + " " + gi.OUTPUTS_NOTE, c[c.index("Photos: "):])
        self.assertTrue(c.endswith("Outputs: noncommercial research use only (FAIR Noncommercial Research License)."), c)

    def test_caption_per_setting(self):
        for config, want in (("rawatta-fast", "accelerated with RAWATTA-Fast (--config rawatta-fast)."),
                             ("exact", "on the exact engine, without attention approximation (--config exact)."),
                             ("dense@bf16", "with --config dense@bf16.")):
            images, run, ply = self.scene("s-" + config.replace("@", "_"), 3, config=config)
            c = gi.entry(self.buddha_json(), images, run, ply, os.path.join(self.tmp, "i.json"))["caption"]
            self.assertIn("from 3 photos, " + want, c)

    def test_run_that_did_not_use_every_photo_is_refused(self):
        images, run, ply = self.scene("buddha", 7, frames=6)
        with self.assertRaises(SystemExit) as cm:
            gi.entry(self.buddha_json(), images, run, ply, os.path.join(self.tmp, "index.json"))
        self.assertIn("holds 7 photos", str(cm.exception))

    def test_synthetic_timing_refused_unless_allowed(self):
        images, run, ply = self.scene("buddha", 3, synthetic=True)
        with self.assertRaises(SystemExit) as cm:
            gi.entry(self.buddha_json(), images, run, ply, os.path.join(self.tmp, "index.json"))
        self.assertIn("synthetic", str(cm.exception))
        e = gi.entry(self.buddha_json(), images, run, ply, os.path.join(self.tmp, "index.json"),
                     allow_synthetic=True)
        self.assertTrue(e["synthetic"])
        self.assertIn("SYNTHETIC test data, not a measurement.", e["caption"])

    def test_order_follows_the_list_not_the_build_order(self):
        order_file = os.path.join(self.tmp, "scenes.txt")
        with open(order_file, "w") as f:
            f.write("# page order\nbuddha\n\nkorno-rock  # outdoor\n")
        self.assertEqual(gi.read_order(order_file), ["buddha", "korno-rock"])
        index = os.path.join(self.tmp, "site", "index.json")
        os.makedirs(os.path.dirname(index))
        with open(index, "w") as f:
            json.dump({"scenes": [{"id": "aaa-extra", "title": "kept"}, {"id": "zzz-extra"}]}, f)
        kr = dict(json.load(open(self.buddha_json())), id="korno-rock")
        kr_json = os.path.join(self.tmp, "korno.json")
        with open(kr_json, "w") as f:
            json.dump(kr, f)
        for sid, meta in (("buddha", self.buddha_json()), ("korno-rock", kr_json), ("buddha", self.buddha_json())):
            images, run, ply = self.scene(sid + str(len(os.listdir(self.tmp))), 3)
            gi.main(["--index", index, "--order-file", order_file, "--scene", meta, images, run, ply])
            ids = [s["id"] for s in json.load(open(index, encoding="utf-8"))["scenes"]]
        self.assertEqual(ids, ["buddha", "korno-rock", "aaa-extra", "zzz-extra"])
        with open(index, encoding="utf-8") as f:
            self.assertIn("VGGT-Ω", f.read())  # written as UTF-8, not \u escapes

    def test_order_without_a_list_is_by_id(self):
        got = gi.order_scenes([{"id": "b"}, {"id": "a"}, {"id": "c"}], [])
        self.assertEqual([s["id"] for s in got], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
