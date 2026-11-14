#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write (or update) the gallery index.json read by the project page viewer (docs/static/js/viewer.js).

  python3 gallery_index.py --index docs/static/examples/index.json --order-file examples/scenes.txt \\
      --scene examples/buddha/scene.json data/examples/buddha/images runs/gallery/buddha \\
              docs/static/examples/buddha/points.ply

Each --scene takes: the scene's scene.json (title, credit, licence, source, ...), the folder of photos the
run used (the caption counts them there), the rawatta output folder of the run (its timing.json gives
setting, frames, latency, repetitions and GPU) and the subsampled points.ply (its folder also holds
cameras.json and thumb.jpg). Entries with the same id are replaced; other entries already in the index are
kept. The scenes are written in the order of --order-file (one id per line, # comments), so rebuilding one
scene does not move it; ids not in that list follow, sorted by id.

Per scene, the viewer reads `id`, `title`, `dir`, `points`, `cameras` and `caption`; the caption is the only
per-scene text the page shows, so it names the model (VGGT-Omega) and the setting (RAWATTA, RAWATTA-Fast or
the exact engine), carries the measured latency, and the photo credit with the licence and its URL (CC BY 4.0
for the Buddha photos requires them). The other fields are kept for reference:
  {"scenes": [{"id", "title", "dir", "points", "cameras", "thumb", "caption",
               "photos", "frames", "config", "latency_ms", "latency_iqr_ms", "reps", "warmup", "gpu",
               "point_count", "attribution", "credit", "source_photos", "licence", "licence_url", "source",
               "author", "changes", "kind"}]}
latency_iqr_ms is null when the run had fewer than 2 timed passes (one sample has no spread).

The photo count in the caption is the number of photos in the given folder, and the run must have used all
of them (timing.json "frames"); otherwise the scene is refused, so the caption cannot contradict itself.
timing.json files marked "synthetic": true (written by tests/fake_rawatta) are refused unless
--allow-synthetic is given; accepted ones are marked "synthetic": true in the index and in the caption.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subsample_ply import read_ply_vertices  # noqa: E402
from timing_table import read_timing  # noqa: E402

MIN_REPS = 3  # fewer timed passes than this gets a warning
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")  # what rawatta reads from a folder, any case
MODEL = "VGGT-Ω"
SETTINGS = {
    "rawatta": "accelerated with RAWATTA",
    "rawatta-fast": "accelerated with RAWATTA-Fast",
    "exact": "on the exact engine, without attention approximation",
}


def count_photos(folder):
    """Photos rawatta reads from the folder: .jpg, .jpeg, .png, .webp files directly inside it."""
    return sum(1 for f in os.listdir(folder)
               if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(folder, f)))


def read_order(path):
    """Scene ids from an order file: one per line, blank lines and # comments ignored."""
    if not path:
        return []
    with open(path) as f:
        return [ln.split("#", 1)[0].strip() for ln in f if ln.split("#", 1)[0].strip()]


def order_scenes(scenes, order):
    """Listed ids in list order, then the others sorted by id."""
    pos = {sid: i for i, sid in enumerate(order)}
    return sorted(scenes, key=lambda e: (pos.get(e.get("id"), len(order)),
                                         "" if e.get("id") in pos else str(e.get("id"))))


def scene_attribution(s, photos):
    """'Photos: <credit>, <licence> (<licence URL>); <n> of <total> photos, downscaled.'"""
    credit = s.get("credit") or ", ".join(b for b in (s.get("author"), s["source"]) if b)
    lic = s["licence"] + (" (%s)" % s["licence_url"] if s.get("licence_url") else "")
    used = "%d of %d photos" % (photos, s["source_photos"]) if s.get("source_photos") else "%d photos" % photos
    return "Photos: %s, %s; %s, downscaled." % (credit, lic, used)


# The point clouds and cameras are outputs of the VGGT-Omega weights, so they are not under the page's CC BY 4.0.
OUTPUTS_NOTE = "Outputs: noncommercial research use only (FAIR Noncommercial Research License)."


def caption(e, attribution):
    """The one line of text the page shows under the viewer: what ran, what it measured, whose photos."""
    setting = SETTINGS.get(e["config"], "with --config %s" % e["config"])
    flag = "" if e["config"] not in SETTINGS else " (--config %s)" % e["config"]
    parts = ["Cameras and point cloud predicted by %s from %d photos, %s%s." % (MODEL, e["photos"], setting, flag)]
    lat = "Forward pass %.1f ms" % e["latency_ms"]
    if e["gpu"]:
        lat += " on %s" % e["gpu"]
    if e["reps"]:
        lat += " (median of %d timed run%s)" % (e["reps"], "" if e["reps"] == 1 else "s")
    parts.append(lat + ".")
    if e.get("synthetic"):
        parts.append("SYNTHETIC test data, not a measurement.")
    parts.append(attribution)
    parts.append(OUTPUTS_NOTE)
    return " ".join(parts)


def entry(scene_json, images_dir, run_dir, ply_path, index_path, default_config=None, allow_synthetic=False):
    with open(scene_json) as f:
        s = json.load(f)
    t = read_timing(run_dir)
    if t["synthetic"] and not allow_synthetic:
        raise SystemExit("%s: timing.json is marked synthetic (tests/fake_rawatta); refusing to put made-up "
                         "latencies in the gallery (--allow-synthetic is for tests only)" % run_dir)
    if t["latency_ms"] is None:
        raise SystemExit("%s: no latency in timing.json" % run_dir)
    photos = count_photos(images_dir)
    if photos == 0:
        raise SystemExit("%s: no photos (%s)" % (images_dir, ", ".join(IMAGE_EXTS)))
    frames = t["frames"]
    cams = os.path.join(run_dir, "cameras.json")
    if frames is None and os.path.exists(cams):
        with open(cams) as f:
            c = json.load(f)
        frames = len(c if isinstance(c, list) else c.get("frames") or c.get("cameras") or [])
    if frames != photos:
        raise SystemExit("%s: the run used %s frames, but %s holds %d photos; the caption would contradict "
                         "itself. Run rawatta on that folder again (without GALLERY_SKIP_RUN=1)."
                         % (run_dir, frames, images_dir, photos))
    reps = t["reps"]
    if reps is None or reps < MIN_REPS:
        print("warning: %s: latency from %s timed pass(es); use at least %d (GALLERY_REPS)"
              % (s["id"], "an unknown number of" if reps is None else reps, MIN_REPS), file=sys.stderr)
    scene_dir = os.path.dirname(os.path.abspath(ply_path))
    rel_dir = os.path.relpath(scene_dir, os.path.dirname(os.path.abspath(index_path))).replace(os.sep, "/")
    attribution = scene_attribution(s, photos)
    e = {
        "id": s["id"],
        "title": s["title"],
        "dir": rel_dir,
        "points": os.path.basename(ply_path),
        "cameras": "cameras.json",
        "thumb": "thumb.jpg",
        "caption": None,  # filled below
        "photos": photos,
        "frames": frames,
        "config": t["config"] or default_config,
        "latency_ms": round(t["latency_ms"], 1),
        "latency_iqr_ms": round(t["iqr_ms"], 1) if t["iqr_ms"] is not None and (reps or 0) >= 2 else None,
        "reps": reps,
        "warmup": t["warmup"],
        "gpu": t["gpu"],
        "point_count": int(len(read_ply_vertices(ply_path))) if os.path.exists(ply_path) else None,
        "attribution": attribution,
        "credit": s.get("credit"),
        "source_photos": s.get("source_photos"),
        "licence": s["licence"],
        "licence_url": s.get("licence_url"),
        "source": s["source"],
        "author": s.get("author"),
        "changes": s.get("changes"),
        "kind": s.get("kind"),
    }
    if t["synthetic"]:
        e["synthetic"] = True
    e["caption"] = caption(e, attribution)
    return e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="index.json to write")
    ap.add_argument("--scene", nargs=4, action="append", required=True,
                    metavar=("SCENE_JSON", "IMAGES_DIR", "RUN_DIR", "POINTS_PLY"))
    ap.add_argument("--order-file", default=None,
                    help="scene ids in page order, one per line (examples/scenes.txt); others follow by id")
    ap.add_argument("--config", default=None, help="config name to record if timing.json has none")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="accept timing.json files marked synthetic (tests only)")
    a = ap.parse_args(argv)

    scenes = []
    if os.path.exists(a.index):
        with open(a.index, encoding="utf-8") as f:
            scenes = json.load(f).get("scenes", [])
    for scene_json, images_dir, run_dir, ply in a.scene:
        e = entry(scene_json, images_dir, run_dir, ply, a.index, a.config, a.allow_synthetic)
        scenes = [s for s in scenes if s.get("id") != e["id"]] + [e]
        print("index: %s  %s photos  %s  %.1f ms over %s rep(s)  %s%s" % (
            e["id"], e["photos"], e["config"], e["latency_ms"], e["reps"], e["gpu"],
            "  (SYNTHETIC)" if e.get("synthetic") else ""))
    scenes = order_scenes(scenes, read_order(a.order_file))
    os.makedirs(os.path.dirname(os.path.abspath(a.index)), exist_ok=True)
    with open(a.index, "w", encoding="utf-8") as f:
        json.dump({"scenes": scenes}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("wrote %s (%d scenes: %s)" % (a.index, len(scenes), ", ".join(str(s.get("id")) for s in scenes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
