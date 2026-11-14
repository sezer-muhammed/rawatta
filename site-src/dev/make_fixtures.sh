#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Regenerate site-src/dev/fixtures/gallery-index.json with the gallery writer the page relies on
# (examples/python/gallery_index.py), from a synthetic run of examples/tests/fake_rawatta and the buddha scene.json.
# Needs any shell with python3 and numpy.
# Usage: site-src/dev/make_fixtures.sh [-h|--help]
#
# The other fixture, fixtures/cameras.rawatta-run.json, is the output of vggt_omega::output::cameras_doc (the serializer
# `rawatta run` uses for cameras.json), printed with serde_json::to_vec_pretty for the camera values of the
# cameras_json_schema test in crates/vggt-omega/src/output.rs and the image names "a.png", "frame 2.png".
# site-src/dev/test_formats.mjs checks that docs/static/js/formats.js reads both.
set -e
case "${1:-}" in
  -h|--help) sed -n '3,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/../.."
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT INT TERM
mkdir -p "$T/images" "$T/site/buddha"
for i in 0 1 2 3; do : > "$T/images/frame_00$i.jpg"; done   # fake_rawatta only reads the file names
python3 examples/tests/fake_rawatta run --images "$T/images" --out "$T/run" --config rawatta --reps 3 --warmup 1
python3 examples/python/subsample_ply.py "$T/run/points.ply" "$T/site/buddha/points.ply" --max-points 2000 --method voxel
python3 examples/python/gallery_index.py --index "$T/site/index.json" --allow-synthetic \
  --scene examples/buddha/scene.json "$T/run" "$T/site/buddha/points.ply"
mkdir -p site-src/dev/fixtures
cp "$T/site/index.json" site-src/dev/fixtures/gallery-index.json
echo "wrote site-src/dev/fixtures/gallery-index.json"
