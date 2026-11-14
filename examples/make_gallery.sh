#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the example gallery for the project page: run rawatta on each example scene and write
#   <GALLERY_OUT>/<scene>/points.ply    point cloud, subsampled to <= 300k points and <= 8 MB
#   <GALLERY_OUT>/<scene>/cameras.json  cameras of the run, unchanged
#   <GALLERY_OUT>/<scene>/thumb.jpg     one input photo, 640 px
#   <GALLERY_OUT>/index.json            {"scenes": [{id, title, caption, points, cameras, ...}]}, in the
#                                       order of examples/scenes.txt (rebuilding one scene does not move it)
# The caption (the text the page shows under the viewer) names the model (VGGT-Omega) and the setting, and
# holds the photo count (counted in the photo folder), the measured latency, the GPU and the photo credit with
# the licence and its URL. Latency, setting and GPU come from the run's timing.json.
#
# usage: examples/make_gallery.sh [scene ...]        (default: the scenes listed in examples/scenes.txt)
#
# env:   EXAMPLES_DATA     folder holding <scene>/images (default: <repo>/data/examples).
#                          Fill it first with examples/<scene>/fetch.sh <EXAMPLES_DATA>/<scene>.
#        GALLERY_OUT       output folder (default: <repo>/docs/static/examples)
#        GALLERY_RUNS      where the full rawatta outputs go (default: <repo>/runs/gallery)
#        GALLERY_CONFIG    setting to run (default: rawatta)
#        GALLERY_REPS      timed forward passes per scene (default 10; the page shows their median)
#        GALLERY_WARMUP    untimed passes before them (default 2)
#        GALLERY_ARGS      more `rawatta run` arguments, split on spaces (e.g. "--conf-percentile 60").
#                          A --reps or --warmup here replaces GALLERY_REPS / GALLERY_WARMUP; use
#                          GALLERY_CONFIG, not --config. No --max-frames: the run uses every photo of
#                          the folder (to use fewer, remove photos from the folder).
#        GALLERY_SKIP_RUN  1 = reuse existing outputs in GALLERY_RUNS/<scene> instead of running rawatta
#        GALLERY_ALLOW_SYNTHETIC  1 = accept timing.json files marked "synthetic" (tests only; refused
#                          when GALLERY_OUT is the default docs/ folder)
#        MAX_POINTS=300000 MAX_BYTES=8000000 THUMB_SIDE=640
#        RAWATTA_RUNNER, RAWATTA_BIN, RAWATTA_IMAGE, RAWATTA_GPUS, RAWATTA_WEIGHTS: see examples/lib/rawatta.sh
#
# The latency shown on the page is a real measurement of this run: run the script with the GPU otherwise idle.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)
# shellcheck source-path=SCRIPTDIR source=lib/rawatta.sh
. "$here/lib/rawatta.sh"

if [ $# -gt 0 ] && { [ "$1" = -h ] || [ "$1" = --help ]; }; then
  usage
  exit 0
fi
data=${EXAMPLES_DATA:-$repo/data/examples}
default_site=$repo/docs/static/examples
site=${GALLERY_OUT:-$default_site}
runs=${GALLERY_RUNS:-$repo/runs/gallery}
config=${GALLERY_CONFIG:-rawatta}
max_points=${MAX_POINTS:-300000}
max_bytes=${MAX_BYTES:-8000000}
thumb_side=${THUMB_SIDE:-640}
have_numpy || die "python3 with numpy is required (point cloud subsampling)"

read -r -a run_args <<<"${GALLERY_ARGS:-}"
has_opt --config ${run_args[@]+"${run_args[@]}"} && die "set the setting with GALLERY_CONFIG, not --config in GALLERY_ARGS"
has_opt --max-frames ${run_args[@]+"${run_args[@]}"} &&
  die "no --max-frames in GALLERY_ARGS: the gallery run uses every photo of the folder (remove photos to use fewer)"
has_opt --reps ${run_args[@]+"${run_args[@]}"} || run_args+=(--reps "${GALLERY_REPS:-10}")
has_opt --warmup ${run_args[@]+"${run_args[@]}"} || run_args+=(--warmup "${GALLERY_WARMUP:-2}")
index_flags=(--config "$config")
realpath_py() { python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }
if [ "${GALLERY_ALLOW_SYNTHETIC:-0}" = 1 ]; then
  if [ "$(realpath_py "$site")" = "$(realpath_py "$default_site")" ]; then
    die "GALLERY_ALLOW_SYNTHETIC=1 with the default GALLERY_OUT ($default_site): synthetic timings never go to the page"
  fi
  index_flags+=(--allow-synthetic)
fi

order_file=$here/scenes.txt
scenes=("$@")
if [ ${#scenes[@]} -eq 0 ]; then
  while read -r s; do scenes+=("$s"); done < <(sed -e 's/#.*//' -e 's/[[:space:]]//g' -e '/^$/d' "$order_file")
fi
[ ${#scenes[@]} -gt 0 ] || die "no scenes (examples/scenes.txt is empty)"

# Read one field of a scene.json (empty if it is missing).
scene_field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$1" "$2"; }

# make_thumb <src> <dst>: fit into THUMB_SIDE x THUMB_SIDE.
make_thumb() {
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -nostdin -loglevel error -y -i "$1" \
      -vf "scale=w=${thumb_side}:h=${thumb_side}:force_original_aspect_ratio=decrease:flags=lanczos" -q:v 4 "$2"
  elif python3 -c 'import PIL' >/dev/null 2>&1; then
    python3 -c 'import sys; from PIL import Image; im = Image.open(sys.argv[1]).convert("RGB"); im.thumbnail((int(sys.argv[3]),) * 2, Image.LANCZOS); im.save(sys.argv[2], quality=85)' "$1" "$2" "$thumb_side"
  elif command -v convert >/dev/null 2>&1; then
    convert "$1" -resize "${thumb_side}x${thumb_side}>" -strip -quality 85 "$2"
  else
    die "no tool to make thumbnails (ffmpeg, Pillow or ImageMagick)"
  fi
}

index_args=()
for s in "${scenes[@]}"; do
  meta=$here/$s/scene.json
  [ -f "$meta" ] || die "unknown scene '$s' (no $meta)"
  images=$data/$s/images
  out=$runs/$s
  dest=$site/$s
  echo "== $s"
  [ -d "$images" ] || die "$images not found; run: examples/$s/fetch.sh $data/$s"
  n=$(count_images "$images")
  [ "$n" -gt 0 ] || die "no $IMAGE_KINDS images in $images"
  expected=$(scene_field "$meta" fetched_photos)
  if [ -n "$expected" ] && [ "$n" -ne "$expected" ]; then
    echo "warning: $images has $n photos, examples/$s/fetch.sh writes $expected; the caption says $n" >&2
  fi

  if [ "${GALLERY_SKIP_RUN:-0}" = 1 ]; then
    echo "reusing $out"
  else
    rawatta_run images "$images" "$out" --config "$config" "${run_args[@]}"
  fi
  for f in points.ply cameras.json timing.json; do
    [ -f "$out/$f" ] || die "$out/$f missing"
  done
  # Made-up timings (tests/fake_rawatta) never reach the gallery folder unless a test asks for them.
  if [ "${GALLERY_ALLOW_SYNTHETIC:-0}" != 1 ] && python3 -c 'import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("synthetic") is True else 1)' "$out/timing.json"; then
    die "$out/timing.json is marked synthetic (tests/fake_rawatta); refusing to publish made-up latencies"
  fi

  # The caption counts the photos in the folder, so the run must have used every one of them.
  frames=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("frames", ""))' "$out/timing.json")
  [ "$frames" = "$n" ] ||
    die "$out used $frames frames, but $images holds $n photos; run rawatta on the folder again (without GALLERY_SKIP_RUN=1)"

  mkdir -p "$dest"
  python3 "$here/python/subsample_ply.py" "$out/points.ply" "$dest/points.ply" \
    --max-points "$max_points" --max-bytes "$max_bytes" --method voxel
  cp "$out/cameras.json" "$dest/cameras.json"
  make_thumb "$images/$(scene_field "$meta" thumb)" "$dest/thumb.jpg"
  index_args+=(--scene "$meta" "$images" "$out" "$dest/points.ply")
done

echo "== index"
python3 "$here/python/gallery_index.py" --index "$site/index.json" --order-file "$order_file" \
  "${index_flags[@]}" "${index_args[@]}"
echo
for s in "${scenes[@]}"; do
  ls -l "$site/$s"
done
