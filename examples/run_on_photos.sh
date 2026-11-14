#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reconstruct cameras, depth maps and a coloured point cloud from a folder of photos.
#
# usage: examples/run_on_photos.sh <image-folder> [out-folder] [--] [more rawatta run arguments...]
#   out-folder defaults to runs/<folder name>-<config> (for .../buddha/images: runs/buddha-rawatta).
#   The extra arguments go to `rawatta run` unchanged. A --config among them replaces RAWATTA_CONFIG
#   (the script then does not add its own, because rawatta rejects an option given twice).
#
# env:   RAWATTA_CONFIG   rawatta (default) | rawatta-fast | exact | a config grammar string
#        RAWATTA_RUNNER, RAWATTA_BIN, RAWATTA_IMAGE, RAWATTA_GPUS, RAWATTA_WEIGHTS: see examples/lib/rawatta.sh
#
# examples:
#   examples/run_on_photos.sh data/examples/buddha/images
#   RAWATTA_CONFIG=exact examples/run_on_photos.sh my_photos runs/my_photos-exact
#   RAWATTA_RUNNER=docker examples/run_on_photos.sh my_photos runs/my_photos --no-depth
#   RAWATTA_BIN=target/release/rawatta examples/run_on_photos.sh my_photos       # a native build
#   examples/run_on_photos.sh my_photos -- --reps 10 --warmup 2     # time 10 forward passes
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR source=lib/rawatta.sh
. "$here/lib/rawatta.sh"

if [ $# -lt 1 ] || [ "$1" = -h ] || [ "$1" = --help ]; then
  usage
  exit 0
fi
images=$1
shift
[ -d "$images" ] || die "not a folder: $images"
out=
if [ $# -gt 0 ]; then case $1 in -*) ;; *) out=$1; shift ;; esac; fi
if [ $# -gt 0 ] && [ "$1" = -- ]; then shift; fi
config=${RAWATTA_CONFIG:-rawatta}
if opt_value --config "$@"; then config=$OPT_VALUE; else set -- --config "$config" "$@"; fi
[ -n "$out" ] || out=runs/$(folder_label "$images")-$(safe_name "$config")

n=$(count_images "$images")
[ "$n" -gt 0 ] || die "no $IMAGE_KINDS images in $images"
runner=$(rawatta_runner)
echo "rawatta: $n images from $images, config $config, runner $runner -> $out"

rawatta_run images "$images" "$out" "$@"

echo
if have_numpy; then
  python3 "$here/python/load_outputs.py" "$out"
else
  ls -l "$out"
fi
