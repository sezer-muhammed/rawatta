#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the exact engine, RAWATTA and RAWATTA-Fast on the same photos and print a latency table.
#
# usage: examples/compare_configs.sh <image-folder> [out-root] [--] [more rawatta run arguments...]
#   Each setting writes its full output to <out-root>/<config>/ (default out-root: runs/compare-<folder name>).
#   The table is read from each run's timing.json: median and IQR over the timed forward passes.
#   Choose the settings with CONFIGS, not --config. A --reps or --warmup among the extra arguments
#   replaces REPS or WARMUP.
#
# env:   CONFIGS   settings to run, in order (default: "exact rawatta rawatta-fast")
#        REPS      timed forward passes per setting (default 10; passed as --reps)
#        WARMUP    untimed passes before them (default 2; passed as --warmup)
#        RAWATTA_RUNNER, RAWATTA_BIN, RAWATTA_IMAGE, RAWATTA_GPUS, RAWATTA_WEIGHTS: see examples/lib/rawatta.sh
#
# Timings are only comparable when nothing else uses the GPU during the runs.
# RAWATTA and RAWATTA-Fast need compute capability 8.9+ (Ada, Hopper, Blackwell; Hopper and sm_100 are built and
# ptxas-checked but not yet run). The exact engine is built and bit-checked for sm_80+, not yet run on Ampere hardware.
# A setting that fails (for example on an older GPU) is reported and the others still run.
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
root=runs/compare-$(folder_label "$images")
if [ $# -gt 0 ]; then case $1 in -*) ;; *) root=$1; shift ;; esac; fi
if [ $# -gt 0 ] && [ "$1" = -- ]; then shift; fi
has_opt --config "$@" && die "choose the settings with CONFIGS=\"exact rawatta ...\", not --config"
has_opt --reps "$@" || set -- --reps "${REPS:-10}" "$@"
has_opt --warmup "$@" || set -- --warmup "${WARMUP:-2}" "$@"
read -r -a configs <<<"${CONFIGS:-exact rawatta rawatta-fast}"
n=$(count_images "$images")
[ "$n" -gt 0 ] || die "no $IMAGE_KINDS images in $images"
runner=$(rawatta_runner)

echo "== GPU"
rawatta_info || echo "(rawatta info failed)"
echo
echo "== $n images from $images, runner $runner, settings: ${configs[*]}"
echo "   rawatta run arguments: $*"

runs=()
failed=()
for c in "${configs[@]}"; do
  out=$root/$(safe_name "$c")
  echo
  echo "== $c -> $out"
  if rawatta_run images "$images" "$out" --config "$c" "$@"; then
    runs+=("$out")
  else
    echo "!! $c failed (exit $?)" >&2
    failed+=("$c")
  fi
done

echo
echo "== latency (forward pass, from timing.json)"
if [ ${#runs[@]} -gt 0 ]; then
  baseline=()
  if [ -d "$root/exact" ]; then baseline=(--baseline exact); fi
  python3 "$here/python/timing_table.py" ${baseline[@]+"${baseline[@]}"} "${runs[@]}"
fi
if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}"
  exit 1
fi
