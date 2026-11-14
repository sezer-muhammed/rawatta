#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# run_official.sh: build the rawatta-official image and run time_official.py in it on the GPU.
#
#   eval/official/run_official.sh [--lock FILE] [--build-only] -- <time_official.py args>
#
# Mounts: $RAWATTA_DATA (default ./data) -> /data (ro), the directory of $VGGT_OMEGA_CKPT
# (default ~/.cache/rawatta/vggt_omega_1b_512.pt, the official .pt, not the converted file)
# -> /models (ro), $OUT (default ./runs/official) -> /out.
# --lock FILE: hold an exclusive flock on FILE for the whole run (use the machine's GPU lock
# for timings; numbers taken without it are preliminary).
# Example (the paper's timing scene, 50 frames):
#   eval/official/run_official.sh --lock ~/gpu.lock -- --data-root /data/mipnerf360 \
#       --frames uniform:50 --max-scenes 1 --no-depth --out /out/official-kitchen-n50.json
# Image hygiene: the previous :latest is kept as :prev; nothing else is removed or retagged.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=rawatta-official
LOCK=
BUILD_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in
    --lock) LOCK=$2; shift 2 ;;
    --build-only) BUILD_ONLY=1; shift ;;
    --) shift; break ;;
    -h|--help) awk 'NR == 1 || /SPDX-License-Identifier/ { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    *) break ;;
  esac
done
if docker image inspect "$IMG:latest" >/dev/null 2>&1; then docker tag "$IMG:latest" "$IMG:prev"; fi
BUILDX_BUILDER=default docker build -q -t "$IMG:latest" -f "$HERE/Dockerfile" "$HERE/.."
[ "$BUILD_ONLY" = 1 ] && exit 0
DATA=$(cd "${RAWATTA_DATA:-data}" && pwd)
CKPT=${VGGT_OMEGA_CKPT:-${XDG_CACHE_HOME:-$HOME/.cache}/rawatta/vggt_omega_1b_512.pt}
OUT=${OUT:-runs/official}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
set -- docker run --rm --gpus all --ipc=host -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$DATA":/data:ro -v "$(dirname "$CKPT")":/models:ro -v "$OUT":/out \
  "$IMG:latest" --checkpoint "/models/$(basename "$CKPT")" ${LOCK:+--lock-held} "$@"
if [ -n "$LOCK" ]; then exec flock -x "$LOCK" "$@"; else exec "$@"; fi
