#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# run.sh: run a research harness on the GPU in pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
# (gcc for GPA-VGGT's Triton kernels) with the clones and wheels research/setup.sh prepared.
#
#   research/run.sh [--lock FILE] <script.py> <args...>
#   research/run.sh --lock ~/gpu.lock vggt1b_sched.py eval --checkpoint /rh/ckpt/model.pt \
#       --configs "dense@bf16,dense@sage" --data-root /data/co3d --frames /frames/co3d_s0.json --no-depth
#
# Mounts: research/ and eval/ (ro) under /rawatta, $RESEARCH_HOME -> /rh, $RAWATTA_DATA (default ./data)
# -> /data (ro), $RESEARCH_OUT (default ./runs/research) -> /results (results.jsonl lands there).
# Checkpoints live in /rh/ckpt (setup.sh --vggt1b / --pi3). The VGGT-Omega weights for run_official.py
# --method omega: put vggt_omega_1b_512.pt in $RESEARCH_HOME/ckpt as well.
# --lock FILE: one exclusive flock on FILE for the whole run (the machine's GPU lock); the paper's
# runs kept every locked run under 15 minutes (`search` subcommands stop after --minutes and exit 3: re-run).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
RH=${RESEARCH_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/rawatta/research}
IMG=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
LOCK=
if [ "${1:-}" = "--lock" ]; then LOCK=$2; shift 2; fi
case "${1:-}" in ""|-h|--help) awk 'NR == 1 || /SPDX-License-Identifier/ { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;; esac
SCRIPT=$1; shift
HELD=0; [ -z "$LOCK" ] || HELD=1
DATA=$(cd "${RAWATTA_DATA:-data}" && pwd)
OUT=${RESEARCH_OUT:-runs/research}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
set -- docker run --rm --gpus all --ipc=host -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -e RESEARCH_HOME=/rh -e RESEARCH_SRC=/rh/src -e RAWATTA_OMEGA_SRC=/rh/vggt-omega -e RAWATTA_DATA=/data \
  -e RESEARCH_RESULTS=/results/results.jsonl -e PO_LOCK_HELD=$HELD -e PYTHONPATH=/rh/pyenv \
  -e PYTHONDONTWRITEBYTECODE=1 -e TRITON_CACHE_DIR=/rh/cache/triton -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "$HERE":/rawatta/research:ro -v "$HERE/../eval":/rawatta/eval:ro -v "$RH":/rh \
  -v "$DATA":/data:ro -v "$OUT":/results -w /results --entrypoint python "$IMG" "/rawatta/research/$SCRIPT" "$@"
if [ -n "$LOCK" ]; then exec flock -x "$LOCK" "$@"; else exec "$@"; fi
