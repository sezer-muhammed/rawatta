#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# setup.sh: one-time setup for the research harnesses (unsupported; as used for the paper).
#   research/setup.sh [--vggt1b] [--pi3]
#  1. clone GPA-VGGT and sparse-vggt (+ its submodules vggt, SpargeAttn, Pi3) at the commits the
#     paper used into $RESEARCH_HOME/src
#  2. build the SpargeAttn (spas_sage_attn, Apache-2.0) wheel for sm_89 in
#     pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel (the same torch as research/run.sh uses)
#  3. install it plus the evaluator's small dependencies into $RESEARCH_HOME/pyenv (on PYTHONPATH in run.sh)
#  4. fetch the official VGGT-Omega source (evaluator) at the pinned commit into $RESEARCH_HOME/vggt-omega
#  5. --vggt1b: download facebook/VGGT-1B model.pt (5 GB); --pi3: yyfz233/Pi3 model.safetensors
# $RESEARCH_HOME defaults to ${XDG_CACHE_HOME:-~/.cache}/rawatta/research. Model licences: see the
# model pages; nothing downloaded here may be redistributed with this repository.
set -euo pipefail
RH=${RESEARCH_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/rawatta/research}
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
GPA_COMMIT=767b172ec2a3feb46609dfc0d9125994b3635fd3
SPARSE_COMMIT=7e8824ff773859ce6f25333bc4fa3f9b5d4cfbbb   # submodules: vggt e37f126, SpargeAttn a2666ad
VGGT1B=0; PI3=0
for a in "$@"; do
  case $a in --vggt1b) VGGT1B=1 ;; --pi3) PI3=1 ;; -h|--help) awk 'NR == 1 || /SPDX-License-Identifier/ { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;; esac
done
mkdir -p "$RH"/{src,ckpt,pyenv,wheels,logs,cache}
cd "$RH/src"
[[ -d gpa ]] || git clone https://github.com/noheat61/GPA-VGGT gpa
git -C gpa checkout -q $GPA_COMMIT
[[ -d sparse ]] || git clone https://github.com/brianwang00001/sparse-vggt sparse
git -C sparse checkout -q $SPARSE_COMMIT
git -C sparse submodule update --init external/vggt external/SpargeAttn external/Pi3

if ! ls "$RH"/wheels/spas_sage_attn-*.whl >/dev/null 2>&1; then
  docker run --rm -v "$RH":/w -e TORCH_CUDA_ARCH_LIST=8.9 -e MAX_JOBS=4 -e HOME=/tmp \
    -u "$(id -u):$(id -g)" -w /w/src/sparse/external/SpargeAttn --entrypoint sh "$IMG" \
    -c "python -m pip wheel --no-build-isolation --no-deps -v -w /w/wheels ." > "$RH/logs/sparge_build.log" 2>&1
  rm -rf "$RH/src/sparse/external/SpargeAttn/build"
  # setup.py generates the kernel instantiations with `python`; without the SM89 sources the wheel has no kernels.
  grep -q "instantiations_sm89/.*\.cu" "$RH/logs/sparge_build.log" || { echo "SM89 instantiations missing" >&2; exit 1; }
  git -C "$RH/src/sparse/external/SpargeAttn" clean -fdq csrc
fi
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$RH":/w --entrypoint sh "$IMG" -c \
  "pip install --no-deps --target /w/pyenv /w/wheels/spas_sage_attn-*.whl opencv-python-headless==4.11.0.86 einops==0.8.1 safetensors==0.5.3"
python3 "$HERE/../eval/omega_src.py" --dir "$RH/vggt-omega"

[[ $VGGT1B = 0 || -s "$RH/ckpt/model.pt" ]] || curl -fL -o "$RH/ckpt/model.pt" \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt
[[ $PI3 = 0 || -s "$RH/ckpt/pi3.safetensors" ]] || curl -fL -o "$RH/ckpt/pi3.safetensors" \
  https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
echo "research setup done: $RH"
