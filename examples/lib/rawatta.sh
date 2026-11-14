# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# Shared helpers for the example scripts: call rawatta either as a local binary or through the Docker image.
#
# Environment:
#   RAWATTA_RUNNER   auto (default) | binary | docker
#                    auto uses the binary if RAWATTA_BIN is found, otherwise Docker.
#   RAWATTA_BIN      binary name or path (default: rawatta; e.g. target/release/rawatta after a native build)
#   RAWATTA_IMAGE    Docker image (default: ghcr.io/sezer-muhammed/rawatta:latest, published at v0.1.0;
#                    until then build it with `docker build -f docker/Dockerfile -t rawatta .` and set
#                    RAWATTA_IMAGE=rawatta)
#   RAWATTA_GPUS     value for `docker run --gpus` (default: all)
#   RAWATTA_WEIGHTS  weights file. Default: ${XDG_CACHE_HOME:-$HOME/.cache}/rawatta/vggt_omega_1b_512.safetensors
#                    The binary reads this variable itself. In Docker mode the file's folder is mounted read-only
#                    at /weights, as in the README's docker command; the image already points RAWATTA_WEIGHTS
#                    at /weights/vggt_omega_1b_512.safetensors, so only a file with another name is passed on.
#
# Docker mode runs the image as the README does (same image, entrypoint, --user and weights mount), but mounts
# only the input and the output folder instead of the whole current folder:
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" -v <weights folder>:/weights:ro \
#     -v <photo folder>:/work/input:ro -v <out folder>:/work/out <image> run --images /work/input --out /work/out ...
# A video is mounted as that one file (/work/input/<name>), never its whole folder.

RAWATTA_BIN=${RAWATTA_BIN:-rawatta}
RAWATTA_IMAGE=${RAWATTA_IMAGE:-ghcr.io/sezer-muhammed/rawatta:latest}
RAWATTA_WEIGHTS_FILE=vggt_omega_1b_512.safetensors

die() { echo "error: $*" >&2; exit 1; }

# Print the comment block at the top of the calling script (its usage text), without the SPDX line.
usage() { awk 'NR > 1 && /^# SPDX-License-Identifier:/ { next } NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

# opt_value <--name> [args...]: set OPT_VALUE to the value of the last `--name V` or `--name=V` among the
# args and return 0; return 1 (OPT_VALUE empty) when the option is absent. A --name with no value after
# it (last argument, or followed by another --option) stops the script here. Call it directly, not in
# $(...): in a subshell that stop would only end the subshell.
# The scripts use it so that an option the user passes replaces the script's own default instead of
# reaching rawatta twice (rawatta rejects an option given twice).
opt_value() {
  local o=$1 found=1
  shift
  OPT_VALUE=
  while [ $# -gt 0 ]; do
    case $1 in
      "$o")
        case ${2-} in '' | --*) die "$o needs a value" ;; esac
        OPT_VALUE=$2; found=0; shift ;;
      "$o"=*)
        OPT_VALUE=${1#*=}
        [ -n "$OPT_VALUE" ] || die "$o needs a value"
        found=0 ;;
    esac
    shift
  done
  return $found
}

# has_opt <--name> [args...]: true if --name or --name=... is among the args.
has_opt() {
  local o=$1 a
  shift
  for a in "$@"; do case $a in "$o" | "$o"=*) return 0 ;; esac; done
  return 1
}

# Absolute path of an existing file or folder.
abs_path() {
  if [ -d "$1" ]; then (cd "$1" && pwd -P)
  else printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd -P)" "$(basename "$1")"; fi
}

rawatta_runner() {
  case ${RAWATTA_RUNNER:-auto} in
    binary) echo binary ;;
    docker) echo docker ;;
    auto)
      if command -v "$RAWATTA_BIN" >/dev/null 2>&1; then echo binary
      elif command -v docker >/dev/null 2>&1; then echo docker
      else die "neither '$RAWATTA_BIN' nor docker found; set RAWATTA_BIN or install Docker"; fi ;;
    *) die "RAWATTA_RUNNER must be auto, binary or docker (got '$RAWATTA_RUNNER')" ;;
  esac
}

rawatta_weights() {
  echo "${RAWATTA_WEIGHTS:-${XDG_CACHE_HOME:-$HOME/.cache}/rawatta/$RAWATTA_WEIGHTS_FILE}"
}

# The one-time weights download, as in the README (Docker only needs the NVIDIA driver on the host).
weights_hint() {
  cat >&2 <<HINT
Download them once (needs a Hugging Face token from 'hf auth login'):
  mkdir -p ~/.cache/huggingface ~/.cache/rawatta
  docker run --rm --user "\$(id -u):\$(id -g)" -v ~/.cache/huggingface:/hf:ro -v ~/.cache/rawatta:/weights \\
    --entrypoint download-weights $RAWATTA_IMAGE --token-file /hf/token --dir /weights
or natively: tools/download_weights.sh (needs curl, python3>=3.10, numpy). Then set RAWATTA_WEIGHTS if
you saved them somewhere else than ~/.cache/rawatta/$RAWATTA_WEIGHTS_FILE.
HINT
}

# _docker_common <need-weights 0|1>: fill DOCKER_ARGS with what every `docker run` gets: GPUs, the
# caller's user and group (files in the output folder stay theirs), and the weights folder at /weights.
_docker_common() {
  local need=$1 w
  DOCKER_ARGS=(--rm --gpus "${RAWATTA_GPUS:-all}" --user "$(id -u):$(id -g)")
  w=$(rawatta_weights)
  if [ -f "$w" ]; then
    w=$(abs_path "$w")
    DOCKER_ARGS+=(-v "$(dirname "$w"):/weights:ro")
    if [ "$(basename "$w")" != "$RAWATTA_WEIGHTS_FILE" ]; then
      DOCKER_ARGS+=(-e "RAWATTA_WEIGHTS=/weights/$(basename "$w")")
    fi
  elif [ "$need" = 1 ]; then
    echo "error: VGGT-Omega weights not found at $w" >&2
    weights_hint
    exit 1
  fi
}

# rawatta_info: print the GPU and which settings can run on it.
rawatta_info() {
  local runner
  runner=$(rawatta_runner) || exit 1
  if [ "$runner" = binary ]; then
    "$RAWATTA_BIN" info
  else
    _docker_common 0
    docker run "${DOCKER_ARGS[@]}" "$RAWATTA_IMAGE" info
  fi
}

# rawatta_run <images|video> <input folder or file> <output folder> [more rawatta arguments...]
# The extra arguments are passed unchanged (e.g. --config rawatta-fast --max-frames 32). In Docker mode
# they must not hold host paths; give the weights with RAWATTA_WEIGHTS, not --weights.
rawatta_run() {
  local kind=$1 input=$2 out=$3 in_abs out_abs runner target
  shift 3
  case $kind in images | video) ;; *) die "rawatta_run: kind must be images or video" ;; esac
  [ -e "$input" ] || die "input not found: $input"
  runner=$(rawatta_runner) || exit 1
  mkdir -p "$out"
  if [ "$runner" = binary ]; then
    "$RAWATTA_BIN" run --"$kind" "$input" --out "$out" "$@"
    return
  fi
  if has_opt --weights "$@"; then
    die "in Docker mode set RAWATTA_WEIGHTS=<file on this machine> instead of passing --weights"
  fi
  _docker_common 1
  in_abs=$(abs_path "$input")
  out_abs=$(abs_path "$out")
  if [ "$kind" = images ]; then
    target=/work/input
  else
    target=/work/input/$(basename "$in_abs")
  fi
  DOCKER_ARGS+=(-v "$in_abs:$target:ro" -v "$out_abs:/work/out")
  docker run "${DOCKER_ARGS[@]}" "$RAWATTA_IMAGE" run --"$kind" "$target" --out /work/out "$@"
}

# safe_name <text>: the text with anything but letters, digits, dot, dash and underscore replaced by _
# (config grammar strings contain / and @, which do not belong in folder names).
safe_name() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }

# folder_label <folder>: a short name for output folders. data/examples/buddha/images -> buddha.
folder_label() {
  local p name
  p=$(abs_path "$1")
  name=$(basename "$p")
  case $name in images | frames) name=$(basename "$(dirname "$p")") ;; esac
  echo "$name"
}

# count_images <folder>: number of photos rawatta reads (.jpg, .jpeg, .png, .webp, any case) directly inside the folder.
IMAGE_KINDS=".jpg/.jpeg/.png/.webp"
count_images() {
  find "$1" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) |
    wc -l | tr -d ' '
}

# True if python3 with numpy is available (needed only for the optional summaries and the gallery).
have_numpy() { python3 -c 'import numpy' >/dev/null 2>&1; }
