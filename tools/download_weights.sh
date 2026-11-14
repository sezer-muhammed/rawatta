#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# download_weights.sh: download the official VGGT-Omega checkpoint (Hugging Face repo
# facebook/VGGT-Omega, file vggt_omega_1b_512.pt, gated) with curl, check its SHA-256 and
# convert it with tools/convert_weights.py into the safetensors file rawatta loads.
# Run with --help for the options. The Docker image installs it as `download-weights`.
#
# Token: the token is never printed and never put on a command line: curl reads the
# Authorization header from a private temporary file (mode 600, deleted on exit).
# Download: resumable; pinned to one Hugging Face revision ($REV below, override with
# RAWATTA_HF_REVISION); the SHA-256 is checked against the LFS object id Hugging Face reports.
# HF_ENDPOINT (a Hugging Face mirror, as in huggingface_hub) is honoured.
# A file of full size that fails the check is deleted and downloaded again.
#
# Licence of this script: Apache-2.0. The weights are under the FAIR Noncommercial Research
# License (research use only): never redistribute the checkpoint or the converted files.
set -eu

REPO=facebook/VGGT-Omega
FILE=vggt_omega_1b_512.pt
# Hugging Face commit of facebook/VGGT-Omega current on 2026-11-14 (last modified 2026-09-08).
REV=${RAWATTA_HF_REVISION:-1041e80fc0e911235d3426b0a3d9a81075111a53}
OUTNAME=vggt_omega_1b_512.safetensors
# rawatta's default weights folder; RAWATTA_CACHE overrides where this script writes (the image
# sets it to /weights).
DEFAULT_DIR=${XDG_CACHE_HOME:-${HOME:-}/.cache}/rawatta
DIR=${RAWATTA_CACHE:-$DEFAULT_DIR}
TOKEN_FILE=
FP8=
KEEP=0
CONVERT=1
PROG=${0##*/}

usage() {
  cat <<EOF
usage: $PROG [--token-file PATH] [--dir DIR] [--fp8] [--keep-pt] [--no-convert]

Downloads the official VGGT-Omega checkpoint ($REPO, $FILE, gated on
Hugging Face) with curl, checks its SHA-256 and converts it to $OUTNAME.

  --token-file PATH  Hugging Face token file. Without it: \$HF_TOKEN, then \$HF_HOME/token,
                     then ~/.cache/huggingface/token (the file 'hf auth login' writes).
  --dir DIR          output folder (default: \$RAWATTA_CACHE if set, else
                     \${XDG_CACHE_HOME:-~/.cache}/rawatta, where rawatta looks by default)
  --fp8              also write vggt_omega_1b_512_fp8.safetensors (only +fp8 research configs need it)
  --keep-pt          keep the downloaded .pt after converting
  --no-convert       download the .pt only (implies --keep-pt)
  -h, --help         show this help

The token's account must have been granted access on https://huggingface.co/$REPO
(Hugging Face answers 401/403 otherwise). Converting needs python3 >= 3.10 with numpy.
The weights are under the FAIR Noncommercial Research License: research use only.
EOF
}

die() { echo "$PROG: $*" >&2; exit 1; }
need_arg() { [ $# -ge 2 ] && [ -n "$2" ] || { echo "$PROG: $1 needs a value (see --help)" >&2; exit 2; }; }
sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
size_of() { wc -c < "$1" | tr -d ' '; }

while [ $# -gt 0 ]; do
  case $1 in
    --token-file) need_arg "$@"; TOKEN_FILE=$2; shift 2 ;;
    --dir) need_arg "$@"; DIR=$2; shift 2 ;;
    --fp8) FP8=--fp8; shift ;;
    --keep-pt) KEEP=1; shift ;;
    --no-convert) CONVERT=0; KEEP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "$PROG: unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

# tools/convert_weights.py sits next to this script; resolve symlinks (the image links this
# script into /usr/local/bin).
SELF=$0
while [ -L "$SELF" ]; do
  LINK=$(readlink "$SELF")
  case $LINK in /*) SELF=$LINK ;; *) SELF=$(dirname "$SELF")/$LINK ;; esac
done
HERE=$(cd "$(dirname "$SELF")" && pwd)

# --- requirements, checked before a 4.6 GB download ----------------------------
command -v curl >/dev/null || die "curl not found"
PY=$(command -v python3 || true)
[ -n "$PY" ] || die "python3 not found"
if [ $CONVERT = 1 ]; then
  [ -f "$HERE/convert_weights.py" ] || die "convert_weights.py not found next to this script ($HERE)"
  "$PY" -c 'import sys
if sys.version_info < (3, 10):
    sys.exit(1)
import numpy' 2>/dev/null || die "converting needs python3 >= 3.10 with numpy (or pass --no-convert)"
fi

# --- token (never echoed) --------------------------------------------------------
TOKEN=
if [ -n "$TOKEN_FILE" ]; then
  [ -s "$TOKEN_FILE" ] || die "token file $TOKEN_FILE is missing or empty"
  TOKEN=$(tr -d '[:space:]' < "$TOKEN_FILE")
elif [ -n "${HF_TOKEN:-}" ]; then
  TOKEN=$HF_TOKEN
else
  for f in "${HF_HOME:+$HF_HOME/token}" "${HOME:+$HOME/.cache/huggingface/token}"; do
    if [ -n "$f" ] && [ -r "$f" ] && [ -s "$f" ]; then TOKEN=$(tr -d '[:space:]' < "$f"); break; fi
  done
fi
[ -n "$TOKEN" ] || die "no Hugging Face token: pass --token-file PATH, set HF_TOKEN, or run 'hf auth login'"

mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
HDR=$(mktemp "${TMPDIR:-/tmp}/rawatta-hf.XXXXXX")
trap 'rm -f "$HDR"' EXIT
trap 'rm -f "$HDR"; exit 130' INT
trap 'rm -f "$HDR"; exit 143' TERM
printf 'Authorization: Bearer %s\n' "$TOKEN" > "$HDR"
unset TOKEN HF_TOKEN

OUT=$DIR/$FILE
ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
API=$ENDPOINT/api/models/$REPO/paths-info/$REV
URL=$ENDPOINT/$REPO/resolve/$REV/$FILE

# --- expected SHA-256 and size (LFS metadata) ------------------------------------
# The last line of the reply is the HTTP status (000: no reply at all).
REPLY=$(curl -sS -w '\n%{http_code}' -H @"$HDR" -X POST --data-urlencode "paths=$FILE" "$API" 2>/dev/null || true)
CODE=$(printf '%s\n' "$REPLY" | tail -n 1)
META=$(printf '%s\n' "$REPLY" | sed '$d')
case $CODE in
  200) ;;
  401) die "Hugging Face rejected the token (401): it is invalid or expired; create a new read token and save it to the token file" ;;
  403) die "this account has no access to $REPO (403): request access on https://huggingface.co/$REPO and wait for approval (a fine-grained token also needs 'Read access to contents of all public gated repos you can access')" ;;
  000|'') die "could not reach Hugging Face at $ENDPOINT (network error); check the connection or proxy and run again" ;;
  *) die "Hugging Face answered HTTP $CODE when asked for $FILE's metadata at revision $REV" ;;
esac
INFO=$(printf '%s' "$META" | "$PY" -c 'import json, re, sys
try:
    lfs = json.load(sys.stdin)[0]["lfs"]
    oid, size = str(lfs["oid"]), int(lfs["size"])
    if re.fullmatch("[0-9a-f]{64}", oid) and size > 0:
        print(oid, size)
except Exception:
    pass' 2>/dev/null || true)
[ -n "$INFO" ] || die "could not read $FILE's metadata at revision $REV from Hugging Face (unexpected reply)"
SHA=${INFO% *}
SIZE=${INFO#* }
echo "expected sha256 $SHA ($SIZE bytes, revision $REV)"

# --- download (resumable) -----------------------------------------------------------
download() {
  i=0
  until curl -fL --retry 3 -C - -H @"$HDR" -o "$OUT" "$URL"; do
    code=$?
    i=$((i + 1))
    if [ $code -eq 22 ]; then die "HTTP error from Hugging Face (401/403: request access on the model page with this account)"; fi
    [ $i -lt 5 ] || die "download failed (curl exit $code); run again to resume"
    echo "retrying ($i) ..." >&2
    sleep 10
  done
}

COMPLETE=0
if [ -s "$OUT" ]; then
  HAVE=$(size_of "$OUT")
  if [ "$HAVE" -ge "$SIZE" ]; then
    if [ "$HAVE" -eq "$SIZE" ] && [ "$(sha256 "$OUT")" = "$SHA" ]; then
      COMPLETE=1
      echo "$OUT already complete"
    else
      echo "$OUT has full size but fails the checksum: deleting it and downloading again" >&2
      rm -f "$OUT"
    fi
  fi
fi
if [ $COMPLETE = 0 ]; then
  RESUMED=0
  [ -s "$OUT" ] && RESUMED=1
  download
  GOT=$(sha256 "$OUT")
  if [ "$GOT" != "$SHA" ] && [ $RESUMED = 1 ]; then
    echo "checksum mismatch after resuming a partial download: downloading from scratch" >&2
    rm -f "$OUT"
    download
    GOT=$(sha256 "$OUT")
  fi
  if [ "$GOT" != "$SHA" ]; then
    rm -f "$OUT"
    die "checksum mismatch: got $GOT, expected $SHA (deleted $OUT)"
  fi
  echo "downloaded $OUT (sha256 ok)"
fi

# --- convert -------------------------------------------------------------------------
if [ $CONVERT = 1 ]; then
  W=$DIR/$OUTNAME
  # shellcheck disable=SC2086  # $FP8 is empty or one word
  "$PY" "$HERE/convert_weights.py" --input "$OUT" --output "$W" $FP8 --check
  [ $KEEP = 1 ] || { rm -f "$OUT"; echo "removed $OUT (pass --keep-pt to keep it)"; }
  if [ "${RAWATTA_WEIGHTS:-}" = "$W" ]; then
    echo "weights ready: $W (RAWATTA_WEIGHTS points there)"
  elif [ -z "${RAWATTA_WEIGHTS:-}" ] && [ -d "$DEFAULT_DIR" ] && [ "$DIR" = "$(cd "$DEFAULT_DIR" && pwd)" ]; then
    echo "weights ready: $W (rawatta's default path)"
  else
    echo "weights ready: $W (pass --weights $W to rawatta, or set RAWATTA_WEIGHTS to it)"
  fi
fi
