# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# Shared download code for the example scenes (sourced by examples/<scene>/fetch.sh).
#
# Each scene folder has a manifest.tsv with one row per frame:
#   <frame> <original file name> <bytes> <sha256 of the original> <url>
# fetch_scene downloads every original, checks its size and SHA-256 (a mismatch is a hard error),
# downscales it so the long side is at most 1024 px and saves it as <dest>/images/<frame>.jpg.
#
# Environment:
#   LONG_SIDE=1024        long side of the saved images in pixels
#   JPEG_QUALITY=92       JPEG quality of the saved images
#   KEEP_ORIGINALS=0      1 keeps the full-size downloads in <dest>/originals
#   RESIZER=auto          pillow | magick | ffmpeg (auto picks the first one available, in that order)

FETCH_UA="rawatta-examples/1.0 (+https://github.com/sezer-muhammed/rawatta)"

_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

_filesize() { wc -c <"$1" | tr -d ' '; }

_pick_resizer() {
  local want=${RESIZER:-auto}
  if [ "$want" = auto ] || [ "$want" = pillow ]; then
    if python3 -c 'import PIL' >/dev/null 2>&1; then echo pillow; return; fi
  fi
  if [ "$want" = auto ] || [ "$want" = magick ]; then
    if command -v magick >/dev/null 2>&1; then echo magick; return; fi
    if command -v convert >/dev/null 2>&1; then echo convert; return; fi
  fi
  if [ "$want" = auto ] || [ "$want" = ffmpeg ]; then
    if command -v ffmpeg >/dev/null 2>&1; then echo ffmpeg; return; fi
  fi
  echo "error: no image resizer found (install Pillow: pip install pillow, or ImageMagick, or ffmpeg)" >&2
  return 1
}

# _resize <tool> <src> <dst.jpg>: fit into LONG_SIDE x LONG_SIDE (never upscale), honour EXIF rotation.
_resize() {
  local tool=$1 src=$2 dst=$3 side=${LONG_SIDE:-1024} q=${JPEG_QUALITY:-92}
  case $tool in
    pillow)
      python3 - "$src" "$dst" "$side" "$q" <<'PY'
import sys
from PIL import Image, ImageOps
src, dst, side, q = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
s = side / max(im.size)
if s < 1:
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
im.save(dst, "JPEG", quality=q, optimize=True)
PY
      ;;
    magick | convert)
      "$tool" "$src" -auto-orient -resize "${side}x${side}>" -strip -quality "$q" "$dst" ;;
    ffmpeg)
      # ffmpeg ignores EXIF orientation; the example originals are all stored upright.
      ffmpeg -nostdin -loglevel error -y -i "$src" \
        -vf "scale=w=${side}:h=${side}:force_original_aspect_ratio=decrease:flags=lanczos" \
        -q:v 2 "$dst" ;;
  esac
}

# fetch_scene <scene dir containing manifest.tsv> <destination dir>
fetch_scene() {
  local scene_dir=$1 dest=$2 manifest=$1/manifest.tsv tool
  [ -f "$manifest" ] || { echo "error: $manifest not found" >&2; return 1; }
  command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; return 1; }
  tool=$(_pick_resizer) || return 1
  mkdir -p "$dest/originals" "$dest/images"
  echo "fetching $(basename "$scene_dir") into $dest (resizer: $tool, long side ${LONG_SIDE:-1024} px)"

  local frame file bytes sha url n=0 have
  while IFS=$'\t' read -r frame file bytes sha url; do
    case $frame in '' | '#'*) continue ;; esac
    local orig=$dest/originals/$file
    if [ -f "$orig" ] && [ "$(_sha256 "$orig")" = "$sha" ]; then
      :
    else
      curl -fL --retry 4 --retry-delay 5 -sS -A "$FETCH_UA" -o "$orig.part" "$url"
      have=$(_filesize "$orig.part")
      if [ "$have" != "$bytes" ]; then
        echo "error: $file: got $have bytes, expected $bytes" >&2; rm -f "$orig.part"; return 1
      fi
      have=$(_sha256 "$orig.part")
      if [ "$have" != "$sha" ]; then
        echo "error: $file: sha256 $have, expected $sha" >&2; rm -f "$orig.part"; return 1
      fi
      mv "$orig.part" "$orig"
    fi
    _resize "$tool" "$orig" "$dest/images/$frame.jpg"
    n=$((n + 1))
    printf '  %s <- %s\n' "$frame" "$file"
  done <"$manifest"

  (cd "$dest/images" && for f in *.jpg; do printf '%s  %s\n' "$(_sha256 "$f")" "$f"; done) >"$dest/SHA256SUMS.images"
  if [ -f "$scene_dir/images.sha256" ]; then
    if cmp -s "$scene_dir/images.sha256" "$dest/SHA256SUMS.images"; then
      echo "resized images are byte-identical to the reference ($scene_dir/images.sha256)"
    else
      echo "note: resized images differ from the reference sums; that is expected with another resizer or"
      echo "      library version (the downloaded originals were verified, the reference used Pillow)"
    fi
  fi
  if [ "${KEEP_ORIGINALS:-0}" != 1 ]; then rm -rf "$dest/originals"; fi
  echo "done: $n images in $dest/images"
}

# fetch_main <scene dir> <default destination> [args of fetch.sh]: argument handling shared by the
# examples/<scene>/fetch.sh scripts (-h/--help prints their header; one optional destination folder).
fetch_main() {
  local scene_dir=$1 default_dest=$2
  shift 2
  case ${1:-} in
    -h | --help)
      awk 'NR > 1 && /^# SPDX-License-Identifier:/ { next } NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
      return 0 ;;
    -*) echo "error: unknown option '$1' (see --help)" >&2; return 2 ;;
  esac
  [ $# -le 1 ] || { echo "error: expected at most one destination folder (see --help)" >&2; return 2; }
  fetch_scene "$scene_dir" "${1:-$default_dest}"
}
