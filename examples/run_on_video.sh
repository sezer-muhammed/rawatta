#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reconstruct cameras, depth maps and a point cloud from a video, using frames spread evenly over it.
#
# usage: examples/run_on_video.sh <video-file> [out-folder] [--] [more rawatta run arguments...]
#   out-folder defaults to runs/<video name>-<config>.
#   The extra arguments go to `rawatta run` unchanged. A --config or --max-frames among them replaces
#   RAWATTA_CONFIG or MAX_FRAMES (the script then does not add its own: rawatta rejects an option given twice).
#
# env:   RAWATTA_CONFIG      rawatta (default) | rawatta-fast | exact | a config grammar string
#        MAX_FRAMES          number of frames taken from the video (default 32; more frames need more GPU memory)
#        VIDEO_MODE          native (default): rawatta reads the video itself (--video --max-frames)
#                            extract: ffmpeg first saves the frames as JPEGs in <out-folder>/frames/, then
#                            rawatta runs on that folder (useful to look at or edit the frames). It saves
#                            exactly MAX_FRAMES frames (every frame if the video has fewer), the same ones
#                            rawatta's --video picks: evenly spaced, first and last included, named
#                            frame_<index in the video>.jpg
#        RAWATTA_RUNNER, RAWATTA_BIN, RAWATTA_IMAGE, RAWATTA_GPUS, RAWATTA_WEIGHTS: see examples/lib/rawatta.sh
#
# examples:
#   examples/run_on_video.sh walkaround.mp4
#   MAX_FRAMES=48 RAWATTA_CONFIG=rawatta-fast examples/run_on_video.sh walkaround.mp4 runs/walk-fast
#   VIDEO_MODE=extract examples/run_on_video.sh walkaround.mp4
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR source=lib/rawatta.sh
. "$here/lib/rawatta.sh"

if [ $# -lt 1 ] || [ "$1" = -h ] || [ "$1" = --help ]; then
  usage
  exit 0
fi
video=$1
shift
[ -f "$video" ] || die "not a file: $video"
out=
if [ $# -gt 0 ]; then case $1 in -*) ;; *) out=$1; shift ;; esac; fi
if [ $# -gt 0 ] && [ "$1" = -- ]; then shift; fi
config=${RAWATTA_CONFIG:-rawatta}
if opt_value --config "$@"; then config=$OPT_VALUE; else set -- --config "$config" "$@"; fi
max_frames=${MAX_FRAMES:-32}
user_max=0
if opt_value --max-frames "$@"; then max_frames=$OPT_VALUE; user_max=1; fi
case $max_frames in '' | *[!0-9]* | 0) die "the frame count must be a positive integer (got '$max_frames')" ;; esac
name=$(basename "$video")
name=${name%.*}
[ -n "$out" ] || out=runs/$name-$(safe_name "$config")
runner=$(rawatta_runner)

# video_frames <packets|frames>: number of frames in the first video stream, from the packet count (fast,
# no decoding) or from the decoded frames (exact, decodes the whole video).
video_frames() {
  local entry=nb_read_packets flag=-count_packets
  if [ "$1" = frames ]; then entry=nb_read_frames; flag=-count_frames; fi
  ffprobe -v error -select_streams v:0 "$flag" -show_entries "stream=$entry" -of csv=p=0 "$video" | tr -dc '0-9'
}

# sample_indices <total> <n>: the frame indices rawatta's --video keeps (vggt_omega::output::sample_indices):
# every frame when total <= n, else n indices spread evenly from 0 to total-1, rounded to the nearest.
sample_indices() {
  local total=$1 n=$2 i
  if [ "$total" -le "$n" ]; then
    for ((i = 0; i < total; i++)); do echo "$i"; done
  elif [ "$n" -eq 1 ]; then
    echo 0
  else
    for ((i = 0; i < n; i++)); do echo $(((i * (total - 1) + (n - 1) / 2) / (n - 1))); done
  fi
}

# extract_frames <total>: save the frames sample_indices picks as <frames_dir>/frame_<index>.jpg and set
# got to the number of frames ffmpeg wrote.
extract_frames() {
  local total=$1 expr='' i f tmpd=$out/.frames-tmp keep=()
  got=0
  while read -r i; do keep+=("$i"); done < <(sample_indices "$total" "$max_frames")
  for i in "${keep[@]}"; do expr+="${expr:++}eq(n\\,$i)"; done
  rm -rf "$tmpd"
  mkdir -p "$tmpd"
  # Variable-frame-rate output (one image per selected frame) is -fps_mode vfr from FFmpeg 5.1 on and
  # -vsync vfr before.
  if ! ffmpeg -nostdin -loglevel error -y -i "$video" -vf "select=$expr" -fps_mode vfr -q:v 2 \
    "$tmpd/%06d.jpg" 2>"$tmpd/err.txt"; then
    if grep -q fps_mode "$tmpd/err.txt"; then
      ffmpeg -nostdin -loglevel error -y -i "$video" -vf "select=$expr" -vsync vfr -q:v 2 "$tmpd/%06d.jpg" ||
        die "ffmpeg could not read $video"
    else
      cat "$tmpd/err.txt" >&2
      die "ffmpeg could not read $video"
    fi
  fi
  for f in "$tmpd"/*.jpg; do
    [ -e "$f" ] || break
    if [ "$got" -lt ${#keep[@]} ]; then mv "$f" "$(printf '%s/frame_%06d.jpg' "$frames_dir" "${keep[$got]}")"; fi
    got=$((got + 1))
  done
  rm -rf "$tmpd"
}

case ${VIDEO_MODE:-native} in
  native)
    echo "rawatta: $video (up to $max_frames frames), config $config, runner $runner -> $out"
    if [ $user_max = 0 ]; then set -- --max-frames "$max_frames" "$@"; fi
    rawatta_run video "$video" "$out" "$@"
    ;;
  extract)
    command -v ffmpeg >/dev/null 2>&1 || die "VIDEO_MODE=extract needs ffmpeg"
    command -v ffprobe >/dev/null 2>&1 || die "VIDEO_MODE=extract needs ffprobe"
    frames_dir=$out/frames
    mkdir -p "$frames_dir"
    rm -f "$frames_dir"/frame_*.jpg # frames of an earlier run with another MAX_FRAMES
    total=$(video_frames packets || true)
    [ -n "$total" ] && [ "$total" -gt 0 ] || die "could not count the frames of $video"
    want=$((total < max_frames ? total : max_frames))
    echo "extracting $want of the $total frames (evenly spaced, first and last included) into $frames_dir"
    extract_frames "$total"
    if [ "$got" -ne "$want" ]; then
      # The packet count can differ from the decodable frames: count those (slower) and extract again.
      total=$(video_frames frames || true)
      [ -n "$total" ] && [ "$total" -gt 0 ] || die "could not count the frames of $video"
      want=$((total < max_frames ? total : max_frames))
      echo "ffmpeg wrote $got frames; the video decodes to $total frames, extracting $want again"
      rm -f "$frames_dir"/frame_*.jpg
      extract_frames "$total"
    fi
    [ "$got" -eq "$want" ] || die "ffmpeg wrote $got frames of $video, expected $want"
    n=$(count_images "$frames_dir")
    [ "$n" -eq "$want" ] || die "$frames_dir holds $n images, expected the $want extracted frames (remove other images)"
    echo "rawatta: $n frames, config $config, runner $runner -> $out"
    rawatta_run images "$frames_dir" "$out" "$@"
    ;;
  *) die "VIDEO_MODE must be native or extract" ;;
esac

echo
if have_numpy; then
  python3 "$here/python/load_outputs.py" "$out"
else
  ls -l "$out"
fi
