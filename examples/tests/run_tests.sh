#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Checks for examples/ that need no GPU and no weights:
#   1. Python files compile; the unit tests of the PLY subsampler and the gallery index pass.
#   2. Every script parses (bash -n), has an SPDX line and answers --help.
#   3. The scripts run end to end against tests/fake_rawatta, a stand-in binary that writes synthetic outputs
#      and, like rawatta, rejects an option given twice: run_on_photos.sh, run_on_video.sh (native, and extract
#      with exact frame counts and the same frames as rawatta's --video), compare_configs.sh, make_gallery.sh
#      (caption, photo count, scene order, refusals), load_outputs.py, timing_table.py, the fetch.sh argument
#      handling and the Gaussian-splatting dataset layout.
#   4. Docker mode against tests/fake_docker, which plays the rawatta image (ENTRYPOINT rawatta, weights at
#      /weights) and checks the mounts.
# The gallery is written to temporary folders, never to docs/. tests/mutation_check.py proves that each check
# can fail: it breaks the scripts on purpose, one way at a time, and expects this suite to fail each time.
# Needs python3 + numpy; ffmpeg/ffprobe for the image and video fixtures (those steps are skipped without them).
#
# usage: examples/tests/run_tests.sh
set -euo pipefail
ex=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
pass=0
step() { echo; echo "== $*"; }
ok() { pass=$((pass + 1)); echo "ok: $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
# tree_digest <dir>: one line per file (path, size, mtime, SHA-256), or "absent"; to prove a folder was not touched.
tree_digest() {
  python3 - "$1" <<'PY'
import hashlib, os, sys
root = sys.argv[1]
if not os.path.exists(root):
    print("absent")
for d, _, files in sorted(os.walk(root)):
    for f in sorted(files):
        p = os.path.join(d, f)
        st = os.stat(p)
        print(os.path.relpath(p, root), st.st_size, st.st_mtime_ns, hashlib.sha256(open(p, "rb").read()).hexdigest())
PY
}

step "python: compile"
for f in "$ex"/python/*.py "$ex"/tests/*.py "$ex"/tests/fake_rawatta "$ex"/tests/fake_docker; do
  python3 -c 'import sys, py_compile; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)' "$f" "$tmp/c.pyc"
done
ok "py_compile"

step "python: unit tests (subsampler, gallery index)"
for t in "$ex"/python/test_*.py; do
  python3 "$t" >"$tmp/unit.txt" 2>&1 || { cat "$tmp/unit.txt"; fail "unit tests in $(basename "$t")"; }
  tail -3 "$tmp/unit.txt"
done
ok "test_subsample_ply.py, test_gallery_index.py"

step "bash -n"
for f in "$ex"/*.sh "$ex"/*/fetch.sh "$ex"/lib/*.sh "$ex"/tests/*.sh; do bash -n "$f"; done
ok "all scripts parse"

step "SPDX headers and --help"
for f in "$ex"/*.sh "$ex"/*/fetch.sh "$ex"/lib/*.sh "$ex"/tests/*.sh "$ex"/python/*.py "$ex"/tests/*.py \
  "$ex"/tests/fake_rawatta "$ex"/tests/fake_docker; do
  head -3 "$f" | grep -q '^# SPDX-License-Identifier: Apache-2.0$' || fail "no SPDX header: $f"
done
for f in "$ex"/*.sh "$ex"/*/fetch.sh; do
  "$f" --help >"$tmp/help.txt" || fail "$f --help failed"
  grep -q "usage:" "$tmp/help.txt" || fail "$f --help: no usage line"
  if grep -q SPDX "$tmp/help.txt"; then fail "$f --help prints the SPDX line"; fi
done
rc=0; "$ex/buddha/fetch.sh" --bogus 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] || fail "fetch.sh --bogus: rc $rc, expected 2"
for f in "$ex"/python/*.py "$ex"/tests/*.py; do
  case $f in *test_*) continue ;; esac
  python3 "$f" --help >/dev/null || fail "$f --help failed"
done
ok "every file has an SPDX line; every script answers --help"

step "timing_table.py reads rawatta's peak_device_mib, not the whole card"
mkdir -p "$tmp/tt"
cat >"$tmp/tt/timing.json" <<'JSON'
{"config": "rawatta", "frames": 4, "reps": 1, "warmup": 1, "times_ms": [100.0], "median_ms": 100.0, "iqr_ms": 0.0,
 "peak_device_mib": 5000.0, "card_used_peak_mib": 7000.0, "gpu": "test"}
JSON
python3 "$ex/python/timing_table.py" "$tmp/tt" >"$tmp/tt.txt"
grep -q " 5000 " "$tmp/tt.txt" && ! grep -q " 7000 " "$tmp/tt.txt" || { cat "$tmp/tt.txt"; fail "timing_table peak memory"; }
grep -q "fewer than 3 timed passes" "$tmp/tt.txt" || { cat "$tmp/tt.txt"; fail "timing_table single-pass note"; }
grep -Eq "\| 100\.0 +\| - " "$tmp/tt.txt" || { cat "$tmp/tt.txt"; fail "one timed pass must show IQR as -"; }
ok "peak 5000 MiB (engine), IQR '-' and a note for a single timed pass"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found: skipping the end-to-end runs"
  echo "passed $pass checks"
  exit 0
fi

export RAWATTA_RUNNER=binary RAWATTA_BIN=$ex/tests/fake_rawatta
cd "$tmp"
mkdir -p photos
ffmpeg -nostdin -loglevel error -f lavfi -i testsrc=size=320x180:rate=4 -frames:v 6 photos/img_%03d.jpg
ffmpeg -nostdin -loglevel error -f lavfi -i testsrc=size=320x180:rate=25 -t 4 -pix_fmt yuv420p clip.mp4 # 100 frames
ffmpeg -nostdin -loglevel error -f lavfi -i testsrc=size=160x90:rate=25 -frames:v 64 -pix_fmt yuv420p clip64.mp4

step "run_on_photos.sh"
"$ex/run_on_photos.sh" photos runs/photos | tail -12 || fail "run_on_photos.sh failed"
for f in cameras.json extrinsics.npy intrinsics.npy depth.npy conf.npy points.ply colmap/images.txt timing.json; do
  [ -f "runs/photos/$f" ] || fail "missing runs/photos/$f"
done
ok "run_on_photos.sh wrote the full output set"

step "run_on_photos.sh default output name and grammar config"
RAWATTA_CONFIG='dense@sagepp~L4=sage.pvh16/L15=sage.pvh16' "$ex/run_on_photos.sh" photos >/dev/null ||
  fail "run_on_photos.sh with a grammar config failed"
[ -f "runs/photos-dense_sagepp_L4_sage.pvh16_L15_sage.pvh16/timing.json" ] || { ls runs; fail "grammar folder name"; }
ok "config grammar string gives a flat folder name"

step "options the scripts set are never passed twice"
rc=0; "$RAWATTA_BIN" run --images photos --out runs/dup --config exact --config rawatta 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] || fail "fake_rawatta accepted a repeated --config (rc $rc)"
"$ex/run_on_photos.sh" photos runs/own-config --config exact >/dev/null || fail "a --config from the user was passed twice"
grep -q '"config": "exact"' runs/own-config/timing.json || fail "the user's --config exact was not used"
"$ex/run_on_photos.sh" photos -- --config=rawatta-fast --reps 4 >/dev/null || fail "--config=V from the user was passed twice"
grep -q '"config": "rawatta-fast"' runs/photos-rawatta-fast/timing.json || fail "--config=rawatta-fast not used"
grep -q '"reps": 4' runs/photos-rawatta-fast/timing.json || fail "--reps 4 not passed on"
"$ex/run_on_photos.sh" photos runs/no-depth --no-depth >/dev/null || fail "run with --no-depth failed"
[ ! -e runs/no-depth/depth.npy ] && [ -f runs/no-depth/timing.json ] || fail "--no-depth not passed on"
rm -rf runs/photos-rawatta
"$ex/run_on_photos.sh" photos --no-depth >/dev/null || fail "run with --no-depth and no output folder failed"
[ -f runs/photos-rawatta/timing.json ] && [ ! -e runs/photos-rawatta/depth.npy ] && [ ! -e ./--no-depth ] ||
  fail "an option right after the folder was taken as the output folder"
ok "a --config/--reps from the user replaces the script default; the output folder can be left out before options"

step "an option the scripts read, given without a value, stops the script (opt_value)"
# expect_no_value <log> <option> <folder that must not appear> <command...>
expect_no_value() {
  local log=$1 opt=$2 dir=$3 rc=0
  shift 3
  "$@" >"$log" 2>&1 || rc=$?
  [ "$rc" -ne 0 ] && grep -q -- "error: $opt needs a value" "$log" && ! grep -q "multiple times" "$log" &&
    [ ! -e "$dir" ] || { cat "$log"; fail "opt_value: '$*' (rc $rc) went on after '$opt needs a value'"; }
}
expect_no_value nv1.txt --config runs/nv1 "$ex/run_on_photos.sh" photos runs/nv1 --config
expect_no_value nv2.txt --config runs/nv2 "$ex/run_on_photos.sh" photos runs/nv2 --config --no-depth
expect_no_value nv3.txt --config runs/nv3 "$ex/run_on_photos.sh" photos runs/nv3 --config=
expect_no_value nv4.txt --config runs/nv4 "$ex/run_on_video.sh" clip.mp4 runs/nv4 --config
expect_no_value nv5.txt --max-frames runs/nv5 "$ex/run_on_video.sh" clip.mp4 runs/nv5 --max-frames
ok "--config / --max-frames as the last argument, before another option or as --config= stop with 'needs a value'"

step "run_on_photos.sh accepts .webp"
mkdir -p webp
for i in 1 2 3; do cp photos/img_00$i.jpg webp/view_$i.webp; done # the fake does not decode; names are what count
"$ex/run_on_photos.sh" webp runs/webp >/dev/null || fail "run on .webp photos failed"
grep -q '"frames": 3' runs/webp/timing.json || fail ".webp photos not counted"
ok "a folder of .webp photos is accepted"

step "run_on_video.sh (native)"
MAX_FRAMES=8 "$ex/run_on_video.sh" clip.mp4 runs/clip | grep -E "^cameras|^timing" || fail "native: video run failed"
grep -q '"frames": 8' runs/clip/timing.json || fail "native: MAX_FRAMES=8 not passed on"
MAX_FRAMES=8 "$ex/run_on_video.sh" clip.mp4 runs/clip5 --max-frames 5 >/dev/null ||
  fail "native: a --max-frames from the user was passed twice"
grep -q '"frames": 5' runs/clip5/timing.json || fail "native: the user's --max-frames 5 not used"
ok "native video mode (MAX_FRAMES, and a --max-frames from the user wins)"

if command -v ffprobe >/dev/null 2>&1; then
  step "run_on_video.sh (extract): exactly MAX_FRAMES frames, the ones rawatta's --video picks"
  # check_extract <video> <total frames> <MAX_FRAMES> <out>: the frames folder holds exactly the frames of
  # vggt_omega::output::sample_indices(total, n) (computed here independently), and rawatta used all of them.
  check_extract() {
    local video=$1 total=$2 n=$3 out=$4
    MAX_FRAMES=$n VIDEO_MODE=extract "$ex/run_on_video.sh" "$video" "$out" >"$out.log" 2>&1 ||
      { cat "$out.log"; fail "extract: $video with MAX_FRAMES=$n failed"; }
    python3 - "$out" "$total" "$n" <<'PY' || { cat "$out.log"; fail "extract: $video with MAX_FRAMES=$n"; }
import json, os, sys
out, total, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
if total <= n:
    want = list(range(total))
elif n == 1:
    want = [0]
else:
    want = [(i * (total - 1) + (n - 1) // 2) // (n - 1) for i in range(n)]
got = sorted(os.listdir(os.path.join(out, "frames")))
names = ["frame_%06d.jpg" % i for i in want]
assert got == names, "FAIL extract frames: got %d %s..., want %d %s..." % (len(got), got[:4], len(names), names[:4])
frames = json.load(open(os.path.join(out, "timing.json")))["frames"]
assert frames == len(want), "FAIL extract: rawatta ran on %d frames, want %d" % (frames, len(want))
PY
  }
  check_extract clip.mp4 100 32 runs/clip-x    # 100 is not a multiple of 32
  check_extract clip64.mp4 64 48 runs/clip64-x # nor 64 of 48
  check_extract clip64.mp4 64 200 runs/clip64-all
  check_extract clip64.mp4 64 1 runs/clip64-one
  check_extract clip.mp4 100 10 runs/clip-x    # again into the same folder with fewer frames: no stale frames
  # The file named frame_000033.jpg holds video frame 33: the same JPEG ffmpeg makes from that frame alone.
  check_extract clip.mp4 100 4 runs/clip-4     # frames 0, 33, 66, 99
  ffmpeg -nostdin -loglevel error -y -i clip.mp4 -vf 'select=eq(n\,33)' -fps_mode vfr -frames:v 1 -q:v 2 ref33.jpg
  cmp -s ref33.jpg runs/clip-4/frames/frame_000033.jpg || fail "extract: frame_000033.jpg is not frame 33 of the video"
  grep -q "^extracting 4 of the 100 frames" runs/clip-4.log || { cat runs/clip-4.log; fail "extract message"; }
  ok "extract mode: 32 of 100, 48 of 64, all 64 when asked for 200, 1, no stale frames, frame names = video indices"
fi

step "compare_configs.sh"
"$ex/compare_configs.sh" photos runs/cmp | tee cmp.txt | tail -7 || fail "compare_configs.sh failed"
grep -q "speed-up vs exact" cmp.txt || fail "compare_configs: no speed-up column"
grep -q "1.43x" cmp.txt || fail "compare_configs: speed-up of the fake stub (exact 10 ms/frame vs rawatta 7) not 1.43x"
for c in exact rawatta rawatta-fast; do
  grep -q '"reps": 10' "runs/cmp/$c/timing.json" && grep -q '"warmup": 2' "runs/cmp/$c/timing.json" ||
    fail "compare_configs.sh did not pass --reps 10 --warmup 2"
done
grep -q "SYNTHETIC timings" cmp.txt || fail "compare_configs: synthetic timings not flagged"
REPS=3 CONFIGS="exact" "$ex/compare_configs.sh" photos runs/cmp3 -- --warmup 0 >/dev/null ||
  fail "compare_configs.sh with REPS and --warmup failed"
grep -q '"reps": 3' runs/cmp3/exact/timing.json && grep -q '"warmup": 0' runs/cmp3/exact/timing.json ||
  fail "compare_configs: REPS / --warmup not used"
rc=0; "$ex/compare_configs.sh" photos runs/cmpx --config exact >cmpx.txt 2>&1 || rc=$?
[ "$rc" -ne 0 ] && grep -q 'choose the settings with CONFIGS=' cmpx.txt && [ ! -e runs/cmpx ] ||
  { cat cmpx.txt; fail "compare_configs.sh --config was not refused with its own message (rc $rc)"; }
ok "latency table with speed-up column; 10 timed passes after 2 warm-ups by default; --config refused"

step "load_outputs.py"
python3 "$ex/python/load_outputs.py" runs/photos >summary.txt || fail "load_outputs.py failed"
grep -q "cameras: 6 frames" summary.txt || fail "load_outputs summary"
python3 - "$ex/python" runs/photos <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from load_outputs import camera_centres, load_run
run = load_run(sys.argv[2])
assert np.allclose(camera_centres(run["w2c"])[0], 0, atol=1e-6)  # world = frame 0
assert np.allclose(run["w2c"], run["extrinsics"], atol=1e-6)      # cameras.json matches extrinsics.npy
assert run["depth"].shape[0] == 6 and run["points"].shape[1] == 3
PY
if python3 -c 'import matplotlib' >/dev/null 2>&1; then
  python3 "$ex/python/load_outputs.py" runs/photos --plot cams.png >/dev/null && [ -s cams.png ] || fail "load_outputs --plot"
  ok "load_outputs.py summary and plot"
else
  ok "load_outputs.py summary (matplotlib not installed: plot skipped)"
fi

step "COLMAP dataset layout for Gaussian splatting (to_gaussian_splatting.md)"
mkdir -p splat/photos/sparse/0
cp runs/photos/colmap/cameras.txt runs/photos/colmap/images.txt runs/photos/colmap/points3D.txt splat/photos/sparse/0/
cp -r runs/photos/colmap/images splat/photos/images
awk '!/^#/ && NF >= 10 { print $10 }' splat/photos/sparse/0/images.txt >names.txt
[ -s names.txt ] || fail "no image lines in images.txt"
while read -r nm; do
  [ -f "splat/photos/images/$nm" ] || fail "images.txt names $nm, not in colmap/images"
done <names.txt
ok "every image named in images.txt is in colmap/images"

step "Docker mode (tests/fake_docker plays the image: ENTRYPOINT rawatta, weights preset /weights/...)"
mkdir -p fakebin weights weights2
ln -s "$ex/tests/fake_docker" fakebin/docker
: >weights/vggt_omega_1b_512.safetensors
: >weights2/my_weights.safetensors
export FAKE_DOCKER_LOG=$tmp/docker.log
docker_env=(env PATH="$tmp/fakebin:$PATH" RAWATTA_RUNNER=docker RAWATTA_WEIGHTS="$tmp/weights/vggt_omega_1b_512.safetensors")
"${docker_env[@]}" "$ex/run_on_photos.sh" photos runs/dock >dock.txt 2>&1 || { cat dock.txt; fail "docker: photos run failed"; }
"${docker_env[@]}" MAX_FRAMES=8 "$ex/run_on_video.sh" clip.mp4 runs/dockv >dockv.txt 2>&1 ||
  { cat dockv.txt; fail "docker: video run failed"; }
"${docker_env[@]}" RAWATTA_WEIGHTS="$tmp/weights2/my_weights.safetensors" "$ex/run_on_photos.sh" photos runs/dockw \
  >dockw.txt 2>&1 || { cat dockw.txt; fail "docker: run with a weights file of another name failed"; }
"${docker_env[@]}" CONFIGS=exact REPS=1 "$ex/compare_configs.sh" photos runs/dockc >dockc.txt 2>&1 ||
  { cat dockc.txt; fail "docker: compare_configs failed"; }
python3 - "$tmp" "$(id -u):$(id -g)" <<'PY'
import json, os, sys
tmp, user = sys.argv[1], sys.argv[2]
calls = [json.loads(l) for l in open(os.path.join(tmp, "docker.log"))]
assert len(calls) == 5, "FAIL docker: %d docker calls, want 5 (photos, video, other weights, info, exact)" % len(calls)
real = lambda p: os.path.realpath(os.path.join(tmp, p))
for c in calls:
    assert c["image"] == "ghcr.io/sezer-muhammed/rawatta:latest", "FAIL docker image %s" % c["image"]
    assert c["entrypoint"] is None, "FAIL docker: --entrypoint %s; the image's ENTRYPOINT is rawatta" % c["entrypoint"]
    assert c["rm"] and c["gpus"] == "all" and c["user"] == user, "FAIL docker --rm/--gpus all/--user: %s" % c["argv"]
    assert "--weights" not in c["args"], "FAIL docker: --weights passed; the image presets RAWATTA_WEIGHTS"
photos, video, other, info, exact = calls
w = [real("weights"), "/weights", True]
assert photos["mounts"] == [w, [real("photos"), "/work/input", True], [real("runs/dock"), "/work/out", False]], \
    "FAIL docker photo mounts %s" % photos["mounts"]
assert photos["env"] == {}, "FAIL docker: RAWATTA_WEIGHTS set although the file has the image's default name"
assert photos["args"] == ["run", "--images", "/work/input", "--out", "/work/out", "--config", "rawatta"], \
    "FAIL docker photo args %s" % photos["args"]
assert video["mounts"] == [w, [real("clip.mp4"), "/work/input/clip.mp4", True], [real("runs/dockv"), "/work/out", False]], \
    "FAIL docker video mounts (the video file only, never its folder): %s" % video["mounts"]
assert video["args"][:3] == ["run", "--video", "/work/input/clip.mp4"], "FAIL docker video args %s" % video["args"]
assert other["mounts"][0] == [real("weights2"), "/weights", True] and \
    other["env"] == {"RAWATTA_WEIGHTS": "/weights/my_weights.safetensors"}, "FAIL docker other weights %s" % other
assert info["args"] == ["info"], "FAIL docker info call %s" % info["args"]
assert exact["args"] == ["run", "--images", "/work/input", "--out", "/work/out", "--config", "exact", "--warmup", "2",
                        "--reps", "1"], "FAIL docker compare_configs args %s" % exact["args"]
for d, n in (("runs/dock", 6), ("runs/dockv", 8), ("runs/dockw", 6), ("runs/dockc/exact", 6)):
    t = json.load(open(os.path.join(tmp, d, "timing.json")))
    assert t["frames"] == n, "FAIL docker: %s has %d frames, want %d" % (d, t["frames"], n)
PY
n_calls=$(wc -l <docker.log | tr -d ' ')
rc=0; "${docker_env[@]}" RAWATTA_WEIGHTS="$tmp/none/vggt_omega_1b_512.safetensors" "$ex/run_on_photos.sh" photos runs/dockn \
  >dockn.txt 2>&1 || rc=$?
[ "$rc" -ne 0 ] && grep -q "weights not found at $tmp/none/" dockn.txt && grep -q -- "--entrypoint download-weights" dockn.txt &&
  [ "$(wc -l <docker.log | tr -d ' ')" = "$n_calls" ] ||
  { cat dockn.txt; fail "docker: missing weights must stop before docker runs, with the download command (rc $rc)"; }
rc=0; "${docker_env[@]}" "$ex/run_on_photos.sh" photos runs/dockx --weights /host/w.safetensors >dockx.txt 2>&1 || rc=$?
[ "$rc" -ne 0 ] && grep -q "instead of passing --weights" dockx.txt && [ "$(wc -l <docker.log | tr -d ' ')" = "$n_calls" ] ||
  { cat dockx.txt; fail "docker: a host --weights path must be refused (rc $rc)"; }
unset FAKE_DOCKER_LOG
ok "canonical image, no --entrypoint, --gpus all, --user, weights folder at /weights:ro (preset name, or RAWATTA_WEIGHTS),\
 photo folder / single video file read-only, output folder; missing weights and host --weights refused"

step "make_gallery.sh (fake runs, temporary output)"
for s in "$ex"/*/scene.json; do
  id=$(basename "$(dirname "$s")")
  mkdir -p "data/$id"
  cp -r photos "data/$id/images"
  # make_gallery takes the thumbnail named in scene.json
  thumb=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["thumb"])' "$s")
  cp photos/img_001.jpg "data/$id/images/$thumb"
done
export EXAMPLES_DATA=$tmp/data GALLERY_RUNS=$tmp/gallery-runs
rc=0; GALLERY_OUT=$tmp/site-refused "$ex/make_gallery.sh" >gal-refused.txt 2>&1 || rc=$?
[ "$rc" -ne 0 ] && grep -q "marked synthetic" gal-refused.txt && [ ! -e "$tmp/site-refused" ] ||
  { cat gal-refused.txt; fail "synthetic timings were not refused before anything was written (rc $rc)"; }
rc=0; GALLERY_OUT=$tmp/site-x GALLERY_ARGS="--config exact" "$ex/make_gallery.sh" >gal-cfg.txt 2>&1 || rc=$?
[ "$rc" -ne 0 ] && grep -q "set the setting with GALLERY_CONFIG" gal-cfg.txt ||
  { cat gal-cfg.txt; fail "make_gallery.sh: --config in GALLERY_ARGS not refused with its own message (rc $rc)"; }
rc=0; GALLERY_OUT=$tmp/site-x GALLERY_ARGS="--max-frames 3" "$ex/make_gallery.sh" >gal-mf.txt 2>&1 || rc=$?
[ "$rc" -ne 0 ] && grep -q "no --max-frames in GALLERY_ARGS" gal-mf.txt && [ ! -e "$tmp/site-x" ] ||
  { cat gal-mf.txt; fail "make_gallery.sh: --max-frames in GALLERY_ARGS not refused (rc $rc)"; }
ok "synthetic timings refused by default; no --config or --max-frames in GALLERY_ARGS"

step "make_gallery.sh never writes synthetic timings to the default docs/ folder"
# A copy of examples/ in a repository layout of its own, so that a broken guard cannot touch the real docs/.
mkdir -p repo/docs/static/examples
cp -R "$ex" repo/examples
echo '{"scenes": []}' >repo/docs/static/examples/index.json
before=$(tree_digest repo/docs)
for out in "" "$tmp/repo/docs/static/../static/examples/"; do
  rc=0; GALLERY_OUT=$out GALLERY_ALLOW_SYNTHETIC=1 GALLERY_SKIP_RUN=1 repo/examples/make_gallery.sh >gal-docs.txt 2>&1 || rc=$?
  [ "$rc" -ne 0 ] && grep -q "never go to the page" gal-docs.txt ||
    { cat gal-docs.txt; fail "GALLERY_ALLOW_SYNTHETIC=1 with GALLERY_OUT='$out' (the default docs/) was not refused"; }
done
[ "$(tree_digest repo/docs)" = "$before" ] || fail "the refused run changed files under docs/"
ok "refused, and docs/ is byte-for-byte unchanged (also when GALLERY_OUT spells the default path differently)"

step "make_gallery.sh: caption, photo count from the folder, files"
GALLERY_ALLOW_SYNTHETIC=1 GALLERY_OUT=$tmp/site GALLERY_ARGS="--conf-percentile 60" \
  "$ex/make_gallery.sh" >gal.txt 2>&1 || { cat gal.txt; fail "make_gallery.sh (fake runs) failed"; }
grep -v "^-" gal.txt | tail -8
python3 - "$tmp/site" "$ex" "$tmp/gallery-runs" "$tmp/data" <<'PY'
import json, os, sys
site, ex, runs, data = sys.argv[1:5]
sys.path.insert(0, os.path.join(ex, "python"))
from subsample_ply import read_ply
idx = json.load(open(os.path.join(site, "index.json"), encoding="utf-8"))
ids = [e["id"] for e in idx["scenes"]]
order = [l.split("#")[0].strip() for l in open(os.path.join(ex, "scenes.txt")) if l.split("#")[0].strip()]
assert ids == order, "FAIL gallery order %s, want scenes.txt order %s" % (ids, order)
for e in idx["scenes"]:
    for k in ("id", "title", "caption", "dir", "points", "cameras", "licence", "licence_url", "source", "photos",
              "frames", "config", "latency_ms", "latency_iqr_ms", "reps", "warmup", "gpu", "point_count"):
        assert e.get(k) not in (None, ""), (e["id"], k)
    scene = json.load(open(os.path.join(ex, e["id"], "scene.json")))
    photos = len([f for f in os.listdir(os.path.join(data, e["id"], "images")) if f.endswith(".jpg")])
    c = e["caption"]
    # What the viewer reads: <dir>/<points> and <dir>/<cameras> relative to index.json, and the caption.
    assert e["dir"] == e["id"] and e["points"] == "points.ply" and e["cameras"] == "cameras.json", e
    assert c.startswith("Cameras and point cloud predicted by VGGT-Ω from %d photos, accelerated with RAWATTA "
                        "(--config rawatta)." % photos), "FAIL caption model/setting/count: " + c
    assert scene["credit"] in c, "FAIL caption: credit missing: " + c
    assert "%s (%s)" % (scene["licence"], scene["licence_url"]) in c, "FAIL caption: licence URL missing: " + c
    assert "%d of %d photos" % (photos, scene["source_photos"]) in c, "FAIL caption: photo count not from the folder: " + c
    assert c.endswith("Outputs: noncommercial research use only (FAIR Noncommercial Research License)."), \
        "FAIL caption: the noncommercial note on the outputs is missing: " + c
    assert ("%.1f ms" % e["latency_ms"]) in c and "median of 10 timed runs" in c, c
    assert e["synthetic"] is True and "SYNTHETIC" in c, e
    assert (e["reps"], e["warmup"]) == (10, 2), (e["reps"], e["warmup"])
    t = json.load(open(os.path.join(runs, e["id"], "timing.json")))
    assert (t["reps"], t["warmup"]) == (10, 2), "make_gallery.sh did not pass --reps 10 --warmup 2"
    d = os.path.join(site, e["dir"])
    for f in (e["points"], e["cameras"], e["thumb"]):
        assert os.path.getsize(os.path.join(d, f)) > 0, f
    size = os.path.getsize(os.path.join(d, e["points"]))
    xyz, rgb = read_ply(os.path.join(d, e["points"]))
    assert len(xyz) == e["point_count"] <= 300_000 and size <= 8_000_000 and rgb is not None, (len(xyz), size)
    print("  %s: %d points, %.2f MB; caption: %s" % (e["id"], len(xyz), size / 1e6, c))
PY
ok "index.json: viewer fields; caption names VGGT-Omega and RAWATTA, counts the folder's photos, has the licence URL"

step "make_gallery.sh: a run that did not use every photo of the folder is refused before anything is written"
cp photos/img_002.jpg data/buddha/images/extra.jpg
before=$(tree_digest site)
rc=0; GALLERY_ALLOW_SYNTHETIC=1 GALLERY_SKIP_RUN=1 GALLERY_OUT=$tmp/site "$ex/make_gallery.sh" buddha >gal-n.txt 2>&1 || rc=$?
[ "$rc" -ne 0 ] && grep -q "holds 8 photos" gal-n.txt && [ "$(tree_digest site)" = "$before" ] ||
  { cat gal-n.txt; fail "a stale run (7 frames) for a folder of 8 photos was not refused cleanly (rc $rc)"; }
GALLERY_ALLOW_SYNTHETIC=1 GALLERY_OUT=$tmp/site "$ex/make_gallery.sh" buddha >/dev/null 2>&1 || fail "rerun of buddha failed"
python3 -c 'import json, sys
e = [s for s in json.load(open(sys.argv[1], encoding="utf-8"))["scenes"] if s["id"] == "buddha"][0]
assert " from 8 photos," in e["caption"] and "8 of 67 photos" in e["caption"], "FAIL caption count: " + e["caption"]' \
  site/index.json
rm data/buddha/images/extra.jpg
ok "refused with the photo count, gallery untouched; after a new run the caption says 8 photos"

step "make_gallery.sh: scene order stays that of scenes.txt whatever the build order"
mkdir -p site-order
echo '{"scenes": [{"id": "aaa-extra", "title": "kept"}]}' >site-order/index.json
for s in buddha korno-rock buddha; do
  GALLERY_ALLOW_SYNTHETIC=1 GALLERY_OUT=$tmp/site-order "$ex/make_gallery.sh" "$s" >/dev/null 2>&1 ||
    fail "make_gallery.sh $s failed"
done
order=$(python3 -c 'import json, sys; print(" ".join(s["id"] for s in json.load(open(sys.argv[1]))["scenes"]))' site-order/index.json)
[ "$order" = "buddha korno-rock aaa-extra" ] || fail "scene order after building buddha, korno-rock, buddha: '$order'"
ok "buddha, korno-rock (scenes.txt), then other scenes already in index.json"

echo
echo "passed $pass checks"
