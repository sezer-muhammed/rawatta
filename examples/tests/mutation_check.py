#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove that examples/tests/run_tests.sh can fail: break the example scripts on purpose, one way at a time,
and check that the suite then fails, and fails at the check written for that break.

Each mutation copies examples/ into a temporary folder, replaces one exact piece of text in one file (it
must occur exactly once, otherwise the mutation is reported as NOT APPLIED), runs the copy's
tests/run_tests.sh, and expects a non-zero exit with the given text in the output. The unmodified copy
must pass first. Results: caught, WRONG REASON (the suite failed elsewhere), SURVIVED (the suite passed).

  python3 examples/tests/mutation_check.py            # all mutations, 4 at a time
  python3 examples/tests/mutation_check.py -j 8 --only opt-subshell docs-guard
  python3 examples/tests/mutation_check.py --list

Needs what run_tests.sh needs (python3 + numpy, ffmpeg/ffprobe). Takes a few minutes. Exit 0 only if every
mutation is caught for the right reason.
"""

import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import tempfile

EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (id, file under examples/, exact text, replacement, text the failing suite must print)
MUTATIONS = [
    # opt_value: an option the scripts read, given without a value
    ("opt-subshell", "run_on_photos.sh",
     'if opt_value --config "$@"; then config=$OPT_VALUE;',
     'if OPT_VALUE=$(opt_value --config "$@" && echo "$OPT_VALUE"); then config=$OPT_VALUE;',
     "FAIL: opt_value"),
    ("opt-no-check", "lib/rawatta.sh",
     """case ${2-} in '' | --*) die "$o needs a value" ;; esac""",
     ': "${2-}"',
     "FAIL: opt_value"),
    # options the scripts set must reach rawatta once
    ("config-twice", "run_on_photos.sh",
     'if opt_value --config "$@"; then config=$OPT_VALUE; else set -- --config "$config" "$@"; fi',
     'if opt_value --config "$@"; then config=$OPT_VALUE; fi; set -- --config "$config" "$@"',
     "FAIL: a --config from the user was passed twice"),
    ("maxframes-twice", "run_on_video.sh",
     'if [ $user_max = 0 ]; then set -- --max-frames "$max_frames" "$@"; fi',
     'set -- --max-frames "$max_frames" "$@"',
     "FAIL: native: a --max-frames from the user was passed twice"),
    ("fake-repeats", "tests/fake_rawatta",
     "    reject_repeated_options(sys.argv[1:])",
     "    pass",
     "FAIL: fake_rawatta accepted a repeated --config"),
    ("compare-config", "compare_configs.sh",
     'has_opt --config "$@" && die "choose the settings with CONFIGS=\\"exact rawatta ...\\", not --config"',
     ":",
     "FAIL: compare_configs.sh --config was not refused"),
    ("compare-reps", "compare_configs.sh",
     'has_opt --reps "$@" || set -- --reps "${REPS:-10}" "$@"',
     ":",
     "FAIL: compare_configs.sh did not pass --reps 10 --warmup 2"),
    # run_on_video.sh VIDEO_MODE=extract
    ("extract-count", "run_on_video.sh",
     "for ((i = 0; i < n; i++)); do echo $(((i * (total - 1)",
     "for ((i = 0; i < n - 1; i++)); do echo $(((i * (total - 1)",
     "FAIL: extract: clip.mp4 with MAX_FRAMES=32"),
    ("extract-rounding", "run_on_video.sh",
     "(n - 1) / 2) / (n - 1)",
     "0) / (n - 1)",
     "FAIL extract frames"),
    ("extract-stale", "run_on_video.sh",
     'rm -f "$frames_dir"/frame_*.jpg # frames of an earlier run with another MAX_FRAMES',
     ": # no clean-up",
     "FAIL: extract: clip.mp4 with MAX_FRAMES=10 failed"),
    ("extract-content", "run_on_video.sh",
     """mv "$f" "$(printf '%s/frame_%06d.jpg' "$frames_dir" "${keep[$got]}")\"""",
     """mv "$f" "$(printf '%s/frame_%06d.jpg' "$frames_dir" "${keep[$((${#keep[@]} - 1 - got))]}")\"""",
     "FAIL: extract: frame_000033.jpg is not frame 33"),
    # Docker mode
    ("docker-image", "lib/rawatta.sh",
     "RAWATTA_IMAGE=${RAWATTA_IMAGE:-ghcr.io/sezer-muhammed/rawatta:latest}",
     "RAWATTA_IMAGE=${RAWATTA_IMAGE:-rawatta:latest}",
     "FAIL docker image"),
    ("docker-entrypoint", "lib/rawatta.sh",
     'DOCKER_ARGS=(--rm --gpus "${RAWATTA_GPUS:-all}" --user "$(id -u):$(id -g)")',
     'DOCKER_ARGS=(--rm --gpus "${RAWATTA_GPUS:-all}" --user "$(id -u):$(id -g)" --entrypoint /usr/local/bin/rawatta)',
     "FAIL: docker: photos run failed"),
    ("docker-user", "lib/rawatta.sh",
     'DOCKER_ARGS=(--rm --gpus "${RAWATTA_GPUS:-all}" --user "$(id -u):$(id -g)")',
     'DOCKER_ARGS=(--rm --gpus "${RAWATTA_GPUS:-all}")',
     "FAIL docker --rm/--gpus all/--user"),
    ("docker-video-folder", "lib/rawatta.sh",
     'DOCKER_ARGS+=(-v "$in_abs:$target:ro" -v "$out_abs:/work/out")',
     'if [ "$kind" = video ]; then DOCKER_ARGS+=(-v "$(dirname "$in_abs"):/work/input:ro"); '
     'else DOCKER_ARGS+=(-v "$in_abs:$target:ro"); fi; DOCKER_ARGS+=(-v "$out_abs:/work/out")',
     "FAIL docker video mounts"),
    ("docker-weights-mount", "lib/rawatta.sh",
     'DOCKER_ARGS+=(-v "$(dirname "$w"):/weights:ro")',
     'DOCKER_ARGS+=(-v "$(dirname "$w"):/models:ro")',
     "FAIL: docker: photos run failed"),
    ("docker-weights-name", "lib/rawatta.sh",
     'if [ "$(basename "$w")" != "$RAWATTA_WEIGHTS_FILE" ]; then',
     "if false; then",
     "FAIL: docker: run with a weights file of another name failed"),
    ("docker-weights-missing", "lib/rawatta.sh",
     'elif [ "$need" = 1 ]; then',
     "elif false; then",
     "FAIL: docker: missing weights must stop"),
    ("docker-host-weights", "lib/rawatta.sh",
     'if has_opt --weights "$@"; then',
     "if false; then",
     "FAIL: docker: a host --weights path must be refused"),
    # make_gallery.sh
    ("gallery-synthetic", "make_gallery.sh",
     """if [ "${GALLERY_ALLOW_SYNTHETIC:-0}" != 1 ] && python3 -c 'import json, sys""",
     """if false && python3 -c 'import json, sys""",
     "FAIL: synthetic timings were not refused before anything was written"),
    ("gallery-config", "make_gallery.sh",
     'has_opt --config ${run_args[@]+"${run_args[@]}"} && die "set the setting with GALLERY_CONFIG, not --config in GALLERY_ARGS"',
     ":",
     "FAIL: make_gallery.sh: --config in GALLERY_ARGS"),
    ("gallery-maxframes", "make_gallery.sh",
     'has_opt --max-frames ${run_args[@]+"${run_args[@]}"} &&',
     "false &&",
     "FAIL: make_gallery.sh: --max-frames in GALLERY_ARGS"),
    ("docs-guard-off", "make_gallery.sh",
     'if [ "$(realpath_py "$site")" = "$(realpath_py "$default_site")" ]; then',
     "if false; then",
     "FAIL: GALLERY_ALLOW_SYNTHETIC=1 with GALLERY_OUT=''"),
    ("docs-guard-spelling", "make_gallery.sh",
     'if [ "$(realpath_py "$site")" = "$(realpath_py "$default_site")" ]; then',
     'if [ "$site" = "$default_site" ]; then',
     "FAIL: GALLERY_ALLOW_SYNTHETIC=1 with GALLERY_OUT='"),
    ("gallery-frames", "make_gallery.sh",
     '[ "$frames" = "$n" ] ||',
     "true ||",
     "FAIL: a stale run (7 frames)"),
    # gallery_index.py
    ("index-frames", "python/gallery_index.py",
     "    if frames != photos:",
     "    if False:",
     "test_run_that_did_not_use_every_photo_is_refused"),
    ("index-order", "python/gallery_index.py",
     "scenes = order_scenes(scenes, read_order(a.order_file))",
     "scenes = scenes",
     "test_order_follows_the_list_not_the_build_order"),
    ("index-licence-url", "python/gallery_index.py",
     'lic = s["licence"] + (" (%s)" % s["licence_url"] if s.get("licence_url") else "")',
     'lic = s["licence"]',
     "test_caption_names_model_setting_licence_url_and_folder_count"),
    ("index-model", "python/gallery_index.py",
     r'MODEL = "VGGT-Ω"',
     'MODEL = "RAWATTA"',
     "test_caption_names_model_setting_licence_url_and_folder_count"),
    ("index-fetch-count", "python/gallery_index.py",
     'used = "%d of %d photos" % (photos, s["source_photos"])',
     'used = "%d of %d photos" % (s["fetched_photos"], s["source_photos"])',
     "test_caption_names_model_setting_licence_url_and_folder_count"),
]


def copy_examples(dest):
    shutil.copytree(EX, os.path.join(dest, "examples"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv"), symlinks=True)
    return os.path.join(dest, "examples")


def run_suite(ex, timeout):
    try:
        p = subprocess.run(["bash", os.path.join(ex, "tests", "run_tests.sh")], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return 124, out + "\n(timed out)"


def mutate(m, timeout):
    mid, rel, old, new, expect = m
    root = tempfile.mkdtemp(prefix="mut-%s-" % mid)
    try:
        ex = copy_examples(root)
        path = os.path.join(ex, rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if text.count(old) != 1:
            return mid, "NOT APPLIED", "text occurs %d times in %s" % (text.count(old), rel)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new))
        rc, out = run_suite(ex, timeout)
        if rc == 0:
            return mid, "SURVIVED", out.strip().splitlines()[-1] if out.strip() else ""
        fails = [ln for ln in out.splitlines() if "FAIL" in ln or "Error" in ln]
        detail = (fails[0] if fails else out.strip().splitlines()[-1] if out.strip() else "").strip()
        if expect not in out:
            return mid, "WRONG REASON", "expected %r; got: %s" % (expect, detail[:200])
        return mid, "caught", detail[:160]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--jobs", type=int, default=4, help="mutations run in parallel (default 4)")
    ap.add_argument("--only", nargs="+", metavar="ID", help="run only these mutations")
    ap.add_argument("--list", action="store_true", help="list the mutations and exit")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per suite run (default 900)")
    a = ap.parse_args(argv)
    muts = MUTATIONS
    if a.only:
        unknown = set(a.only) - {m[0] for m in MUTATIONS}
        if unknown:
            ap.error("unknown mutation(s): %s" % ", ".join(sorted(unknown)))
        muts = [m for m in MUTATIONS if m[0] in a.only]
    if a.list:
        for m in muts:
            print("%-24s %-26s expects: %s" % (m[0], m[1], m[4]))
        return 0

    root = tempfile.mkdtemp(prefix="mut-baseline-")
    try:
        rc, out = run_suite(copy_examples(root), a.timeout)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if rc != 0:
        print(out)
        print("baseline: the unmodified suite fails (rc %d); fix that first" % rc)
        return 1
    print("baseline: %s" % out.strip().splitlines()[-1])

    results = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        for mid, status, detail in pool.map(lambda m: mutate(m, a.timeout), muts):
            results[mid] = status
            print("%-24s %-12s %s" % (mid, status, detail), flush=True)
    bad = [k for k, v in results.items() if v != "caught"]
    print("\n%d of %d mutations caught for the right reason%s" % (
        len(results) - len(bad), len(results), "" if not bad else "; not caught: " + ", ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
