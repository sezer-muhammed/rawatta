#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Download the "korno-rock" example scene, verify every original against its SHA-256 and save
# 1024 px JPEGs as <dest>/images/frame_000.jpg, frame_001.jpg, ...
# Source, author and licence: SOURCE.md next to this script.
#
# usage: examples/korno-rock/fetch.sh [dest-dir]      (default: <repo>/data/examples/korno-rock)
# env:   KEEP_ORIGINALS=1 keeps the full-size downloads; RESIZER=pillow|magick|ffmpeg; LONG_SIDE=1024
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR source=../lib/fetch_common.sh
. "$here/../lib/fetch_common.sh"
fetch_main "$here" "$here/../../data/examples/korno-rock" "$@"
