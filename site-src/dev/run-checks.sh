#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Run the page checks (check.sh: format unit tests, then the browser checks) in the Playwright image and collect the
# screenshots.
# Usage: site-src/dev/run-checks.sh [-h|--help] [SCREENSHOT_DIR]     (default: site-src/dev/out/shots, git-ignored)
#   RAWATTA_CHECK_HOST=<ssh host>  run on that Docker host instead of locally (the tree is rsynced there first)
#   RAWATTA_CHECK_DIR=<dir>        directory on that host (default: rawatta-site-check)
set -e
case "${1:-}" in
  -h|--help) sed -n '3,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  -*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
SHOTS=${1:-$WT/site-src/dev/out/shots}
IMG=mcr.microsoft.com/playwright:v1.63.0-noble
RUN="docker run --rm --name rawatta-site-check --shm-size=1g -u \$(id -u):\$(id -g) -e HOME=/tmp -v \$PWD:/w -w /w $IMG sh site-src/dev/check.sh"
mkdir -p "$SHOTS"
if [ -n "$RAWATTA_CHECK_HOST" ]; then
  RDIR=${RAWATTA_CHECK_DIR:-rawatta-site-check}
  rsync -az --delete --exclude .git --exclude target/ --exclude __pycache__ --exclude .venv "$WT"/ "$RAWATTA_CHECK_HOST:$RDIR/"
  set +e
  ssh -o BatchMode=yes "$RAWATTA_CHECK_HOST" "cd $RDIR && $RUN"
  rc=$?
  set -e
  rsync -az "$RAWATTA_CHECK_HOST:$RDIR/site-src/dev/out/shots/" "$SHOTS/"
else
  set +e
  (cd "$WT" && eval "$RUN")
  rc=$?
  set -e
  [ "$SHOTS" = "$WT/site-src/dev/out/shots" ] || cp -R "$WT/site-src/dev/out/shots/." "$SHOTS/"
fi
exit $rc
