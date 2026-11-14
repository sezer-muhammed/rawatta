#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Run the format unit tests (test_formats.mjs), then serve docs/ under /rawatta/ (as on GitHub Pages) with
# python -m http.server and run the browser checks (check.cjs). Screenshots and report.json go to site-src/dev/out/shots/.
# Runs inside mcr.microsoft.com/playwright:v1.63.0-noble with the repository mounted at /w (use run-checks.sh); see site-src/README.md.
# Usage: sh site-src/dev/check.sh [-h|--help]
set -e
case "${1:-}" in
  -h|--help) sed -n '3,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
esac
cd /w
PW=/w/.venv/pw
[ -d "$PW/node_modules/playwright" ] || npm install --silent --no-audit --no-fund --prefix "$PW" playwright@1.63.0 axe-core@4.10.2
OUTD=/w/site-src/dev/out
rm -rf "$OUTD/serve" "$OUTD/shots" "$OUTD/examples"
mkdir -p "$OUTD/serve/ship" "$OUTD/serve/dev" "$OUTD/shots"
python3 -I site-src/dev/make_synthetic_scene.py "$OUTD/examples"
node site-src/dev/test_formats.mjs "$OUTD/examples"
ln -s /w/docs "$OUTD/serve/ship/rawatta"                 # shipped page, unchanged
cp -r /w/docs "$OUTD/serve/dev/rawatta"                  # dev copy with the synthetic scene
cp -r "$OUTD/examples/." "$OUTD/serve/dev/rawatta/static/examples/"
# dev copy also flips the arXiv constant to a fake ID, to test the Paper/PDF buttons and the BibTeX substitution
sed -i 's/^const ARXIV_ID = "";/const ARXIV_ID = "2611.00001";/' "$OUTD/serve/dev/rawatta/static/js/main.js"
python3 -m http.server 8811 --bind 127.0.0.1 --directory "$OUTD/serve/ship" >"$OUTD/http-ship.log" 2>&1 &
python3 -m http.server 8812 --bind 127.0.0.1 --directory "$OUTD/serve/dev" >"$OUTD/http-dev.log" 2>&1 &
sleep 1
PW_MODULES="$PW/node_modules" node site-src/dev/check.cjs "$OUTD/shots"
