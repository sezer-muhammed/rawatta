#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Regenerate the page figures (site-src/make_figs.py) in any shell with python3 (3.12), for example
# docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/w -w /w python:3.12 sh site-src/dev/figs.sh
# Uses a venv at .venv in the repository root (created on first use). Arguments go to make_figs.py.
# Usage: sh site-src/dev/figs.sh [-h|--help] [DATA_DIR] [OUT_DIR]
set -e
case "${1:-}" in
  -h|--help) sed -n '3,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
cd "$(dirname "$0")/../.."
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r site-src/requirements.txt
.venv/bin/python -I site-src/make_figs.py "$@"
