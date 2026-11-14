#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Python checks, as run by .github/workflows/ci.yml (and locally, from any directory, with bash 3.2 or
# newer, so macOS /bin/bash works too):
#   1. every Python file in the repository compiles (py_compile);
#   2. the pytest suites of tools/, eval/, calibration/, research/ and examples/ pass;
#   3. examples/tests/run_tests.sh (the example scripts end to end against a fake binary) passes.
# Needs python3 with numpy and pytest. The optional parts (byte identity with the original converter,
# the official evaluator) also need torch (the CPU wheel is enough), opencv-python-headless and Pillow;
# without them those tests skip themselves.
#
# usage: scripts/ci/python.sh [--help]   (PYTHON=... to pick the interpreter)
set -euo pipefail
case ${1:-} in
  -h|--help)
    # The header comment above (from line 3 to the usage line) is the help text.
    sed -n '3,/^# usage:/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
    exit 0 ;;
  "") ;;
  *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=${PYTHON:-python3}

echo "== $("$PY" --version)"

echo "== py_compile"
# A while-read loop rather than mapfile, so that the script also runs on bash 3.2 (macOS /bin/bash).
files=()
while IFS= read -r f; do
  files+=("$f")
done < <(
  find . \( -path ./.git -o -path ./data -o -path ./runs -o -path ./out \
    -o -name target -o -name .venv -o -name node_modules -o -name __pycache__ \) -prune \
    -o -type f -name '*.py' -print | LC_ALL=C sort
)
cache=$(mktemp -d)
trap 'rm -rf "$cache"' EXIT
if [ ${#files[@]} -gt 0 ]; then
  PYTHONPYCACHEPREFIX=$cache "$PY" -m py_compile "${files[@]}"
fi
echo "ok: ${#files[@]} files compile"

echo "== pytest"
suites=()
for d in tools/tests eval/tests calibration/tests research/tests examples/python; do
  [ -d "$d" ] && suites+=("$d")
done
if [ ${#suites[@]} -gt 0 ]; then
  PYTHONPYCACHEPREFIX=$cache "$PY" -m pytest -q -p no:cacheprovider -rs "${suites[@]}"
else
  echo "no test suites found"
fi

if [ -x examples/tests/run_tests.sh ]; then
  echo "== examples/tests/run_tests.sh"
  PYTHONPYCACHEPREFIX=$cache examples/tests/run_tests.sh
fi
