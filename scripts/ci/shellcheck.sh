#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ShellCheck over every shell script in the repository, as run by .github/workflows/ci.yml (and
# locally, from any directory, with bash 3.2 or newer): files named *.sh, plus executable files whose
# first line is a sh/bash/dash shebang. Default severity (style and up). Sourced helpers are followed
# (-x) relative to each script.
#
# Two groups get one check turned off each:
#   - sourced libraries (*.sh without a shebang, e.g. examples/lib/rawatta.sh): SC2034 ("appears
#     unused"), because they define variables for the scripts that source them;
#   - test scripts (any path with a tests/ folder): SC2015 ("A && B || C is not if-then-else"),
#     because they assert with `check_a && check_b || fail`, where running `fail` whenever A or B
#     fails is exactly what is meant.
#
# usage: scripts/ci/shellcheck.sh [--help]   (SHELLCHECK=... to pick the binary)
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
SC=${SHELLCHECK:-shellcheck}

"$SC" --version | sed -n 2p
shebang_re='^#!.*[/ ](ba|da)?sh([[:space:]]|$)'
scripts=()
tests=()
libs=()
add_script() {
  case $1 in
    */tests/*) tests+=("$1") ;;
    *) scripts+=("$1") ;;
  esac
}
while IFS= read -r f; do
  case $f in
    *.sh)
      first=$(head -n 1 -- "$f")
      if [[ $first == '#!'* ]]; then add_script "$f"; else libs+=("$f"); fi
      ;;
    *)
      if [ -x "$f" ]; then
        first=$(head -n 1 -- "$f" | tr -d '\0')
        if [[ $first =~ $shebang_re ]]; then add_script "$f"; fi
      fi
      ;;
  esac
done < <(
  find . \( -path ./.git -o -path ./data -o -path ./runs -o -path ./out \
    -o -name target -o -name .venv -o -name node_modules \) -prune -o -type f -print | LC_ALL=C sort
)

rc=0
if [ ${#scripts[@]} -gt 0 ]; then
  "$SC" -x --source-path=SCRIPTDIR "${scripts[@]}" || rc=1
fi
if [ ${#tests[@]} -gt 0 ]; then
  "$SC" -x --source-path=SCRIPTDIR -e SC2015 "${tests[@]}" || rc=1
fi
if [ ${#libs[@]} -gt 0 ]; then
  "$SC" -x --source-path=SCRIPTDIR -e SC2034 "${libs[@]}" || rc=1
fi
echo "checked ${#scripts[@]} scripts, ${#tests[@]} test scripts and ${#libs[@]} sourced libraries"
if [ $rc -eq 0 ]; then echo "ok: shellcheck passes"; fi
exit $rc
