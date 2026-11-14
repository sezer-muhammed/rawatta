# SPDX-License-Identifier: Apache-2.0
"""The research harnesses are not re-run for this release (see research/README.md). This test
only checks that every script prints its --help without the upstream model packages (VGGT-1B,
sparse-vggt, SpargeAttn, pi3, safetensors): they are blocked in the subprocess, so a top-level
import of one of them fails the test. PyTorch is a base dependency of the research container;
the four harnesses that import it at the top are skipped where it is not installed.

    python3 -m pytest research/tests -q
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RESEARCH = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(p.name for p in RESEARCH.glob("*.py"))
BLOCKED = ("vggt", "sparse_vggt", "spas_sage_attn", "pi3", "safetensors", "gpa_vggt")
NEEDS_TORCH = {"run_official.py", "v31gen_search.py", "v3_generality.py", "vggt1b_sched.py"}
RUN_BLOCKED = ("import runpy, sys\n"
               f"for m in {BLOCKED!r}:\n"
               "    sys.modules[m] = None\n"
               "sys.argv = [sys.argv[1], '--help']\n"
               "runpy.run_path(sys.argv[0], run_name='__main__')\n")


def test_all_scripts_listed():
    assert SCRIPTS == ["export_maps.py", "run_official.py", "tables_v31gen.py", "tables_v3gen.py",
                       "v31gen_search.py", "v3_generality.py", "vggt1b_sched.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_python_help(script):
    if script in NEEDS_TORCH:
        pytest.importorskip("torch")
    r = subprocess.run([sys.executable, "-c", RUN_BLOCKED, str(RESEARCH / script)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert r.stdout.startswith("usage:"), r.stdout[:500]


@pytest.mark.parametrize("script,shell", [("run.sh", "sh"), ("setup.sh", "bash")])
def test_shell_help(script, shell):
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not installed")
    r = subprocess.run([shell, str(RESEARCH / script), "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.startswith(f"{script}: "), r.stdout[:500] + r.stderr[-500:]
    assert "set -e" not in r.stdout and "SPDX" not in r.stdout
