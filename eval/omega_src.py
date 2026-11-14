#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch and load the official VGGT-Omega evaluation code at the pinned commit.

The official evaluator (facebookresearch/vggt-omega, `eval/metrics.py`,
`eval/datasets.py`, `eval/model.py`, `eval/prepare.py`) is under the FAIR
Noncommercial Research License, so it is NOT copied into this repository. This
module downloads the source tarball of the commit the paper's evaluation used
into a cache directory the first time it is needed and imports modules from
there by file path.

Cache: $RAWATTA_OMEGA_SRC if set (an existing checkout to use as is), else
${RAWATTA_CACHE:-${XDG_CACHE_HOME:-~/.cache}/rawatta}/vggt-omega-<commit[:7]>.

    python3 eval/omega_src.py            # fetch (if needed) and print the path
    python3 eval/omega_src.py --check    # also import eval/metrics.py
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import shutil
import sys
import tarfile
import types
import urllib.request
from pathlib import Path

OMEGA_REPO = "facebookresearch/vggt-omega"
# The commit of the official code behind every paper number (pinned in the authors' reference image).
OMEGA_COMMIT = "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"


def cache_root() -> Path:
    base = os.environ.get("RAWATTA_CACHE")
    if not base:
        base = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
                            "rawatta")
    return Path(base)


def default_dir() -> Path:
    env = os.environ.get("RAWATTA_OMEGA_SRC")
    return Path(env) if env else cache_root() / f"vggt-omega-{OMEGA_COMMIT[:7]}"


def _safe_extract(data: bytes, dest: Path) -> None:
    """Extracts a GitHub archive tarball into dest, dropping the top directory."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            parts = Path(m.name).parts
            if len(parts) < 2 or m.issym() or m.islnk() or m.isdev():
                continue
            rel = Path(*parts[1:])
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"unsafe path in archive: {m.name}")
            out = dest / rel
            if m.isdir():
                out.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                out.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(out, "wb") as f:
                    shutil.copyfileobj(src, f)


def ensure(dest: Path | None = None, quiet: bool = False) -> Path:
    """Path of the official source tree (downloaded on first use)."""
    dest = Path(dest) if dest else default_dir()
    if (dest / "eval" / "metrics.py").is_file():
        return dest
    if os.environ.get("RAWATTA_OMEGA_SRC") and dest == Path(os.environ["RAWATTA_OMEGA_SRC"]):
        raise FileNotFoundError(f"RAWATTA_OMEGA_SRC={dest} has no eval/metrics.py")
    url = f"https://github.com/{OMEGA_REPO}/archive/{OMEGA_COMMIT}.tar.gz"
    if not quiet:
        print(f"[omega_src] fetching {url} -> {dest} (FAIR Noncommercial Research License)", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "rawatta-eval"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    _safe_extract(data, tmp)
    (tmp / ".rawatta-commit").write_text(OMEGA_COMMIT + "\n")
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


_MODULES: dict[str, types.ModuleType] = {}


def eval_module(name: str, src: Path | None = None) -> types.ModuleType:
    """Imports `<src>/eval/<name>.py` by path (once). `eval/` and the repo root go on sys.path
    because the official modules import each other by bare name (`from metrics import ...`)."""
    if name in _MODULES:
        return _MODULES[name]
    src = ensure(src)
    for p in (str(src), str(src / "eval")):
        if p not in sys.path:
            sys.path.insert(0, p)
    if name == "model":
        _stub_model_imports()
    spec = importlib.util.spec_from_file_location(f"omega_eval_{name}", src / "eval" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    _MODULES[name] = mod
    return mod


def _stub_model_imports() -> None:
    """`eval/model.py` imports the VGGT-Omega network at module level; the scorer only needs
    its `_source_crop`. If the package's own dependencies (einops, torchvision...) are missing,
    stand-ins let the module import; the real package is used whenever it imports."""
    try:
        import vggt_omega.models  # noqa: F401,PLC0415
        import vggt_omega.utils.load_fn  # noqa: F401,PLC0415
        import vggt_omega.utils.pose_enc  # noqa: F401,PLC0415
        return
    except Exception:  # noqa: BLE001
        pass

    def unavailable(*_a, **_k):
        raise RuntimeError("the VGGT-Omega network package could not be imported here")

    for mod, attrs in (("vggt_omega", {}), ("vggt_omega.models", {"VGGTOmega": unavailable}),
                       ("vggt_omega.utils", {}),
                       ("vggt_omega.utils.load_fn", {"load_and_preprocess_images": unavailable}),
                       ("vggt_omega.utils.pose_enc", {"encoding_to_camera": unavailable})):
        m = types.ModuleType(mod)
        m.__dict__.update(attrs)
        sys.modules[mod] = m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", help="where to put / find the source (default: the cache path above)")
    p.add_argument("--check", action="store_true", help="import eval/metrics.py after fetching")
    a = p.parse_args()
    src = ensure(Path(a.dir) if a.dir else None)
    print(src)
    if a.check:
        m = eval_module("metrics", src)
        print("eval/metrics.py ok:", ", ".join(n for n in ("pose_errors", "depth_metrics", "_auc") if hasattr(m, n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
