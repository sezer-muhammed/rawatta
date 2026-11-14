#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write or verify the manifest of a prepared evaluation dataset.

A manifest (eval/manifests/<dataset>.json) lists per scene the frame count, every
stored file with its SHA-256, the frame maps used for sampling, and the source
notes the preparation script wrote. Stdlib only.

  manifest.py write  <dataset root> <out json> [<frame map json> ...]
  manifest.py verify <dataset root> <manifest json> [--maps <frame map dir>] [--scenes a,b]

`verify` re-hashes the files of every scene present under the root (missing scenes
are reported, not fatal unless --strict) and, with --maps, checks that regenerated
frame maps equal the manifest's. Exit status 1 on any mismatch.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(a) -> int:
    root, out = Path(a.root), Path(a.out)
    scenes = {}
    total = 0
    for sd in sorted(p for p in root.iterdir() if (p / "cameras.npz").is_file()):
        files = {}
        for f in sorted(x for x in sd.rglob("*") if x.is_file()):
            files[str(f.relative_to(sd))] = sha(f)
            total += f.stat().st_size
        scenes[sd.name] = {"frames": len((sd / "frames.txt").read_text().split()), "files": files}
    maps = {Path(m).name: {"sha256": sha(Path(m)), "map": json.loads(Path(m).read_text())} for m in a.maps}
    sources = {p.name: json.loads(p.read_text()) for p in sorted(root.glob("source*.json"))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": root.name, "bytes": total, "scenes": scenes, "frame_maps": maps,
                               "sources": sources}, indent=1, sort_keys=True) + "\n")
    print(out, len(scenes), "scenes", round(total / 2**20, 1), "MiB")
    return 0


def verify(a) -> int:
    root = Path(a.root)
    m = json.loads(Path(a.manifest).read_text())
    want = a.scenes.split(",") if a.scenes else sorted(m["scenes"])
    bad, missing, ok = [], [], 0
    for sc in want:
        info = m["scenes"].get(sc)
        if info is None:
            bad.append(f"{sc}: not in manifest")
            continue
        sd = root / sc
        if not sd.is_dir():
            missing.append(sc)
            continue
        for rel, h in info["files"].items():
            f = sd / rel
            if not f.is_file():
                bad.append(f"{sc}/{rel}: missing")
            elif sha(f) != h:
                bad.append(f"{sc}/{rel}: sha256 differs")
            else:
                ok += 1
    if a.maps:
        for name, fm in m.get("frame_maps", {}).items():
            p = Path(a.maps) / name
            if not p.is_file():
                bad.append(f"map {name}: not regenerated")
            elif json.loads(p.read_text()) != fm["map"]:
                bad.append(f"map {name}: differs from the manifest")
    for b in bad[:50]:
        print("MISMATCH", b)
    if missing:
        print(f"{len(missing)} scene(s) not present: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    print(f"{ok} files verified, {len(bad)} mismatches")
    return 1 if bad or (a.strict and missing) else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="hash a prepared dataset into a manifest")
    w.add_argument("root")
    w.add_argument("out")
    w.add_argument("maps", nargs="*")
    v = sub.add_parser("verify", help="check a prepared dataset against a manifest")
    v.add_argument("root")
    v.add_argument("manifest")
    v.add_argument("--maps", help="directory with regenerated frame maps to compare")
    v.add_argument("--scenes")
    v.add_argument("--strict", action="store_true", help="missing scenes are an error")
    a = p.parse_args()
    return write(a) if a.cmd == "write" else verify(a)


if __name__ == "__main__":
    sys.exit(main())
