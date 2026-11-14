#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the evaluation frame maps the research harnesses take as --frames.

    python3 research/export_maps.py --out $DATA/_frames [--seeds 0,1,2]

Writes, from eval/manifests (calibration sequences excluded, as eval/run_eval.py does):
eth3d.json (official frames), co3d_eval_s<seed>.json, re10k_eval_s<seed>.json,
7scenes_s<seed>.json, nrgbd_s<seed>.json, tum_dynamic_s<seed>.json. The paper's harness
runs used `re10k_eval_v2` (38 sequences), which was not kept; re10k_eval_s<seed> has the
190 sequences outside the calibration list (see eval/DATASETS.md).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run_eval  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0,1,2")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eth3d.json").write_text(json.dumps(run_eval.frame_map("eth3d", 0), indent=1) + "\n")
    for s in (int(x) for x in a.seeds.split(",")):
        for ds, name in (("co3d", "co3d_eval"), ("re10k", "re10k_eval"), ("7scenes", "7scenes"),
                         ("nrgbd", "nrgbd"), ("tum_dynamic", "tum_dynamic")):
            m = run_eval.frame_map(ds, s)
            (out / f"{name}_s{s}.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    print(f"wrote maps to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
