#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tables for v31-gen-search from a results.jsonl (default runs/research/results.jsonl).

  tables_v31gen.py [results.jsonl]

Research code, unsupported: kept as used for the paper (see research/README.md).

Cell sources (latest record wins): exp v31-gen-search (new cells, seeded frame maps),
v3-generality seed 0 (CO3D/RE10K/7-Scenes/ETH3D; its s1/s2 reused the s0 maps, so
only s0 is taken), E8-vggt1b (VGGT-1B dense seed-0 cells, seed field empty).
ETH3D has only official frames: its seed-0 value stands for every seed.
"""
import json
import statistics as st
import sys
from pathlib import Path

if "-h" in sys.argv or "--help" in sys.argv:
    print("usage: tables_v31gen.py [results.jsonl]\n" + __doc__)
    sys.exit(0)
RES = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/research/results.jsonl")
DOMS = ["co3d", "re10k", "7scenes", "nrgbd", "tum_dynamic", "eth3d"]
NAMES = {"co3d": "CO3D", "re10k": "RE10K", "7scenes": "7-Scenes", "nrgbd": "NRGBD",
         "tum_dynamic": "TUM-dyn", "eth3d": "ETH3D"}
RULE = {
    "VGGT-1B": "L0:merge4,L1:merge4,L2:merge4,L3:merge4,L4:merge4,L5:merge4,L6:merge4,L7:merge4,L8:merge4,L9:merge4,L16:merge4,L17:merge4,L18:merge4,L19:merge4,L20:merge4,L22:merge4,L23:merge4",
    "Pi3": "L0:merge4nx,L1:merge4nx,L2:merge4nx,L3:merge4nx,L4:merge4nx,L5:merge4nx,L6:merge4nx,L7:merge4nx,L8:merge4nx,L9:merge4nx,L10:merge4nx,L17:merge4nx",
}
SEARCHED = {
    "VGGT-1B": "L1:frame,L2:frame,L3:frame,L5:merge4,L6:merge4,L7:frame,L8:merge4,L18:merge4,L20:merge4,L22:merge4,L23:merge4",
    "Pi3": "L0:merge4nx,L1:merge4nx,L3:frame,L5:merge4nx,L7:merge4nx",
}


def cfg_of(r):
    sage = "sage" in r.get("precision", "") or r.get("config", "").endswith("@sage")
    if "+fp8" in r.get("config", ""):
        return None
    sch = r.get("schedule") or r.get("config", "").split("@")[0]
    if r.get("config") == "vggt1b-rule-b":
        sch = RULE["VGGT-1B"]
    if r.get("config") == "pi3-rule-b":
        sch = RULE["Pi3"]
    model = r.get("model")
    if sch == "dense":
        return "dense@sage" if sage else "dense@bf16"
    if sage and sch == RULE.get(model):
        return "rule"
    if sage and sch == SEARCHED.get(model):
        return "searched"
    return None


cells, timing = {}, {}
for line in RES.read_text().splitlines():
    try:
        r = json.loads(line)
    except ValueError:
        continue
    model, exp = r.get("model"), r.get("exp")
    if model not in RULE or "metrics" not in r:
        continue
    c = cfg_of(r)
    if c is None:
        continue
    ds = r.get("dataset")
    if exp in ("v31-gen-timing", "v3-gen-timing") and r.get("latency_ms", {}).get("median"):
        key = (model, c, ds)
        if exp == "v31-gen-timing" or key not in timing:
            timing[key] = (r["latency_ms"]["median"], r["peak_mem_bytes"] / 2**30,
                           r["gpu"].get("lock_held"), exp)
        continue
    seed = r.get("seed") or 0
    if exp == "v31-gen-search":
        pri = 3
    elif exp == "v3-generality" and seed == 0:
        pri = 2
    elif exp == "E8-vggt1b" and c.startswith("dense"):
        pri = 1
    else:
        continue
    k = (model, c, ds, seed)
    if k not in cells or pri >= cells[k][0]:
        cells[k] = (pri, r["metrics"]["AUC@3"], r["metrics"]["AUC@30"], exp)


def get(model, c, d, s):
    v = cells.get((model, c, d, 0 if d == "eth3d" else s))
    return None if v is None else v[1:3]


CFGS = ["dense@bf16", "dense@sage", "rule", "searched"]
LABEL = {"dense@bf16": "dense bf16", "dense@sage": "dense@sage (Sage kernels)",
         "rule": "rule (Rule-B, no search)", "searched": "searched (regime-constrained greedy)"}
for model in RULE:
    print(f"\n### {model}: AUC@3 / AUC@30, mean ± std over seeds 0,1,2\n")
    print("| config | " + " | ".join(NAMES[d] for d in DOMS) + " |")
    print("|---|" + "---|" * len(DOMS))
    for c in CFGS:
        row = []
        for d in DOMS:
            v = [get(model, c, d, s) for s in (0, 1, 2)]
            if any(x is None for x in v):
                row.append("n/a" if all(x is None for x in v) else
                           "partial " + str([s for s in (0, 1, 2) if v[s] is None]))
                continue
            a3 = [x[0] for x in v]
            a30 = [x[1] for x in v]
            row.append(f"{st.mean(a3):.2f}±{st.pstdev(a3):.2f} / {st.mean(a30):.2f}")
        print(f"| {LABEL[c]} | " + " | ".join(row) + " |")
    for base in ("dense@sage", "dense@bf16"):
        print(f"\n{model}: Δ vs {base} (per seed: worst = min over 6 domains; mean ± std over seeds)\n")
        print("| config | worst ΔAUC@3 | worst ΔAUC@30 | mean ΔAUC@3 | mean ΔAUC@30 | worst domain (s0/s1/s2) |")
        print("|---|---|---|---|---|---|")
        for c in CFGS:
            if c == base:
                continue
            w3, w30, m3, m30, wd = [], [], [], [], []
            ok = True
            for s in (0, 1, 2):
                d3, d30 = {}, {}
                for d in DOMS:
                    a, b = get(model, c, d, s), get(model, base, d, s)
                    if a is None or b is None:
                        ok = False
                        break
                    d3[d], d30[d] = a[0] - b[0], a[1] - b[1]
                if not ok:
                    break
                wk = min(d3, key=d3.get)
                w3.append(d3[wk])
                wd.append(NAMES[wk])
                w30.append(min(d30.values()))
                m3.append(st.mean(d3.values()))
                m30.append(st.mean(d30.values()))
            if not ok:
                print(f"| {LABEL[c]} | missing cells |||||")
                continue
            f = lambda x: f"{st.mean(x):+.2f} ± {st.pstdev(x):.2f}"  # noqa: E731
            print(f"| {LABEL[c]} | {f(w3)} | {f(w30)} | {f(m3)} | {f(m30)} | {'/'.join(wd)} |")
    print(f"\n{model}: latency (median of 5 after 2 warm-up)\n")
    print("| frames | config | median ms | peak GiB | speed-up vs dense bf16 | vs dense@sage | lock | source exp |")
    print("|---|---|---|---|---|---|---|---|")
    for ds in ("mipnerf360-n50", "kitchen-n200"):
        b = timing.get((model, "dense@bf16", ds))
        sg = timing.get((model, "dense@sage", ds))
        for c in CFGS:
            t = timing.get((model, c, ds))
            if t is None:
                print(f"| {ds} | {LABEL[c]} | OOM / not run | | | | | |")
                continue
            sp = f"{b[0] / t[0]:.2f}×" if b else "n/a"
            ss = f"{sg[0] / t[0]:.2f}×" if sg else "n/a"
            print(f"| {ds} | {LABEL[c]} | {t[0]:.0f} | {t[1]:.2f} | {sp} | {ss} | {t[2]} | {t[3]} |")
