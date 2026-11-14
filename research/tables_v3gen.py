#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze v3-generality results from results.jsonl.

  tables_v3gen.py [runs/research/results.jsonl] [--e8-baseline]

Research code, unsupported: kept as used for the paper (see research/README.md).

Reads all rows with task=v3-generality (and paper-e8-gen for baselines),
prints cross-model tables and writes v3-generality.md-ready tables.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path, tasks=("v3-generality", "paper-e8-gen")):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") in tasks or r.get("exp", "").startswith("v3-gen"):
                rows.append(r)
    return rows


def key(r):
    return (r.get("model", ""), r.get("dataset", ""), r.get("config", ""),
            r.get("seed", 0))


def group(rows):
    """(model, dataset, config) -> {seed: metrics}"""
    out = defaultdict(dict)
    for r in rows:
        k = (r.get("model", ""), r.get("dataset", ""), r.get("config", ""))
        s = r.get("seed", 0)
        out[k][s] = r.get("metrics", {})
    return out


def mean_seeds(seeds_dict):
    """Average metrics over seeds."""
    keys = set()
    for m in seeds_dict.values():
        keys.update(m.keys())
    out = {}
    for k in keys:
        vals = [m[k] for m in seeds_dict.values() if k in m]
        if vals and isinstance(vals[0], (int, float)):
            out[k] = float(np.mean(vals))
    return out


def delta(metrics, base):
    d = {}
    for k in ("AUC@3", "AUC@30"):
        if k in metrics and k in base:
            d[f"d{k}"] = metrics[k] - base[k]
    return d


def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        print("usage: tables_v3gen.py [results.jsonl] [--e8-baseline]\n" + __doc__)
        return
    path = next((a for a in sys.argv[1:] if not a.startswith("--")), "runs/research/results.jsonl")
    rows = load(path)
    if not rows:
        print("No v3-generality rows found")
        return

    g = group(rows)
    models = sorted({k[0] for k in g})
    datasets = sorted({k[1] for k in g})
    configs = sorted({k[2] for k in g})

    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Configs: {configs}")
    print(f"Total rows: {len(rows)}")

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        base_cfg = "dense@bf16"
        for ds in datasets:
            base_k = (model, ds, base_cfg)
            if base_k not in g:
                continue
            base = mean_seeds(g[base_k])
            print(f"\n  {ds} (dense baseline: AUC@3={base.get('AUC@3', '?'):.2f} "
                  f"/ @30={base.get('AUC@30', '?'):.2f}, "
                  f"{len(g[base_k])} seeds)")
            for cfg in configs:
                if cfg == base_cfg:
                    continue
                k = (model, ds, cfg)
                if k not in g:
                    continue
                m = mean_seeds(g[k])
                d = delta(m, base)
                seeds = sorted(g[k].keys())
                per_seed = " ".join(f"s{s}={g[k][s].get('AUC@3', 0):.2f}"
                                    for s in seeds)
                cfg_short = cfg[:50] + "..." if len(cfg) > 50 else cfg
                print(f"    {cfg_short:55s} "
                      f"AUC@3={m.get('AUC@3', 0):6.2f} "
                      f"(d={d.get('dAUC@3', 0):+.2f}) "
                      f"@30={m.get('AUC@30', 0):6.2f} "
                      f"(d={d.get('dAUC@30', 0):+.2f})  "
                      f"[{per_seed}]")

    # worst-domain table
    print(f"\n{'='*60}")
    print("Worst-domain ΔAUC@3/@30 (mean over seeds)")
    print(f"{'='*60}")
    for model in models:
        base_cfg = "dense@bf16"
        all_cfgs = [c for c in configs if c != base_cfg and
                    any((model, ds, c) in g for ds in datasets)]
        if not all_cfgs:
            continue
        print(f"\n{model}:")
        print(f"  {'config':55s} worst3  worst30  mean3  mean30")
        for cfg in all_cfgs:
            d3s, d30s = [], []
            for ds in datasets:
                bk = (model, ds, base_cfg)
                ck = (model, ds, cfg)
                if bk not in g or ck not in g:
                    continue
                bm = mean_seeds(g[bk])
                cm = mean_seeds(g[ck])
                d = delta(cm, bm)
                d3s.append(d.get("dAUC@3", 0))
                d30s.append(d.get("dAUC@30", 0))
            if d3s:
                cfg_short = cfg[:50] + "..." if len(cfg) > 50 else cfg
                print(f"  {cfg_short:55s} "
                      f"{min(d3s):+6.2f}  {min(d30s):+6.2f}  "
                      f"{np.mean(d3s):+6.2f}  {np.mean(d30s):+6.2f}")

    # timing
    timing_rows = [r for r in rows if r.get("exp", "").startswith("v3-gen-timing")]
    if timing_rows:
        print(f"\n{'='*60}")
        print("Timing (ms)")
        print(f"{'='*60}")
        for r in timing_rows:
            lat = r.get("latency_ms", {})
            cfg = r.get("config", "")[:50]
            print(f"  {r.get('model',''):8s} {r.get('dataset',''):20s} "
                  f"{cfg:50s} {lat.get('median', '?')} ms  "
                  f"{r.get('peak_mem_bytes', 0) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
