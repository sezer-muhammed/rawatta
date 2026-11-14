#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate rawatta configs on the paper's datasets: build a bench job from the
manifests, run `rawatta bench`, score with the official evaluator, print a table.

    python3 eval/run_eval.py --data $DATA --datasets paper --configs exact,rawatta,rawatta-fast \
        [--seeds 0,1,2] [--reference exact] [--out runs/eval-<stamp>] [--reps 0] [--no-depth]

Datasets (prepared under `$DATA/<name>` by eval/data/*.py; manifests in eval/manifests/):

| name | layout | frames (map) | notes |
|---|---|---|---|
| eth3d | omega | VGGT-Omega's official 10-frame map, every seed | depth |
| sintel | omega | official 10-frame map (from the pinned official source) | depth; not in the paper's tables |
| co3d | omega | `co3d_s<seed>` | pose only; eval split = the 8 categories outside the calibration set (40 seqs) |
| re10k | omega | `re10k_s<seed>` | pose only; eval split = sequences outside the calibration list |
| 7scenes, nrgbd, tum_dynamic | omega | `<name>_s<seed>` (10 random frames), or `--map n50` | depth |
| mipnerf360 | colmap | `--frames N` evenly spaced (`--seeds` >= 1: seeded random) | pose only (COLMAP GT) |

`--datasets paper` = eth3d,co3d,re10k,7scenes,nrgbd,tum_dynamic (the six evaluation datasets).
`--map` overrides the map for every dataset (`s0|s1|s2|n50|official|<file.json>`).

Steps per (dataset, seed): `<out>/<dataset>-s<seed>/job.json` -> `rawatta bench --job`
-> `score.py` -> `<out>/results.json`. Then the table: AUC@3, AUC@30 (and delta1.25,
AbsRel where there is depth) per dataset and config, the change against `--reference`
(default `exact`, the exact engine), and the worst-dataset change per config (per-seed
minimum over datasets; mean +- sample std over seeds when several seeds ran).
Latency columns appear when `--reps` > 0 (median over the cell's timed runs,
averaged over sequences); timings are only meaningful on an otherwise idle GPU
(take a GPU lock around the whole command if the machine is shared).

Other flags: `--rawatta PATH` (binary, default `rawatta` on PATH), `--weights` (sets
RAWATTA_WEIGHTS), `--score-python` (interpreter with numpy/torch/opencv/Pillow for the
scorer, default this one), `--skip-bench` (re-score existing cells), `--dry-run` (write the
jobs and print the commands only), `--keep` (keep depth/conf .npy; deleted after scoring
by default), `--max-scenes`, `--split all` (no calibration exclusion), `--points`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"
PAPER = ("eth3d", "co3d", "re10k", "7scenes", "nrgbd", "tum_dynamic")
OMEGA_DATASETS = ("eth3d", "sintel", "co3d", "re10k", "7scenes", "nrgbd", "tum_dynamic")
DEPTH = {"eth3d", "sintel", "7scenes", "nrgbd", "tum_dynamic"}
SEED_FREE = {"eth3d", "sintel"}  # official frames: the same map (and result) for every seed
COLMAP_DATASETS = ("mipnerf360",)
NAMES = {"eth3d": "ETH3D", "sintel": "Sintel", "co3d": "CO3D", "re10k": "RE10K", "7scenes": "7-Sc.",
         "nrgbd": "NRGBD", "tum_dynamic": "TUM-dyn", "mipnerf360": "MipNeRF360"}


# --------------------------------------------------------------------------
# frame maps and splits
# --------------------------------------------------------------------------

def splits() -> dict:
    return json.loads((MANIFESTS / "splits.json").read_text())


def manifest(ds: str) -> dict:
    return json.loads((MANIFESTS / f"{ds}.json").read_text())


def frame_map(ds: str, seed: int, map_arg: str | None = None, split: str = "eval") -> dict[str, list[int]]:
    """{scene: [frame indices]} for one omega-layout dataset and seed."""
    if map_arg and Path(map_arg).is_file():
        m = json.loads(Path(map_arg).read_text())
    elif ds in SEED_FREE or map_arg == "official":
        if ds == "eth3d" and map_arg in (None, "official"):
            m = manifest("eth3d")["frame_maps"]["eth3d.json"]["map"]
        else:
            sys.path.insert(0, str(HERE))
            import omega_src  # noqa: PLC0415

            m = json.loads((omega_src.ensure() / "eval" / "frames" / f"{ds}.json").read_text())
    else:
        key = map_arg or f"s{seed}"
        fm = manifest(ds)["frame_maps"]
        name = f"{ds}_{key}.json"
        if name not in fm:
            raise SystemExit(f"{ds}: no frame map {name} in eval/manifests/{ds}.json (have {sorted(fm)})")
        m = fm[name]["map"]
    if split == "eval":
        cal = set(splits().get(ds, {}).get("calibration", []))
        m = {k: v for k, v in m.items() if k not in cal}
    return dict(sorted(m.items())) if ds not in SEED_FREE else m


def colmap_pick(n_images: int, n: int, seed: int) -> list[int]:
    """Evenly spaced (seed 0) or seeded-random (seed >= 1) frames, as the paper's evaluation driver chose them."""
    k = min(n, n_images)
    if seed:
        return sorted(random.Random(seed).sample(range(n_images), k))
    return sorted({round(i * (n_images - 1) / max(k - 1, 1)) for i in range(k)})


def colmap_sequences(root: Path, scenes: list[str], n: int, seed: int) -> list[dict]:
    out = []
    for sc in scenes:
        d = root / sc
        img_dir = next((d / x for x in ("images_4", "images_2", "images") if (d / x).is_dir()), None)
        if img_dir is None:
            raise SystemExit(f"{d}: no images_4/images_2/images directory")
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        out.append({"name": sc, "images": [str(imgs[i]) for i in colmap_pick(len(imgs), n, seed)],
                    "gt_scene": str(d)})
    return out


def omega_sequences(root: Path, fmap: dict[str, list[int]]) -> tuple[list[dict], dict]:
    """Bench sequences for the scenes present under root; returns (sequences, map actually used)."""
    seqs, used = [], {}
    for scene, idx in fmap.items():
        sd = root / scene
        if not (sd / "cameras.npz").is_file():
            continue
        names = (sd / "frames.txt").read_text().split()
        paths = [sd / "images" / names[i] for i in idx]
        if not all(p.is_file() for p in paths):
            print(f"[run_eval] {scene}: sampled frames missing, skipped", file=sys.stderr)
            continue
        seqs.append({"name": scene, "images": [str(p) for p in paths]})
        used[scene] = idx
    return seqs, used


def bench_job(out: Path, configs: list[str], seqs: list[dict], reps: int, warmup: int, depth: bool,
              crop_aspect: float | None) -> dict:
    job = {"out": str(out), "configs": configs, "reps": reps, "warmup": warmup, "depth": depth,
           "sequences": [{"name": s["name"], "images": s["images"]} for s in seqs]}
    if crop_aspect:
        job["crop_aspect"] = crop_aspect
    return job


def config_dir(cells: Path, cfg: str) -> Path:
    for name in (cfg.replace("/", "%"), cfg):
        if (cells / name).is_dir():
            return cells / name
    return cells / cfg.replace("/", "%")


def latency(cells: Path, cfg: str) -> float | None:
    meds = []
    for c in sorted(config_dir(cells, cfg).rglob("cell.json")):
        j = json.loads(c.read_text())
        if j.get("timed") and "median_ms" in j:
            meds.append(float(j["median_ms"]))
    return float(np.mean(meds)) if meds else None


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def deltas(res: dict, reference: str) -> dict:
    """res[seed][dataset][config] = metrics -> {seed: {config: {dataset: (dAUC@3, dAUC@30)}}}."""
    out = {}
    for seed, by_ds in res.items():
        out[seed] = {}
        for ds, by_cfg in by_ds.items():
            ref = by_cfg.get(reference)
            if not ref:
                continue
            for cfg, m in by_cfg.items():
                if not m or "AUC@3" not in m:
                    continue
                out[seed].setdefault(cfg, {})[ds] = (m["AUC@3"] - ref["AUC@3"], m["AUC@30"] - ref["AUC@30"])
    return out


def worst(dl: dict) -> dict:
    """{config: {"worst3": [per seed], "worst30": [...], "at": [dataset per seed]}}."""
    out: dict = {}
    for seed, by_cfg in dl.items():
        for cfg, by_ds in by_cfg.items():
            if not by_ds:
                continue
            w = out.setdefault(cfg, {"worst3": [], "worst30": [], "at": []})
            ds3 = min(by_ds, key=lambda d: by_ds[d][0])
            w["worst3"].append(by_ds[ds3][0])
            w["at"].append(ds3)
            w["worst30"].append(min(v[1] for v in by_ds.values()))
    return out


def pm(xs: list[float]) -> str:
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:+.2f}"
    return f"{statistics.mean(xs):+.2f} ± {statistics.stdev(xs):.2f}"


def mean_metric(res: dict, ds: str, cfg: str, key: str) -> float | None:
    v = [res[s][ds][cfg][key] for s in res if ds in res[s] and cfg in res[s][ds] and res[s][ds][cfg]
         and key in res[s][ds][cfg]]
    return float(np.mean(v)) if v else None


def format_table(res: dict, configs: list[str], datasets: list[str], reference: str,
                 lat: dict | None = None) -> str:
    seeds = sorted(res)
    lines = [f"Seeds: {', '.join(map(str, seeds))}. Values: mean over seeds. Δ: against `{reference}`.", ""]
    hdr = "| dataset | config | AUC@3 | AUC@30 | δ1.25 | AbsRel | ΔAUC@3 | ΔAUC@30 |"
    lines += [hdr, "|---|---|---|---|---|---|---|---|"]
    dl = deltas(res, reference)
    for ds in datasets:
        for cfg in configs:
            a3 = mean_metric(res, ds, cfg, "AUC@3")
            if a3 is None:
                continue
            a30 = mean_metric(res, ds, cfg, "AUC@30")
            d125 = mean_metric(res, ds, cfg, "delta125")
            ar = mean_metric(res, ds, cfg, "AbsRel")
            d3 = [dl[s][cfg][ds][0] for s in seeds if cfg in dl.get(s, {}) and ds in dl[s][cfg]]
            d30 = [dl[s][cfg][ds][1] for s in seeds if cfg in dl.get(s, {}) and ds in dl[s][cfg]]
            lines.append(f"| {NAMES.get(ds, ds)} | {cfg} | {a3:.2f} | {a30:.2f} | "
                         f"{'-' if d125 is None else f'{d125:.2f}'} | {'-' if ar is None else f'{ar:.4f}'} | "
                         f"{pm(d3)} | {pm(d30)} |")
    lines += ["", "Worst-dataset change (per-seed minimum over the datasets above; mean ± std over seeds):", "",
              "| config | worst ΔAUC@3 | worst ΔAUC@30 | worst dataset (AUC@3) |" + (" latency ms |" if lat else ""),
              "|---|---|---|---|" + ("---|" if lat else "")]
    for cfg, w in worst(dl).items():
        at = ", ".join(sorted({NAMES.get(d, d) for d in w["at"]}))
        row = f"| {cfg} | {pm(w['worst3'])} | {pm(w['worst30'])} | {at} |"
        if lat:
            v = lat.get(cfg)
            row += f" {'-' if v is None else f'{v:.1f}'} |"
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def run(cmd: list[str], log: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    print("[run_eval] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if log:
        log.write_text(p.stdout + p.stderr)
    return p


def cleanup(cells: Path) -> None:
    for p in cells.rglob("*.npy"):
        if p.name in ("depth.npy", "conf.npy"):
            p.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("RAWATTA_DATA", "data"), help="dataset root ($RAWATTA_DATA)")
    ap.add_argument("--datasets", default="paper", help="comma list, or `paper`")
    ap.add_argument("--configs", default="exact,rawatta,rawatta-fast")
    ap.add_argument("--reference", default="exact")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--map", default=None, help="override frame map: s0|s1|s2|n50|official|<file>")
    ap.add_argument("--split", choices=("eval", "all"), default="eval")
    ap.add_argument("--frames", type=int, default=50, help="colmap datasets: frames per scene")
    ap.add_argument("--scenes", default=None, help="colmap datasets: comma list (default: all in --data)")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--crop-aspect", type=float, default=None)
    ap.add_argument("--points", action="store_true")
    ap.add_argument("--rawatta", default=os.environ.get("RAWATTA_BIN", "rawatta"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--score-python", default=sys.executable)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    datasets = list(PAPER) if a.datasets == "paper" else a.datasets.split(",")
    configs = [c for c in a.configs.split(",") if c]
    if a.reference not in configs:
        configs.insert(0, a.reference)
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(a.out or f"runs/eval-{dt.datetime.now():%Y%m%d-%H%M%S}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data).resolve()
    env = dict(os.environ)
    if a.weights:
        env["RAWATTA_WEIGHTS"] = str(Path(a.weights).resolve())

    res: dict = {s: {} for s in seeds}
    lat: dict = {}
    records = []
    for ds in datasets:
        if ds not in OMEGA_DATASETS + COLMAP_DATASETS:
            raise SystemExit(f"unknown dataset {ds}")
        root = data / ds
        done_seed_free = None
        for seed in seeds:
            if ds in SEED_FREE and done_seed_free is not None and a.map in (None, "official"):
                res[seed][ds] = res[done_seed_free][ds]
                continue
            cells = out / f"{ds}-s{seed}"
            cells.mkdir(parents=True, exist_ok=True)
            if ds in COLMAP_DATASETS:
                scenes = a.scenes.split(",") if a.scenes else sorted(
                    p.name for p in root.iterdir() if (p / "sparse/0/images.bin").is_file())
                if a.max_scenes:
                    scenes = scenes[: a.max_scenes]
                seqs = colmap_sequences(root, scenes, a.frames, seed)
                (cells / "score_job.json").write_text(json.dumps({"sequences": seqs}, indent=1))
                depth = False
            else:
                fmap = frame_map(ds, seed, a.map, a.split)
                if a.max_scenes:
                    fmap = dict(list(fmap.items())[: a.max_scenes])
                seqs, used = omega_sequences(root, fmap)
                (cells / "fmap.json").write_text(json.dumps(used))
                depth = ds in DEPTH and not a.no_depth
            if not seqs:
                print(f"[run_eval] {ds} s{seed}: no sequences under {root} (prepare it with eval/data/)",
                      file=sys.stderr)
                continue
            job = bench_job(cells, configs, seqs, a.reps, a.warmup, depth, a.crop_aspect)
            (cells / "job.json").write_text(json.dumps(job, indent=1))
            bench_cmd = [a.rawatta, "bench", "--job", str(cells / "job.json")]
            if ds in COLMAP_DATASETS:
                score_cmd = [a.score_python, str(HERE / "score.py"), "--cells", str(cells), "--layout", "colmap",
                             "--job", str(cells / "score_job.json")]
            else:
                score_cmd = [a.score_python, str(HERE / "score.py"), "--cells", str(cells), "--layout", "omega",
                             "--root", str(root), "--map", str(cells / "fmap.json")]
                if a.points and depth:
                    score_cmd.append("--points")
            if a.dry_run:
                print(" ".join(shlex.quote(c) for c in bench_cmd))
                print(" ".join(shlex.quote(c) for c in score_cmd))
                continue
            if not a.skip_bench:
                p = run(bench_cmd, cells / "bench.log", env)
                if p.returncode != 0:
                    print(p.stdout[-2000:], p.stderr[-2000:], file=sys.stderr)
                    raise SystemExit(f"rawatta bench failed for {ds} s{seed} (log: {cells / 'bench.log'})")
            p = run(score_cmd, cells / "score.log")
            if p.returncode != 0:
                print(p.stderr[-3000:], file=sys.stderr)
                raise SystemExit(f"scoring failed for {ds} s{seed}")
            scored = json.loads(p.stdout.strip().splitlines()[-1])
            res[seed][ds] = {}
            for cfg in configs:
                sc = scored.get(config_dir(cells, cfg).name, {})
                res[seed][ds][cfg] = sc.get("metrics")
                if a.reps:
                    lat.setdefault(cfg, []).append(latency(cells, cfg))
                records.append({"dataset": ds, "seed": seed, "config": cfg, "metrics": sc.get("metrics"),
                                "per_scene": sc.get("per_scene"), "cells": str(cells)})
            if not a.keep:
                cleanup(cells)
            if ds in SEED_FREE:
                done_seed_free = seed
    if a.dry_run:
        return 0
    lat_mean = {c: float(np.mean([x for x in v if x is not None])) for c, v in lat.items()
                if any(x is not None for x in v)}
    table = format_table(res, configs, datasets, a.reference, lat_mean or None)
    (out / "results.json").write_text(json.dumps({
        "configs": configs, "reference": a.reference, "datasets": datasets, "seeds": seeds,
        "results": {str(s): v for s, v in res.items()}, "records": records, "latency_ms": lat_mean,
        "protocol": "VGGT-Omega eval/evaluate.py per-scene logic, official functions (pinned commit)",
    }, indent=1))
    (out / "table.md").write_text(table + "\n")
    print(table)
    print(f"\n[run_eval] results: {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
