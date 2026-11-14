#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""v31-gen-search: regime-constrained greedy schedule search on VGGT-1B and π3.

Same protocol as the VGGT-Ω schedule search of the paper (Sec. 4.4), on the PyTorch
models of research/v3_generality.py:
  - five TRAIN calibration domains (outdoor, object, indoor, rgbd, dynamic),
  - every trial at Sage precision, baseline = the model's dense@sage,
  - candidates only on the blocks the model's regime map marks context/late
    diffuse or anchoring; operators frame and merge4 (+log count; frame 0 kept
    dense on VGGT-1B, all frames merged on π3),
  - single-layer sensitivity first; (L, mode) pairs whose single-layer worst
    drop misses the budget are pruned,
  - greedy in order of mean single-layer frame drop; per layer try frame, then
    merge4; accept the first whose CUMULATIVE schedule meets the budget
    (worst ΔAUC@3 >= -b3 and worst ΔAUC@30 >= -b30) on every domain.

Evaluations are cached in a state JSON; a run stops after --minutes and exits 3
so the caller can release the GPU lock and re-queue (research/run.sh search loops).

Research code, unsupported: kept as used for the paper (see research/README.md).
The paper's CO3Dv2 and RE10K calibration maps (co3d_train.json / re10k_train.json)
are released as lists in eval/manifests/paper/calibration.json; if the two files are not in --cal-maps, the
object domain uses the banana/donut sequences of the CO3Dv2 manifest (seed-0 map)
and the indoor domain the 10 RE10K calibration sequences of eval/manifests/splits.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "v3_generality", Path(__file__).resolve().parent / "v3_generality.py")
vg = importlib.util.module_from_spec(_spec)
sys.modules["v3_generality"] = vg
_spec.loader.exec_module(vg)
ro = vg.ro

# Regime maps from v3-generality.md (v3-anatomy statistics, N = 10).
CANDIDATES = {
    # context 0-9, context boundary 10 (diffuse), late diffuse 17,18,20-23, anchoring 19
    "vggt1b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23],
    # context 0-10, late diffuse 16-17 (π3 has no reference frame)
    "pi3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17],
}
MATCHING = {"vggt1b": list(range(11, 17)), "pi3": list(range(11, 16))}

CAL = [  # (domain, data root under --data, frame map file under --cal-maps)
    ("outdoor", "mipnerf360", {s: 50 for s in ("bicycle", "garden", "stump", "kitchen")}),
    ("object", "co3d", "co3d_train.json"),
    ("indoor", "re10k", "re10k_train.json"),
    ("rgbd", "7scenes_train", "7scenes_train_n50.json"),
    ("dynamic", "tum_dynamic_train", "tum_dynamic_train_n50.json"),
]


def cal_map(name: str, maps: Path) -> dict | str:
    """The calibration frame map: a file in --cal-maps, else the fallback described above."""
    if (maps / name).is_file():
        return str(maps / name)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    import run_eval  # noqa: PLC0415

    ds = {"co3d_train.json": "co3d", "re10k_train.json": "re10k"}.get(name)
    if ds is None:
        raise SystemExit(f"missing calibration map {maps / name} (eval/data/*.py --maps writes it)")
    keep = set(run_eval.splits()[ds]["calibration"])
    return {k: v for k, v in run_eval.frame_map(ds, 0, split="all").items() if k in keep}


def merge_mode(model):
    return "merge4nx" if model == "pi3" else "merge4"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("vggt1b", "pi3"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--b3", type=float, default=1.0)
    p.add_argument("--b30", type=float, default=0.5)
    p.add_argument("--minutes", type=float, default=12.0)
    p.add_argument("--data", default=os.environ.get("RAWATTA_DATA", "data"), help="dataset root")
    p.add_argument("--cal-maps", default=None, help="calibration frame maps (default <data>/_frames)")
    a = p.parse_args()
    vg._upstream()
    maps = Path(a.cal_maps) if a.cal_maps else Path(a.data) / "_frames"
    ro.metrics_mod = ro.omega_module("metrics")
    metrics = ro.metrics_mod

    st_path = Path(a.state)
    st_path.parent.mkdir(parents=True, exist_ok=True)
    st = json.loads(st_path.read_text()) if st_path.exists() else {"evals": {}}
    st.setdefault("evals", {})
    t_end = time.time() + a.minutes * 60

    def save():
        st_path.write_text(json.dumps(st, indent=1))

    model = None
    imgs = None

    def load():
        nonlocal model, imgs
        model = vg.build_model(a.model, a.checkpoint, depth=False)
        imgs = {}
        for d, root, fm in CAL:
            if isinstance(fm, str):
                fm = cal_map(fm, maps)
            if isinstance(fm, dict):
                tmp = Path(tempfile.gettempdir()) / f"v31cal-{d}.json"
                tmp.write_text(json.dumps(fm))
                fm = str(tmp)
            scenes = ro.load_scenes(Path(a.data) / root, fm, None)
            imgs[d] = [(vg.load_and_preprocess_images(sc.image_paths), sc.extrinsics)
                       for sc in scenes]
            print(f"[cal] {d}: {len(scenes)} seqs", flush=True)

    def score(sched):
        key = vg.sched_str(sched)
        if key in st["evals"]:
            return st["evals"][key]
        if time.time() > t_end:
            save()
            print("[search] time slice used up; re-run to continue", flush=True)
            sys.exit(3)
        if model is None:
            load()
        model.configure(sched, True, False)
        t0 = time.time()
        out = {}
        for d, lst in imgs.items():
            a3, a30 = [], []
            for im, gt in lst:
                imc = im.cuda()
                pred = model(imc)
                ext = model.extract_poses(imc, pred)
                pm = metrics.pose_metrics(*metrics.pose_errors(ext, gt))
                a3.append(pm["AUC@3"])
                a30.append(pm["AUC@30"])
                del pred, imc
            out[d] = {"AUC@3": float(np.mean(a3)), "AUC@30": float(np.mean(a30)),
                      "n_seq": len(lst)}
        st["evals"][key] = out
        save()
        print(f"[eval {time.time() - t0:.0f}s] {key}@sage: "
              + " ".join(f"{d}={v['AUC@3']:.2f}/{v['AUC@30']:.2f}" for d, v in out.items()),
              flush=True)
        return out

    def drops(r, base):
        return (min(r[d]["AUC@3"] - base[d]["AUC@3"] for d in base),
                min(r[d]["AUC@30"] - base[d]["AUC@30"] for d in base))

    def ok(w):
        return w[0] >= -a.b3 and w[1] >= -a.b30

    mm = merge_mode(a.model)
    modes = ["frame", mm]
    cand = CANDIDATES[a.model]
    base = score({})
    if "dense@bf16" not in st:
        # reference only: dense bf16 on the calibration set
        if time.time() > t_end:
            save()
            sys.exit(3)
        if model is None:
            load()
        model.configure({}, False, False)
        out = {}
        for d, lst in imgs.items():
            a3, a30 = [], []
            for im, gt in lst:
                imc = im.cuda()
                pred = model(imc)
                pm = metrics.pose_metrics(*metrics.pose_errors(model.extract_poses(imc, pred), gt))
                a3.append(pm["AUC@3"])
                a30.append(pm["AUC@30"])
                del pred, imc
            out[d] = {"AUC@3": float(np.mean(a3)), "AUC@30": float(np.mean(a30))}
        st["dense@bf16"] = out
        save()

    sens = {}
    for L in cand:
        for m in modes:
            sens[f"{L}:{m}"] = drops(score({L: m}), base)
    st["sensitivity"] = sens
    order = sorted(cand, key=lambda L: -np.mean(
        [score({L: "frame"})[d]["AUC@3"] - base[d]["AUC@3"] for d in base]))
    st["order"] = order
    save()

    sched, steps = {}, []
    for L in order:
        for m in modes:
            if not ok(sens[f"{L}:{m}"]):
                steps.append({"layer": L, "mode": m, "result": "pruned (single-layer)",
                              "single": sens[f"{L}:{m}"]})
                continue
            trial = {**sched, L: m}
            w = drops(score(trial), base)
            acc = ok(w)
            steps.append({"layer": L, "mode": m, "cum_worst": w,
                          "result": "accepted" if acc else "rejected (cumulative)"})
            if acc:
                sched = trial
                break
    st["steps"] = steps
    st["schedule"] = vg.sched_str(sched)
    st["cal_worst"] = drops(score(sched), base) if sched else (0.0, 0.0)
    st["flop_ratio_n50"] = vg.flop_ratio(sched, len(cand) + len(MATCHING[a.model]),
                                         keep0=a.model == "vggt1b")
    st["done"] = True
    save()
    print(f"[search] {a.model}: {st['schedule']}  cal worst {st['cal_worst']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
