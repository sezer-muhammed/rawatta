#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Calibrate a RAWATTA configuration for VGGT-Omega: per-layer precision search, operator
schedule search and the worst-case calibration check of the paper (Sec. 4.4), run through
`rawatta bench` on the calibration datasets, ending in a config TOML in the configs/ format.

    calibrate.py precision --data $DATA [--seed 0] [--low sagepp] [--high sage.pvh16]
    calibrate.py schedule  --data $DATA [--seed 0] --base "dense@sagepp~L4=sage.pvh16/L15=sage.pvh16"
    calibrate.py check     --data $DATA --config <name or grammar> [--seeds 0,1,2] [--budget operator|precision]
                           # default operator budget: the "Cal." column of the paper's Table 1
    calibrate.py all       --data $DATA --name my-setting --emit configs/my-setting.toml
    calibrate.py emit      --config <grammar> --name <name> --emit <file.toml> [--state <out>/state.json]
    calibrate.py rule      --stats stats.json                  # threshold schedule (Alg. 1) from attention statistics

The check (paper Sec. 4.4, "The check"): evaluate the complete configuration on the five
calibration datasets and accept it only if the accuracy drop against the exact engine on
the WORST dataset stays within the budget: ΔAUC@3 >= -0.5 and ΔAUC@30 >= -0.3 when choosing
numeric formats (`precision` budget), ΔAUC@3 >= -1 and ΔAUC@30 >= -0.5 when choosing
operators (`operator` budget). The worst dataset, not the mean, so a loss on one dataset
cannot hide behind gains on others. The "Cal." column of Table 1 (calibration samples of
three passing the check) applies the operator budget to every row, RAWATTA included (paper
Sec. 4.4, "Protocol sensitivity": RAWATTA passes it on all three samples and the stricter
precision budget on the original sample only), so `check` uses the operator budget unless
--budget says otherwise.

Precision search ("give back"): start from FP8 P.V everywhere (`dense@<low>`), measure for every
dense global layer the gain of giving only that layer 16-bit P.V (`~L<i>=<high>`), gain = change
of the worst-dataset ΔAUC@3 + 0.5 x change of the mean ΔAUC@3; then give layers 16-bit P.V in
order of gain, re-measuring the cumulative config each step, until the precision budget is met
(a layer is kept only if it helps the worst dataset by more than -0.1 or improves the mean;
stop at the first non-positive gain). In the paper this stopped after blocks 15 and 4 = RAWATTA.

Schedule search: with the precision fixed (`--base`), measure every single-layer change
(frame or merge4 on each of the 19 non-register global layers) under the base precision;
visit layers in order of their mean single-layer frame-attention cost (cheapest first); on
each layer try frame attention, then merge4, skipping any (layer, operator) whose single-layer
config already misses the budget, and accept the first whose CUMULATIVE config passes the
operator budget. A layer that becomes frame attention drops its 16-bit override (frame layers
always run on the bf16 kernel); a merged layer keeps it (merged layers run on the quantised
kernel, see configs/GRAMMAR.md). In the paper: 49 configurations (38
single-layer, 11 greedy trials), result frame on 0, 4, 5 and merge4 on 3, 19 = RAWATTA-Fast.

Calibration data (paper Sec. 4.4, "Data"; prepare with eval/data/, see eval/DATASETS.md):
  outdoor  Mip-NeRF 360 bicycle, garden, stump, kitchen; 50 frames (COLMAP ground truth)
  object   CO3Dv2 banana + donut, 10 sequences x 10 frames
  indoor   RealEstate10K calibration sequences x 10 frames (--re10k-cal; see the README)
  rgbd     7-Scenes train (first TrainSplit sequence per scene), 50 frames ($DATA/7scenes_train)
  dynamic  TUM fr3 *_validation takes, 50 frames ($DATA/tum_dynamic_train)
A frame sample is one draw of frames per sequence (--seed; 0 = the search's own sample).
Samples 1 and 2 of the 50-frame train splits use the <ds>_n50_s<seed> maps the data scripts
write with --seeded50 (50 random frames; falls back to the 10-frame <ds>_s<seed> map).

Every (config, dataset, seed) result is cached in <out>/cells.json, so an interrupted run
resumes; each bench job holds at most --chunk configs and, with --lock FILE, runs under an
exclusive flock on FILE (use the machine's GPU lock). Needs the scorer's environment
(eval/requirements.txt) and the `rawatta` binary with weights.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"
sys.path.insert(0, str(EVAL))
import run_eval  # noqa: E402

REGISTER = (2, 6, 9, 14, 20)  # register-only global layers of VGGT-Omega: never changed
GLOBAL = [i for i in range(24) if i not in REGISTER]
BUDGETS = {"precision": (0.5, 0.3), "operator": (1.0, 0.5)}
ALIASES = {"exact": "dense@bf16", "rawatta": "dense@sagepp~L4=sage.pvh16/L15=sage.pvh16",
           "rawatta-fast": "L0:frame/L3:merge4/L4:frame/L5:frame/L19:merge4@sagepp~L15=sage.pvh16"}
MODES = ("frame", "merge4")


# --------------------------------------------------------------------------
# calibration domains
# --------------------------------------------------------------------------

@dataclass
class Domain:
    name: str
    layout: str  # omega | colmap
    root: Path
    fmap: dict = field(default_factory=dict)  # omega: {scene: [frame indices]}
    scenes: list = field(default_factory=list)  # colmap
    frames: int = 50


def _map_file(maps: Path, ds: str, seed: int) -> dict:
    """Frame maps the data scripts wrote for the 50-frame train splits (not part of eval/manifests).

    Sample 0: the evenly spaced 50 frames (`<ds>_n50`). Samples 1 and 2: 50 seeded random frames
    (`<ds>_n50_s<seed>`, written with --seeded50, as in the paper's three-sample check); data
    prepared without --seeded50 falls back to the 10-frame `<ds>_s<seed>` map, with a warning."""
    names = [f"{ds}_n50.json"] if seed == 0 else [f"{ds}_n50_s{seed}.json", f"{ds}_s{seed}.json"]
    for i, name in enumerate(names):
        p = maps / name
        if p.is_file():
            if i:
                print(f"[calibrate] warning: {maps / names[0]} missing, using the 10-frame map {p} "
                      "(re-run the data script with --seeded50 for the paper's 50-frame samples)", file=sys.stderr)
            return json.loads(p.read_text())
    raise SystemExit(f"missing frame map {maps / names[0]} (written by eval/data/*.py --maps {maps})")


def domains(data: Path, seed: int, maps: Path | None = None, re10k_cal: list[str] | None = None,
            only: list[str] | None = None) -> list[Domain]:
    sp = run_eval.splits()
    maps = maps or data / "_frames"
    out = []
    want = only or ["outdoor", "object", "indoor", "rgbd", "dynamic"]
    if "outdoor" in want:
        out.append(Domain("outdoor", "colmap", data / "mipnerf360", scenes=sp["mipnerf360"]["calibration"], frames=50))
    if "object" in want:
        cal = set(sp["co3d"]["calibration"])
        fm = run_eval.frame_map("co3d", seed, split="all")
        out.append(Domain("object", "omega", data / "co3d", fmap={k: v for k, v in fm.items() if k in cal}))
    if "indoor" in want:
        cal = set(re10k_cal or sp["re10k"]["calibration"])
        fm = run_eval.frame_map("re10k", seed, split="all")
        out.append(Domain("indoor", "omega", data / "re10k", fmap={k: v for k, v in fm.items() if k in cal}))
    if "rgbd" in want:
        out.append(Domain("rgbd", "omega", data / "7scenes_train", fmap=_map_file(maps, "7scenes_train", seed)))
    if "dynamic" in want:
        out.append(Domain("dynamic", "omega", data / "tum_dynamic_train",
                          fmap=_map_file(maps, "tum_dynamic_train", seed)))
    return out


# --------------------------------------------------------------------------
# config strings
# --------------------------------------------------------------------------

def resolve(cfg: str) -> str:
    return ALIASES.get(cfg, cfg)


def parse_config(cfg: str) -> tuple[dict[int, str], str, dict[int, str]]:
    """`<schedule>@<precision>[~L<i>=<prec>/...]` -> (schedule {layer: op}, precision, overrides)."""
    cfg = resolve(cfg)
    if "+fp8" in cfg:
        raise ValueError("+fp8 configs are not calibrated by this tool")
    body, _, prec = cfg.partition("@")
    prec = prec or "bf16"
    prec, _, ov = prec.partition("~")
    sched = {}
    if body != "dense":
        for item in body.replace(";", ",").replace("/", ",").split(","):
            if item:
                layer, op = item.split(":")
                sched[int(layer.lstrip("L"))] = op
    overrides = {}
    for item in ov.replace("/", ",").split(","):
        if item:
            layer, p = item.split("=")
            overrides[int(layer.lstrip("L"))] = p
    return sched, prec, overrides


def make_config(sched: dict[int, str], prec: str, overrides: dict[int, str]) -> str:
    """Inverse of parse_config, in the engine's canonical order; a frame layer drops its override."""
    body = "/".join(f"L{i}:{m}" for i, m in sorted(sched.items())) or "dense"
    ov = "/".join(f"L{i}={p}" for i, p in sorted(overrides.items()) if sched.get(i) != "frame")
    return f"{body}@{prec}" + (f"~{ov}" if ov else "")


def precision_config(low: str, high: str, restored: list[int]) -> str:
    return make_config({}, low, {i: high for i in restored})


# --------------------------------------------------------------------------
# running and caching cells
# --------------------------------------------------------------------------

class Runner:
    """Runs (config, domain, seed) cells through `rawatta bench` + eval/score.py, cached in cells.json."""

    def __init__(self, a):
        self.a = a
        self.out = Path(a.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.out / "cells.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.is_file() else {}
        self.data = Path(a.data).resolve()
        self.evaluated = 0
        self.bench_s = 0.0
        self._domains: dict[int, list[Domain]] = {}

    @staticmethod
    def key(cfg: str, dom: str, seed: int) -> str:
        return f"{resolve(cfg)}|{dom}|{seed}"

    def save(self) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=1, sort_keys=True))
        tmp.replace(self.cache_path)

    @contextlib.contextmanager
    def lock(self):
        if not self.a.lock:
            yield
            return
        import fcntl  # noqa: PLC0415

        with open(self.a.lock, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def ensure(self, configs: list[str], seed: int) -> dict[str, dict[str, dict]]:
        """{config: {domain: {"AUC@3", "AUC@30"}}} for every config, running the missing cells."""
        configs = list(dict.fromkeys(resolve(c) for c in configs))
        if seed not in self._domains:
            self._domains[seed] = domains(self.data, seed, Path(self.a.maps) if self.a.maps else None,
                                          self.a.re10k_cal_list, self.a.domains_list)
        doms = self._domains[seed]
        for d in doms:
            todo = [c for c in configs if self.key(c, d.name, seed) not in self.cache]
            for i in range(0, len(todo), self.a.chunk):
                self._run(d, seed, todo[i:i + self.a.chunk])
        return {c: {d.name: self.cache.get(self.key(c, d.name, seed)) for d in doms} for c in configs}

    def _run(self, d: Domain, seed: int, cfgs: list[str]) -> None:
        cells = self.out / "cells" / f"{d.name}-s{seed}-{len(self.cache):05d}"
        cells.mkdir(parents=True, exist_ok=True)
        if d.layout == "colmap":
            seqs = run_eval.colmap_sequences(d.root, d.scenes, d.frames, seed)
            (cells / "score_job.json").write_text(json.dumps({"sequences": seqs}, indent=1))
            score = ["--layout", "colmap", "--job", str(cells / "score_job.json")]
        else:
            seqs, used = run_eval.omega_sequences(d.root, d.fmap)
            (cells / "fmap.json").write_text(json.dumps(used))
            score = ["--layout", "omega", "--root", str(d.root), "--map", str(cells / "fmap.json")]
        if not seqs:
            raise SystemExit(f"calibration domain {d.name}: no sequences under {d.root} (see eval/DATASETS.md)")
        job = run_eval.bench_job(cells, cfgs, seqs, 0, 0, False, None)
        (cells / "job.json").write_text(json.dumps(job, indent=1))
        bench = [self.a.rawatta, "bench", "--job", str(cells / "job.json")]
        score_cmd = [self.a.score_python, str(EVAL / "score.py"), "--cells", str(cells), *score]
        print(f"[calibrate] {d.name} s{seed}: {len(cfgs)} config(s), {len(seqs)} sequences", flush=True)
        if self.a.dry_run:
            print("  $", " ".join(shlex.quote(c) for c in bench))
            print("  $", " ".join(shlex.quote(c) for c in score_cmd))
            for c in cfgs:  # mark as seen so the dry run walks the whole search with zero deltas
                self.cache[self.key(c, d.name, seed)] = {"AUC@3": 0.0, "AUC@30": 0.0, "dry_run": True}
            return
        env = dict(os.environ)
        if self.a.weights:
            env["RAWATTA_WEIGHTS"] = str(Path(self.a.weights).resolve())
        t0 = time.time()
        with self.lock():
            p = subprocess.run(bench, text=True, capture_output=True, env=env)
        self.bench_s += time.time() - t0
        (cells / "bench.log").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            raise SystemExit(f"rawatta bench failed (log: {cells / 'bench.log'})")
        p = subprocess.run(score_cmd, text=True, capture_output=True)
        (cells / "score.log").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            raise SystemExit(f"scoring failed (log: {cells / 'score.log'})")
        scored = json.loads(p.stdout.strip().splitlines()[-1])
        for c in cfgs:
            m = scored.get(run_eval.config_dir(cells, c).name, {}).get("metrics")
            if not m:
                raise SystemExit(f"no metrics for {c} on {d.name} (log: {cells / 'score.log'})")
            self.cache[self.key(c, d.name, seed)] = {"AUC@3": m["AUC@3"], "AUC@30": m["AUC@30"],
                                                     "pairs": m.get("pairs"), "scenes": m.get("scenes")}
            self.evaluated += 1
        self.save()
        for f in cells.rglob("*.npy"):
            if f.name in ("depth.npy", "conf.npy"):
                f.unlink()


# --------------------------------------------------------------------------
# the check and the searches (pure functions over measured cells)
# --------------------------------------------------------------------------

def deltas(res: dict, reference: str) -> dict:
    """Per config: {domain: (ΔAUC@3, ΔAUC@30)}, worst3, worst30, mean3, worst_domain."""
    ref = res[resolve(reference)] if resolve(reference) in res else res[reference]
    out = {}
    for c, doms in res.items():
        row = {}
        for d, m in doms.items():
            if m is None or ref.get(d) is None:
                continue
            row[d] = (m["AUC@3"] - ref[d]["AUC@3"], m["AUC@30"] - ref[d]["AUC@30"])
        if not row:
            out[c] = {"per_domain": {}, "worst3": None, "worst30": None, "mean3": None, "worst_domain": None}
            continue
        wd = min(row, key=lambda k: row[k][0])
        out[c] = {"per_domain": row, "worst3": row[wd][0], "worst30": min(v[1] for v in row.values()),
                  "mean3": sum(v[0] for v in row.values()) / len(row), "worst_domain": wd}
    return out


def passes(r: dict, budget: tuple[float, float]) -> bool:
    """The worst-case check: worst-dataset ΔAUC@3 >= -b3 and ΔAUC@30 >= -b30."""
    return r["worst3"] is not None and r["worst3"] >= -budget[0] and r["worst30"] >= -budget[1]


def precision_search(measure, reference: str, low: str, high: str, budget=BUDGETS["precision"],
                     max_restore: int = 8, layers=GLOBAL, log=print) -> dict:
    """`measure(configs) -> deltas dict`. Returns the search record."""
    base = precision_config(low, high, [])
    singles = {i: precision_config(low, high, [i]) for i in layers}
    dl = measure([reference, base, *singles.values()])
    w0 = dl[base]
    gain = {i: (dl[c]["worst3"] - w0["worst3"]) + 0.5 * (dl[c]["mean3"] - w0["mean3"]) for i, c in singles.items()}
    order = sorted(layers, key=lambda i: -gain[i])
    log(f"[precision] base {base}: worst3 {w0['worst3']:+.2f} worst30 {w0['worst30']:+.2f}")
    log("[precision] gains: " + ", ".join(f"L{i} {gain[i]:+.2f}" for i in order))
    restored: list[int] = []
    cur = w0
    steps = [{"restored": [], "config": base, **_brief(w0)}]
    for i in order:
        if passes(cur, budget) or len(restored) >= max_restore:
            break
        if gain[i] <= 0 and restored:
            break
        trial = restored + [i]
        c = precision_config(low, high, trial)
        r = measure([reference, c])[c]
        log(f"[precision] +L{i}: worst3 {r['worst3']:+.2f} worst30 {r['worst30']:+.2f} mean3 {r['mean3']:+.2f}")
        steps.append({"restored": trial, "config": c, **_brief(r)})
        if r["worst3"] >= cur["worst3"] - 0.1 or r["mean3"] > cur["mean3"]:
            restored, cur = trial, r
    final = precision_config(low, high, restored)
    return {"kind": "precision", "low": low, "high": high, "budget": list(budget), "gain": gain, "order": order,
            "steps": steps, "restored": sorted(restored), "final": final, "passes": passes(cur, budget),
            "final_deltas": _brief(cur)}


def schedule_search(measure, reference: str, base: str, budget=BUDGETS["operator"], layers=GLOBAL,
                    modes=MODES, log=print) -> dict:
    sched0, prec, overrides = parse_config(base)
    if sched0:
        raise ValueError("--base must be a dense schedule (precision only)")
    single = {(i, m): make_config({i: m}, prec, overrides) for i in layers for m in modes}
    dl = measure([reference, base, *single.values()])
    order = sorted(layers, key=lambda i: -dl[single[(i, "frame")]]["mean3"])
    log("[schedule] order: " + " ".join(f"L{i}({dl[single[(i, 'frame')]]['mean3']:+.2f})" for i in order))
    sched: dict[int, str] = {}
    cur = dl[resolve(base)]
    steps = [{"try": None, "config": resolve(base), **_brief(cur)}]
    for i in order:
        for m in modes:
            if not passes(dl[single[(i, m)]], budget):
                steps.append({"try": f"L{i}:{m}", "result": "pruned (single-layer misses the budget)",
                              **_brief(dl[single[(i, m)]])})
                continue
            trial = {**sched, i: m}
            c = make_config(trial, prec, overrides)
            r = measure([reference, c])[c]
            ok = passes(r, budget)
            log(f"[schedule] +L{i}:{m}: worst3 {r['worst3']:+.2f} worst30 {r['worst30']:+.2f} "
                f"mean3 {r['mean3']:+.2f} {'OK' if ok else 'reject'}")
            steps.append({"try": f"L{i}:{m}", "config": c, "result": "accepted" if ok else "rejected (cumulative)",
                          **_brief(r)})
            if ok:
                sched, cur = trial, r
                break
    final = make_config(sched, prec, overrides)
    return {"kind": "schedule", "base": resolve(base), "budget": list(budget), "order": order,
            "schedule": {str(k): v for k, v in sorted(sched.items())}, "steps": steps, "final": final,
            "passes": passes(cur, budget), "final_deltas": _brief(cur),
            "configs_measured": len(single) + 1 + sum(1 for s in steps if s.get("config") and s["try"])}


def _brief(r: dict) -> dict:
    return {k: r[k] for k in ("worst3", "worst30", "mean3", "worst_domain") if k in r} | {
        "per_domain": {d: [round(x, 4) for x in v] for d, v in r.get("per_domain", {}).items()}}


def threshold_schedule(stats: dict, frame_in_stage: bool = False) -> dict[int, str]:
    """Paper Alg. 1. stats: {layer: {"e", "e_min", "f", "r"}} for the global blocks (register-only skipped).

    label: matching if e < 0.80; else anchoring if f >= 0.28; else context. The context stage ends at
    the first block that is matching, anchoring or has r >= 1.5. Context in the stage -> merge4 (or
    frame); context with e_min >= 0.75 -> merge4; anchoring with e_min >= 0.40 -> merge4 (frame 0
    complete); everything else dense."""
    layers = sorted(int(k) for k in stats if int(k) not in REGISTER)
    s = {int(k): v for k, v in stats.items()}
    label = {}
    for i in layers:
        if s[i]["e"] < 0.80:
            label[i] = "matching"
        elif s[i]["f"] >= 0.28:
            label[i] = "anchoring"
        else:
            label[i] = "context"
    stage_end = next((i for i in layers if label[i] != "context" or s[i]["r"] >= 1.5), 24)
    out = {}
    for i in layers:
        if label[i] == "context" and i < stage_end:
            out[i] = "frame" if frame_in_stage else "merge4"
        elif label[i] == "context" and s[i]["e_min"] >= 0.75:
            out[i] = "merge4"
        elif label[i] == "anchoring" and s[i]["e_min"] >= 0.40:
            out[i] = "merge4"
    return out


# --------------------------------------------------------------------------
# config TOML (configs/ format)
# --------------------------------------------------------------------------

def min_cc(cfg: str) -> str:
    sched, prec, ov = parse_config(cfg)
    quant = prec != "bf16" or any(p != "bf16" for p in ov.values()) or any(m.startswith("merge") for m in sched.values())
    return "8.9" if quant else "8.0"


def describe(cfg: str) -> str:
    sched, prec, ov = parse_config(cfg)
    parts = []
    frame = [i for i, m in sorted(sched.items()) if m == "frame"]
    merged = {}
    for i, m in sorted(sched.items()):
        if m.startswith("merge"):
            merged.setdefault(m, []).append(i)
    if frame:
        parts.append("frame attention (bf16 kernel) on global blocks " + ", ".join(map(str, frame)))
    mprec = prec if prec.startswith("sage") else "sage"  # merged layers have no bf16 path (GRAMMAR.md)
    for m, ls in merged.items():
        parts.append(f"{m} (frame 0 complete, quantised kernel in {mprec}) on blocks " + ", ".join(map(str, ls)))
    pv = {"bf16": "exact bf16 attention", "sagepp": "SageAttention2++ (INT8 QK^T, FP8 P.V, FP16 block accumulation)",
          "sage": "SageAttention2 (INT8 QK^T, FP8 P.V, FP32 accumulation)"}.get(prec, prec)
    parts.append(f"the other global layers dense on {pv}")
    ovs = [f"{p} on block {i}" for i, p in sorted(ov.items()) if sched.get(i) != "frame"]
    if ovs:
        parts.append("overrides: " + ", ".join(ovs))
    return "; ".join(parts) + "."


def toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def emit_toml(path: Path, name: str, cfg: str, record: dict | None = None, display_name: str | None = None) -> str:
    """Writes a config in the configs/ format (name, display_name, grammar, description,
    min_compute_capability, paper_reference); calibration provenance goes into comments."""
    cfg = resolve(cfg)
    shown = "<path.toml>" if str(path) == "-" else str(path)
    lines = ["# SPDX-License-Identifier: Apache-2.0",
             f"# Generated by calibration/calibrate.py on {dt.date.today().isoformat()}.",
             f"# Use it with `rawatta run --config {shown}`, or pass the grammar string below (see configs/GRAMMAR.md)."]
    if record:
        lines.append(f"# Calibration: datasets {', '.join(record.get('domains', []))}; frame sample(s) "
                     f"{record.get('seeds')}; reference {record.get('reference')}; budget worst ΔAUC@3 >= "
                     f"-{record['budget'][0]}, ΔAUC@30 >= -{record['budget'][1]}.")
        fd = record.get("final_deltas") or {}
        if fd.get("worst3") is not None:
            lines.append(f"# Measured on the calibration set: worst ΔAUC@3 {fd['worst3']:+.2f} ({fd.get('worst_domain')}), "
                         f"worst ΔAUC@30 {fd['worst30']:+.2f}, passes: {record.get('passes')}.")
            for d, v in sorted((fd.get("per_domain") or {}).items()):
                lines.append(f"#   {d}: ΔAUC@3 {v[0]:+.2f}, ΔAUC@30 {v[1]:+.2f}")
    lines += ["", f"name = {toml_str(name)}", f"display_name = {toml_str(display_name or name)}",
              f"grammar = {toml_str(cfg)}", f"description = {toml_str(describe(cfg))}",
              f"min_compute_capability = {toml_str(min_cc(cfg))}",
              'paper_reference = "Not a paper setting: produced by the calibration procedure of Sec. 4.4 '
              '(calibration/README.md)."', ""]
    text = "\n".join(lines)
    if str(path) != "-":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def make_measure(runner: Runner, seed: int, reference: str):
    def measure(configs):
        return deltas(runner.ensure([reference, *configs], seed), reference)
    return measure


def save_state(out: Path, key: str, record: dict) -> None:
    p = out / "state.json"
    st = json.loads(p.read_text()) if p.is_file() else {}
    st[key] = record
    p.write_text(json.dumps(st, indent=1, sort_keys=True))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("precision", "schedule", "check", "all", "emit", "rule"))
    p.add_argument("--data", default=os.environ.get("RAWATTA_DATA", "data"))
    p.add_argument("--maps", default=None, help="frame maps of the train splits (default $DATA/_frames)")
    p.add_argument("--out", default="runs/calibration")
    p.add_argument("--seed", type=int, default=0, help="frame sample for the searches")
    p.add_argument("--seeds", default="0,1,2", help="check: frame samples")
    p.add_argument("--reference", default="exact")
    p.add_argument("--low", default="sagepp", help="precision search: FP8 P.V precision")
    p.add_argument("--high", default="sage.pvh16", help="precision search: 16-bit P.V precision")
    p.add_argument("--base", default=None, help="schedule search: precision-only config (default: precision result)")
    p.add_argument("--config", default=None, help="check/emit: config name or grammar")
    p.add_argument("--budget", choices=tuple(BUDGETS), default="operator",
                   help="check: which budget (default: operator, as in the Cal. column of Table 1)")
    p.add_argument("--budget3", type=float, default=None, help="override the ΔAUC@3 budget")
    p.add_argument("--budget30", type=float, default=None, help="override the ΔAUC@30 budget")
    p.add_argument("--max-restore", type=int, default=8)
    p.add_argument("--re10k-cal", default=None, help="file with RE10K calibration sequence ids, one per line")
    p.add_argument("--domains", default=None, help="subset of outdoor,object,indoor,rgbd,dynamic")
    p.add_argument("--chunk", type=int, default=6, help="configs per bench job")
    p.add_argument("--lock", default=None, help="flock this file around every bench job (GPU lock)")
    p.add_argument("--rawatta", default=os.environ.get("RAWATTA_BIN", "rawatta"))
    p.add_argument("--weights", default=None)
    p.add_argument("--score-python", default=sys.executable)
    p.add_argument("--name", default="rawatta-calibrated")
    p.add_argument("--emit", default=None, help="write the resulting config TOML here")
    p.add_argument("--state", default=None, help="emit: state.json of an earlier run (default <out>/state.json)")
    p.add_argument("--stats", default=None, help="rule: JSON {layer: {e, e_min, f, r}}")
    p.add_argument("--frame-in-stage", action="store_true", help="rule: frame instead of merge4 in the context stage")
    p.add_argument("--dry-run", action="store_true", help="write jobs and print commands; deltas are all zero")
    a = p.parse_args()
    a.re10k_cal_list = [x.strip() for x in Path(a.re10k_cal).read_text().split() if x.strip()] if a.re10k_cal else None
    a.domains_list = a.domains.split(",") if a.domains else None

    def budget(kind):
        b3, b30 = BUDGETS[kind]
        return (a.budget3 if a.budget3 is not None else b3, a.budget30 if a.budget30 is not None else b30)

    if a.cmd == "rule":
        if not a.stats:
            p.error("rule needs --stats")
        sched = threshold_schedule(json.loads(Path(a.stats).read_text()), a.frame_in_stage)
        print(make_config(sched, "sagepp", {}).split("@")[0])
        return 0
    out = Path(a.out).resolve()
    if a.cmd == "emit":
        if not (a.config and a.emit):
            p.error("emit needs --config and --emit")
        st_path = Path(a.state) if a.state else out / "state.json"
        st = json.loads(st_path.read_text()) if st_path.is_file() else {}
        rec = next((r for r in st.values() if r.get("final") == resolve(a.config)), None)
        print(emit_toml(Path(a.emit), a.name, a.config, rec))
        return 0

    runner = Runner(a)
    doms = [d.name for d in domains(runner.data, a.seed, Path(a.maps) if a.maps else None, a.re10k_cal_list,
                                    a.domains_list)]
    t0 = time.time()
    common = {"reference": a.reference, "domains": doms, "seeds": [a.seed]}
    record = None
    if a.cmd in ("precision", "all"):
        record = precision_search(make_measure(runner, a.seed, a.reference), a.reference, a.low, a.high,
                                  budget("precision"), a.max_restore) | common
        save_state(out, f"precision-s{a.seed}", record)
        print(json.dumps({k: record[k] for k in ("final", "passes", "restored", "final_deltas")}))
    if a.cmd in ("schedule", "all"):
        base = a.base or (record["final"] if record else None)
        if not base:
            p.error("schedule needs --base (a precision-only config) or run `all`")
        record = schedule_search(make_measure(runner, a.seed, a.reference), a.reference, base,
                                 budget("operator")) | common
        save_state(out, f"schedule-s{a.seed}", record)
        print(json.dumps({k: record[k] for k in ("final", "passes", "schedule", "final_deltas")}))
    if a.cmd == "check":
        if not a.config:
            p.error("check needs --config")
        kind = a.budget
        b = budget(kind)
        res = {}
        for s in (int(x) for x in a.seeds.split(",")):
            r = deltas(runner.ensure([a.reference, a.config], s), a.reference)[resolve(a.config)]
            res[s] = {**_brief(r), "passes": passes(r, b)}
            print(f"[check] seed {s}: worst ΔAUC@3 {r['worst3']:+.2f} ({r['worst_domain']}), worst ΔAUC@30 "
                  f"{r['worst30']:+.2f} -> {'PASS' if res[s]['passes'] else 'FAIL'} ({kind} budget -{b[0]}/-{b[1]})")
        n = sum(v["passes"] for v in res.values())
        print(f"[check] {resolve(a.config)}: passes {n}/{len(res)} frame samples")
        record = {"kind": "check", "final": resolve(a.config), "budget": list(b), "per_seed": res,
                  "passes": n == len(res), "final_deltas": res[min(res)], **common, "seeds": sorted(res)}
        save_state(out, f"check-{resolve(a.config)}", record)
    print(f"[calibrate] {runner.evaluated} cells evaluated in this run, bench wall {runner.bench_s:.0f} s, "
          f"total {time.time() - t0:.0f} s; cache {runner.cache_path}")
    if a.emit and record:
        print(emit_toml(Path(a.emit), a.name, record["final"], record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
