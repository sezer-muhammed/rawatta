# SPDX-License-Identifier: Apache-2.0
"""Tests for calibration/calibrate.py: config strings, the worst-case check, both greedy
searches on a synthetic accuracy model, the threshold rule, the TOML output, and (with the
official evaluator available) a full `check` run through eval/tests/fake_rawatta.py.

    python3 -m pytest calibration/tests -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
CAL = HERE.parent
EVAL_TESTS = CAL.parent / "eval" / "tests"
sys.path.insert(0, str(CAL))
sys.path.insert(0, str(EVAL_TESTS))

import calibrate as c  # noqa: E402

CORE = {  # configs/*.toml grammar strings of the three named settings
    "exact": "dense@bf16",
    "rawatta": "dense@sagepp~L4=sage.pvh16/L15=sage.pvh16",
    "rawatta-fast": "L0:frame/L3:merge4/L4:frame/L5:frame/L19:merge4@sagepp~L15=sage.pvh16",
}


def test_config_roundtrip_and_frame_drops_override():
    for g in CORE.values():
        assert c.make_config(*c.parse_config(g)) == g
    _, prec, ov = c.parse_config(CORE["rawatta"])
    fast = c.make_config({0: "frame", 3: "merge4", 4: "frame", 5: "frame", 19: "merge4"}, prec, ov)
    assert fast == CORE["rawatta-fast"]
    assert c.precision_config("sagepp", "sage.pvh16", [15, 4]) == CORE["rawatta"]
    assert c.min_cc("exact") == "8.0" and c.min_cc("rawatta") == "8.9"


DOMS = ("outdoor", "object", "indoor", "rgbd", "dynamic")


def synthetic_measure(penalty):
    """deltas for configs from an additive model: penalty(config) -> {domain: (d3, d30)}."""
    calls = []

    def measure(configs):
        calls.append(list(configs))
        res = {}
        for cfg in configs:
            pen = penalty(c.resolve(cfg))
            res[c.resolve(cfg)] = {d: {"AUC@3": 50 + pen[d][0], "AUC@30": 90 + pen[d][1]} for d in DOMS}
        return c.deltas(res, "exact")
    measure.calls = calls
    return measure


def test_worst_case_check_uses_the_worst_dataset():
    r = {"worst3": -0.49, "worst30": -0.29}
    assert c.passes(r, c.BUDGETS["precision"])
    assert not c.passes({"worst3": -0.51, "worst30": 0.0}, c.BUDGETS["precision"])
    assert not c.passes({"worst3": 0.0, "worst30": -0.31}, c.BUDGETS["precision"])
    assert c.passes({"worst3": -0.99, "worst30": -0.49}, c.BUDGETS["operator"])
    res = {"exact": {d: {"AUC@3": 50, "AUC@30": 90} for d in DOMS},
           "x": {d: {"AUC@3": 50 + (3 if d != "rgbd" else -0.6), "AUC@30": 90} for d in DOMS}}
    dl = c.deltas(res, "exact")["x"]
    assert dl["worst3"] == pytest.approx(-0.6) and dl["worst_domain"] == "rgbd"
    assert dl["mean3"] > 0  # the mean hides the loss; the check does not
    assert not c.passes(dl, c.BUDGETS["precision"])


def test_precision_search_gives_back_largest_gains_first():
    cost = {15: 0.7, 4: 0.5, 5: 0.15, 8: 0.1}  # FP8 P.V cost on the worst domain per layer

    def penalty(cfg):
        if cfg == "dense@bf16":
            return {d: (0.0, 0.0) for d in DOMS}
        _, _, ov = c.parse_config(cfg)
        lost = sum(v for k, v in cost.items() if k not in ov)
        return {d: ((-lost, -0.1) if d == "dynamic" else (-0.1, 0.0)) for d in DOMS}

    rec = c.precision_search(synthetic_measure(penalty), "exact", "sagepp", "sage.pvh16", log=lambda *_: None)
    assert rec["restored"] == [4, 15]
    assert rec["final"] == CORE["rawatta"]
    assert rec["passes"] and rec["final_deltas"]["worst3"] == pytest.approx(-0.25)


def test_schedule_search_frame_first_prune_and_cumulative_budget():
    single = {(0, "frame"): -0.2, (3, "frame"): -2.0, (3, "merge4"): -0.3, (4, "frame"): -0.25,
              (5, "frame"): -0.2, (19, "frame"): -30.0, (19, "merge4"): -0.2, (22, "frame"): -5.0,
              (22, "merge4"): -0.9}

    def penalty(cfg):
        sched, _, _ = c.parse_config(cfg)
        tot = sum(single.get((i, m), -50.0) for i, m in sched.items())
        return {d: ((tot, tot / 4) if d == "indoor" else (0.0, 0.0)) for d in DOMS}

    rec = c.schedule_search(synthetic_measure(penalty), "exact", CORE["rawatta"], log=lambda *_: None)
    assert rec["schedule"] == {"0": "frame", "3": "merge4", "4": "frame", "5": "frame"}
    assert rec["final_deltas"]["worst3"] == pytest.approx(-0.95)
    tries = {s["try"]: s.get("result") for s in rec["steps"] if s["try"]}
    assert tries["L3:frame"].startswith("pruned")
    assert tries["L19:merge4"] == "rejected (cumulative)"


def test_threshold_rule():
    stats = {}
    for i in range(24):
        stats[i] = {"e": 0.9, "e_min": 0.8, "f": 0.1, "r": 1.0}  # diffuse context
    for i in (10, 11, 12, 13, 15, 16):
        stats[i] = {"e": 0.7, "e_min": 0.3, "f": 0.2, "r": 1.6}  # matching
    for i in (19, 21, 23):
        stats[i] = {"e": 0.85, "e_min": 0.45, "f": 0.55, "r": 5.5}  # anchoring
    stats[21]["e_min"] = 0.3  # one selective head keeps it dense
    stats[7]["r"] = 1.0
    stats[17]["e_min"] = 0.7  # late context, a selective head -> dense
    sched = c.threshold_schedule(stats)
    assert sched == {0: "merge4", 1: "merge4", 3: "merge4", 4: "merge4", 5: "merge4", 7: "merge4", 8: "merge4",
                     18: "merge4", 19: "merge4", 22: "merge4", 23: "merge4"}
    assert c.threshold_schedule(stats, frame_in_stage=True)[0] == "frame"


def test_emit_toml_matches_configs_format(tmp_path):
    tomllib = pytest.importorskip("tomllib")
    rec = {"domains": list(DOMS), "seeds": [0], "reference": "exact", "budget": [1.0, 0.5], "passes": True,
           "final_deltas": {"worst3": -0.84, "worst30": -0.35, "worst_domain": "rgbd",
                            "per_domain": {"rgbd": [-0.84, -0.2]}}}
    f = tmp_path / "x.toml"
    c.emit_toml(f, "my-setting", CORE["rawatta-fast"], rec)
    t = tomllib.loads(f.read_text())
    assert set(t) == {"name", "display_name", "grammar", "description", "min_compute_capability", "paper_reference"}
    assert t["grammar"] == CORE["rawatta-fast"] and t["min_compute_capability"] == "8.9"
    assert "frame attention (bf16 kernel) on global blocks 0, 4, 5" in t["description"]
    assert "merge4 (frame 0 complete, quantised kernel in sagepp) on blocks 3, 19" in t["description"]
    assert f"`rawatta run --config {f}`" in f.read_text().splitlines()[2]
    assert "`rawatta run --config <path.toml>`" in c.emit_toml(Path("-"), "x", "dense@bf16")
    assert "quantised kernel in sage)" in c.describe("L3:merge4@bf16")  # merged layers have no bf16 path


def test_map_file_prefers_seeded_50_frame_maps(tmp_path, capsys):
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("common", CAL.parent / "eval" / "data" / "common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    valid = {"seqA": [i for i in range(300) if i % 7]}
    m = common.seeded_maps(valid)
    assert set(m) == {"_n50_s1", "_n50_s2"} and len(m["_n50_s1"]["seqA"]) == 50
    assert set(m["_n50_s1"]["seqA"]) <= set(valid["seqA"]) and m["_n50_s1"] != m["_n50_s2"]
    common.write_maps({**common.frame_maps({"seqA": 300}), **m}, tmp_path, "7scenes_train")
    assert c._map_file(tmp_path, "7scenes_train", 0) == json.loads((tmp_path / "7scenes_train_n50.json").read_text())
    assert c._map_file(tmp_path, "7scenes_train", 1) == m["_n50_s1"]
    (tmp_path / "7scenes_train_n50_s2.json").unlink()
    assert len(c._map_file(tmp_path, "7scenes_train", 2)["seqA"]) == 10  # old data: 10-frame fallback
    assert "--seeded50" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        c._map_file(tmp_path, "tum_dynamic_train", 1)


def test_domains_use_the_paper_calibration_data(tmp_path):
    """Object and indoor domains: the paper's CO3Dv2 and RE10K calibration sequences and frames for
    every frame sample (eval/manifests/paper/calibration.json); outdoor: its four scenes."""
    import run_eval  # noqa: PLC0415

    pc = run_eval.paper_calibration()["domains"]
    for seed in (0, 1, 2):
        doms = {d.name: d for d in c.domains(tmp_path, seed, only=["outdoor", "object", "indoor"])}
        assert doms["object"].fmap == pc["object"]["frames"][str(seed)]
        assert doms["indoor"].fmap == pc["indoor"]["frames"][str(seed)] and len(doms["indoor"].fmap) == 30
        assert doms["outdoor"].scenes == ["bicycle", "garden", "stump", "kitchen"]
    own = c.domains(tmp_path, 1, re10k_cal=pc["indoor"]["sequences"][:3], only=["indoor"])[0]
    assert list(own.fmap) == sorted(pc["indoor"]["sequences"][:3])


def test_help():
    r = subprocess.run([sys.executable, str(CAL / "calibrate.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "worst" in r.stdout and "default: operator" in r.stdout


def _official():
    try:
        import test_eval  # noqa: PLC0415

        return test_eval.OFFICIAL
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _official(), reason="official evaluator / torch / opencv / Pillow unavailable")
def test_check_end_to_end_with_fake_bench(tmp_path):
    import run_eval  # noqa: PLC0415
    import test_eval as te  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    data = tmp_path / "data"
    pc = run_eval.paper_calibration()["domains"]
    for ds, dom in (("co3d", "object"), ("re10k", "indoor")):
        fm = {k: v for k, v in run_eval.frame_map(ds, 0, split="all").items() if k in pc[dom]["sequences"][:2]}
        te.make_omega_dataset(data, ds, {k: max(v) + 1 for k, v in fm.items()}, False, sample=fm)
    rng = np.random.default_rng(0)
    for ds in ("7scenes_train", "tum_dynamic_train"):
        fm = {"seqA": list(range(0, 60, 6))}
        te.make_omega_dataset(data, ds, {"seqA": 60}, True, sample=fm)
        (data / "_frames").mkdir(exist_ok=True)
        (data / "_frames" / f"{ds}_n50.json").write_text(json.dumps(fm))
    for sc in pc["outdoor"]["sequences"][:1]:
        sd = data / "mipnerf360" / sc
        (sd / "images_4").mkdir(parents=True)
        poses = {}
        for i in range(8):
            nm = f"F{i:03d}.JPG"
            Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(sd / "images_4" / nm, "JPEG")
            poses[nm] = np.concatenate([te.rand_rot(rng), rng.standard_normal((3, 1))], 1)
        te.write_images_bin(sd / "sparse/0/images.bin", poses)
    sp_file = tmp_path / "splits_override"  # only the first mip scene exists: restrict via --domains below
    del sp_file
    wrapper = tmp_path / "rawatta"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {EVAL_TESTS / 'fake_rawatta.py'} \"$@\"\n")
    wrapper.chmod(0o755)
    out = tmp_path / "cal"
    r = subprocess.run([sys.executable, str(CAL / "calibrate.py"), "check", "--data", str(data), "--config", "rawatta",
                        "--seeds", "0", "--domains", "object,indoor,rgbd,dynamic", "--rawatta", str(wrapper),
                        "--out", str(out), "--emit", str(tmp_path / "x.toml"), "--name", "x"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "FAIL" in r.stdout  # the fake perturbs every non-exact config by up to 2 degrees
    cells = json.loads((out / "cells.json").read_text())
    assert cells["dense@bf16|rgbd|0"]["AUC@3"] == pytest.approx(100.0)
    assert (tmp_path / "x.toml").is_file()
