# SPDX-License-Identifier: Apache-2.0
"""Tests for the paper's evaluation and calibration subsets (eval/paper_subsets.py,
eval/manifests/paper/) and their use by eval/run_eval.py.

The recorded values below come from the authors' experiment records (sequence lists, file hashes
and sizes) and from the paper (Supp. Tab. 10); the tests check that the rules and the manifests
reproduce them.

    python3 -m pytest eval/tests/test_paper_subsets.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))

import paper_subsets as ps  # noqa: E402
import run_eval  # noqa: E402

# The RE10K evaluation sequences, in the order the per-sequence records of the paper's runs list them.
RECORDED_RE10K_EVAL = [
    "10c551ef9644ea03", "178cef169356d4a5", "1fcc400e42725a95", "24ad46fd2b26b208", "2bff9ec89ca982c9",
    "3558c5283169e6ca", "3aaed2e6422d7d57", "4393f3a15ed6fb9a", "4dc410cbe3dd4d2a", "5179e23afde37cfd",
    "5451cefde53f06f1", "575fc7cbb9b2d352", "5d9d555bb2294681", "64f9a0b6b7667ce7", "6d4b3603c5edee50",
    "70888447007609c0", "75c7801702268d0d", "7a34348316608aee", "840b7acb6b1d5165", "888c0f8b1eb412e9",
    "8f0218121c335789", "9ab639ce124d7b51", "a29f700543ac753b", "aa4a4768c8635fde", "b43aa92e530e2aa5",
    "bb2f6009491cea51", "bf800a3d2af8f88c", "c4ed784ee6c885e5", "cbc66cef2ae88877", "d08577a59254fa98",
    "d42055153cb42cfa", "d7d2e3d202d77bc0", "df6aa9c2baa43f3a", "e5dc905f2d09e08f", "e7a91fda857970df",
    "ec1e63259aa2d61a", "f8d39b718e9986dc", "fb7719fc97e9f6e8",
]
# SHA-256 of the earlier protocol's seed-0 evaluation maps (CO3Dv2: 40 sequences, RE10K: 190), the
# parents of the paper's subsets, as kept in the authors' research codebase (json indent 1, sorted keys).
RECORDED_EVAL_S0_SHA256 = {
    "co3d": "a1f9fad86a8019506ef72cc8981645e8e879d66006818fb356256f8db8857363",
    "re10k": "f25abd24957fb9f8dcaf1a70fa368cff569690e29524115aa55bd9509b328449",
}
# Sizes in bytes of the paper's CO3Dv2 and RE10K calibration maps, recorded when they were written
# (json.dump with indent 1, sequences in the order of the rule).
RECORDED_CALIBRATION_BYTES = {"co3d": 931, "re10k": 2546}
# Supp. Tab. 10 of the paper (seed 0): AUC@3 of the 14 configurations on 7-Scenes and on RE10K.
SEED0_AUC3 = {
    "7scenes": [22.63, 21.44, 22.39, 22.63, 22.76, 23.05, 23.17, 22.39, 21.89, 21.65, 21.52, 21.81, 21.03, 21.23],
    "re10k": [38.09, 37.04, 37.76, 37.88, 37.99, 37.74, 37.95, 37.08, 38.13, 37.47, 37.82, 37.76, 36.67, 36.86],
}
PAPER_COUNTS = {"eth3d": 13, "co3d": 40, "re10k": 38, "7scenes": 18, "nrgbd": 9, "tum_dynamic": 8}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _common():
    spec = importlib.util.spec_from_file_location("common", EVAL / "data" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_files_match_the_rules():
    r = subprocess.run([sys.executable, str(EVAL / "paper_subsets.py"), "check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_counts_frames_and_seeds():
    for ds, n in PAPER_COUNTS.items():
        sub = run_eval.paper_subset(ds)
        assert len(sub["sequences"]) == n and sub["seeds"] == [0, 1, 2]
        for s in ("0", "1", "2"):
            fr = sub["frames"][s]
            assert list(fr) == sub["sequences"]
            assert all(len(v) == 10 and len(set(v)) == 10 for v in fr.values())
            if ds != "eth3d":  # the official ETH3D frames keep their order (the first is the reference)
                assert all(v == sorted(v) for v in fr.values())
            if ds == "eth3d":  # official frames, the same for every seed
                assert fr == run_eval.manifest("eth3d")["frame_maps"]["eth3d.json"]["map"]
            else:  # the manifest's seed-s map restricted to the subset
                full = run_eval.manifest(ds)["frame_maps"][f"{ds}_s{s}.json"]["map"]
                assert fr == {k: full[k] for k in sub["sequences"]}
    assert run_eval.paper_subset("7scenes")["frames"]["0"] != run_eval.paper_subset("7scenes")["frames"]["1"]


def test_frames_regenerate_from_frame_counts():
    """Frame selection is a deterministic function of (frame count, seed): eval/data/common.py."""
    common = _common()
    for ds in ("co3d", "re10k", "7scenes", "nrgbd", "tum_dynamic"):
        sub = run_eval.paper_subset(ds)
        counts = {k: v["frames"] for k, v in run_eval.manifest(ds)["scenes"].items() if k in sub["sequences"]}
        maps = common.frame_maps(counts)
        for s in ("0", "1", "2"):
            assert maps[f"_s{s}"] == sub["frames"][s], (ds, s)


def test_manifest_maps_are_the_hashed_files():
    for ds in ("co3d", "re10k", "7scenes", "nrgbd", "tum_dynamic"):
        for name, m in run_eval.manifest(ds)["frame_maps"].items():
            assert _sha(json.dumps(m["map"], indent=1, sort_keys=True) + "\n") == m["sha256"], name


def test_re10k_rule_reproduces_the_recorded_sequences():
    assert ps.evaluation_sequences()["re10k"] == RECORDED_RE10K_EVAL
    assert run_eval.paper_subset("re10k")["sequences"] == RECORDED_RE10K_EVAL


def test_parent_maps_reproduce_the_recorded_hashes():
    for ds, want in RECORDED_EVAL_S0_SHA256.items():
        m = run_eval.frame_map(ds, 0, subsets="full")  # outside the earlier calibration lists
        assert _sha(json.dumps(m, indent=1, sort_keys=True)) == want, ds
    # the paper's subsets are taken from these maps, frames unchanged
    for ds in ("co3d", "re10k"):
        parent = run_eval.frame_map(ds, 0, subsets="full")
        sub = run_eval.paper_subset(ds)["frames"]["0"]
        assert all(parent[k] == v for k, v in sub.items())


def test_calibration_rules_reproduce_the_recorded_files():
    s0 = {ds: ps.manifest_map(ds, 0) for ds in ("co3d", "re10k")}
    co3d = {k: v for k, v in s0["co3d"].items() if k.startswith(("banana", "donut"))}
    order = ps.re10k_calibration(s0["re10k"], ps.evaluation_sequences()["re10k"], ps.earlier_calibration("re10k"))
    re10k = {k: s0["re10k"][k] for k in order}
    assert len(json.dumps(co3d, indent=1)) == RECORDED_CALIBRATION_BYTES["co3d"]
    assert len(json.dumps(re10k, indent=1)) == RECORDED_CALIBRATION_BYTES["re10k"]
    cal = run_eval.paper_calibration()["domains"]
    assert sorted(cal["object"]["sequences"]) == sorted(co3d)
    assert sorted(cal["indoor"]["sequences"]) == sorted(order) and len(order) == 30
    assert order[:10] == run_eval.splits()["re10k"]["calibration"]  # the earlier list comes first
    for dom, ds in (("object", "co3d"), ("indoor", "re10k")):
        for s in ("0", "1", "2"):
            full = run_eval.manifest(ds)["frame_maps"][f"{ds}_s{s}.json"]["map"]
            assert cal[dom]["frames"][s] == {k: full[k] for k in sorted(cal[dom]["sequences"])}


def test_evaluation_is_sequence_disjoint_from_calibration():
    cal = run_eval.paper_calibration()["domains"]
    assert not set(cal["object"]["sequences"]) & set(run_eval.paper_subset("co3d")["sequences"])
    assert not set(cal["indoor"]["sequences"]) & set(run_eval.paper_subset("re10k")["sequences"])
    assert not set(cal["rgbd"]["sequences"]) & set(run_eval.paper_subset("7scenes")["sequences"])
    tum_eval = set(run_eval.paper_subset("tum_dynamic")["sequences"])
    assert {s.removesuffix("_validation") for s in cal["dynamic"]["sequences"]} <= tum_eval  # other takes
    assert not set(cal["dynamic"]["sequences"]) & tum_eval
    assert cal["outdoor"]["sequences"] == run_eval.splits()["mipnerf360"]["calibration"]
    assert sorted(cal["object"]["sequences"]) == sorted(run_eval.splits()["co3d"]["calibration"])


def _on_grid(v: float, pairs: int) -> bool:
    """A 2-decimal AUC@3 that is k * 100 / (3 * pairs) for some integer k (the official AUC)."""
    step = 100 / (3 * pairs)
    return abs(v - round(v / step) * step) <= 0.005 + 1e-9


def test_paper_values_fit_the_subset_sizes():
    """Every 7-Scenes AUC@3 of Supp. Tab. 10 fits 18 sequences (810 pairs), none fits 10 (450):
    the paper's "10 test sequences" (Supp. Tab. 12) is a wrong count. RE10K fits 38 (1710 pairs)."""
    assert all(_on_grid(v, 18 * 45) for v in SEED0_AUC3["7scenes"])
    assert not any(_on_grid(v, 10 * 45) for v in SEED0_AUC3["7scenes"])
    assert all(_on_grid(v, 38 * 45) for v in SEED0_AUC3["re10k"])


def test_run_eval_paper_and_full_subsets():
    for ds, n in PAPER_COUNTS.items():
        for s in (0, 1, 2):
            assert run_eval.frame_map(ds, s, subsets="paper") == run_eval.paper_subset(ds)["frames"][str(s)]
    assert len(run_eval.frame_map("re10k", 0, subsets="full")) == 190
    assert len(run_eval.frame_map("re10k", 0, split="all", subsets="full")) == 200
    # an explicit map is restricted to the paper's sequences
    assert run_eval.frame_map("re10k", 2, "s2", subsets="paper") == run_eval.paper_subset("re10k")["frames"]["2"]
    n50 = run_eval.frame_map("7scenes", 0, "n50", subsets="paper")
    assert len(n50) == 18 and all(len(v) == 50 for v in n50.values())
    with pytest.raises(SystemExit):
        run_eval.frame_map("re10k", 3, subsets="paper")
    # datasets outside the paper's six are not affected
    assert run_eval.PAPER == tuple(PAPER_COUNTS)


def test_run_eval_dry_run_uses_the_paper_subsets(tmp_path):
    data = tmp_path / "data"
    # two evaluation sequences, one of the paper's calibration sequences that the full set keeps
    # (10e9...) and one of the earlier calibration list that both drop (004e...)
    scenes = RECORDED_RE10K_EVAL[:2] + ["10e931268a81f228", "004e9db3337e8206"]
    counts = run_eval.manifest("re10k")["scenes"]
    for sc in scenes:
        d = data / "re10k" / sc
        (d / "images").mkdir(parents=True)
        (d / "cameras.npz").write_bytes(b"")
        names = [f"{i}.jpg" for i in range(counts[sc]["frames"])]
        (d / "frames.txt").write_text("\n".join(names) + "\n")
        for nm in names:
            (d / "images" / nm).write_bytes(b"")
    for subsets, want in (("paper", scenes[:2]), ("full", scenes[:3])):
        out = tmp_path / subsets
        r = subprocess.run([sys.executable, str(EVAL / "run_eval.py"), "--data", str(data), "--datasets", "re10k",
                            "--configs", "exact", "--seeds", "1", "--subsets", subsets, "--out", str(out),
                            "--dry-run"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        used = json.loads((out / "re10k-s1" / "fmap.json").read_text())
        assert sorted(used) == sorted(want)
        if subsets == "paper":
            assert used == {k: run_eval.paper_subset("re10k")["frames"]["1"][k] for k in want}
