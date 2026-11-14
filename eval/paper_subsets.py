#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The sequences and per-seed frame maps of the paper's evaluation (N = 10, seeds 0, 1, 2) and
calibration data, rebuilt from the frame manifests with the rules the authors recorded when they
made each subset (recovered from the authors' experiment records), written to manifests/paper/.

    python3 eval/paper_subsets.py check     # manifests/paper/*.json equal what the rules give
    python3 eval/paper_subsets.py write     # rewrite them

Frame selection itself is deterministic per (sequence, seed): the data scripts draw 10 frames
with numpy default_rng(seed) (eval/data/common.py, sample_random), and the manifests keep the
resulting maps `<ds>_s0/_s1/_s2` with the SHA-256 of each map file. The paper's runs used those
maps restricted to the subsets below (seed s -> map s<s>); ETH3D uses VGGT-Omega's official
frames for every seed.

Evaluation (manifests/paper/<dataset>.json):

| dataset | sequences | rule |
|---|---|---|
| eth3d | 13 | all scenes, official frames for every seed |
| co3d | 40 | the 8 categories outside the calibration categories banana and donut |
| re10k | 38 | every fifth of the 190 sorted ids outside the earlier 10-sequence calibration list |
| 7scenes | 18 | all TestSplit sequences (the paper's Supp. Tab. 12 says 10: manifests/paper/7scenes.json) |
| nrgbd | 9 | all scenes |
| tum_dynamic | 8 | all fr3 dynamic sequences |

Calibration (manifests/paper/calibration.json):

| domain | data | rule |
|---|---|---|
| outdoor | Mip-NeRF 360 bicycle, garden, stump, kitchen | 50 frames, run_eval.colmap_pick (seed 0: evenly spaced) |
| object | CO3Dv2 banana + donut, 10 sequences | co3d_s<seed> restricted to them |
| indoor | RealEstate10K, 30 sequences | the earlier 10-sequence list + the first 20 sorted ids outside it and the 38 |
| rgbd | 7-Scenes train: first TrainSplit sequence per scene | 7scenes_train_n50 (seed 0), _n50_s1/_s2 (--seeded50) |
| dynamic | TUM fr3 *_validation takes (4) | tum_dynamic_train_n50 (seed 0), _n50_s1/_s2 (--seeded50) |
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"
OUT = MANIFESTS / "paper"
SEEDS = (0, 1, 2)
EVAL_DATASETS = ("eth3d", "co3d", "re10k", "7scenes", "nrgbd", "tum_dynamic")

# Sequences of the 50-frame calibration domains, as the data scripts name them (the frames are
# chosen per frame with a finite pose by eval/data/sevenscenes.py / tum_dynamic.py --seeded50,
# so they are not part of the manifests).
SEVENSCENES_TRAIN = ["chess_seq-01", "fire_seq-01", "heads_seq-02", "office_seq-01", "pumpkin_seq-02",
                     "redkitchen_seq-01", "stairs_seq-02"]
TUM_TRAIN = ["sitting_xyz_validation", "walking_halfsphere_validation", "walking_rpy_validation",
             "walking_xyz_validation"]

NAMES = {"eth3d": "ETH3D", "co3d": "CO3Dv2", "re10k": "RealEstate10K", "7scenes": "7-Scenes", "nrgbd": "NRGBD",
         "tum_dynamic": "TUM-dynamic"}


# --------------------------------------------------------------------------
# the recorded rules
# --------------------------------------------------------------------------

def manifest_map(ds: str, seed: int) -> dict[str, list[int]]:
    """The manifest's seed-`seed` map of one dataset (ETH3D: the official map), sorted by sequence."""
    m = json.loads((MANIFESTS / f"{ds}.json").read_text())["frame_maps"]
    if ds == "eth3d":
        return dict(m["eth3d.json"]["map"])
    return dict(sorted(m[f"{ds}_s{seed}.json"]["map"].items()))


def earlier_calibration(ds: str) -> list[str]:
    """The calibration list of the earlier protocol (manifests/splits.json)."""
    return list(json.loads((MANIFESTS / "splits.json").read_text())[ds]["calibration"])


def co3d_calibration(sequences) -> list[str]:
    """CO3Dv2 calibration: the banana and donut sequences."""
    return sorted(k for k in sequences if k.startswith(("banana", "donut")))


def co3d_evaluation(sequences) -> list[str]:
    """CO3Dv2 evaluation: the 40 sequences of the 8 other categories."""
    cal = set(co3d_calibration(sequences))
    return sorted(k for k in sequences if k not in cal)


def re10k_evaluation(sequences, earlier_cal) -> list[str]:
    """RE10K evaluation (38): every fifth id of the sorted ids outside the earlier calibration list
    (the earlier protocol's 190-sequence evaluation map), as the subset was made."""
    cal = set(earlier_cal)
    return sorted(k for k in sequences if k not in cal)[::5]


def re10k_calibration(sequences, evaluation, earlier_cal) -> list[str]:
    """RE10K calibration (30), in the order it was written: the earlier 10-sequence list, then the
    first 20 sorted ids that are neither in it nor in the evaluation subset."""
    ev, cal = set(evaluation), list(earlier_cal)
    rest = [k for k in sorted(sequences) if k not in ev and k not in cal]
    return cal + rest[:20]


def restrict(m: dict[str, list[int]], keys) -> dict[str, list[int]]:
    keep = set(keys)
    return {k: v for k, v in m.items() if k in keep}


# --------------------------------------------------------------------------
# the files
# --------------------------------------------------------------------------

def evaluation_sequences() -> dict[str, list[str]]:
    s0 = {ds: manifest_map(ds, 0) for ds in EVAL_DATASETS}
    return {
        "eth3d": list(s0["eth3d"]),
        "co3d": co3d_evaluation(s0["co3d"]),
        "re10k": re10k_evaluation(s0["re10k"], earlier_calibration("re10k")),
        "7scenes": sorted(s0["7scenes"]),
        "nrgbd": sorted(s0["nrgbd"]),
        "tum_dynamic": sorted(s0["tum_dynamic"]),
    }


RULES = {
    "eth3d": "All 13 training scenes of the ETH3D high-res multi-view benchmark with VGGT-Omega's official "
             "10-frame map (eval/manifests/eth3d.json); the same frames for every seed.",
    "co3d": "The 40 sequences of eval/manifests/co3d.json outside the calibration categories banana and donut "
            "(8 categories x 5 sequences). Seed s: the manifest's co3d_s<s> map restricted to them.",
    "re10k": "Every fifth sequence id, in sorted order, of the 190 ids of eval/manifests/re10k.json outside the "
             "earlier 10-sequence calibration list (eval/manifests/splits.json). Seed s: the manifest's re10k_s<s> "
             "map restricted to them.",
    "7scenes": "All 18 TestSplit sequences of the 7 scenes. Seed s: the manifest's 7scenes_s<s> map.",
    "nrgbd": "All 9 Neural RGB-D synthetic scenes. Seed s: the manifest's nrgbd_s<s> map.",
    "tum_dynamic": "All 8 TUM RGB-D fr3 walking/sitting sequences (xyz, static, halfsphere, rpy). Seed s: the "
                   "manifest's tum_dynamic_s<s> map.",
}

PROVENANCE = {
    "eth3d": "Recovered from the authors' experiment records: the paper's runs list these 13 scenes and used the "
             "official frames for seeds 0, 1 and 2 (only those images were prepared; the engine is deterministic, "
             "so the seed standard deviation is 0).",
    "co3d": "Recovered from the authors' experiment records: the per-sequence records of the paper's runs list "
            "exactly these 40 sequences, and the seed-0 map equals, byte for byte, the earlier protocol's CO3Dv2 "
            "evaluation map kept in the authors' research codebase. Seeds 1 and 2 restrict the s1/s2 maps to the "
            "seed-0 sequences, as the paper's runs did.",
    "re10k": "Recovered from the authors' experiment records: the subset was made with exactly this rule from the "
             "earlier protocol's 190-sequence map, and the per-sequence records of the paper's runs list these 38 "
             "ids in this order. The pair counts agree: every RE10K AUC@3 of Supp. Tab. 10 (seed 0) equals, to its "
             "two decimals, a multiple of 100/(3 x 1710), 1710 = 38 x 45 pairs. Seeds 1 and 2 restrict the s1/s2 "
             "maps to these sequences, as the paper's runs did.",
    "7scenes": "Recovered from the authors' experiment records: the paper's runs used the full 7scenes_s<seed> "
               "maps, and their per-sequence records list all 18 test sequences. Supp. Tab. 12 of the paper says 10 "
               "test sequences; that count is wrong: each of the 14 7-Scenes AUC@3 values of Supp. Tab. 10 "
               "(seed 0) equals, to its two decimals, a multiple of 100/(3 x 810), 810 = 18 x 45 pairs, and none "
               "equals a multiple of 100/(3 x 450), the 10-sequence pair count.",
    "nrgbd": "Recovered from the authors' experiment records: the paper's runs used the full nrgbd_s<seed> maps "
             "(9 scenes in their per-sequence records).",
    "tum_dynamic": "Recovered from the authors' experiment records: the paper's runs used the full "
                   "tum_dynamic_s<seed> maps (8 sequences in their per-sequence records).",
}


def evaluation_file(ds: str, seqs: list[str]) -> dict:
    frames = {str(s): restrict(manifest_map(ds, s), seqs) for s in SEEDS}
    if ds == "eth3d":  # keep the official map's order
        frames = {str(s): dict(manifest_map("eth3d", 0)) for s in SEEDS}
    return {"dataset": ds, "name": NAMES[ds], "role": "evaluation", "frames_per_sequence": 10,
            "seeds": list(SEEDS), "sequences": seqs, "rule": RULES[ds], "provenance": PROVENANCE[ds],
            "frames": frames}


def calibration_file(eval_re10k: list[str]) -> dict:
    co3d = co3d_calibration(manifest_map("co3d", 0))
    re10k = re10k_calibration(manifest_map("re10k", 0), eval_re10k, earlier_calibration("re10k"))
    mip = earlier_calibration("mipnerf360")
    return {"role": "calibration", "seeds": list(SEEDS), "domains": {
        "outdoor": {
            "dataset": "mipnerf360", "layout": "colmap", "sequences": mip, "frames_per_sequence": 50,
            "rule": "COLMAP ground truth, images_4. Seed 0: 50 evenly spaced frames; seeds 1 and 2: 50 frames drawn "
                    "with random.Random(seed) from all frames, sorted (run_eval.colmap_pick, the rule of the "
                    "paper's evaluation driver). The frames depend only on the number of images of each scene.",
        },
        "object": {
            "dataset": "co3d", "sequences": co3d, "frames_per_sequence": 10,
            "rule": "The banana and donut sequences of eval/manifests/co3d.json. Seed s: the manifest's co3d_s<s> "
                    "map restricted to them.",
            "provenance": "Recovered from the authors' experiment records: the paper's calibration map was written "
                          "with exactly this rule, and its recorded file size is reproduced byte for byte.",
            "frames": {str(s): restrict(manifest_map("co3d", s), co3d) for s in SEEDS},
        },
        "indoor": {
            "dataset": "re10k", "sequences": sorted(re10k), "frames_per_sequence": 10,
            "rule": "The earlier 10-sequence calibration list (eval/manifests/splits.json) plus the first 20 "
                    "sorted ids of eval/manifests/re10k.json that are neither in it nor in the 38-sequence "
                    "evaluation subset. Seed s: the manifest's re10k_s<s> map restricted to them.",
            "provenance": "Recovered from the authors' experiment records: the paper's calibration map was written "
                          "with exactly this rule, and its recorded file size is reproduced byte for byte. The "
                          "calibration records of the paper hold 1350 = 30 x 45 pairs on this dataset.",
            "frames": {str(s): restrict(manifest_map("re10k", s), re10k) for s in SEEDS},
        },
        "rgbd": {
            "dataset": "7scenes_train", "sequences": SEVENSCENES_TRAIN, "frames_per_sequence": 50,
            "rule": "The first TrainSplit sequence of each scene (eval/data/sevenscenes.py --split train). Seed 0: "
                    "7scenes_train_n50 (50 evenly spaced frames with a finite pose); seeds 1 and 2: "
                    "7scenes_train_n50_s<seed> (50 frames with a finite pose drawn with numpy default_rng(seed); "
                    "written with --seeded50).",
        },
        "dynamic": {
            "dataset": "tum_dynamic_train", "sequences": TUM_TRAIN, "frames_per_sequence": 50,
            "rule": "The four fr3 *_validation takes (eval/data/tum_dynamic.py --seqs ...). Seed 0: "
                    "tum_dynamic_train_n50; seeds 1 and 2: tum_dynamic_train_n50_s<seed> (written with --seeded50).",
        },
    }}


def build() -> dict[str, dict]:
    seqs = evaluation_sequences()
    files = {f"{ds}.json": evaluation_file(ds, seqs[ds]) for ds in EVAL_DATASETS}
    files["calibration.json"] = calibration_file(seqs["re10k"])
    return files


def dumps(obj) -> str:
    """JSON with one line per frame list."""
    s = json.dumps(obj, indent=1, ensure_ascii=False)
    s = re.sub(r"\[\s*(-?\d+(?:,\s*-?\d+)*)\s*\]",
               lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", s)
    return s + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "write"))
    a = ap.parse_args()
    bad = []
    for name, obj in build().items():
        p = OUT / name
        text = dumps(obj)
        if a.cmd == "write":
            OUT.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            print(p)
        elif not p.is_file() or p.read_text() != text:
            bad.append(name)
    if bad:
        print(f"out of date: {', '.join(bad)} (run `python3 eval/paper_subsets.py write`)", file=sys.stderr)
        return 1
    if a.cmd == "check":
        print(f"ok: {len(build())} files in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
