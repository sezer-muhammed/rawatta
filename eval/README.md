<!-- SPDX-License-Identifier: Apache-2.0 -->
# Evaluation

Accuracy (and optionally latency) of rawatta configs under the official VGGT-Ω evaluation
protocol, on the paper's six evaluation datasets, plus a container that runs the Official
VGGT-Ω PyTorch code on the same frames for the baseline column.

| piece | what it does |
|---|---|
| `run_eval.py` | builds a `rawatta bench` job per dataset and seed from `manifests/`, runs it, scores it, prints the table (AUC@3, AUC@30, δ1.25, AbsRel, Δ against a reference config, worst-dataset Δ) and writes `results.json` / `table.md` |
| `score.py` | scores bench output with the official evaluator functions (`eval/evaluate.py` per-scene logic: `Dataset.load`, `_source_crop`, `crop_depth`, `resize_depth`, `depth_metrics`, `pose_errors`, AUC pooled over all pairs); COLMAP ground truth for Mip-NeRF 360; optional point-map metrics |
| `omega_src.py` | downloads the official VGGT-Ω source at the pinned commit (`48b23c8`) into a cache and imports its modules; the official code is FAIR-licensed and not vendored |
| `data/` | per-dataset download and conversion scripts (see `DATASETS.md` for sources and licences), `manifest.py write|verify` |
| `manifests/` | frame maps per seed and SHA-256 of every prepared file (extracted from the authors' research codebase); `paper/`: the sequences and per-seed frames of the paper's evaluation and calibration data; `splits.json`: the earlier protocol's calibration lists (excluded by `--subsets full`) |
| `paper_subsets.py` | rebuilds `manifests/paper/` from the manifests with the rules the paper's subsets were made with (`check` / `write`) |
| `official/` | `Dockerfile` (PyTorch 2.9.1, CUDA 12.8, official repo at the pinned commit), `time_official.py` (official `eval/model.py` Predictor: accuracy + CUDA-event latency + peak memory), `run_official.sh` (build + run, optional GPU lock) |

## Setup

```sh
pip install -r eval/requirements.txt        # numpy, opencv-python-headless, Pillow, torch (CPU is fine)
python3 eval/omega_src.py --check           # fetches the official evaluator into ~/.cache/rawatta/
```

Prepare the datasets as described in `DATASETS.md` (only sampled frames are downloaded).

## Run

```sh
# the paper's protocol: six datasets, the paper's sequences, 10 frames per sequence, seeds 0, 1, 2
python3 eval/run_eval.py --data $DATA --datasets paper --configs exact,rawatta,rawatta-fast --seeds 0,1,2

# every manifest sequence instead (RE10K: 190 instead of 38; the other datasets are the same)
python3 eval/run_eval.py --data $DATA --datasets paper --configs exact,rawatta --seeds 0,1,2 --subsets full

# one dataset, write the jobs only
python3 eval/run_eval.py --data $DATA --datasets eth3d --configs exact,rawatta --dry-run

# Official VGGT-Ω on the frames a run used (needs the official .pt checkpoint)
eval/official/run_official.sh -- --data-root /data/eth3d --frames /out/fmap.json --out /out/official-eth3d.json
```

`run_eval.py` deletes the predicted depth/confidence maps after scoring unless `--keep`. Pass
`--reps 5` for timed runs; timings are only meaningful on an otherwise idle GPU.

Paper reference values (Tab. 1 of the paper, exact engine, AUC@3, N = 10, mean over three seeds):
ETH3D 49.91, CO3D 51.21, RE10K 38.49, 7-Scenes 21.11, NRGBD 85.46, TUM-dynamic 39.66; worst-dataset
ΔAUC@3 against the exact engine: RAWATTA −0.51 ± 0.11, RAWATTA-Fast −0.80 ± 0.17.

## The paper's subsets

`manifests/paper/<dataset>.json` lists, for each of the six evaluation datasets, the sequences the
paper's tables used and their frames for seeds 0, 1 and 2 (`calibration.json`: the same for the
five calibration datasets). They were recovered from the authors' experiment records; `--subsets
paper`, the default of `run_eval.py`, evaluates exactly them.

| dataset | sequences | frames for seed s |
|---|---|---|
| ETH3D | 13 (all) | VGGT-Ω's official 10 frames, every seed |
| CO3Dv2 | 40: the 8 categories outside banana and donut | `co3d_s<s>` |
| RealEstate10K | 38: every fifth of the 190 sorted ids outside the earlier 10-sequence calibration list | `re10k_s<s>` |
| 7-Scenes | 18: all test sequences | `7scenes_s<s>` |
| NRGBD | 9 (all) | `nrgbd_s<s>` |
| TUM-dynamic | 8 (all) | `tum_dynamic_s<s>` |

Frame selection is a deterministic function of the sequence's frame count and the seed (10
frames drawn with numpy `default_rng(seed)`, `data/common.py`); every seeded map in `manifests/`
regenerates from the frame counts, and the paper's runs used these maps restricted to the
sequences above. The paper's Supp. Tab. 12 gives 10 7-Scenes test sequences; the records, and the
pair counts behind every 7-Scenes value of Supp. Tab. 10, show that all 18 were used
(`manifests/paper/7scenes.json`). `python3 eval/paper_subsets.py check` confirms that the files
follow the rules; `tests/test_paper_subsets.py` checks them against the recorded sequence lists,
file hashes and sizes.

## Tests

```sh
python3 -m pytest eval/tests -q
```

The pipeline test runs `run_eval.py` end to end with `tests/fake_rawatta.py` in place of the
binary (ground-truth poses for `exact`, perturbed poses otherwise), so it needs no GPU or weights;
tests that call the official evaluator skip themselves when it (or torch/opencv/Pillow) is missing.
