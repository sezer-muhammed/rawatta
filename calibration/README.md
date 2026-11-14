<!-- SPDX-License-Identifier: Apache-2.0 -->
# Calibration

`calibrate.py` reproduces how the paper chose its named settings (Sec. 4.4, "Calibration check
and search") and lets you calibrate your own: candidate configs run through `rawatta bench` on five
calibration datasets, are scored with the official VGGT-Ω evaluator (`eval/score.py`), and are
accepted only if the **worst** calibration dataset stays within an accuracy budget against the exact
engine. The result is written as a config TOML in the `configs/` format.

## The check

Evaluate the complete configuration on all five calibration datasets. With Δ = config − exact
engine per dataset, accept iff

| choosing | worst-dataset ΔAUC@3 | worst-dataset ΔAUC@30 |
|---|---|---|
| numeric formats (`--budget precision`) | ≥ −0.5 | ≥ −0.3 |
| operators (`--budget operator`) | ≥ −1.0 | ≥ −0.5 |

The worst dataset, not the mean: a loss on one dataset must not hide behind gains on the others.
Single-layer costs do not add up, so every accepted step re-measures the whole configuration.

The precision search uses the precision budget and the schedule search the operator budget. The
"Cal." column of the paper's Table 1 (calibration samples, of three, that pass the check) applies
the **operator** budget to every row, RAWATTA included: by Sec. 4.4 ("Protocol sensitivity"),
RAWATTA passes the operator budget on all three samples but the stricter precision budget on the
original sample only. `check` therefore uses the operator budget by default; pass
`--budget precision` for the stricter one.

## Procedure

```sh
export DATA=/path/to/data            # prepared with eval/data/ (see eval/DATASETS.md)
pip install -r eval/requirements.txt

# 1. precision: FP8 P.V everywhere, give layers 16-bit P.V by single-layer gain until the budget holds
python3 calibration/calibrate.py precision --data $DATA --lock /path/to/gpu.lock
# 2. schedule: from the precision result, greedy frame-then-merge4 per layer under the operator budget
python3 calibration/calibrate.py schedule --data $DATA --base "dense@sagepp~L4=sage.pvh16/L15=sage.pvh16"
# 3. check a config on three frame samples (the "Cal." column of Table 1: operator budget, the default)
python3 calibration/calibrate.py check --data $DATA --config rawatta-fast --seeds 0,1,2
# or 1 + 2 + TOML in one go
python3 calibration/calibrate.py all --data $DATA --name my-setting --emit configs/my-setting.toml
```

The TOML that `--emit` writes is used with `rawatta run --config <path.toml>`, or through the
grammar string it contains.

All results are cached in `<out>/cells.json` (default `runs/calibration/`), so an interrupted run
resumes where it stopped, and `state.json` keeps every search step. `--dry-run` writes the bench jobs
and prints the commands without running anything. `--lock FILE` holds an exclusive flock on FILE
around every bench job; each job runs at most `--chunk` (default 6) configs.

`calibrate.py rule --stats stats.json` applies the paper's threshold schedule (Alg. 1) to per-block
attention statistics `{layer: {e, e_min, f, r}}` (normalised entropy, least diffuse head, frame-0
mass, its ratio to uniform). The engine does not export these statistics yet, so this subcommand
only implements the rule.

## Calibration data

| domain | data | sequences × frames |
|---|---|---|
| outdoor | Mip-NeRF 360 bicycle, garden, stump, kitchen (COLMAP ground truth) | 4 × 50 |
| object | CO3Dv2 banana and donut (the evaluation uses the 8 other categories) | 10 × 10 |
| indoor | RealEstate10K sequences outside the evaluation split | see below × 10 |
| rgbd | 7-Scenes train: first `TrainSplit` sequence of each scene | 7 × 50 |
| dynamic | TUM fr3 `*_validation` takes (walking xyz/halfsphere/rpy, sitting xyz) | 4 × 50 |

Frame sample 0 (`--seed 0`) is the search's own: evenly spaced 50 frames for the 50-frame domains,
the manifests' `s0` maps for CO3Dv2 and RE10K. Samples 1 and 2 use seeded random frames: Mip-NeRF 360
draws 50 of all frames; CO3Dv2/RE10K use the `s1`/`s2` maps; the 7-Scenes and TUM train splits use
50 random frames with a finite pose (`<ds>_n50_s1`/`_s2`, the sampling of the paper's three-sample
check), which the data scripts write and store when run with `--seeded50` (see
`eval/DATASETS.md`). Data prepared without `--seeded50` falls back to the 10-frame `s1`/`s2` maps,
with a warning.

Deviations from the paper that cannot be closed from the authors' research codebase: the paper's
RE10K calibration used 30 sequences outside its 38-sequence evaluation subset, and neither list was
kept (`eval/DATASETS.md`); the CO3Dv2 calibration frames of the paper (its `co3d_train` map) were not
kept either, so the object domain uses the manifest's seeded maps of the 10 banana/donut sequences. By default the indoor domain uses the 10 RE10K sequences of the earlier
calibration protocol (`eval/manifests/splits.json`); pass your own list with `--re10k-cal FILE`.
Searches run with these defaults are therefore not guaranteed to return exactly the paper's
settings.

## Cost

Counts, from the procedure. Brackets give what the paper states, or a count derived from it
(marked *derived*):

- one config = one bench run per domain over 35 sequences on the default set (4 + 7 + 4 at 50
  frames, 10 + 10 at 10 frames; 55 with the paper's 30 RE10K sequences) plus scoring; the reference
  (exact engine) is measured once per frame sample;
- precision search: 1 base + 19 single-layer configs, then one config per accepted or tried layer
  [paper, Sec. 4.4: the search "stops after two layers, block 15 and block 4". *Derived*, not stated
  in the paper: by this procedure that is 21 configs (exact reference + base + 19 single-layer)
  on the 5 datasets, then 2 cumulative steps];
- schedule search: 1 base + 38 single-layer configs (19 layers × frame/merge4), then one config per
  greedy trial that survives pruning [paper, Sec. 4.4: "49 configurations (38 single-layer
  evaluations, 11 greedy trials)"];
- check on three frame samples: 2 configs × 3 samples (the reference is shared).

GPU time is dominated by the 50-frame sequences. For scale, one RAWATTA forward at 50 frames takes
1734 ms on an RTX 4070 SUPER (paper Table 1; not measured in this build); bench model loading,
decoding and scoring come on top.
Wall-clock numbers for a full search on this release are pending (they need the weights).

## Tests

```sh
python3 -m pytest calibration/tests -q
```

They check the config strings against `configs/*.toml`, the worst-dataset rule, both searches on a
synthetic additive accuracy model, the threshold rule, the TOML keys and comment, the choice of
frame maps, and (when the official evaluator is available) a full `check` run with
`eval/tests/fake_rawatta.py` standing in for the binary.
