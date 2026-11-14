<!-- SPDX-License-Identifier: Apache-2.0 -->
# Evaluation and calibration datasets

The scripts in `eval/data/` download every dataset from its original source (or, where the
original source has no pixels, from the source the paper used, marked below) and convert it
into the official VGGT-Ω evaluation layout. **Nothing is redistributed by this repository**:
it ships only frame maps and SHA-256 manifests (`eval/manifests/`). Check each licence before
use; several are non-commercial.

Licences were read from the pages linked below on 2026-11-13. "Not stated" means we found no
licence on the source page; that does not make the data free to use.

| dataset | used for | licence (as stated by the source) | download source used by the script | script |
|---|---|---|---|---|
| ETH3D (high-res multi-view, 13 training scenes) | evaluation | CC BY-NC-SA 4.0 ([eth3d.net](https://www.eth3d.net/)) | Hugging Face mirror `Livioni/eth3d_omnivggt` @ `64154bd`, the mirror pinned by the official VGGT-Ω `eval/prepare.py` (no dataset card); `prepare_official.py` can instead use the official ETH3D archives | `eth3d.py` (+ `eth3d_fetch.py`) |
| MPI Sintel (training, 16 scenes) | not in the paper's tables; supported | MPI non-commercial research licence, no redistribution ([sintel.is.tue.mpg.de/license](http://sintel.is.tue.mpg.de/license)) | official zips `files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip`, `.../MPI-Sintel-depth-training-20150305.zip` (range reads) | `sintel.py` (+ `sintel_fetch.py`) |
| CO3Dv2 (10 categories x 5 sequences) | evaluation (8 categories), calibration (banana, donut) | CC BY-NC 4.0 (`LICENSE` of [facebookresearch/co3d](https://github.com/facebookresearch/co3d); its README states this for the codebase, check the terms shown at download) | `dl.fbaipublicfiles.com/co3dv2_231130/` via the release's `links.json` (range reads) | `co3d.py` |
| RealEstate10K (200 test sequences) | evaluation, calibration | camera data: CC BY 4.0, Google LLC ([download page](https://google.github.io/realestate10k/download.html)); frames come from YouTube videos and remain their owners' | **not the original source**: the original release has poses and YouTube ids only; the script reads pixelSplat's processed test chunks from the Hugging Face mirror `lhmd/re10k_torch` (card says MIT; that cannot relicense the frames) | `re10k.py` |
| 7-Scenes (Microsoft) | evaluation (test sequences), calibration (`--split train`) | non-commercial use only, Microsoft licence agreement ([project page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)) | `download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/<scene>.zip` (range reads) | `sevenscenes.py`, `sevenscenes_long.py` |
| Neural RGB-D synthetic scenes (9) | evaluation | per scene CC BY or CC0 (BlendSwap assets; table in [the authors' README](https://github.com/dazinovic/neural-rgbd-surface-reconstruction)) | `kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip` (range reads) | `nrgbd.py` |
| TUM RGB-D fr3 dynamic (8 sequences; 4 `_validation` takes for calibration) | evaluation, calibration | CC BY 4.0 ([TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)) | `cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_<seq>.tgz` | `tum_dynamic.py` |
| Mip-NeRF 360 (bicycle, garden, stump, kitchen) | calibration (outdoor), timing (kitchen) | not stated ([project page](https://jonbarron.info/mipnerf360/)) | `storage.googleapis.com/gresearch/refraw360/360_v2.zip` (range reads of the chosen scenes) | `mipnerf360.py` |

## Preparing

```sh
export DATA=/path/to/data          # e.g. /data on the build machine
pip install -r eval/requirements.txt
python3 eval/data/eth3d.py       --out $DATA/eth3d                       # ~1.8 GB
python3 eval/data/co3d.py        --out $DATA/co3d        --maps $DATA/_frames
python3 eval/data/re10k.py       --out $DATA/re10k       --tmp $DATA/tmp --maps $DATA/_frames
python3 eval/data/sevenscenes.py --out $DATA/7scenes     --tmp $DATA/tmp --maps $DATA/_frames
python3 eval/data/nrgbd.py       --out $DATA/nrgbd       --maps $DATA/_frames
python3 eval/data/tum_dynamic.py --out $DATA/tum_dynamic --tmp $DATA/tmp --maps $DATA/_frames
# calibration only
python3 eval/data/sevenscenes.py --split train --seeded50 --name 7scenes_train --out $DATA/7scenes_train --tmp $DATA/tmp --maps $DATA/_frames
python3 eval/data/tum_dynamic.py --seqs walking_xyz_validation,walking_halfsphere_validation,walking_rpy_validation,sitting_xyz_validation \
    --seeded50 --name tum_dynamic_train --out $DATA/tum_dynamic_train --tmp $DATA/tmp --maps $DATA/_frames
python3 eval/data/mipnerf360.py  --out $DATA/mipnerf360
# check against the paper's manifests (files and regenerated frame maps)
python3 eval/data/manifest.py verify $DATA/eth3d eval/manifests/eth3d.json
python3 eval/data/manifest.py verify $DATA/co3d  eval/manifests/co3d.json --maps $DATA/_frames
```

`--seeded50` adds the 50-frame seeded maps (`<name>_n50_s1`, `_s2`) that frame samples 1 and 2 of
`calibration/calibrate.py check` use for these two train splits, and stores their frames. The train
splits have no manifest.

Sizes of the prepared subsets (from the manifests): ETH3D 1.79 GB, CO3Dv2 0.15 GB, RE10K 0.32 GB,
7-Scenes 0.69 GB, NRGBD 0.36 GB, TUM-dynamic 0.36 GB. Only sampled frames are stored.

Scripts that call the official VGGT-Ω `eval/prepare.py` or `eval/datasets.py` (ETH3D, Sintel,
`shrink_depth.py`) fetch the official source at the pinned commit into a cache at run time
(`eval/omega_src.py`); that code is under the FAIR Noncommercial Research License and is not part
of this repository.

## The paper's subsets

The paper's tables used subsets of these datasets. They were recovered from the authors'
experiment records and are released in `eval/manifests/paper/` with the frames of every sequence
for seeds 0, 1 and 2 (rules and evidence in each file and in `eval/paper_subsets.py`):

| dataset | evaluation (`<dataset>.json`) | calibration (`calibration.json`) |
|---|---|---|
| ETH3D | 13 scenes, official frames | - |
| CO3Dv2 | 40 sequences, 8 categories | banana and donut, 10 sequences × 10 frames |
| RealEstate10K | 38 sequences: every fifth of the 190 sorted ids outside the earlier 10-sequence calibration list | 30 sequences: that list plus the first 20 sorted ids outside it and outside the 38 |
| 7-Scenes | all 18 test sequences | first TrainSplit sequence of each scene, 7 × 50 frames |
| NRGBD | 9 scenes | - |
| TUM-dynamic | 8 sequences | the four `*_validation` takes, 4 × 50 frames |
| Mip-NeRF 360 | - | bicycle, garden, stump, kitchen, 4 × 50 frames |

For seed s, the 10-frame datasets use the manifest's `<dataset>_s<s>` map restricted to the
subset; these maps are a deterministic function of each sequence's frame count and the seed
(`common.sample_random`), so the data scripts regenerate them exactly. The 50-frame calibration
frames of 7-Scenes train and TUM validation come from the data scripts (`<name>_n50`, and with
`--seeded50` the seeds 1 and 2), and those of Mip-NeRF 360 from `run_eval.colmap_pick`.

The paper's Supp. Tab. 12 gives "10 test sequences" for 7-Scenes. That count is wrong: the
paper's runs used all 18 (see `eval/manifests/paper/7scenes.json`).

`run_eval.py --subsets full` evaluates every manifest sequence outside the earlier protocol's
calibration lists (`eval/manifests/splits.json`) instead; that differs from the paper only on
RealEstate10K (190 sequences, 20 of which are in the paper's RE10K calibration set).
