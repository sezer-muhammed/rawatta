<!-- SPDX-License-Identifier: Apache-2.0 -->
# Frame manifests

The JSON files in this folder and in `paper/` are under Apache-2.0 (JSON has no comments, so the licence is
recorded here). They hold frame indices, file names and SHA-256 sums of the prepared datasets, not
dataset content; the datasets keep their own licences ([../DATASETS.md](../DATASETS.md)).

| file | content |
|---|---|
| `<dataset>.json` | per scene the frame count and the SHA-256 of every prepared file; the frame maps (`<dataset>_s0/_s1/_s2`: 10 frames per seed, `_n50`: 50 evenly spaced frames; ETH3D: the official map) with the SHA-256 of each map file; the source notes of the data script |
| `paper/<dataset>.json` | the paper's evaluation subset of one of the six evaluation datasets: its sequences and their frames for seeds 0, 1 and 2, with the rule and the evidence it was recovered from (written by `eval/paper_subsets.py`) |
| `paper/calibration.json` | the paper's five calibration datasets: sequences, frames per seed where the manifests determine them, and the rule for the others |
| `splits.json` | the earlier protocol's calibration lists, dropped by `run_eval.py --subsets full` |
