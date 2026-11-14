# Reproducing the paper

The commands behind each table of *Training-Free Acceleration of Multi-View 3D Transformers via Regime-Aware
Attention Approximation*, the values the paper reports, and what this repository cannot reproduce.

Every expected value below is from the paper (arXiv XXXX.XXXXX, version 1) and names its table ("Tab." = main paper,
"Supp. Tab." = supplementary material). The paper's runs used the authors' research codebase, from which this release
was extracted; the kernel sources are unchanged apart from renames ([crates/ARCHITECTURE.md](crates/ARCHITECTURE.md)).
End-to-end parity with the paper's runs is pending until the weights are available.

> **Status.** The full runs with the VGGT-Ω weights have not yet been done for this release, so no table below has
> been re-measured with it. What has been checked without the weights: the config strings in this file parse in the
> release binary, and the evaluation and calibration pipelines, scored with the official evaluator, pass their
> end-to-end tests with a stand-in binary (`python3 -m pytest eval/tests calibration/tests`). This file will list
> measured values once they exist.

## 1. Hardware

| | Paper |
|---|---|
| Main GPU | one NVIDIA GeForce RTX 4070 SUPER, 12 GB, CUDA 12.8, Linux, no other process on the GPU |
| Second GPU (Supp. Tab. 18) | RTX PRO 6000 Blackwell Server Edition, 96 GB, CUDA 13.0, card exclusive, one fresh process per cell |

Accuracy runs need a GPU that can run the configs (sm_89+ for anything quantised; sm_80+ for `exact`, which is built
and bit-checked for sm_80+ but not yet run on Ampere hardware) and has room for 10-frame sequences. Absolute
latencies are only comparable on the same GPU model; compare the speed-up ratios otherwise.

## 2. Software and weights

- The `rawatta` binary, from the Docker image or a native build ([README](README.md#quickstart)). `eval/run_eval.py`
  and `calibration/calibrate.py` call it as `rawatta` on `PATH` (or `--rawatta PATH` / `RAWATTA_BIN`).
- Python 3.10+ with the evaluation dependencies (numpy, opencv-python-headless, Pillow, torch; CPU torch is enough):
  ```sh
  pip install -r eval/requirements.txt
  python3 eval/omega_src.py --check      # fetches the official VGGT-Ω evaluator at the pinned commit into a cache
  ```
  The official evaluator is under the FAIR Noncommercial Research License and is fetched at run time, not vendored.
- The converted weights, downloaded as in the README. The Official VGGT-Ω baseline (Tab. 1, Tab. 4) runs the original
  PyTorch checkpoint `vggt_omega_1b_512.pt`, which the download deletes after converting: append `--keep-pt` to the
  download command to keep it. `eval/official/run_official.sh` reads it from `~/.cache/rawatta/` (or
  `VGGT_OMEGA_CKPT`).
- Docker, for the Official baseline: `eval/official/Dockerfile` builds PyTorch 2.9.1 + CUDA 12.8 with the official
  repository at the pinned commit.

## 3. Datasets

[eval/DATASETS.md](eval/DATASETS.md) lists every dataset, its licence and its download script; only the sampled frames
are downloaded. Six evaluation datasets (ETH3D, CO3Dv2, RealEstate10K, 7-Scenes, NRGBD, TUM-dynamic) and five
calibration datasets (Mip-NeRF 360, CO3Dv2 banana/donut, RealEstate10K, 7-Scenes train, TUM-dynamic validation).

```sh
export DATA=/path/to/data
# the commands of eval/DATASETS.md, "Preparing", then check the files against the paper's manifests:
python3 eval/data/manifest.py verify $DATA/eth3d eval/manifests/eth3d.json
python3 eval/data/manifest.py verify $DATA/co3d  eval/manifests/co3d.json --maps $DATA/_frames
```

The frame maps per seed and the SHA-256 of every prepared file are in [eval/manifests/](eval/manifests). ETH3D, CO3Dv2
(40 sequences of 8 categories), NRGBD (9 scenes) and TUM-dynamic (8 takes) match the paper's evaluation sets.
RealEstate10K and 7-Scenes do not; see [Known gaps](#7-known-gaps).

## 4. Timing protocol and the GPU lock

The paper times the full forward pass (trunk, camera and dense heads) with CUDA events on pre-loaded images: median of
5 to 10 runs after 2 warm-ups, Mip-NeRF 360 kitchen at 512×336, under an exclusive `flock` on the GPU. In Tab. 1
the exact engine and RAWATTA use the mean of three locked sessions, each a fresh process; RAWATTA-Fast uses the
1616.4 ms session in which the rivals were timed (three-session mean 1611.8 ms; Supp. Tab. 23).

- Pass `--reps 5 --warmup 2` (or more reps) and keep the GPU otherwise idle: no other jobs, no training, no display
  load if you can avoid it.
- On a shared machine, give every GPU job the same lock file and take it around the whole timing command:
  `flock -x /path/to/gpu.lock <command>`. `calibration/calibrate.py` and `eval/official/run_official.sh` take
  `--lock FILE` themselves; wrap `eval/run_eval.py` and `rawatta run` in `flock`. Treat timings taken without the lock
  as preliminary.
- `timing.json` and `cell.json` report peak memory from `cuMemGetInfo`, which counts the whole card, so other
  processes inflate it.

## 5. Tables of the main paper

The examples write into `runs/`; `run_eval.py` prints the table and writes `results.json` and `table.md` there. It
deletes the predicted depth maps after scoring unless `--keep`.

### Tab. 1: operating points (accuracy)

```sh
python3 eval/run_eval.py --data $DATA --datasets paper --seeds 0,1,2 \
    --configs exact,rawatta,rawatta-fast,threshold,greedy-b2 --out runs/tab1
```

The sensitivity-only greedy row (post hoc) has no alias; Supp. Tab. 6 describes it as merge4 on blocks 0, 1, 3, 4, 8,
19, 22, frame on 5, at RAWATTA's precision, which in the grammar reads
`L0:merge4/L1:merge4/L3:merge4/L4:merge4/L5:frame/L8:merge4/L19:merge4/L22:merge4@sagepp~L4=sage.pvh16/L15=sage.pvh16`.

Expected (Tab. 1; AUC@3 of the exact engine and ΔAUC@3 against it, N = 10, mean over three seeds; worst = per-seed
minimum over the six datasets, mean ± std over seeds):

| | ETH3D | CO3D | RE10K | 7-Sc. | NRGBD | TUM-dyn | worst Δ3 | worst Δ30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact engine (AUC@3) | 49.91 | 51.21 | 38.49 | 21.11 | 85.46 | 39.66 | 0 | 0 |
| RAWATTA | −0.23 | −0.15 | −0.19 | +0.01 | −0.08 | −0.22 | −0.51 ± 0.11 | −0.24 |
| RAWATTA-Fast | −0.34 | +0.07 | −0.22 | −0.56 | −0.49 | −0.12 | −0.80 ± 0.17 | −1.20 |
| threshold schedule | +0.17 | −0.42 | +0.24 | −0.22 | −0.36 | −0.84 | −0.99 ± 0.54 | −1.23 |
| greedy search, budget −2 | −1.03 | −0.10 | −0.45 | −0.73 | −1.48 | −0.16 | −1.59 ± 0.50 | −0.75 |
| sensitivity-only greedy | +0.40 | −0.27 | −0.61 | −0.51 | −0.03 | −0.40 | −0.91 ± 0.25 | −0.21 |

Tolerance: the paper treats differences below about 0.5 AUC@3 as seed noise, and the ± above is the sample standard
deviation over the three seeds. The engine is deterministic: re-runs on the same machine are bit-identical
(Supp. reproducibility checklist). Another GPU, driver or cuBLASLt version can change results slightly. On the
RE10K and 7-Scenes columns, and on any worst-dataset value they decide, expect differences from the paper because the
manifests hold more sequences than the paper evaluated ([Known gaps](#7-known-gaps)).

The "vs Off." column and the "cal." column of Tab. 1 come from Tab. 4 (below) and from the calibration check
(Sec. 4.4, [section 6](#6-supplementary-material-calibration-and-the-rest)).

### Tab. 1 and Tab. 2: latency at 50 and 200 frames

Mip-NeRF 360 kitchen, prepared with `eval/data/mipnerf360.py` (quarter-resolution images, run at 512×336). One
session, under the GPU lock:

```sh
flock -x /path/to/gpu.lock python3 eval/run_eval.py --data $DATA --datasets mipnerf360 --scenes kitchen \
    --frames 50 --configs exact,rawatta,rawatta-fast --reps 5 --warmup 2 --out runs/time-n50-1
```

Run it three times as separate processes (`runs/time-n50-1`, `-2`, `-3`) and average the session medians; use
`--frames 200` for N = 200. `rawatta run --images $DATA/mipnerf360/kitchen/images_4 --max-frames 50 --reps 5
--warmup 2 --no-depth --config <config> --out <dir>` measures the same forward pass for one config and writes it to
`timing.json`.

Official VGGT-Ω on the same scene (it runs out of memory at N ≥ 100 on 12 GB):

```sh
mkdir -p runs/official && echo '{"kitchen": 50}' > runs/official/kitchen-n50.json
RAWATTA_DATA=$DATA OUT=runs/official eval/official/run_official.sh --lock /path/to/gpu.lock -- \
    --data-root /data/mipnerf360 --frames /out/kitchen-n50.json --no-depth --out /out/official-kitchen-n50.json
```

Expected (Tab. 1, Tab. 2):

| | N = 50 (ms) | × Official | N = 200 (s) | × exact | step × (Tab. 2) |
|---|---:|---:|---:|---:|---|
| Official VGGT-Ω | 3605 | 1.00 | OOM | – | – |
| Exact engine | 2237 | 1.61 | 20.74 | 1.00 | engine 1.61 |
| RAWATTA | 1734 | 2.08 | 12.51 | 1.66 | precision 1.29 (N = 50), 1.66 (N = 200) |
| RAWATTA-Fast | 1616 | 2.23 | 10.56 | 1.97 | schedule 1.07 (N = 50), 1.19 (N = 200) |

Of the 2.08× at 50 frames, 1.61× is the CUDA engine and 1.29× RAWATTA itself; at 200 frames RAWATTA alone gives 1.66×
(RAWATTA-Fast 1.97×) over the exact engine. The lower rows of Tab. 1: threshold schedule 1544 ms / 9.39 s, greedy
search budget −2 1547 ms / 9.51 s, sensitivity-only greedy 1558 ms / 9.67 s.

Tolerance: on the paper's card, locked sessions of the same config agree within 10 ms (Supp. Tab. 23: exact engine
2236.6 ± 1.5, RAWATTA 1734.2 ± 1.3, RAWATTA-Fast 1611.8 ± 5.1 ms over three sessions; Tab. 1 uses the 1616.4 ms
session for RAWATTA-Fast). Another card of the same model can differ by more (clocks, cooling, driver).

The per-component kernel times of Tab. 2 came from one Nsight Systems session; this release has no script for that
breakdown.

### Tab. 3: numeric format (seed 0)

```sh
python3 eval/run_eval.py --data $DATA --datasets paper --seeds 0 \
    --configs exact,sage2,sage2pp,pvh16,rawatta --out runs/tab3
```

| P·V format (accumulation) | config | worst Δ3 | worst Δ30 | mean Δ3 | ms (N = 50) |
|---|---|---:|---:|---:|---:|
| bf16 (exact engine) | `exact` | 0 | 0 | 0 | 2237 |
| FP8 (FP32), SageAttention2 | `sage2` | −2.14 | −0.85 | −0.91 | 1799 |
| FP8 (FP16), SageAttention2++ | `sage2pp` | −0.56 | −1.12 | −0.38 | 1713 |
| 16-bit (FP16) | `pvh16` | −0.21 | −0.06 | −0.01 | 1943 |
| RAWATTA: FP8 (FP16), 16-bit on 4, 15 | `rawatta` | −0.41 | −0.12 | −0.15 | 1738 |

Supp. Tab. 10 gives the per-dataset seed-0 AUC@3 / AUC@30 of these configs. The "FP8 (FP32), row sum of rounded P"
row (`dense@sage.pq`) is not available in this release: the binary rejects that config.

### Tab. 4: published accelerators on the same GPU

In-engine rows, three seeds (the SageAttention2 row is seed 0 in the paper):

```sh
python3 eval/run_eval.py --data $DATA --datasets paper --seeds 0,1,2 \
    --configs exact,rawatta,rawatta-fast,sage2,avggt-pub,avggt-pub-frame,gpa-fast --out runs/tab4
```

and the latency commands above with the same configs (`--frames 50` and `--frames 200`).

The Official VGGT-Ω row and the "vs Official" deltas need Official's accuracy on the frames each run used.
`run_eval.py` writes them to `<run>/<dataset>-s<seed>/fmap.json`; run the official model on each, for example:

```sh
RAWATTA_DATA=$DATA OUT=runs/tab4 eval/official/run_official.sh --lock /path/to/gpu.lock -- \
    --data-root /data/eth3d --frames /out/eth3d-s0/fmap.json --no-timing --out /out/official-eth3d-s0.json
```

and subtract its AUC@3 / AUC@30 from the configs' values in `runs/tab4/results.json` (no script merges the two).

Expected (Tab. 4; worst ΔAUC against Official over the six datasets, mean ± std over seeds, * = seed 0):

| | config | s (N = 50) | × | worst Δ3 | worst Δ30 | s (N = 200) |
|---|---|---:|---:|---:|---:|---:|
| Official VGGT-Ω | (eval/official) | 3.61 | 1.00 | 0 | 0 | OOM |
| SageAttention2, uniform | `sage2` | 1.80 | 2.00 | −2.47* | −1.10 | 13.57 |
| frame 0–8, merge2 10–23 | `avggt-pub` | 1.36 | 2.65 | −37.89 ± 0.00 | −28.41 | 6.63 |
| frame 0–8 only | `avggt-pub-frame` | 1.56 | 2.31 | −2.97 ± 0.31 | −0.82 | 9.72 |
| GPA-style merge, RAWATTA-Fast's layers | `gpa-fast` | 1.65 | 2.18 | −1.03 ± 0.00 | −1.25 | 11.13 |
| Exact engine | `exact` | 2.24 | 1.61 | −0.86 ± 0.11 | −0.30 | 20.74 |
| RAWATTA | `rawatta` | 1.73 | 2.08 | −1.03 ± 0.00 | −0.39 | 12.51 |
| RAWATTA-Fast | `rawatta-fast` | 1.62 | 2.23 | −1.14 ± 0.00 | −1.13 | 10.56 |

The rows for SparseVGGT†, GPA-VGGT† and FastVGGT† (released code ported onto VGGT-Ω) are not reproducible here: the
ports are not part of this release.

### Tab. 5: ablations of RAWATTA-Fast

At RAWATTA's precision, against RAWATTA (`--reference rawatta`), three seeds:

```sh
python3 eval/run_eval.py --data $DATA --datasets paper --seeds 0,1,2 --reference rawatta --out runs/tab5 --configs \
"rawatta,rawatta-fast,\
L0:frame/L3:merge2/L4:frame/L5:frame/L19:merge2@sagepp~L15=sage.pvh16,\
L0:frame/L3:merge4/L4:frame/L5:frame/L19:merge4@sagepp~L15=sage.pvh16/L3=sagepp.nb/L19=sagepp.nb,\
L0:frame/L3:merge4/L4:frame/L5:frame/L19:merge4@sagepp~L15=sage.pvh16/L19=sagepp.f0,\
L0:frame/L12:merge4/L4:frame/L5:frame/L13:merge4@sagepp~L15=sage.pvh16"
```

| Row | config | worst Δ3 vs RAWATTA | mean Δ3 | ms (N = 50) |
|---|---|---:|---:|---:|
| RAWATTA-Fast | `rawatta-fast` | −0.70 ± 0.19 | −0.14 | 1608 |
| (a) merge 2×2 instead of 4×4 | `...L3:merge2...L19:merge2...` | −0.80 ± 0.09 | −0.12 | 1620 |
| (b) no +log c bias | `...~L15=sage.pvh16/L3=sagepp.nb/L19=sagepp.nb` | −1.14 ± 0.17 | −0.05 | 1604 |
| (c) frame 0 merged in block 19 | `...~L15=sage.pvh16/L19=sagepp.f0` | −2.68 ± 0.00 | −0.73 | 1606 |
| (e′) only the two merges moved to 12, 13 | `L0:frame/L12:merge4/L4:frame/L5:frame/L13:merge4@...` | −19.60 ± 0.00 | −8.88 | 1613 |

The config strings of (a) to (e′) are our transcription of the row descriptions into the grammar
([configs/GRAMMAR.md](configs/GRAMMAR.md): `.nb` drops the bias on a merged layer, `.f0` pools frame 0 too). Row (e)
does not state its layers, so it is not listed. The draws of (d₁) and (d₂) are listed layer by layer in Supp.
Tab. 25 and can be written the same way; the layers of row (d) are not given in the paper.

## 6. Supplementary material, calibration and the rest

| Result | How | Expected (paper) | Reproducible here? |
|---|---|---|---|
| Configuration names (Supp. Tab. 6) | [configs/GRAMMAR.md](configs/GRAMMAR.md), "Other named configs" | – | yes: every name maps to a config string, except the configs listed under "Not available" there |
| Seed-0 results per dataset (Supp. Tab. 10) | the Tab. 3 command | Supp. Tab. 10 | yes, with the RE10K / 7-Scenes caveat |
| Depth and point maps at N = 10 (Supp. Tab. 22) | `run_eval.py` prints δ1.25 and AbsRel for ETH3D, 7-Scenes, NRGBD, TUM-dynamic; add `--points` for the point-map metrics | δ1.25, exact engine / RAWATTA: ETH3D 99.83 / 99.83 (AbsRel 0.0115 / 0.0116), 7-Scenes 94.61 / 94.60, NRGBD 99.67 / 99.67 | yes, with the 7-Scenes caveat |
| Small N, 2 to 10 frames (Supp. Tab. 24) | the latency command with `--frames 2`, `4`, `10` | N = 2 / 4 / 10: Official 90.5 / 167.5 / 482.2 ms, exact engine 52.0 / 108.5 / 285.4 ms, RAWATTA 51.4 / 105.4 / 265.5 ms, RAWATTA-Fast 51.3 / 104.7 / 260.7 ms | yes |
| Calibration check and searches (Sec. 4.4, Supp. Tab. 13) | [calibration/README.md](calibration/README.md): `calibrate.py precision`, `schedule`, `check --seeds 0,1,2` | precision search stops after blocks 15 and 4 (RAWATTA); schedule search: 49 configurations, frame on 0, 4, 5 and merge4 on 3, 19 (RAWATTA-Fast); check on three samples: RAWATTA 3/3, RAWATTA-Fast 2/3, threshold schedule 0/3 | partly: the paper's 30-sequence RE10K calibration subset is lost, so a search is not guaranteed to return the paper's settings |
| Accuracy at 50, 100 and 200 frames (Supp. Tab. 11) | – | RAWATTA within 0.32 and RAWATTA-Fast within 1.20 AUC@3 of the exact engine on the worst dataset | no: the seeded 100- and 200-frame maps of the five datasets are not in the manifests (`--map n50` gives one evenly spaced 50-frame map for 7-Scenes, NRGBD and TUM-dynamic only) |
| ETH3D at 50 frames (Supp. Tab. 21) | – | RAWATTA-Fast −1.65 against the exact engine, −2.22 against Official | no: the manifests hold ETH3D's official 10-frame map only |
| Memory on 7-Scenes-long (Supp. Tab. 17) | `eval/data/sevenscenes_long.py` prepares the sequences and the N-frame maps; run them with a `rawatta bench` job | exact engine at N = 200: 10.63 GiB peak, 32.8 AUC@3 | partly: no driver script; the chunked VGGT-Ω and rival rows are not included |
| Second GPU class (Supp. Tab. 18) | the latency commands on an RTX PRO 6000 Blackwell | RAWATTA 1.86× and 1.76× faster than Official at 50 and 200 frames, exact engine 1.41× and 1.05× | needs that GPU |
| Generality on VGGT-1B and π³ (Sec. 5.6, Supp. Tab. 19) | [research/README.md](research/README.md) | Supp. Tab. 19 | unsupported: the harnesses are kept as used for the paper and were not re-run for this release |
| Regime statistics (Fig. 2 top, Supp. Fig. 3), PyTorch emulation (Supp. Fig. 5, Supp. Secs. E and K), NVS (Supp. Tab. 20), top-k routing control (Supp. Sec. R) | – | – | no: these harnesses are not part of this release ([research/README.md](research/README.md), "Not included", covers the first two and the routing control) |

The figures of the project page are regenerated from the paper's logged values with `site-src/make_figs.py`, not from
new runs ([site-src/README.md](site-src/README.md)).

## 7. Known gaps

- **Evaluation subsets that were not kept.** The paper evaluated 38 RealEstate10K sequences and 10 7-Scenes test
  sequences; neither list survived. `run_eval.py` evaluates all 18 7-Scenes test sequences and the 190 RealEstate10K
  sequences outside the earlier protocol's 10-sequence calibration list; they may include sequences of the paper's
  lost 30-sequence calibration subset, so RE10K results for the shipped settings are not guaranteed sequence-disjoint
  from their calibration. These two columns, and any worst-dataset value they decide, are not expected to equal the
  paper's. The paper's 30-sequence RealEstate10K calibration subset and its CO3Dv2 calibration
  map are lost too ([eval/DATASETS.md](eval/DATASETS.md)).
- **Rival ports not included.** The ports of SparseVGGT†, GPA-VGGT† and FastVGGT† onto VGGT-Ω, chunked VGGT-Ω
  (Tab. 4, Supp. Tab. 17) and the top-k routing control are not part of this release. Their in-engine counterparts
  (`avggt-pub`, `avggt-pub-frame`, `gpa-fast`, `sage2`) are.
- **Unavailable configs.** The engine rejects `dense@sage.pq` (Tab. 3), exact GPA-VGGT, block-sparse attention,
  FastVGGT token merging and the SageAttention v1 precision ([configs/GRAMMAR.md](configs/GRAMMAR.md), "Not
  available").
- **`research/` is unsupported.** The VGGT-1B and π³ harnesses are kept as used for the paper, are not tested beyond
  `--help`, and were not re-run for this release; the regime-statistics and PyTorch-emulation harnesses were lost.
- **No measurements with this release yet.** See the status note at the top.

## 8. Reporting your numbers

Say where every number comes from: the paper (with its table) or your measurement, with the GPU, driver, config,
frames, number of timed runs and warm-ups, and whether the GPU was locked. `rawatta info` prints most of it.
