# vggt-omega

A native CUDA inference engine for **VGGT-Ω**, Meta's multi-view 3D
transformer: it maps converted weights, preprocesses frames like the official
PyTorch loader for same-size inputs (PIL-exact bicubic resize, libjpeg
decoding; frames of another aspect ratio are centre-cropped to the first
frame's instead of padded), runs the DINOv3 patch embedder, the 24
alternating frame / global blocks, the camera head and the DPT dense head,
and plans device memory so that 200 frames fit on a 12 GB card (paper
Sec. 5.5; planner limit about 259 frames at 512×336, Supp. Sec. M). Global attention layers follow a per-layer *attention schedule*
(dense bf16, SageAttention2-style INT8/FP8, merged keys and values, or
per-frame attention) written in a small config grammar
(`src/bench/configs.rs`); the named settings `exact`, `rawatta` and
`rawatta-fast` are aliases in that grammar.

The CUDA building blocks (GEMMs, LayerNorm, attention kernels) live in the
Apache-2.0 crate [`rawatta-nn`](../rawatta-nn).

## Licence

This crate is a port of Meta's VGGT-Ω architecture and is distributed under
the **FAIR Noncommercial Research License** (`LICENSE-VGGT-OMEGA` at the
repository root): research use only, no commercial use. The VGGT-Ω weights
are gated on Hugging Face under the same licence; they are never bundled with
this code. The DINOv3 backbone code it ports (`src/trunk/{dino,block,rope}.rs`)
is also under the DINOv3 License (`LICENSE-DINOV3`), as in upstream VGGT-Ω. Download and convert them once (the Docker image's
`download-weights` entrypoint or `tools/download_weights.sh`, which runs
`tools/convert_weights.py`; see the repository README) and point the engine
at the result (`RAWATTA_WEIGHTS`, or the default location, see
`default_weights_path`).

## Tests

- `cargo test -p vggt-omega` runs the CPU tests (preprocessing, weight-file
  parsing, memory planning, config grammar, metrics, outputs).
- GPU and data-dependent tests are `#[ignore]`, in three tiers (see
  CONTRIBUTING.md): weight-free (`cargo test --release --locked -p vggt-omega
  ptx_tests -- --include-ignored`, plus all of `rawatta-nn`), weights only
  (`gpu_weights_upload_and_views`, `gpu_fp8_weights_load`,
  `gpu_engine_load_drop_releases_pinned`, `gpu_bench_engine`), and the
  authors' parity suite, which needs inputs that do not ship with the
  repository. A test whose input is missing fails with the path and the
  variable to set. The inputs:
  - `RAWATTA_WEIGHTS`: the converted `vggt_omega_1b_512.safetensors`;
  - `RAWATTA_ORACLE`: parity dumps of the official model (default
    `/data/oracle`; `<set>/`, `trunk/`, `heads/` subdirectories, written by
    `src/trunk/trunk_oracle.py`, `src/heads/capture.py` and the official
    model run);
  - `RAWATTA_DATA`: datasets (default `/data`);
  - outputs (`gpu_pose_outputs_on_truck`) go to the workspace `target/`
    (or `$CARGO_TARGET_DIR`);
  - `RAWATTA_BENCH_JOB`: a benchmark job file for `bench::cell`.
