# Crate architecture

Three crates in one Cargo workspace. CUDA libraries (driver, NVRTC, cuBLASLt)
are loaded at runtime through `cudarc` 0.19 (`fallback-dynamic-loading`,
API floor CUDA 12.8), and every kernel is CUDA C++ embedded with
`include_str!` and compiled by NVRTC on first use. The workspace therefore
builds and runs its CPU tests on machines without CUDA; no `nvcc` is needed.

```text
rawatta-cli  (Apache-2.0, binary `rawatta`)
   └── vggt-omega  (FAIR Noncommercial Research License)
          └── rawatta-nn  (Apache-2.0)
```

## rawatta-nn: CUDA building blocks

| Module | Does | Kernels |
|---|---|---|
| `lib.rs` | `Nn` (stream, device facts, NVRTC module cache, PTX cache per arch), `compile_ptx` / `nvrtc_arch` / `module_sources` (what `Nn` compiles, for the `ptxas` check), error re-exports (`anyhow`), `CancelFlag` | `cuda/common.cuh` (bf16 / FP8 packing: the FP8 instruction on sm_89+, a bit-identical software conversion on sm_80 to sm_88) |
| `gpu` | device probe (`probe`) and contexts (`context`) | |
| `gemm` | cuBLASLt bf16 and FP8 E4M3 GEMMs with bias / GELU epilogues | |
| `norm` | LayerNorm fused with FP8 quantisation, LayerScale residual, row epilogue (sm_80+) | `cuda/norm.cu` |
| `qkv` | QKV prologue: split, QK-norm, 2D RoPE, head-major layout | `cuda/qkv.cu` |
| `attention` | FlashAttention-2-style bf16 attention (head dim 64), fp16-P·V variant | `cuda/attention.cu` |
| `sage` | SageAttention2-style attention: INT8 QKᵀ, FP8 or FP16 P·V, FP32 / FP16 accumulation, 2×2 / 4×4 KV merging with +log(count) bias | `cuda/sage_{common,quant,attn,peak}.cu` |
| `conv` | NHWC implicit-GEMM convolution, depth-to-space, bilinear resize | `cuda/conv.cu` |
| `rowops` | in-place activation, head-dim-128 attention on short sequences | `cuda/rowops.cu` |
| `residual` | LayerScale + bias + residual add | `cuda/residual.cu` |
| `host` | bf16 / f16 / FP8 conversions and f64 CPU references (test oracles) | |
| `sage::reference` | CPU reference of the sage kernels | |
| `tests`, `conv_tests`, `sage::tests`, `testkit` | CPU tests and `#[ignore]` GPU tests against the CPU references (and optional PyTorch fixtures from `scripts/torch_ref.py`) | |

## vggt-omega: the VGGT-Ω engine

| Module | Does | Kernels |
|---|---|---|
| `lib.rs` | `VggtEngine` (load, plan, run, download), `VggtConfig`, `AttnSchedule` / `InterAttn`, FP8 block selection, `default_weights_path` | `cuda/vggt_runtime.cu` |
| `weights` | safetensors header parsing, mmap, one device arena per file set, typed views, FP8 side file | |
| `preprocess` | aspect crop, PIL-exact 8-bit bicubic resize, RGBA compositing; `Target` (output size from the first frame) and `prepare_with` (frames decoded and resized one at a time) | |
| `image` | `Rgb8Image`, JPEG via libjpeg (feature `libjpeg`, PIL-exact) or the `image` crate | |
| `workspace` | `Geometry`, memory `Plan` (dense chunks, level offload, trunk release), device buffers | |
| `graph` | CUDA-graph capture of the trunk per (geometry, config) | |
| `hostbuf` | pinned host buffers for offloaded cache levels | |
| `trunk` | DINOv3 patch embedder (`dino`), token assembly (`aggregator`), blocks (`block`), RoPE tables (`rope`), approximate global attention (`sched`: sage, merge2/4, frame) | `cuda/vggt_trunk.cu` |
| `heads` | camera head (`camera`), DPT dense head (`dense`), position embedding (`pos_embed`) | `cuda/vggt_heads.cu` |
| `output` | frame sampling, point cloud, PLY, COLMAP text, `cameras.json`, `.npy` | |
| `bench::configs` | config grammar `<schedule>[@<precision>][~<overrides>][+fp8[:<blocks>]]` and named aliases (`exact`, `rawatta`, `rawatta-fast`, controls) | |
| `bench::metrics` | pose AUC / RRA / RTA, depth δ<1.25 / AbsRel, Umeyama, point metrics | |
| `bench::cell` | benchmark job runner (`Job`, `run_job`): predictions + timing per (config, sequence) | |
| `tests`, `schedule_tests`, `trunk::tests`, `heads::tests`, `testkit` | CPU tests and `#[ignore]` GPU parity tests (`RAWATTA_WEIGHTS`, `RAWATTA_ORACLE`, `RAWATTA_DATA`) | |
| `ptx_tests` | `#[ignore]`, no GPU: every NVRTC module the engine loads (`module_sources`), compiled for each GPU family from sm_80 to sm_120 and assembled with `ptxas`; the sage kernels for sm_89+ | |

## rawatta-cli

The `rawatta` binary (clap derive).

| Module | Does |
|---|---|
| `main.rs` | commands `info`, `run`, `bench`; `bench` checks the configs, weights and GPU, then calls `vggt_omega::bench::run_job` (job JSON, output layout and `cell.json` unchanged) |
| `support.rs` | config names, grammar strings and `.toml` config files (`rawatta`, `rawatta-fast`, `exact`, any alias or grammar string, a file in the `configs/` format), compute-capability gate (sm_89+ for quantised attention or FP8 GEMMs, sm_80+ otherwise), weights location and the missing-weights message (names `RAWATTA_WEIGHTS` and the Docker and native download commands), `--crop-aspect` parsing |
| `info.rs` | GPU name, compute capability, VRAM, driver CUDA version (a driver that lists no device, `CUDA_ERROR_NO_DEVICE`, is told apart from a missing driver), RAWATTA / exact-engine support, NVRTC and cuBLASLt checks, weights location |
| `run.rs` | input path checked first (before the weights and the GPU); images folder (sorted, sampled) or video; output size from the first frame, memory-plan check, then every frame decoded and resized one at a time; warm-up and timed passes, outputs: `cameras.json`, `extrinsics.npy`, `intrinsics.npy`, `depth.npy`, `conf.npy`, `points.ply` (xyz float + rgb uchar), `colmap/` (`cameras.txt`, `images.txt`, `points3D.txt`, `images/` with the preprocessed frames), `timing.json` |
| `video.rs` | uniform frame sampling with `ffprobe` / `ffmpeg` subprocesses (`-fps_mode vfr`, falling back to `-vsync vfr` before FFmpeg 5.1) |
| `config_files.rs` | test: `configs/*.toml` and `configs/GRAMMAR.md` agree with the aliases |

## GPU coverage

Every GPU test runs on sm_89 (RTX 4070 SUPER). For other families the
`ptx_tests` check assembles every module with `ptxas` for sm_80, sm_86,
sm_87, sm_89, sm_90, sm_100 and sm_120 (sage: sm_89 and newer), and
`gpu_soft_fp8_matches_cvt` shows that the software FP8 conversion used
below sm_89 equals the sm_89 instruction on all 2^32 inputs;
`gpu_norm_soft_fp8_path_matches_hardware` runs the LayerNorm and row
epilogue kernels built on that path and gets the same FP8 bytes. So the exact
engine is built and bit-checked for sm_80+, not yet run on Ampere
hardware (sm_80 to sm_88): cuBLASLt's bf16 algorithm choice, the shared-memory
limits of the tile choices on sm_86 and the memory planner are untested there.

## Provenance

Extracted from the authors' research codebase; the kernel sources are
unchanged apart from renames (the table below lists every addition), and
host-specific defaults were replaced by environment variables. End-to-end
parity with the paper's runs is pending until the weights are available.

| Here | Came from |
|---|---|
| `rawatta-nn/src/{attention,conv,gemm,host,norm,qkv,residual,rowops,testkit,tests,conv_tests}.rs`, `src/sage/**`, `cuda/*`, `scripts/*` | the platform `nn` crate (unchanged apart from renames, paths and comments, and, added here: the software FP8 conversion in `cuda/common.cuh` that lets `norm` load on sm_80 to sm_88, `sage::defines`, the fixture skip in `testkit`, and the GPU tests `gpu_soft_fp8_matches_cvt` and `gpu_norm_soft_fp8_path_matches_hardware`) |
| `rawatta-nn/src/lib.rs` | the platform `nn` crate's `lib.rs`, plus the error re-exports of the platform `core` crate (`Result`, `bail`, `anyhow`, `Context`) and a minimal `CancelFlag` replacing its `CancellationToken`; the op registry, run context, artifacts and events were dropped |
| `rawatta-nn/src/gpu.rs` | the platform `gpu` crate (`probe`, `context`), without the JSON-schema / TypeScript derives and the unused `load_cuda_source` |
| `vggt-omega/src/{lib,graph,hostbuf,preprocess,weights,workspace,tests}.rs`, `src/trunk/**`, `src/heads/**`, `cuda/vggt_*.cu` | the reconstruction app's `vggt` module and its `cuda/vggt_*.cu` kernels |
| `vggt-omega/src/schedule_tests.rs` | the CPU tests of the app's attention-schedule tests; the GPU experiment harnesses (evaluation dumps, deviation and FP8-sensitivity sweeps) were dropped |
| `vggt-omega/src/bench/{configs,metrics}.rs` | the app's `bench` module; the experiment aliases were replaced by the published setting names |
| `vggt-omega/src/bench/cell.rs` | the app's `gpu_bench_cell` test, turned into a library function (`run_job`) with the same job format and output layout |
| `vggt-omega/src/output.rs` | the output helpers of the app's pose op (`sample_indices`, `point_cloud`, `write_points_ply`, `mat_to_quat`, `colmap_text`, `cameras.json`) and the bench cell's `.npy` writer |
| `vggt-omega/src/image.rs` | the app's image decoding (libjpeg path and `image` crate fallback) |
| `vggt-omega/src/testkit/npz.rs` | the app's test kit `.npz` reader |

Not carried over: the app's pipeline integration tests (pose op vs COLMAP,
scaling benchmark, evaluation dumps), which depend on the app's op runtime;
the app's training, export, ONNX and UI code; the attention-schedule
schedule names of the experiment log (the `v1` schedule remains as a
control).

Renames: the platform crates' paths → `anyhow` + `rawatta_nn`;
`CancellationToken` → `rawatta_nn::CancelFlag`; the early control schedule
and its wide variant → `v1` / `v1-wide` (`AttnSchedule::v1`, `v1_wide`); its
FP8 block set → `LATE_FP8_BLOCKS`. Environment variables: `RAWATTA_WEIGHTS`,
`RAWATTA_ORACLE`, `RAWATTA_DATA`, `RAWATTA_BENCH_JOB`,
`RAWATTA_METRICS_FIXTURE`, `RAWATTA_NN_FIXTURES`, `RAWATTA_OUT`.
