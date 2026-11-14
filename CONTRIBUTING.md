# Contributing

Bug reports, fixes and measurements on GPUs the paper did not use are welcome. Questions about the VGGT-Ω model
itself, its training or its PyTorch code belong in the [official repository](https://github.com/facebookresearch/vggt-omega).

## Reporting a bug

Use the bug report form on GitHub. Include the output of `rawatta info` (GPU, compute capability, driver CUDA version,
NVRTC and cuBLASLt status), the exact command, the config, and the number of frames. Do not attach weights or
datasets.

## Building and testing

The CUDA libraries are loaded at run time, so the workspace builds and its CPU tests run on any Linux machine with
Rust 1.99+ and a C compiler; no GPU or CUDA toolkit is needed for that. The checks CI runs:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
scripts/ci/python.sh          # py_compile, pytest of tools/, eval/, calibration/, research/, examples/ (numpy, pytest)
scripts/ci/shellcheck.sh      # ShellCheck 0.11.0
python3 scripts/ci/check_yaml.py
```

GPU tests are `#[ignore]`d. They come in three tiers; a test whose input is missing fails with the path and the
variable to set, so a run never reports a test as passed that did not run (the `torch_*` tests of `rawatta-nn` are
the one exception: they print a visible `SKIPPED` line when the optional PyTorch fixtures are absent).

1. **Weight-free** (required for every kernel or engine change, and in the launch pre-flight). On a CUDA machine with
   a CUDA 12.8+ toolkit:
   ```sh
   cargo test --release --locked -p rawatta-nn -- --ignored          # add --skip bench_ --skip gpu_bench to skip timings
   cargo test --release --locked -p vggt-omega ptx_tests -- --include-ignored
   ```
   `kernels_assemble_for_every_supported_gpu` needs NVRTC and `ptxas` but no GPU; it assembles every kernel for sm_80
   to sm_120.
2. **Weights only** (`RAWATTA_WEIGHTS` = the converted `vggt_omega_1b_512.safetensors`; `gpu_fp8_weights_load` also
   needs the FP8 side file from `tools/download_weights.sh --fp8`):
   ```sh
   RAWATTA_WEIGHTS=/path/to/vggt_omega_1b_512.safetensors cargo test --release --locked -p vggt-omega -- --ignored \
       gpu_weights_upload_and_views gpu_fp8_weights_load gpu_engine_load_drop_releases_pinned
   ```
   plus `rawatta run` on an example scene (`examples/run_on_photos.sh`) with each named setting.
3. **Oracle and dataset parity**: the authors' parity suite, every other ignored test of `vggt-omega`. It needs
   `RAWATTA_ORACLE` dumps of the official model (written by `src/trunk/trunk_oracle.py`, `src/heads/capture.py` and
   the official model run), `RAWATTA_DATA` (the Tanks and Temples truck images, `fixtures/omega_metrics_parity.json`)
   and `RAWATTA_BENCH_JOB`; none of these inputs ships with the repository. See
   [crates/vggt-omega/README.md](crates/vggt-omega/README.md).

The kernels are CUDA C++ files embedded with `include_str!` and compiled by NVRTC at run time. If you copy a tree
with its modification times preserved (for example with `rsync -a`), `touch` the sources before building, or cargo
may keep a binary with older kernels.

For a change to a kernel or to the engine, run at least the weight-free GPU tests (tier 1) and paste the `rawatta info`
lines and the tests you ran into the pull request. [crates/ARCHITECTURE.md](crates/ARCHITECTURE.md) maps every module to its kernels and tests.

## Numbers

Every performance or accuracy number in code, docs or a pull request says where it comes from: the paper (with its
table), or a measurement with the GPU, driver, config, frames and protocol (median of 5 to 10 timed runs after 2
warm-ups, nothing else on the GPU). See [REPRODUCE.md](REPRODUCE.md).

## Licences and what not to commit

- New files carry an SPDX line. Code copied or adapted from VGGT-Ω stays in `crates/vggt-omega/` (FAIR Noncommercial
  Research License) or carries that licence in its header; everything else is Apache-2.0.
- Contributions are licensed under the licence of the files they change: Apache-2.0, or the FAIR Noncommercial
  Research License for `crates/vggt-omega/`.
- Never commit weights (`*.pt`, `*.pth`, `*.safetensors`), datasets, run outputs, credentials or personal paths.
