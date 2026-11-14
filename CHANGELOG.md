# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First public release, with the paper *Training-Free Acceleration of Multi-View 3D Transformers via Regime-Aware
Attention Approximation*. Extracted from the authors' research codebase; the kernel sources are unchanged apart from
renames (crates/ARCHITECTURE.md). End-to-end parity with the paper's runs is pending until the weights are available.

### Added

- `rawatta` command line (`info`, `run`, `bench`) on a native Rust + CUDA engine for VGGT-Ω; kernels compiled at run
  time with NVRTC; static memory planner.
- Named settings `rawatta` (default), `rawatta-fast` and `exact`, the config grammar for per-layer schedules and
  precisions, and the paper's other configurations as named aliases (`configs/GRAMMAR.md`).
- Outputs: `cameras.json`, camera and depth arrays, `points.ply`, COLMAP text model, `timing.json`.
- Torch-free weight converter and download script (`tools/`).
- Evaluation under the official VGGT-Ω protocol on the paper's six datasets, dataset scripts and frame manifests, and a
  container for the Official VGGT-Ω baseline (`eval/`).
- Calibration check and precision / schedule searches (`calibration/`).
- PyTorch harnesses used for the paper's VGGT-1B and π³ experiments, unsupported (`research/`).
- Docker image (`docker/`), CI and release workflows, example scripts and scenes (`examples/`), project page (`docs/`).

### Known limitations

- Linux x86_64 and NVIDIA GPUs only. RAWATTA and RAWATTA-Fast need sm_89+; the exact engine is built and bit-checked
  for sm_80+ but not yet run on Ampere hardware.
- The paper's RealEstate10K and 7-Scenes evaluation subsets were not kept, and the ports of the rival methods onto
  VGGT-Ω are not included (`REPRODUCE.md`).

[0.1.0]: https://github.com/sezer-muhammed/rawatta/releases/tag/v0.1.0
