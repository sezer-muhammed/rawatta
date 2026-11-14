## What this changes

<!-- What and why, in a few sentences. Link the issue it fixes, if any. -->

## How it was checked

- [ ] `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` pass (CI runs them without a GPU)
- [ ] Python or shell changes: `scripts/ci/python.sh` and `scripts/ci/shellcheck.sh` pass
- [ ] Kernel or engine changes: the weight-free GPU tests pass (`cargo test --release --locked -p rawatta-nn -- --ignored` and `cargo test --release --locked -p vggt-omega ptx_tests -- --include-ignored`; tiers in CONTRIBUTING.md), and the weights-only tests if you have the weights; GPU model and `rawatta info` output:

<!-- Paste the GPU tests you ran and the `rawatta info` lines here. -->

## Numbers

- [ ] No new performance or accuracy numbers, **or** every new number says where it comes from: the paper (table) or a measurement (GPU, driver, config, frames, protocol: median of 5 to 10 runs after 2 warm-ups on an idle GPU)

## Licences and data

- [ ] No weights (`*.pt`, `*.safetensors`), datasets, credentials or personal paths are added
- [ ] Code copied or adapted from VGGT-Omega stays under `crates/vggt-omega/` (FAIR Noncommercial Research License) or carries that licence in its header
