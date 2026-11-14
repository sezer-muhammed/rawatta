# Docker image

`ghcr.io/sezer-muhammed/rawatta` packages the `rawatta` binary with everything it needs at run time.
The host needs only Docker and an NVIDIA driver. The model weights are **not** in the image: one
command downloads and converts them into a folder on the host, which you then mount.

The image is published at v0.1.0; pulling it before then fails with "error from registry: denied".
Until then build it from a checkout (`git clone https://github.com/sezer-muhammed/rawatta && cd rawatta`)
with `docker build -f docker/Dockerfile -t rawatta .` (needs BuildKit, see [Build details](#build-details))
and use `rawatta` in place of `ghcr.io/sezer-muhammed/rawatta:latest` in the commands below, or build
with `-t ghcr.io/sezer-muhammed/rawatta:latest` to use them unchanged.

## Quick start

```sh
mkdir -p ~/.cache/huggingface ~/.cache/rawatta
# weights, once (host needs only Docker + NVIDIA driver; token file from 'hf auth login' or saved by hand)
docker run --rm --user "$(id -u):$(id -g)" -v ~/.cache/huggingface:/hf:ro -v ~/.cache/rawatta:/weights \
  --entrypoint download-weights ghcr.io/sezer-muhammed/rawatta:latest --token-file /hf/token --dir /weights
# reconstruct
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v ~/.cache/rawatta:/weights:ro -v "$PWD":/work \
  ghcr.io/sezer-muhammed/rawatta:latest run --images photos --config rawatta --out out
```

- `mkdir -p ~/.cache/huggingface ~/.cache/rawatta` first: a folder that is missing when Docker mounts
  it is created owned by root, and then neither the download nor `hf auth login` can write to it.
- The token file: `hf auth login` (from `pip install -U huggingface_hub`; older versions:
  `huggingface-cli login`) saves it as `~/.cache/huggingface/token`; you can also write the token into
  that file by hand (`(umask 077; cat > ~/.cache/huggingface/token)`, paste, Enter, Ctrl-D). The
  weights are gated on Hugging Face under the FAIR Noncommercial Research License: request access on
  <https://huggingface.co/facebook/VGGT-Omega>. Meta reviews requests by hand, which can take a while;
  downloads are refused (401/403) until the account is approved. A fine-grained token also needs "Read
  access to contents of all public gated repos you can access". The token is never printed or put on
  a command line.
- `download-weights` downloads `vggt_omega_1b_512.pt` (resumable, SHA-256 checked), converts it to
  `~/.cache/rawatta/vggt_omega_1b_512.safetensors` and deletes the `.pt`. Its options (`--fp8`,
  `--keep-pt`, ...) are listed by `--help`.
- `run` reads `photos/` and writes `out/` in the current folder (mounted at `/work`, the container's
  working directory). `--user "$(id -u):$(id -g)"` makes the outputs belong to you.

## What is in the image

| In the image | Where |
|---|---|
| `rawatta` (release build), the entrypoint | `/usr/local/bin/rawatta` |
| `download-weights` (runs `tools/download_weights.sh`) | `/usr/local/bin/download-weights` |
| The two CUDA 12.8 libraries `rawatta` loads: NVRTC (kernels are compiled at run time, no nvcc needed) and cuBLASLt | `/usr/local/cuda-12.8/lib64` |
| ffmpeg (for `--video`), python3 + numpy (download/converter, example helpers), curl | Ubuntu 24.04 packages |
| Weight converter and download script | `/opt/rawatta/tools/` |
| Example scripts and Python helpers | `/opt/rawatta/examples/` |
| Config files and the config grammar (`GRAMMAR.md`) | `/opt/rawatta/configs/` |
| Licence files, including the licences of the linked Rust crates | `/opt/rawatta/LICENSE*`, `/opt/rawatta/THIRD-PARTY-LICENSES.md`, `/opt/rawatta/NOTICE` |

The image runs under any `--user uid:gid`: `HOME` is `/tmp`, and nothing depends on a user account.
Without `--user` it runs as uid 1000 (named `rawatta`), never as root.

Size, measured on the build machine: about 0.84 GB to download (compressed layers), 1.6 GB unpacked, of
which 1.0 GB are the CUDA libraries (cuBLASLt alone is 0.72 GB). (`docker image ls` shows about 2.6 GB
on Docker's containerd image store, which counts the compressed layers and the unpacked files together.)

## Requirements

- Linux x86_64 with an NVIDIA GPU. RAWATTA and RAWATTA-Fast need compute capability 8.9 or newer
  (Ada, Hopper, Blackwell; run on Ada (RTX 4070 SUPER) and, in the paper, Blackwell sm_120 (RTX PRO
  6000); Hopper (sm_90) and sm_100 are built and ptxas-checked but not yet run). The exact engine (`--config exact`) is built and bit-checked for sm_80+
  (Ampere), but has not yet been run on Ampere hardware.
- An NVIDIA driver that supports CUDA 12.8 or newer (`nvidia-smi` shows "CUDA Version: 12.8" or
  higher).
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  so that `docker run --gpus all` works.

## Check the GPU

```sh
docker run --rm --gpus all --user "$(id -u):$(id -g)" ghcr.io/sezer-muhammed/rawatta:latest info
```

`info` prints the GPU, its compute capability and memory, the driver's CUDA version, whether
RAWATTA and the exact engine can run, whether NVRTC and cuBLASLt load, and whether the weights are
found. Without `--gpus` the container sees no GPU and `info` says so.

## Weights

The image sets `RAWATTA_WEIGHTS=/weights/vggt_omega_1b_512.safetensors`, so mount the folder that
holds the converted weights at `/weights` (read-only is enough for `run`). For a file with another
name or in another folder, mount its folder and point `RAWATTA_WEIGHTS` at it (or pass `--weights` to
`rawatta run`):

```sh
-v /data/models:/models:ro -e RAWATTA_WEIGHTS=/models/vggt_omega_1b_512.safetensors
```

Configs with `+fp8` (none of the named settings uses it) also need the FP8 side file
`vggt_omega_1b_512_fp8.safetensors` in the same folder: add `--fp8` to the `download-weights` command.

To convert a checkpoint you already have:

```sh
docker run --rm --user "$(id -u):$(id -g)" -v ~/Downloads:/in:ro -v ~/.cache/rawatta:/weights \
  --entrypoint python3 ghcr.io/sezer-muhammed/rawatta:latest \
  /opt/rawatta/tools/convert_weights.py --input /in/vggt_omega_1b_512.pt \
  --output /weights/vggt_omega_1b_512.safetensors --check
```

Without Docker, the same script runs natively (`tools/download_weights.sh`, needs curl, python3 >= 3.10
and numpy) next to `cargo build --release -p rawatta-cli`; see the main README.

## Run

Data in another place: mount it too and use the path inside the container.

```sh
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v ~/.cache/rawatta:/weights:ro -v /mnt/datasets:/data:ro -v "$PWD":/work \
  ghcr.io/sezer-muhammed/rawatta:latest \
  run --images /data/garden/images --config rawatta-fast --out out/garden
```

A video (frames are sampled with the ffmpeg in the image):

```sh
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v ~/.cache/rawatta:/weights:ro -v "$PWD":/work \
  ghcr.io/sezer-muhammed/rawatta:latest run --video clip.mp4 --max-frames 32 --out out/clip
```

`--config` takes `rawatta` (the default), `rawatta-fast`, `exact`, or a config grammar string
(`/opt/rawatta/configs/GRAMMAR.md`). The outputs (`cameras.json`, `.npy` arrays, `points.ply`, a
COLMAP text model, `timing.json`) are described in the main README.

### Choosing a GPU

`rawatta` runs on the first GPU it sees. Pick one with Docker, for example `--gpus device=1`.

### Python helpers and examples

```sh
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/work --entrypoint python3 \
  ghcr.io/sezer-muhammed/rawatta:latest /opt/rawatta/examples/python/load_outputs.py out
```

The example scripts in `examples/` can also run `rawatta` through this image from the host
(`RAWATTA_RUNNER=docker`); see `examples/README.md`.

### Timing inside Docker

`timing.json` reports the forward latency and the peak GPU memory. Time on an idle GPU with the
paper's protocol (`--reps 5 --warmup 2`). Peak memory comes from `cuMemGetInfo`, which counts the
whole card, so other processes on the GPU inflate it.

## Troubleshooting

| Message | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Install the NVIDIA Container Toolkit and restart Docker. |
| `info` says `Driver CUDA version: none` | The container was started without `--gpus all`, or the host has no NVIDIA driver. |
| `info` says the driver is too old | Update the driver to one that supports CUDA 12.8 or newer. |
| `download-weights: the output folder /weights is not writable` | `sudo chown "$(id -u):$(id -g)" ~/.cache/rawatta` (Docker created it as root because it was missing), and pass `--user "$(id -u):$(id -g)"`. The wrapper prints this command when the folder belongs to root. |
| `download-weights: cannot read the token file` | Run `hf auth login` on the host, or save the token as `~/.cache/huggingface/token`, and mount that folder at `/hf`. If Docker created `~/.cache/huggingface` as root (it was missing), first `sudo chown "$(id -u):$(id -g)" ~/.cache/huggingface`; the wrapper says so when it sees an empty root-owned folder. |
| `Hugging Face rejected the token (401): it is invalid or expired` | Create a new Read token and save it to the token file. |
| `this account has no access to facebook/VGGT-Omega (403)` | Request access on the model page with the account the token belongs to and wait for approval; a fine-grained token also needs read access to public gated repos. |
| `could not reach Hugging Face ... (network error)` | Check the network or proxy (curl got no reply), then run the command again. |
| `VGGT-Omega weights not found at /weights/...` | Mount the host folder that holds the weights at `/weights` (`-v ~/.cache/rawatta:/weights:ro`), or set `RAWATTA_WEIGHTS` to the path inside the container (see [Weights](#weights)). |
| `Permission denied` when writing outputs | Pass `--user "$(id -u):$(id -g)"` and check that the current folder is yours. |
| `config "rawatta" needs compute capability 8.9+` | The GPU is Ampere or older: use `--config exact` (sm_80+; not yet run on Ampere hardware). |

## Build details

- Building the image needs BuildKit (the default since Docker 23; on older Docker run
  `DOCKER_BUILDKIT=1 docker build ...`): the Dockerfile uses `RUN --mount=type=cache`.
- `docker/Dockerfile` has two stages, both pinned by digest (the tag is in a comment next to each
  `FROM`). The builder (`nvidia/cuda:12.8.1-devel-ubuntu24.04`) installs Rust 1.99.1 with the official
  `rustup-init` binary, pinned by version and SHA-256, and runs
  `cargo build --release --locked -p rawatta-cli` with cache mounts for the cargo registry and the
  target folder. It touches the workspace sources first, so the three workspace crates (and their
  `include_str!` kernels) are always rebuilt from the files in the build context; the crates.io
  dependencies come from the cache.
- The runtime stage starts from `nvidia/cuda:12.8.1-base-ubuntu24.04` (CUDA runtime only) and adds the
  NVIDIA packages `cuda-nvrtc-12-8` and `libcublas-12-8` at the versions of the 12.8.1 release. A dpkg
  filter leaves out the files of those packages that `rawatta` never loads: `libcublas.so.12` (it
  calls cuBLASLt only, and `libcublasLt` does not depend on it), `libnvblas` and the alternative NVRTC
  builds. Compared with the `-runtime` base image, this drops cuFFT, cuRAND, cuSPARSE, cuSOLVER, NPP and
  nvJPEG, which `rawatta` does not use.
- The build runs `docker/check_cuda_libs.py`, which loads `libnvrtc.so.12` and `libcublasLt.so.12`
  the way `rawatta` does and compiles a test kernel with NVRTC for every GPU target the engine uses
  (compute_80 to compute_120). The build fails if a library is missing. It needs no GPU.
- Build arguments: `RUST_VERSION` (1.99.1), `RUSTUP_VERSION` / `RUSTUP_INIT_SHA256`,
  `NVRTC_PKG_VERSION` / `CUBLAS_PKG_VERSION`, and `VERSION` / `REVISION` for the image labels.
- `.dockerignore` is an allowlist: only `Cargo.toml`, `Cargo.lock`, `crates/`, `configs/`, `tools/`,
  `examples/`, the licence files (including `THIRD-PARTY-LICENSES.md`), the library check and `docker/download-weights` enter the build
  context. Weights, datasets and run outputs never do.
- Release images are built by `.github/workflows/docker.yml` when a `v*` tag is pushed and published
  as `ghcr.io/sezer-muhammed/rawatta:<tag>`; a release version (no pre-release part) also moves
  `:latest`.

## Licences

The image is for research use. The rawatta code in it is under these licences:

| Code | Licence | File in the image |
|---|---|---|
| The VGGT-Omega engine (`crates/vggt-omega`), compiled into the `rawatta` binary | FAIR Noncommercial Research License, like the weights | `/opt/rawatta/LICENSE-VGGT-OMEGA` |
| The DINOv3 backbone code ported in that engine (patch embedder, blocks, RoPE) | also the DINOv3 License | `/opt/rawatta/LICENSE-DINOV3` |
| Everything else: the CUDA building blocks and attention kernels, the CLI, configs, tools and examples | Apache-2.0 | `/opt/rawatta/LICENSE` |
| The Rust crates statically linked into the binary | their own licences (MIT, Apache-2.0, ISC, Zlib, BSD-3-Clause, IJG, Unicode-3.0, ...) | `/opt/rawatta/THIRD-PARTY-LICENSES.md` |

Because the binary contains the engine, using the binary is bound by the FAIR licence too, including
its noncommercial terms. The example scripts do not include the example photos; each scene's
`SOURCE.md` under `/opt/rawatta/examples/` names their source and licence. The CUDA libraries come
from NVIDIA's `nvidia/cuda` base image and NVIDIA's CUDA packages under the NVIDIA Deep Learning
Container License (`/NGC-DL-CONTAINER-LICENSE` in the image) and the CUDA EULA. The Ubuntu packages
(ffmpeg, Python, numpy, ...) keep their own licences (`/usr/share/doc/*/copyright`); ffmpeg is
Ubuntu's GPL-2.0-or-later build, and `NOTICE` says where to get its source.
