# SPDX-License-Identifier: Apache-2.0
#
# rawatta container image: the `rawatta` binary, the two CUDA libraries it loads (NVRTC and
# cuBLASLt), ffmpeg (for --video) and python3 + numpy (for the weight download/converter and the
# example helpers). No weights are inside: mount them at run time.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t rawatta .
# Use (see docker/README.md):
#   docker run --rm --user "$(id -u):$(id -g)" -v ~/.cache/huggingface:/hf:ro -v ~/.cache/rawatta:/weights \
#     --entrypoint download-weights rawatta --token-file /hf/token --dir /weights
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" -v ~/.cache/rawatta:/weights:ro -v "$PWD":/work \
#     rawatta run --images photos --config rawatta --out out
#
# The kernels are compiled at run time with NVRTC, so the image needs no nvcc. The binary loads
# libnvrtc.so.12 (which loads libnvrtc-builtins) and libcublasLt.so.12 from this image and
# libcuda.so.1 from the host driver (the NVIDIA Container Toolkit mounts it when you pass --gpus).
#
# Base images are pinned by digest (the multi-arch index digest of the tag in the comment). To move
# to a newer patch release: `docker buildx imagetools inspect nvidia/cuda:<tag>` and update both the
# tag comment and the digest.

# --------------------------------------------------------------------------------------------------
# Builder: CUDA devel image + Rust 1.99.1 (rustup-init, checksum-verified), release build of the CLI.
# --------------------------------------------------------------------------------------------------
# nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM nvidia/cuda@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS builder

ARG RUST_VERSION=1.99.1
# rustup-init from static.rust-lang.org, pinned by version and SHA-256
# (https://static.rust-lang.org/rustup/archive/<version>/x86_64-unknown-linux-gnu/rustup-init.sha256).
ARG RUSTUP_VERSION=1.29.3
ARG RUSTUP_INIT_SHA256=b1c8cb14e30889e7e2999e6b69e62d9e9b9ffc0f4df8180564d306d9680e6c55
ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:${PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl pkg-config \
 && rm -rf /var/lib/apt/lists/*

# rustup itself verifies the toolchain it downloads against the signed channel manifest.
RUN curl --proto '=https' --tlsv1.2 -sSfLo /tmp/rustup-init \
      "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
 && echo "${RUSTUP_INIT_SHA256}  /tmp/rustup-init" | sha256sum -c - \
 && chmod +x /tmp/rustup-init \
 && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain "${RUST_VERSION}" \
 && rm /tmp/rustup-init \
 && rustc --version \
 && cargo --version

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY configs configs
COPY crates crates

# The registry and target caches persist between local builds; the binary is copied out of the
# cached target directory in the same step.
# Cargo decides what to rebuild from file mtimes, and COPY keeps the mtimes of the build context. A
# tree whose changed files are older than the cached build (an older commit extracted with
# `git archive`, a restored backup) would therefore install a stale binary (include_str! kernels
# included). Touching the workspace sources first makes cargo rebuild the three workspace crates on
# every build; the crates.io dependencies stay cached (cargo keys them by version and features).
RUN --mount=type=cache,id=rawatta-cargo-registry,target=/opt/cargo/registry \
    --mount=type=cache,id=rawatta-cargo-target,target=/src/target \
    find crates configs -type f -exec touch {} + \
 && cargo build --release --locked -p rawatta-cli \
 && install -m 0755 target/release/rawatta /usr/local/bin/rawatta \
 && /usr/local/bin/rawatta --version

# --------------------------------------------------------------------------------------------------
# Runtime: CUDA base image (cudart only) + the two CUDA libraries rawatta loads, pinned to the
# versions of the 12.8.1 release, plus ffmpeg, python3, numpy and curl. Runs under any uid:gid
# (`--user "$(id -u):$(id -g)"`); the default user is `rawatta` (uid 1000).
# --------------------------------------------------------------------------------------------------
# nvidia/cuda:12.8.1-base-ubuntu24.04
FROM nvidia/cuda@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2 AS runtime

ARG VERSION=dev
ARG REVISION=unknown
# ref.name overrides the base image's misleading `ubuntu`.
LABEL org.opencontainers.image.title="rawatta" \
      org.opencontainers.image.description="RAWATTA (Regime-AWare ATTention Approximation): VGGT-Omega multi-view 3D reconstruction on a native CUDA engine. Research use only (FAIR Noncommercial Research License for the VGGT-Omega engine and weights). No weights included." \
      org.opencontainers.image.source="https://github.com/sezer-muhammed/rawatta" \
      org.opencontainers.image.url="https://github.com/sezer-muhammed/rawatta" \
      org.opencontainers.image.documentation="https://github.com/sezer-muhammed/rawatta/blob/main/docker/README.md" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-FAIR-Noncommercial-Research" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.ref.name="rawatta" \
      org.opencontainers.image.base.name="docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04" \
      org.opencontainers.image.base.digest="sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2"

# cuda-nvrtc-12-8: libnvrtc.so.12 + libnvrtc-builtins.so.12.8. libcublas-12-8: libcublasLt.so.12.
# Versions = those in nvidia/cuda:12.8.1-runtime-ubuntu24.04. The dpkg filter leaves out the files of
# those packages that rawatta never loads (~220 MB): libcublas.so.12 (rawatta calls cuBLASLt only, and
# libcublasLt does not depend on it), libnvblas, and the alternative NVRTC builds (*.alt.so).
ARG NVRTC_PKG_VERSION=12.8.93-1
ARG CUBLAS_PKG_VERSION=12.8.4.1-1
ENV DEBIAN_FRONTEND=noninteractive
RUN printf '%s\n' \
      'path-exclude=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so*' \
      'path-exclude=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvblas.so*' \
      'path-exclude=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.alt.so*' \
      'path-exclude=/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc-builtins.alt.so*' \
      > /etc/dpkg/dpkg.cfg.d/rawatta-cuda-slim \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      "cuda-nvrtc-12-8=${NVRTC_PKG_VERSION}" "libcublas-12-8=${CUBLAS_PKG_VERSION}" \
      ca-certificates curl ffmpeg python3 python3-numpy \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb

# Fail the build if NVRTC (with its builtins) or cuBLASLt is missing or cannot be loaded, or if
# NVRTC cannot compile for one of the GPU targets the engine uses. Needs no GPU.
COPY docker/check_cuda_libs.py /opt/rawatta/docker/check_cuda_libs.py
RUN python3 /opt/rawatta/docker/check_cuda_libs.py

COPY --from=builder /usr/local/bin/rawatta /usr/local/bin/rawatta
COPY docker/download-weights /usr/local/bin/download-weights
COPY LICENSE LICENSE-VGGT-OMEGA LICENSE-DINOV3 THIRD-PARTY-LICENSES.md NOTICE* /opt/rawatta/
COPY configs /opt/rawatta/configs
COPY tools /opt/rawatta/tools
COPY examples /opt/rawatta/examples

# The CUDA image ships an `ubuntu` user with uid/gid 1000; rename it to `rawatta` (no supplementary
# groups, no home folder). Nothing depends on it: any `--user uid:gid` works, because HOME is /tmp
# (world-writable) and /work and /weights are writable by everyone when nothing is mounted there.
RUN if id -u ubuntu >/dev/null 2>&1; then \
      groupmod -n rawatta ubuntu && usermod -l rawatta -d /tmp -G "" ubuntu && rm -rf /home/ubuntu; \
    else \
      useradd -M -d /tmp -u 1000 -U -s /usr/sbin/nologin rawatta; \
    fi \
 && chmod 0755 /usr/local/bin/download-weights \
 && mkdir -p /work /weights \
 && chmod 1777 /work /weights \
 && rawatta --version \
 && download-weights --help >/dev/null

# Weights: mount the folder that holds vggt_omega_1b_512.safetensors (and the optional
# vggt_omega_1b_512_fp8.safetensors) at /weights, or point RAWATTA_WEIGHTS / --weights elsewhere.
# RAWATTA_CACHE is where download-weights writes when --dir is not given.
ENV RAWATTA_WEIGHTS=/weights/vggt_omega_1b_512.safetensors \
    RAWATTA_CACHE=/weights \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1

USER 1000:1000
WORKDIR /work
ENTRYPOINT ["rawatta"]
CMD ["--help"]
