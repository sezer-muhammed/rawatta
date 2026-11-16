<!-- SPDX-License-Identifier: Apache-2.0 -->
# Weight tools

| file | what it does |
|---|---|
| `download_weights.sh` | downloads `facebook/VGGT-Omega` → `vggt_omega_1b_512.pt` with curl at a pinned Hugging Face revision, checks its SHA-256 against Hugging Face's LFS id (resumable; a corrupt file is replaced), then converts it. The token is never printed or put on a command line. The Docker image installs it as `download-weights` |
| `convert_weights.py` | converts the official checkpoint to the `rawatta-vggt-1` safetensors layout the engine loads (`--fp8` adds the FP8 side file); **no PyTorch needed** (`--use-torch` to load with torch instead) |
| `tests/` | `pytest tools/tests`: casts, the torch-free reader (including malformed and hostile pickles), layout, byte identity with the original converter (`tests/reference_convert_vggt.py`) on synthetic checkpoints with the real key names (`tests/synthetic_checkpoint.py`), and `download_weights.sh` against a local mock of the Hugging Face API |

## Getting the weights

The checkpoint is gated: request access on <https://huggingface.co/facebook/VGGT-Omega> with your
Hugging Face account (Meta reviews requests by hand; downloads return 401/403 until you are
approved), then save a Read token (`hf auth login` writes `~/.cache/huggingface/token`, or save the
token in that file by hand).

With Docker (the host needs only Docker and the NVIDIA driver), once. The image
is `ghcr.io/sezer-muhammed/rawatta:latest` (or build it with
`docker build -f docker/Dockerfile -t ghcr.io/sezer-muhammed/rawatta:latest .`). Create both host folders first, so that they belong to you and
not to root:

```sh
mkdir -p ~/.cache/huggingface ~/.cache/rawatta
docker run --rm --user "$(id -u):$(id -g)" -v ~/.cache/huggingface:/hf:ro -v ~/.cache/rawatta:/weights \
  --entrypoint download-weights ghcr.io/sezer-muhammed/rawatta:latest --token-file /hf/token --dir /weights
```

Natively (needs curl, python3 >= 3.10 and numpy):

```sh
tools/download_weights.sh        # -> ~/.cache/rawatta/vggt_omega_1b_512.safetensors, rawatta's default path
```

`download_weights.sh [--token-file PATH] [--dir DIR] [--fp8] [--keep-pt] [--no-convert]`: without
`--token-file` the token comes from `$HF_TOKEN`, `$HF_HOME/token` or `~/.cache/huggingface/token`.
`--fp8` also writes `vggt_omega_1b_512_fp8.safetensors` (only `+fp8` research configs use it),
`--keep-pt` keeps the 4.6 GB `.pt`, `--no-convert` only downloads it. An interrupted download
resumes on the next run. `RAWATTA_CACHE` changes the default output folder (the image sets it to
`/weights`), `RAWATTA_HF_REVISION` selects another revision of the Hugging Face repo and
`HF_ENDPOINT` a mirror. `rawatta` itself reads the weights from `--weights`, `RAWATTA_WEIGHTS` or
`${XDG_CACHE_HOME:-~/.cache}/rawatta/vggt_omega_1b_512.safetensors`; the script says which applies.

With a checkpoint you already have:

```sh
python3 tools/convert_weights.py --input vggt_omega_1b_512.pt --check
```

## The converter

It needs Python 3.10+ and numpy. It reads the torch zip archive with a restricted unpickler (only
the globals a tensor state dict uses; anything else is refused, so no code from the file runs; a
malformed file is reported as an error) and memory-maps the storages. The tests show byte identity
with the original torch converter on synthetic checkpoints (1/32 of the real widths, the real key
names and dtypes). A synthetic checkpoint with the real shapes (4.58 GB float32, 1413 tensors
including the text head the converter drops, wrapped in `model`, with strided views) also gave
byte-identical BF16/F32 and FP8 files (checked in this build on 2026-11-14). The real checkpoint has
not been converted in this build yet: the gated download is pending.

The VGGT-Omega weights are under the FAIR Noncommercial Research License: research use only. Do not
redistribute the checkpoint or the converted files, and do not bake them into images.
