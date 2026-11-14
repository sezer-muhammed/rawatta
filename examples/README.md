# Examples

Scripts that run VGGT-Ω with RAWATTA on your own photos or videos, compare the three settings (RAWATTA,
RAWATTA-Fast and the exact engine), read the outputs from Python, and build the example gallery of the project page.

| File | What it does |
|---|---|
| `run_on_photos.sh` | Cameras, depth and a coloured point cloud from a folder of photos |
| `run_on_video.sh` | The same from a video, using frames spread evenly over it |
| `compare_configs.sh` | Runs the exact engine, RAWATTA and RAWATTA-Fast on the same photos and prints a latency table |
| `python/load_outputs.py` | Loads `cameras.json`, the `.npy` arrays and `points.ply` with numpy only, prints a summary, optionally plots the cameras |
| `python/subsample_ply.py` | Shrinks a point cloud (voxel grid or random, keeps colours), numpy only |
| `python/timing_table.py` | Latency table from several `timing.json` files |
| `to_gaussian_splatting.md` | How to train Gaussian splats (gsplat, nerfstudio, Brush) from the `colmap/` output |
| `make_gallery.sh` | Runs the example scenes and writes the gallery files of the project page (`python/gallery_index.py` writes its `index.json`) |
| `scenes.txt` | The gallery scenes, in the order the page shows them |
| `buddha/`, `korno-rock/` | The two example scenes: `fetch.sh`, source and licence (`SOURCE.md`), checksums, `scene.json` (title and credit for the gallery) |
| `tests/run_tests.sh` | Checks that need no GPU and no weights (uses `tests/fake_rawatta`, which writes synthetic outputs, and `tests/fake_docker`) |
| `tests/mutation_check.py` | Breaks the scripts on purpose, one way at a time, and checks that `run_tests.sh` catches each break |

## Requirements

- Linux x86_64 with an NVIDIA driver for CUDA 12.8 or newer.
- RAWATTA and RAWATTA-Fast need compute capability 8.9+ (Ada, Hopper, Blackwell; run on Ada (RTX 4070 SUPER) and,
  in the paper, Blackwell sm_120 (RTX PRO 6000); Hopper (sm_90) and sm_100 are built and ptxas-checked but not yet
  run). The exact engine is built and bit-checked for sm_80+, not yet run on Ampere hardware. `rawatta info` tells you what your GPU can run.
- Either Docker with the NVIDIA container toolkit (the host needs only Docker and the NVIDIA driver), or a native
  build of the `rawatta` binary.
- The VGGT-Ω weights, converted for rawatta (research use only, FAIR Noncommercial Research License). Download
  them once; the token file comes from `hf auth login` (or save your Hugging Face token in it by hand). The image
  `ghcr.io/sezer-muhammed/rawatta` is published at v0.1.0, and pulling it before then fails with "error from
  registry: denied": until then build it from the repository root with `docker build -f docker/Dockerfile -t rawatta .`
  and use `rawatta` as the image name (or tag the build `-t ghcr.io/sezer-muhammed/rawatta:latest`). Create both host
  folders first, so that they belong to you and not to root:

  ```bash
  mkdir -p ~/.cache/huggingface ~/.cache/rawatta
  docker run --rm --user "$(id -u):$(id -g)" -v ~/.cache/huggingface:/hf:ro -v ~/.cache/rawatta:/weights \
    --entrypoint download-weights ghcr.io/sezer-muhammed/rawatta:latest --token-file /hf/token --dir /weights
  ```

  Natively: `tools/download_weights.sh` (needs curl, python3>=3.10 and numpy). Both save
  `~/.cache/rawatta/vggt_omega_1b_512.safetensors`, where the scripts look by default; set `RAWATTA_WEIGHTS` to use
  another file.
- For the Python helpers: python3 with numpy. matplotlib is only needed for `load_outputs.py --plot`.

Until v0.1.0, with the image built as `rawatta` as above, set `RAWATTA_IMAGE=rawatta` for the scripts.
For a native build: `cargo build --release -p rawatta-cli`, then `RAWATTA_BIN=target/release/rawatta`.

## Binary or Docker

All scripts call rawatta through `lib/rawatta.sh`. Choose how with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `RAWATTA_RUNNER` | `auto` | `binary`, `docker`, or `auto` (the binary if found, otherwise Docker) |
| `RAWATTA_BIN` | `rawatta` | name or path of the binary, e.g. `target/release/rawatta` |
| `RAWATTA_IMAGE` | `ghcr.io/sezer-muhammed/rawatta:latest` | Docker image (see above until v0.1.0) |
| `RAWATTA_GPUS` | `all` | value for `docker run --gpus` |
| `RAWATTA_WEIGHTS` | `~/.cache/rawatta/vggt_omega_1b_512.safetensors` | weights file |
| `RAWATTA_CONFIG` | `rawatta` | setting for `run_on_photos.sh` / `run_on_video.sh`: `rawatta`, `rawatta-fast`, `exact` or a grammar string (`configs/GRAMMAR.md`) |

In Docker mode the scripts run the image as the main README does (same image, entrypoint, `--user` and weights
mount), but mount only the input and the output folder instead of the whole current folder:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -v ~/.cache/rawatta:/weights:ro \
  -v <photo folder>:/work/input:ro -v <out folder>:/work/out \
  ghcr.io/sezer-muhammed/rawatta:latest run --images /work/input --out /work/out --config rawatta ...
```

The folder of `RAWATTA_WEIGHTS` is mounted read-only at `/weights`; the image already points rawatta at
`/weights/vggt_omega_1b_512.safetensors`, so a file of another name is passed as `-e RAWATTA_WEIGHTS=/weights/<name>`.
A video is mounted as that one file, never its whole folder. Output files belong to you (`--user`). Extra
arguments must not hold host paths in Docker mode (use `RAWATTA_WEIGHTS`, not `--weights`). If the weights are
missing, the scripts stop before Docker starts and print the download command.

## Quick start

```bash
examples/buddha/fetch.sh                                  # 34 photos into data/examples/buddha/images
examples/run_on_photos.sh data/examples/buddha/images     # -> runs/buddha-rawatta/
python3 examples/python/load_outputs.py runs/buddha-rawatta --plot cameras.png
```

Your own photos:

```bash
examples/run_on_photos.sh ~/photos/garden runs/garden
RAWATTA_CONFIG=rawatta-fast examples/run_on_photos.sh ~/photos/garden runs/garden-fast
```

Anything after the photo folder and the optional output folder goes to `rawatta run` unchanged (`--` is
optional), for example `examples/run_on_photos.sh ~/photos/garden runs/garden --no-depth` or
`examples/run_on_photos.sh ~/photos/garden -- --reps 10 --warmup 2`. The scripts already pass `--config`
(from `RAWATTA_CONFIG`), and `run_on_video.sh` also passes `--max-frames` (from `MAX_FRAMES`). rawatta rejects
an option given twice, so when you put one of these among the extra arguments the script uses your value and
leaves out its own. `compare_configs.sh` takes the settings from `CONFIGS` only.

The scripts accept `.jpg`, `.jpeg`, `.png` and `.webp` photos, like rawatta.

A video:

```bash
MAX_FRAMES=32 examples/run_on_video.sh walk.mp4 runs/walk
VIDEO_MODE=extract examples/run_on_video.sh walk.mp4 runs/walk     # save the frames with ffmpeg first
```

Both modes use the same frames: exactly `MAX_FRAMES` of them (every frame if the video has fewer), evenly spaced
with the first and last frame included. `VIDEO_MODE=extract` saves them as `runs/walk/frames/frame_<index>.jpg`,
where `<index>` is the frame's number in the video, before running rawatta on that folder.

More frames need more GPU memory. If a run does not fit on your GPU, use fewer frames.

## Outputs

`rawatta run --out <dir>` writes:

| File | Content |
|---|---|
| `cameras.json` | per frame: world-to-camera 3x4 matrix (OpenCV axes: x right, y down, z forward), 3x3 intrinsics, image name. World = frame 0 |
| `extrinsics.npy`, `intrinsics.npy` | the same cameras, arrays of shape (N,3,4) and (N,3,3) |
| `depth.npy`, `conf.npy` | depth and confidence maps, (N,H,W) (not written with `--no-depth`) |
| `points.ply` | coloured point cloud from the depth maps, confidence-filtered |
| `colmap/` | COLMAP text model (`cameras.txt`, `images.txt`, `points3D.txt`) and `images/`: the frames as the network saw them (PNG, the size in `cameras.txt`) |
| `timing.json` | forward latency (`median_ms`, `iqr_ms` over `reps` timed passes after `warmup` untimed ones), peak GPU memory of the run (`peak_device_mib`), setting, GPU name |

`rawatta run` times a single forward pass unless you pass `--reps` (and `--warmup`). One pass is enough to
get the outputs; for a latency you want to quote, use `--reps 10 --warmup 2` with the GPU otherwise idle.
`load_outputs.py` and `timing_table.py` show the number of timed passes next to the median.

Load them in your own code:

```python
import sys; sys.path.insert(0, "examples/python")
from load_outputs import load_run, camera_centres
run = load_run("runs/garden")
print(run["w2c"].shape, camera_centres(run["w2c"])[:3])
print(run["points"].shape, run["colours"].dtype)          # (P,3) float64, (P,3) uint8
```

## Comparing the settings

```bash
examples/compare_configs.sh data/examples/buddha/images runs/compare-buddha
```

This runs `exact`, `rawatta` and `rawatta-fast` one after another, each with `--reps 10 --warmup 2`, and prints
a table read from their `timing.json` files: timed passes, median and IQR of the forward latency, peak memory,
and the speed-up relative to the exact engine. Run it with the GPU otherwise idle; a busy GPU makes the numbers
meaningless. `CONFIGS="exact rawatta"` runs a subset; `REPS` and `WARMUP` change the number of passes.

## Example scenes

| Scene | Kind | Photos | Licence | Source |
|---|---|---|---|---|
| `buddha` | object | 34 of 67 | CC BY 4.0 | [Buddha Dataset](https://github.com/alicevision/dataset_buddha) by S. Gasparini, F. Castan, Y. Lanthony (AliceVision), [doi:10.5281/zenodo.1203283](https://doi.org/10.5281/zenodo.1203283) |
| `korno-rock` | outdoor (rock face) | 32 of 96 | CC0 1.0 | [Photogrammetry of Korno climbing rock](https://commons.wikimedia.org/wiki/Category:Photogrammetry_of_Korno_climbing_rock) by Zbytovsky, Wikimedia Commons |

The photos are not stored in this repository. `examples/<scene>/fetch.sh [dest]` downloads the originals,
verifies each one against the SHA-256 in `manifest.tsv`, and saves 1024 px JPEGs as `<dest>/images/frame_NNN.jpg`
(default `dest`: `data/examples/<scene>`). It needs curl and one of Pillow, ImageMagick or ffmpeg. Each
`SOURCE.md` records the source page, author, licence text and every change we made. Outputs made from the
Buddha photos are adaptations of a CC BY 4.0 work and must carry its attribution.

## Building the gallery

```bash
for s in buddha korno-rock; do examples/$s/fetch.sh data/examples/$s; done
examples/make_gallery.sh
```

For each scene listed in `scenes.txt` this runs `rawatta run --config rawatta --reps 10 --warmup 2` on every
photo of `data/examples/<scene>/images`, then writes to `docs/static/examples/<scene>/`: `points.ply`
(voxel-subsampled to at most 300k points and 8 MB), `cameras.json` and `thumb.jpg`, and finally
`docs/static/examples/index.json`, shaped `{"scenes": [ {...}, ... ]}`. The scenes appear in the order of
`scenes.txt` (the page opens the first one), however often one of them is rebuilt; scenes already in
`index.json` but not in the list follow, sorted by id. The page viewer (`docs/static/js/viewer.js`) reads `id`,
`title`, `dir`, `points`, `cameras` and `caption`; the other fields are kept for reference.

| Field | From |
|---|---|
| `id`, `title` | `examples/<scene>/scene.json` |
| `dir`, `points`, `cameras`, `thumb` | the scene folder relative to `index.json`, and the file names in it (`points.ply`, `cameras.json`, `thumb.jpg`) |
| `caption` | the text the page shows under the viewer: the model and the setting, the photo count, the measured latency with GPU and number of timed passes, and the photo credit with the licence and its URL, e.g. `Cameras and point cloud predicted by VGGT-Ω from 34 photos, accelerated with RAWATTA (--config rawatta). Forward pass <median> ms on <GPU> (median of 10 timed runs). Photos: Buddha Dataset by ..., CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); 34 of 67 photos, downscaled.` |
| `photos` | the number of photos in the scene's image folder (the caption's count) |
| `frames`, `config`, `latency_ms`, `latency_iqr_ms`, `reps`, `warmup`, `gpu` | the run's `timing.json` (`latency_iqr_ms` is null for a single timed pass) |
| `point_count` | number of points in the subsampled `points.ply` |
| `attribution` | the `Photos: ...` part of the caption |
| `credit`, `source_photos`, `licence`, `licence_url`, `source`, `author`, `changes`, `kind` | `examples/<scene>/scene.json` |

The photo count comes from the image folder, not from `scene.json`: if you add or remove photos, the caption
follows (`make_gallery.sh` warns when the folder differs from what `fetch.sh` writes). A run that did not use
every photo of the folder (for example an old run reused with `GALLERY_SKIP_RUN=1`) is refused before anything
is written, so the caption cannot contradict itself.

The latency is what was measured on the machine that built the gallery, so run it with the GPU otherwise idle.
`GALLERY_REPS` and `GALLERY_WARMUP` change the number of passes, `GALLERY_ARGS` adds `rawatta run` arguments, and
`GALLERY_CONFIG` the setting. Set `EXAMPLES_DATA`, `GALLERY_OUT` or `GALLERY_RUNS` to use other folders, and
`GALLERY_SKIP_RUN=1` to rebuild the gallery files from earlier runs without the GPU. A `timing.json` marked
`"synthetic": true` (from `tests/fake_rawatta`) stops the script; tests allow it with
`GALLERY_ALLOW_SYNTHETIC=1`, which is refused when `GALLERY_OUT` is the default `docs/` folder.

## Tests

```bash
examples/tests/run_tests.sh                  # about half a minute
python3 examples/tests/mutation_check.py     # a few minutes: proves the checks can fail
```

`run_tests.sh` compiles the Python files, runs the unit tests (`python/test_*.py`), and runs every script end to
end against `tests/fake_rawatta`, a stand-in that writes files in the rawatta output layout filled with
synthetic data (its timings are made up and marked `"synthetic": true`). Like rawatta, the stand-in rejects an
option given twice. Docker mode runs against `tests/fake_docker`, which plays the image (ENTRYPOINT rawatta,
weights at `/weights`) and checks the mounts. The tests also check the frame counts of `VIDEO_MODE=extract`,
the gallery caption, photo count and scene order, and that the gallery refuses synthetic timings and never
writes them under `docs/`. No GPU or weights needed.

`mutation_check.py` breaks the scripts on purpose, one way at a time (30 breaks, such as passing `--config`
twice, mounting a video's whole folder, or dropping the licence URL from the caption), and checks that
`run_tests.sh` then fails at the check written for that break.

## Licence

The scripts in this folder are Apache-2.0 (each file carries an SPDX line), like the rest of the RAWATTA code.
The example photos keep their own licences, recorded in each scene's `SOURCE.md`.
