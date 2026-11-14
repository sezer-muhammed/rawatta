# Training Gaussian splats from a rawatta run

`rawatta run` writes a COLMAP text model to `<out>/colmap/`:

| File | Content |
|---|---|
| `cameras.txt` | one `PINHOLE` camera per frame, at the size the network ran at |
| `images.txt` | one pose per frame: world-to-camera rotation (quaternion) and translation, camera id, image file name |
| `points3D.txt` | coloured 3D points from the depth maps (filtered by confidence) |
| `images/` | the frames as the network saw them (cropped and resized), saved as PNG under the names in `images.txt` |

The names in `images.txt` are those of `colmap/images/`: the input file name with a `.png` extension
(`frame_000.jpg` becomes `frame_000.png`; spaces become `_`; if two inputs share a name, the frame number is
put in front, as in `00001_a.png`). They are not the names of your original photos,
and the pixel size is that of `cameras.txt`, not of the photos.

Splatting tools that read COLMAP can start from this model instead of running structure-from-motion first.
The poses are in the frame of the first photo (frame 0 at the origin). There is no metric scale and no gravity
direction, which these tools do not need: gsplat and nerfstudio re-centre and re-scale the scene themselves.

## 1. Make a dataset folder

Most tools expect this layout:

```
splat/<scene>/
  images/       the frames named in sparse/0/images.txt (copied from <out>/colmap/images/)
  sparse/0/     cameras.txt, images.txt, points3D.txt
```

```bash
scene=buddha
out=runs/buddha-rawatta                 # the --out folder of the rawatta run
mkdir -p "splat/$scene/sparse/0"
cp "$out"/colmap/{cameras,images,points3D}.txt "splat/$scene/sparse/0/"
cp -r "$out/colmap/images" "splat/$scene/images"
```

This works the same for photo and video runs: `colmap/images/` holds the frames rawatta used either way, and
their file names and pixel sizes match `images.txt` and `cameras.txt` by construction. Check it with:

```bash
grep -v '^#' "splat/$scene/sparse/0/cameras.txt" | head -3      # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
awk '!/^#/ && NF >= 10 { print $10 }' "splat/$scene/sparse/0/images.txt" | head -3   # file names
ls "splat/$scene/images" | head -3
```

### Training at the full photo resolution (optional)

The frames in `colmap/images/` are small: the network's input size, 512 px on the long side with the short side
rounded to a multiple of 16 (for example 512 x 288 for 16:9 photos). To train on
your original photos instead, you have to rename them to the names in `images.txt` and rescale the cameras. The
rescaling is only correct when rawatta resized the photos as a whole without cropping: no `--crop-aspect`, all
photos the same size, and an aspect ratio between 1:2 and 2:1 (rawatta centre-crops anything outside that range).
Then, for each `PINHOLE` line (`fx fy cx cy`), multiply `fx` and `cx` by `photo_width / WIDTH` and `fy` and `cy` by
`photo_height / HEIGHT`, and write the new `WIDTH HEIGHT`. If the network size was rounded to whole 16-pixel
patches, the scale factors differ slightly between width and height; that is expected. When in doubt, use
`colmap/images/` as above.

### Optional: thin out the initial points

`points3D.txt` can hold millions of points. That is fine for initialisation, but it makes the first iterations
slower. To start from fewer points, keep every k-th point line:

```bash
awk '/^#/ || (NR % 4 == 0)' "$out/colmap/points3D.txt" > "splat/$scene/sparse/0/points3D.txt"
```

The 2D-track fields in COLMAP files are not used by these trainers.

## 2a. gsplat

[gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0). Its example trainer reads `images/` and
`sparse/0/` (falling back to `sparse/`) through pycolmap, which reads text and binary models. It supports
`PINHOLE`, `SIMPLE_PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV` and `OPENCV_FISHEYE` cameras.

```bash
git clone https://github.com/nerfstudio-project/gsplat && cd gsplat/examples
pip install -r requirements.txt
python simple_trainer.py default --data_dir ../../splat/buddha --data_factor 1 --result_dir ../../results/buddha
```

Keep `--data_factor 1`: the frames from `colmap/images/` are already small. With a factor above 1, gsplat looks
for (or makes) an `images_<factor>` folder.

## 2b. nerfstudio (splatfacto)

[nerfstudio](https://github.com/nerfstudio-project/nerfstudio) (Apache-2.0). Its `colmap` dataparser reads text
models (`cameras.txt`, `images.txt`, `points3D.txt`) as well as binary ones, and by default loads the 3D points to
initialise splatfacto. Its default model path is `colmap/sparse/0`, so point it at `sparse/0`:

```bash
ns-train splatfacto --data splat/buddha colmap --colmap-path sparse/0 --images-path images --downscale-factor 1
```

Run `ns-train splatfacto colmap --help` to see the dataparser options of your version. The defaults
`orientation_method=up`, `center_method=poses` and `auto_scale_poses=True` re-orient and re-scale the rawatta
frame-0 coordinates, which is what you want here.

## 2c. Brush

[Brush](https://github.com/ArthurBrussee/brush) (Apache-2.0) trains splats on the GPU through WebGPU and loads
COLMAP and nerfstudio datasets. Open the `splat/<scene>` folder (or a zip of it) in the Brush app, or see
`brush --help` for the command line. If your Brush version does not accept the text model, convert it to binary
with COLMAP:

```bash
colmap model_converter --input_path splat/buddha/sparse/0 --output_path splat/buddha/sparse/0 --output_type BIN
```

## Notes

- Poses from a feed-forward model are good starting points, not bundle-adjusted ones. Tools that refine camera
  poses during training (for example gsplat's `--pose_opt`) can help on large scenes.
- The example photos have their own licences (see `examples/<scene>/SOURCE.md`). Splats trained on them are
  derived works: keep the attribution (CC BY 4.0 for the Buddha set).
- The cameras and points are outputs of the VGGT-Ω weights (FAIR Noncommercial Research License): splats trained
  from them are for noncommercial research only.
