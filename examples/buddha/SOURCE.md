# Buddha head (object-centric example)

34 photos of a Buddha head on a table, taken all around the object.

| | |
|---|---|
| Dataset | Buddha Dataset, version 1.0 (AliceVision) |
| Source | https://github.com/alicevision/dataset_buddha (files taken from commit `9067f679eb5a772c3e818b7a12dc35882c071858`, folder `buddha/`) |
| Archive copy | Zenodo, https://doi.org/10.5281/zenodo.1203283 |
| Authors | Simone Gasparini, Fabien Castan, Yann Lanthony |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ |
| Licence checked | 2026-11-13, on the GitHub repository (README, `LICENSE.md`, GitHub licence field `cc-by-4.0`) and on the Zenodo record |

## Licence text (excerpts from the source pages)

From the repository README, section "License":

> This dataset is released under the Creative Commons Attribution 4.0 license (see [LICENSE.md](https://github.com/alicevision/dataset_buddha/blob/9067f679eb5a772c3e818b7a12dc35882c071858/LICENSE.md); the link in the source README is relative to that repository)

From the repository `LICENSE.md`: the full "Creative Commons Attribution 4.0 International Public License".

From the Zenodo record 1203283: licence "Creative Commons Attribution 4.0 International".

The README asks for this citation:

```
Simone Gasparini, Fabien Castan, & Yann Lanthony. (2017). Buddha Dataset (Version 1.0) [Data set].
Zenodo. http://doi.org/10.5281/zenodo.1203283
```

The README also notes that the dataset was developed during the LADIO project, funded by the European Union's
Horizon 2020 research and innovation programme under grant agreement No 731970.

## What we use and what we changed

- 34 of the 67 images in `buddha/`: every second file, `00001._c.png`, `00003._c.png`, ..., `00067._c.png`.
- `00033._c.png` is moved to the front (`frame_000`) because RAWATTA puts the world frame at frame 0 and this
  view is upright. The other frames follow in file-name order. The exact list, with the SHA-256 of every original,
  is `manifest.tsv`.
- Each image (2736 x 1540 PNG) is downscaled to 1024 x 576 (Lanczos) and saved as JPEG, quality 92.
- The projection matrices and seed files that ship with the dataset are not used.

The images are not stored in this repository. `fetch.sh` downloads the originals, checks their size and
SHA-256, and writes the downscaled copies. `images.sha256` lists the sums of the downscaled files as produced with
Pillow 10.2.0; another resizer gives different (equally valid) bytes.

## Attribution for derived outputs

Point clouds, cameras and thumbnails made from these photos are adaptations of a CC BY 4.0 work. Show them with
the credit, the licence with its URL, and the changes, for example:

> Photos: Buddha Dataset by Simone Gasparini, Fabien Castan and Yann Lanthony (AliceVision),
> doi:10.5281/zenodo.1203283, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); 34 of 67 photos,
> downscaled. Cameras and point cloud predicted by VGGT-Ω, accelerated with RAWATTA.

Give the number of photos you actually used. `make_gallery.sh` writes this line into the gallery caption with
the photo count taken from the image folder (credit, licence and URL come from `scene.json`).
