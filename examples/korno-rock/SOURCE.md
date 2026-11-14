# Korno climbing rock (outdoor example)

32 photos of a climbing rock face, taken with a DJI drone camera (FC9670), from wide views of the whole
face to close views along its base.

| | |
|---|---|
| Collection | "Photogrammetry of Korno climbing rock" on Wikimedia Commons ("Photogrammetry set of raw photos") |
| Source | https://commons.wikimedia.org/wiki/Category:Photogrammetry_of_Korno_climbing_rock |
| Example file page | https://commons.wikimedia.org/wiki/File:Photogrammetry_dataset_of_Korno_climbing_rock_01.jpg |
| Author | Zbytovsky (Wikimedia Commons user), "Own work", photos taken 11 August 2026 |
| Licence | Creative Commons CC0 1.0 Universal Public Domain Dedication, https://creativecommons.org/publicdomain/zero/1.0/ |
| Licence checked | 2026-11-13, on each file page through the Commons API (`LicenseShortName` = `CC0` for all 32 files used) and on the file page of photo 01 |

## Licence text (excerpt from the file pages)

> This file is made available under the Creative Commons CC0 1.0 Universal Public Domain Dedication.

The category holds 97 files: 96 CC0 photos by Zbytovsky and one CC BY-SA 4.0 photo by another author
("Tomáškův lom - Korno.jpg"). That CC BY-SA photo is **not** used.

CC0 needs no attribution, but we credit the author anyway.

## What we use and what we changed

- The first 32 of the 96 CC0 photos in capture order (EXIF `DateTimeOriginal`, 21:10 to 21:27). In file-name terms
  these are photos 01-22, 97, 98 and 25-32 (photos 97 and 98 were taken between 22 and 25). They are renamed
  `frame_000` ... `frame_031` in that order. `manifest.tsv` lists every file with its size, SHA-256 and URL; the
  SHA-1 values published by Commons were checked when the manifest was made.
- Each image (8000 x 4500 JPEG, about 30 MB) is downscaled to 1024 x 576 (Lanczos) and saved as JPEG, quality 92.

The images are not stored in this repository (the originals are about 1 GB). `fetch.sh` downloads them, checks
their size and SHA-256 and writes the downscaled copies; it deletes the originals afterwards unless
`KEEP_ORIGINALS=1`. `images.sha256` lists the sums of the downscaled files as produced with Pillow 10.2.0.

## Credit line for derived outputs

> Photos: Photogrammetry of Korno climbing rock by Zbytovsky, Wikimedia Commons, CC0 1.0
> (https://creativecommons.org/publicdomain/zero/1.0/); 32 of 96 photos, downscaled. Cameras and point cloud
> predicted by VGGT-Ω, accelerated with RAWATTA.

Give the number of photos you actually used. `make_gallery.sh` writes this line into the gallery caption with
the photo count taken from the image folder (credit, licence and URL come from `scene.json`).
