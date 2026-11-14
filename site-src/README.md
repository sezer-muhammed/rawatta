# site-src: sources of the project page

The project page is the static site in [`../docs/`](../docs/) (GitHub Pages, served from `docs/` under `/rawatta/`).
It has no build step and no framework: `docs/index.html`, `docs/static/css/style.css`, `docs/static/js/{main,formats,viewer}.js`,
figures in `docs/static/img/`. All URLs the page loads are relative, so it works under any sub-path. The only absolute
URLs are the OpenGraph/Twitter meta tags and `<link rel="canonical">`, because link previews require them.

This folder holds what is needed to regenerate the figures and to check the page.

| Path | What it is |
|---|---|
| `make_figs.py` | The paper's figure script adapted to write `docs/static/img/{teaser,anatomy,scaling}.{svg,png}` and the social preview `og.png` (1200x630). The only layout change is a two-row legend in the scaling figure (the one-row legend is clipped in the paper PDF). |
| `data/` | The values the figures plot, extracted from the paper's logged runs and rounded from the paper's sources: latencies to 0.1 ms, ΔAUC to 1e-4, attention statistics to 1e-6 (see the `note` fields; `scaling.csv` holds the logged medians at 0.1 ms). The rounding does not change the figures visibly. |
| `requirements.txt` | Pinned Python packages for `make_figs.py` (matplotlib 3.11.2, the version used for the paper figures). |
| `dev/figs.sh` | Runs `make_figs.py` in a venv at `.venv` (any shell with python3 3.12; the script header shows a Docker command). |
| `dev/check.sh`, `dev/check.cjs`, `dev/run-checks.sh` | Page checks: the format unit tests below, then serve `docs/` with `python -m http.server` under `/rawatta/`, drive it with Playwright at 1280, 390 and 360 px in light and dark mode, fail on console errors, failed requests, broken images or internal links, horizontal scroll, axe-core WCAG 2.1 A/AA violations, a quick start that does not show the canonical commands verbatim, or a viewer that does not load the synthetic scene with its cameras and licence link; save screenshots. |
| `dev/test_formats.mjs` | Node unit tests of `docs/static/js/formats.js` (the viewer's readers for `index.json`, `cameras.json` and `points.ply`) against the fixtures. |
| `dev/fixtures/` | `cameras.rawatta-run.json`: printed by the serializer `rawatta run` uses for `cameras.json`; `gallery-index.json`: written by `examples/python/gallery_index.py`. `dev/make_fixtures.sh` says how both were made and regenerates the second. |
| `dev/make_synthetic_scene.py` | Writes a tiny synthetic scene in the same formats for testing the 3D viewer. It is only used in the check's dev copy of the site and is never shipped. |
| `dev/audit_numbers.py` | Lists every number in the page text and finds it in the paper's LaTeX sources (pass the source directory; the paper sources are not in this repository). A found number only shows that it occurs somewhere in the sources; the authors keep the hand-checked mapping of every page number to the paper's tables and sections with the paper sources. |

Every script prints its usage with `--help`.

## Regenerate the figures

```sh
python3 -m venv .venv && .venv/bin/pip install -r site-src/requirements.txt
.venv/bin/python -I site-src/make_figs.py          # writes docs/static/img/
```

## Check the page

```sh
site-src/dev/run-checks.sh                                # local Docker; screenshots in site-src/dev/out/shots/
RAWATTA_CHECK_HOST=<ssh host> site-src/dev/run-checks.sh  # or on a remote Docker host (the tree is rsynced there)
```

Both run `site-src/dev/check.sh` in `mcr.microsoft.com/playwright:v1.63.0-noble` with the repository mounted at `/w`.
It needs network access once, to install `playwright@1.63.0` and `axe-core@4.10.2` into `.venv/pw/`, and the dev copy
loads three.js from cdn.jsdelivr.net.

## Flip the arXiv link

Set `ARXIV_ID` at the top of `docs/static/js/main.js` (for example `"2611.01234"`). The Paper and PDF buttons become
links to `arxiv.org/abs/<id>` and `arxiv.org/pdf/<id>`, and the BibTeX block shows the identifier. Also update the
`XXXX.XXXXX` placeholder in the BibTeX text of `docs/index.html` so that readers without JavaScript see it too.

## Add interactive examples

The "Interactive example" section stays hidden while `docs/static/examples/index.json` is missing or lists no scenes.
three.js (`three@0.170.0`, pinned in the import map of `index.html`) is only fetched when the section scrolls into view.

`examples/make_gallery.sh` (see `examples/README.md`) runs `rawatta run` on the example scenes and writes everything the
viewer reads into `docs/static/examples/`:

- `index.json`, written by `examples/python/gallery_index.py`:
  `{"scenes": [{"id", "title", "dir", "points", "cameras", "thumb", "caption", "frames", "config", "latency_ms",
  "latency_iqr_ms", "reps", "warmup", "gpu", "point_count", "licence", "licence_url", "source", "author", "attribution",
  "changes", "kind"}]}`. The page reads `id`, `title`, `dir` (relative to `index.json`), `points`, `cameras`, `caption`,
  `licence`, `licence_url` and `source`: the caption text is shown under the viewer, followed by a link to the photo
  licence (`licence_url`, labelled with `licence`) and to the photo source.
- `<dir>/points.ply`: the run's point cloud, subsampled; binary little-endian PLY with `x y z` (float) and
  `red green blue` (uchar).
- `<dir>/cameras.json`: the run's cameras, unchanged, as `rawatta run` writes them
  (`crates/vggt-omega/src/output.rs`): `{"convention", "width", "height", "frames": [{"image", "w2c", "intrinsics",
  "fov_y", "fov_x", "pose_enc"}]}` with `w2c` the world-to-camera 3x4 (OpenCV axes, world = frame 0) and `intrinsics`
  the 3x3 matrix in pixels of the `width` x `height` network input. Frame 0 is drawn in orange.

Keep each `points.ply` small (`make_gallery.sh` caps it at 300k points and 8 MB) because GitHub Pages serves it to every
visitor.

## Licences

Page text, figures and figure data: CC BY 4.0, except `docs/static/examples/`: the point clouds and cameras there are
outputs of the VGGT-Ω weights (FAIR Noncommercial Research License, noncommercial research use only); photos and
thumbnails keep their own licences (`examples/*/SOURCE.md`). Scripts in this folder: Apache-2.0.
