# Launch checklist

`sezer-muhammed/rawatta` is already **public**: the owner decided on 2026-11-13 to keep rawatta open
source and public from the start. Everything pushed to it is visible at once, so the pre-flight
checks in step 1 apply to every push to `main`, not only to the launch.

What is still to do for the launch, in order: set the arXiv ID, turn on GitHub Pages from `main:/docs`,
tag `v0.1.0` (which publishes the Docker image), make the GHCR package public, and check the result.
None of these has been done by the build: each one is a decision for the owner. Until then the
arXiv buttons say "coming soon", and the docs describe the image `ghcr.io/sezer-muhammed/rawatta` as
"published at v0.1.0; until then build it with `docker build -f docker/Dockerfile -t rawatta .`".

Run the commands from an up-to-date checkout of `main`, with `gh` logged in as the repository owner.
Push only `main` (and the release tag): never the `ws/*` work branches or any local backup branch, whose
history names the private research codebase.

```sh
REPO=sezer-muhammed/rawatta
ARXIV_ID=YYMM.NNNNN        # replace with the real identifier once arXiv has announced the paper
VERSION=0.1.0              # must equal `version` in Cargo.toml ([workspace.package])
```

## 0. Repository settings (owner, once, before the first push of the rewritten `main`)

- [ ] Private vulnerability reporting, which `SECURITY.md` points reporters to:
  `gh api -X PUT repos/$REPO/private-vulnerability-reporting`
  (check: `gh api repos/$REPO/private-vulnerability-reporting` prints `{"enabled":true}`).
- [ ] Secret scanning and push protection (free for public repositories): Settings → Code security, or
  ```sh
  gh api -X PATCH repos/$REPO --input - <<'JSON'
  {"security_and_analysis": {"secret_scanning": {"status": "enabled"},
                             "secret_scanning_push_protection": {"status": "enabled"}}}
  JSON
  ```
- [ ] Author e-mail: commits use `74321576+sezer-muhammed@users.noreply.github.com`
  (`git config user.email` in this checkout); turn on "Block command line pushes that expose my email" in
  the GitHub e-mail settings.
- [ ] After the first push: a ruleset for `main` (no force push, no deletion) and one for `v*` tags (only the
  owner may create them), under Settings → Rules. Optionally require SHA-pinned actions under Settings →
  Actions (every workflow already pins its actions by SHA).

## 1. Pre-flight (before every push to `main`, and once more before the tag)

- [ ] CI checks pass locally (the same commands `.github/workflows/ci.yml` runs):
  ```sh
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets --locked -- -D warnings
  cargo test --workspace --locked
  scripts/ci/python.sh          # needs numpy, pytest; torch==2.9.1 (CPU), opencv-python-headless, Pillow for all tests
  scripts/ci/shellcheck.sh      # ShellCheck 0.11.0 (pip install shellcheck-py==0.11.0.1)
  scripts/ci/check_yaml.py      # PyYAML: every YAML file under .github/ (issue forms too) and CITATION.cff
  python3 scripts/third_party_licenses.py --check    # THIRD-PARTY-LICENSES.md matches Cargo.lock (needs cargo fetch)
  docker run --rm -v "$PWD":/repo -w /repo rhysd/actionlint:1.7.12 .github/workflows/ci.yml .github/workflows/docker.yml
  ```
- [ ] Weight-free GPU tests pass on a GPU machine (tier 1 in `CONTRIBUTING.md`):
  ```sh
  cargo test --release --locked -p rawatta-nn -- --ignored
  cargo test --release --locked -p vggt-omega ptx_tests -- --include-ignored
  ```
  Once the weights are available, also the weights-only tests (tier 2: `gpu_weights_upload_and_views`,
  `gpu_fp8_weights_load`, `gpu_engine_load_drop_releases_pinned` with `RAWATTA_WEIGHTS`). The oracle and
  dataset parity suite (tier 3) needs inputs that do not ship with the repository and stays pending until
  they exist; do not tick it from a partial run.
- [ ] The image builds and the documented commands work with it (Docker + NVIDIA driver on the host):
  ```sh
  docker build -f docker/Dockerfile -t rawatta .
  docker run --rm --gpus all --user "$(id -u):$(id -g)" rawatta info     # NVRTC: ok, cuBLASLt: ok
  mkdir -p ~/.cache/rawatta
  docker run --rm --user "$(id -u):$(id -g)" -v ~/.cache/huggingface:/hf:ro -v ~/.cache/rawatta:/weights \
    --entrypoint download-weights rawatta --token-file /hf/token --dir /weights
  docker run --rm --gpus all --user "$(id -u):$(id -g)" -v ~/.cache/rawatta:/weights:ro -v "$PWD":/work \
    rawatta run --images photos --config rawatta --out out
  ```
- [ ] Nothing private in what you push, **history included** (every commit pushed to `main` is public,
  not only `HEAD`). In particular, nothing may name the authors' private research codebase, its
  branches, commit ids or URL; describe the code as "extracted from the authors' research codebase".
  ```sh
  # Keep the patterns (home folders, user names, machine names, private repository and branch names,
  # commit ids, the private draft labels, the personal e-mail) in a file outside the repository.
  git log origin/main..HEAD -p | grep -nE -f ~/rawatta-private-patterns.txt
  git log origin/main..HEAD --format='%an <%ae> %cn <%ce>' | sort -u   # only the noreply address
  git log origin/main..HEAD --name-only --format= | sort -u | grep -iE '\.(pt|pth|safetensors|npz|tex)$|PERFORMANCE_LEDGER'
  gitleaks detect --source . --log-opts="origin/main..HEAD"    # or any other secret scanner
  ```
  Each command should print nothing. Once something private is pushed, deleting it in a later commit
  does not remove it: it stays in the public history.
- [ ] Every number in `README.md`, `REPRODUCE.md` and `docs/index.html` is a paper number (with its table)
  or a number measured in this build (with the GPU and protocol). Speed claims carry the split: of the
  2.08x at 50 frames on the RTX 4070 SUPER, 1.61x is the CUDA engine and 1.29x RAWATTA itself; at 200
  frames RAWATTA alone gives 1.66x (RAWATTA-Fast 1.97x) over the exact engine. Ampere support for the
  exact engine says "built and bit-checked for sm_80+, not yet run on Ampere hardware"; Hopper and sm_100
  for RAWATTA say "built and ptxas-checked but not yet run". The authors keep the hand-checked audit of the
  site's numbers with the paper sources, outside this repository.
- [ ] Licence files and headers are in place: `LICENSE` (Apache-2.0), `LICENSE-VGGT-OMEGA` (FAIR
  Noncommercial Research License, for `crates/vggt-omega/` and the weights), `LICENSE-DINOV3` (the DINOv3
  backbone code in `crates/vggt-omega/src/trunk/`), `THIRD-PARTY-LICENSES.md`, `NOTICE`, `CITATION.cff`.

## 1b. Amend the paper before it is posted (owner)

The paper and this release must not disagree once both are public:

- [ ] The two sentences that promise per-seed frame maps of every evaluation sequence (supplementary material:
  the data-split paragraph and the reproducibility checklist) become: "Frame maps for ETH3D, CO3Dv2, NRGBD and TUM-dynamic are
  released with the code; the RE10K and 7-Scenes evaluation subsets were not kept."
- [ ] Sec. 4.2, "Merge and frame operators use the layer's format", becomes: "Frame layers run on the bf16
  kernel and merged layers on the 8-bit kernel." (as `configs/GRAMMAR.md` documents).
- [ ] Optionally, the Tab. 1 caption: "against a worst-dataset loss of 1.20/1.03 for RAWATTA-Fast".
- [ ] Acknowledge the use of DINOv3 materials (DINOv3 License s.1.b.ii), as the README does.

## 2. Set the arXiv ID

- [ ] `docs/static/js/main.js`: set `const ARXIV_ID = "";` to the identifier. This turns on the Paper and PDF
  buttons and fills the BibTeX on the page.
- [ ] Replace the placeholder `XXXX.XXXXX` where it is shown to readers: the BibTeX block in
  `docs/index.html`, `README.md`, `CITATION.cff` and the version pin in `REPRODUCE.md` ("arXiv XXXX.XXXXX,
  version 1", which pins the expected values to `arXiv:$ARXIV_ID v1`). Only these four files: the other
  hits must stay as they are (see below).
  ```sh
  F="docs/index.html README.md CITATION.cff REPRODUCE.md"
  git grep -n 'XXXX\.XXXXX' -- $F
  git grep -l 'XXXX\.XXXXX' -- $F | xargs -r perl -pi -e "s/XXXX\.XXXXX/$ARXIV_ID/g"
  git grep -n 'XXXX\.XXXXX' -- $F    # prints nothing now
  git grep -n 'XXXX\.XXXXX'          # only the expected hits below
  ```
  The hits that stay: the `replace("XXXX.XXXXX", ARXIV_ID)` call in `docs/static/js/main.js` (it fills
  the BibTeX for readers with JavaScript and does nothing once the placeholder is gone),
  `site-src/dev/check.cjs` (its no-arXiv check expects the placeholder), `site-src/README.md` and this
  checklist. Review any other hit by hand.
- [ ] For readers without JavaScript, give the two `data-arxiv` buttons in `docs/index.html` real links
  (`href="https://arxiv.org/abs/$ARXIV_ID"` and `href="https://arxiv.org/pdf/$ARXIV_ID"`), and remove
  `aria-disabled="true"`, the "will appear here" `title` and the `coming soon` span.
- [ ] Replace every other "coming soon" arXiv mention in `README.md` (and `REPRODUCE.md`, if any) with the link.
- [ ] `CITATION.cff`: add the arXiv DOI (`10.48550/arXiv.$ARXIV_ID`) under `identifiers` and the arXiv URL.
- [ ] Run the pre-flight checks of step 1, commit ("Set the arXiv ID") and push to `main`. Wait for CI:
  `gh run list --repo $REPO --workflow ci.yml --limit 1`.

## 3. Enable GitHub Pages from `main` `/docs`

There is no Pages workflow: GitHub serves `docs/` from `main` directly (`docs/.nojekyll` turns off Jekyll).
This step does not depend on the arXiv ID and can be done right after the first push of `main` (the README
and `CITATION.cff` already link the page). Afterwards drop "(live at launch)" from the README.

```sh
gh api --method POST repos/$REPO/pages -f 'source[branch]=main' -f 'source[path]=/docs'
# If Pages already exists, change its source instead:
#   gh api --method PUT repos/$REPO/pages -f 'source[branch]=main' -f 'source[path]=/docs'
gh api repos/$REPO/pages --jq '.html_url, .source.branch, .source.path, .status'
gh api repos/$REPO/pages/builds/latest --jq '.status'            # "built" after a minute or two
gh api --method PUT repos/$REPO/pages -F https_enforced=true     # once the certificate is issued
gh repo edit $REPO --homepage "https://sezer-muhammed.github.io/rawatta/"
```

## 4. Tag `v0.1.0` to publish the Docker image

`.github/workflows/docker.yml` runs on `v*` tags. It checks that the tag equals `v` + the Cargo version,
builds `docker/Dockerfile`, smoke-tests the image without a GPU (also under an arbitrary `--user`), and
pushes `ghcr.io/sezer-muhammed/rawatta:v0.1.0` with the workflow's `GITHUB_TOKEN`. A release version
(no pre-release part) also moves `:latest`; a tag such as `v0.2.0-rc1` pushes only its own tag. The
image's `org.opencontainers.image.version` label is the version without the `v` (`0.1.0`).

```sh
git switch main && git pull --ff-only
grep -m1 '^version' Cargo.toml                                    # version = "0.1.0"
git tag -a "v$VERSION" -m "rawatta v$VERSION"
git push origin "v$VERSION"
gh run watch --repo $REPO "$(gh run list --repo $REPO --workflow docker.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

Optional: a GitHub release for the tag (`gh release create "v$VERSION" --repo $REPO --title "rawatta v$VERSION" --notes-file <notes>`).

If the push fails with `denied: permission_denied`, a package with that name already exists without
access for this repository: on the package settings page (step 5), under "Manage Actions access", add
`sezer-muhammed/rawatta` with the Write role, then re-run the workflow (`gh run rerun <id> --repo $REPO`).

## 5. Make the GHCR package public

The first push creates the package `ghcr.io/sezer-muhammed/rawatta` as **private**. The REST API has no
call to change a package's visibility, so use the web page:

- [ ] https://github.com/users/sezer-muhammed/packages/container/rawatta/settings → Danger Zone →
  Change visibility → Public.
- [ ] The package page shows it is linked to `sezer-muhammed/rawatta` (from the image label
  `org.opencontainers.image.source`).
- [ ] Check it from the API: `gh api users/sezer-muhammed/packages/container/rawatta --jq .visibility` prints `public`.

## 6. Verify

- [ ] Signed out of GHCR, the published image runs the documented commands:
  ```sh
  docker logout ghcr.io
  docker pull ghcr.io/sezer-muhammed/rawatta:latest
  docker run --rm --gpus all --user "$(id -u):$(id -g)" ghcr.io/sezer-muhammed/rawatta:latest info
  docker run --rm --user "$(id -u):$(id -g)" -v ~/.cache/huggingface:/hf:ro -v ~/.cache/rawatta:/weights \
    --entrypoint download-weights ghcr.io/sezer-muhammed/rawatta:latest --token-file /hf/token --dir /weights
  docker run --rm --gpus all --user "$(id -u):$(id -g)" -v ~/.cache/rawatta:/weights:ro -v "$PWD":/work \
    ghcr.io/sezer-muhammed/rawatta:latest run --images photos --config rawatta --out out
  ```
- [ ] Open https://sezer-muhammed.github.io/rawatta/: figures and the example viewer load, the arXiv
  buttons open the abstract and the PDF, the BibTeX shows the identifier, the browser console shows no
  errors.
- [ ] Update the docs that describe the image as "published at v0.1.0; until then build it ..." so
  they simply say `docker pull ghcr.io/sezer-muhammed/rawatta:latest`, and drop the "pulling it before then
  fails" clause (`README.md`, `docker/README.md`, `docs/index.html`, `examples/README.md`, `tools/README.md`).
- [ ] CI passes on `main`: `gh run list --repo $REPO --workflow ci.yml --limit 3`.
- [ ] The README's links (site, image, arXiv, licences) work from the public page.
- [ ] Announce only numbers that appear in the README or the paper, with the speed split of step 1.
