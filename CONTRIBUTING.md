# Contributing

Include a reproducible example with each feature or fix. Describe the behavior, the method used, and any limits that affect the result.

## Package development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,md,preview]'
ruff format src tests examples scripts skills
ruff check src tests examples scripts skills
pytest
python -m build
```

GPU and hardware-export tests are marked `gpu` and should be run on a native GPU adapter. Some optional example-rendering scripts need `ffmpeg` and `ffprobe` (`brew install ffmpeg` on macOS). CPU checks can run with `pytest -m "not gpu"`.

GitHub Actions runs CPU tests and linting on Ubuntu and Windows, then builds and installs the wheel in a clean environment. For rendering or animation changes, run the local GPU suite on Apple silicon or Windows/NVIDIA and inspect the output. Video checks decode frames with PyAV and do not require an FFmpeg executable. Windows symlink tests skip when Developer Mode or administrator privileges are unavailable. Record the setup and results in the validation report.

## Documentation development

The website uses Next.js static export, React, and Tailwind CSS. Website development requires Node 22 or later. The Python package has its own dependencies.

```bash
cd website
npm ci
npm run dev
```

Open `http://localhost:3000/proteinmotion/`. Edit Markdown in `docs/` or the site components in `website/`. The complete NMR source is inserted directly from `examples/nmr_regions.py` at build time, so it stays in sync.

Before submitting:

```bash
npm run build
npm run check
npm run verify
```

The build exports static HTML into `website/out/`. `PAGES_BASE_PATH` defaults to `/proteinmotion`; set it to a different repository path (or an empty string for root hosting) when building for another deployment. Keep site metadata in `website/lib/config.ts` consistent with the final public URL.

`npm run verify` checks exported pages, internal links, fragment targets, and media references. Browser checks should cover desktop/mobile layout, search, code copying, video controls, and reduced motion.

Follow the [writing guide](docs/writing-guide.md) when editing page text, headings, captions, and metadata.

### Rendered documentation examples

Short examples are defined in `examples/docs_examples.py`. Their marked code sections appear beside the rendered output in the guides. To update the clips and snippets, run:

```bash
python scripts/render_docs_examples.py
```

This requires a working GPU. The command renders 720p/60 fps clips, checks every encoded frame, saves posters, and updates the matching Markdown snippets. Run it before building the website. The build checks source, input, and media hashes to catch outdated previews.

Use `output=<example-id>` after the language in a Markdown code fence to pair a snippet with its clip. Existing gallery videos use `output=gallery-<demo-id>` beside their render commands.

## Publishing

The Pages workflow builds on pull requests and pushes. Only `main` pushes or a manual workflow run from `main` deploy. GitHub Pages must use **GitHub Actions** as its source. The workflow uploads the static export and deploys it with the official Pages actions. GitHub Pages serves the exported static files.

Compressed web previews are committed in `website/public/media/`. Calmodulin in focus retains 1080p/60 fps. Other previews use 720p, with the continuous feature tour at 60 fps and shorter examples at 30 fps. Their source filenames and sizes are recorded in `provenance.json`; the scientific data provenance lives in the documentation. Original rendered videos, caches, local environments, and build outputs are ignored.

## Changes and issues

Open an issue with a minimal scene, input format, Python version, and the output of `proteinmotion doctor`. For rendering defects, include the relevant time or frame. Use a public example structure when reporting an issue in the public repository.

A pull request should explain the behavior change and relevant checks. Record the source of example data and distinguish deposited structures, simulated trajectories, and interpolated coordinates.

## Package and skill releases

The AI agent skill source is `skills/proteinmotion-movies`. Its resource files are mapped into the wheel by `pyproject.toml`; keep that mapping in sync when adding a skill resource. Run the skill-creator validator when changing its instructions. The starter scene is shared by the skill and `proteinmotion init`, so maintain it once.

Build with `python -m build` (the wheel is built from the sdist). Install that wheel into a fresh environment, change out of the checkout, and run `python -I /path/to/repo/scripts/check_installed_package.py --render` on a native GPU. CI runs the same installed-resource check without rendering. Distribute the wheel, source archive, skill ZIP and checksums as versioned GitHub Release assets. Publish versioned release assets after the release checks pass.


### PyPI publishing

The `publish.yml` workflow builds the source archive and wheel, validates the PyPI description, and checks the installed wheel on Linux, Windows, and macOS. Linux installation checks cover Python 3.11–3.14. A manual run with `publish` left false performs these checks without uploading.

Configure a pending GitHub publisher in the PyPI account that will own the project:

| Field | Value |
|---|---|
| PyPI project | `proteinmotion` |
| GitHub owner | `pdpppd` |
| Repository | `proteinmotion` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

Create the `pypi` environment in GitHub repository settings and restrict its deployments to release tags. PyPI Trusted Publishing uses GitHub's short-lived identity credentials; it needs no stored PyPI API token. PyPI creates the project on the first successful upload. See the [PyPI setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

For a release:

1. Update `pyproject.toml`, `proteinmotion.__version__`, and the website package version, then run `python scripts/build_reference.py`.
2. Run package checks, `python -m build`, and `python -m twine check --strict dist/*`. Verify an isolated installation and a native GPU render with `scripts/check_installed_package.py --render`.
3. Commit the release and create its matching `v<version>` tag. Publish a GitHub Release for that tag to start the PyPI workflow. The upload job requires the tag to match the package version and waits for all installation checks.
4. Confirm the version on PyPI, then install it with `python -m pip install --index-url https://pypi.org/simple proteinmotion==<version>` in a fresh environment.

A manual publishing run must select the version tag and set `publish` to true. PyPI distribution filenames are permanent: change the version for a corrected release instead of replacing files. Keep previous release artifacts in separate directories when building a new version.

### Reference manual

The manual reads signatures, defaults, inheritance, and source locations from the Python files. Descriptions and example links are maintained in `docs/reference/catalog.json`. After changing either, regenerate the reference data:

```bash
python3 scripts/build_reference.py
```

The generator uses Python’s standard library. The website build checks that the generated data matches the source and that every package export has a reference entry. Describe new parameters and methods in the catalog before building.
