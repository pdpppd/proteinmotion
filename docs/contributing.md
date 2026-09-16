# Contributing

ProteinMotion is an early implementation. Keep new features small, reproducible, and honest about scientific and rendering limits.

## Package development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,md,preview]'
ruff format src tests examples
ruff check src tests examples
pytest
python -m build
```

GPU and hardware-export tests are marked `gpu` and should be run on a real native adapter. Optional video checks need `ffmpeg` and `ffprobe` (`brew install ffmpeg` on macOS). CPU checks can run with `pytest -m "not gpu"`.

The hosted GitHub checks run CPU tests and linting on Ubuntu, then build the package. They do not replace the local Apple silicon GPU checks recorded in the validation report. When changing rendering or animation, run the complete local suite and inspect actual output.

## Documentation development

The website uses Next.js static export, React, and Tailwind CSS. It is a separate development tool; Node is not a dependency of the Python renderer. Use Node 22 or newer.

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

## Publishing

The Pages workflow builds on pull requests and pushes. Only `main` pushes or a manual workflow run from `main` deploy. GitHub Pages must use **GitHub Actions** as its source. The workflow uploads the static export and deploys it with the official Pages actions. No server or API keys are required by the site.

Compressed web previews are committed in `website/public/media/`. They are 720p/30 fps versions of the original 1080p/60 fps films. Their source filenames and sizes are recorded in `provenance.json`; the scientific data provenance lives in the documentation. Original rendered videos, caches, local environments, and build outputs are ignored.

## Changes and issues

Open an issue with a minimal scene, input format, Python version, and the output of `proteinmotion doctor`. For rendering defects, include the relevant time or frame. Avoid uploading private or unpublished structures unless you intend to share them publicly.

A pull request should explain the concrete behavior change and relevant validation. Keep source attribution for new example data. Do not label interpolated coordinates as measured dynamics or physically valid transitions without supporting methodology.
