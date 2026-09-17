# Get started

ProteinMotion is an installable Python package. Python 3.11+ and a native GPU are required; macOS with Apple silicon is the verified platform. The renderer uses Metal there, with VideoToolbox hardware encoding for MP4 export.

## Install the release

Create an environment wherever you want to author movies, then install the versioned wheel from [GitHub Releases](https://github.com/pdpppd/proteinmotion/releases/tag/v0.7.0):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install \
  https://github.com/pdpppd/proteinmotion/releases/download/v0.7.0/proteinmotion-0.7.0-py3-none-any.whl
proteinmotion --version
proteinmotion doctor
```

No clone or editable installation is needed. The wheel includes the shaders, fonts and their licenses, a sample structure, a starter scene, and the optional Codex skill. PyAV includes FFmpeg libraries, so rendering does not require an external FFmpeg executable, browser, Blender, or Xcode build.

On Apple silicon, use native arm64 Python. `doctor` reports the GPU backend and available encoders; look for `Metal` and `h264_videotoolbox`.

## Create and render your first film

```bash
proteinmotion init my-movie
proteinmotion render my-movie/film.py ProteinMovie \
  --fps 60 --width 1920 --height 1080 -o my-movie/film.mp4
```

`init` writes an editable `film.py` and the deposited **1UBQ ubiquitin** structure into your movie folder. The scene rotates the cartoon, colors and labels its helix, eases the camera into that region, switches to ball-and-stick, and returns to a ribbon view. Input paths are resolved relative to the script, so you can render it from another directory. Existing scene or structure files are never overwritten.

[Read the complete starter script](https://github.com/pdpppd/proteinmotion/blob/main/skills/proteinmotion-movies/assets/film.py). Change the input structure and selections together when making a movie of a different protein.

## Make movies with Codex

```bash
proteinmotion install-skill
```

This installs **`$proteinmotion-movies`** into `${CODEX_HOME:-~/.codex}/skills/proteinmotion-movies`. Start a new Codex conversation if it was already open, then ask:

> Use $proteinmotion-movies to make a 20-second movie of my calmodulin structure, with a helix callout, a smooth zoom, and a ball-and-stick close-up.

[Skill usage, installation options and source](codex-skill.md).

## Optional dependencies

Install extras from the same release URL:

```bash
python -m pip install \
  "proteinmotion[preview,md] @ https://github.com/pdpppd/proteinmotion/releases/download/v0.7.0/proteinmotion-0.7.0-py3-none-any.whl"
```

| Extra | Purpose |
|---|---|
| `preview` | Interactive native window |
| `md` | Lazy MDAnalysis readers for XTC, DCD, TRR and related formats |
| `dev` | Tests, linting, package builds |
| `plots` | Contact-map report figures |

Core PDB/mmCIF reading, multi-model state playback, surfaces, annotations, matching, interactions and movie export work without these extras.

## Preview, stills and export

```bash
proteinmotion still my-movie/film.py ProteinMovie --time 5 -o my-movie/frame.png
proteinmotion preview my-movie/film.py ProteinMovie
proteinmotion render my-movie/film.py ProteinMovie --fps 60 --bitrate 20M -o my-movie/film.mp4
```

Preview requires the `preview` extra: drag to orbit; wheel to zoom; Space to play/pause; Left/Right to seek; Home to rewind; R to reset the preview camera; Escape to close. Camera overrides in preview do not change the authored film.

`scene.seek(t)` is deterministic, including backward seeks. `scene.render_frame(t)` returns an RGBA array; `output='frame.png'` saves it. Reuse a `Renderer` for many stills. MP4 export selects `h264_videotoolbox` on macOS when available, with `allow_sw=0` to require hardware encoding. `--codec libx264` explicitly chooses CPU encoding; molecular rendering still requires a GPU. `hevc_videotoolbox` is also supported. Export replaces the requested output only after encoding succeeds.

The CLI is also available as `python -m proteinmotion`.

## Source installation and repository examples

For package development or the larger included examples:

```bash
git clone https://github.com/pdpppd/proteinmotion.git
cd proteinmotion
python -m pip install -e '.[dev,md,preview]'
proteinmotion render examples/feature_showcase.py FeatureShowcase --fps 60 -o showcase.mp4
```

See [the continuous feature tour](showcase.md) or [the full NMR script](full-example.md). Additional example structures live in the repository; the wheel bundles the small starter input only.
