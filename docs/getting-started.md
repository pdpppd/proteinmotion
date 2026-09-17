# Get started

ProteinMotion requires Python 3.11 or later and a GPU. On macOS, it uses Metal for rendering and VideoToolbox for video encoding. The examples and rendering tests use Apple silicon Macs.

## Install the release

Create a virtual environment and install v0.9.1 from [GitHub Releases](https://github.com/pdpppd/proteinmotion/releases/tag/v0.9.1):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install \
  https://github.com/pdpppd/proteinmotion/releases/download/v0.9.1/proteinmotion-0.9.1-py3-none-any.whl
proteinmotion --version
proteinmotion doctor
```

The package includes shaders, fonts, licenses, a sample structure, a starter scene, and the AI agent skill. PyAV supplies the FFmpeg libraries used for export.

On Apple silicon, use native arm64 Python. `doctor` reports the GPU backend and available encoders; look for `Metal` and `h264_videotoolbox`.

## Create and render a video

```bash output=starter
proteinmotion init my-movie
proteinmotion render my-movie/film.py ProteinMovie \
  --fps 60 --width 1920 --height 1080 -o my-movie/film.mp4
```

`init` creates `film.py` and the **1UBQ ubiquitin** structure in your video folder. The scene rotates the cartoon, colors and labels its helix, zooms into that region, switches to ball-and-stick, and returns to a ribbon view. The script resolves input paths relative to its own directory. Choose an empty folder; the command stops if either file already exists.

[Read the complete starter script](https://github.com/pdpppd/proteinmotion/blob/main/skills/proteinmotion-movies/assets/film.py). Change the input structure and selections together when making a movie of a different protein.

## Use with an AI agent

The ProteinMotion Movies skill provides instructions and examples for agents that can read local files and run Python. Install it in your agent's skills directory:

```bash
proteinmotion install-skill --path /path/to/skills/proteinmotion-movies
```

You can also point an agent directly to the installed `SKILL.md`. See the [AI agent guide](agent-skill.md) for setup and example requests, including the default Codex installation.

## Optional dependencies

Install extras from the same release URL:

```bash
python -m pip install \
  "proteinmotion[preview,md] @ https://github.com/pdpppd/proteinmotion/releases/download/v0.9.1/proteinmotion-0.9.1-py3-none-any.whl"
```

| Extra | Purpose |
|---|---|
| `preview` | Interactive native window |
| `md` | Read XTC, DCD, TRR, and related formats with MDAnalysis |
| `dev` | Tests, linting, package builds |
| `plots` | Contact-map report figures |

The base package includes PDB/mmCIF reading, state playback, surfaces, annotations, matching, interactions, and video export.

## Preview, stills and export

```bash
proteinmotion still my-movie/film.py ProteinMovie --time 5 -o my-movie/frame.png
proteinmotion preview my-movie/film.py ProteinMovie
proteinmotion render my-movie/film.py ProteinMovie --fps 60 --bitrate 20M -o my-movie/film.mp4
```

Preview requires the `preview` extra: drag to orbit; wheel to zoom; Space to play/pause; Left/Right to seek; Home to rewind; R to reset the preview camera; Escape to close. Preview camera adjustments apply to the preview window only.

`scene.seek(t)` restores the scene at time `t`, including when seeking backward. `scene.render_frame(t)` returns an RGBA array; `output='frame.png'` saves it. Reuse a `Renderer` for many stills. MP4 export selects `h264_videotoolbox` on macOS when available, with `allow_sw=0` to require hardware encoding. `--codec libx264` explicitly chooses CPU encoding; molecular rendering still requires a GPU. `hevc_videotoolbox` is also supported. Export replaces the requested output only after encoding succeeds.

The CLI is also available as `python -m proteinmotion`.

## Source installation and repository examples

For package development or the larger included examples:

```bash
git clone https://github.com/pdpppd/proteinmotion.git
cd proteinmotion
python -m pip install -e '.[dev,md,preview]'
proteinmotion render examples/feature_showcase.py FeatureShowcase --fps 60 -o showcase.mp4
```

See the [feature demo](showcase.md) or [NMR example](full-example.md). The repository contains the input structures for these examples. The release package includes the ubiquitin starter structure.

The [reference manual](https://pdpppd.github.io/proteinmotion/reference/) lists classes, functions, parameters, and methods by module.

## Optional Blender EEVEE renderer

Install [Blender 4.5 or later](https://www.blender.org/download/) for EEVEE depth of field. EEVEE is bundled with Blender; a separate Python `bpy` installation is unnecessary. The native renderer remains the default.

```bash
proteinmotion render my-movie/film.py ProteinMovie --renderer eevee --fps 60 -o film.mp4
```

Use `--blender /path/to/blender` for a custom installation. See [EEVEE and lens focus](eevee.md) for residue selections, animated focus pulls, and a complete example.
