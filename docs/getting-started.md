# Get started

ProteinMotion requires Python 3.11 or later and a GPU. Windows NVIDIA systems use Vulkan and NVENC. macOS uses Metal and VideoToolbox. Rendering tests cover Apple silicon and an NVIDIA RTX 5070 Ti on Windows 11.

The same scene scripts work on both platforms. Each render automatically detects the operating system and available GPU, then selects the backend and video encoder. The render command prints the detected platform, GPU/backend and encoder before exporting. Platform flags are optional overrides.

## Install the release

ProteinMotion is available on [PyPI](https://pypi.org/project/proteinmotion/). Install it into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install proteinmotion
proteinmotion --version
proteinmotion doctor --check-encoders
```

### Install the command globally

Use [pipx](https://pipx.pypa.io/stable/installation/) to make the command available from any directory:

```bash
pipx install proteinmotion
pipx ensurepath
```

Open a new terminal after `ensurepath`, then run `proteinmotion init my-movie`. pipx keeps the package and its dependencies in an isolated environment. To import ProteinMotion from your own Python process or notebook, install it with pip in that environment.

The package includes shaders, fonts, licenses, a sample structure, a starter scene, and the AI agent skill. PyAV supplies the FFmpeg libraries used for export.

On Apple silicon, use native arm64 Python. `doctor` reports the GPU backend and available encoders; look for `Metal` and `h264_videotoolbox`.

### Update an installation

For a pip installation, activate its Python environment and run:

```bash
python -m pip install --upgrade proteinmotion
proteinmotion --version
```

For the global command, run `pipx upgrade proteinmotion`. See the [release notes](https://github.com/pdpppd/proteinmotion/releases) for changes in each version.

## Windows and NVIDIA GPUs

Use 64-bit Python 3.11 or later and a current NVIDIA driver. In PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "proteinmotion[preview]"
.\.venv\Scripts\python.exe -m proteinmotion doctor --check-encoders
.\.venv\Scripts\python.exe -m proteinmotion init my-movie
.\.venv\Scripts\python.exe -m proteinmotion render my-movie/film.py ProteinMovie --fps 60 -o first-film.mp4
```

These commands use the virtual environment directly and do not require changing PowerShell's execution policy. The package supplies wgpu-native and PyAV's FFmpeg libraries. CUDA Toolkit and the `ffmpeg` command are not required.

`doctor` lists compiled encoders; `--check-encoders` also tests whether they can encode a small frame. Check that `selected_adapter` identifies your NVIDIA GPU with `backend_type: Vulkan`, and that `h264_nvenc` is usable. Automatic selection prefers a discrete GPU over integrated graphics. To select a GPU explicitly:

```powershell
.\.venv\Scripts\python.exe -m proteinmotion render my-movie/film.py ProteinMovie --gpu-backend Vulkan --gpu-adapter NVIDIA --codec h264_nvenc --fps 60 -o first-film.mp4
```

`--codec auto` prefers H.264 NVENC on Windows/Linux and VideoToolbox on macOS. If hardware encoder initialization fails, it warns and uses `libx264`. An explicit `--codec h264_nvenc` reports a failure instead of switching encoders. `hevc_nvenc` and `av1_nvenc` are also available when supported by the GPU and PyAV build.

GPU selection also works for `still`, `preview`, and `doctor`. For Python scripts, set `$env:PROTEINMOTION_GPU_BACKEND = "Vulkan"` and `$env:PROTEINMOTION_GPU_ADAPTER = "NVIDIA"` before launching Python, or pass `backend="Vulkan", adapter_name="NVIDIA"` to `Renderer`. Vulkan is the tested Windows backend; DirectX 12 currently has device-loss failures in the transparency tests on the tested driver. See [rendering and platform limits](rendering.md).

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

Install the extras you need:

```bash
python -m pip install "proteinmotion[preview,md]"
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

`scene.seek(t)` restores the scene at time `t`, including when seeking backward. `scene.render_frame(t)` returns an RGBA array; `output='frame.png'` saves it. Reuse a `Renderer` for many stills. MP4 export automatically tries `h264_videotoolbox` on macOS and `h264_nvenc` on Windows/Linux, then warns and falls back to `libx264` if hardware initialization fails. `--codec libx264` explicitly chooses CPU encoding; molecular rendering still requires a GPU. `hevc_videotoolbox`, `hevc_nvenc` and `av1_nvenc` are also supported when available. Export replaces the requested output only after encoding succeeds.

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
