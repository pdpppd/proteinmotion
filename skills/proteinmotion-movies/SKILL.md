---
name: proteinmotion-movies
description: Create and render protein, DNA, and RNA movies with ProteinMotion from PDB/mmCIF structures or trajectories, including base representations, surfaces, ligands, metal ions and side chains drawn over cartoons, eased motion, region annotations, and protein morphs. Use for making or revising molecular animation scripts and their rendered videos.
---

# ProteinMotion Movies

Deliver an editable Python scene and a rendered movie using ProteinMotion. Preserve the user's molecular structure, selected regions, scientific message, duration, and output format. ProteinMotion is a standalone renderer with Manim-inspired syntax.

## Get a working renderer

Use the task's existing Python environment if it contains ProteinMotion. Otherwise create a project-local virtual environment with Python 3.11+ and install from PyPI:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "proteinmotion>=0.11.0"
.venv/bin/proteinmotion doctor
```

On Windows, use `py -3 -m venv .venv`, `.venv\Scripts\python.exe`, and `.venv\Scripts\proteinmotion.exe`. Windows/NVIDIA support requires version 0.10.1 or later, 64-bit Python, and a current NVIDIA driver. `doctor --check-encoders` should identify a Vulkan adapter and usable `h264_nvenc` encoder. On Apple silicon, use native arm64 Python; the corresponding backend and encoder are Metal and `h264_videotoolbox`.

PyAV supplies the FFmpeg libraries. Add extras as needed: `"proteinmotion[md,preview]>=0.11.0"`. `md` adds trajectory readers; `preview` adds an interactive GPU window. For an existing global command installed with pipx, use that command to render scenes; install the package in the task's environment when importing it into another Python process.

If hardware rendering fails, diagnose the reported adapter and encoder before changing renderers. `--codec libx264` selects CPU encoding; molecular rendering still needs a GPU.

## Author the requested film

For a new project, `proteinmotion init movie` creates `movie/film.py` and `movie/1ubq.cif`. The bundled [starter scene](assets/film.py) and [ubiquitin structure](assets/1ubq.cif) are also directly reusable. Use this structure only for a demo with no requested input. Substitute the user's structure and actual selections when provided; do not keep the starter's residue numbers or labels on a different protein.

Load inputs and inspect chain IDs, residue numbers, atom availability, bound ligands and ions (`p.topology.residue_categories`), and state counts before choosing labels or analyses. Cartoons draw ligands, ions, and loaded waters by default; hide crystallization additives that are not part of the story. Resolve structure paths relative to `Path(__file__).parent` so the script works from any current directory. Keep input structures alongside the script or use explicit user paths.

Compose a normal `ProteinScene.construct()` with `add`, `play`, and `wait`. Configure each object's initial geometry, palette, radii, and transform **before adding it**; frame the camera before the first `play`/`wait`. Use animations for subsequent changes so seeking and export reproduce the timeline. Clip durations add up; `run_time` belongs to `play`, not animation constructors. Angles are radians and coordinates are Å.

For an explanatory tour, prefer a continuous model and eased focus/representation transitions when they suit the user's storyboard. Leave enough time to read callouts. If the user wants cuts or a montage, follow that direction. Render at 60 fps for smooth final motion unless the user specifies otherwise.

Read only the relevant supporting reference:

- [DNA and RNA](references/nucleic-acids.md): nucleotide backbones, base styles, modified residues, surfaces, color, opacity, and trajectories.
- [Ligands, ions, and side chains](references/ligands-and-side-chains.md): default ligand/ion display, category and distance selectors, `ShowSideChains`/`ShowAtoms`, colors, and binding-site storyboards.
- [Animation and state changes](references/animation.md): representations, residue styling, timelines, trajectories, same-topology deformation, and different-protein contact-map morphs.
- [Numerical values, plots, and density](references/data-visualization.md): B factors, RMSF, custom residue values, synchronized plots, MRC/CCP4 contours, and slices.
- [Annotations and interactions](references/annotations-and-interactions.md): selectors, focus, text, 3D regions, distance rulers, hydrogen bonds, and imported-charge electrostatics.

## Render and check

For the generated starter, the class name is `ProteinMovie`:

```bash
proteinmotion still movie/film.py ProteinMovie --time 5 \
  --width 1280 --height 720 -o movie/review.png
proteinmotion render movie/film.py ProteinMovie --fps 60 \
  --width 1920 --height 1080 --msaa 4 -o movie/film.mp4
```

Inspect representative frames for camera fit, label collisions, line visibility, and transitions. For a long film, start with a lower-resolution draft or selected stills. Check the movie's duration, frame rate and successful decoding, especially around changed transitions; use PyAV if no `ffprobe` is available. Use normal CPU tests only for logic that needs them, not for every authored clip.

Surface rebuilding is CPU-bound during coordinate changes; adjust grid resolution, output size, or storyboard if it dominates. Keep the GPU renderer alive when rendering multiple stills. Match the scene and renderer dimensions/MSAA.

Return the movie and runnable source with a concise description of what was shown. Record PDB IDs, trajectory origin, selections, and any approximations in the script or accompanying description. Morphs, NMR interpolation and procedural deformations are visual illustrations; do not present them as MD or physical pathways. Publish to a site/repository only within the user's requested scope.

For API details beyond these references, use the [official documentation](https://pdpppd.github.io/proteinmotion/docs/api/) or inspect the installed package. Do not invent Manim methods that ProteinMotion does not implement.

## Optional EEVEE depth of field

Use `--renderer eevee` for lens depth of field. Blender 4.5+ must be installed separately; EEVEE is included. Find it with `proteinmotion doctor`, set `PROTEINMOTION_BLENDER`, or pass `--blender /path/to/blender`. macOS uses Metal. The native renderer remains the default for fast previews and large trajectories.

Before the first play/wait, call `self.camera.set_focus(protein, chain="A", residues=(10, 20), atoms="CA", fstop=5.6)` with an actual selection from the input. This sets a focus plane through the mean selected position without moving the camera. Use `FocusPull(self.camera, region, fstop=4)` inside `play()` for an eased change; it can run alongside orbit or zoom. Targets follow motion by default. `camera.set_focus(None)` disables DOF.

EEVEE renders opacity groups as separate opaque layers and mixes them smoothly. Keep the number of distinct simultaneous opacities small for efficient rendering. Same-opacity geometry retains opaque visibility ordering; this is a group fade approximation. Defaults are 64 samples and 1.5× spatial supersampling. Labels are composited after DOF. Inspect focus, silhouettes, fades, and labels during motion. See the [EEVEE guide](https://pdpppd.github.io/proteinmotion/docs/eevee/) for a complete example.
