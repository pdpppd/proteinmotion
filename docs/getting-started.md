# Get started

Python 3.11+ and a native GPU are required. macOS with Apple silicon is the verified platform. The core renderer uses Metal; MP4 export uses VideoToolbox hardware encoding.

## Install from source

```bash
git clone https://github.com/pdpppd/proteinmotion.git
cd proteinmotion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[preview]'
proteinmotion doctor
```

The package is installed from this repository; no PyPI release is assumed. To enable MD file readers, use `python -m pip install -e '.[preview,md]'`.

`doctor` lists the detected GPU backend and available video encoders. On Apple silicon, look for `Metal` and `h264_videotoolbox`. PyAV bundles FFmpeg libraries; an external FFmpeg executable is not needed for rendering.

## Render your first film

```bash
proteinmotion render examples/quickstart.py Quickstart -o first-film.mp4 --fps 60
```

This loads the included ubiquitin structure, rotates a cartoon, highlights its helix, focuses the camera, then switches to ball-and-stick. Run commands from the repository root so example paths resolve.

## A complete scene

```python
"""A first film: rotation, a live region highlight, focus, and ball-and-stick."""

from pathlib import Path

from proteinmotion import FadeIn, FadeOut, Protein, ProteinScene, Representation, Rotate

DATA = Path(__file__).parent / "data" / "1ubq.cif"


class Quickstart(ProteinScene):
    def construct(self):
        protein = Protein.from_file(DATA).cartoon().center()
        self.add(protein)
        self.camera.frame(protein, aspect=self.width / self.height)

        self.play(FadeIn(protein), run_time=0.8)
        self.play(Rotate(protein, 1.2), run_time=2)

        helix = protein.select(chain="A", residues=(23, 34))
        highlight = helix.highlight(style="box", color="#f2ba67")
        self.play(FadeIn(highlight), run_time=0.6)
        self.focus(helix, margin=1.7, run_time=1.5)
        self.play(Rotate(protein, 0.5), run_time=1.5)
        self.play(FadeOut(highlight), run_time=0.6)
        self.focus(protein, run_time=1.2)

        self.play(Representation(protein, "ball_and_stick"), run_time=1)
        self.wait(0.8)


if __name__ == "__main__":
    Quickstart(width=1920, height=1080, fps=60).render("first-film.mp4")
```

## Preview and export

```bash
proteinmotion render examples/showcase.py Showcase -o movie.mp4 --width 1920 --height 1080 --fps 60
proteinmotion still examples/showcase.py Showcase -o frame.png --time 4.5
proteinmotion preview examples/showcase.py Showcase
```

Preview: drag to orbit; wheel to zoom; Space to play/pause; Left/Right to seek;
Home to rewind; R to reset the preview camera; Escape to close. Preview uses a native
GPU window and does not read rendered pixels back to the CPU. Camera overrides in
preview are temporary and do not change the authored film.

`scene.seek(t)` is deterministic, including backward seeks. `scene.render_frame(t)`
returns an RGBA array; `output='frame.png'` saves it. Use a persistent `Renderer`
when generating many individual frames. MP4 export selects `h264_videotoolbox` on
macOS if available, with `allow_sw=0` to ensure hardware encoding. `codec='libx264'`
is an explicit CPU fallback. `hevc_videotoolbox` is also supported. Export writes a
temporary file and replaces the requested output only when encoding succeeds.

## Optional dependencies

| Extra | Purpose |
|---|---|
| `preview` | Interactive native window |
| `md` | MDAnalysis trajectory readers |
| `dev` | Tests, linting, package builds |
| `plots` | Contact-map report figures |

See [the full NMR script](full-example.md) for state playback and multiple highlights.
