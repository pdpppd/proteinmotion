# EEVEE and depth of field

Render a ProteinMotion scene through Blender EEVEE and focus on selected atoms or residues. Camera movement, colors, opacity, labels, trajectories, and morphs use the existing scene API.

[Watch Calmodulin in focus](calmodulin-in-focus.md): a 68-second film with helix close-ups, focus pulls, atomic detail, and a molecular surface. The page includes the full script beside its output.

## Install Blender

Install [Blender 4.5 or later](https://www.blender.org/download/) separately. EEVEE is included with Blender. ProteinMotion starts it in the background; the Blender interface can stay closed. This integration was tested with Blender 5.2 on an Apple M3 Max using Metal and on Windows with an NVIDIA RTX 5070 Ti.

ProteinMotion finds `blender` on `PATH`, standard Windows installations under `Program Files/Blender Foundation/Blender <version>`, or the macOS installation at `/Applications/Blender.app`. Windows discovery chooses the newest installed version. For another location:

```bash
export PROTEINMOTION_BLENDER="/path/to/blender"
proteinmotion doctor
```

The equivalent command-line option is `--blender /path/to/blender`. Python packages and video encoding use the normal ProteinMotion installation. Blender uses its bundled Python environment.

## Render a scene

```bash
proteinmotion render examples/eevee_focus.py HelixFocus --renderer eevee \
    --width 960 --height 540 --fps 60 -o helix-focus.mp4
```

Use `--renderer native` for the default renderer. Interactive `preview` uses the native renderer; use `still --renderer eevee` to review EEVEE frames.

## Focus on residues

Place initial lens settings before the scene's first `play()` or `wait()`:

```python
# Focus on one residue's atom centroid.
self.camera.set_focus(protein, chain="A", residues=5, fstop=5.6)

# Focus on the Cα centroid of an inclusive residue range.
self.camera.set_focus(protein, chain="A", residues=(10, 20), atoms="CA", fstop=4)

# Reuse any existing selection.
helix = protein.select(chain="A", residues=(5, 19))
self.camera.set_focus(helix)
```

`fstop` controls blur. Lower values give a shallower depth of field. The focus point is the mean world position of the selected atoms. A region defines one focus plane; atoms farther from that plane can still blur.

Selections follow rotations, translations, deformation, and trajectory playback. Set `follow=False` to fix the initial focus point in space. You can also pass a world point, such as `(0, 0, 10)`, in ångströms. `camera.set_focus(None)` disables depth of field during initial setup.

Lens focus leaves camera position and zoom unchanged. `camera.focus()` and `scene.focus()` frame a selection. Native renders preserve the scene's framing and show a sharp image.

## Animate focus

```python
from proteinmotion import FocusPull

region = protein.select(chain="A", residues=(82, 92), atoms="CA")
self.play(
    FocusPull(self.camera, region, fstop=4),
    self.camera.animate.orbit(theta=0.3),
    run_time=2,
)
```

You can also write `self.camera.animate.set_focus(region, fstop=4)`. Focus pulls use the scene's eased animation timing and can run together with camera movement or protein deformation.

## Complete example and output

The example highlights calmodulin residues 5–19, fades the surrounding cartoon to 6% opacity, and rotates the camera. It then moves focus to residues 82–92 and returns to the highlighted helix. Download or clone the repository to use the script with `examples/data/1cll.cif`.

```python output=eevee-focus
"""Calmodulin 1CLL: select a helix, fade its context, then pull lens focus.

Render with:
    proteinmotion render examples/eevee_focus.py HelixFocus --renderer eevee \
        --width 960 --height 540 --fps 60 -o helix-focus.mp4

Requires Blender 4.5+. The camera motion illustrates the deposited structure.
"""

from pathlib import Path

from proteinmotion import FocusPull, Protein, ProteinScene, SetOpacity, Text, Write

DATA = Path(__file__).resolve().parent / "data"


class HelixFocus(ProteinScene):
    def construct(self):
        protein = Protein.from_file(DATA / "1cll.cif", chains="A").cartoon(color="#7299b4").center()
        protein.rotate(0.9, axis=(0, 0, 1))
        helix = protein.select(chain="A", residues=(5, 19))
        helix.set_color("#ffb34c")
        # The context selection includes every atom outside the highlighted helix.
        context = protein.select(chain="A", residues=[4, *range(20, 149)])
        far_region = protein.select(chain="A", residues=(82, 92), atoms="CA")
        self.add(protein)
        self.camera.frame(protein, aspect=self.width / self.height).zoom(1.5)
        self.camera.set_focus(protein, chain="A", residues=(5, 19), atoms="CA", fstop=4)
        self.add(Text("Calmodulin · 1CLL", position=(0.055, 0.9), font_size=30))
        label = helix.callout(
            "Alpha helix",
            subtitle="Chain A · residues 5–19",
            position=(0.68, 0.13),
            color="#ffb34c",
            font_size=38,
        )
        self.play(Write(label), run_time=0.8)
        self.play(SetOpacity(context, 0.06), run_time=1)
        self.play(self.camera.animate.orbit(theta=0.28), run_time=2)
        self.wait(0.4)
        self.play(FocusPull(self.camera, far_region, fstop=2.8), SetOpacity(context, 0.35), run_time=1.4)
        self.wait(0.4)
        self.play(FocusPull(self.camera, helix, fstop=4), SetOpacity(context, 0.06), run_time=1.2)
        self.wait(0.8)
```

## Quality and performance

```python
from proteinmotion import EEVEEOptions

scene = HelixFocus(width=1920, height=1080, fps=60)
scene.render(
    "helix.mp4",
    renderer="eevee",
    eevee=EEVEEOptions(samples=64, supersampling=1.5, max_blur=24),
)
scene.render_frame(3, output="helix.png", renderer="eevee")
```

The default 64 samples and 1.5× spatial supersampling balance edge quality and render time. Use `--samples 128 --supersampling 2` for a more expensive export. Shadows, camera jitter, and screen-space ray tracing are disabled to reduce temporal noise. Text and callout lines are composited after depth of field, so they remain sharp.

Blender stays open for the full export. The backend reuses meshes when connectivity remains unchanged and updates positions and colors for each frame. Rendering and GPU composition run in EEVEE; mesh evaluation and file transfer add CPU overhead. The native renderer is faster for large scenes and long trajectories.

Opacity uses a smooth blend of opaque render layers. Same-opacity geometry retains opaque visibility ordering. Each distinct opacity value adds a render layer, so group fades are more efficient than many staggered fades. See [rendering methods](rendering.md) for the approximation and layer limit.

Both backends support cartoons, ribbons, ball-and-stick models, molecular surfaces, and 3D annotations. EEVEE uses its own studio lighting and color management; its colors can differ from the native renderer. The native `depth_cue` and `msaa` options are replaced by EEVEE's lighting and sampling settings.
