# Regions and focus

Select a region once, then reuse it for camera focus, annotations, or alignment:

```python
# Inside construct(), after adding p and framing the camera:
helix = p.select(chain="A", residues=(23, 34))
marker = helix.highlight(style="box", color="#f2ba67", padding=1.5)
self.play(FadeIn(marker), run_time=0.6)
self.focus(helix, run_time=1.5, margin=1.6)
self.play(PlayTrajectory(p, state_easing=smooth), run_time=8)
self.play(FadeOut(marker), run_time=0.6)
self.focus(p, run_time=1.5)  # Return to the whole protein.
```

Selections use **PDB author residue numbers**, not array indices. A tuple `(23, 34)`
includes every number from 23 through 34; a list `[8, 44, 70]` selects only those
numbers. `residues=23` selects one residue. Add `atoms="CA"` or
`atoms=["N", "CA", "C", "O"]` to filter by atom name. `chain` accepts a string or
list of chains. All supplied criteria must match. Omitted criteria accept everything;
empty selections raise an error. Combine selections from the same protein with
`region_a | region_b`. Residues with different insertion codes but the same author
number are selected together; use `Region(p, atom_indices)` for exact index selections.

Three highlight styles are available:

| Style | Appearance | Useful options |
|---|---|---|
| `"sphere"` | Translucent sphere enclosing selected atom centers | `opacity=0.18`, `padding=2.0` |
| `"box"` | Wire box aligned with the protein's local axes | `opacity=0.9`, `padding=2.0`, `line_width=0.12` |
| `"atoms"` | Enlarged tinted halos around selected atoms | `opacity=0.18`, `padding=0.3` |

All accept `color="#RRGGBB"`; padding and line width are in ångströms. These are 3D
annotations, not molecular solvent surfaces. They use depth testing and smooth
transparency, and follow the parent's rotation, translation, scaling, deformation,
and trajectory states. Bounds are recomputed as the region changes shape. Animate
their opacity with `FadeIn`, `FadeOut`, or `.animate.set_opacity(...)`; their positions
and transforms belong to the parent. Annotation opacity is independent of protein
opacity. Small annotation geometry updates happen on the CPU; molecular state
interpolation still runs on the GPU.

`self.focus(region, ...)` is an eased camera animation using the scene's aspect ratio.
Tracking is enabled by default: the camera follows the selected region's center while
keeping a steady zoom. Increase `margin` for regions that expand substantially.
Pass `follow=False` to retain the pose fitted when the animation starts. Initial,
immediate placement uses `self.camera.focus(region, aspect=self.width/self.height)`.
`self.camera.frame(p, ...)` fits the whole protein and disables tracking.

For concurrent focus, playback, or fades, use `Focus` explicitly:

```python
self.play(
    Focus(self.camera, helix, margin=1.6, aspect=self.width / self.height),
    PlayTrajectory(p, start=0, end=20, state_easing=smooth),
    run_time=4,
)
```

Focus evaluates after molecular motion in the same clip. It preserves camera angles
and field of view. `self.camera.animate.focus(region, ...)` is also supported; camera
focus and camera orbit/zoom must run in separate clips because both write camera state.
