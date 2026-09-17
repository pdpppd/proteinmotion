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

Selections use **PDB author residue numbers**. A tuple `(23, 34)` selects the inclusive range; a list `[8, 44, 70]` selects those three residues. `residues=23` selects one residue.

Use `atoms="CA"` or `atoms=["N", "CA", "C", "O"]` to filter by atom name. `chain` accepts one chain ID or a list. All supplied filters must match. An empty selection raises an error.

Combine selections from the same protein with `region_a | region_b`. Residues that share an author number but have different insertion codes are selected together. For exact atom selection, use `Region(p, atom_indices)`.

Three highlight styles are available:

| Style | Appearance | Useful options |
|---|---|---|
| `"sphere"` | Translucent sphere enclosing selected atom centers | `opacity=0.18`, `padding=2.0` |
| `"box"` | Wire box aligned with the protein's local axes | `opacity=0.9`, `padding=2.0`, `line_width=0.12` |
| `"atoms"` | Enlarged tinted halos around selected atoms | `opacity=0.18`, `padding=0.3` |

All highlight styles accept `color="#RRGGBB"`. Padding and line width are in ångströms. Highlights are 3D shapes that move with the selected region. Nearby atoms can hide parts of them.

The highlight resizes as the region changes shape. Animate its opacity with `FadeIn`, `FadeOut`, or `.animate.set_opacity(...)`. Its position follows the parent protein; its opacity is independent. Highlight geometry updates on the CPU, and coordinate interpolation runs on the GPU.

`self.focus(region, ...)` moves the camera to a region with easing and uses the scene’s aspect ratio. By default, the camera then follows the region’s center at a fixed zoom. Increase `margin` for a region that expands during playback. Use `follow=False` to keep the camera at the position fitted when the animation starts.

For immediate placement, use `self.camera.focus(region, aspect=self.width/self.height)`. Use `self.camera.frame(p, ...)` to fit the whole protein and stop region tracking.

For concurrent focus, playback, or fades, use `Focus` explicitly:

```python
self.play(
    Focus(self.camera, helix, margin=1.6, aspect=self.width / self.height),
    PlayTrajectory(p, start=0, end=20, state_easing=smooth),
    run_time=4,
)
```

`Focus` updates after molecular motion in the same `play()` call and preserves the camera angles and field of view. `self.camera.animate.focus(region, ...)` is also available. Put camera focus and orbit or zoom in separate calls because each changes the camera state.

## Text labels and callouts

Use `region.callout("α helix")` to place a label with a line to a selected region. Use `region.label()` for one amino acid, or `protein.label_residues(residues=[8, 44, 70])` for several names. `Write` and `Unwrite` animate these labels. See [text and labels](text.md).