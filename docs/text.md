# Text, labels, and callouts

Write vector text into your molecular films, label individual amino acids, and draw leaders that follow live regions. All overlays render in the native GPU pipeline, including interactive preview and video export.

Watch the [label film and writing study](https://pdpppd.github.io/proteinmotion/gallery/#labels), or run the [complete script](labels_and_callouts.py).

## Manim-style writing

```python
from proteinmotion import Text, Write, Unwrite

title = Text("A protein in motion", font_size=64, font="semibold")
self.play(Write(title, lag_ratio=0.12, stroke_width=1.8), run_time=2.5)
self.wait(1)
self.play(Unwrite(title), run_time=1.5)
```

`Write` traces each glyph's actual outline, then brings in its fill as the outline thins. Neighboring glyphs start with staggered timing. This reproduces Manim's two-stage writing behavior, not a character-by-character text replacement. `Unwrite` reverses the reveal, erasing the last glyph first by default.

| Option | Meaning |
|---|---|
| `run_time` on `play()` | Total duration in seconds, including the stagger; explicitly choose it for each scene |
| `lag_ratio=None` | Manim-style default `min(4 / max(1, glyph_count), 0.2)` |
| `lag_ratio=0` | All glyphs draw together |
| `stroke_width=1.5` | Outline width in pixels at a 1080-pixel render height |
| `reverse=True` | Reverse the order of the shaped glyphs |
| `rate_func=linear` | Default writing clock; pass `smooth` for easing across the whole animation |

A lag of 0.1 starts each glyph one tenth of an individual glyph's animation duration after its predecessor. The total remains `run_time`. Whitespace does not take a slot; a ligature is one glyph. Text and annotation objects are added automatically when first animated. Set their style before adding them or compiling their timeline.

The animation timing is adapted from MIT-licensed Manim. The renderer uses HarfBuzz shaping, FontTools outlines, cached vector triangles and GPU contour strokes. It does not require a Manim or Pango installation. [Licenses and exact upstream source](https://github.com/pdpppd/proteinmotion/blob/main/THIRD_PARTY.md).

## Place text

```python
title = Text("α helix / β sheet", font_size=48, color="#50e0d0",
             position=(0.06, 0.08))
footer = Text("PDB 2K39", font_size=24).to_corner("DR", buff=0.06)
self.play(Write(title), Write(footer), run_time=2)
self.play(title.animate.move_to((0.10, 0.15)), run_time=1)
self.play(title.animate.set_opacity(0.4), run_time=0.5)
```

Positions are normalized screen coordinates: `(0, 0)` is top-left and `(1, 1)` is bottom-right. Font size, outline width and residue-label offsets scale with render height, so changing from 1080p to 720p preserves the composition. `align="left"|"center"|"right"` positions the text block relative to its anchor. `to_corner` accepts `UL`, `UR`, `DL`, or `DR`.

Use `\n` for explicit line breaks and `line_spacing=1.3` to control baseline spacing. There is no automatic wrapping. The bundled fonts are Source Sans 3 `"regular"` and `"semibold"`; `font="/path/to/font.ttf"` or an `.otf` path selects another font. Kerning and ligatures are enabled. Set `ligatures=False` if you want separate glyphs for combinations such as “fi”. Unsupported characters raise an error rather than silently rendering empty boxes.

## Point to a region

```python
helix = protein.select(chain="A", residues=(23, 34), atoms="CA")
callout = helix.callout(
    "α helix", subtitle="Residues 23–34",
    position=(0.73, 0.25), font_size=36,
    color="#f2ba67", line_width=1.5, tip="arrow",
)
highlight = helix.highlight(style="box", color="#f2ba67")
self.play(Write(callout), FadeIn(highlight), run_time=2)
self.focus(helix, margin=1.8, run_time=1.5)
self.play(PlayTrajectory(protein), run_time=8)
```

`Callout(region, text, ...)` is the equivalent constructor. The text stays at its screen position while the leader endpoint follows the selected atoms' 3D centroid. Select `atoms="CA"` for the backbone centroid or omit the atom filter to include all selected atoms. The anchor updates after coordinate interpolation, deformation, transforms and camera tracking, including interactive orbit/zoom.

During `Write`, the leader draws first and overlaps the start of the lettering. Tips can be `"dot"`, `"arrow"`, or `"none"`. `line_color` overrides the leader color; `subtitle_color` controls the smaller second line. Callout text is clamped to the viewport by default (`clamp=False` disables this). Targets outside the view frustum hide the entire callout.

By default, `follow_opacity=True` makes the annotation follow the mean atom opacity in its selection, including unmatched-residue fades. Set it to `False` to keep a note visible when the parent fades. Callouts remain bound to the original region's atom identities; a different-protein morph does not automatically reassign the label to its target protein.

## Label amino acids

```python
# A single label follows its Cα atom. Default text: “A · Leu 8”.
leu = protein.select(chain="A", residues=8).label(offset=(-180, -60))
self.play(Write(leu), run_time=1)
self.play(leu.animate.set_offset((-220, -90)), run_time=0.8)

# Lists mean specific author residue numbers; tuples mean an inclusive range.
labels = protein.label_residues(
    chain="A", residues=[8, 44, 70],
    font_size=30, color="#f2ba67", format="three_letter",
    offsets={8: (-220, -60), 44: (180, -40)},
)
self.play(Write(labels, lag_ratio=0.08), run_time=2)
```

The direct constructors are `ResidueLabel(region, text=None, ...)` and `ResidueLabels(region, ...)`. Names include the chain, amino-acid name, PDB author number and insertion code. Set `format="one_letter"` for `A · L 8`, `include_chain=False` to hide the chain, or supply custom text to a single label.

Offsets are relative to the projected Cα atom in 1080p design pixels. Use `set_offset()` or `label.animate.set_offset()` to move a single residue label. Use a `Callout` when you want a fixed screen position instead. A label on a single residue without a Cα uses the selected centroid; groups skip residues without Cα atoms.

`ResidueLabels` tries several offsets to reduce overlap inside its own group. Set `avoid_overlap=False` to disable this. Entries in `offsets` are fixed and bypass automatic placement. Dictionary keys can be residue numbers or `(chain, number, insertion_code)` tuples. Automatic placement is deterministic at each frame but can switch positions as models move; explicit offsets are preferable for a carefully composed film.

## Rendering details and limits

Text stays upright and renders as an overlay after molecular transparency. Leaders and text remain readable even when their anchors are behind other atoms; they are not depth-occluded. Combine them with a depth-tested 3D `highlight()` to show spatial context.

Glyph shaping and tessellation happen once per cached text layout, not on every frame. Writing updates small GPU uniforms; only the moving leaders need tiny geometry updates. Completed text skips the contour pass. All output passes through the existing Metal → NV12 → VideoToolbox export path. An 18.8-second, 1080p/60 fps label film exported in 3.74 seconds on the tested M3 Max (single run, excluding scene construction). [Validation and measurements](VALIDATION.md).

This is a focused text/annotation API, not a complete Manim text engine. It does not yet provide MathTex/LaTeX, Pango markup, per-word styles inside one object, font fallback, emoji/color fonts, full mixed-direction paragraph layout or automatic text wrapping. Static TrueType/OpenType outline fonts are supported; arbitrary overlapping/self-intersecting custom glyph outlines have not been verified. Overlap avoidance considers labels within one group, not the molecular silhouette or all scene objects.

## Complete runnable film

The source checkout includes the required 2K39 data file.

```bash
proteinmotion render examples/labels_and_callouts.py ProteinLabels -o labels.mp4 --fps 60
proteinmotion render examples/labels_and_callouts.py WritingStudy -o writing.mp4 --fps 60
```

{{LABEL_SOURCE}}
