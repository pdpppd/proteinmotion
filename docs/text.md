# Text, labels, and callouts

Add text, amino acid names, and labels for selected regions. A callout connects a label to a region with a line. These annotations work in preview windows, still images, and exported videos.

Watch the [label and text animation examples](https://pdpppd.github.io/proteinmotion/gallery/#labels), or run the [complete script](labels_and_callouts.py).

## Manim-style writing

```python
from proteinmotion import Text, Write, Unwrite

title = Text("A protein in motion", font_size=64, font="semibold")
self.play(Write(title, lag_ratio=0.12, stroke_width=1.8), run_time=2.5)
self.wait(1)
self.play(Unwrite(title), run_time=1.5)
```

`Write` draws each letter’s outline, then fills it in as the outline fades. Letters start in sequence. `Unwrite` erases the text, starting with the last letter by default.

| Option | Meaning |
|---|---|
| `run_time` on `play()` | Total duration in seconds, including the stagger; explicitly choose it for each scene |
| `lag_ratio=None` | Manim-style default `min(4 / max(1, glyph_count), 0.2)` |
| `lag_ratio=0` | All glyphs draw together |
| `stroke_width=1.5` | Outline width in pixels at a 1080-pixel render height |
| `reverse=True` | Reverse the order of the shaped glyphs |
| `rate_func=linear` | Default writing clock; pass `smooth` for easing across the whole animation |

A `lag_ratio` of 0.1 starts each glyph one tenth of a glyph’s animation duration after the previous one. The total animation lasts `run_time` seconds. Spaces are skipped; a ligature counts as one glyph.

Animating a text or annotation object adds it to the scene. Set its style before adding it or calling `play()`.

The animation timing comes from MIT-licensed Manim. ProteinMotion renders the text using HarfBuzz, FontTools, and its GPU renderer. See the [source attribution and licenses](https://github.com/pdpppd/proteinmotion/blob/main/THIRD_PARTY.md).

## Place text

```python
title = Text("α helix / β sheet", font_size=48, color="#50e0d0",
             position=(0.06, 0.08))
footer = Text("PDB 2K39", font_size=24).to_corner("DR", buff=0.06)
self.play(Write(title), Write(footer), run_time=2)
self.play(title.animate.move_to((0.10, 0.15)), run_time=1)
self.play(title.animate.set_opacity(0.4), run_time=0.5)
```

Positions use fractions of the image width and height: `(0, 0)` is the top-left corner and `(1, 1)` is the bottom-right. Font sizes, outline widths, and residue-label offsets use pixels at a 1080-pixel image height. They scale with the output height.

Set `align` to `"left"`, `"center"`, or `"right"` to align the text at its position. `to_corner` accepts `UL`, `UR`, `DL`, or `DR`.

Use `\n` to add line breaks. Set `line_spacing=1.3` to control the space between lines. Text wrapping is manual.

The bundled fonts are Source Sans 3 `"regular"` and `"semibold"`. To use another font, pass its `.ttf` or `.otf` path. Kerning and ligatures are enabled by default. Use `ligatures=False` to draw combinations such as “fi” as separate glyphs. A character missing from the selected font raises an error.

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

`Callout(region, text, ...)` creates the same annotation. The label stays at the specified image position. A line connects it to the center of the selected atoms and updates as the protein or camera moves.

Select `atoms="CA"` to use the Cα atoms for the center. Omit the atom filter to use all selected atoms.

`Write` starts drawing the line before the lettering. Choose `"dot"`, `"arrow"`, or `"none"` for the line tip. `line_color` sets the line color; `subtitle_color` sets the second line of text.

Callout text stays within the image by default. Set `clamp=False` to allow it outside the image. The callout is hidden when its target leaves the camera view.

`follow_opacity=True` makes the annotation fade with the average opacity of its selected atoms. Set it to `False` to keep the label visible when the protein fades.

A callout remains attached to its original selection. After a morph to a different protein, create a callout for the target selection.

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

A residue label’s offset is measured from its Cα position, using pixels at a 1080-pixel image height. Use `set_offset()` or `label.animate.set_offset()` to move it. Use a `Callout` for a fixed image position.

For a single residue that lacks a Cα atom, the label uses the center of the selection. Label groups skip residues that lack Cα atoms.

`ResidueLabels` tries several positions to reduce overlap within the group. Set `avoid_overlap=False` to use fixed placement. You can also supply an `offsets` dictionary, keyed by residue number or `(chain, number, insertion_code)`. Those entries use the supplied positions.

Automatic placement can move a label as the structure changes. Use explicit offsets when you need stable label positions throughout a video.

## Rendering details and limits

Text and callout lines are drawn over the protein image. Atoms and surfaces behind or in front of a selected region leave its label visible. Add a 3D `highlight()` to show the region’s position within the structure.

The renderer caches the text geometry. During `Write`, the GPU controls how much of each outline and fill is visible. Moving callout lines update each frame. See [rendering](rendering.md) for the implementation and [benchmarks](VALIDATION.md) for measured export times.

Supported fonts use static TrueType or OpenType outlines. Current limitations include LaTeX/MathTex, markup, styles within a single text object, automatic font fallback, color emoji, mixed-direction paragraphs, and automatic wrapping. Custom glyphs with overlapping or self-intersecting outlines remain untested.

Overlap avoidance checks labels within one group. Check the rendered frames for collisions with the protein and other scene objects.

## Complete example

The source checkout includes the required 2K39 data file.

```bash
proteinmotion render examples/labels_and_callouts.py ProteinLabels -o labels.mp4 --fps 60
proteinmotion render examples/labels_and_callouts.py WritingStudy -o writing.mp4 --fps 60
```

{{LABEL_SOURCE}}
