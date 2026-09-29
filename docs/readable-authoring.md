# Readable authoring

ProteinMotion uses the same selections and styles for setup and animation. Ordinary setters take effect at the current authoring time; `play()` interpolates changes. Seeking a scene does not change where subsequent authoring continues: new actions append to the end.

## A complete rendered example

The helix color, surrounding opacity, camera orbit, and caption have independent
durations. A shared style configures the cartoon; the scene stores its Studio look.
Download [the script](studio_examples.py) and place
[1UBQ](data/1ubq.cif) in `data/` beside it. The script defines
`DATA = Path(__file__).parent / "data"` and includes the imports.
These authoring improvements require ProteinMotion 0.13.0 or later. Upgrade with
`python -m pip install --upgrade "proteinmotion>=0.13.0"`.

```python output=readable-timing
class ReadableTiming(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="film")

    def construct(self):
        style = MolecularStyle("cartoon", {"color": "#8fe3c0"})
        protein = Protein.from_file(DATA / "1ubq.cif", chains="A").set_style(style).center()
        helix = protein.select(residue_range=(23, 34))
        self.camera.frame(protein, margin=1.12, screen_position=(0.64, 0.54))
        caption = Text("Independent timing", position=(0.05, 0.09), font_size=48)
        self.play(FadeIn(protein).during(2), Write(caption).during(1, delay=0.5))
        self.play(
            helix.animate.set_color("#ffc46b").show_atoms().during(2),
            (~helix).animate.set_opacity(0.25).during(1, delay=0.5),
            self.camera.animate.orbit(0.6).during(3),
        )
        self.play(
            (~helix).animate.set_opacity(1),
            caption.animate.set_text("One scene · Shared styles"),
            run_time=1.5,
        )
        self.wait(1)
```

## One scene configuration

Store the renderer and look on the scene so the CLI, interactive preview, stills, and movie export use the same settings:

```python
from proteinmotion import ProteinScene, StudioLook

class Movie(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="clean")

    def construct(self):
        ...

movie = Movie(fps=60)
movie.preview()
movie.render_frame(2, output="frame.png")
movie.render("movie.mp4")
```

The CLI preserves constructor defaults unless you supply overrides. When a file defines just one scene, its class name is optional:

```sh
proteinmotion render movie.py -o movie.mp4
proteinmotion preview movie.py --renderer studio --effects film
```

Blender EEVEE still uses offline rendering; interactive preview supports native and Studio. `camera.lens_focus()` controls EEVEE depth of field, while `camera.frame()` and `camera.animate.frame()` control framing. The older `focus()` and `set_focus()` names remain available.

## Selections stay readable

```python
helix = protein.select(chain="A", residue_range=(23, 34))
carbons = protein.select(element="C")
backbone = protein.select(polymer=True, atoms=["N", "CA", "C", "O"])
sheets = protein.select(secondary="E")
inserted = protein.select(chain="A", residues=42, icode="A")
rest = ~helix
helix_carbons = helix & carbons
side = helix - backbone
site = pocket | ligand
ligand.filter(element="C").set_color("#e7bd62")
```

`residue_range=(23, 34)` explicitly selects an inclusive author-number range. `residues=[23, 34]` selects exactly two numbers. The legacy tuple spelling `residues=(23, 34)` remains an inclusive range.

`protein.summary()` lists counts, chains, residue categories and names, and frame count. Optional static selections use `required=False` and return `None` when absent:

```python
ions = protein.select(ions=True, required=False)
if ions is not None:
    ions.set_color("#aa88ff")
```

A normal spatial query freezes membership. `updating=True` re-evaluates it whenever indices or positions are read:

```python
near_ligand = protein.select(within=4, of=ligand, updating=True)
initial_pocket = near_ligand.freeze()
```

An updating region can temporarily be empty. Use it for live queries and interaction filters; freeze it before geometry whose atom count must stay fixed or before a styling animation. Set operations capture their operands' current membership. `len(region)` counts atoms and an empty region is false.

## Setup and animation use the same intent

```python
scene.add(protein)
protein.set_opacity(0.25)  # Recorded even though add() has already happened.
scene.wait(1)
scene.play(protein.animate.set_opacity(1))

scene.play(
    helix.animate.set_color("#50e0d0").set_opacity(0.5).show_atoms(),
    protein.animate.shift((3, 0, 0)),
    run_time=2,
)
```

`FadeOut(object)` followed by `FadeIn(object)` restores its previous visible opacity. Initial zero-opacity objects fade in to one.

Opacity has an explicit scope:

| Call | Scope |
| --- | --- |
| `protein.set_opacity(0.5)` | Global object multiplier |
| `protein.animate.set_opacity(0.5)` | Global object multiplier |
| `SetOpacity(protein, 0.5)` | Global object multiplier |
| `SetOpacity(region, 0.5)` | Selected atom/residue opacity |
| `SetOpacity(protein, 1, scope="residues")` | Reset all per-atom opacity multipliers |

The global multiplier and local opacities still combine. Restore a local fade through its region or explicit `scope="residues"`. `protein.reset_style()` clears atom color/detail/opacity overrides and cartoon thickness, and restores global opacity to one; it retains the active representation and base palette.

Topology-changing setup such as assigning secondary structure should happen before building the timeline.

## Configure the representation where it is used

```python
from proteinmotion import MolecularStyle, Representation

surface = MolecularStyle("surface", {"kind": "vdw", "grid_spacing": 0.6})
scene.play(Representation(protein, surface), run_time=1.5)
scene.play(Representation(protein, "ball_and_stick", atom_scale=0.25, bond_radius=0.1))
scene.play(protein.animate.representation("cartoon", color="secondary"))
```

Use the same style for initial setup with `protein.set_style(surface)`. Representation options are applied at the transition's start while representation weights crossfade. Per-atom color overrides remain until explicitly cleared. `cartoon(atom_scale=..., bond_radius=...)` configures ligand/side-chain detail without briefly activating another representation.

`grid_spacing` is in Å; smaller spacing increases surface detail. `resolution` remains an alias. Geometry styles belong to the protein; Studio's global `cartoon_style` and `surface_mode` choose rendering algorithms for the entire renderer.

## Easing and independent timing

Use `easing="smooth"`, `easing="linear"`, or a bounded callable. `scene.play(..., easing=...)` applies that easing to each action, including each residue of a staggered animation. Legacy `rate_func` remains a clip-clock override; use `easing` for new code.

```python
scene.play(Colorize(helix, "#f08040", stagger=0.6), run_time=1)
scene.play(
    protein.animate.rotate(0.8).during(3),
    Write(caption).during(1, delay=0.5),
)
```

`during()` uses seconds. The scene infers a three-second group duration above; supplying `run_time` scales the whole group. `AnimationGroup(..., durations=[...], offsets=[...])` is the explicit form. Actions may write the same property sequentially but overlapping writes are rejected.

For repeated items, `delay_seconds` means seconds between residue, chain, or glyph starts. `Colorize(..., stagger=0.6)` reserves 60% of the clip for residue starts and fits any residue count. `Thread` and `BackboneMorph` expose the same total-fraction behavior as `stagger_fraction=0.6`. Their older `stagger`, `residue_delay`, and text `lag_ratio` arguments retain their historical meanings. Use either a delay in seconds or a fraction, not both.

Threading takes the scene's camera and aspect automatically:

```python
scene.play(Thread(protein, easing="smooth", glow=0), run_time=5)
```

Cartoon, ribbon and ball-and-stick geometry are supported. Surface threading raises an error. For the older wire mode, use `Thread(protein, mode="wire", wire_options={"settle": 0.2})`. Legacy `head`, `settle`, and `resolution` warn when explicitly supplied in geometry mode because they have no effect there.

## Camera composition and cutaways

```python
scene.camera.frame(protein, screen_position=(0.65, 0.5))
scene.focus(ligand, margin=2, screen_position=(0.65, 0.5))
scene.play(Reveal(scene.camera, ligand, padding=2.5, surface_keep=0.35,
                  rim="#8cebdc", window=1.5, shape="tunnel"))
```

`screen_position` uses normalized coordinates from the top-left. Camera framing, focus and threading inherit the scene aspect ratio. Explicit `aspect` overrides remain available. Cutaway shape, padding, kept surface fraction and rim color can all be supplied to `Reveal`.

## Consistent measurements and trajectories

```python
ruler = Distance(residue_a, residue_b, anchor="backbone")
plot = ruler.plot(trajectory=trajectory)
scene.add(ruler, plot)
scene.play(PlayTrajectory(protein, trajectory))
```

`Distance` and `TimeSeriesPlot.distance` now share the same default backbone anchors. Pass `anchor="centroid"` for the old distance-plot behavior. The plot uses model-space distances in Å. Explicit `trajectory=` binds the plot to the same trajectory being played without mutating `protein.trajectory`.

`Trajectory(..., times=physical_times, time_unit="ns")` stores a physical clock. MDAnalysis trajectories expose physical timestamps in ps, read lazily. Plots use those timestamps automatically; inputs without them use frame indices. `trajectory.aligned(selection=protein.select(atoms="CA"))` accepts a Region and preserves timestamps.

Manual morph correspondence can use author residue identities:

```python
match = ContactMatch.from_residues(source, destination, [
    (("A", 10), ("B", 25)),
    (("A", 11), ("B", 26)),
    (("A", 12, "A"), ("B", 27)),
])
morph = BackboneMorph(source, destination, match=match)
scene.play(morph)
scene.play(morph.result.animate.rotate(0.4))
```

A backbone morph's `result` is its destination; labels and regions for subsequent shots should refer to that object.

## Interactions, labels and maps

`protein.hydrogen_bonds().diagnostics()` reports explicit and virtual-backbone donors, donors lacking usable H, acceptor counts and detected pairs. Load structures with `include_hydrogens=True` when explicit side-chain or nucleotide hydrogen positions are present. Diagnostics do not add hydrogens or infer nonstandard chemistry.

```python
bonds = protein.hydrogen_bonds()
network = bonds.highlight(between=(ligand, pocket), max_pairs=200)
```

`between` keeps interactions crossing the two selections. The older `region` filter keeps interactions touching either endpoint. Truncation at `max_pairs` emits a warning with displayed and total counts.

`SequenceTrack` and `ContactMap` use `display_region` to limit rows and `highlight_region` to mark selected rows. Their older `region` and `selection` names are aliases.

```python
body_style = TextStyle(font_size=26, color="#d5dde8")
caption = TextGroup(
    Text("Buried ligand", font_size=42),
    Text("A pocket inside the protein", style=body_style, max_width=460),
    position=(0.06, 0.1), gap=18,
)
scene.play(Write(caption))
scene.play(note.animate.set_text("Hydrogen-bond network"))
```

Text width and group gaps use 1080p design pixels. Text replacement occurs at the animation midpoint and supports backward seeking. Residue labels now include ligand and ion centroids when no backbone anchor exists.

`density.crop(region)` associates the cropped map with that protein's transforms. For an uncropped map, use `density.attach_to(protein)`. Subsequent surfaces and slices inherit the association; pass `follow=False` for a fixed world-space mesh. `density.slice(samples=128)` makes the sampling count explicit; `resolution` remains an alias.

## Studio preset reuse

```python
look = StudioLook(lighting="soft", material="medium", effects="film", rim_color="#ff8800")
clean = look.with_presets(material="rough", effects="clean")
```

`with_presets`, `dataclasses.replace`, and preset assignment re-resolve inherited settings, preserving explicit overrides. Colors accept RGB triples or hex strings. Recreate an existing renderer after changing its lighting/material or geometry settings; film-effect uniforms remain adjustable on a live Studio renderer. Scene render methods create a renderer using the current scene look.
