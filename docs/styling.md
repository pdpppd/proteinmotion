# Residue colors and surfaces

Apply color and opacity to a protein or a selected region. The settings carry across cartoon, ribbon, ball-and-stick, and surface views. Put the animation examples below inside `ProteinScene.construct()`.

## Choose and color residues

```python
from proteinmotion import Protein

p = Protein.from_file("examples/data/2k39.cif", chains="A").cartoon().center()
helix = p.select(chain="A", residues=(23, 34))
strand = p.select(chain="A", residues=(2, 7))
helix.set_color("#50e0d0")
strand.set_color("#f2ba67")
p.select(residues=(71, 76)).set_opacity(0.2)
```

Select a whole residue to color its atoms, bonds, and backbone. Adding `atoms="CA"` colors the Cα atoms and the associated cartoon or ribbon segments. Residue numbers use PDB author numbering. A tuple selects an inclusive range; a list selects individual residues.

`p.set_color(color)` applies an override to the whole protein. `p.color_residues(color, chain="A", residues=[8, 44, 70])` is a convenience method. Pass `None` to `set_color()` to clear an override and recover the representation's base palette. Base palettes include `secondary`, `rainbow`, `chain` and a hex color for cartoon/ribbon/surface; ball-and-stick starts with element colors.

A color override replaces the base color. Colors blend along bonds and backbone segments. Surface vertices take colors from their nearest atoms, and colors blend between vertices. This gives a gradual change at residue boundaries.

## Animate a color change

```python
from proteinmotion import Colorize, SetOpacity

self.add(p)
self.camera.frame(p, aspect=self.width / self.height)
self.play(
    Colorize(helix, "#a89bfa", residue_delay=0.06),
    Colorize(strand, "#f59b75", residue_delay=0.1),
    run_time=2,
)
self.play(helix.animate.set_color("#50e0d0"), run_time=1)
self.play(Colorize(helix, None), run_time=1)  # Fade back to the base palette.
```

`Colorize(target, color, residue_delay=0, reverse=False, easing="smooth")` accepts a protein or region. With a zero delay, every selected residue changes together. A positive delay is the start-time gap in **seconds**, following topology residue order from N to C within each chain. `reverse=True` reverses that order. Atoms in one residue share its timing.

Set `run_time > (number_of_selected_residues - 1) * residue_delay` so every residue has time to change color. Each transition lasts the remaining duration after the last residue has started.

Use `easing="smooth"` for a gradual start and stop, or `"linear"` for a constant rate. Keep the `play()` clock linear; a nonlinear `rate_func` raises an error because it would change the delays in seconds.

Disjoint selections can animate concurrently. Color and opacity can animate on the same selection together. Two animations writing the same color or opacity channel on overlapping atoms raise an error. Coordinates, camera motion and styling can run together, with reproducible backward seeking.

## Control transparency

```python
# Initial setup:
p.set_opacity(0.8)                         # Global multiplier.
p.set_residue_opacity(0.2, residues=(71, 76))

# Timeline animations:
self.play(SetOpacity(helix, 0.15, residue_delay=0.05), run_time=1.5)
self.play(helix.animate.set_opacity(1), run_time=0.8)
self.play(p.animate.set_opacity(1), run_time=0.8)
```

`SetOpacity` uses the same delay, order, and easing options as `Colorize`. It changes the opacity of selected atoms. `p.set_opacity()` and `p.animate.set_opacity()` apply a separate multiplier to the whole protein. The final opacity combines both settings and any morph fades. Use `SetOpacity(region, 1)` to restore a local fade.

Fades use weighted blended transparency with approximate depth ordering. Bonds use the lower opacity of their two atoms, so hiding either atom hides the bond. Labels fade with their selection unless you set `follow_opacity=False`.

## Render a molecular surface

```python
p.surface(
    kind="ses",
    probe_radius=1.4,
    resolution=0.7,
    color="secondary",
    update="rebuild",
    max_voxels=8_000_000,
)
```

All distances above are ångströms. `resolution` is voxel spacing: a smaller value improves detail and increases preparation time and memory.

| Kind | Geometry |
|---|---|
| `"vdw"` | Union of atom van der Waals spheres, using Gemmi element radii |
| `"sas"` | Union of spheres expanded by `probe_radius`; solvent-accessible surface |
| `"ses"` | Approximate solvent-excluded envelope from voxel dilation, cavity filling and distance-transform erosion |

The solvent-excluded surface (SES) is calculated on a voxel grid. Enclosed inaccessible cavities are filled, and grid spacing affects the result. Marching cubes extracts the mesh for GPU rendering. See the [distance-transform surface method](https://doi.org/10.1371/journal.pone.0008140) and [scikit-image marching cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) for the underlying methods.

`max_voxels` limits the number of grid cells. A larger grid raises an error. Distance fields, mesh arrays, and mesh extraction also require memory. Increase `resolution` to use fewer cells for a large assembly.

### Switch representations

```python
from proteinmotion import Representation

# Configure surface settings, then start the scene as a cartoon.
p.surface(kind="ses", resolution=0.5).cartoon()
self.add(p)
self.camera.frame(p, aspect=self.width / self.height)
self.play(Representation(p, "surface"), run_time=1.2)
self.play(Representation(p, "ball_and_stick"), run_time=1.2)
```

Surface settings and residue styles persist across transitions. Calling a representation method during initial setup switches immediately; `Representation` crossfades on the timeline.

### Follow trajectories and deformations

With the default `update="rebuild"`, changed coordinates produce a new mesh. This keeps the surface attached to moving atoms through NMR, MD and ordinary `Morph`/`Deform` animations. CPU voxel construction and mesh extraction can dominate export during coordinate motion. Rotation, camera movement, color and opacity changes reuse the mesh and render on the GPU.

`update="deform"` keeps one reference mesh and moves each vertex with four nearby atoms on the GPU. Use this mode for small displacements. Large changes can fold or tear the mesh, and cavities retain their original shape. Use `update="rebuild"` for larger coordinate changes.

For contact-guided morphs between different protein topologies, use cartoon, ribbon, or ball-and-stick. Surface opacity hides patches assigned to fading atoms. The surface boundary is still calculated from the full atom set, which limits its use for this type of morph.

## Run the example

```bash
proteinmotion render examples/molecular_tools.py StylingAndSurface -o styling-and-surface.mp4 --fps 60
```

The example colors two ubiquitin regions, fades the rest, changes representation, adds a Cα distance label, and rebuilds a surface through NMR conformations. [Watch the video](https://pdpppd.github.io/proteinmotion/gallery/#surfaces) or read the [complete script](molecular-example.md).