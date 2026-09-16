# Residue colors and surfaces

Color and opacity belong to the selected atoms, so the same styling carries across cartoon, ribbon, ball-and-stick and surface representations. The examples below go inside `ProteinScene.construct()` unless they only configure a protein.

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

Select the whole residue to color its atoms, bonds and backbone together. `atoms="CA"` styles only the selected Cα atoms and their associated cartoon/ribbon samples. Residue numbers are inclusive PDB author numbers, not array offsets. A list selects discrete residues; a tuple selects a range.

`p.set_color(color)` applies an override to the whole protein. `p.color_residues(color, chain="A", residues=[8, 44, 70])` is a convenience method. Pass `None` to `set_color()` to clear an override and recover the representation's base palette. Base palettes include `secondary`, `rainbow`, `chain` and a hex color for cartoon/ribbon/surface; ball-and-stick starts with element colors.

Overrides replace the base color completely. Bonds interpolate endpoint colors, cartoon/ribbon colors interpolate along backbone segments, and surfaces interpolate vertex colors whose ownership is assigned to the nearest atom. Residue boundaries are therefore blended rather than hard segmentation edges.

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

The last residue must have time to finish: `run_time > (number_of_selected_residues - 1) * residue_delay`. Each residue uses the remaining span and ends before the clip ends. `easing="smooth"` is a quintic ease-in/ease-out; `"linear"` is also available. Keep the outer scene clock linear for these clips; passing a nonlinear `rate_func` to `play()` is rejected because it would change delays expressed in seconds.

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

`SetOpacity` has the same delay, reverse and easing options as `Colorize`. It changes selected atom/residue opacity. `p.set_opacity()` and `p.animate.set_opacity()` change a separate global multiplier. Their product also includes any morph visibility. Restoring global opacity does not erase local fades; use `SetOpacity(region, 1)` to restore those.

Fades use smooth weighted blended transparency, including per-residue surface fades. There is no screen-door dithering. Transparent fragments retain approximate depth/color ordering rather than exact sorted blending or refractive glass. A hidden atom hides its connecting bond; surviving bonds use the lower endpoint opacity. Labels follow selected opacity unless configured with `follow_opacity=False`.

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

The SES implementation is a voxel approximation, not an exact analytic rolling-probe surface. Enclosed inaccessible cavities are filled; results depend on grid spacing. Marching cubes extracts the final mesh, and its normals are shaded on Metal. See the [distance-transform surface method](https://doi.org/10.1371/journal.pone.0008140) for the underlying dilation/erosion idea and [scikit-image marching cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) for mesh extraction.

`max_voxels` bounds grid size and raises a useful error instead of allocating an arbitrarily large grid. It is not a total RAM limit: distance fields, mesh arrays and extraction scratch memory also use memory. Increase the spacing for large assemblies.

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

`update="deform"` is an optional fast preview: one reference mesh follows four neighboring atoms per vertex through GPU skinning. It is suitable only for small displacements. Large conformer changes can fold or tear the mesh; solvent topology and cavities do not update. Use `rebuild` for those changes. Neither mode turns a visual interpolation into a physical simulation.

For a contact-guided morph between **different topologies**, continue to use cartoon, ribbon or ball-and-stick. The surface builder does not recompute a solvent boundary around just the currently visible subset of atoms; fading residues hides their surface patches. It is not a general surface correspondence algorithm.

## Try the film

```bash
proteinmotion render examples/molecular_tools.py StylingAndSurface -o styling-and-surface.mp4 --fps 60
```

The film colors two ubiquitin regions, fades the rest, changes representation, draws a live Cα ruler and rebuilds a surface through NMR conformers. [Watch the rendered example](https://pdpppd.github.io/proteinmotion/gallery/#surfaces), read the [complete script](molecular-example.md), or continue to [distances and interactions](interactions.md).
