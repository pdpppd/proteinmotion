# Distances and interactions

Attach live rulers to atoms or residues, then use the same line styling for hydrogen bonds and screened electrostatic contacts. Coordinates and distances are evaluated from the current model state, so annotations follow trajectory playback and deformation.

## Measure between residues or atoms

```python
from proteinmotion import Distance, Write

a = p.select(chain="A", residues=23)
b = p.select(chain="A", residues=34)
ruler = Distance(a, b, mode="2d", color="#50e0d0", prefix="Cα · ")
self.play(Write(ruler), run_time=1.5)

# Explicit atom selections measure those atoms instead.
oxygen = p.select(chain="A", residues=23, atoms="O")
nitrogen = p.select(chain="A", residues=27, atoms="N")
atomic_ruler = oxygen.distance_to(nitrogen, mode="3d", style="dashed")
self.add(atomic_ruler)
```

A whole single-residue selection uses its Cα by default, if one exists. A one-atom selection always uses that atom. `anchor="centroid"` uses the mean of the selected coordinates; selections spanning multiple residues also use their centroid. Set `unit="nm"` to convert the displayed distance from ångströms to nanometers, or `unit=""` for no suffix. The `.distance` property remains in ångströms for model space.

| Option | Default | Meaning |
|---|---|---|
| `mode` | `"3d"` | Depth-tested cylinders in the actual scene, or `"2d"` overlay leaders |
| `space` | `"model"` | Physical coordinate distance, ignoring display transforms; `"world"` includes them |
| `style` | `"dashed"` | Dashed or solid line |
| `color`, `label_color` | gold, inherits line | Independent line and text colors |
| `radius` | `0.09` | 3D cylinder radius in scene units |
| `line_width` | `1.7` | 2D width in pixels at 1080p height |
| `dash_count`, `dash_ratio` | `12`, `0.6` | Number of dashes and filled fraction |
| `font_size`, `precision` | `27`, `1` | Text size at 1080p and decimal places |
| `show_distance` | `True` | Hide the caption while retaining the line if false |
| `follow_opacity` | `True` | Fade with the less-visible endpoint |
| `opacity` | `1` | Additional annotation opacity multiplier |

The number sits in a gap at the line's midpoint and updates as the coordinates move. Labels are screen-facing overlays in **both** modes. In 3D mode only the line is depth-tested; opaque atoms and surfaces can occlude it. In 2D mode both line and text stay above the protein, like a callout. A ruler hides when either endpoint leaves the viewport. This version does not automatically resolve collisions between multiple ruler captions.

Use `space="world"` for endpoints on different protein objects. World distances measure scene geometry and are physical only if the objects share a common coordinate frame and scale. Rotating or scaling a display model does not affect the default same-protein model-space measurement.

`Write`, `Unwrite`, `FadeIn`, `FadeOut` and `.animate.set_opacity()` work on a ruler. The line draws before the text fills. [Text animation options](text.md).

## Compute hydrogen bonds

```python
from proteinmotion import HydrogenBonds

hb = HydrogenBonds(
    p,
    donors=p.select(chain="A", residues=(23, 34), atoms="N"),
    max_distance=3.5,
    min_angle=150,
    hydrogens="auto",
)
for pair in hb.pairs:
    print(pair.a, pair.b, pair.distance, pair.angle, pair.inferred_hydrogen)

bonds = hb.highlight(
    mode="3d", color="#50e0d0", radius=0.08,
    style="dashed", dash_count=7,
    show_distances=True, font_size=24, max_pairs=12,
)
self.play(Write(bonds), run_time=1.5)
```

`p.hydrogen_bonds(**options)` is equivalent. `.pairs` returns immutable `Interaction` records: zero-based donor/acceptor atom indices `a`/`b`, donor–acceptor distance in Å, D–H–A angle in degrees, an explicit hydrogen index when available, and an `inferred_hydrogen` flag. The labels show **donor–acceptor** distance, not H–acceptor distance.

Detection requires `1.5 Å < D–A ≤ max_distance` and `D–H–A ≥ min_angle`. Defaults are **3.5 Å and 150°**, both configurable. Donor and acceptor in the same residue are excluded. Multiple qualifying hydrogens for one D–A pair collapse to the most linear angle. These are geometric criteria; there is no hydrogen-bond energy calculation. The angle/distance definitions follow the conventional [hydrogen-bond analysis](https://docs.mdanalysis.org/2.9.0/documentation_pages/analysis/hydrogenbonds.html), with our explicitly chosen distance cutoff.

### Explicit and missing hydrogens

| `hydrogens` | Behavior |
|---|---|
| `"explicit"` | Only hydrogens present in the loaded coordinates |
| `"auto"` | Explicit hydrogens first; infer a backbone amide H when none is available |
| `"backbone"` | Use only inferred backbone amide hydrogens |

Load explicit H with `Protein.from_file(path, include_hydrogens=True)`. Explicit hydrogens are assigned to the nearest candidate donor in the same residue within `donor_h_cutoff=1.3 Å` (and farther than 0.4 Å). Adjust that cutoff for longer donor–H bonds, such as sulfur–H.

Virtual backbone H sits 1.01 Å from N along the outward bisector of its bonds to the preceding peptide carbonyl C and its own Cα. Inference requires same-chain peptide connectivity and skips proline, terminal/missing geometry and invalid bond lengths. This is an approximate geometry construction, not protonation or hydrogen placement by a force field. Missing side-chain hydrogens are never invented.

Default donor/acceptor templates cover common standard amino-acid groups. Histidine tautomerism, unusual protonation, ligands, waters and nonstandard residues require appropriate `donors=` / `acceptors=` selections and explicit hydrogens. Selections may be `Region` objects or integer atom-index arrays. Do not interpret a missing inferred bond as evidence that no interaction exists.

## Screened Coulomb electrostatics

```python
from proteinmotion import Electrostatics

# q is one charge in elementary-charge units per selected atom, in topology order.
field = Electrostatics(
    p, charges=q,
    dielectric=80, screening_length=8, cutoff=12, min_energy=0.05,
)
contacts = field.highlight(
    mode="2d", show_distances=True, style="solid",
    attractive_color="#58b8fa", repulsive_color="#ef7484",
    max_pairs=20,
)
self.play(Write(contacts), run_time=1.5)
```

`p.electrostatics(charges=q, **options)` is equivalent. Pair energies use:

```text
Eij = 332.0637133 × qi × qj × exp(−rij / screening_length)
      / (dielectric × rij)
```

Charges are in elementary charges, distance and screening length in Å, dielectric is dimensionless, and energy is **kcal/mol**. A negative energy is attractive and a positive energy is repulsive. Set `screening_length=None` for unscreened Coulomb. The conversion constant corresponds to the conventional [Coulomb interaction](https://manual.gromacs.org/2024.3/reference-manual/functions/nonbonded-interactions.html); exponential screening is the chosen approximation.

`.pairs` lists charged-atom contacts within `cutoff` whose `abs(energy) >= min_energy`, sorted strongest first. By default it excludes same-residue pairs and directly bonded atoms. It does not implement a force field's full 1–3/1–4 exclusion/scaling rules. `.pair_energy(a, b)` computes a specified pair directly, without the display cutoff/filter. Coincident charges raise an error there and are omitted from the contact list.

These estimates are not Poisson–Boltzmann/APBS, PME, binding free energies or a complete electrostatic model. There is no automatic dielectric boundary, pH/protonation assignment, ion distribution or solvent response. Choosing charge, dielectric and screening parameters is part of preparing the visualization.

### Import prepared charges

```python
# Load the structure whose atom identities match the prepared PQR.
p = Protein.from_file("prepared.pdb", include_hydrogens=True).ball_and_stick()
field = Electrostatics.from_pqr(
    p, "prepared.pqr", dielectric=80, screening_length=8,
)

# Or use an atom-ordered array from a trajectory/force-field preparation step:
# field = Electrostatics(p, charges=charges_in_topology_order)
```

PQR charges are matched by **chain, author residue number, insertion code, residue name and atom name**, rather than line order. Missing, duplicate or nonfinite charges raise an error. Extra PQR atoms also raise an error by default so omitted hydrogen charges cannot silently disappear. `allow_extra=True` explicitly permits an intentionally selected subset. Chain-less PQR files require a matching chain-less topology. Imported radii and coordinates do not replace the displayed model; only charges are imported.

`charges_from_pqr(p, path, allow_extra=False)` returns the mapped array separately. Whitespace-delimited standard PQR records with or without a chain field are supported.

For a quick illustration, `charges="formal"` places −1 across the Asp/Glu carboxyl oxygens and +1 across the Lys/Arg charged side-chain atoms. This is the default convenience model. It omits histidine, termini, pKa effects and force-field partial charges. The example film labels these charges as illustrative; use an imported prepared model for your intended chemical state.

### Evaluate potential at points

```python
# points has shape (n, 3) in the model coordinate frame, in Å.
values = field.potential(points, softening=1.0, chunk_size=2048)
```

Potential is in **kcal/mol per elementary charge**. This sums all nonzero charges without the pair-display cutoff. It uses the softened distance `sqrt(r² + softening²)` in both the inverse-distance and screening factors. Set `softening=0` for unsmoothed points away from charges; evaluation directly at a charge then raises an error. The CPU calculation is chunked to bound its distance matrix. Potential values are returned for your own analysis or color mapping; this API does not automatically paint a surface potential texture.

## Style and animate interaction highlights

Both analyses provide `.highlight(...)` using the same `Distance` line/font options. Use `show_distances=True` on the group (the singular `show_distance` is for an individual ruler). A uniform `color` overrides attraction/repulsion colors. `label_color` can independently override text color. `follow_opacity=False` keeps lines visible when you dim the model.

`region=selection` retains contacts with **either** endpoint in that region. `max_pairs` bounds the displayed set and the reusable GPU slot pool; it defaults to 100. `.visible_pairs` and `.total_pairs` are refreshed when the annotation is evaluated. Hydrogen bonds are ordered by atom indices, electrostatic contacts by absolute energy. Contacts update during coordinate motion; crossing a detection threshold can make a line appear or disappear immediately. There is no automatic contact-lifetime smoothing.

Interactions and mesh building run on the CPU; resulting lines, text and surfaces render through Metal. Analysis results are cached while coordinates and settings remain unchanged, so rotation or styling alone does not rerun the calculation. Preprocess periodic boundaries before loading a trajectory.

## Run the example

```bash
proteinmotion render examples/molecular_tools.py InteractionsAndDistances -o interactions-and-distances.mp4 --fps 60
```

The film compares 3D and 2D hydrogen-bond rulers through ubiquitin NMR conformers, then focuses on a screened electrostatic contact using clearly labeled formal charges. [Watch it](https://pdpppd.github.io/proteinmotion/gallery/#interactions) or read the [complete script](molecular-example.md).
