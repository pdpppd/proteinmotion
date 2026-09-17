# Distances and interactions

The rendered excerpts use [docs_examples.py](docs_examples.py). Its `ubiquitin()` helper loads PDB 1UBQ, and `frame()` adds the model and sets the camera. Run each excerpt inside `construct()`; the full script includes the imports and setup. Run the full script from a repository checkout, which includes the input structures.

Draw a line between two atoms or residues and label its distance. Use the same line styles to highlight hydrogen bonds and electrostatic contacts. Measurements update as the coordinates change.

## Measure between residues or atoms

The `helix_view()` helper selects residues 23–34, dims the rest of ubiquitin, and frames the selected helix.

```python output=distances
p = helix_view(self)
a = p.select(residues=23, atoms="CA")
b = p.select(residues=34, atoms="CA")
for mode in ("3d", "2d"):
    ruler = Distance(
        a,
        b,
        mode=mode,
        font_size=42,
        color="#50e0d0",
        prefix="Cα · ",
    )
    title = Text(f"{mode.upper()} line", font_size=48)
    self.play(Write(ruler), Write(title), run_time=1.5)
    self.play(self.camera.animate.orbit(0.35), run_time=2)
    self.play(FadeOut(ruler), FadeOut(title), run_time=0.5)
```

A whole single-residue selection uses its Cα by default, if one exists. A one-atom selection always uses that atom. `anchor="centroid"` uses the mean of the selected coordinates; selections spanning multiple residues also use their centroid. Set `unit="nm"` to convert the displayed distance from ångströms to nanometers, or `unit=""` for no suffix. The `.distance` property remains in ångströms for model space.

| Option | Default | Meaning |
|---|---|---|
| `mode` | `"3d"` | 3D lines that atoms can hide, or `"2d"` lines drawn over the image |
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

The distance appears at the line’s midpoint and updates as the atoms move. In 3D mode, atoms and surfaces can hide the line. In 2D mode, the line is drawn over the protein image. The text is drawn over the image in both modes.

A ruler is hidden when either endpoint leaves the image. Check label placement when using several rulers; captions can overlap.

Use `space="world"` for endpoints on different protein objects. This includes object translation, rotation, and scale. Interpret the result as a physical distance only when the objects share a coordinate frame and scale. The default `space="model"` measures within one protein’s coordinates, independently of display transforms.

`Write`, `Unwrite`, `FadeIn`, `FadeOut` and `.animate.set_opacity()` work on a ruler. The line draws before the text fills. [Text animation options](text.md).

## Alpha-helix diagnostic

The [alpha-helix example](alpha-helix.md) shows all 12 expected i→i+4 hydrogen bonds in an idealized backbone. Visible hydrogens and separate H···O and N···O measurements show which atoms each line connects.

## Compute hydrogen bonds

```python output=hydrogen-bonds
p = helix_view(self)
hb = HydrogenBonds(
    p,
    donors=p.select(residues=(23, 34), atoms="N"),
    max_distance=3.5,
    min_angle=150,
    hydrogens="backbone",
)
lines = hb.highlight(
    mode="3d",
    color="#f2ba67",
    radius=0.07,
    dash_count=5,
    show_distances=False,
)
self.play(Write(lines), run_time=2)
self.play(self.camera.animate.orbit(0.6), run_time=3)
self.wait(1)
```

`p.hydrogen_bonds(**options)` is equivalent. `.pairs` returns immutable `Interaction` records: zero-based donor/acceptor atom indices `a`/`b`, donor–acceptor distance in Å, D–H–A angle in degrees, an explicit hydrogen index when available, and an `inferred_hydrogen` flag. The default labels show **donor–acceptor** distance. Set `hb.highlight(endpoints="hydrogen_acceptor", ...)` to draw H···acceptor lines and label that distance instead; `Interaction.distance` remains donor–acceptor distance. `hb.hydrogen_position(pair)` returns the explicit or virtual H position in model Å.

A pair passes detection when `1.5 Å < D–A ≤ max_distance` and `D–H–A ≥ min_angle`. The defaults are **3.5 Å and 150°**. Donors and acceptors must belong to different residues. If several hydrogens qualify for one pair, the detector keeps the most linear angle. Detection uses geometry only. The definitions follow [MDAnalysis hydrogen-bond analysis](https://docs.mdanalysis.org/2.9.0/documentation_pages/analysis/hydrogenbonds.html), with the distance cutoff given here.

### Explicit and missing hydrogens

| `hydrogens` | Behavior |
|---|---|
| `"explicit"` | Only hydrogens present in the loaded coordinates |
| `"auto"` | Explicit hydrogens first; infer a backbone amide H when none is available |
| `"backbone"` | Use only inferred backbone amide hydrogens |

Load explicit H with `Protein.from_file(path, include_hydrogens=True)`. Explicit hydrogens are assigned to the nearest candidate donor in the same residue within `donor_h_cutoff=1.3 Å` (and farther than 0.4 Å). Adjust that cutoff for longer donor–H bonds, such as sulfur–H.

An inferred backbone H is placed 1.01 Å from N along the outward bisector of the bonds to the preceding carbonyl C and the residue’s Cα. This construction requires a connected peptide backbone. It skips proline, terminal or missing geometry, and invalid bond lengths.

Inference is limited to backbone amide hydrogens. Supply prepared coordinates for protonation-dependent analyses and side-chain hydrogens.

The default donor and acceptor templates cover common standard amino-acid groups. For histidine tautomers, unusual protonation, ligands, water, or nonstandard residues, supply explicit hydrogens and suitable `donors=` and `acceptors=` selections. These can be `Region` objects or integer atom-index arrays. A missing contact can reflect missing atoms or a geometry cutoff.

## Screened Coulomb electrostatics

```python output=charge-contacts
p = ubiquitin().ball_and_stick()
field = Electrostatics(
    p,
    charges="formal",
    dielectric=80,
    screening_length=8,
    cutoff=6,
    min_energy=0.08,
)
pair = field.pairs[0]
endpoints = Region(p, [pair.a, pair.b])
residues = [p.topology.atoms[i].resid for i in (pair.a, pair.b)]
selected = p.select(residues=residues)
other = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), selected.atom_indices))
other.set_opacity(0.06)
frame(self, p, endpoints, margin=4.5)
contacts = field.highlight(
    mode="2d",
    max_pairs=1,
    show_distances=True,
    font_size=42,
    style="solid",
    line_width=2,
)
self.play(Write(contacts), run_time=1.5)
self.play(self.camera.animate.orbit(0.5), run_time=3)
self.wait(1)
```

`p.electrostatics(charges=q, **options)` is equivalent. Pair energies use:

```text
Eij = 332.0637133 × qi × qj × exp(−rij / screening_length)
      / (dielectric × rij)
```

Charges are in elementary charges, distance and screening length in Å, dielectric is dimensionless, and energy is **kcal/mol**. A negative energy is attractive and a positive energy is repulsive. Set `screening_length=None` for unscreened Coulomb. The conversion constant corresponds to the conventional [Coulomb interaction](https://manual.gromacs.org/2024.3/reference-manual/functions/nonbonded-interactions.html); exponential screening is the chosen approximation.

`.pairs` lists charged-atom contacts within `cutoff` whose `abs(energy) >= min_energy`, sorted strongest first. By default it excludes same-residue pairs and directly bonded atoms. It does not implement a force field's full 1–3/1–4 exclusion/scaling rules. `.pair_energy(a, b)` computes a specified pair directly, without the display cutoff/filter. Coincident charges raise an error there and are omitted from the contact list.

This model estimates pairwise interactions for supplied charges in a uniform dielectric with exponential screening. Prepare the charges, dielectric, and screening length for the visualization. Calculations that require solvent boundaries, ion distributions, or binding free energies need a separate electrostatics method.

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

PQR charges are matched by **chain, author residue number, insertion code, residue name, and atom name**. Missing, duplicate, or nonfinite charges raise an error. Extra PQR atoms also raise an error by default. Use `allow_extra=True` when you intentionally load a subset.

A PQR file with no chain field requires a topology with empty chain IDs. The import reads charges only; the displayed coordinates and radii come from the protein model.

`charges_from_pqr(p, path, allow_extra=False)` returns the mapped array separately. Whitespace-delimited standard PQR records with or without a chain field are supported.

`charges="formal"` is the default example charge model. It distributes −1 over Asp/Glu carboxyl oxygens and +1 over Lys/Arg charged side-chain atoms. It is limited to those groups. Use prepared charges when histidine, termini, protonation, or force-field partial charges matter.

### Evaluate potential at points

```python
# points has shape (n, 3) in the model coordinate frame, in Å.
values = field.potential(points, softening=1.0, chunk_size=2048)
```

Potential is in **kcal/mol per elementary charge**. The calculation sums all nonzero charges and uses `sqrt(r² + softening²)` in the distance and screening terms. `softening=0` gives the unsmoothed value away from charges; evaluation directly at a charge then raises an error.

The CPU calculation processes points in chunks to limit matrix size. The returned array can be used for analysis or for a color map you define.

## Style and animate interaction highlights

Both analyses provide `.highlight(...)` using the same `Distance` line/font options. Use `show_distances=True` on the group (the singular `show_distance` is for an individual ruler). A uniform `color` overrides attraction/repulsion colors. `label_color` can independently override text color. `follow_opacity=False` keeps lines visible when you dim the model.

`region=selection` includes contacts with either endpoint in the region. `max_pairs` limits the displayed contacts and allocated GPU slots; its default is 100. `.visible_pairs` and `.total_pairs` update when the annotation is evaluated.

Hydrogen bonds are ordered by atom index. Electrostatic contacts are ordered by absolute energy. A line appears or disappears as a contact crosses the detection threshold.

Interaction calculations and surface construction run on the CPU. Metal renders the resulting lines, text, and surfaces. Results are cached until coordinates or analysis settings change. Prepare whole, unwrapped molecules before loading a periodic trajectory.

## Run the example

```bash
proteinmotion render examples/molecular_tools.py InteractionsAndDistances -o interactions-and-distances.mp4 --fps 60
```

The example compares 3D and 2D hydrogen-bond lines through ubiquitin NMR conformations, then focuses on an electrostatic contact calculated from example formal charges. [Watch the video](https://pdpppd.github.io/proteinmotion/gallery/#interactions) or read the [complete script](molecular-example.md).
