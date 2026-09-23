# Annotations and interactions

## Selections, focus and regions

```python
helix = p.select(chain="A", residues=(23, 34))  # Inclusive PDB author residue numbers.
ends = p.select(residues=[23, 34], atoms="CA")  # An explicit list, not a range.
backbone = p.select(residues=(23, 34), atoms=["N", "CA", "C", "O"])
```

Select bound groups and their surroundings by category, name, or distance:

```python
ions = p.select(ions=True)  # Single-atom residues outside the traced chains.
ligand = p.select(resname="ATP")  # Also: ligands=True, water=True (with include_water=True).
pocket = p.select(within=4.0, of=ligand)  # Whole residues; excludes the ligand itself.
```

Inspect the input first; chain IDs and residue numbers are not assumed contiguous or zero-based. `Region.atom_indices` refers to topology atom indices. Selections follow their protein through motion. To dim everything outside a selection:

```python
rest = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), helix.atom_indices))
self.play(
    SetOpacity(rest, 0.08), Focus(self.camera, helix, margin=1.8, aspect=self.width / self.height), run_time=2
)
box = helix.highlight(style="box", color="#50e0d0", opacity=0.5, padding=1.0)
self.play(FadeIn(box), run_time=1)
```

Other highlight styles are `sphere` and `atoms`. Atom halos are effective for a few selected Cαs; a box or sphere suits a larger region. Avoid opaque highlights that hide the selected structure.

## Text, residue names and leaders

```python
title = Text("A closer view", font_size=58, font="semibold", position=(0.06, 0.08))
note = helix.callout(
    "α helix", subtitle="Residues 23–34", position=(0.07, 0.42), font_size=38, color="#50e0d0"
)
labels = p.label_residues(residues=[23, 34], font_size=30, offsets={23: (230, -60), 34: (230, 60)})
self.play(Write(title), Write(note), run_time=1.5)
self.play(Write(labels, lag_ratio=0.05), run_time=1.5)
self.play(Unwrite(labels), Unwrite(note), run_time=1)
```

Positions are normalized screen coordinates with a top-left origin. Font sizes, residue-label offsets, and 2D line widths are **1080p design pixels**, scaled by `height / 1080` in the output. For example, `font_size=60` becomes 20 output pixels at 360p; to obtain 30 output pixels there, use 90. Choose sizes for the intended viewing resolution. Labels derive actual residue names from topology. `Write` draws glyph contours then fills them. `FadeIn`/`FadeOut` are simpler choices for secondary captions. Use one or two readable labels in a close-up instead of labeling every residue.

## Distance rulers

```python
a, b = p.select(residues=23, atoms="CA"), p.select(residues=34, atoms="CA")
ruler = Distance(
    a, b, mode="3d", color="#f2ba67", style="dashed", radius=0.09, font_size=32, precision=1, dash_count=12
)
self.play(Write(ruler), run_time=1.5)
```

`mode="3d"` draws a depth-tested model line. `mode="2d"` draws an overlay; use `line_width` in 1080p design pixels instead of model-space `radius`. The measured distance sits in a central gap and updates as coordinates move. Select one atom at each end for an unambiguous atom distance. For whole residues, use `anchor="ca"` (default) or explicitly request `anchor="centroid"`.

## Hydrogen bonds

```python
hb = HydrogenBonds(
    p,
    donors=p.select(residues=(27, 34), atoms="N"),
    acceptors=p.select(residues=(23, 30), atoms="O"),
    hydrogens="backbone",
    max_distance=3.5,
    min_angle=150,
)
network = hb.highlight(
    mode="3d", color="#f2ba67", radius=0.08, endpoints="donor_acceptor", show_distances=False, max_pairs=30
)
self.play(Write(network), run_time=1.5)
```

Inspect `hb.pairs`; a visually plausible helix need not pass every geometric cutoff. Do not invent contacts to produce an expected count. `hydrogens="backbone"` infers missing amide H for angle tests; label that assumption. `endpoints="donor_acceptor"` draws N···O. `endpoints="hydrogen_acceptor"` draws H···O and changes optional distance labels accordingly. For explicit H, load with `include_hydrogens=True` and use `hydrogens="explicit"`. Missing side-chain hydrogens are not inferred.

Keep the network legible by focusing a short backbone section and dimming the rest. Avoid a number on every short H-bond; annotate one selected pair or show a separate longer ruler.

## Electrostatics and charge import

```python
charges = np.load(charge_file, allow_pickle=False)  # One charge in e per displayed atom.
field = Electrostatics(p, charges=charges, dielectric=80, screening_length=8, cutoff=8, min_energy=0.05)
contacts = field.highlight(
    mode="2d", region=selected_region, max_pairs=8, style="dashed", show_distances=True
)
self.play(Write(contacts), run_time=1.5)
```

Charge arrays must match displayed atom order, not merely have the same length. Use `Electrostatics.from_pqr(p, path, ...)` for identity-mapped PQR charges. `charges="formal"` supplies illustrative side-chain formal charges, not a force-field assignment. State the charge source and model parameters. This is a screened-Coulomb approximation with energies in kcal/mol, not a Poisson–Boltzmann/APBS calculation. An interaction energy is not a binding free energy. Recompute/update a static energy caption if geometry changes; rigid rotation alone leaves it unchanged.
