# Torsions and secondary structure

Build peptides from a sequence, measure and animate backbone (φ, ψ, ω) and side-chain (χ) torsion angles, and show each residue on a live Ramachandran plot. Torsion changes rotate atoms about bonds, so bond lengths and angles stay fixed. The cartoon follows the change in secondary structure.

Requires ProteinMotion 0.14.0 or later. Upgrade with `python -m pip install --upgrade "proteinmotion>=0.14.0"`.

## Complete example and output

The example builds a 12-residue alanine peptide as an extended β strand. It turns ψ of Ala6, then moves every residue to α-helix angles, starting at the N terminus.

```python output=torsions
"""Turn one ψ angle, then wind a whole peptide into an α helix.

A 12-residue alanine peptide is built as an extended β strand. ψ of Ala6 turns to
−45° with its atoms shown over the cartoon; then every residue moves to φ = −57°,
ψ = −47°, starting at the N terminus. DSSP assigns the helix from the final
coordinates, and the cartoon blends from coil to helix as the angles change.
"""

from proteinmotion import (
    FadeIn,
    Protein,
    ProteinScene,
    RamachandranPlot,
    SetTorsions,
    Text,
    Unwrite,
    Write,
)


class TorsionBasics(ProteinScene):
    def construct(self):
        peptide = Protein.build("AAAAAAAAAAAA", "strand").cartoon()
        middle = peptide.select(residues=6)
        middle.show_atoms()
        self.add(peptide)
        self.camera.frame(peptide, margin=1.0, aspect=self.width / self.height, screen_position=(0.32, 0.55))
        self.camera.phi = 0.3
        plot = RamachandranPlot(peptide, highlight=middle, position=(0.62, 0.14), size=(0.35, 0.7))
        self.add(Text("Torsion angles", font_size=48, font="semibold", position=(0.05, 0.07)))
        marker = middle.torsion_marker("psi", radius=1.6)
        self.play(FadeIn(plot), Write(marker), run_time=1.5)
        self.play(SetTorsions(middle, psi=-45, anchor="n"), run_time=2.5)
        self.play(Unwrite(marker), run_time=0.6)
        self.play(SetTorsions(peptide, conformation="helix", stagger=0.5), run_time=5)
        self.play(self.camera.animate.orbit(0.8), run_time=2.5)
```

```bash
proteinmotion render examples/torsion_basics.py TorsionBasics --fps 60 -o torsions.mp4
```

The [torsion film](https://github.com/pdpppd/proteinmotion/blob/main/examples/torsions.py) is a longer example. It turns φ and ψ of one residue, winds Baldwin's peptide into an α helix with its hydrogen bonds, folds the protein G β-hairpin from angles measured in PDB 1PGA, and plots ubiquitin.

## Build a peptide

`Protein.build(sequence, conformation)` makes an ideal peptide with every heavy atom. The sequence uses one-letter codes for the 20 standard amino acids.

```python
peptide = Protein.build("AEAAAKEAAAKEAAAKA", "helix")
strand = Protein.build("GEWTYDDATKTFTVTE", "strand", first=41, hydrogens=True)
custom = Protein.build("AAAAAA", phi=[-60, -70, -80, -90, -100, -110], psi=-45)
```

| Conformation | φ, ψ (degrees) |
|---|---|
| `helix` or `alpha` | −57, −47 |
| `310` | −49, −26 |
| `pi` | −57, −70 |
| `strand` or `beta` | −135, 135 |
| `antiparallel`, `parallel` | −139, 135 and −119, 113 |
| `ppii` | −75, 145 |
| `extended` (default) | 180, 180 |
| `left` | 57, 47 |

`phi`, `psi`, and `omega` override the conformation. Each accepts one value or one value per residue. ω defaults to 180° (trans). Proline keeps φ = −65° because its ring fixes that bond.

Bond lengths and angles use standard values. Each side chain starts in a common rotamer. A side chain that would touch the backbone turns χ1 to the clearest of −65°, 180°, and 60°. `hydrogens=True` adds amide H atoms, which `HydrogenBonds(..., hydrogens="explicit")` can use. `chain` and `first` set the chain ID and first residue number. The chain is centered at the origin along the x axis, with the N terminus on the left.

## Measure torsions

```python
angles = protein.torsions()            # every residue
helix = protein.torsions(residues=(23, 34))
print(helix.phi, helix.psi)            # one value per residue, in degrees
print(angles["A", 28])                 # {'phi': -66.1, 'psi': -38.1, 'omega': ..., 'chi1': ...}
```

`region.torsions()` measures a selection. Values are in degrees, from −180 to 180. A torsion that a residue lacks is NaN, such as φ of the first residue or χ1 of glycine. The definitions follow IUPAC:

| Torsion | Atoms |
|---|---|
| φ | C(i−1)–N–Cα–C |
| ψ | N–Cα–C–N(i+1) |
| ω | Cα–C–N(i+1)–Cα(i+1), the peptide bond after the residue |
| χ1 … χ5 | Side-chain atoms, for example N–Cα–Cβ–Cγ for χ1 |

Torsion angles are in degrees. Scene rotations and camera angles stay in radians.

## Animate torsions

`SetTorsions` moves torsions to new values. `RotateTorsions` turns them by an amount.

```python
self.play(SetTorsions(peptide, conformation="helix", stagger=0.5), run_time=5)
self.play(SetTorsions(protein.select(residues=44), chi1=-60), run_time=1)
self.play(SetTorsions(peptide, phi={5: -60, 6: -70}, psi={5: -45}), run_time=2)
self.play(RotateTorsions(protein.select(residues=8), chi1=360), run_time=2)
```

A value can be one angle for every selected residue, a list with one angle per selected residue, or a dictionary keyed by residue number or `(chain, number)`. `conformation=` sets φ and ψ together. To change torsions at authoring time without animation, use `protein.set_torsions(...)` or `region.set_torsions(...)`.

| Option | Default | Meaning |
|---|---|---|
| `anchor` | `"center"` | The part that stays still: `"center"` keeps the middle of each chain in place, `"n"` or `"c"` a terminus. A Region is held in place as a whole. |
| `path` | `"allowed"` | φ and ψ turn the way that keeps the residue in favored and allowed Ramachandran regions. `"shortest"` takes the smaller turn. χ angles always take the smaller turn. |
| `stagger` | `None` | Fraction of the clip used to start residues one after another, N to C |
| `reverse` | `False` | Stagger from C to N |
| `secondary` | `"auto"` | Blend the cartoon toward the DSSP assignment of the final conformation; `"keep"` leaves it unchanged |
| `easing` | `"smooth"` | Easing of each torsion's change |

Atoms rotate about each torsion's bond, so the result has exact bond lengths and angles. Torsions inside rings cannot rotate: proline φ and side-chain torsions within aromatic rings are skipped, and `animation.locked` lists them. An animation in which every requested torsion is locked raises an error. Disulfides, metal contacts, and ligands bonded to several residues stay where they are; their bonds stretch when the chain moves.

The motion between the start and end angles is an illustration, not a folding pathway. A long chain swings through large arcs when torsions near one end change, and atoms can pass through each other. Use a morph or a trajectory to show motion between two experimental structures.

## Show a torsion

`TorsionMarker` draws a torsion's bond, an arc from the first atom's direction to the last atom's around that bond, and the current angle. The arc and value update as the torsion changes.

```python
residue = peptide.select(residues=6)
phi = residue.torsion_marker("phi", color="#f5d477")
self.play(Write(phi), run_time=1)
self.play(SetTorsions(residue, phi=-60), run_time=2)
```

| Option | Default | Meaning |
|---|---|---|
| `torsion` | `"phi"` | `phi`, `psi`, `omega`, or `chi1`–`chi5` |
| `radius` | `1.25` | Arc radius in Å |
| `color`, `font_size`, `precision` | gold, 30, 0 | Line and label style; decimals in the label |
| `label` | symbol | Text before the value, for example `"ψ"` |
| `backing` | `"#0b1220"` | Color of the card behind the label; `None` removes it |

The marker needs a single-residue Region. It is drawn over the image, like a 2D distance line.

## Ramachandran plots

`RamachandranPlot(protein)` plots φ against ψ for every residue that has both. Points follow the protein through torsion animations, morphs, and trajectories, and use the residue colors drawn in 3D. Glycine is a triangle and proline is a square.

```python
plot = RamachandranPlot(protein, highlight=protein.select(resname="GLY"), position=(0.62, 0.14), size=(0.35, 0.7))
self.add(plot.with_panel())
```

| Option | Default | Meaning |
|---|---|---|
| `region` | all residues | Residues to plot |
| `highlight`, `labels` | `None`, `True` | Ring, enlarge, and label a Region's residues |
| `background` | `"general"` | Reference regions: `general`, `glycine`, `proline`, `preproline`, or `None` |
| `paths` | `True` | During `SetTorsions`, draw each moving residue's path from a hollow start marker |
| `color` | residue colors | One color for every point |

The darker regions contain 98% (favored) and the lighter regions 99.8% (allowed) of residues in a reference set: one chain from each 30% sequence-identity cluster of X-ray entries at 1.4 Å resolution or better, with R-free at most 0.20. Residues with alternate locations, partial occupancy, or B factors of 30 Å² or more are excluded. Glycine, trans proline, and the residue before a proline have separate references. `scripts/build_ramachandran_reference.py` rebuilds the regions from the PDB, and `src/proteinmotion/data/ramachandran.json` lists the entries.

`plot.with_panel()` draws a solid card behind the plot so molecules that pass behind it stay hidden. Every plot, legend, and heatmap has this method.

## Secondary structure

ProteinMotion assigns helix (H), strand (E), and coil (C) with DSSP when a file has no helix or sheet records. Predicted models and MD topologies usually lack them. `Protein.from_file(path, secondary="dssp")` always computes the assignment; `secondary="file"` uses only the file's records.

```python
print(protein.secondary_structure)          # "CEEEEEECCCCEEEEECCC…"
protein.assign_secondary_structure()        # DSSP from the current coordinates
protein.set_secondary_structure("CCHHHHHHCC")
self.play(SecondaryStructure(protein), run_time=1)  # blend to DSSP of the current coordinates
```

The DSSP implementation follows Kabsch and Sander: C=O···H–N energies, n-turns, minimal helices, and β-bridges grouped into ladders with bulges. Helix includes α, 3₁₀, and π helices. Strand requires a ladder of two or more bridges, so a single extended chain is drawn as coil until it pairs with another strand. Isolated bridges, turns, and bends are coil. On nine structures (9,595 residues, including GroEL and an AlphaFold model), the assignment agreed with `mdtraj.compute_dssp` reduced to the same three states for every residue.

Cartoons blend the coil, helix, and strand cross-sections, colors, and sheet arrowheads when the assignment changes. `SetTorsions` blends each residue as its own torsions change. Use `SecondaryStructure(protein)` after a `Morph` or `PlayTrajectory` to update the cartoon to the new coordinates. `select(secondary="H")` uses the current assignment.
