# Torsions, peptides, and secondary structure

Requires ProteinMotion 0.14.0 or later. Import the names below from `proteinmotion`. Torsion angles are in degrees; scene rotations and camera angles stay in radians.

## Build a peptide

```python
pep = Protein.build("AEAAAKEAAAKEAAAKA", "strand", hydrogens=True)  # one-letter codes
hairpin = Protein.build("GEWTYDDATKTFTVTE", first=41).cartoon()
custom = Protein.build("AAAAAA", phi=[-60, -70, -80, -90, -100, -110], psi=-45)
```

Conformations: `helix`/`alpha` (−57, −47), `310`, `pi`, `strand`/`beta` (−135, 135), `parallel`, `antiparallel`, `ppii` (−75, 145), `extended` (180, 180, the default), `left` (57, 47). `phi`, `psi`, and `omega` override them, as one value or one per residue. Proline keeps φ = −65°. Side chains start in common rotamers that avoid the backbone. `hydrogens=True` adds amide H for `HydrogenBonds(..., hydrogens="explicit")`. The chain is centered on the origin along x, N terminus on the left. DSSP assigns its secondary structure.

## Measure and change torsions

```python
angles = protein.torsions(residues=(23, 34))  # TorsionAngles: .phi, .psi, .omega, .chi1 … arrays
angles["A", 28]  # dict for one residue
self.play(SetTorsions(pep, conformation="helix", stagger=0.5), run_time=6)
self.play(SetTorsions(residue, phi=-60, anchor="n"), run_time=2.5)
self.play(SetTorsions(pep, phi={5: -60, 6: -70}, psi={5: -45}), run_time=2)
self.play(RotateTorsions(side_chain_residue, chi1=120), run_time=1.5)
protein.set_torsions(conformation="ppii")  # immediate, at authoring time
```

Values are one angle for every selected residue, one per residue, or `{residue_number: angle}`. Atoms rotate about each torsion's bond: bond lengths and angles stay exact. `anchor="center"` (default) keeps each chain's middle still; `"n"`/`"c"` keep a terminus still, which suits "turning φ swings everything after it"; a Region anchor is held in place as a whole. `path="allowed"` (default) turns φ/ψ through allowed Ramachandran regions (extended → helix goes through β and the bridge); `"shortest"` takes the smaller turn. `stagger` (clip fraction) starts residues N to C, `reverse=True` C to N. Ring torsions (proline φ, aromatic rings) cannot rotate; `animation.locked` lists skipped ones. Disulfides and multi-residue ligand bonds stretch.

Concurrent `SetTorsions` on the same protein both write geometry; combine them in one call with dictionaries. Torsion paths are illustrations, not folding pathways; large changes near one end swing long arms and atoms can pass through each other. Prefer short peptides for single-bond demonstrations, fold hairpins from an extended chain (not from a helix), and turn the result upright before showing it, for example with `protein.animate.rotate(angle, axis)` toward the helix axis.

## Show torsions

```python
residue = pep.select(residues=3)
phi = residue.torsion_marker("phi", color="#f5d477", radius=1.5, font_size=34)
psi = residue.torsion_marker("psi", color="#50e0d0", radius=1.5, font_size=34)
plot = RamachandranPlot(pep, highlight=residue, position=(0.62, 0.16), size=(0.35, 0.68)).with_panel()
self.add(plot)
self.play(Write(phi), run_time=1.5)
```

`TorsionMarker` draws the bond, an arc around it, and the live value on a dark card (`backing=None` removes the card). It needs a single-residue Region. `RamachandranPlot` points follow any motion and use the residue colors; Gly is a triangle, Pro a square. `highlight` rings and labels residues. During `SetTorsions` it shows each moving residue's path and start. `background="glycine"`, `"proline"`, `"preproline"`, or `None` changes the reference regions (98% favored, 99.8% allowed, from ≤1.4 Å PDB structures). `.with_panel()` hides molecules passing behind any plot. Leave the right ~38% of a 16:9 frame for the plot and frame the molecule with `screen_position=(0.31, 0.56)`.

## Secondary structure

Files without helix/sheet records (predicted models, MD topologies) are assigned by DSSP automatically. `Protein.from_file(path, secondary="dssp")` always computes it; `"file"` never does. `protein.secondary_structure` is the current H/E/C string; `set_secondary_structure(codes)` and `assign_secondary_structure()` change it at authoring time. `SetTorsions(..., secondary="auto")` blends the cartoon to the final DSSP assignment; after `Morph` or `PlayTrajectory`, play `SecondaryStructure(protein)`. DSSP marks strand only for paired strands, so a lone extended chain is coil.

The website has a complete guide with rendered output: [Torsions and secondary structure](https://pdpppd.github.io/proteinmotion/docs/torsions/). `examples/torsions.py` is a full film.
