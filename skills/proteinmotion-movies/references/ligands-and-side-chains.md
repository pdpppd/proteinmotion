# Ligands, ions, and side chains

Requires ProteinMotion 0.11.0 or later. Atoms can be drawn as ball-and-stick over a cartoon, ribbon, or surface. Use this to show a bound ligand, metal ions, the residues that contact them, or any side chains the story needs.

## What cartoons show by default

A residue belongs to a traced chain when the cartoon draws it: two or more amino acids joined through Cα, or nucleotides joined along the backbone. Every other residue draws as ball-and-stick over a cartoon or ribbon without any code:

| Category | Rule | Examples |
|---|---|---|
| `ligand` | Outside the traced chains; not water or a single-atom ion | HEM, ATP, spermine, and also crystallization additives such as EOH, GOL, SO4 |
| `ion` | One heavy atom that is not C, N, or O | Ca²⁺, Mg²⁺, Zn²⁺, Mn²⁺, Cd²⁺, Na⁺, Cl⁻ |
| `water` | HOH, WAT, SOL, and common MD names | Only present when loaded with `include_water=True` |

Inspect `p.topology.residue_categories` (one string per residue: `polymer`, `ligand`, `ion`, `water`) before planning the storyboard. Additives often distract: hide them before `add`, for example `p.select(resname="EOH").hide_atoms()`. `p.hide_atoms()` removes all default detail. MD selections default to `"protein or nucleic"`, so solvent is not loaded unless requested.

## Select by category, name, or distance

```python
ions = p.select(ions=True)
ligand = p.select(resname="ATP")  # One name or a list; case-insensitive.
bound = p.select(ligands=True, ions=True)  # Categories combine with each other.
calcium = p.select(ions=True, residues=149)  # Other filters must all match.
site = p.select(within=3.0, of=calcium)  # Whole residues, excluding the reference itself.
backbone_near = p.select(within=5.0, of=ions, atoms=["P", "OP1", "OP2"])
```

- `within` measures the current coordinates once, when the selection is made, in world space. `of` can be a region of another protein object; its own residues are only excluded when it belongs to the same protein.
- Report `within` as a distance criterion (for example "residues within 3 Å of the ion"), not as a proven interaction or binding energy.
- Metal sites: 3.0 Å finds direct coordination. Some ions bind only through water, which is usually not loaded; use 5 Å and say so, or check the neighbors first.
- A ligand made of several deposited residues, or several copies of a ligand, needs `chain`/`residues` to pick one copy. A callout on a multi-copy selection points at their average position.

## Show, hide, and animate atoms

Initial state, before `add`:

```python
p.select(ions=True).hide_atoms()
site.side_chains().show_atoms()  # Region of side-chain atoms plus Cα.
```

Animations, inside `play`:

```python
self.play(ShowSideChains(site, residue_delay=0.2), run_time=2)
self.play(ShowAtoms(ligand), run_time=1)
self.play(HideSideChains(site, reverse=True), HideAtoms(p.select(resname="SO4")), run_time=1.5)
```

- `ShowSideChains`/`HideSideChains` include Cα so sticks join the cartoon; proline keeps N to close its ring. Glycine, nucleotides, ligands, and ions are skipped. They accept a Region or a whole Protein.
- `ShowAtoms`/`HideAtoms` take any atoms, including whole nucleotides or phosphate groups.
- All four accept `residue_delay`, `reverse`, and `easing`, like `SetOpacity`. Keep the default linear clip clock; `run_time` must exceed `(residue_count - 1) * residue_delay`. `region.animate.show_atoms()` is equivalent.
- Two animations cannot change the same atoms in one `play`. Color, opacity, camera, and trajectory animations can run alongside.
- A bond appears only when both of its atoms are shown.

## Appearance

- Heteroatoms use element colors; side-chain carbons use the cartoon color of their residue. Ligand carbons are light green with the default `secondary`/`base` palettes. `Colorize` and `SetOpacity` tint and fade these atoms like the rest of the model.
- Ions are drawn 1.3× larger than a normal atom of the same element. Radii use the protein's `atom_scale` and `bond_radius`; set them with `ball_and_stick(atom_scale=..., bond_radius=...).cartoon()` before `add`.
- Bonds come from covalent radii, so an ion close to a coordinating oxygen is joined to it by a stick once both are shown. Describe these as coordination contacts, not covalent bonds.
- `Representation(p, "ball_and_stick")` draws every atom and blends colors into the ball-and-stick palette; returning to `"cartoon"` keeps the atoms that were shown.
- Detail atoms follow trajectories, morphs, and transforms, and they render with EEVEE. A long `residue_delay` creates many simultaneous opacity levels; EEVEE may need a higher `EEVEEOptions(max_opacity_layers=...)`.
- Molecular surfaces still enclose ligand atoms, so a ligand inside its own surface is hidden. Prefer a cartoon, or a translucent surface, for pocket shots.

## Patterns that read well

Pocket reveal: fade residues far from the ligand, focus on the pocket, then grow the side chains.

```python
ligand = p.select(ligands=True, resname="ADP")
pocket = p.select(within=4.0, of=ligand)
shell = p.select(within=8.0, of=ligand) | ligand
distant = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), shell.atom_indices))
self.play(
    Focus(self.camera, pocket | ligand, margin=1.2, aspect=self.width / self.height),
    SetOpacity(distant, 0.2),
    run_time=2,
)
note = ligand.callout("ADP", subtitle="Nucleotide pocket", position=(0.70, 0.26), color="#50e0d0")
self.play(Write(note), ShowSideChains(pocket, residue_delay=0.1), run_time=2.5)
```

- Metal sites: focus on `site | ion` with `margin` about 1.2–1.4, show the side chains with a 0.1–0.25 s delay, then label the ion with a callout. Several residue labels crowd a metal site; prefer one callout per ion or explicit label `offsets`.
- Nucleic acids: `BaseStyle(dna, "none")` alongside `ShowAtoms(contacts)` replaces base slabs with atoms, so the two do not overlap. Thin `bases="ladder"` rods keep ion and phosphate detail readable on a large RNA.
- Ensembles: side chains move with `PlayTrajectory`; `ShowSideChains(p, residue_delay=0.03)` grows every side chain N to C during playback.
- Move the camera to a second object with an animated `Focus` in `play`, not `camera.frame()` in the middle of `construct()`; direct camera changes after the first `play` are not part of the timeline.

## Complete examples

Scripts in the repository, each with a video in the [gallery](https://pdpppd.github.io/proteinmotion/gallery/):

- `examples/binding_sites_film.py`: calmodulin Ca²⁺ sites (1CLL), then a GroEL ADP/Mg²⁺ pocket (1AON chain A).
- `examples/troponin_sites.py`: two Cd²⁺ EF-hand sites and a sulfate on Arg47 (1NCX), with a representation round trip.
- `examples/side_chain_ensemble.py`: ubiquitin side chains across the 2K39 NMR conformers.
- `examples/nucleic_ions.py`: ions on tRNA (1EHZ) and spermine on Z-DNA (2DCG).
- `examples/ligands_and_side_chains.py`: a short calmodulin example, explained in the [guide](https://pdpppd.github.io/proteinmotion/docs/ligands-and-side-chains/).
