# Ligands, ions, and side chains

Cartoons and ribbons draw the traced protein and nucleic acid chains. Atoms outside those chains, such as bound ligands, metal ions, and loaded waters, are drawn as ball-and-stick models over the cartoon by default. You can also show individual atoms or amino acid side chains in the same way, and animate them in or out residue by residue.

## Complete example and output

This script uses calmodulin, PDB 1CLL. The four Ca²⁺ ions appear with the cartoon. The side chains within 3 Å of the first ion then grow from the cartoon.

```python output=ligands-and-side-chains
"""Show 1CLL Ca²⁺ ions on the cartoon and grow the side chains that coordinate them."""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, ShowSideChains, Unwrite, Write

DATA = Path(__file__).parent / "data"


class LigandsAndSideChains(ProteinScene):
    def construct(self):
        # Ions and ligands outside the traced chain draw as ball-and-stick detail.
        p = Protein.from_file(DATA / "1cll.cif").cartoon().center()
        self.add(p)
        self.camera.frame(p, margin=1.0, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi = 0.4, 0.15
        self.camera.depth_cue = 0.3
        self.play(self.camera.animate.orbit(0.4), run_time=1.5)

        # Whole residues with an atom within 3 Å of the first Ca²⁺ ion.
        ion = p.select(ions=True, residues=149)
        site = p.select(within=3.0, of=ion)
        self.focus(site | ion, margin=1.3, run_time=1.5)
        self.play(ShowSideChains(site, residue_delay=0.2), run_time=1.8)
        labels = p.label_residues(
            residues=[20, 22, 24, 26, 31],
            font_size=30,
            color="#f2ba67",
            offsets={20: (-260, 90), 22: (-300, -40), 24: (-120, -170), 26: (260, -130), 31: (300, 70)},
        )
        self.play(Write(labels, lag_ratio=0.1), run_time=1.2)
        self.play(self.camera.animate.orbit(0.5), run_time=2.5)
        self.play(Unwrite(labels), run_time=0.6)

        # Every residue within 3 Å of any of the four ions, staggered N to C.
        self.focus(p, margin=1.0, run_time=1.5)
        self.play(
            ShowSideChains(p.select(within=3.0, of=p.select(ions=True)), residue_delay=0.04),
            self.camera.animate.orbit(0.5),
            run_time=2.5,
        )
```

[Download the script](ligands_and_side_chains.py). Run it from a repository checkout, which includes `examples/data/1cll.cif`:

```bash
proteinmotion render examples/ligands_and_side_chains.py LigandsAndSideChains --fps 60 -o ligands.mp4
```

## Select ligands, ions, and nearby residues

`protein.select()` accepts residue categories, residue names, and a distance:

```python
ions = p.select(ions=True)
ligand = p.select(resname="ATP")
bound = p.select(ligands=True, ions=True)
pocket = p.select(within=4.0, of=ligand)
```

| Option | Selects |
|---|---|
| `ligands=True` | Residues outside the traced chains, except water and single-atom ions |
| `ions=True` | Single-atom residues outside the traced chains, except carbon, nitrogen, and oxygen |
| `water=True` | Water residues such as HOH, WAT, and SOL. Load crystal waters with `include_water=True` |
| `resname="HEM"` | Residues with that name; a list selects several names. Names are not case-sensitive |
| `within=4.0, of=region` | Whole residues with at least one atom within 4 Å of `region` |

`ligands`, `ions`, and `water` combine with each other: `ligands=True, ions=True` selects both. All other options must match together, as in `p.select(chain="B", ions=True)`.

`within` measures the coordinates once, when you make the selection. It uses world coordinates, so `of` can be a region of another protein object. The residues of `of` itself are left out when it belongs to the same protein. The distance is a geometric criterion; it does not identify a chemical interaction.

A residue belongs to a traced chain when the cartoon draws it: two or more amino acids joined through Cα, or nucleotides joined along the backbone. Other residues, including a nucleotide ligand such as ATP, count as ligands.

## Show side chains

`ShowSideChains(target)` adds the side chains of the target's amino acids. Each side chain includes Cα, where the cartoon passes, so the sticks join the cartoon. Proline also keeps its backbone N to close the ring. Glycine, nucleotides, ligands, and ions are skipped.

```python
site = p.select(within=3.0, of=p.select(ions=True, residues=149))
self.play(ShowSideChains(site, residue_delay=0.2), run_time=2)
self.play(HideSideChains(site, reverse=True), run_time=1)
```

With `residue_delay`, residues start in N-to-C order, or C-to-N order with `reverse=True`. `run_time` must be longer than the total delay. `easing` sets the transition of each residue, as for `SetOpacity`. A bond appears once both of its atoms are visible.

Use `region.side_chains()` to get the same atoms as a region, for example to color them.

## Show or hide any atoms

`ShowAtoms` and `HideAtoms` work on any protein or region:

```python
self.play(ShowAtoms(p.select(residues=(20, 31))), run_time=1)   # Backbone and side chains
self.play(HideAtoms(p.select(ligands=True)), run_time=1)        # Remove a ligand
```

To set the state before the first animation, use `region.show_atoms()`, `region.hide_atoms()`, `protein.show_atoms()`, or `protein.hide_atoms()`. `protein.hide_atoms()` also removes the default ligands and ions.

These atoms are drawn over cartoons, ribbons, and molecular surfaces. They follow trajectories, morphs, and transforms, and they use the protein's `atom_scale` and `bond_radius`. `Colorize` and `SetOpacity` apply to them as to the rest of the model. `Representation(p, "ball_and_stick")` draws every atom, and a transition back to a cartoon keeps the atoms you selected.

## Colors and sizes

Over a cartoon, oxygen, nitrogen, sulfur, and other heteroatoms use element colors. Side-chain carbons use the cartoon color of their residue. Ligand carbons are light green with the default `secondary` and `base` palettes. As a representation changes to ball-and-stick, the colors blend into that representation's palette.

Single-atom ions are drawn 1.3 times larger than other atoms of the same element, so that they remain visible next to the cartoon.

Bonds are inferred from covalent radii. Metal ions within bonding distance of an atom, such as Ca²⁺ and a coordinating oxygen, are joined by a stick when both atoms are shown.

## More examples

The [video gallery](https://pdpppd.github.io/proteinmotion/gallery/) includes four longer films with their scripts:

- [Calcium sites and a nucleotide pocket](https://pdpppd.github.io/proteinmotion/gallery/#binding-sites): calmodulin Ca²⁺ sites, then the ADP and Mg²⁺ pocket of a GroEL subunit (PDB 1AON, chain A). Residues far from the nucleotide are faded with `SetOpacity`.
- [Troponin C metal sites](https://pdpppd.github.io/proteinmotion/gallery/#troponin-sites): two Cd²⁺ EF-hand sites, a sulfate held by Arg47, and a change to ball-and-stick and back.
- [Side chains in an NMR ensemble](https://pdpppd.github.io/proteinmotion/gallery/#side-chain-ensemble): side chains follow the interpolated 2K39 ubiquitin conformers.
- [Ions and ligands on nucleic acids](https://pdpppd.github.io/proteinmotion/gallery/#nucleic-ions): Mg²⁺ and Mn²⁺ on transfer RNA, then spermine across a Z-DNA duplex.
