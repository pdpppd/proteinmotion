# DNA and RNA movies

Requires ProteinMotion 0.10.0 or later. Inspect chain IDs, author residue numbers, modified residues, atom names, and state counts before writing a scene. For a protein–nucleic acid complex, use `Protein.from_file()`; it draws both polymer types. `NucleicAcid.from_file()` uses the base color palette by default and requires at least one nucleotide.

```python
from pathlib import Path
from proteinmotion import (
    NucleicAcid,
    ProteinScene,
    BaseStyle,
    Colorize,
    SetOpacity,
    Representation,
    Write,
)


class DNAMovie(ProteinScene):
    def construct(self):
        # Example selections below are for PDB 1BNA.
        dna = NucleicAcid.from_file(Path(__file__).parent / "1bna.cif").cartoon(
            bases="slabs",
            color="base",
            backbone_radius=0.38,
            base_thickness=0.36,
            base_radius=0.16,
        )
        self.add(dna)
        self.camera.frame(dna, aspect=self.width / self.height)
        self.camera.set_focus(dna, chain="A", residues=(5, 8), fstop=5.6)
        site = dna.select(chain="A", residues=(5, 8))
        self.play(BaseStyle(dna, "rings"), run_time=2)
        self.play(
            Colorize(site, "#ffd47b", residue_delay=0.1),
            SetOpacity(dna.select(chain="B"), 0.2),
            run_time=2,
        )
        self.play(Write(site.callout("Selected bases")), run_time=1.5)
        self.play(Representation(dna, "surface"), run_time=2)
        self.wait(2)
```

`slabs` are rounded plates fitted to each base. `rings` are filled base outlines. `sticks` show covalent bonds within the bases. `ladder` uses schematic rods directed toward the bases. `none` shows the backbone alone. Ladder rods do not identify hydrogen bonds or base pairs. Use `BaseStyle` for transitions and `set_bases` for initial setup. `ribbon(bases="rings")` and `ball_and_stick(color="element")` are also available.

Use `base`, `chain`, `rainbow`, or a fixed color for the initial palette. Surface and ball-and-stick views also accept `element`. `Colorize`, `ColorByProperty`, `SetOpacity`, and residue delays apply across representations. Configure lens focus before the timeline, then animate it with `FocusPull`. EEVEE uses the same base meshes as the native renderer.

Nucleotide labels, sequence tracks, contact maps, B factors, and RMSF use C4′ backbone anchors, falling back to P for coarse models. Atom selectors accept prime or star names. Modified bases retain their deposited names; parent-base metadata supplies their palette and one-letter labels. Check warnings for incomplete rings, which use sticks instead. Models with only P atoms have a backbone without inferred bases.

Use multi-model files or `Trajectory.from_mdanalysis(...)` for playback. The MD reader selects `protein or nucleic` by default; use `selection="nucleic"` to restrict it. All states need matching atom identities. Slabs and filled rings follow a rigid plane through three ring atoms. Sticks follow individual coordinates. Same-topology `Morph` uses C1′ for rigid alignment. Contact-matched `BackboneMorph` also uses C1′ for DNA/RNA contact maps, alignment, and whole-residue motion. Select one strand per endpoint. The configurable `residue_delay` follows deposited order, normally 5′ to 3′; unmatched residues fade. Automatic matching skips missing C1′ atoms, and explicit mappings require them. `C1*` names are accepted. The displayed trace continues to use C4′.

For hydrogen bonds between standard bases, load explicit H coordinates and choose `hydrogens="explicit"`. Modified bases need explicit donor/acceptor selections. Supply imported atomic charges for DNA/RNA electrostatics. Contact-map distances show backbone proximity rather than base-pair assignments.

The repository examples use PDB 1BNA (DNA) and 1EHZ (yeast tRNA). Keep the actual input structures beside delivered scripts and state whether motion comes from deposited models, MD, or a visual deformation.
