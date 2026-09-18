# DNA and RNA

Load DNA or RNA with `NucleicAcid.from_file()`. It uses the same scene, camera, selection, color, opacity, and trajectory controls as `Protein`. For a protein–DNA or protein–RNA complex, use `Protein.from_file()` to draw both polymer types together.

These features are included in v0.10.0. Follow the [installation guide](getting-started.md), then download a script and its input structures below. The repository also includes the structures under `examples/data/`.

## Base styles

| Style | Appearance |
| --- | --- |
| `slabs` | Rounded plates fitted to each base plane. |
| `rings` | Filled ring outlines, with distinct purine and pyrimidine shapes. |
| `sticks` | Covalent bonds within each base. |
| `ladder` | A simple rod from each backbone anchor toward its base. |
| `none` | Backbone only. |

```python
from proteinmotion import NucleicAcid, BaseStyle, Representation

dna = NucleicAcid.from_file("dna.cif").cartoon(
    bases="slabs", color="base", backbone_radius=0.38,
    base_thickness=0.36, base_radius=0.16,
)
self.add(dna)
self.camera.frame(dna)
self.play(BaseStyle(dna, "rings"), run_time=1.5)
self.play(Representation(dna, "surface"), run_time=1.5)
self.play(Representation(dna, "cartoon", bases="ladder"), run_time=1.5)
```

Dimensions are in ångströms. `BaseStyle` changes the bases while keeping the backbone visible. `set_bases()` sets the initial style and dimensions before the timeline begins. `ribbon(bases="rings")` combines a ribbon with bases. `ball_and_stick(color="element")` displays every loaded atom and covalent bond.

## Colors, opacity and labels

The `base` palette assigns colors to A, C, G, T, U, and I. `chain`, `rainbow`, `secondary`, and fixed colors also work. On nucleotides, `secondary` uses the base palette. Ball-and-stick and surface views also accept `element` colors.

Residue overrides apply to the backbone, bases, atoms, bonds, and surface. Keep a selection and use it across the movie:

```python
from proteinmotion import Colorize, SetOpacity, Write

site = dna.select(chain="A", residues=(5, 8))
other_strand = dna.select(chain="B")
self.camera.set_focus(dna, chain="A", residues=(5, 8), fstop=4)  # Before the first play().
self.play(
    Colorize(site, "#ffd47b", residue_delay=0.1),
    SetOpacity(other_strand, 0.15),
    run_time=2,
)
self.play(Write(site.callout("Selected bases")), run_time=1)
self.add(dna.label_residues(chain="A", residues=(5, 8), format="one_letter"))
```

Residue numbers are PDB author numbers. A tuple gives an inclusive range. Color and opacity delays follow file order along each strand, usually 5′ to 3′. Atom selectors accept both `C4'` and older `C4*` names.

## Surfaces and measurements

Use `surface(kind="ses", color="base")`, `surface(kind="sas")`, or `surface(kind="vdw")`. Surface resolution and deformation controls are the same as for proteins. The SES is a voxel approximation.

`ResidueValues.b_factors()` and `ResidueValues.rmsf()` use Cα for proteins and C4′ for nucleotides, with P as a fallback for coarse models. Use `atoms="P"` for phosphate values or `atoms=None` for a per-residue mean. Sequence tracks and contact maps use these same backbone anchors. Contact-map distances measure backbone proximity; they do not assign base pairs.

Distance labels, region highlights, imported property values and density-map overlays accept nucleotide selections. For DNA/RNA electrostatics, supply imported atom charges. Standard base hydrogen-bond analysis uses explicit hydrogen coordinates: load with `include_hydrogens=True` and use `hydrogens="explicit"`. Modified bases need explicit donor and acceptor selections.

## States and trajectories

Multi-model PDB/mmCIF and NumPy trajectories use the usual `PlayTrajectory` API. MDAnalysis readers select `protein or nucleic` by default; pass `selection="nucleic"` for nucleic acids alone. All states must have the same selected atom identities. `Morph` and `Deform` support the same topology. Contact-matched `BackboneMorph` uses C1′ for DNA/RNA and Cα for proteins. It moves whole matched residues in deposited order, with configurable delays and fades for unmatched residues.

The native renderer keeps base geometry on the GPU and reads interpolated atom positions for each frame. Slabs and rings follow a rigid base plane defined by three ring atoms. Sticks follow each bond endpoint. EEVEE renders the same geometry and supports residue-based depth of field.

The loader uses C4′ backbone traces and respects chain breaks. Modified residues keep their deposited names. Their base colors use parent-residue annotations when present, with a neutral color for unknown parents. Incomplete ring templates fall back to sticks with a warning. P-only models show the deposited backbone. Ladder rods are a schematic base representation, not hydrogen bonds.

## DNA example

Download [dna_styles.py](dna_styles.py) and [1bna.cif](data/1bna.cif). Keep the structure in a `data/` directory beside the script.

This script uses the [1BNA B-DNA dodecamer](https://www.rcsb.org/structure/1BNA). It changes base styles, highlights part of one strand, and switches to atoms and a surface.

```python output=dna-styles
"""DNA base styles, residue colors, transparency, labels and molecular surface.

Input: RCSB PDB 1BNA, a deposited B-DNA dodecamer (no simulated motion).
Render: proteinmotion render examples/dna_styles.py DNAStyles --fps 60 -o dna.mp4
The same scene accepts --renderer eevee; Blender 4.5+ is required for EEVEE.
"""

from pathlib import Path

from proteinmotion import (
    BaseStyle,
    Colorize,
    FadeOut,
    NucleicAcid,
    ProteinScene,
    Representation,
    SetOpacity,
    Text,
    Write,
)

DATA = Path(__file__).resolve().parent / "data"


class DNAStyles(ProteinScene):
    def construct(self):
        dna = NucleicAcid.from_file(DATA / "1bna.cif").cartoon(bases="slabs").center()
        dna.rotate(1.35, axis=(1, 0, 0)).rotate(-0.28, axis=(0, 0, 1))
        self.add(dna)
        self.camera.frame(dna, aspect=self.width / self.height).zoom(1.22)
        self.camera.depth_cue = 0.3
        self.camera.set_focus(dna, chain="A", residues=(5, 8), fstop=5.6)
        self.add(Text("DNA · 1BNA", position=(0.055, 0.88), font_size=30))
        title = Text("Base slabs", position=(0.055, 0.10), font_size=46)
        self.play(Write(title), self.camera.animate.orbit(theta=0.12), run_time=1)
        self.play(self.camera.animate.orbit(theta=0.18), run_time=2)
        for style, heading in (
            ("rings", "Filled base rings"),
            ("sticks", "Base sticks"),
            ("ladder", "Ladder rods"),
        ):
            self.play(
                FadeOut(title), BaseStyle(dna, style), self.camera.animate.orbit(theta=0.10), run_time=0.8
            )
            title = Text(heading, position=(0.055, 0.10), font_size=46)
            self.play(Write(title), self.camera.animate.orbit(theta=0.12), run_time=0.8)
            self.play(self.camera.animate.orbit(theta=0.18), run_time=1.4)
        region = dna.select(chain="A", residues=(5, 8))
        other = dna.select(chain="B")
        self.play(
            FadeOut(title),
            BaseStyle(dna, "slabs"),
            Colorize(region, "#ffd47b", residue_delay=0.12),
            SetOpacity(other, 0.16),
            run_time=1.5,
        )
        label = region.callout(
            "Residues 5–8",
            subtitle="Color and opacity follow the selection",
            position=(0.67, 0.25),
            font_size=34,
            color="#ffd47b",
        )
        self.play(Write(label), self.camera.animate.orbit(theta=0.12), run_time=1)
        self.play(Representation(dna, "ball_and_stick"), run_time=1.2)
        self.play(self.camera.animate.orbit(theta=0.20), run_time=2)
        self.play(FadeOut(label), SetOpacity(other, 1), Colorize(region, None), run_time=1)
        title = Text("Molecular surface", position=(0.055, 0.10), font_size=46)
        self.play(Representation(dna, "surface"), Write(title), run_time=1.5)
        self.play(self.camera.animate.orbit(theta=0.30), run_time=2.5)
        self.wait(0.5)
```

```bash
proteinmotion render examples/dna_styles.py DNAStyles --fps 60 -o dna.mp4
proteinmotion render examples/dna_styles.py DNAStyles --renderer eevee --fps 60 -o dna-eevee.mp4
```

EEVEE requires Blender 4.5 or later. See [EEVEE and depth of field](eevee.md) for installation and quality settings.

## RNA example

Download [rna_styles.py](rna_styles.py) and [1ehz.cif](data/1ehz.cif). Keep the structure in a `data/` directory beside the script.

This script uses [1EHZ yeast phenylalanine tRNA](https://www.rcsb.org/structure/1EHZ). It selects the anticodon loop, changes base styles, and colors the molecular surface by deposited B factors. The preview shows camera motion around one experimental conformation.

```python output=rna-styles
"""Yeast tRNA: base rings, residue focus, B-factor colors and surface.

Input: RCSB PDB 1EHZ. Modified bases use deposited parent-residue annotations.
Render: proteinmotion render examples/rna_styles.py RNAStyles --fps 60 -o rna.mp4
"""

from pathlib import Path

from proteinmotion import (
    BaseStyle,
    ColorByProperty,
    Colorize,
    ColorLegend,
    ColorScale,
    FadeOut,
    NucleicAcid,
    ProteinScene,
    Representation,
    ResidueValues,
    SetOpacity,
    Text,
    Write,
)

DATA = Path(__file__).resolve().parent / "data"


class RNAStyles(ProteinScene):
    def construct(self):
        rna = NucleicAcid.from_file(DATA / "1ehz.cif").cartoon(bases="rings").center()
        rna.rotate(-0.5, axis=(0, 1, 0)).rotate(0.15, axis=(0, 0, 1))
        self.add(rna)
        self.camera.frame(rna, aspect=self.width / self.height).zoom(1.3)
        self.camera.depth_cue = 0.3
        self.camera.set_focus(rna, chain="A", residues=(32, 38), fstop=5.6)
        self.add(Text("Transfer RNA · 1EHZ", position=(0.055, 0.10), font_size=30))
        loop = rna.select(chain="A", residues=(32, 38))
        context = rna.select(chain="A", residues=[*range(1, 32), *range(39, 77)])
        self.play(self.camera.animate.orbit(theta=0.35), run_time=3)
        self.play(Colorize(loop, "#ffd47b", residue_delay=0.08), SetOpacity(context, 0.18), run_time=1.5)
        callout = loop.callout(
            "Anticodon loop", subtitle="Residues 32–38", position=(0.67, 0.20), font_size=38, color="#ffd47b"
        )
        self.play(Write(callout), self.camera.animate.orbit(theta=0.15), run_time=1.5)
        self.play(BaseStyle(rna, "slabs"), self.camera.animate.orbit(theta=0.15), run_time=1.5)
        self.play(FadeOut(callout), SetOpacity(context, 1), Colorize(loop, None), run_time=1)
        values = ResidueValues.b_factors(rna)  # C4′ values, including modified nucleotides.
        scale = ColorScale.from_values(values, colors=("#5ca7dc", "#77d1c0", "#f6c675"))
        legend = ColorLegend(scale, title="Deposited B factor", unit="Å²")
        self.play(ColorByProperty(rna, values, scale=scale), Write(legend), run_time=1.5)
        self.play(Representation(rna, "surface"), self.camera.animate.orbit(theta=0.15), run_time=1.5)
        self.play(self.camera.animate.orbit(theta=0.30), run_time=2.5)
        self.wait(0.5)
```

```bash
proteinmotion render examples/rna_styles.py RNAStyles --fps 60 -o rna.mp4
```

## DNA morph

C1′ positions drive contact matching, alignment, and motion. The same `BackboneMorph` API works with `NucleicAcid`. The example uses chain A from [1BNA](https://www.rcsb.org/structure/1BNA) and [2DCG](https://www.rcsb.org/structure/2DCG). These are deposited DNA structures; the interpolated path is a visual transition.

Five nucleotides match under the 0.20 contact-error limit. Seven source nucleotides fade out and one target nucleotide fades in. Gold marks the matched residues. Motion starts in 5′-to-3′ order with a 0.25-second delay between residues.

Download [dna_morph.py](dna_morph.py), [1bna.cif](data/1bna.cif), and [2dcg.cif](data/2dcg.cif). Keep the structures in a `data/` directory beside the script.

```python output=dna-morph
"""C1′ contact-guided morph between chain A of PDB 1BNA and 2DCG.

Matched nucleotides are gold. Unmatched source residues fade out and unmatched
destination residues fade in. This is a visual interpolation, not simulated dynamics.
Render: proteinmotion render examples/dna_morph.py DNAMorph --fps 60 -o dna-morph.mp4
"""

from pathlib import Path

from proteinmotion import (
    BackboneMorph,
    NucleicAcid,
    ProteinScene,
    Representation,
    Text,
    Write,
    match_backbones,
)

DATA = Path(__file__).resolve().parent / "data"


class DNAMorph(ProteinScene):
    def construct(self):
        source = (
            NucleicAcid.from_file(DATA / "1bna.cif", chains="A")
            .cartoon(bases="rings", color="#75b8d9")
            .center()
        )
        target = (
            NucleicAcid.from_file(DATA / "2dcg.cif", chains="A")
            .cartoon(bases="rings", color="#df92a0")
            .center()
        )
        source.rotate(1.35, axis=(1, 0, 0)).rotate(-0.28, axis=(0, 0, 1))
        target.rotate(1.35, axis=(1, 0, 0)).rotate(-0.28, axis=(0, 0, 1))
        match = match_backbones(source, target, max_contact_error=0.20, max_candidates=None, search_seconds=2)
        self.match = match
        for p, ids in ((source, match.source_indices), (target, match.target_indices)):
            p.color_residues("#ffd47b", chain="A", residues=[p.topology.residues[i].resid for i in ids])
            extras = [r.resid for r in p.topology.residues if not r.is_nucleic]
            if extras:
                p.set_residue_opacity(0, chain="A", residues=extras)
        self.add(source)
        self.camera.frame(source, aspect=self.width / self.height).zoom(1.1)
        self.camera.depth_cue = 0.25
        self.add(Text("1BNA A → 2DCG A · matched nucleotides in gold", position=(0.055, 0.89), font_size=27))
        self.play(Write(Text("DNA morph · C1′ anchors", position=(0.055, 0.10), font_size=40)), run_time=1)
        self.wait(1)
        self.play(
            BackboneMorph(source, target, match=match, residue_delay=0.25, align=True),
            run_time=5,
        )
        self.focus(target, margin=1.4, run_time=1.5)
        self.play(Representation(target, "ball_and_stick"), run_time=1.2)
        self.play(self.camera.animate.orbit(theta=0.35), run_time=2)
        self.wait(0.5)
```

```bash
proteinmotion render examples/dna_morph.py DNAMorph --fps 60 -o dna-morph.mp4
```

Each endpoint uses one strand. Automatic matching skips nucleotides without C1′; explicit correspondences require it. Older C1* atom names are accepted. The displayed backbone remains a C4′ trace. See [morphing](morphing.md) for the contact-map objective, search limits, and saved correspondences.
