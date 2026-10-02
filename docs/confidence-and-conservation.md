# Heatmaps, confidence, and conservation

Show residue-by-residue matrices, such as AlphaFold predicted aligned error (PAE) or Cα distances, next to the structure. Color residues by AlphaFold pLDDT, hydropathy, or sequence conservation with preset color scales.

## Complete example and output

The first part colors the AlphaFold DB model of calmodulin by pLDDT and shows its PAE. Each lobe is predicted with confidence, but the position of one lobe relative to the other is not. The second part colors the ubiquitin surface by hydropathy, then by conservation across the Pfam ubiquitin family.

```python output=confidence-and-conservation
"""AlphaFold confidence for calmodulin, then hydropathy and conservation on ubiquitin.

Part 1 colors the AlphaFold DB model of human calmodulin (AF-P0DP23-F1, v6) by
pLDDT and shows its predicted aligned error. The two lobes are confident on their
own, but their relative position is not: the PAE blocks between them are high.
Part 2 colors the ubiquitin surface (PDB 1UBQ) by Kyte–Doolittle hydropathy, then
by conservation across the Pfam ubiquitin family seed alignment (PF00240).
"""

from pathlib import Path

from proteinmotion import (
    ColorByProperty,
    ColorLegend,
    FadeIn,
    FadeOut,
    Focus,
    Heatmap,
    Protein,
    ProteinScene,
    ResidueValues,
    Text,
    Unwrite,
    Write,
)

DATA = Path(__file__).parent / "data"
MUTED = "#a3b3c7"


class ConfidenceAndConservation(ProteinScene):
    def construct(self):
        aspect = self.width / self.height
        model = Protein.from_file(DATA / "AF-P0DP23-F1-model_v6.cif").cartoon().center()
        model.color_by("plddt")
        c_lobe = model.select(residues=(82, 148))
        pae = Heatmap.pae(
            model,
            DATA / "AF-P0DP23-F1-predicted_aligned_error_v6.json",
            highlight=c_lobe,
            position=(0.63, 0.14),
            size=(0.34, 0.6),
        ).with_panel()
        plddt = ColorLegend(ResidueValues.plddt(model), position=(0.05, 0.82))
        title = Text("AlphaFold confidence", font_size=46, font="semibold", position=(0.05, 0.07))
        subtitle = Text("Calmodulin, AF-P0DP23-F1", font_size=26, color=MUTED, position=(0.051, 0.135))
        self.add(model)
        self.camera.frame(model, margin=0.95, aspect=aspect, screen_position=(0.32, 0.5))
        self.play(Write(title), FadeIn(subtitle), FadeIn(pae), FadeIn(plddt), run_time=1.5)
        self.play(self.camera.animate.orbit(0.8), run_time=4)
        self.play(FadeOut(model), FadeOut(pae), FadeOut(plddt), FadeOut(subtitle), Unwrite(title), run_time=1)

        ubiquitin = Protein.from_file(DATA / "1ubq.cif").surface(color="hydropathy").center()
        hydropathy = ResidueValues.hydropathy(ubiquitin)
        conservation = ResidueValues.conservation(ubiquitin, DATA / "pf00240-seed.sto")
        title = Text("Hydropathy", font_size=46, font="semibold", position=(0.05, 0.07))
        subtitle = Text("Ubiquitin surface, PDB 1UBQ", font_size=26, color=MUTED, position=(0.051, 0.135))
        legend = ColorLegend(hydropathy, position=(0.05, 0.82))
        self.camera.theta, self.camera.phi = 0.0, 0.0
        self.play(
            FadeIn(ubiquitin),
            Write(title),
            FadeIn(subtitle),
            FadeIn(legend),
            Focus(self.camera, ubiquitin, margin=0.9, aspect=aspect),
            run_time=1.5,
        )
        self.play(self.camera.animate.orbit(0.9), run_time=3)
        swap = Text("Conservation", font_size=46, font="semibold", position=(0.05, 0.07))
        source = Text(
            "Pfam PF00240 seed alignment, 59 sequences", font_size=26, color=MUTED, position=(0.051, 0.135)
        )
        consurf = ColorLegend(conservation, position=(0.05, 0.82))
        self.play(
            ColorByProperty(ubiquitin, conservation, stagger=0.4),
            Unwrite(title),
            FadeOut(subtitle),
            FadeOut(legend),
            Write(swap),
            FadeIn(source),
            FadeIn(consurf),
            run_time=2.5,
        )
        self.play(self.camera.animate.orbit(1.2), run_time=4)
```

```bash
proteinmotion render examples/confidence_and_conservation.py ConfidenceAndConservation --fps 60 -o confidence.mp4
```

## Heatmaps

`Heatmap(matrix)` draws a 2D array as colored cells with a color bar. A function that returns an array is called again whenever the coordinates change, so the plot follows the movie.

```python
pae = Heatmap.pae(protein, "AF-P0DP23-F1-predicted_aligned_error_v6.json", highlight=lobe)
distances = Heatmap.distances(protein, region=protein.select(residues=(1, 70)))
change = Heatmap.distances(protein, reference=other_conformation)
custom = Heatmap(matrix, protein=protein, rows=residue_indices, title="Coupling", unit="")
self.add(pae.with_panel())
```

| Option | Default | Meaning |
|---|---|---|
| `protein`, `rows`, `columns` | `None` | Residues for each row and column (topology indices or Regions); label the axes with residue numbers and mark chain boundaries |
| `highlight` | `None` | Mark a Region's rows and columns at the edges |
| `scale`, `unit` | fitted, `""` | `ColorScale` for the cells and the unit shown on the color bar |
| `xlabel`, `ylabel` | `""` | Axis captions |
| `max_cells` | `160` | Larger matrices are averaged into at most this many cells per side |
| `colorbar` | `True` | Draw the color bar |

`Heatmap.pae(protein, source)` reads AlphaFold DB (`predicted_aligned_error_v*.json`), ColabFold (`scores*.json`), and AlphaFold 3 (`full_data*.json`) files, a `.npy` file, or an array. Rows are the residue each prediction is aligned on, and columns the residue whose error is shown, as on AlphaFold DB pages. The matrix must have one row per polymer residue of the loaded model. AlphaFold 3 tokens for ligand atoms are averaged onto their residues. The default scale runs from 0 Å (dark green) to the file's maximum error (near white).

`Heatmap.distances(protein)` shows Cα (or C4′) distances from the current coordinates. `anchor="centroid"` uses residue centroids. With `reference=` (a Protein with the same residues, or coordinates in the same atom order), the plot shows the change in distance: blue where residues moved closer, red where they moved apart. Live matrices are limited to 600 residues by default (`max_residues`).

## AlphaFold pLDDT

AlphaFold writes pLDDT into the B-factor column of its models. `ResidueValues.plddt(protein)` reads it from there, or from an AlphaFold DB `confidence_v*.json` or ColabFold scores file:

```python
protein.color_by("plddt")                              # AlphaFold's four bands
legend = ColorLegend(ResidueValues.plddt(protein))
```

`ColorScale.plddt()` uses the AlphaFold DB bands: very low below 50, low below 70, confident below 90, and very high above. Banded scales use `ColorScale(vmin, vmax, colors=..., boundaries=...)` with one more color than boundaries.

## Hydropathy

```python
protein.surface(color="hydropathy")                    # a named palette, like "secondary"
values = ResidueValues.hydropathy(protein)             # Kyte–Doolittle, −4.5 to 4.5
values = ResidueValues.hydropathy(protein, "eisenberg")
self.play(ColorByProperty(protein, "hydropathy"), run_time=2)
```

Hydrophobic residues are orange, hydrophilic residues blue, and residues near zero almost white. The scale is symmetric about zero. Non-amino acids are gray. Modified residues use their parent amino acid when the structure records one.

## Conservation

`ResidueValues.conservation(protein, alignment)` reads a FASTA, A3M, Stockholm, or Clustal alignment, or a list of aligned sequences.

```python
conservation = ResidueValues.conservation(protein, "family.sto")
self.play(ColorByProperty(protein, conservation, stagger=0.4), run_time=2.5)
self.add(ColorLegend(conservation))
```

The score of each alignment column is 1 − H/ln 20, where H is the Shannon entropy of its amino acids. Sequences are weighted by the Henikoff method so that similar sequences count less. The score is multiplied by the fraction of sequences without a gap in that column. Invariant columns score 1.

The query is the alignment sequence most similar to the structure, unless `query=` names it or gives its index. The query is aligned to every protein chain whose sequence it matches with at least 90% identity, or to the closest chain otherwise; `chain=` chooses chains. Residues outside the alignment are NaN and drawn gray. The preset ConSurf palette runs from turquoise (variable) through white to maroon (conserved) and spans the 5th–95th percentile of the protein's scores, so the most and least conserved residues stand out.

## Data sources

The calmodulin model and PAE are AlphaFold DB entry AF-P0DP23-F1, version 6 (CC BY 4.0). The ubiquitin family alignment is the Pfam PF00240 seed alignment from InterPro (CC0). See [third-party notices](https://github.com/pdpppd/proteinmotion/blob/main/THIRD_PARTY.md).
