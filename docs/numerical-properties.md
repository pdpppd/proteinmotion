# Numerical properties

Map one value per residue to color and cartoon thickness. Values can come from a structure file, an aligned trajectory, or your own analysis. Colors apply to cartoons, ribbons, ball-and-stick models, and molecular surfaces. Thickness changes the cartoon cross-section.

## Complete example and output

This script uses the Cα B factors in PDB 1UBQ. The color scale stays fixed while the representation changes.

```python output=numerical-properties
"""Map 1UBQ Cα B factors to residue color and cartoon thickness."""

from pathlib import Path

from proteinmotion import (
    ColorByProperty,
    ColorLegend,
    ColorScale,
    Protein,
    ProteinScene,
    Representation,
    ResidueValues,
    Rotate,
    Text,
)

DATA = Path(__file__).parent / "data"


class NumericalProperties(ProteinScene):
    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").center()
        values = ResidueValues.b_factors(protein)
        scale = ColorScale(0, 40)
        self.add(protein)
        self.camera.frame(protein, margin=1.08)
        self.add(Text("Residue properties", position=(0.06, 0.07), font_size=42))
        self.add(Text("Ubiquitin · Cα B factors", position=(0.06, 0.13), font_size=25))
        self.add(ColorLegend(scale, title="B factor", unit="Å²", position=(0.06, 0.8)))
        self.play(
            ColorByProperty(protein, values, scale=scale, thickness=(0.6, 1.8), residue_delay=0.012),
            Rotate(protein, 0.25),
            run_time=2.5,
        )
        self.wait(0.5)
        self.play(Representation(protein, "ball_and_stick"), run_time=1.5)
        self.play(Rotate(protein, 0.3), run_time=1)
        self.play(Representation(protein, "surface"), run_time=1.5)
        self.play(Rotate(protein, 0.25), run_time=1.5)
```

[Download the script](numerical_properties.py). Run it from a repository checkout, which includes `examples/data/1ubq.cif`:

```bash
proteinmotion render examples/numerical_properties.py NumericalProperties --fps 60 -o properties.mp4
```

## Supply values

`ResidueValues` takes one number for every residue in `protein.topology.residues`, including any selected ligands. Use `NaN` for missing measurements. `from_mapping` is useful when an analysis covers part of the structure:

```python
from proteinmotion import ResidueValues, ColorScale

values = ResidueValues.from_mapping(
    protein,
    {("A", 10): 0.3, ("A", 11): 0.7, ("A", 12): 0.9},
    name="Conservation",
)
scale = ColorScale(0, 1, colors=("#355f9e", "#f5d477"))
protein.color_by(values, scale=scale, thickness=(0.6, 1.8))
```

Keys use chain ID and PDB author residue number. Include an insertion code as the third item when needed, such as `("A", 10, "B")`. Unknown or ambiguous keys raise an error. Apply immediate styling before `self.add(protein)`.

A `ColorScale` uses fixed limits and evenly spaced color stops. Values beyond the limits use the endpoint colors. Missing values use gray by default; change this with `missing="#RRGGBB"`. `ColorScale.from_values(values)` sets limits from the finite values. Reuse the same scale in the structure, sequence track, and legend.

`thickness=(0.6, 1.8)` gives the low end 0.6 times the usual cartoon width and height, and the high end 1.8 times. Missing values keep the usual thickness. Ball radii and surface geometry retain their existing sizes.

## B factors and confidence

```python
b_factors = ResidueValues.b_factors(protein)  # Cα values from the first model
mean_b = ResidueValues.b_factors(protein, atoms=None)  # mean over selected atoms per residue
confidence = ResidueValues.b_factors(protein, name="pLDDT", unit="")
```

Use the confidence label only for files that store confidence in the B-factor field. ProteinMotion reads the stored numbers; the file's source defines their meaning.

## RMSF from an ensemble or trajectory

```python
core = protein.select(chain="A", residues=(5, 60), atoms="CA")
rmsf = ResidueValues.rmsf(protein, alignment=core)
```

RMSF is the square root of the mean squared displacement from each atom's mean position, in ångströms. Frames are aligned to the first frame using backbone anchors by default: Cα for proteins, C4′ for nucleotides, or P when C4′ is absent. `alignment` chooses another region for the fit; `align=False` uses coordinates as supplied. `atoms=None` averages atomic mean-square fluctuations within each residue before taking the square root. `stride` controls frame sampling. Computation reads one frame at a time.

For NMR ensembles, RMSF describes variation across the supplied models. For MD, unwrap periodic coordinates before analysis. Choose an alignment region that fits the question you want to show.

## Animate a property change

```python
from proteinmotion import ColorByProperty, ColorLegend

self.add(ColorLegend(scale, title=values.name, unit=values.unit))
self.play(
    ColorByProperty(protein, values, scale=scale, thickness=(0.6, 1.8),
                    residue_delay=0.01),
    run_time=3,
)
```

`residue_delay` is in seconds. `run_time` must exceed the delay across all residues. Set `reverse=True` for C-to-N order, or `easing="linear"` for a linear change within each residue. The default uses smooth easing. Use separate animations in the same `play` call for camera movement or opacity changes.
