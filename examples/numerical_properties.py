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
