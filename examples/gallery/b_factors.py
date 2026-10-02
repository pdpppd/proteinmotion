"""Color and thicken the cartoon by crystallographic B factor.

Input: ubiquitin, PDB 1UBQ, Cα B factors from the deposited file. Higher B
factors mark atoms with more displacement or disorder in the crystal; the
C-terminal tail is the most mobile.
"""

from pathlib import Path

from proteinmotion import (
    ColorByProperty,
    ColorLegend,
    ColorScale,
    Protein,
    ProteinScene,
    ResidueValues,
    StudioLook,
    Text,
)

DATA = Path(__file__).resolve().parent.parent / "data"


class BFactors(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon(color="#b8c4d6").center()
        values = ResidueValues.b_factors(protein)
        scale = ColorScale(2, 30)
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.52))
        self.add(protein, Text("B factors", position=(0.05, 0.08), font_size=46, font="semibold"))
        self.add(
            Text(
                "Color and cartoon width by Cα B factor",
                position=(0.05, 0.145),
                font_size=28,
                color="#a3b3c7",
            )
        )
        legend = ColorLegend(scale, title="B factor", unit="Å²", position=(0.05, 0.8))
        self.wait(0.5)
        self.add(legend)
        self.play(
            ColorByProperty(protein, values, scale=scale, thickness=(0.6, 2.0), residue_delay=0.02),
            self.camera.animate.orbit(0.5),
            run_time=3,
        )
        self.play(self.camera.animate.orbit(1.2), run_time=5)
