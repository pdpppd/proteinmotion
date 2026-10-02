"""Hydropathy, then sequence conservation, painted on a molecular surface.

Input: ubiquitin, PDB 1UBQ, and the Pfam PF00240 seed alignment (CC0).
Conservation is 1 − H/ln 20 per alignment column with Henikoff weights; the
ConSurf palette runs from turquoise (variable) to maroon (conserved).
"""

from pathlib import Path

from proteinmotion import (
    ColorByProperty,
    ColorLegend,
    FadeIn,
    FadeOut,
    Protein,
    ProteinScene,
    ResidueValues,
    StudioLook,
    Text,
)

DATA = Path(__file__).resolve().parent.parent / "data"


class Conservation(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").surface(color="hydropathy").center()
        hydropathy = ResidueValues.hydropathy(protein)
        conservation = ResidueValues.conservation(protein, DATA / "pf00240-seed.sto")
        self.camera.frame(protein, margin=0.95, screen_position=(0.6, 0.52))
        self.add(protein, Text("Residue properties", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text("Kyte–Doolittle hydropathy", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        first = ColorLegend(hydropathy, position=(0.05, 0.8))
        second = ColorLegend(conservation, position=(0.05, 0.8))
        self.add(caption, first)
        self.play(self.camera.animate.orbit(1.0), run_time=3.5)
        self.play(
            ColorByProperty(protein, conservation, stagger=0.4),
            caption.animate.set_text("Conservation · Pfam PF00240, 59 sequences"),
            FadeOut(first),
            FadeIn(second),
            run_time=2.5,
        )
        self.play(self.camera.animate.orbit(1.2), run_time=4)
