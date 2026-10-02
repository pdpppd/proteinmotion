"""Tint selected residues one after another, then restore the palette.

Input: ubiquitin, PDB 1UBQ. Colorize() with residue_delay starts each residue's
color change after the previous one; Colorize(region, None) restores it.
"""

from pathlib import Path

from proteinmotion import Colorize, Protein, ProteinScene, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class ResidueColors(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon(color="#b8c4d6").center()
        helix = protein.select(residues=(23, 34))
        hairpin = protein.select(residues=(1, 17))
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.52))
        self.add(protein, Text("Residue colors", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "Colorize(region, color, residue_delay=…)", position=(0.05, 0.145), font_size=28, color="#a3b3c7"
        )
        self.add(caption)
        self.wait(0.5)
        self.play(
            Colorize(helix, "#50e0d0", residue_delay=0.1),
            Colorize(hairpin, "#ffb45e", residue_delay=0.08),
            self.camera.animate.orbit(0.5),
            run_time=2.5,
        )
        self.play(
            Colorize(protein.select(residues=(40, 76)), "#ff7f9e", residue_delay=0.03, reverse=True),
            run_time=2,
        )
        self.play(self.camera.animate.orbit(0.4), run_time=1.5)
        self.play(caption.animate.set_text("Colorize(protein, None)"), Colorize(protein, None), run_time=1.5)
        self.wait(0.5)
