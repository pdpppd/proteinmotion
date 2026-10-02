"""Calcium ions drawn on the cartoon, and the side chains that coordinate them.

Input: calmodulin with four bound Ca²⁺, PDB 1CLL. Cartoons draw ligands and
ions by default. within=3.0 selects whole residues with an atom within 3 Å of
the ions.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, ShowSideChains, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class CalciumSites(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1cll.cif").cartoon().center()
        ions = protein.select(ions=True)
        site = protein.select(within=3.0, of=ions)
        first = protein.select(ions=True, residues=149)
        self.camera.theta, self.camera.phi = 0.4, 0.15
        self.camera.frame(protein, margin=1.15, screen_position=(0.6, 0.55))
        self.add(protein, Text("Ions and ligands", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text("Calmodulin · four Ca²⁺ ions", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        self.add(caption)
        self.play(self.camera.animate.orbit(0.4), run_time=1.5)
        self.play(
            caption.animate.set_text("ShowSideChains(within 3 Å of the ions)"),
            ShowSideChains(site, residue_delay=0.04),
            self.camera.animate.orbit(0.4),
            run_time=2.5,
        )
        self.play(caption.animate.set_text("EF-hand I"), run_time=0.5)
        self.focus(
            protein.select(within=3.0, of=first) | first,
            margin=2.4,
            screen_position=(0.6, 0.55),
            run_time=1.5,
        )
        self.play(self.camera.animate.orbit(0.8), run_time=3)
        self.focus(protein, margin=1.15, screen_position=(0.6, 0.55), run_time=1.5)
