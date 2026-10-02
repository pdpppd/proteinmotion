"""Ghost the surroundings to bring one region forward.

Input: ubiquitin, PDB 1UBQ. ~region selects everything else; SetOpacity with
residue_delay fades residues in sequence.
"""

from pathlib import Path

from proteinmotion import HideSideChains, Protein, ProteinScene, SetOpacity, ShowSideChains, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class Opacity(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        sheet = protein.select(secondary="E")
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.52))
        self.add(protein, Text("Transparency", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "SetOpacity(~sheet, 0.18, residue_delay=0.03)",
            position=(0.05, 0.145),
            font_size=28,
            color="#a3b3c7",
        )
        self.add(caption)
        self.wait(0.5)
        self.play(SetOpacity(~sheet, 0.18, residue_delay=0.03), self.camera.animate.orbit(0.4), run_time=2.5)
        self.play(ShowSideChains(sheet, residue_delay=0.04), self.camera.animate.orbit(0.5), run_time=2)
        self.play(
            caption.animate.set_text("SetOpacity(protein, 0.35)"),
            HideSideChains(sheet),
            SetOpacity(~sheet, 1),
            SetOpacity(protein, 0.35),
            run_time=1.5,
        )
        self.play(
            caption.animate.set_text("SetOpacity(protein, 1)"),
            SetOpacity(protein, 1),
            self.camera.animate.orbit(0.3),
            run_time=1.5,
        )
        self.wait(0.5)
