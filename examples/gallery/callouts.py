"""Label a region with a callout and residues with three-letter names.

Input: ubiquitin, PDB 1UBQ. Leu8, Ile44, and Val70 form the hydrophobic
patch that ubiquitin-binding domains recognize. Labels track their residues
as the camera moves.
"""

from pathlib import Path

from proteinmotion import (
    FadeIn,
    FadeOut,
    Protein,
    ProteinScene,
    ShowSideChains,
    StudioLook,
    Text,
    Unwrite,
    Write,
)

DATA = Path(__file__).resolve().parent.parent / "data"


class Callouts(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        helix = protein.select(residues=(23, 34), atoms="CA")
        patch = protein.select(residues=[8, 44, 70])
        self.camera.theta = 0.6
        self.camera.frame(protein, margin=0.95, screen_position=(0.6, 0.52))
        self.add(protein, Text("Callouts and labels", position=(0.05, 0.08), font_size=46, font="semibold"))
        note = helix.callout(
            "α helix", subtitle="Residues 23–34", position=(0.12, 0.42), font_size=40, tip="arrow"
        )
        box = helix.highlight(style="box", color="#f2ba67")
        self.play(Write(note), FadeIn(box), run_time=1.5)
        self.play(self.camera.animate.orbit(0.4), run_time=2)
        self.play(Unwrite(note), FadeOut(box), run_time=0.8)
        labels = protein.label_residues(
            residues=[8, 44, 70],
            font_size=34,
            color="#ffc46b",
            offsets={8: (-320, -60), 44: (340, -170), 70: (340, 130)},
        )
        self.play(ShowSideChains(patch, residue_delay=0.2), run_time=1.2)
        self.play(Write(labels, lag_ratio=0.15), run_time=1.5)
        self.play(self.camera.animate.orbit(0.5), run_time=2.5)
        self.play(Unwrite(labels), run_time=0.8)
