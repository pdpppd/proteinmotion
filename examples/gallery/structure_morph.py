"""Adenylate kinase closes: a morph between two crystal structures.

Inputs: E. coli adenylate kinase, open (PDB 4AKE) and closed around the
inhibitor Ap5A (PDB 1AKE), chain A, superposed on the CORE domain. Residues
follow screw paths between the endpoints; the path is an illustration, not a
computed transition. Ap5A exists only in 1AKE, so it fades in.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, StructureMorph, StudioLook, Text, domain_motion

DATA = Path(__file__).resolve().parent.parent / "data"


class AdenylateKinase(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="subtle")

    def construct(self):
        open_ = Protein.from_file(DATA / "4ake.cif", chains="A").cartoon(color="#7f93b3")
        closed = Protein.from_file(DATA / "1ake.cif", chains="A").cartoon()
        core = open_.select(residues=[*range(1, 30), *range(60, 122), *range(160, 215)])
        lid, nmp = open_.select(residues=(122, 159)), open_.select(residues=(30, 59))
        lid.set_color("#f5d477")
        nmp.set_color("#50e0d0")
        swing = domain_motion(open_, closed, moving=lid, fixed=core)
        open_.center()
        self.camera.look_along(swing.axis, open_)
        self.camera.frame(open_, margin=0.8, screen_position=(0.6, 0.53))
        self.add(open_, Text("Structure morphs", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "Adenylate kinase · open, PDB 4AKE", position=(0.05, 0.145), font_size=28, color="#a3b3c7"
        )
        self.add(caption)
        self.add(Text("LID", position=(0.05, 0.78), font_size=30, color="#f5d477"))
        self.add(Text("NMP-binding", position=(0.05, 0.83), font_size=30, color="#50e0d0"))
        self.add(Text("CORE", position=(0.05, 0.88), font_size=30, color="#7f93b3"))
        self.wait(1)
        self.play(
            StructureMorph(open_, closed, align=core),
            caption.animate.set_text(f"Closed around Ap5A, PDB 1AKE · LID turns {swing.angle:.0f}°"),
            run_time=5,
        )
        self.wait(0.5)
        self.play(self.camera.animate.orbit(0.8), run_time=3)
