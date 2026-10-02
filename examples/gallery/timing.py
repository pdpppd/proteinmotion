"""Give concurrent actions their own durations and delays.

Input: ubiquitin, PDB 1UBQ. animation.during(seconds, delay=...) offsets an
action inside one play() call; the call lasts until its last action ends.
"""

from pathlib import Path

from proteinmotion import FadeIn, MolecularStyle, Protein, ProteinScene, StudioLook, Text, Write

DATA = Path(__file__).resolve().parent.parent / "data"


class IndependentTiming(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="subtle")

    def construct(self):
        style = MolecularStyle("cartoon", {"color": "#8fe3c0"})
        protein = Protein.from_file(DATA / "1ubq.cif").set_style(style).center()
        helix = protein.select(residue_range=(23, 34))
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.52))
        title = Text("Independent timing", position=(0.05, 0.08), font_size=46, font="semibold")
        caption = Text("during(seconds, delay=…)", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        self.play(
            FadeIn(protein).during(2), Write(title).during(1, delay=0.5), FadeIn(caption).during(1, delay=1)
        )
        self.play(
            helix.animate.set_color("#ffc46b").show_atoms().during(2),
            (~helix).animate.set_opacity(0.25).during(1, delay=0.75),
            self.camera.animate.orbit(0.8).during(3.5),
        )
        self.play((~helix).animate.set_opacity(1), helix.animate.hide_atoms(), run_time=1.5)
        self.wait(0.5)
