"""Thread a cartoon into view, then reverse it.

Input: E. coli ribose-binding protein, PDB 2DRI chain A. Thread moves the
actual cartoon geometry along eased paths; it is an entrance animation, not a
folding simulation. Unthread plays the motion in reverse.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, StudioLook, Text, Thread, Unthread

DATA = Path(__file__).resolve().parent.parent / "data"


class Threading(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").cartoon(color="rainbow").center()
        self.camera.frame(protein, margin=1.05, screen_position=(0.6, 0.53))
        self.add(Text("Threading", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            'Thread(protein, easing="smooth")', position=(0.05, 0.145), font_size=28, color="#a3b3c7"
        )
        self.add(caption)
        self.play(Thread(protein, easing="smooth", swirl=0.25, glow=0, seed=4), run_time=5)
        self.play(self.camera.animate.orbit(0.5), run_time=2)
        self.play(
            caption.animate.set_text("Unthread(protein)"),
            Unthread(protein, easing="smooth", glow=0, seed=4),
            run_time=3.5,
        )
        self.wait(0.3)
