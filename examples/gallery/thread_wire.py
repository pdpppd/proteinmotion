"""Glowing wires trace each chain of a tetramer, then the cartoon fades in.

Input: human deoxyhemoglobin, PDB 4HHB, chains A–D. mode="wire" draws a wire
that runs along the backbone from the C terminus; stagger offsets the chains.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, StudioLook, Text, Thread

DATA = Path(__file__).resolve().parent.parent / "data"


class ThreadWire(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="film", grain="off")

    def construct(self):
        protein = (
            Protein.from_file(DATA / "4hhb.cif", chains=["A", "B", "C", "D"]).cartoon(color="chain").center()
        )
        self.camera.frame(protein, margin=1.0, screen_position=(0.6, 0.53))
        self.add(Text("Wire threading", position=(0.05, 0.08), font_size=46, font="semibold"))
        self.add(
            Text(
                'Thread(protein, self.camera, mode="wire")',
                position=(0.05, 0.145),
                font_size=28,
                color="#a3b3c7",
            )
        )
        self.play(Thread(protein, self.camera, mode="wire", stagger=0.16), run_time=7)
        self.play(self.camera.animate.orbit(0.6), run_time=2.5)
