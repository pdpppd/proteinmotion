"""Built-in color schemes: secondary structure, rainbow, chain, and hydropathy.

Input: human deoxyhemoglobin, PDB 4HHB, four chains. Representation() with a
color option blends the cartoon from one palette to the next.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, Representation, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class ColorSchemes(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "4hhb.cif", chains=["A", "B", "C", "D"]).cartoon().center()
        self.camera.frame(protein, margin=1.0, screen_position=(0.6, 0.52))
        self.add(protein, Text("Color schemes", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text('color="secondary"', position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        self.add(caption)
        self.play(self.camera.animate.orbit(0.4), run_time=1.5)
        for scheme in ("rainbow", "chain", "hydropathy"):
            self.play(
                Representation(protein, "cartoon", color=scheme),
                caption.animate.set_text(f'color="{scheme}"'),
                self.camera.animate.orbit(0.4),
                run_time=1.5,
            )
            self.wait(0.8)
