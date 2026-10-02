"""DNA base styles: slabs, filled rings, sticks, and ladder rods.

Input: B-DNA dodecamer, PDB 1BNA, in four copies. Slabs and rings follow the
deposited base planes; ladder rods are schematic. BaseStyle(dna, style)
animates a change from one style to another.
"""

from pathlib import Path

from proteinmotion import NucleicAcid, ProteinScene, Rotate, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class DNABases(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        dna = NucleicAcid.from_file(DATA / "1bna.cif").center()
        dna.rotate(1.35, axis=(1, 0, 0)).rotate(-0.28, axis=(0, 0, 1))
        styles = ("slabs", "rings", "sticks", "ladder")
        copies = [
            dna.copy().cartoon(bases=style).shift(((i - 1.5) * 32, 0, 0)) for i, style in enumerate(styles)
        ]
        self.camera.theta, self.camera.phi = 0, 0
        self.camera.frame(*copies, margin=0.95, screen_position=(0.5, 0.5))
        self.camera.zoom(1.8)
        self.add(*copies, Text("DNA base styles", position=(0.05, 0.08), font_size=46, font="semibold"))
        for i, style in enumerate(styles):
            self.add(
                Text(
                    f'bases="{style}"', position=(0.5 + (i - 1.5) * 0.236, 0.9), font_size=28, align="center"
                )
            )
        self.play(*(Rotate(p, 3.14) for p in copies), run_time=8)
