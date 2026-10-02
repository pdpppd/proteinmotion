"""Transfer RNA: highlight the anticodon loop, then show the surface.

Input: yeast tRNA-Phe, PDB 1EHZ. Modified nucleotides keep their names and
take their parent base's color.
"""

from pathlib import Path

from proteinmotion import (
    Colorize,
    NucleicAcid,
    ProteinScene,
    Representation,
    SetOpacity,
    StudioLook,
    Text,
    Unwrite,
    Write,
)

DATA = Path(__file__).resolve().parent.parent / "data"


class TransferRNA(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        rna = NucleicAcid.from_file(DATA / "1ehz.cif").cartoon(bases="rings").center()
        rna.rotate(-0.5, axis=(0, 1, 0)).rotate(0.15, axis=(0, 0, 1))
        loop = rna.select(residues=(32, 38))
        self.camera.frame(rna, margin=1.0, screen_position=(0.6, 0.52))
        self.add(rna, Text("Transfer RNA", position=(0.05, 0.08), font_size=46, font="semibold"))
        self.add(Text("Yeast tRNA-Phe · PDB 1EHZ", position=(0.05, 0.145), font_size=28, color="#a3b3c7"))
        self.play(self.camera.animate.orbit(0.35), run_time=2)
        note = loop.callout(
            "Anticodon loop", subtitle="Residues 32–38", position=(0.12, 0.45), font_size=36, color="#ffd47b"
        )
        self.play(
            Colorize(loop, "#ffd47b", residue_delay=0.08), SetOpacity(~loop, 0.2), Write(note), run_time=1.8
        )
        self.play(self.camera.animate.orbit(0.25), run_time=1.5)
        self.play(Unwrite(note), SetOpacity(~loop, 1), run_time=1)
        self.play(Representation(rna, "surface"), self.camera.animate.orbit(0.3), run_time=1.8)
        self.play(self.camera.animate.orbit(0.4), run_time=2)
