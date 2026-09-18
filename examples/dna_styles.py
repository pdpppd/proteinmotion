"""DNA base styles, residue colors, transparency, labels and molecular surface.

Input: RCSB PDB 1BNA, a deposited B-DNA dodecamer (no simulated motion).
Render: proteinmotion render examples/dna_styles.py DNAStyles --fps 60 -o dna.mp4
The same scene accepts --renderer eevee; Blender 4.5+ is required for EEVEE.
"""

from pathlib import Path

from proteinmotion import (
    BaseStyle,
    Colorize,
    FadeOut,
    NucleicAcid,
    ProteinScene,
    Representation,
    SetOpacity,
    Text,
    Write,
)

DATA = Path(__file__).resolve().parent / "data"


class DNAStyles(ProteinScene):
    def construct(self):
        dna = NucleicAcid.from_file(DATA / "1bna.cif").cartoon(bases="slabs").center()
        dna.rotate(1.35, axis=(1, 0, 0)).rotate(-0.28, axis=(0, 0, 1))
        self.add(dna)
        self.camera.frame(dna, aspect=self.width / self.height).zoom(1.22)
        self.camera.depth_cue = 0.3
        self.camera.set_focus(dna, chain="A", residues=(5, 8), fstop=5.6)
        self.add(Text("DNA · 1BNA", position=(0.055, 0.88), font_size=30))
        title = Text("Base slabs", position=(0.055, 0.10), font_size=46)
        self.play(Write(title), self.camera.animate.orbit(theta=0.12), run_time=1)
        self.play(self.camera.animate.orbit(theta=0.18), run_time=2)
        for style, heading in (
            ("rings", "Filled base rings"),
            ("sticks", "Base sticks"),
            ("ladder", "Ladder rods"),
        ):
            self.play(
                FadeOut(title), BaseStyle(dna, style), self.camera.animate.orbit(theta=0.10), run_time=0.8
            )
            title = Text(heading, position=(0.055, 0.10), font_size=46)
            self.play(Write(title), self.camera.animate.orbit(theta=0.12), run_time=0.8)
            self.play(self.camera.animate.orbit(theta=0.18), run_time=1.4)
        region = dna.select(chain="A", residues=(5, 8))
        other = dna.select(chain="B")
        self.play(
            FadeOut(title),
            BaseStyle(dna, "slabs"),
            Colorize(region, "#ffd47b", residue_delay=0.12),
            SetOpacity(other, 0.16),
            run_time=1.5,
        )
        label = region.callout(
            "Residues 5–8",
            subtitle="Color and opacity follow the selection",
            position=(0.67, 0.25),
            font_size=34,
            color="#ffd47b",
        )
        self.play(Write(label), self.camera.animate.orbit(theta=0.12), run_time=1)
        self.play(Representation(dna, "ball_and_stick"), run_time=1.2)
        self.play(self.camera.animate.orbit(theta=0.20), run_time=2)
        self.play(FadeOut(label), SetOpacity(other, 1), Colorize(region, None), run_time=1)
        title = Text("Molecular surface", position=(0.055, 0.10), font_size=46)
        self.play(Representation(dna, "surface"), Write(title), run_time=1.5)
        self.play(self.camera.animate.orbit(theta=0.30), run_time=2.5)
        self.wait(0.5)
