"""Yeast tRNA: base rings, residue focus, B-factor colors and surface.

Input: RCSB PDB 1EHZ. Modified bases use deposited parent-residue annotations.
Render: proteinmotion render examples/rna_styles.py RNAStyles --fps 60 -o rna.mp4
"""

from pathlib import Path

from proteinmotion import (
    BaseStyle,
    ColorByProperty,
    Colorize,
    ColorLegend,
    ColorScale,
    FadeOut,
    NucleicAcid,
    ProteinScene,
    Representation,
    ResidueValues,
    SetOpacity,
    Text,
    Write,
)

DATA = Path(__file__).resolve().parent / "data"


class RNAStyles(ProteinScene):
    def construct(self):
        rna = NucleicAcid.from_file(DATA / "1ehz.cif").cartoon(bases="rings").center()
        rna.rotate(-0.5, axis=(0, 1, 0)).rotate(0.15, axis=(0, 0, 1))
        self.add(rna)
        self.camera.frame(rna, aspect=self.width / self.height).zoom(1.3)
        self.camera.depth_cue = 0.3
        self.camera.set_focus(rna, chain="A", residues=(32, 38), fstop=5.6)
        self.add(Text("Transfer RNA · 1EHZ", position=(0.055, 0.10), font_size=30))
        loop = rna.select(chain="A", residues=(32, 38))
        context = rna.select(chain="A", residues=[*range(1, 32), *range(39, 77)])
        self.play(self.camera.animate.orbit(theta=0.35), run_time=3)
        self.play(Colorize(loop, "#ffd47b", residue_delay=0.08), SetOpacity(context, 0.18), run_time=1.5)
        callout = loop.callout(
            "Anticodon loop", subtitle="Residues 32–38", position=(0.67, 0.20), font_size=38, color="#ffd47b"
        )
        self.play(Write(callout), self.camera.animate.orbit(theta=0.15), run_time=1.5)
        self.play(BaseStyle(rna, "slabs"), self.camera.animate.orbit(theta=0.15), run_time=1.5)
        self.play(FadeOut(callout), SetOpacity(context, 1), Colorize(loop, None), run_time=1)
        values = ResidueValues.b_factors(rna)  # C4′ values, including modified nucleotides.
        scale = ColorScale.from_values(values, colors=("#5ca7dc", "#77d1c0", "#f6c675"))
        legend = ColorLegend(scale, title="Deposited B factor", unit="Å²")
        self.play(ColorByProperty(rna, values, scale=scale), Write(legend), run_time=1.5)
        self.play(Representation(rna, "surface"), self.camera.animate.orbit(theta=0.15), run_time=1.5)
        self.play(self.camera.animate.orbit(theta=0.30), run_time=2.5)
        self.wait(0.5)
