"""1UBQ electron density from PDBe, with a contour and a moving density slice.

Source: https://www.ebi.ac.uk/pdbe/coordinates/files/1ubq.ccp4
Retrieved 2026-09-17. The supplied map covers a complete crystallographic unit
cell. Cropping extends periodic data across the cell boundary when needed.
Sigma contours use the mean and standard deviation of the original map.
"""

from pathlib import Path

from proteinmotion import (
    ColorLegend,
    ColorScale,
    DensityMap,
    FadeIn,
    FadeOut,
    Protein,
    ProteinScene,
    Text,
)

DATA = Path(__file__).parent / "data"


class DensityMaps(ProteinScene):
    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").ball_and_stick().center()
        protein.set_residue_opacity(0.08)
        helix = protein.select(chain="A", residues=(23, 34))
        helix.set_opacity(1)
        density = DensityMap.from_file(DATA / "1ubq.ccp4")
        local = density.crop(helix, padding=2.5)
        shell = local.isosurface(1.5, opacity=0.28, follow=protein)
        scale = ColorScale(-0.5, 2, colors=("#10243d", "#438ca4", "#f5df93"))
        section = local.slice("z", 0.15, scale=scale, follow=protein, resolution=96)
        self.add(protein, shell)
        self.camera.frame(helix, margin=1.55)
        self.camera.orbit(theta=0.22, phi=0.18)
        self.add(Text("Electron density", position=(0.06, 0.07), font_size=42))
        self.add(Text("1UBQ · helix 23–34 · PDBe map", position=(0.06, 0.13), font_size=25))
        self.play(FadeIn(shell), run_time=1)
        self.play(shell.animate.set_level(2.5), self.camera.animate.orbit(theta=0.25), run_time=2)
        self.play(shell.animate.set_level(1.5), run_time=1.5)
        self.play(FadeIn(section), run_time=1)
        self.add(ColorLegend(scale, title="Map value", position=(0.06, 0.81)))
        self.play(section.animate.set_slice(0.85), run_time=3)
        self.play(FadeOut(section), run_time=1)
        self.wait(0.5)
