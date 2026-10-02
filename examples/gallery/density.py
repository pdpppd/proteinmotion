"""An electron-density contour and a sliding density plane around a helix.

Input: ubiquitin, PDB 1UBQ, and its 2mFo–DFc map from PDBe (1ubq.ccp4). The
contour level is in σ of the full map; the slice samples map values on a
plane that moves through the cropped box.
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
    StudioLook,
    Text,
)

DATA = Path(__file__).resolve().parent.parent / "data"


class DensityMaps(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = (
            Protein.from_file(DATA / "1ubq.cif").ball_and_stick(atom_scale=0.25, bond_radius=0.11).center()
        )
        helix = protein.select(residues=(23, 34))
        (~helix).set_opacity(0.06)
        local = DensityMap.from_file(DATA / "1ubq.ccp4").crop(helix, padding=1.8)
        shell = local.isosurface(1.5, opacity=0.3, color="#7fc8ff", follow=protein)
        scale = ColorScale(-0.5, 2, colors=("#10243d", "#438ca4", "#f5df93"))
        section = local.slice("z", 0.15, scale=scale, opacity=0.9, follow=protein, resolution=96)
        self.camera.frame(helix, margin=1.6, screen_position=(0.6, 0.52))
        self.camera.orbit(theta=0.22, phi=0.18)
        self.add(protein, Text("Density maps", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text("isosurface(1.5 σ)", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        self.add(caption)
        self.play(FadeIn(shell), run_time=1)
        self.play(
            shell.animate.set_level(2.5),
            caption.animate.set_text("set_level(2.5 σ)"),
            self.camera.animate.orbit(0.3),
            run_time=2,
        )
        self.play(shell.animate.set_level(1.5), caption.animate.set_text("set_level(1.5 σ)"), run_time=1.5)
        self.play(
            FadeIn(section),
            FadeIn(ColorLegend(scale, title="Map value", position=(0.05, 0.8))),
            caption.animate.set_text('slice("z")'),
            run_time=1,
        )
        self.play(section.animate.set_slice(0.85), run_time=3.5)
        self.play(FadeOut(section), run_time=0.8)
