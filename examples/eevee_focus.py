"""Calmodulin 1CLL: select a helix, fade its context, then pull lens focus.

Render with:
    proteinmotion render examples/eevee_focus.py HelixFocus --renderer eevee \
        --width 960 --height 540 --fps 60 -o helix-focus.mp4

Requires Blender 4.5+. The camera motion illustrates the deposited structure.
"""

from pathlib import Path

from proteinmotion import FocusPull, Protein, ProteinScene, SetOpacity, Text, Write

DATA = Path(__file__).resolve().parent / "data"


class HelixFocus(ProteinScene):
    def construct(self):
        protein = Protein.from_file(DATA / "1cll.cif", chains="A").cartoon(color="#7299b4").center()
        protein.rotate(0.9, axis=(0, 0, 1))
        helix = protein.select(chain="A", residues=(5, 19))
        helix.set_color("#ffb34c")
        # The context selection includes every atom outside the highlighted helix.
        context = protein.select(chain="A", residues=[4, *range(20, 149)])
        far_region = protein.select(chain="A", residues=(82, 92), atoms="CA")
        self.add(protein)
        self.camera.frame(protein, aspect=self.width / self.height).zoom(1.5)
        self.camera.set_focus(protein, chain="A", residues=(5, 19), atoms="CA", fstop=4)
        self.add(Text("Calmodulin · 1CLL", position=(0.055, 0.9), font_size=30))
        label = helix.callout(
            "Alpha helix",
            subtitle="Chain A · residues 5–19",
            position=(0.68, 0.13),
            color="#ffb34c",
            font_size=38,
        )
        self.play(Write(label), run_time=0.8)
        self.play(SetOpacity(context, 0.06), run_time=1)
        self.play(self.camera.animate.orbit(theta=0.28), run_time=2)
        self.wait(0.4)
        self.play(FocusPull(self.camera, far_region, fstop=2.8), SetOpacity(context, 0.35), run_time=1.4)
        self.wait(0.4)
        self.play(FocusPull(self.camera, helix, fstop=4), SetOpacity(context, 0.06), run_time=1.2)
        self.wait(0.8)
