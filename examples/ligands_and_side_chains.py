"""Show 1CLL Ca²⁺ ions on the cartoon and grow the side chains that coordinate them."""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, ShowSideChains, Unwrite, Write

DATA = Path(__file__).parent / "data"


class LigandsAndSideChains(ProteinScene):
    def construct(self):
        # Ions and ligands outside the traced chain draw as ball-and-stick detail.
        p = Protein.from_file(DATA / "1cll.cif").cartoon().center()
        self.add(p)
        self.camera.frame(p, margin=1.0, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi = 0.4, 0.15
        self.camera.depth_cue = 0.3
        self.play(self.camera.animate.orbit(0.4), run_time=1.5)

        # Whole residues with an atom within 3 Å of the first Ca²⁺ ion.
        ion = p.select(ions=True, residues=149)
        site = p.select(within=3.0, of=ion)
        self.focus(site | ion, margin=1.3, run_time=1.5)
        self.play(ShowSideChains(site, residue_delay=0.2), run_time=1.8)
        labels = p.label_residues(
            residues=[20, 22, 24, 26, 31],
            font_size=30,
            color="#f2ba67",
            offsets={20: (-260, 90), 22: (-300, -40), 24: (-120, -170), 26: (260, -130), 31: (300, 70)},
        )
        self.play(Write(labels, lag_ratio=0.1), run_time=1.2)
        self.play(self.camera.animate.orbit(0.5), run_time=2.5)
        self.play(Unwrite(labels), run_time=0.6)

        # Every residue within 3 Å of any of the four ions, staggered N to C.
        self.focus(p, margin=1.0, run_time=1.5)
        self.play(
            ShowSideChains(p.select(within=3.0, of=p.select(ions=True)), residue_delay=0.04),
            self.camera.animate.orbit(0.5),
            run_time=2.5,
        )
