"""Calmodulin -> troponin C using contact-guided CA correspondence."""

from pathlib import Path

import numpy as np

from proteinmotion import BackboneMorph, ContactMatch, Protein, ProteinScene, Rotate

DATA = Path(__file__).parent / "data"


class BackboneDemo(ProteinScene):
    show_atoms = False

    def construct(self):
        source = Protein.from_file(DATA / "1cll.cif", chains="A").cartoon(color="#56d8c0").center()
        target = Protein.from_file(DATA / "1ncx.cif", chains="A").cartoon(color="#f2ba67").center()
        if self.show_atoms:
            source.ball_and_stick(atom_scale=0.30, bond_radius=0.14)
            target.ball_and_stick(atom_scale=0.30, bond_radius=0.14)
        match = ContactMatch.load(DATA / "calmodulin-troponin-match.json")
        self.add(source)
        self.camera.frame(source, margin=1.12, aspect=self.width / self.height)
        self.camera.theta = 0.4
        self.camera.phi = 0.15
        self.camera.depth_cue = 0.4
        self.wait(1)
        self.play(
            BackboneMorph(
                source,
                target,
                match=match,
                residue_delay=0.025,
                fade_out=(0, 0.35),
                fade_in=(0.65, 1),
                align=True,
            ),
            run_time=6,
        )
        self.wait(1)
        self.play(Rotate(target, np.pi * 0.55), run_time=3)
        self.wait(0.5)


class BallAndStickDemo(BackboneDemo):
    """The same correspondence, camera and timing, with all selected heavy atoms shown."""

    show_atoms = True


if __name__ == "__main__":
    BackboneDemo(fps=60).render("backbone-morph.mp4")
