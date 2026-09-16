"""Actual GroEL/GroES coordinates: 21 chains, not a replicated synthetic system."""

from pathlib import Path

import numpy as np

from proteinmotion import Protein, ProteinScene, Representation, Rotate, linear

DATA = Path(__file__).parent / "data" / "1aon.cif"


class GroELComplex(ProteinScene):
    def construct(self):
        p = Protein.from_file(DATA).cartoon(color="chain").center()
        # Orient the chaperonin axis toward the vertical image axis.
        ca = np.array([p.positions[r.ca] for r in p.topology.residues if r.ca >= 0])
        _, _, axes = np.linalg.svd(ca - ca.mean(0), full_matrices=False)
        p.orientation = np.stack((axes[1], axes[0], axes[2]))
        if np.linalg.det(p.orientation) < 0:
            p.orientation[2] *= -1
        p.center()
        self.add(p)
        self.camera.frame(p, margin=1.05, aspect=self.width / self.height)
        self.camera.theta = 0.3
        self.camera.phi = 0.2
        self.camera.depth_cue = 0.45
        self.wait(0.5)
        self.play(Rotate(p, np.pi), run_time=4, rate_func=linear)
        self.play(Representation(p, "ball_and_stick"), run_time=1)
        self.play(Rotate(p, np.pi / 2), run_time=3, rate_func=linear)
        self.wait(0.5)


class GroELSubunit(ProteinScene):
    def construct(self):
        p = Protein.from_file(DATA, chains="A").cartoon().center()
        self.add(p)
        self.camera.frame(p, margin=1.05, aspect=self.width / self.height)
        self.camera.depth_cue = 0.45
        self.play(Rotate(p, 2 * np.pi), run_time=6, rate_func=linear)


if __name__ == "__main__":
    GroELComplex(fps=60).render("groel-complex.mp4")
