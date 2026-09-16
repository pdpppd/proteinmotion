"""Real ubiquitin NMR conformers (2K39), camera focus and moving 3D region highlights.

Deposited model order is a visualization order, not a measured time series.
"""

from pathlib import Path

from proteinmotion import (
    FadeIn,
    FadeOut,
    Focus,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Representation,
    smooth,
)

DATA = Path(__file__).parent / "data/2k39.cif"


def ubiquitin():
    loaded = Protein.from_file(DATA, chains="A")
    # Align the core, leaving the flexible C-terminal tail free to move.
    core = loaded.select(chain="A", residues=(1, 70), atoms="CA")
    aligned = loaded.trajectory.aligned(indices=core.atom_indices)
    return Protein.from_trajectory(aligned).cartoon().center()


class NMRStates(ProteinScene):
    """All 116 deposited states, smoothly interpolated at five transitions per second."""

    show_atoms = False

    def construct(self):
        p = ubiquitin()
        if self.show_atoms:
            p.ball_and_stick()
        self.add(p)
        self.camera.frame(p, margin=1.25, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi = 0.7, 0.18
        self.camera.depth_cue = 0.4
        self.wait(0.5)
        self.play(PlayTrajectory(p, state_easing=smooth), run_time=23)
        self.wait(0.5)


class NMRAtoms(NMRStates):
    show_atoms = True


class RegionTour(ProteinScene):
    """Gold: helix 23–34; cyan: tail 71–76. Highlights follow the moving model."""

    def construct(self):
        p = ubiquitin()
        helix = p.select(chain="A", residues=(23, 34))
        tail = p.select(chain="A", residues=(71, 76))
        sphere = helix.highlight(style="sphere", color="#f2ba67", opacity=0.13, padding=1.4)
        box = helix.highlight(style="box", color="#f2ba67", padding=1.6, line_width=0.09)
        halo = tail.highlight(style="atoms", color="#50e0d0", opacity=0.38, padding=0.4)
        tail_box = tail.highlight(style="box", color="#50e0d0", padding=1.4, line_width=0.09)
        self.add(p)
        self.camera.frame(p, margin=1.2, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi = 0.7, 0.18
        self.camera.depth_cue = 0.38
        self.wait(0.5)
        self.play(PlayTrajectory(p, start=0, end=20, state_easing=smooth), run_time=4)
        self.play(FadeIn(sphere), FadeIn(box), run_time=0.8)
        self.focus(helix, margin=1.8, run_time=1.8)
        self.play(PlayTrajectory(p, start=20, end=35, state_easing=smooth), run_time=4.5)
        self.play(
            FadeOut(sphere),
            FadeOut(box),
            Focus(self.camera, p, margin=1.2, aspect=self.width / self.height),
            run_time=1.8,
        )
        self.play(FadeIn(halo), FadeIn(tail_box), run_time=0.8)
        self.focus(tail, margin=2.0, run_time=1.8)
        self.play(Representation(p, "ball_and_stick"), run_time=0.8)
        self.play(PlayTrajectory(p, start=35, end=45, state_easing=smooth), run_time=5)
        self.play(
            FadeOut(halo),
            FadeOut(tail_box),
            Focus(self.camera, p, margin=1.2, aspect=self.width / self.height),
            run_time=1.8,
        )
        self.wait(0.5)


if __name__ == "__main__":
    RegionTour(fps=60).render("nmr-region-tour.mp4")
