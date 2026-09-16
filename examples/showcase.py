"""A real ubiquitin structure; the animated trajectory is synthetic, not an MD simulation."""

from pathlib import Path

import numpy as np

from proteinmotion import (
    Deform,
    FadeIn,
    Morph,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Representation,
    Rotate,
    Trajectory,
    linear,
)

DATA = Path(__file__).parent / "data" / "1ubq.cif"


def synthetic_motion(xyz, phase):
    centered = xyz - xyz.mean(0)
    x, y, z = centered.T
    displacement = np.column_stack(
        (1.6 * np.sin(y * 0.12 + phase), 0.9 * np.sin(z * 0.13 + phase), 1.6 * np.sin(x * 0.10 + phase))
    )
    return xyz + displacement


class Showcase(ProteinScene):
    def construct(self):
        p = Protein.from_file(DATA).cartoon().center().rotate(-0.4, axis=(0, 0, 1))
        self.add(p)
        self.camera.frame(p, margin=1.05, aspect=self.width / self.height)
        self.camera.theta = -0.3
        self.camera.phi = 0.12
        self.camera.depth_cue = 0.48
        self.play(FadeIn(p), run_time=0.8)
        self.play(Rotate(p, np.pi * 0.7), run_time=3.2)
        self.play(Representation(p, "ribbon"), run_time=1.2)
        self.play(Rotate(p, 0.9, axis=(1, 0.5, 0)), run_time=2)
        self.play(Representation(p, "ball_and_stick"), run_time=1.2)
        self.play(self.camera.animate.orbit(theta=0.9, phi=0.15), run_time=2)
        self.play(Representation(p, "cartoon"), run_time=1.2)
        original = p.positions.copy()
        self.play(Deform(p, lambda xyz: xyz.mean(0) + (xyz - xyz.mean(0)) * [1.22, 0.9, 1]), run_time=1.5)
        self.play(Morph(p, original, align=False), run_time=1.5)
        frames = [synthetic_motion(original, phase) for phase in np.linspace(0, 2 * np.pi, 61)]
        trajectory = Trajectory(frames, topology=p.topology)
        self.play(Morph(p, frames[0], align=False), run_time=0.6)
        self.play(PlayTrajectory(p, trajectory), Rotate(p, 1.5), run_time=4, rate_func=linear)
        self.play(Morph(p, original, align=False), run_time=0.6)
        self.wait(0.7)


class Gallery(ProteinScene):
    def construct(self):
        p = Protein.from_file(DATA).center().rotate(-0.4, axis=(0, 0, 1))
        a = p.copy().cartoon().shift((-36, 0, 0))
        b = p.copy().ribbon(color="rainbow")
        c = p.copy().ball_and_stick().shift((36, 0, 0))
        self.add(a, b, c)
        self.camera.theta = 0
        self.camera.phi = 0
        self.camera.target = np.array([0.0, 0.0, 0.0])
        self.camera.radius = 28
        self.camera.distance = 115
        self.camera.depth_cue = 0.4
        self.play(
            Rotate(a, 2 * np.pi), Rotate(b, 2 * np.pi), Rotate(c, 2 * np.pi), run_time=12, rate_func=linear
        )


if __name__ == "__main__":
    Showcase().render("showcase.mp4")
