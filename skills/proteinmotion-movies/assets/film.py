"""An editable first film using deposited ubiquitin, PDB 1UBQ.

Change STRUCTURE and the region/label selections together for a different protein.
Render: proteinmotion render film.py ProteinMovie --fps 60 -o film.mp4
Data: https://www.rcsb.org/structure/1UBQ
"""

from pathlib import Path

import numpy as np

from proteinmotion import (
    Colorize,
    FadeIn,
    FadeOut,
    Focus,
    Protein,
    ProteinScene,
    Region,
    Representation,
    Rotate,
    SetOpacity,
    Text,
    Unwrite,
    Write,
)

STRUCTURE = Path(__file__).parent / "1ubq.cif"


class ProteinMovie(ProteinScene):
    def construct(self):
        p = (
            Protein.from_file(STRUCTURE, chains="A")
            .ball_and_stick(atom_scale=0.28, bond_radius=0.10)
            .cartoon()
            .center()
        )
        helix = p.select(residues=(23, 34))
        rest = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), helix.atom_indices))
        self.add(p)
        self.camera.frame(p, margin=1.35, aspect=self.width / self.height)
        self.camera.depth_cue = 0.3
        title = Text("A protein in motion", font_size=58, font="semibold", position=(0.06, 0.07))
        note = helix.callout(
            "α helix",
            subtitle="Ubiquitin · residues 23–34",
            font_size=34,
            color="#50e0d0",
            position=(0.07, 0.43),
        )
        box = helix.highlight(style="box", color="#50e0d0", opacity=0.5, padding=1.0)
        self.play(FadeIn(p), Write(title), run_time=1.3)
        self.play(Rotate(p, 0.8), run_time=2)
        self.play(Colorize(helix, "#50e0d0", residue_delay=0.05), FadeIn(box), Write(note), run_time=2)
        self.play(
            Focus(self.camera, helix, margin=1.8, aspect=self.width / self.height),
            SetOpacity(rest, 0.08),
            run_time=2.4,
        )
        self.play(Representation(p, "ball_and_stick"), FadeOut(box), run_time=1.5)
        self.play(self.camera.animate.orbit(0.5), run_time=2)
        self.play(Unwrite(note), run_time=1)
        self.play(
            Focus(self.camera, p, margin=1.35, aspect=self.width / self.height), SetOpacity(p, 1), run_time=2
        )
        self.play(Representation(p, "ribbon"), run_time=1.5)
        self.play(Rotate(p, 0.4), run_time=1.8)
        self.play(FadeOut(p), Unwrite(title), run_time=1.2)


if __name__ == "__main__":
    ProteinMovie(width=1920, height=1080, fps=60).render(Path(__file__).with_suffix(".mp4"))
