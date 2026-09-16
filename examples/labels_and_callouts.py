"""Vector writing and live amino-acid callouts on the real 2K39 NMR ensemble.

Run from a source checkout:
    proteinmotion render examples/labels_and_callouts.py ProteinLabels --fps 60 -o labels.mp4

NMR conformers are interpolated for illustration; their order is not physical time.
"""

from pathlib import Path

from proteinmotion import (
    FadeIn,
    FadeOut,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Representation,
    Text,
    Unwrite,
    Write,
    smooth,
)


class ProteinLabels(ProteinScene):
    def construct(self):
        loaded = Protein.from_file(Path(__file__).parent / "data/2k39.cif", chains="A")
        core = loaded.select(chain="A", residues=(1, 70), atoms="CA")
        p = Protein.from_trajectory(loaded.trajectory.aligned(indices=core.atom_indices)).cartoon().center()
        self.add(p)
        self.camera.frame(p, margin=1.05, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi = 0.65, 0.18
        self.camera.depth_cue = 0.35

        title = Text("Every residue has a story.", font_size=64, font="semibold", position=(0.065, 0.07))
        subtitle = Text(
            "Ubiquitin · PDB 2K39 · 116 NMR conformers", font_size=26, color="#a3b3c7", position=(0.067, 0.15)
        )
        footer = Text(
            "Interpolated NMR conformers · model order is not physical time", font_size=21, color="#74869e"
        ).to_corner("DL", buff=0.065)
        self.play(Write(title, stroke_width=1.8), run_time=2.2)
        self.play(FadeIn(subtitle), FadeIn(footer), run_time=0.6)

        # Use all selected atoms for a region centroid, or atoms='CA' for the backbone.
        helix = p.select(chain="A", residues=(23, 34), atoms="CA")
        tail = p.select(chain="A", residues=(71, 76), atoms="CA")
        helix_note = helix.callout(
            "α helix",
            subtitle="Residues 23–34",
            position=(0.075, 0.44),
            font_size=38,
            color="#f2ba67",
            tip="dot",
        )
        tail_note = tail.callout(
            "C-terminal tail",
            subtitle="Residues 71–76",
            position=(0.76, 0.35),
            font_size=38,
            color="#50e0d0",
            tip="arrow",
        )
        helix_glow = helix.highlight(style="sphere", color="#f2ba67", opacity=0.10, padding=2)
        tail_glow = tail.highlight(style="atoms", color="#50e0d0", opacity=0.32, padding=0.6)
        self.play(Write(helix_note), Write(tail_note), FadeIn(helix_glow), FadeIn(tail_glow), run_time=2)
        self.play(
            PlayTrajectory(p, start=0, end=16, state_easing=smooth),
            self.camera.animate.orbit(0.35),
            run_time=4,
        )
        self.play(
            Unwrite(helix_note), Unwrite(tail_note), FadeOut(helix_glow), FadeOut(tail_glow), run_time=1.2
        )

        # Automatic names use PDB residue numbers, insertion codes and chain IDs.
        # Explicit offsets (in 1080p design pixels) make this composition repeatable.
        self.play(Representation(p, "ball_and_stick"), run_time=0.8)
        labels = p.label_residues(
            chain="A",
            residues=[8, 44, 70],
            font_size=34,
            color="#f2ba67",
            offsets={8: (-270, -90), 44: (520, -100), 70: (400, 150)},
        )
        patch = p.select(chain="A", residues=[8, 44, 70], atoms="CA")
        patch_glow = patch.highlight(style="atoms", color="#f2ba67", opacity=0.45, padding=0.7)
        self.play(Write(labels, lag_ratio=0.09), FadeIn(patch_glow), run_time=2)
        self.play(
            PlayTrajectory(p, start=16, end=36, state_easing=smooth),
            self.camera.animate.orbit(0.4),
            run_time=5,
        )
        self.wait(1)


class WritingStudy(ProteinScene):
    """A close-up of actual glyph outlines, including Greek letters and ligatures."""

    def construct(self):
        title = Text("Protein motion", font_size=146, font="semibold", position=(0.10, 0.29))
        detail = Text(
            "α helix  ·  β sheet  ·  Cα backbone", font_size=54, color="#50e0d0", position=(0.105, 0.52)
        )
        self.play(Write(title, lag_ratio=0.12, stroke_width=2), run_time=4)
        self.play(Write(detail, lag_ratio=0.06), run_time=3)
        self.wait(1)
        self.play(Unwrite(detail), Unwrite(title), run_time=2)


if __name__ == "__main__":
    ProteinLabels(fps=60).render("labels.mp4")
