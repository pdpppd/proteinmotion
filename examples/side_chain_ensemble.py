"""Ubiquitin side chains across the 2K39 NMR ensemble.

Leu8, Ile44 and Val70 form the hydrophobic patch that many ubiquitin-binding domains
recognize. Their side chains follow the deposited conformers, then every side chain
grows from the cartoon in N-to-C order. Model order is a visualization order, not a
measured time series; interpolation between models is illustrative.

Run from a source checkout:
    proteinmotion render examples/side_chain_ensemble.py SideChainEnsemble --fps 60 -o side-chains.mp4
"""

from pathlib import Path

from proteinmotion import (
    FadeIn,
    HideSideChains,
    PlayTrajectory,
    Protein,
    ProteinScene,
    ShowSideChains,
    Text,
    Unwrite,
    Write,
    smooth,
)

DATA = Path(__file__).parent / "data/2k39.cif"
GOLD, MUTED = "#f2ba67", "#a3b3c7"


class SideChainEnsemble(ProteinScene):
    def construct(self):
        loaded = Protein.from_file(DATA, chains="A")
        # Align the core, leaving the flexible C-terminal tail free to move.
        core = loaded.select(residues=(1, 70), atoms="CA")
        p = Protein.from_trajectory(loaded.trajectory.aligned(indices=core.atom_indices)).cartoon().center()
        self.add(p)
        self.camera.frame(p, margin=0.85, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi = 0.2, 0.35
        self.camera.depth_cue = 0.35

        title = Text("Side chains in an NMR ensemble", font_size=56, font="semibold", position=(0.06, 0.07))
        subtitle = Text(
            "Ubiquitin · PDB 2K39 · 116 conformers", font_size=26, color=MUTED, position=(0.062, 0.14)
        )
        self.play(Write(title, stroke_width=1.6), FadeIn(subtitle), run_time=1.6)

        # The hydrophobic patch, grown one residue at a time.
        patch = p.select(residues=[8, 44, 70])
        self.focus(patch, margin=1.7, run_time=1.8)
        self.play(ShowSideChains(patch, residue_delay=0.3), run_time=1.6)
        labels = p.label_residues(
            residues=[8, 44, 70],
            font_size=30,
            color=GOLD,
            include_chain=False,
            offsets={8: (-260, -90), 44: (280, -110), 70: (260, 120)},
        )
        self.play(Write(labels, lag_ratio=0.1), run_time=1.2)
        self.play(
            PlayTrajectory(p, start=0, end=20, state_easing=smooth),
            self.camera.animate.orbit(0.5),
            run_time=6,
        )
        self.play(Unwrite(labels), run_time=0.7)

        # All side chains, staggered N to C, while the conformers continue.
        self.focus(p, margin=0.85, run_time=1.5)
        self.play(
            ShowSideChains(p, residue_delay=0.03),
            PlayTrajectory(p, start=20, end=32, state_easing=smooth),
            self.camera.animate.orbit(0.6),
            run_time=4,
        )
        self.play(
            PlayTrajectory(p, start=32, end=48, state_easing=smooth),
            self.camera.animate.orbit(0.6),
            run_time=5,
        )
        self.play(HideSideChains(p, residue_delay=0.02, reverse=True), Unwrite(title), run_time=2.5)
        self.wait(0.4)
