"""Residue styling, surfaces, live rulers, hydrogen bonds and screened electrostatics.

Run either scene from the checkout. All analyses below use actual 2K39 coordinates.
NMR ordering is illustrative; inferred H and formal charges are labeled explicitly.
"""

from pathlib import Path

import numpy as np

from proteinmotion import (
    Colorize,
    Distance,
    Electrostatics,
    FadeIn,
    FadeOut,
    Focus,
    HydrogenBonds,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Region,
    Representation,
    SetOpacity,
    Text,
    Write,
    smooth,
)


def ubiquitin():
    loaded = Protein.from_file(Path(__file__).parent / "data/2k39.cif", chains="A")
    core = loaded.select(chain="A", residues=(1, 70), atoms="CA")
    return Protein.from_trajectory(loaded.trajectory.aligned(indices=core.atom_indices)).center()


def caption(text, y=0.155, color="#9eafc4"):
    return Text(text, font_size=26, color=color, position=(0.065, y))


class StylingAndSurface(ProteinScene):
    def construct(self):
        p = ubiquitin().surface(kind="ses", resolution=0.5).cartoon(color="#8299ad")
        helix = p.select(chain="A", residues=(23, 34))
        strand = p.select(chain="A", residues=(2, 7))
        rest = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), (helix | strand).atom_indices))
        self.add(p)
        self.camera.frame(p, margin=1.08, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi, self.camera.depth_cue = 0.6, 0.18, 0.3
        title = Text("Color is part of the story.", font_size=64, font="semibold", position=(0.065, 0.07))
        note = caption("Residue colors · staggered transitions · local transparency")
        footer = Text(
            "Ubiquitin · 2K39 · interpolated conformers are not physical time", font_size=21, color="#74869e"
        ).to_corner("DL", buff=0.065)
        self.play(Write(title), FadeIn(note), FadeIn(footer), run_time=1.7)
        self.play(
            Colorize(helix, "#50e0d0", residue_delay=0.09),
            Colorize(strand, "#f2ba67", residue_delay=0.15),
            run_time=2.5,
        )
        helix_note = helix.callout(
            "α helix", subtitle="Residues 23–34", position=(0.075, 0.43), color="#50e0d0", font_size=36
        )
        strand_note = strand.callout(
            "β strand", subtitle="Residues 2–7", position=(0.77, 0.35), color="#f2ba67", font_size=36
        )
        self.play(Write(helix_note), Write(strand_note), run_time=1.2)
        self.play(SetOpacity(rest, 0.12, residue_delay=0.008), run_time=1.3)
        self.play(self.camera.animate.orbit(0.35), run_time=2)
        self.play(SetOpacity(rest, 1), FadeOut(helix_note), FadeOut(strand_note), run_time=1)
        self.play(Representation(p, "ball_and_stick"), run_time=1)
        ruler = Distance(
            p.select(residues=23),
            p.select(residues=34),
            mode="2d",
            color="#50e0d0",
            font_size=32,
            prefix="Cα · ",
            style="dashed",
        )
        self.play(Write(ruler), run_time=1.5)
        self.play(PlayTrajectory(p, start=0, end=12, state_easing=smooth), run_time=3)
        self.play(FadeOut(ruler), FadeOut(note), run_time=0.6)
        surface_note = caption(
            "Solvent-excluded surface · 1.4 Å probe · 0.5 Å grid · meshes follow the coordinates"
        )
        self.play(Representation(p, "surface"), Write(surface_note), run_time=1.4)
        self.play(
            PlayTrajectory(p, start=12, end=24, state_easing=smooth),
            self.camera.animate.orbit(0.4),
            run_time=3,
        )
        self.play(SetOpacity(rest, 0.2, residue_delay=0.008), run_time=1.2)
        self.play(self.camera.animate.orbit(0.4), run_time=2)
        self.play(SetOpacity(rest, 1), run_time=0.8)
        self.wait(0.6)


class InteractionsAndDistances(ProteinScene):
    def construct(self):
        p = ubiquitin().ball_and_stick(atom_scale=0.27)
        helix = p.select(residues=(23, 34))
        rest = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), helix.atom_indices))
        rest.set_opacity(0.08)
        self.add(p)
        self.camera.frame(helix, margin=1.75, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi, self.camera.depth_cue = 0.8, 0.18, 0.15
        title = Text("Measure the connections.", font_size=64, font="semibold", position=(0.065, 0.07))
        note = caption("3D hydrogen-bond rulers · depth-tested lines · live donor–acceptor distances")
        method = Text(
            "D–A ≤ 3.5 Å · D–H–A ≥ 150° · virtual backbone H · NMR interpolation is illustrative",
            font_size=21,
            color="#74869e",
        ).to_corner("DL", buff=0.065)
        self.play(Write(title), FadeIn(note), FadeIn(method), run_time=1.7)
        hbonds = HydrogenBonds(
            p, donors=p.select(residues=(23, 34), atoms="N"), max_distance=3.5, min_angle=150
        )
        h3 = hbonds.highlight(
            mode="3d",
            show_distances=True,
            max_pairs=1,
            color="#50e0d0",
            font_size=26,
            radius=0.09,
            style="dashed",
            dash_count=7,
            follow_opacity=False,
        )
        self.play(Write(h3), run_time=1.6)
        self.play(
            PlayTrajectory(p, start=0, end=10, state_easing=smooth),
            self.camera.animate.orbit(0.18),
            run_time=3.5,
        )
        h2 = hbonds.highlight(
            mode="2d",
            show_distances=True,
            max_pairs=1,
            color="#f2ba67",
            font_size=26,
            line_width=2,
            style="dashed",
            dash_count=7,
            follow_opacity=False,
        )
        note2 = caption("2D hydrogen-bond rulers · lines remain visible over the model")
        self.play(FadeOut(h3), FadeOut(note), Write(h2), Write(note2), run_time=1.4)
        self.play(PlayTrajectory(p, start=10, end=16, state_easing=smooth), run_time=3)
        self.play(FadeOut(h2), FadeOut(note2), FadeOut(method), run_time=0.6)

        # Use charges=<atom-ordered array> or Electrostatics.from_pqr(p, "prepared.pqr")
        # for a prepared charge model. Formal templates here are only illustrative.
        field = Electrostatics(
            p, charges="formal", dielectric=80, screening_length=8, cutoff=6, min_energy=0.08
        )
        strongest = field.pairs[0]
        endpoints = Region(p, [strongest.a, strongest.b])
        residue_numbers = [p.topology.atoms[i].resid for i in (strongest.a, strongest.b)]
        charged = p.select(chain="A", residues=residue_numbers)
        other = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), charged.atom_indices))
        charge_note = caption("Screened electrostatic contacts · blue: attraction · coral: repulsion")
        charge_method = Text(
            "Illustrative formal side-chain charges · εr = 80 · screening length = 8 Å · not Poisson–Boltzmann",
            font_size=21,
            color="#74869e",
        ).to_corner("DL", buff=0.065)
        self.play(
            SetOpacity(charged, 1),
            SetOpacity(other, 0.035),
            Focus(self.camera, endpoints, margin=4.8, aspect=self.width / self.height),
            Write(charge_note),
            FadeIn(charge_method),
            run_time=2,
        )
        contacts = field.highlight(
            mode="2d",
            region=charged,
            max_pairs=3,
            show_distances=True,
            font_size=29,
            style="solid",
            line_width=2,
            follow_opacity=False,
        )
        self.play(Write(contacts), run_time=1.6)
        self.play(
            self.camera.animate.orbit(0.3),
            run_time=4,
        )
        self.wait(0.8)


if __name__ == "__main__":
    StylingAndSurface(fps=60).render("styling-and-surface.mp4")
    InteractionsAndDistances(fps=60).render("interactions-and-distances.mp4")
