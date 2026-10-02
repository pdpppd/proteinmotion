"""Two angles per residue: φ and ψ fold a polypeptide.

Act 1 builds an alanine pentapeptide as an extended β strand and turns φ, then ψ,
of Ala3. Act 2 sets every residue of Baldwin's helical peptide AEAAAKEAAAKEAAAKA
to φ = −57°, ψ = −47°: the chain winds into an α helix while its i → i+4
hydrogen bonds form. Act 3 copies
the φ/ψ angles of protein G residues 41–56 (PDB 1PGA) onto an extended chain,
which folds into the β-hairpin. Act 4 plots the 74 residues of ubiquitin (PDB 1UBQ).

Torsion changes rotate atoms about bonds, so bond lengths and angles never change.
The motion between the start and end angles is an illustration, not a folding
pathway; φ and ψ move along paths through allowed Ramachandran regions.

Run from a source checkout:
    proteinmotion render examples/torsions.py TorsionsFilm --fps 60 -o torsions.mp4
"""

from pathlib import Path

import numpy as np

from proteinmotion import (
    FadeIn,
    FadeOut,
    Focus,
    HydrogenBonds,
    Protein,
    ProteinScene,
    RamachandranPlot,
    Representation,
    SetTorsions,
    StudioLook,
    Text,
    Unwrite,
    Write,
)

DATA = Path(__file__).parent / "data"
GOLD, TEAL, MUTED = "#f5d477", "#50e0d0", "#a3b3c7"
PLOT = dict(position=(0.62, 0.16), size=(0.35, 0.68))
SCREEN = (0.31, 0.56)  # where the molecule sits, leaving the right side for the plot


def caption(text):
    return Text(text, font_size=30, color="#edf3fc", position=(0.05, 0.215))


class TorsionsFilm(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="clean")

    def frame_on(self, target, margin, *, follow=True):
        return Focus(
            self.camera,
            target,
            margin=margin,
            aspect=self.width / self.height,
            screen_position=SCREEN,
            follow=follow,
        )

    def construct(self):
        self.camera.depth_cue = 0.25

        # Act 1. One residue's two rotatable backbone bonds.
        short = Protein.build("AAAAA", "strand", hydrogens=True).ball_and_stick(
            atom_scale=0.26, bond_radius=0.11
        )
        middle = short.select(residues=3)
        self.add(short)
        self.camera.frame(short, margin=0.95, aspect=self.width / self.height, screen_position=SCREEN)
        self.camera.theta, self.camera.phi = 0.0, 0.35

        title = Text("Two angles per residue", font_size=60, font="semibold", position=(0.05, 0.07))
        subtitle = Text(
            "Alanine pentapeptide, built as an extended β strand",
            font_size=28,
            color=MUTED,
            position=(0.051, 0.145),
        )
        rama = RamachandranPlot(short, highlight=middle, **PLOT).with_panel()
        self.play(Write(title), FadeIn(subtitle), FadeIn(short), run_time=1.8)
        self.play(FadeIn(rama), self.camera.animate.orbit(0.25), run_time=1.5)

        phi = middle.torsion_marker("phi", color=GOLD, radius=1.5, font_size=34)
        psi = middle.torsion_marker("psi", color=TEAL, radius=1.5, font_size=34)
        note = caption("φ turns the chain about the N–Cα bond")
        self.play(Write(phi), Write(note), run_time=1.5)
        self.play(SetTorsions(middle, phi=-60, anchor="n"), run_time=3)
        self.wait(0.5)
        self.play(Unwrite(note), run_time=0.5)
        note = caption("ψ turns it about the Cα–C bond")
        self.play(Write(psi), Write(note), run_time=1.5)
        self.play(SetTorsions(middle, psi=-45, anchor="n"), run_time=3)
        self.wait(1)
        self.play(Unwrite(phi), Unwrite(psi), Unwrite(note), run_time=0.8)
        self.play(FadeOut(short), FadeOut(rama), FadeOut(subtitle), run_time=1)

        # Act 2. The same angles at every residue make an α helix.
        pep = Protein.build("AEAAAKEAAAKEAAAKA", "strand", hydrogens=True)
        pep.ball_and_stick(atom_scale=0.26, bond_radius=0.11)
        subtitle = Text(
            "Baldwin's peptide AEAAAKEAAAKEAAAKA, from an extended β strand",
            font_size=28,
            color=MUTED,
            position=(0.051, 0.145),
        )
        rama = RamachandranPlot(pep, **PLOT).with_panel()
        self.camera.theta, self.camera.phi = 0.0, 0.35
        note = caption("Set every residue to φ = −57°, ψ = −47°")
        self.play(
            FadeIn(pep), FadeIn(rama), FadeIn(subtitle), Write(note), self.frame_on(pep, 1.0), run_time=1.8
        )
        bonds = HydrogenBonds(pep, hydrogens="explicit")
        lines = bonds.highlight(
            mode="3d", endpoints="hydrogen_acceptor", color=GOLD, radius=0.06, dash_count=5, dash_ratio=0.65
        )
        self.add(lines)
        self.play(SetTorsions(pep, conformation="helix", stagger=0.55), run_time=7)
        # Turn the new helix upright so it is seen from the side.
        trace = pep.positions[[r.ca for r in pep.topology.residues]]
        axis = np.linalg.svd(trace - trace.mean(0))[2][0]
        axis *= np.sign(np.dot(axis, trace[-1] - trace[0]))
        up = np.array([0.0, 1.0, 0.0])
        turn = np.arccos(np.clip(np.dot(axis, up), -1, 1))
        self.play(
            Unwrite(note),
            pep.animate.rotate(turn, np.cross(axis, up)),
            self.frame_on(pep, 1.35),
            run_time=1.5,
        )
        note = caption("13 hydrogen bonds, each from C=O of residue i to N–H of i + 4")
        self.play(Write(note), self.camera.animate.orbit(0.6), run_time=3)
        self.play(Representation(pep, "cartoon"), FadeOut(lines), run_time=1.8)
        self.play(self.camera.animate.orbit(0.7, 0.1), run_time=3)
        self.play(FadeOut(pep), FadeOut(rama), Unwrite(note), FadeOut(subtitle), run_time=1.2)

        # Act 3. Angles copied from a crystal structure fold a β-hairpin.
        crystal = Protein.from_file(DATA / "1pga.cif").torsions(residues=(41, 56))
        hairpin = Protein.build("GEWTYDDATKTFTVTE", first=41).cartoon()
        hairpin.color_residues(GOLD, residues=(42, 46)).color_residues(TEAL, residues=(51, 55))
        subtitle = Text(
            "Protein G residues 41–56, PDB 1PGA", font_size=28, color=MUTED, position=(0.051, 0.145)
        )
        rama = RamachandranPlot(hairpin, **PLOT).with_panel()
        self.camera.theta, self.camera.phi = 0.0, 0.2
        self.play(FadeIn(hairpin), FadeIn(rama), FadeIn(subtitle), self.frame_on(hairpin, 1.0), run_time=1.5)
        note = caption("Copy φ and ψ from the crystal structure")
        self.play(Write(note), run_time=1.2)
        angles = dict(phi=np.nan_to_num(crystal.phi, nan=-120), psi=np.nan_to_num(crystal.psi, nan=140))
        self.play(SetTorsions(hairpin, **angles), run_time=6)
        backbone = HydrogenBonds(hairpin, hydrogens="backbone")
        cross = backbone.highlight(
            mode="3d",
            endpoints="hydrogen_acceptor",
            color="#edf3fc",
            radius=0.07,
            dash_count=5,
            between=(hairpin.select(residues=(41, 47)), hairpin.select(residues=(50, 56))),
        )
        self.play(Unwrite(note), self.frame_on(hairpin, 1.2), run_time=1.5)
        note = caption("Two strands pair through backbone hydrogen bonds: a β-hairpin")
        self.play(Write(note), Write(cross), run_time=1.8)
        self.play(self.camera.animate.orbit(0.9, 0.1), run_time=4)
        self.play(
            FadeOut(hairpin), FadeOut(cross), FadeOut(rama), Unwrite(note), FadeOut(subtitle), run_time=1.2
        )

        # Act 4. A real protein.
        ubiquitin = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        glycines = ubiquitin.select(resname="GLY")
        subtitle = Text("Ubiquitin, PDB 1UBQ", font_size=28, color=MUTED, position=(0.051, 0.145))
        rama = RamachandranPlot(ubiquitin, highlight=glycines, labels=False, **PLOT).with_panel()
        self.camera.theta, self.camera.phi = 0.4, 0.15
        self.play(
            FadeIn(ubiquitin), FadeIn(rama), FadeIn(subtitle), self.frame_on(ubiquitin, 1.0), run_time=1.5
        )
        note = caption("All 74 residues lie in favored regions; glycines (triangles) reach positive φ")
        self.play(Write(note), self.camera.animate.orbit(1.2), run_time=5)
        self.wait(1)
        self.play(Unwrite(note), Unwrite(title), FadeOut(subtitle), run_time=1.2)
        self.wait(0.3)
