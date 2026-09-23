"""Ligands, ions and side chains drawn over cartoons, from deposited structures.

Act 1: calmodulin (PDB 1CLL) with its four Ca²⁺ ions, which cartoons show by
default. The side chains that coordinate each ion grow from the cartoon at Cα.
Act 2: one GroEL subunit (PDB 1AON, chain A) with its bound ADP and Mg²⁺, and the
side chains that line the nucleotide pocket.

Coordinating residues are selected geometrically (any atom within 3 Å of an ion, or
4 Å of ADP/Mg²⁺); this is a distance criterion, not an interaction energy. Residues
more than 8 Å from the nucleotide are faded for visibility.

Run from a source checkout:
    proteinmotion render examples/binding_sites_film.py BindingSites --fps 60 -o binding-sites.mp4
"""

from pathlib import Path

import numpy as np

from proteinmotion import (
    FadeIn,
    FadeOut,
    Focus,
    HideSideChains,
    Protein,
    ProteinScene,
    Region,
    SetOpacity,
    ShowSideChains,
    Text,
    Unwrite,
    Write,
)

DATA = Path(__file__).parent / "data"
GOLD, TEAL, MUTED = "#f2ba67", "#50e0d0", "#a3b3c7"


class BindingSites(ProteinScene):
    def construct(self):
        aspect = self.width / self.height
        self.camera.depth_cue = 0.35

        # Act 1. Ions and ligands outside the traced chain draw as ball-and-stick detail.
        calmodulin = Protein.from_file(DATA / "1cll.cif").cartoon().center()
        self.add(calmodulin)
        self.camera.frame(calmodulin, margin=1.0, aspect=aspect)
        self.camera.theta, self.camera.phi = 0.4, 0.15

        title = Text("Ligands, ions and side chains", font_size=60, font="semibold", position=(0.06, 0.07))
        subtitle = Text(
            "Calmodulin · PDB 1CLL · four Ca²⁺ ions", font_size=26, color=MUTED, position=(0.062, 0.14)
        )
        self.play(Write(title, stroke_width=1.6), FadeIn(subtitle), run_time=1.8)
        self.play(self.camera.animate.orbit(0.5), run_time=2.5)

        ions = calmodulin.select(ions=True)
        ion = calmodulin.select(ions=True, residues=149)
        site = calmodulin.select(within=3.0, of=ion)
        note = ion.callout("Ca²⁺", subtitle="EF-hand 1", position=(0.72, 0.30), font_size=40, color=TEAL)
        self.play(Write(note), run_time=1.2)
        self.focus(site | ion, margin=1.25, run_time=2)
        self.play(Unwrite(note), run_time=0.6)

        # Side chains grow residue by residue, joined to the cartoon at Cα.
        self.play(ShowSideChains(site, residue_delay=0.25), run_time=2.4)
        labels = calmodulin.label_residues(
            residues=[20, 22, 24, 26, 31],
            font_size=30,
            color=GOLD,
            offsets={20: (-260, 90), 22: (-300, -40), 24: (-120, -170), 26: (260, -130), 31: (300, 70)},
        )
        self.play(Write(labels, lag_ratio=0.1), run_time=1.6)
        self.play(self.camera.animate.orbit(0.7, 0.1), run_time=4)
        self.play(Unwrite(labels), run_time=0.8)

        # All four sites at once, staggered along the chain.
        others = calmodulin.select(within=3.0, of=ions)
        self.focus(calmodulin, margin=1.0, run_time=2)
        self.play(ShowSideChains(others, residue_delay=0.08), self.camera.animate.orbit(0.4), run_time=3)
        self.play(self.camera.animate.orbit(0.6), run_time=3)
        self.play(FadeOut(calmodulin), FadeOut(subtitle), HideSideChains(others, reverse=True), run_time=1.2)

        # Act 2. A bound nucleotide and the residues lining its pocket.
        groel = Protein.from_file(DATA / "1aon.cif", chains="A").cartoon().center()
        adp = groel.select(ligands=True) | groel.select(ions=True)
        pocket = groel.select(within=4.0, of=adp)
        # Residues more than 8 Å from the nucleotide fade back to reveal the pocket.
        shell = groel.select(within=8.0, of=adp) | adp
        distant = Region(groel, np.setdiff1d(np.arange(len(groel.topology.atoms)), shell.atom_indices))
        subtitle = Text(
            "GroEL subunit · PDB 1AON chain A · ADP and Mg²⁺",
            font_size=26,
            color=MUTED,
            position=(0.062, 0.14),
        )
        self.play(
            FadeIn(groel),
            FadeIn(subtitle),
            Focus(self.camera, groel, margin=1.0, aspect=aspect),
            run_time=1.5,
        )
        self.play(self.camera.animate.orbit(0.5), run_time=2)
        self.play(
            Focus(self.camera, pocket | adp, margin=1.2, aspect=aspect),
            SetOpacity(distant, 0.2),
            run_time=2.2,
        )
        note = adp.callout(
            "ADP · Mg²⁺", subtitle="Nucleotide pocket", position=(0.70, 0.26), font_size=40, color=TEAL
        )
        self.play(Write(note), ShowSideChains(pocket, residue_delay=0.1), run_time=2.5)
        self.play(self.camera.animate.orbit(0.9, 0.12), run_time=5)
        self.play(Unwrite(note), Unwrite(title), FadeOut(subtitle), run_time=1.2)
        self.wait(0.5)
