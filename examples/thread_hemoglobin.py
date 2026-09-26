"""Hemoglobin, one chain at a time: four wires thread human deoxyhemoglobin (PDB 4HHB).

The α subunits are drawn in rose and the β subunits in teal. Each chain's wire flies in
from its own side of the screen, enters at the C terminus and traces the backbone to the N
terminus; the chains arrive in turn. The cartoon and the four hemes then fade in. The
flight paths are seeded illustrations, not physical motion. The two phosphates, deposited
as P atoms only, are hidden.

    proteinmotion render examples/thread_hemoglobin.py ThreadHemoglobin --fps 60 -o hemoglobin.mp4
"""

from pathlib import Path

import numpy as np

from proteinmotion import (
    FadeIn,
    FadeOut,
    Protein,
    ProteinScene,
    Region,
    Text,
    Thread,
    Unthread,
    Unwrite,
    Write,
)

DATA = Path(__file__).parent / "data"
SUBUNITS = {"A": ("α1", "#ef8f8f"), "B": ("β1", "#5fc9c0"), "C": ("α2", "#d9707f"), "D": ("β2", "#4aa3c2")}
HEME, GOLD, MUTED = "#e0564f", "#f2ba67", "#a3b3c7"


def label_position(scene, region, push=0.2):
    """Screen position just outside the region, away from the image center."""
    from proteinmotion.annotations import project_region

    x, y = project_region(region, scene.camera, scene.width, scene.height)
    u, v = x / scene.width - 0.5, y / scene.height - 0.5
    length = max(np.hypot(u, v), 1e-6)
    return (
        float(np.clip(0.5 + u + push * u / length, 0.08, 0.88)),
        float(np.clip(0.5 + v + push * v / length, 0.2, 0.9)),
    )


class ThreadHemoglobin(ProteinScene):
    def construct(self):
        aspect = self.width / self.height
        p = Protein.from_file(DATA / "4hhb.cif").cartoon().center()
        p.select(resname="PO4").hide_atoms()
        # Tint each subunit's polymer atoms; wires take the same colors.
        hemes = p.select(resname="HEM")
        for chain, (_, color) in SUBUNITS.items():
            atoms = np.setdiff1d(p.select(chain=chain).atom_indices, hemes.atom_indices)
            Region(p, atoms).set_color(color)
        carbons = [i for i in hemes.atom_indices if p.topology.atoms[i].element == "C"]
        Region(p, np.array(carbons)).set_color(HEME)
        self.camera.frame(p, margin=1.15, aspect=aspect)
        self.camera.theta, self.camera.phi = 0.35, 0.22
        self.camera.depth_cue = 0.45

        title = Text("Hemoglobin, one chain at a time", font_size=56, font="semibold", position=(0.06, 0.07))
        subtitle = Text(
            "Human deoxyhemoglobin · PDB 4HHB · two α and two β subunits",
            font_size=26,
            color=MUTED,
            position=(0.062, 0.14),
        )
        self.play(Write(title, stroke_width=1.6), FadeIn(subtitle), run_time=1.2)

        # Four wires, one per chain, arriving in turn from four sides.
        self.play(
            Thread(
                p, self.camera, radius=0.42, head=2.2, glow=16.0, glow_brightness=1.2, stagger=0.16, seed=7
            ),
            self.camera.animate.orbit(0.25, 0.02),
            run_time=7.5,
        )

        labels = []
        for chain, (name, color) in SUBUNITS.items():
            region = p.select(chain=chain, atoms="CA")
            position = label_position(self, region)
            labels.append(region.callout(name, position=position, font_size=38, color=color))
        self.play(*(Write(label) for label in labels), self.camera.animate.orbit(0.3), run_time=1.6)
        self.play(self.camera.animate.orbit(0.35), run_time=1.6)
        self.play(*(Unwrite(label) for label in labels), run_time=0.8)

        # Each subunit holds one heme.
        glows = [
            p.select(chain=chain, resname="HEM").highlight(
                style="sphere", color=GOLD, opacity=0.55, padding=3.0
            )
            for chain in SUBUNITS
        ]
        note = Text("One heme in each subunit", font_size=30, color="#edf3fc", position=(0.06, 0.88))
        self.play(
            *(FadeIn(g) for g in glows), FadeIn(note), self.camera.animate.orbit(0.45, 0.05), run_time=2
        )
        self.play(*(FadeOut(g) for g in glows), FadeOut(note), self.camera.animate.orbit(0.35), run_time=1.6)

        # And out again, last chain first.
        self.play(
            Unthread(
                p, self.camera, radius=0.42, head=2.2, glow=16.0, glow_brightness=1.2, stagger=0.16, seed=12
            ),
            run_time=5.5,
        )
        self.play(Unwrite(title), FadeOut(subtitle), run_time=1)
        self.wait(0.3)
