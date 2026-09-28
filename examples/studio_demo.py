"""Prototype: the studio renderer, in the style of the ProteinMotion intro film.

Glossy lighting with a mint rim light, ambient occlusion, bloom on the threading tip,
and molecular surfaces computed on the GPU every frame, so they follow NMR playback and
deformation in real time. Uses ubiquitin (PDB 2K39, NMR models) and GroEL–GroES (1AON).

    proteinmotion render examples/studio_demo.py StudioDemo --renderer studio --fps 60 -o studio.mp4
"""

from pathlib import Path

import numpy as np

from proteinmotion import (
    Conceal,
    Deform,
    FadeIn,
    FadeOut,
    Focus,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Region,
    Representation,
    Reveal,
    Text,
    Thread,
    Unwrite,
    Write,
    smooth,
)

DATA = Path(__file__).parent / "data"
INK, MINT, CORAL, CREAM, MUTED = "#0B1210", "#8FE3C0", "#FF7F5C", "#F3EEE3", "#8FA89C"


def palette(p):
    """Intro palette: mint helices, coral strands, cream coils."""
    colors = {"H": MINT, "E": CORAL, "C": CREAM}
    for code, color in colors.items():
        atoms = [
            i
            for i, a in enumerate(p.topology.atoms)
            if p.topology.residues[a.residue_index].secondary == code
            and p.topology.residue_categories[a.residue_index] == "polymer"
        ]
        if atoms:
            Region(p, np.array(atoms)).set_color(color)
    return p


def upright(p):
    xyz = p.positions[[r.ca for r in p.topology.residues if r.ca >= 0]]
    axis = np.linalg.svd(xyz - xyz.mean(0), full_matrices=False)[2][0]
    axis = axis if axis[1] >= 0 else -axis
    turn = np.cross(axis, [0, 1, 0])
    if np.linalg.norm(turn) > 1e-6:
        p.rotate(float(np.arccos(np.clip(axis[1], -1, 1))), turn / np.linalg.norm(turn))
    return p.center()


class StudioDemo(ProteinScene):
    def __init__(self, **kwargs):
        kwargs.setdefault("background", INK)
        super().__init__(**kwargs)

    def construct(self):
        aspect = self.width / self.height
        self.camera.depth_cue = 0.45

        def caption(title, detail):
            return (
                Text(title, font_size=54, font="semibold", color=CREAM, position=(0.06, 0.74)),
                Text(detail, font_size=26, color=MUTED, position=(0.062, 0.82)),
            )

        # 1. Ubiquitin threads in; the tip blooms.
        loaded = Protein.from_file(DATA / "2k39.cif", chains="A")
        core = loaded.select(residues=(1, 70), atoms="CA")
        ubq = Protein.from_trajectory(loaded.trajectory.aligned(indices=core.atom_indices))
        ubq = palette(ubq.surface(resolution=0.6).cartoon().center())
        self.camera.frame(ubq, margin=1.05, aspect=aspect)
        self.camera.theta, self.camera.phi = 0.5, 0.2
        title, detail = caption("Studio rendering", "Glossy light · ambient occlusion · bloom")
        self.play(Write(title, stroke_width=1.6), FadeIn(detail), run_time=1.2)
        self.play(
            Thread(
                ubq,
                self.camera,
                mode="wire",
                radius=0.32,
                head=2.2,
                glow=22.0,
                glow_color=MINT,
                glow_brightness=1.6,
                seed=2,
            ),
            self.camera.animate.orbit(0.35, 0.02),
            run_time=4.5,
        )
        self.play(self.camera.animate.orbit(0.8, 0.05), run_time=4)
        self.play(Unwrite(title), FadeOut(detail), run_time=0.8)

        # 2. A GPU surface that follows the NMR models, every frame.
        title, detail = caption("Live surfaces", "Computed on the GPU each frame · 116 NMR models")
        self.play(Representation(ubq, "surface"), Write(title), FadeIn(detail), run_time=1.6)
        self.play(
            PlayTrajectory(ubq, start=0, end=30, state_easing=smooth),
            self.camera.animate.orbit(0.9),
            run_time=7,
        )
        self.play(Representation(ubq, "cartoon"), Unwrite(title), FadeOut(detail), run_time=1.4)
        self.play(FadeOut(ubq), run_time=0.8)

        # 3. GroEL–GroES: 21 chains, a deforming surface, and a cutaway into the ring.
        groel = upright(palette(Protein.from_file(DATA / "1aon.cif").surface(resolution=1.0).cartoon()))
        ligand = groel.select(chain="A", resname=["ADP", "MG"])
        site = groel.select(within=4.0, of=ligand) | ligand
        title, detail = caption("59,000 atoms", "GroEL–GroES · GPU surface while the complex breathes")
        self.play(
            FadeIn(groel),
            Focus(self.camera, groel, margin=0.95, aspect=aspect),
            Write(title),
            FadeIn(detail),
            run_time=1.8,
        )
        self.play(Representation(groel, "surface"), self.camera.animate.orbit(0.5), run_time=2.2)
        rest = groel.positions.copy()
        center = rest.mean(0)
        self.play(
            Deform(groel, lambda xyz: center + (xyz - center) * np.array([1.06, 0.96, 1.06])),
            self.camera.animate.orbit(0.5),
            run_time=2.5,
        )
        self.play(Deform(groel, lambda xyz: rest), self.camera.animate.orbit(0.4), run_time=2.5)
        self.play(Unwrite(title), FadeOut(detail), run_time=0.6)
        title, detail = caption("Cutaways", "The surface is carved on the GPU, down to the ADP pocket")
        self.play(
            Focus(self.camera, site, margin=1.5, aspect=aspect), Write(title), FadeIn(detail), run_time=2
        )
        label = ligand.callout(
            "ADP · Mg²⁺", subtitle="Subunit A", position=(0.72, 0.28), font_size=38, color=CORAL
        )
        self.play(Reveal(self.camera, site, window=1.3, shape="tunnel"), run_time=2)
        self.play(Write(label), self.camera.animate.orbit(0.2, 0.08).depth_cue(0.7), run_time=2)
        self.play(self.camera.animate.orbit(0.25, 0.05), run_time=2.5)
        self.play(Conceal(self.camera), Unwrite(label), Unwrite(title), FadeOut(detail), run_time=1.5)
        self.wait(0.3)
