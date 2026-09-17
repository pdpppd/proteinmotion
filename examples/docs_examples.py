"""Small scenes rendered beside the documentation snippets.

Render one: proteinmotion render examples/docs_examples.py ColorChange --fps 60 -o color.mp4
Render all and refresh the docs: python scripts/render_docs_examples.py
Inputs: deposited 1UBQ and 2K39 coordinates in examples/data.
"""

from pathlib import Path

import numpy as np

from proteinmotion import (
    Colorize,
    Deform,
    Distance,
    Electrostatics,
    FadeIn,
    FadeOut,
    HydrogenBonds,
    Morph,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Region,
    Representation,
    Rotate,
    SetOpacity,
    Text,
    Unwrite,
    Write,
    smooth,
)

DATA = Path(__file__).parent / "data"


def ubiquitin():
    return Protein.from_file(DATA / "1ubq.cif", chains="A").cartoon().center()


def frame(scene, p, target=None, margin=0.85):
    """Set up the model and camera before the first animation."""
    scene.add(p)
    scene.camera.frame(target or p, margin=margin, aspect=scene.width / scene.height)
    scene.camera.theta, scene.camera.phi = 0.65, 0.18
    scene.camera.depth_cue = 0.3


def helix_view(scene):
    p = ubiquitin().ball_and_stick(atom_scale=0.28, bond_radius=0.10)
    helix = p.select(residues=(23, 34))
    rest = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), helix.atom_indices))
    rest.set_opacity(0.05)
    frame(scene, p, helix, margin=1.1)
    return p


class Motion(ProteinScene):
    def construct(self):
        # docs:start motion
        p = ubiquitin()
        frame(self, p)
        self.play(FadeIn(p), run_time=1)
        self.play(Rotate(p, np.pi), run_time=3)
        self.play(Representation(p, "ribbon"), run_time=1)
        self.play(
            p.animate.shift((5, 0, 0)),
            self.camera.animate.orbit(theta=0.6),
            run_time=2,
        )
        self.wait(0.5)
        # docs:end motion


class Representations(ProteinScene):
    def construct(self):
        # docs:start representations
        p = ubiquitin().surface(resolution=0.7).cartoon()
        frame(self, p)
        self.wait(1)
        for name in ("ribbon", "ball_and_stick", "surface"):
            self.play(Representation(p, name), run_time=1.2)
            self.play(Rotate(p, 0.45), run_time=1.5)
        # docs:end representations


class RegionFocus(ProteinScene):
    def construct(self):
        # docs:start region-focus
        p = ubiquitin()
        frame(self, p)
        helix = p.select(chain="A", residues=(23, 34))
        marker = helix.highlight(
            style="box",
            color="#f2ba67",
            padding=1.5,
        )
        self.play(FadeIn(marker), run_time=0.6)
        self.focus(helix, margin=1.6, run_time=1.5)
        self.play(self.camera.animate.orbit(0.4), run_time=2)
        self.play(FadeOut(marker), run_time=0.6)
        self.focus(p, run_time=1.5)
        # docs:end region-focus


class HighlightStyles(ProteinScene):
    def construct(self):
        # docs:start highlights
        p = ubiquitin()
        frame(self, p)
        helix = p.select(residues=(23, 34))
        for style in ("sphere", "box", "atoms"):
            marker = helix.highlight(
                style=style,
                color="#f2ba67",
                padding=1.0,
                opacity=0.3,
            )
            self.play(FadeIn(marker), run_time=0.6)
            self.wait(1.2)
            self.play(FadeOut(marker), run_time=0.6)
        # docs:end highlights


class Writing(ProteinScene):
    def construct(self):
        # docs:start writing
        title = Text(
            "Protein motion",
            font_size=130,
            font="semibold",
            position=(0.1, 0.38),
        )
        self.play(
            Write(title, lag_ratio=0.12, stroke_width=1.8),
            run_time=2.5,
        )
        self.wait(1)
        self.play(Unwrite(title), run_time=1.5)
        # docs:end writing


class TextPlacement(ProteinScene):
    def construct(self):
        # docs:start text-placement
        title = Text(
            "α helix / β sheet",
            font_size=80,
            color="#50e0d0",
            position=(0.06, 0.08),
        )
        footer = Text("PDB 1UBQ", font_size=40).to_corner("DR")
        self.play(Write(title), Write(footer), run_time=2)
        self.play(title.animate.move_to((0.10, 0.35)), run_time=1)
        self.play(title.animate.set_opacity(0.4), run_time=0.5)
        self.wait(1)
        # docs:end text-placement


class Callout(ProteinScene):
    def construct(self):
        # docs:start callout
        p = ubiquitin()
        frame(self, p, margin=1.05)
        helix = p.select(residues=(23, 34), atoms="CA")
        note = helix.callout(
            "α helix",
            subtitle="Residues 23–34",
            position=(0.07, 0.38),
            font_size=48,
            color="#f2ba67",
            tip="arrow",
        )
        box = helix.highlight(style="box", color="#f2ba67")
        self.play(Write(note), FadeIn(box), run_time=2)
        self.play(self.camera.animate.orbit(0.5), run_time=3)
        self.wait(1)
        # docs:end callout


class ResidueLabels(ProteinScene):
    def construct(self):
        # docs:start residue-labels
        p = ubiquitin()
        frame(self, p, margin=1.05)
        labels = p.label_residues(
            chain="A",
            residues=[8, 44, 70],
            font_size=40,
            color="#f2ba67",
            offsets={8: (-270, -90), 44: (440, -100), 70: (380, 150)},
        )
        self.play(Write(labels, lag_ratio=0.08), run_time=2)
        self.play(self.camera.animate.orbit(0.35), run_time=2)
        self.wait(1)
        # docs:end residue-labels


class ResidueColors(ProteinScene):
    def construct(self):
        # docs:start residue-colors
        p = ubiquitin()
        helix = p.select(residues=(23, 34))
        strand = p.select(residues=(2, 7))
        helix.set_color("#50e0d0")
        strand.set_color("#f2ba67")
        p.select(residues=(71, 76)).set_opacity(0.2)
        frame(self, p)
        self.play(Rotate(p, 0.7), run_time=3)
        # docs:end residue-colors


class ColorChange(ProteinScene):
    def construct(self):
        # docs:start color-change
        p = ubiquitin()
        frame(self, p)
        helix = p.select(residues=(23, 34))
        strand = p.select(residues=(2, 7))
        self.play(
            Colorize(helix, "#50e0d0", residue_delay=0.09),
            Colorize(strand, "#f2ba67", residue_delay=0.15),
            run_time=2.5,
        )
        self.wait(1)
        self.play(Colorize(helix, None), run_time=1.5)
        self.wait(0.5)
        # docs:end color-change


class Opacity(ProteinScene):
    def construct(self):
        # docs:start opacity
        p = ubiquitin().ball_and_stick()
        helix = p.select(residues=(23, 34))
        helix.set_color("#50e0d0")
        frame(self, p)
        self.wait(0.5)
        self.play(
            SetOpacity(helix, 0.1, residue_delay=0.06),
            run_time=2,
        )
        self.wait(0.7)
        self.play(SetOpacity(helix, 1), run_time=1.5)
        self.wait(0.5)
        # docs:end opacity


class Surface(ProteinScene):
    def construct(self):
        # docs:start surface
        p = ubiquitin().surface(
            kind="ses",
            probe_radius=1.4,
            resolution=0.7,
            color="secondary",
            update="rebuild",
            max_voxels=8_000_000,
        )
        frame(self, p)
        self.play(Rotate(p, 0.8), run_time=3)
        # docs:end surface


class DistanceModes(ProteinScene):
    def construct(self):
        # docs:start distances
        p = helix_view(self)
        a = p.select(residues=23, atoms="CA")
        b = p.select(residues=34, atoms="CA")
        for mode in ("3d", "2d"):
            ruler = Distance(
                a,
                b,
                mode=mode,
                font_size=42,
                color="#50e0d0",
                prefix="Cα · ",
            )
            title = Text(f"{mode.upper()} line", font_size=48)
            self.play(Write(ruler), Write(title), run_time=1.5)
            self.play(self.camera.animate.orbit(0.35), run_time=2)
            self.play(FadeOut(ruler), FadeOut(title), run_time=0.5)
        # docs:end distances


class HydrogenBondLines(ProteinScene):
    def construct(self):
        # docs:start hydrogen-bonds
        p = helix_view(self)
        hb = HydrogenBonds(
            p,
            donors=p.select(residues=(23, 34), atoms="N"),
            max_distance=3.5,
            min_angle=150,
            hydrogens="backbone",
        )
        lines = hb.highlight(
            mode="3d",
            color="#f2ba67",
            radius=0.07,
            dash_count=5,
            show_distances=False,
        )
        self.play(Write(lines), run_time=2)
        self.play(self.camera.animate.orbit(0.6), run_time=3)
        self.wait(1)
        # docs:end hydrogen-bonds


class ChargeContacts(ProteinScene):
    def construct(self):
        # docs:start charge-contacts
        p = ubiquitin().ball_and_stick()
        field = Electrostatics(
            p,
            charges="formal",
            dielectric=80,
            screening_length=8,
            cutoff=6,
            min_energy=0.08,
        )
        pair = field.pairs[0]
        endpoints = Region(p, [pair.a, pair.b])
        residues = [p.topology.atoms[i].resid for i in (pair.a, pair.b)]
        selected = p.select(residues=residues)
        other = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), selected.atom_indices))
        other.set_opacity(0.06)
        frame(self, p, endpoints, margin=4.5)
        contacts = field.highlight(
            mode="2d",
            max_pairs=1,
            show_distances=True,
            font_size=42,
            style="solid",
            line_width=2,
        )
        self.play(Write(contacts), run_time=1.5)
        self.play(self.camera.animate.orbit(0.5), run_time=3)
        self.wait(1)
        # docs:end charge-contacts


class Deformation(ProteinScene):
    def construct(self):
        # docs:start deformation
        p = ubiquitin()
        frame(self, p, margin=1.05)
        rest = p.positions.copy()

        def stretch(xyz):
            center = xyz.mean(axis=0)
            return center + (xyz - center) * [1.4, 1, 1]

        self.play(Deform(p, stretch), run_time=2)
        self.wait(0.5)
        self.play(Morph(p, rest, align=False), run_time=2)
        self.wait(0.5)
        # docs:end deformation


class NMRStates(ProteinScene):
    def construct(self):
        # docs:start nmr-states
        loaded = Protein.from_file(DATA / "2k39.cif", chains="A")
        core = loaded.select(residues=(1, 70), atoms="CA")
        aligned = loaded.trajectory.aligned(indices=core.atom_indices)
        p = Protein.from_trajectory(aligned).cartoon().center()
        frame(self, p)
        self.play(
            PlayTrajectory(p, start=0, end=2, state_easing=smooth),
            run_time=6,
        )
        self.wait(0.5)
        # docs:end nmr-states


# ID, scene, title, caption, poster time. Snippet markers above are also the docs IDs.
EXAMPLES = [
    ("motion", Motion, "Rotation and movement", "Ubiquitin, PDB 1UBQ.", 3),
    (
        "representations",
        Representations,
        "Representation changes",
        "Cartoon, ribbon, ball-and-stick, and SES surface.",
        8.3,
    ),
    ("region-focus", RegionFocus, "Camera focus", "Ubiquitin residues 23–34.", 3),
    (
        "highlights",
        HighlightStyles,
        "3D highlights",
        "Sphere, box, and atom highlights on the same helix.",
        3.7,
    ),
    ("writing", Writing, "Write and Unwrite", "Letter outlines are drawn, filled, and erased.", 3),
    (
        "text-placement",
        TextPlacement,
        "Text position and opacity",
        "Positions use fractions of the image size.",
        2.8,
    ),
    ("callout", Callout, "Region callout", "A line follows the selected helix as the camera moves.", 3),
    ("residue-labels", ResidueLabels, "Amino acid labels", "Ubiquitin residues 8, 44, and 70.", 3),
    (
        "residue-colors",
        ResidueColors,
        "Residue colors",
        "Helix in cyan, strand in gold, and a transparent tail.",
        1.5,
    ),
    ("color-change", ColorChange, "Color animation", "Residues change color in sequence from N to C.", 3),
    ("opacity", Opacity, "Residue transparency", "The selected helix fades and returns.", 2.8),
    ("surface", Surface, "Molecular surface", "Approximate SES with a 1.4 Å probe and 0.7 Å grid.", 1.5),
    (
        "distances",
        DistanceModes,
        "3D and 2D distance lines",
        "Distance between Cα23 and Cα34 in PDB 1UBQ.",
        5.5,
    ),
    (
        "hydrogen-bonds",
        HydrogenBondLines,
        "Hydrogen bonds",
        "Geometry-based N···O contacts with inferred backbone H.",
        3,
    ),
    (
        "charge-contacts",
        ChargeContacts,
        "Electrostatic contact",
        "Screened Coulomb estimate using example formal charges.",
        3,
    ),
    (
        "deformation",
        Deformation,
        "Deformation and morph",
        "A synthetic stretch followed by a return to the original coordinates.",
        2,
    ),
    (
        "nmr-states",
        NMRStates,
        "NMR state playback",
        "PDB 2K39 models 1–3. Interpolation shows structural variation.",
        3,
    ),
]
