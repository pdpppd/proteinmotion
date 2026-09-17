"""One film covering ProteinMotion's visual tools, rendered entirely by the package.

    proteinmotion render examples/feature_showcase.py FeatureShowcase --fps 60 -o showcase.mp4

The XTC fixture contains deposited 2K39 NMR conformers, not an MD simulation.
The imported charge array contains illustrative formal charges, not a force field.
Regenerate both fixtures with prepare_inputs(); this needs proteinmotion[md].
"""

from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np

from proteinmotion import (
    BackboneMorph,
    Colorize,
    ContactMatch,
    Deform,
    Distance,
    Electrostatics,
    FadeIn,
    FadeOut,
    Focus,
    Morph,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Region,
    Representation,
    Rotate,
    SetOpacity,
    Text,
    Trajectory,
    Unwrite,
    Write,
    smooth,
)

try:  # Both normal imports and the proteinmotion scene loader are supported.
    from .alpha_helix_hbonds import analyze_helix, ideal_helix
except ImportError:
    from alpha_helix_hbonds import analyze_helix, ideal_helix

DATA = Path(__file__).parent / "data"
TEAL, GOLD, BLUE, MUTED = "#50e0d0", "#f2ba67", "#638cff", "#9eafc4"


@lru_cache(maxsize=1)
def ensemble():
    loaded = Protein.from_file(DATA / "2k39.cif", chains="A")
    core = loaded.select(residues=(1, 70), atoms="CA")
    return loaded.trajectory.aligned(indices=core.atom_indices)


def ubiquitin():
    return Protein.from_trajectory(ensemble()).cartoon().center()


def complement(region):
    p = region.protein
    return Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), region.atom_indices))


def prepare_inputs():
    """Repackage known coordinates as XTC; do not invent or simulate dynamics."""
    import MDAnalysis as mda

    p = ubiquitin()
    topology = DATA / "showcase-nmr.pdb"
    with topology.open("w") as out:
        out.write("REMARK 2K39 NMR conformers repackaged for trajectory I/O, not MD\n")
        for i, (atom, xyz) in enumerate(zip(p.topology.atoms, p.positions), 1):
            x, y, z = xyz
            out.write(
                f"ATOM  {i:5d} {atom.name:>4s} {atom.resname:3s} {atom.chain:1s}{atom.resid:4d}"
                f"{atom.icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2s}\n"
            )
        out.write("END\n")
    u = mda.Universe(str(topology))
    with mda.Writer(str(DATA / "showcase-nmr.xtc"), n_atoms=len(p.topology.atoms)) as writer:
        for i in range(len(p.trajectory)):
            u.atoms.positions = p.trajectory.frame(i)
            writer.write(u.atoms)
    # Export and import the array to demonstrate the user-charge workflow honestly.
    np.save(DATA / "showcase-formal-charges.npy", Electrostatics(p, charges="formal").charges)


class Chapter(ProteinScene):
    number = ""
    name = ""

    def stage(self, *proteins, margin=1.20, theta=0.6, phi=0.15):
        self.add(*proteins)
        self.camera.frame(*proteins, margin=margin, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi, self.camera.depth_cue = theta, phi, 0.3

    def heading(self, title, detail):
        kicker = Text(
            f"{self.number} / {self.name.upper()}", font_size=24, color=TEAL, position=(0.06, 0.055)
        )
        title = Text(title, font_size=68, font="semibold", position=(0.06, 0.102))
        footer = Text(detail, font_size=24, color=MUTED, position=(0.06, 0.915))
        self.play(FadeIn(kicker), Write(title, lag_ratio=0.015), FadeIn(footer), run_time=0.7)

    def badge(self, text, color=TEAL):
        return Text(text, font_size=38, color=color, position=(0.94, 0.825), align="right")


class Representations(Chapter):
    number, name = "01", "Representations"

    def construct(self):
        p = ubiquitin().surface(kind="ses", resolution=0.5).cartoon()
        self.stage(p, theta=0.4)
        self.heading("One structure. Every view.", "Cartoon · ribbon · ball and stick · molecular surface")
        label = self.badge("Cartoon")
        self.play(Write(label), Rotate(p, 0.45), run_time=1.3)
        for rep, name, color in [
            ("ribbon", "Ribbon", BLUE),
            ("ball_and_stick", "Ball and stick", GOLD),
            ("surface", "Solvent-excluded surface", TEAL),
        ]:
            replacement = self.badge(name, color)
            self.play(FadeOut(label), Representation(p, rep), Write(replacement), run_time=0.8)
            self.play(Rotate(p, 0.55), run_time=1.4)
            label = replacement
        self.wait(0.4)


class ResidueStyle(Chapter):
    number, name = "02", "Color + transparency"

    def construct(self):
        p = ubiquitin().surface(kind="ses", resolution=0.5).cartoon(color="#8198ae")
        helix, sheet = p.select(residues=(23, 34)), p.select(residues=(2, 7))
        rest = complement(helix | sheet)
        self.stage(p)
        self.heading("Make a region stand out.", "Animated color · residue staggering · smooth transparency")
        self.play(
            Colorize(helix, TEAL, residue_delay=0.07), Colorize(sheet, GOLD, residue_delay=0.12), run_time=1.5
        )
        self.play(SetOpacity(rest, 0.08, residue_delay=0.01), self.camera.animate.orbit(0.25), run_time=1.2)
        self.play(Representation(p, "ball_and_stick"), run_time=0.8)
        self.play(SetOpacity(rest, 1, residue_delay=0.008), run_time=0.9)
        self.play(Representation(p, "surface"), run_time=0.8)
        self.play(PlayTrajectory(p, start=0, end=3, state_easing=smooth), Rotate(p, 0.4), run_time=1.8)
        self.play(SetOpacity(rest, 0.12), run_time=0.7)


class FocusLabels(Chapter):
    number, name = "03", "Focus + annotation"

    def construct(self):
        p = ubiquitin().cartoon(color="#8198ae")
        helix = p.select(residues=(23, 34))
        helix.set_color(TEAL)
        rest = complement(helix)
        self.stage(p, theta=0.9)
        self.heading("Guide the eye.", "Camera focus · 3D highlights · vector writing · amino-acid labels")
        sphere = helix.highlight(style="sphere", color=TEAL, opacity=0.10, padding=0.8)
        box = helix.highlight(style="box", color=TEAL, opacity=0.6, padding=1.3, line_width=0.055)
        note = helix.callout(
            "α helix", subtitle="Residues 23–34", font_size=42, color=TEAL, position=(0.07, 0.42)
        )
        self.play(FadeIn(sphere), FadeIn(box), Write(note), run_time=1.0)
        self.play(
            Focus(self.camera, helix, margin=2.15, aspect=self.width / self.height),
            SetOpacity(rest, 0.06),
            run_time=1.3,
        )
        labels = p.label_residues(
            residues=[24, 32], font_size=34, color=GOLD, offsets={24: (290, -60), 32: (310, 65)}
        )
        halos = p.select(residues=[24, 32], atoms="CA").highlight(
            style="atoms", color=GOLD, opacity=0.4, padding=0.4
        )
        self.play(
            FadeOut(box),
            FadeOut(sphere),
            Representation(p, "ball_and_stick"),
            Write(labels),
            FadeIn(halos),
            run_time=1.2,
        )
        self.play(self.camera.animate.orbit(0.25), run_time=1.6)
        self.play(Unwrite(labels), Unwrite(note), FadeOut(halos), run_time=0.7)


class EnsemblePlayback(Chapter):
    number, name = "04", "Ensembles + MD readers"

    def construct(self):
        left = ubiquitin().shift((-21, 0, 0))
        trajectory = Trajectory.from_mdanalysis(DATA / "showcase-nmr.pdb", DATA / "showcase-nmr.xtc")
        right = (
            Protein.from_trajectory(trajectory)
            .ball_and_stick(atom_scale=0.24, bond_radius=0.10)
            .center()
            .shift((21, 0, 0))
        )
        self.stage(left, right, margin=1.02, theta=0, phi=0)
        self.heading(
            "Let structures move.",
            "2K39 states 1–25 · same NMR coordinates in XTC · interpolation is illustrative",
        )
        self.add(
            Text("NMR ensemble", font_size=37, color=TEAL, position=(0.29, 0.26), align="center"),
            Text("Lazy XTC playback", font_size=37, color=GOLD, position=(0.71, 0.26), align="center"),
        )
        self.play(
            PlayTrajectory(left, start=0, end=24, state_easing=smooth),
            PlayTrajectory(right, start=0, end=24, state_easing=smooth),
            Rotate(left, 0.8),
            Rotate(right, 0.8),
            run_time=6.3,
        )


class Deformation(Chapter):
    number, name = "05", "Deformation + easing"

    def construct(self):
        p = ubiquitin().cartoon(color=TEAL)
        self.stage(p, margin=1.6, theta=0.5)
        self.heading(
            "Deform. Ease. Return.",
            "Procedural coordinate deformation · an illustration, not a dynamics simulation",
        )
        original = p.positions.copy()

        def bend(xyz):
            center = xyz.mean(0)
            d = xyz - center
            d[:, 0] = d[:, 0] * 1.45 + 2.8 * np.sin(d[:, 1] * 0.15)
            d[:, 2] += 2.0 * np.sin(d[:, 0] * 0.12)
            return d + center

        self.play(Deform(p, bend), Rotate(p, 0.35), run_time=2.2)
        self.play(Morph(p, original, align=False), Rotate(p, 0.35), run_time=2.1)


class ContactMorph(Chapter):
    number, name = "06", "Contact-guided morphs"

    def construct(self):
        source = (
            Protein.from_file(DATA / "1cll.cif", chains="A")
            .ball_and_stick(atom_scale=0.27, bond_radius=0.10)
            .center()
        )
        target = (
            Protein.from_file(DATA / "1ncx.cif", chains="A")
            .ball_and_stick(atom_scale=0.27, bond_radius=0.10)
            .center()
        )
        match = ContactMatch.load(DATA / "calmodulin-troponin-match.json")
        source.set_color("#8299ad")
        target.set_color("#8299ad")
        source.select(residues=[source.topology.residues[i].resid for i in match.source_indices]).set_color(
            TEAL
        )
        target.select(residues=[target.topology.residues[i].resid for i in match.target_indices]).set_color(
            GOLD
        )
        self.stage(source, margin=1.5, theta=0.35)
        self.heading(
            "Different proteins. Shared geometry.",
            "114 paired Cα · 25 ms N → C delay · unmatched residues fade out / in",
        )
        self.add(
            Text("Calmodulin → troponin C", font_size=34, color=GOLD, position=(0.94, 0.83), align="right")
        )
        self.play(BackboneMorph(source, target, match=match, residue_delay=0.025), run_time=6.0)
        self.play(Rotate(target, 0.45), run_time=1.3)


class HelixBonds(Chapter):
    number, name = "07", "Hydrogen bonds"

    def construct(self):
        p = ideal_helix().ball_and_stick(atom_scale=0.25, bond_radius=0.09)
        hb = analyze_helix(p)
        self.stage(p, margin=1.40, theta=0.7, phi=0.08)
        self.heading(
            "Reveal the connections.",
            "Ideal α helix · explicit H · geometric detection recovers 12 of 12 i → i+4 pairs",
        )
        network = hb.highlight(
            mode="3d",
            endpoints="hydrogen_acceptor",
            show_distances=False,
            color=GOLD,
            radius=0.055,
            dash_count=5,
        )
        self.play(Write(network), self.camera.animate.orbit(0.35), run_time=1.5)
        record = next(r for r in hb.pairs if p.topology.atoms[r.b].resid == 5)
        turn = p.select(residues=(5, 9))
        h, o = Region(p, [record.hydrogen]), Region(p, [record.b])
        bond = Distance(h, o, mode="3d", show_distance=False, color=GOLD, radius=0.07, dash_count=5)
        self.play(
            FadeOut(network),
            SetOpacity(complement(turn), 0.06),
            Focus(self.camera, turn, margin=2.0, aspect=self.width / self.height),
            run_time=1.3,
        )
        note = h.callout(
            f"H···O   {bond.distance:.2f} Å",
            subtitle=f"N–H···O   {record.angle:.1f}°",
            color=GOLD,
            font_size=40,
            position=(0.07, 0.43),
        )
        self.play(Write(bond), Write(note), run_time=0.9)
        self.play(self.camera.animate.orbit(0.25), run_time=1.9)


class Rulers(Chapter):
    number, name = "08", "Distance labels"

    def construct(self):
        p = ubiquitin().ball_and_stick(atom_scale=0.25, bond_radius=0.10)
        helix = p.select(residues=(23, 34))
        complement(helix).set_opacity(0.05)
        a, b = p.select(residues=23, atoms="CA"), p.select(residues=34, atoms="CA")
        a.set_color(TEAL)
        b.set_color(TEAL)
        self.stage(p, margin=1.4, theta=0.8)
        self.camera.frame(helix, margin=1.85, aspect=self.width / self.height)
        self.heading(
            "Measure in three dimensions.",
            "Live distances between residues or atoms · labels centered on the line",
        )
        solid = Distance(
            a, b, mode="3d", color=TEAL, font_size=38, radius=0.085, style="dashed", dash_count=13
        )
        overlay = Distance(
            a, b, mode="2d", color=GOLD, font_size=38, line_width=2.5, style="dashed", dash_count=13
        )
        label = self.badge("3D · depth tested")
        self.play(Write(solid), Write(label), self.camera.animate.orbit(0.25), run_time=1.3)
        next_label = self.badge("2D · overlay", GOLD)
        self.play(FadeOut(solid), FadeOut(label), Write(overlay), Write(next_label), run_time=0.8)
        self.play(
            PlayTrajectory(p, start=0, end=6, state_easing=smooth),
            self.camera.animate.orbit(0.4),
            run_time=2.2,
        )


class ChargeContacts(Chapter):
    number, name = "09", "Electrostatics"

    def construct(self):
        p = ubiquitin().ball_and_stick(atom_scale=0.26, bond_radius=0.09)
        field = Electrostatics(
            p,
            charges=np.load(DATA / "showcase-formal-charges.npy"),
            dielectric=80,
            screening_length=8,
            cutoff=6,
            min_energy=0.08,
        )
        pair = next(pair for pair in field.pairs if pair.energy < 0)
        numbers = [p.topology.atoms[i].resid for i in (pair.a, pair.b)]
        charged = p.select(residues=numbers)
        complement(charged).set_opacity(0.035)
        p.select(residues=numbers[0]).set_color(BLUE)
        p.select(residues=numbers[1]).set_color(GOLD)
        self.stage(p, theta=0.65)
        self.camera.frame(Region(p, [pair.a, pair.b]), margin=5, aspect=self.width / self.height)
        self.heading(
            "See charge interactions.",
            "Imported illustrative formal charges · screened Coulomb · εr = 80 · screening length 8 Å",
        )
        contacts = field.highlight(
            mode="2d",
            region=charged,
            max_pairs=2,
            show_distances=False,
            style="dashed",
            line_width=2.4,
            follow_opacity=False,
        )
        note = Region(p, [pair.a, pair.b]).callout(
            "Attractive contact",
            subtitle=f"{pair.energy:.2f} kcal/mol",
            font_size=40,
            color=BLUE,
            position=(0.065, 0.43),
        )
        self.play(Write(contacts), Write(note), run_time=1.1)
        self.play(self.camera.animate.orbit(0.4), run_time=3.2)


class LargeComplex(Chapter):
    number, name = "10", "Scale + native rendering"

    def construct(self):
        p = Protein.from_file(DATA / "1aon.cif").cartoon(color="chain").center()
        ca = p.positions[[r.ca for r in p.topology.residues if r.ca >= 0]]
        _, _, axes = np.linalg.svd(ca - ca.mean(0), full_matrices=False)
        p.orientation = np.stack((axes[1], axes[0], axes[2]))
        if np.linalg.det(p.orientation) < 0:
            p.orientation[2] *= -1
        p.center()
        self.stage(p, margin=1.35, theta=0.3, phi=0.2)
        self.heading(
            "From one residue to 58,870 atoms.",
            "GroEL/GroES · PDB 1AON · 21 chains · free rotation + depth cueing",
        )
        self.play(Rotate(p, 0.7), run_time=1.4)
        self.play(Representation(p, "ball_and_stick"), run_time=0.8)
        self.play(Rotate(p, 0.45), run_time=1.3)
        self.play(Representation(p, "cartoon"), run_time=0.7)
        brand = Text(
            "ProteinMotion", font_size=62, font="semibold", color=TEAL, position=(0.94, 0.82), align="right"
        )
        self.play(Write(brand), Rotate(p, 0.2), run_time=0.65)
        self.wait(0.45)


class FeatureShowcase(ProteinScene):
    """One seekable movie, with clean editorial cuts between ten native scenes."""

    chapter_types = (
        Representations,
        ResidueStyle,
        FocusLabels,
        EnsemblePlayback,
        Deformation,
        ContactMorph,
        HelixBonds,
        Rulers,
        ChargeContacts,
        LargeComplex,
    )

    def construct(self):
        self.chapters, self.starts = [], []
        for chapter_type in self.chapter_types:
            self.starts.append(self.duration)
            chapter = chapter_type(
                width=self.width, height=self.height, fps=self.fps, background="#091321", msaa=self.msaa
            ).build()
            self.chapters.append(chapter)
            self.duration += chapter.duration

    def seek(self, time):
        self.build()
        if not np.isfinite(time):
            raise ValueError("Time must be finite")
        time = float(np.clip(time, 0, self.duration))
        index = min(bisect_right(self.starts, time) - 1, len(self.chapters) - 1)
        chapter = self.chapters[index]
        visible = chapter.seek(time - self.starts[index])
        self.camera, self.background = chapter.camera, chapter.background
        return visible

    def chapter_manifest(self):
        self.build()
        return [
            dict(number=c.number, title=c.name, start=round(start, 3), duration=round(c.duration, 3))
            for start, c in zip(self.starts, self.chapters)
        ]


if __name__ == "__main__":
    FeatureShowcase(width=1920, height=1080, fps=60).render("proteinmotion-showcase.mp4")
