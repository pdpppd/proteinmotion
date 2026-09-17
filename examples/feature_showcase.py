"""A continuous calmodulin → troponin C feature tour, with no scene cuts.

    proteinmotion render examples/feature_showcase.py FeatureShowcase --fps 60 -o showcase.mp4

The XTC fixture maps deposited 1CFC NMR conformers onto 1CLL atom identities.
It is not an MD simulation. Interpolation, deformation and the protein-to-protein
morph are illustrative. Regenerate fixtures with prepare_inputs() (requires [md]).
"""

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
    HydrogenBonds,
    Morph,
    PlayTrajectory,
    Protein,
    ProteinScene,
    Region,
    Representation,
    SetOpacity,
    Text,
    Trajectory,
    Unwrite,
    Write,
    smooth,
)
from proteinmotion.math3d import align_coordinates

DATA = Path(__file__).parent / "data"
TEAL, GOLD, BLUE, MUTED = "#50e0d0", "#f2ba67", "#638cff", "#9eafc4"


def complement(region):
    p = region.protein
    return Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), region.atom_indices))


def mapped_ensemble():
    """Keep all 1CLL protein atoms; align the 25 1CFC models on common Cαs.

    Six nonprotein atoms absent from 1CFC retain their reference coordinates.
    The scene hides them throughout NMR playback. No missing protein atoms are
    fabricated: every one of the 1,130 displayed heavy atoms has an exact match.
    """
    p = Protein.from_file(DATA / "1cll.cif", chains="A")
    nmr = Protein.from_file(DATA / "1cfc.cif", chains="A")
    lookup = {key: i for i, key in enumerate(nmr.topology.keys)}
    common = np.array([i for i, key in enumerate(p.topology.keys) if key in lookup])
    mapping = np.array([lookup[p.topology.keys[i]] for i in common])
    ca = [j for j, i in enumerate(common) if p.topology.atoms[i].name == "CA"]
    protein_atoms = [
        i for i, a in enumerate(p.topology.atoms) if p.topology.residues[a.residue_index].ca >= 0
    ]
    if set(common) != set(protein_atoms):
        raise ValueError("NMR fixture must map every protein heavy atom exactly")
    frames = []
    for state in range(len(nmr.trajectory)):
        xyz = p.positions.copy()
        xyz[common] = align_coordinates(nmr.trajectory.frame(state)[mapping], xyz[common], ca)
        frames.append(xyz)
    return p, Trajectory(frames, topology=p.topology)


def prepare_inputs():
    """Write mapped experimental conformers as XTC, plus illustrative charges."""
    import MDAnalysis as mda

    p, trajectory = mapped_ensemble()
    topology = DATA / "showcase-calmodulin.pdb"
    with topology.open("w") as out:
        out.write("REMARK 1CFC mapped to 1CLL atom order; NMR conformers, not MD\n")
        for i, (atom, xyz) in enumerate(zip(p.topology.atoms, p.positions), 1):
            x, y, z = xyz
            record = "ATOM  " if p.topology.residues[atom.residue_index].ca >= 0 else "HETATM"
            out.write(
                f"{record}{i:5d} {atom.name:>4s} {atom.resname:3s} {atom.chain:1s}{atom.resid:4d}"
                f"{atom.icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2s}\n"
            )
        out.write("END\n")
    u = mda.Universe(str(topology))
    with mda.Writer(str(DATA / "showcase-calmodulin.xtc"), n_atoms=len(p.topology.atoms)) as writer:
        for i in range(len(trajectory)):
            u.atoms.positions = trajectory.frame(i)
            writer.write(u.atoms)
    troponin = Protein.from_file(DATA / "1ncx.cif", chains="A")
    np.save(DATA / "showcase-troponin-charges.npy", Electrostatics(troponin, charges="formal").charges)


class FeatureShowcase(ProteinScene):
    """One camera, two protein objects, and an ordinary seekable animation timeline.

    Chapter markers only change captions. They never replace scene objects,
    reset the camera, or override seek(). The target of BackboneMorph is the
    very same troponin object used for every subsequent demonstration.
    """

    def heading(self, name, title, detail, *, intro=()):
        self._chapters.append(dict(number=f"{len(self._chapters) + 1:02d}", title=name, start=self.duration))
        if self._heading:
            self.play(*(FadeOut(t) for t in self._heading), self.camera.animate.orbit(0.012), run_time=0.4)
        self._heading = [
            Text(
                f"{len(self._chapters):02d} / {name.upper()}",
                font_size=24,
                color=TEAL,
                position=(0.06, 0.055),
            ),
            Text(title, font_size=62, font="semibold", position=(0.06, 0.103)),
            Text(detail, font_size=23, color=MUTED, position=(0.06, 0.925)),
        ]
        self.play(
            *(FadeIn(obj) for obj in intro),
            FadeIn(self._heading[0]),
            Write(self._heading[1], lag_ratio=0.015),
            FadeIn(self._heading[2]),
            self.camera.animate.orbit(0.025),
            run_time=1.0,
        )

    def badge(self, text, color=TEAL):
        return Text(text, font_size=34, color=color, position=(0.94, 0.83), align="right")

    def fit(self, region, margin):
        return Focus(self.camera, region, margin=margin, aspect=self.width / self.height)

    def construct(self):
        self._chapters, self._heading = [], []
        p = Protein.from_file(DATA / "1cll.cif", chains="A")
        p.surface(kind="ses", resolution=0.5).ball_and_stick(atom_scale=0.26, bond_radius=0.10).cartoon()
        # Establish one initial pose. Every later change goes through the timeline.
        ca = p.positions[[r.ca for r in p.topology.residues if r.ca >= 0]]
        _, _, axes = np.linalg.svd(ca - ca.mean(0), full_matrices=False)
        p.orientation = np.stack((axes[1], axes[0], axes[2]))
        if np.linalg.det(p.orientation) < 0:
            p.orientation[2] *= -1
        p.center()
        self.calmodulin = p
        self.add(p)
        self.camera.frame(p, margin=1.40, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi, self.camera.depth_cue = 0.22, 0.08, 0.28
        identity = Text("CALMODULIN · 1CLL", font_size=23, color=MUTED, position=(0.94, 0.055), align="right")
        self.add(identity)

        self.heading(
            "Representations",
            "One protein. Every view.",
            "Cartoon · ribbon · ball and stick · solvent-excluded surface",
            intro=(p, identity),
        )
        label = self.badge("Cartoon")
        self.play(Write(label), self.camera.animate.orbit(0.12), run_time=1.8)
        for rep, name, color in [
            ("ribbon", "Ribbon", BLUE),
            ("ball_and_stick", "Ball and stick", GOLD),
            ("surface", "Molecular surface", TEAL),
        ]:
            next_label = self.badge(name, color)
            self.play(
                FadeOut(label),
                Representation(p, rep),
                Write(next_label),
                self.camera.animate.orbit(0.12),
                run_time=1.9,
            )
            self.play(self.camera.animate.orbit(0.09), run_time=1.1)
            label = next_label

        helix = p.select(residues=(69, 88))
        lobe = p.select(residues=(5, 38))
        self.heading(
            "Color + transparency",
            "Color flows through residues.",
            "One selection across every representation · staggered color and opacity",
        )
        self.play(
            FadeOut(label),
            Colorize(helix, TEAL, residue_delay=0.065),
            Colorize(lobe, GOLD, residue_delay=0.04),
            self.camera.animate.orbit(0.12),
            run_time=2.5,
        )
        self.play(
            SetOpacity(complement(helix | lobe), 0.2, residue_delay=0.006),
            self.camera.animate.orbit(0.10),
            run_time=2.0,
        )
        self.play(Representation(p, "cartoon"), self.camera.animate.orbit(0.10), run_time=1.8)
        self.play(SetOpacity(p, 1), self.camera.animate.orbit(0.08), run_time=1.5)

        self.heading(
            "Focus + annotation",
            "Stay with the same structure.",
            "Eased focus · 3D regions · vector Write / Unwrite · amino-acid callouts",
        )
        sphere = helix.highlight(style="sphere", color=TEAL, opacity=0.08, padding=0.8)
        box = helix.highlight(style="box", color=TEAL, opacity=0.5, padding=1.0, line_width=0.055)
        note = helix.callout(
            "Central helix",
            subtitle="Calmodulin · residues 69–88",
            font_size=38,
            color=TEAL,
            position=(0.065, 0.43),
        )
        self.play(
            self.fit(helix, 1.9),
            SetOpacity(complement(helix), 0.055),
            FadeIn(sphere),
            FadeIn(box),
            Write(note),
            run_time=2.8,
        )
        labels = p.label_residues(
            residues=[75, 84], font_size=32, color=GOLD, offsets={75: (320, -65), 84: (300, 75)}
        )
        halos = p.select(residues=[75, 84], atoms="CA").highlight(
            style="atoms", color=GOLD, opacity=0.35, padding=0.35
        )
        self.play(
            FadeOut(sphere),
            FadeOut(box),
            Representation(p, "ball_and_stick"),
            Write(labels),
            FadeIn(halos),
            run_time=1.8,
        )
        self.play(self.camera.animate.orbit(0.22), run_time=2.6)
        self.play(Unwrite(labels), Unwrite(note), FadeOut(halos), run_time=1.0)
        self.play(self.fit(p, 1.45), SetOpacity(p, 1), Colorize(p, None), run_time=2.5)

        self.heading(
            "States + deformation",
            "Calmodulin keeps moving.",
            "1CFC NMR conformers via XTC · slow illustrative interpolation, not MD",
        )
        trajectory = Trajectory.from_mdanalysis(
            DATA / "showcase-calmodulin.pdb", DATA / "showcase-calmodulin.xtc", selection="all"
        )
        original = p.positions.copy()
        nonprotein = Region(
            p, [i for i, a in enumerate(p.topology.atoms) if p.topology.residues[a.residue_index].ca < 0]
        )
        nmr_identity = Text(
            "CALMODULIN · 1CFC", font_size=23, color=MUTED, position=(0.94, 0.055), align="right"
        )
        self.play(
            Representation(p, "cartoon"),
            SetOpacity(nonprotein, 0),
            FadeOut(identity),
            Write(nmr_identity),
            run_time=1.5,
        )
        self.play(Morph(p, trajectory.frame(0), align=False), self.camera.animate.orbit(0.08), run_time=3.0)
        self.play(
            PlayTrajectory(p, trajectory, start=0, end=2, state_easing=smooth),
            self.camera.animate.orbit(0.24),
            run_time=7.0,
        )
        # Bridge back to the exact original coordinates before the interprotein morph.
        returned_identity = Text(
            "CALMODULIN · 1CLL", font_size=23, color=MUTED, position=(0.94, 0.055), align="right"
        )
        self.play(
            Morph(p, original, align=False), FadeOut(nmr_identity), Write(returned_identity), run_time=2.6
        )
        deformation = self.badge("Procedural deformation · eased return", GOLD)

        def bend(xyz):
            d = xyz - xyz.mean(0)
            xyz[:, 0] += 2.2 * np.sin(d[:, 1] * 0.10)
            return xyz

        self.play(Write(deformation), Deform(p, bend), run_time=1.8)
        self.play(
            Morph(p, original, align=False), FadeOut(deformation), SetOpacity(nonprotein, 1), run_time=1.8
        )

        self.heading(
            "Contact-guided morph",
            "Calmodulin becomes troponin C.",
            "114 matched Cα · 25 ms N → C delay · unmatched residues fade out / in",
        )
        target = (
            Protein.from_file(DATA / "1ncx.cif", chains="A")
            .ball_and_stick(atom_scale=0.26, bond_radius=0.10)
            .center()
        )
        match = ContactMatch.load(DATA / "calmodulin-troponin-match.json")
        matched_source = p.select(residues=[p.topology.residues[i].resid for i in match.source_indices])
        target.set_color("#8299ad")
        target.select(residues=[target.topology.residues[i].resid for i in match.target_indices]).set_color(
            GOLD
        )
        self.troponin = target
        self.play(
            Representation(p, "ball_and_stick"),
            Colorize(matched_source, TEAL),
            Colorize(complement(matched_source), "#8299ad"),
            run_time=1.8,
        )
        troponin_identity = Text(
            "TROPONIN C · 1NCX", font_size=23, color=GOLD, position=(0.94, 0.055), align="right"
        )
        self.morph_start = self.duration
        self.play(
            BackboneMorph(p, target, match=match, residue_delay=0.025),
            FadeOut(returned_identity),
            Write(troponin_identity),
            self.camera.animate.orbit(0.18),
            run_time=8.5,
        )
        self.morph_end = self.duration
        self.play(self.fit(target, 1.40), run_time=1.8)

        # Continue with the actual morph destination: never reload or recenter it.
        p = target
        helix = p.select(residues=(84, 99))
        backbone = p.select(residues=(84, 99), atoms=["N", "CA", "C", "O"])
        self.heading(
            "Hydrogen bonds",
            "Look inside the new structure.",
            "Troponin C backbone · N···O contacts · angles use inferred amide H",
        )
        self.play(
            self.fit(backbone, 1.35), Colorize(p, None), SetOpacity(complement(backbone), 0.015), run_time=2.8
        )
        hb = HydrogenBonds(
            p,
            donors=p.select(residues=(88, 99), atoms="N"),
            acceptors=p.select(residues=(84, 95), atoms="O"),
            hydrogens="backbone",
        )
        self.hbond_count = len(hb.pairs)
        network = hb.highlight(
            mode="3d", endpoints="donor_acceptor", show_distances=False, color=GOLD, radius=0.10, dash_count=5
        )
        self.play(Write(network), self.camera.animate.orbit(0.10), run_time=1.6)
        self.play(self.camera.animate.orbit(0.22), run_time=2.4)

        self.heading(
            "Live distances",
            "Measure without losing context.",
            "The same two Cα atoms · a depth-tested 3D line becomes a 2D overlay",
        )
        a, b = p.select(residues=84, atoms="CA"), p.select(residues=97, atoms="CA")
        solid = Distance(a, b, mode="3d", color=TEAL, font_size=38, radius=0.08, dash_count=13)
        overlay = Distance(a, b, mode="2d", color=GOLD, font_size=38, line_width=2.5, dash_count=13)
        ruler_label = self.badge("3D · in the model")
        self.play(FadeOut(network), Write(solid), Write(ruler_label), run_time=1.5)
        self.play(self.camera.animate.orbit(0.16), run_time=1.7)
        next_label = self.badge("2D · above the model", GOLD)
        self.play(FadeOut(solid), FadeOut(ruler_label), Write(overlay), Write(next_label), run_time=1.3)
        self.play(self.camera.animate.orbit(0.16), run_time=1.7)

        self.heading(
            "Electrostatics",
            "From a helix to a contact.",
            "Imported illustrative formal charges · screened Coulomb · εr = 80 · λ = 8 Å",
        )
        field = Electrostatics(
            p,
            charges=np.load(DATA / "showcase-troponin-charges.npy"),
            dielectric=80,
            screening_length=8,
            cutoff=6,
            min_energy=0.08,
        )
        pair = next(
            pair
            for pair in field.pairs
            if {p.topology.atoms[i].resid for i in (pair.a, pair.b)} == {91, 95} and pair.energy < 0
        )
        charged = p.select(residues=[91, 95])
        contact_atoms = Region(p, [pair.a, pair.b])
        self.charge_energy = pair.energy
        self.play(
            FadeOut(overlay),
            FadeOut(next_label),
            self.fit(charged, 2.15),
            SetOpacity(charged, 1),
            SetOpacity(complement(charged), 0.025),
            Colorize(p.select(residues=91), BLUE),
            Colorize(p.select(residues=95), GOLD),
            run_time=2.8,
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
        note = contact_atoms.callout(
            "Attractive contact",
            subtitle=f"{pair.energy:.2f} kcal/mol",
            font_size=38,
            color=BLUE,
            position=(0.065, 0.43),
        )
        self.play(Write(contacts), Write(note), run_time=1.5)
        self.play(self.camera.animate.orbit(0.2), run_time=2.5)

        self.heading(
            "One continuous scene",
            "Every detail. One connected story.",
            "Calmodulin → troponin C · authored in Python · rendered natively on Metal",
        )
        self.play(
            Unwrite(note),
            FadeOut(contacts),
            SetOpacity(p, 1),
            Colorize(p, None),
            self.fit(p, 1.40),
            run_time=3.0,
        )
        self.play(Representation(p, "cartoon"), self.camera.animate.orbit(0.18), run_time=2.0)
        brand = self.badge("ProteinMotion", TEAL)
        self.play(Write(brand), self.camera.animate.orbit(0.16), run_time=2.0)
        self.wait(1.0)
        self.play(
            FadeOut(p),
            FadeOut(brand),
            FadeOut(troponin_identity),
            *(FadeOut(t) for t in self._heading),
            run_time=1.2,
        )
        self.wait(0.2)
        self.duration = round(self.duration, 6)

    def chapter_manifest(self):
        self.build()
        ends = [c["start"] for c in self._chapters[1:]] + [self.duration]
        return [
            dict(c, start=round(c["start"], 3), duration=round(end - c["start"], 3))
            for c, end in zip(self._chapters, ends)
        ]


if __name__ == "__main__":
    FeatureShowcase(width=1920, height=1080, fps=60).render("proteinmotion-showcase.mp4")
