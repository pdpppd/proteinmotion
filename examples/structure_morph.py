"""Morphs between two experimental structures of the same protein.

HemoglobinMorph moves human deoxyhemoglobin (T state, PDB 2DN2) to
oxyhemoglobin (R state, PDB 2DN1, whose deposited αβ dimer is expanded to the
tetramer with assembly 1). The structures are superposed on α1β1, so the change
appears as the rotation of α2β2. A heatmap shows how each Cα–Cα distance changes.

AdenylateKinaseMorph closes E. coli adenylate kinase: open and empty (PDB 4AKE)
to closed around the inhibitor Ap5A (PDB 1AKE), superposed on the CORE domain.
The LID and NMP-binding domains swing shut; Ap5A, present only in 1AKE, fades in.

Cas9Morph moves S. pyogenes Cas9 with its guide RNA (PDB 4ZT0) to the R-loop
complex with target DNA (PDB 5F9R). By default the morph superposes the parts
that do not move, here mostly the nuclease lobe; the HNH nuclease domain and
REC2 then swing as the DNA, present only in 5F9R, fades in. The two entries are
downloaded from the PDB on first use and cached.

The morph moves each residue along a screw path between the two deposited
structures. It shows how the endpoints differ, not the pathway between them.

Run from a source checkout:
    proteinmotion render examples/structure_morph.py HemoglobinMorph --fps 60 -o hemoglobin.mp4
    proteinmotion render examples/structure_morph.py AdenylateKinaseMorph --fps 60 -o adenylate-kinase.mp4
    proteinmotion render examples/structure_morph.py Cas9Morph --fps 60 -o cas9.mp4
"""

from pathlib import Path

from proteinmotion import (
    Distance,
    FadeIn,
    FadeOut,
    Focus,
    Heatmap,
    Protein,
    ProteinScene,
    SetOpacity,
    StructureMorph,
    StudioLook,
    Text,
    Unwrite,
    Write,
    domain_motion,
)

DATA = Path(__file__).parent / "data"
MUTED, GOLD, TEAL, ROSE, SLATE = "#a3b3c7", "#f5d477", "#50e0d0", "#ef8a9c", "#7f93b3"
ORANGE, PALE = "#f2a65a", "#e8eef8"
LOOK = StudioLook(lighting="soft", material="medium", effects="clean")


def caption(text):
    return Text(text, font_size=30, color="#edf3fc", position=(0.05, 0.205))


class HemoglobinMorph(ProteinScene):
    renderer = "studio"
    look = LOOK

    def construct(self):
        deoxy = Protein.from_file(DATA / "2dn2.cif").cartoon()
        oxy = Protein.from_file(DATA / "2dn1.cif", assembly="1").cartoon()
        oxy.select(resname="MBN").hide_atoms()  # toluene from crystallization
        for chain, color in (("A", SLATE), ("B", "#9fb0c9"), ("C", TEAL), ("D", GOLD)):
            deoxy.select(chain=chain, polymer=True).set_color(color)
        dimer = deoxy.select(chain=["A", "B"])
        motion = domain_motion(deoxy, oxy, moving=deoxy.select(chain=["C", "D"]), fixed=dimer)
        deoxy.center()
        self.add(deoxy)
        # Look down the axis of the α2β2 rotation so it turns in the image plane.
        self.camera.look_along(motion.axis, deoxy)
        self.camera.frame(deoxy, margin=1.08, aspect=self.width / self.height, screen_position=(0.32, 0.6))

        title = Text("Hemoglobin: T to R", font_size=56, font="semibold", position=(0.05, 0.07))
        subtitle = Text(
            "Deoxy (PDB 2DN2) to oxy (PDB 2DN1, assembly 1), superposed on α1β1",
            font_size=26,
            color=MUTED,
            position=(0.051, 0.14),
        )
        change = Heatmap.distances(
            deoxy,
            reference=deoxy.copy(),
            max_distance=6,
            title="Change in Cα distance",
            position=(0.64, 0.16),
            size=(0.33, 0.6),
        ).with_panel()
        self.play(Write(title), FadeIn(subtitle), FadeIn(change), run_time=1.6)
        note = caption("Deoxy: the T state, α1β1 in blue-gray and α2β2 in teal and gold")
        self.play(Write(note), run_time=1.4)
        self.wait(1)
        self.play(Unwrite(note), run_time=0.6)
        note = caption(f"Oxygen bound: α2β2 turns {motion.angle:.0f}° relative to α1β1")
        self.play(Write(note), StructureMorph(deoxy, oxy, align=dimer), run_time=6)
        oxygen = oxy.select(resname="OXY", chain="C").callout(
            "O₂", position=(0.5, 0.86), font_size=36, color=ROSE
        )
        self.play(Write(oxygen), run_time=1.2)
        self.play(Unwrite(note), run_time=0.6)
        note = caption("Distances within each αβ dimer barely change; those between dimers do")
        self.play(Write(note), self.camera.animate.orbit(0.5, 0.1), run_time=3)
        self.wait(1.5)
        self.play(Unwrite(note), Unwrite(title), Unwrite(oxygen), run_time=0.8)


class AdenylateKinaseMorph(ProteinScene):
    renderer = "studio"
    look = LOOK

    def construct(self):
        open_ = Protein.from_file(DATA / "4ake.cif", chains="A").cartoon()
        closed = Protein.from_file(DATA / "1ake.cif", chains="A").cartoon()
        core = open_.select(residues=list(range(1, 30)) + list(range(60, 122)) + list(range(160, 215)))
        lid, nmp = open_.select(residues=(122, 159)), open_.select(residues=(30, 59))
        open_.set_color(SLATE)
        open_.color_residues(GOLD, residues=(122, 159)).color_residues(TEAL, residues=(30, 59))
        swing = domain_motion(open_, closed, moving=lid, fixed=core)
        shut = domain_motion(open_, closed, moving=nmp, fixed=core)
        open_.center()
        self.add(open_)
        self.camera.look_along(swing.axis, open_)
        self.camera.frame(open_, margin=1.0, aspect=self.width / self.height, screen_position=(0.45, 0.6))

        title = Text("Adenylate kinase closes", font_size=56, font="semibold", position=(0.05, 0.07))
        subtitle = Text(
            "Open (PDB 4AKE) to closed around Ap5A (PDB 1AKE), superposed on the CORE",
            font_size=26,
            color=MUTED,
            position=(0.051, 0.14),
        )
        ruler = Distance(
            open_.select(residues=151, atoms="CA"),
            open_.select(residues=41, atoms="CA"),
            color=ROSE,
            prefix="LID–NMP  ",
            font_size=30,
            follow_opacity=False,
        )
        self.play(Write(title), FadeIn(subtitle), run_time=1.6)
        note = caption("LID in gold, NMP-binding domain in teal, CORE in gray")
        self.play(Write(note), Write(ruler), run_time=1.5)
        self.wait(1)
        self.play(Unwrite(note), run_time=0.6)
        note = caption(
            f"The LID turns {swing.angle:.0f}° and the NMP domain {shut.angle:.0f}° to close around Ap5A"
        )
        self.play(Write(note), StructureMorph(open_, closed, align=core), run_time=6)
        inhibitor = closed.select(resname="AP5").callout(
            "Ap5A", subtitle="bisubstrate inhibitor", position=(0.7, 0.42), font_size=36, color=ROSE
        )
        self.play(Write(inhibitor), run_time=1.2)
        self.play(self.camera.animate.orbit(0.6, 0.15), run_time=3.5)
        self.wait(1)
        self.play(Unwrite(note), Unwrite(title), Unwrite(inhibitor), run_time=0.8)


class Cas9Morph(ProteinScene):
    renderer = "studio"
    look = LOOK

    def construct(self):
        # Thicker nucleotide tubes keep the guide RNA and DNA readable among 1,368 residues.
        nucleotides = dict(backbone_radius=0.7, base_thickness=0.45)
        bound = Protein.fetch("4ZT0", chains=["A", "B"]).cartoon(**nucleotides)  # first of two copies
        rloop = Protein.fetch("5F9R").cartoon(**nucleotides)
        rec_residues = list(range(94, 180)) + list(range(308, 714))
        hnh = bound.select(chain="A", residues=(775, 908))
        rec2 = bound.select(chain="A", residues=(180, 307))
        bound.set_color(SLATE)
        bound.select(chain="A", residues=rec_residues).set_color(MUTED)
        rec2.set_color(TEAL)
        hnh.set_color(GOLD)
        bound.select(chain="B").set_color(ORANGE)  # guide RNA
        rloop.select(chain="A").set_color(ORANGE)  # guide RNA, including nucleotides only 5F9R resolves
        rloop.select(chain="C").set_color(ROSE)  # target strand
        rloop.select(chain="D").set_color(PALE)  # non-target strand
        turn = domain_motion(bound, rloop, moving=hnh)
        swing = domain_motion(bound, rloop, moving=rec2)
        bound.center()
        self.add(bound)
        # Look down the HNH rotation axis so the domain turns in the image plane.
        self.camera.look_along(turn.axis, bound)
        place = dict(aspect=self.width / self.height, screen_position=(0.56, 0.57))
        self.camera.frame(bound, margin=0.95, **place)

        title = Text("Cas9 grips its DNA target", font_size=56, font="semibold", position=(0.05, 0.07))
        subtitle = Text(
            "Guide-bound Cas9 (PDB 4ZT0) to the R-loop complex with DNA (PDB 5F9R)",
            font_size=26,
            color=MUTED,
            position=(0.051, 0.14),
        )
        self.play(Write(title), FadeIn(subtitle), run_time=1.6)
        note = caption("Guide RNA in orange; HNH nuclease domain in gold, REC2 in teal")
        self.play(Write(note), run_time=1.4)
        self.wait(1)
        self.play(Unwrite(note), run_time=0.6)
        note = caption(
            f"Target DNA pairs with the guide: HNH turns {turn.angle:.0f}°, REC2 {swing.angle:.0f}°"
        )
        self.play(Write(note), StructureMorph(bound, rloop), run_time=7)
        self.play(Unwrite(note), run_time=0.6)
        # Fade REC1 and REC3 to show the guide–DNA hybrid against HNH.
        strand = rloop.select(chain="C").callout(
            "target strand", subtitle="cut by HNH", position=(0.84, 0.8), font_size=34, color=ROSE
        )
        note = caption("With the REC lobe faded, HNH rests against the strand it cuts")
        self.play(
            SetOpacity(rloop.select(chain="B", residues=rec_residues), 0.18),
            Focus(self.camera, rloop, margin=0.92, follow=False, **place),
            Write(note),
            Write(strand),
            run_time=2,
        )
        self.play(self.camera.animate.orbit(0.35, 0.08), run_time=3.5)
        self.wait(1)
        self.play(Unwrite(note), Unwrite(title), Unwrite(strand), FadeOut(subtitle), run_time=0.8)
