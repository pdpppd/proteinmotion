"""Metal ions on transfer RNA and spermine bound to Z-DNA.

Part 1: yeast tRNA-Phe (PDB 1EHZ) with its deposited Mg²⁺ and Mn²⁺ ions, which cartoons
show by default. Phosphate groups with an atom within 5 Å of an ion are then shown, and
the camera moves to the Mn²⁺ ion beside G19 and dihydrouridine 16.
Several ions bind through water molecules, which are not loaded here.
Part 2: the Z-DNA hexamer duplex d(CGCGCG) (PDB 2DCG) with two spermine molecules
and a Mg²⁺ ion; nucleotides within 3.5 Å of one spermine are shown as ball-and-stick.

Run from a source checkout:
    proteinmotion render examples/nucleic_ions.py NucleicIons --fps 60 -o nucleic-ions.mp4
"""

from pathlib import Path

from proteinmotion import (
    BaseStyle,
    FadeIn,
    FadeOut,
    Focus,
    NucleicAcid,
    ProteinScene,
    ShowAtoms,
    Text,
    Unwrite,
    Write,
)

DATA = Path(__file__).parent / "data"
TEAL, MUTED = "#50e0d0", "#a3b3c7"
PHOSPHATE = ["P", "OP1", "OP2", "O5'", "O3'"]


class NucleicIons(ProteinScene):
    def construct(self):
        aspect = self.width / self.height
        self.camera.depth_cue = 0.3

        # Part 1. Ions outside the traced strand draw over the cartoon.
        trna = NucleicAcid.from_file(DATA / "1ehz.cif").cartoon(bases="ladder").center()
        self.add(trna)
        self.camera.frame(trna, margin=0.8, aspect=aspect)
        self.camera.theta, self.camera.phi = 0.9, 0.1
        title = Text(
            "Ions and ligands on nucleic acids", font_size=56, font="semibold", position=(0.06, 0.07)
        )
        subtitle = Text(
            "Transfer RNA · PDB 1EHZ · Mg²⁺ and Mn²⁺ ions", font_size=26, color=MUTED, position=(0.062, 0.14)
        )
        self.play(Write(title, stroke_width=1.6), FadeIn(subtitle), run_time=1.6)
        self.play(self.camera.animate.orbit(0.7), run_time=3)

        ions = trna.select(ions=True)
        phosphates = trna.select(within=5.0, of=ions, atoms=PHOSPHATE)
        self.play(ShowAtoms(phosphates, residue_delay=0.03), self.camera.animate.orbit(0.5), run_time=2.5)
        manganese = trna.select(resname="MN", residues=530)
        site = trna.select(within=5.0, of=manganese)
        self.focus(site | manganese, margin=1.4, run_time=2)
        note = manganese.callout(
            "Mn²⁺", subtitle="Beside G19 and D16", position=(0.72, 0.28), font_size=38, color=TEAL
        )
        self.play(Write(note), ShowAtoms(site, residue_delay=0.2), run_time=1.5)
        self.play(self.camera.animate.orbit(0.8, 0.1), run_time=4)
        self.play(Unwrite(note), FadeOut(trna), FadeOut(subtitle), run_time=1.2)

        # Part 2. Spermine lies across the Z-DNA duplex; its carbons use the ligand color.
        dna = NucleicAcid.from_file(DATA / "2dcg.cif").cartoon(bases="slabs").center()
        # Spermine 15 spans both strands of the duplex.
        spermine = dna.select(resname="SPM", residues=15)
        contacts = dna.select(within=3.5, of=spermine)
        subtitle = Text(
            "Z-DNA d(CGCGCG) · PDB 2DCG · spermine and Mg²⁺",
            font_size=26,
            color=MUTED,
            position=(0.062, 0.14),
        )
        self.play(
            FadeIn(dna), FadeIn(subtitle), Focus(self.camera, dna, margin=1.1, aspect=aspect), run_time=1.5
        )
        self.play(self.camera.animate.orbit(0.8), run_time=3)
        note = spermine.callout(
            "Spermine", subtitle="Across both strands", position=(0.72, 0.28), font_size=38, color=TEAL
        )
        self.play(Write(note), run_time=1.2)
        # Base slabs give way to the atoms of the contacted nucleotides.
        self.play(
            BaseStyle(dna, "none"),
            ShowAtoms(contacts, residue_delay=0.1),
            self.camera.animate.orbit(0.4),
            run_time=2.5,
        )
        self.play(self.camera.animate.orbit(0.9, 0.08), run_time=4.5)
        self.play(Unwrite(note), Unwrite(title), FadeOut(subtitle), run_time=1)
        self.wait(0.4)
