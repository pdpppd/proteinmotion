"""Backbone hydrogen bonds of an α helix, i to i+4.

Input: ubiquitin, PDB 1UBQ, helix residues 23–34. Amide hydrogens are placed
on the backbone (hydrogens="backbone"); bonds need N···O ≤ 3.5 Å and an
N–H···O angle ≥ 150°.
"""

from pathlib import Path

from proteinmotion import HydrogenBonds, Protein, ProteinScene, StudioLook, Text, Write

DATA = Path(__file__).resolve().parent.parent / "data"


class HelixHydrogenBonds(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = (
            Protein.from_file(DATA / "1ubq.cif").ball_and_stick(atom_scale=0.26, bond_radius=0.1).center()
        )
        helix = protein.select(residues=(23, 34))
        backbone = protein.select(atoms=["N", "CA", "C", "O"])
        (~helix).set_opacity(0.05)
        (helix - backbone).set_opacity(0.2)
        bonds = HydrogenBonds(
            protein, donors=protein.select(residues=(23, 34), atoms="N"), hydrogens="backbone"
        )
        lines = bonds.highlight(mode="3d", color="#ffd27a", radius=0.13, dash_count=5)
        self.camera.theta, self.camera.phi = 0.2, 0.1
        self.camera.frame(helix, margin=0.95, screen_position=(0.6, 0.52))
        self.add(protein, Text("Hydrogen bonds", position=(0.05, 0.08), font_size=46, font="semibold"))
        self.add(
            Text(
                f"Ubiquitin helix 23–34 · {len(bonds.pairs)} backbone N–H···O=C",
                position=(0.05, 0.145),
                font_size=28,
                color="#a3b3c7",
            )
        )
        self.wait(0.5)
        self.play(Write(lines), run_time=2)
        self.play(self.camera.animate.orbit(1.2), run_time=5)
        self.wait(0.5)
