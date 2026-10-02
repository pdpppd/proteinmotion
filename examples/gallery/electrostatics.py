"""Screened Coulomb estimates between charged side chains.

Input: ubiquitin, PDB 1UBQ, with formal charges on Lys, Arg, Asp, Glu, and
the termini. Energies use a Debye-screened Coulomb term (dielectric 80,
8 Å screening): an illustrative estimate, not a Poisson–Boltzmann result.
Blue lines attract; red lines repel.
"""

from pathlib import Path

from proteinmotion import Electrostatics, Protein, ProteinScene, ShowSideChains, StudioLook, Text, Write

DATA = Path(__file__).resolve().parent.parent / "data"


class ChargePairs(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon(color="#b8c4d6").center()
        charged = protein.select(resname=["LYS", "ARG", "ASP", "GLU"])
        field = Electrostatics(protein, charges="formal", cutoff=7, min_energy=0.1)
        contacts = field.highlight(mode="3d", max_pairs=16, radius=0.2)
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.52))
        self.add(protein, Text("Charge interactions", position=(0.05, 0.08), font_size=46, font="semibold"))
        self.add(
            Text("Formal charges · screened Coulomb", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        )
        self.add(Text("attractive", position=(0.05, 0.84), font_size=28, color="#58b8fa"))
        self.add(Text("repulsive", position=(0.05, 0.89), font_size=28, color="#ef7484"))
        self.play(ShowSideChains(charged, residue_delay=0.03), self.camera.animate.orbit(0.3), run_time=2)
        self.play(Write(contacts), run_time=1.5)
        self.play(self.camera.animate.orbit(1.1), run_time=5)
