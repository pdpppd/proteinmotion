"""Open a ligand's binding pocket: fade the fold, grow the pocket side chains.

Input: E. coli ribose-binding protein with bound β-D-ribose (RIP A272),
PDB 2DRI. The pocket is every residue with an atom within 4 Å of the ligand.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, SetOpacity, ShowSideChains, StudioLook, Text, Unwrite, Write

DATA = Path(__file__).resolve().parent.parent / "data"


class BindingPocket(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").cartoon(color="#b8c4d6").center()
        ligand = protein.select(resname="RIP")
        pocket = protein.select(within=4.0, of=ligand)
        theta, phi = self.camera.clearest_view(ligand)
        self.camera.theta, self.camera.phi = theta, phi
        self.camera.frame(protein, margin=0.95, screen_position=(0.6, 0.52))
        self.add(protein, Text("Binding pocket", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            "Ribose-binding protein · RIP A272", position=(0.05, 0.145), font_size=28, color="#a3b3c7"
        )
        self.add(caption)
        self.wait(0.5)
        self.focus(ligand | pocket, margin=1.9, screen_position=(0.6, 0.52), run_time=1.8)
        self.play(
            caption.animate.set_text("protein.select(within=4.0, of=ligand)"),
            SetOpacity(~(pocket | ligand), 0.1),
            ShowSideChains(pocket, residue_delay=0.08),
            run_time=2,
        )
        labels = protein.label_residues(residues=[89, 141, 190, 215], font_size=30, color="#ffc46b")
        self.play(Write(labels, lag_ratio=0.1), self.camera.animate.orbit(0.3), run_time=1.5)
        self.play(self.camera.animate.orbit(0.6), run_time=2.5)
        self.play(Unwrite(labels), SetOpacity(protein, 1, scope="residues"), run_time=1)
        self.focus(protein, margin=0.95, screen_position=(0.6, 0.52), run_time=1.5)
