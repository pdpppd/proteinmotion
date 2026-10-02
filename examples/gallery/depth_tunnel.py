"""Show how deep a buried site lies with a ringed tunnel.

Input: E. coli ribose-binding protein with bound β-D-ribose (RIP A272),
PDB 2DRI, as a 0.6 Å voxel SES. shape="tunnel" draws a ring every 5 Å from
the outer surface to the ligand; the tunnel stays fixed to the protein.
"""

from pathlib import Path

from proteinmotion import Conceal, Focus, Protein, ProteinScene, Reveal, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class DepthTunnel(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="dramatic", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.surface(kind="ses", grid_spacing=0.6, color="#9fb7e8")
        ligand = protein.select(resname="RIP")
        ligand.filter(element="C").set_color("#ffc46b")
        self.camera.theta, self.camera.phi = 1.5, 0.25
        self.camera.frame(protein, margin=1.1, screen_position=(0.6, 0.53))
        self.add(protein, Text("Depth tunnels", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text(
            'Reveal(camera, ligand, shape="tunnel", rings=5)',
            position=(0.05, 0.145),
            font_size=28,
            color="#a3b3c7",
        )
        self.add(caption)
        self.wait(0.8)
        self.play(
            Reveal(self.camera, ligand, window=1.6, shape="tunnel", rings=5, padding=2, surface_keep=0.15),
            Focus(self.camera, ligand, margin=5, screen_position=(0.6, 0.53)),
            run_time=2,
        )
        depth = self.camera.cutaway_geometry()[3]
        self.play(
            caption.animate.set_text(f"Ribose lies {depth:.0f} Å below the surface · ring every 5 Å"),
            run_time=0.6,
        )
        self.play(self.camera.animate.orbit(0.35, 0.15), run_time=3)
        self.play(self.camera.animate.orbit(-0.35, -0.15), run_time=2)
        self.play(
            Conceal(self.camera),
            Focus(self.camera, protein, margin=1.1, screen_position=(0.6, 0.53)),
            run_time=1.5,
        )
