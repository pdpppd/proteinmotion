"""Van der Waals, solvent-accessible, and solvent-excluded surfaces side by side.

Input: ubiquitin, PDB 1UBQ. Each copy is a voxel surface at 0.6 Å spacing. SES
uses cavity filling and erosion; it approximates the analytical surface.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, Representation, Rotate, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class SurfaceKinds(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="subtle")

    def construct(self):
        base = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        kinds = {"vdw": "van der Waals", "sas": "Solvent accessible", "ses": "Solvent excluded"}
        copies = [base.copy().shift(((i - 1) * 52, 0, 0)) for i in range(3)]
        self.camera.theta, self.camera.phi = 0, 0
        self.camera.frame(*copies, margin=1.0, screen_position=(0.5, 0.5))
        self.camera.zoom(1.5)
        self.add(*copies, Text("Molecular surfaces", position=(0.05, 0.08), font_size=46, font="semibold"))
        for i, (kind, label) in enumerate(kinds.items()):
            self.add(Text(f'kind="{kind}"', position=(0.2 + 0.3 * i, 0.8), font_size=30, align="center"))
            self.add(
                Text(label, position=(0.2 + 0.3 * i, 0.855), font_size=24, color="#a3b3c7", align="center")
            )
        self.wait(0.5)
        self.play(
            *(Representation(p, "surface", kind=kind, grid_spacing=0.6) for p, kind in zip(copies, kinds)),
            run_time=2,
        )
        self.play(*(Rotate(p, 1.6) for p in copies), run_time=4)
        self.play(
            *(Representation(p, "surface", kind=kind, color="hydropathy") for p, kind in zip(copies, kinds)),
            run_time=1.5,
        )
        self.play(*(Rotate(p, 1.2) for p in copies), run_time=3)
