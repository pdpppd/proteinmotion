"""The four Studio lighting presets on the same surface.

Input: E. coli ribose-binding protein, PDB 2DRI chain A, as a 0.6 Å voxel SES.
Lighting is fixed when the renderer is created, so each preset is its own
scene; the gallery tiles the four renders into one 2 × 2 video.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class LightingScene(ProteinScene):
    renderer = "studio"
    lighting = "studio"

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.surface(kind="ses", grid_spacing=0.6, color="secondary")
        self.camera.theta, self.camera.phi = 1.2, 0.25
        self.camera.frame(protein, margin=1.25, screen_position=(0.6, 0.56))
        self.add(protein, Text(f'lighting="{self.lighting}"', position=(0.05, 0.06), font_size=64))
        self.play(self.camera.animate.orbit(1.3), run_time=6)


class StudioLighting(LightingScene):
    lighting = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="clean")


class SoftLighting(LightingScene):
    lighting = "soft"
    look = StudioLook(lighting="soft", material="medium", effects="clean")


class DramaticLighting(LightingScene):
    lighting = "dramatic"
    look = StudioLook(lighting="dramatic", material="medium", effects="clean")


class FlatLighting(LightingScene):
    lighting = "flat"
    look = StudioLook(lighting="flat", material="medium", effects="clean")
