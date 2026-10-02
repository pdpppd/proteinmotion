"""Studio materials and film effects on the same cartoon.

Input: E. coli ribose-binding protein, PDB 2DRI chain A, with bound ribose.
Materials and effects are fixed when the renderer is created, so each look
is its own scene; the gallery tiles the four renders into one 2 × 2 video.
"""

from pathlib import Path

from proteinmotion import Protein, ProteinScene, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class MaterialScene(ProteinScene):
    renderer = "studio"
    label = ""

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.cartoon(color="#e8cda5")
        protein.select(resname="RIP").show_atoms()
        self.camera.theta, self.camera.phi = 0.4, 0.2
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.56))
        self.add(protein, Text(self.label, position=(0.05, 0.06), font_size=64))
        self.play(self.camera.animate.orbit(1.3), run_time=6)


class RoughMaterial(MaterialScene):
    label = 'material="rough"'
    look = StudioLook(lighting="studio", material="rough", effects="clean")


class MediumMaterial(MaterialScene):
    label = 'material="medium"'
    look = StudioLook(lighting="studio", material="medium", effects="clean")


class GlossyMaterial(MaterialScene):
    label = 'material="glossy"'
    look = StudioLook(lighting="studio", material="glossy", effects="clean")


class FilmEffects(MaterialScene):
    label = 'effects="film"'
    look = StudioLook(lighting="studio", material="glossy", effects="film", halation="warm")
