"""Mark a selection with a sphere, a wire box, or an atom halo.

Input: ubiquitin, PDB 1UBQ, helix residues 23–34. Highlights follow the
selection's coordinates; FadeIn/FadeOut animate their opacity.
"""

from pathlib import Path

from proteinmotion import FadeIn, FadeOut, Protein, ProteinScene, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class Highlights(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        helix = protein.select(residues=(23, 34))
        self.camera.frame(protein, margin=0.9, screen_position=(0.6, 0.52))
        self.add(protein, Text("Region highlights", position=(0.05, 0.08), font_size=46, font="semibold"))
        caption = Text("", position=(0.05, 0.145), font_size=28, color="#a3b3c7")
        self.add(caption)
        for style, color, opacity in (
            ("sphere", "#ffc46b", 0.3),
            ("box", "#ff8fb1", 0.9),
            ("atoms", "#9fe6ff", 0.6),
        ):
            marker = helix.highlight(style=style, color=color, padding=1.0, opacity=opacity)
            self.play(
                caption.animate.set_text(f'helix.highlight(style="{style}")'),
                FadeIn(marker),
                self.camera.animate.orbit(0.25),
                run_time=0.8,
            )
            self.play(self.camera.animate.orbit(0.45), run_time=1.6)
            self.play(FadeOut(marker), run_time=0.6)
        self.wait(0.3)
