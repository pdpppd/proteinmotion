"""Write, restyle, move, and erase text over a scene.

Write traces each glyph's outline and then fills it; set_text crossfades the
string; TextGroup stacks a title and body. Text uses 1080p pixel sizes and
normalized screen positions, independent of the output resolution.
"""

from pathlib import Path

from proteinmotion import (
    FadeIn,
    Protein,
    ProteinScene,
    StudioLook,
    Text,
    TextGroup,
    TextStyle,
    Unwrite,
    Write,
)

DATA = Path(__file__).resolve().parent.parent / "data"


class TextAnimation(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "1ubq.cif").cartoon().center()
        self.camera.frame(protein, margin=0.9, screen_position=(0.68, 0.55))
        title = Text("Ubiquitin", font_size=110, font="semibold", position=(0.06, 0.3))
        self.play(Write(title, lag_ratio=0.12, stroke_width=1.8), FadeIn(protein), run_time=2)
        body = TextStyle(font_size=30, color="#a3b3c7")
        caption = TextGroup(
            Text("76 residues · PDB 1UBQ", style=body),
            Text("β-grasp fold: a β sheet wrapped around one α helix", style=body, max_width=560),
            position=(0.065, 0.5),
            gap=14,
        )
        self.play(Write(caption), self.camera.animate.orbit(0.3), run_time=1.5)
        self.play(title.animate.move_to((0.06, 0.12)), caption.animate.move_to((0.065, 0.3)), run_time=1.2)
        greek = Text("α → β", font_size=80, color="#50e0d0", position=(0.06, 0.55))
        self.play(Write(greek), self.camera.animate.orbit(0.4), run_time=1.5)
        self.play(greek.animate.set_text("β → α"), run_time=1)
        self.wait(0.5)
        self.play(Unwrite(title), Unwrite(caption), Unwrite(greek), run_time=1.5)
