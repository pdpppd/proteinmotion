"""AlphaFold confidence: pLDDT on the model and predicted aligned error.

Input: AlphaFold DB model of human calmodulin, AF-P0DP23-F1 v6, with its PAE
file (CC BY 4.0). Each lobe is predicted with confidence, but the high PAE
between lobes shows that their relative position is not.
"""

from pathlib import Path

from proteinmotion import ColorLegend, FadeIn, Heatmap, Protein, ProteinScene, ResidueValues, StudioLook, Text

DATA = Path(__file__).resolve().parent.parent / "data"


class AlphaFoldConfidence(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="medium", effects="subtle")

    def construct(self):
        model = Protein.from_file(DATA / "AF-P0DP23-F1-model_v6.cif").cartoon().center()
        model.color_by("plddt")
        c_lobe = model.select(residues=(82, 148))
        pae = Heatmap.pae(
            model,
            DATA / "AF-P0DP23-F1-predicted_aligned_error_v6.json",
            highlight=c_lobe,
            position=(0.63, 0.16),
            size=(0.34, 0.6),
        ).with_panel()
        self.camera.frame(model, margin=0.85, screen_position=(0.31, 0.52))
        self.add(model, Text("AlphaFold confidence", position=(0.05, 0.08), font_size=46, font="semibold"))
        self.add(Text("Calmodulin · AF-P0DP23-F1", position=(0.05, 0.145), font_size=28, color="#a3b3c7"))
        self.play(
            FadeIn(pae), FadeIn(ColorLegend(ResidueValues.plddt(model), position=(0.05, 0.82))), run_time=1.2
        )
        self.play(self.camera.animate.orbit(1.4), run_time=7)
