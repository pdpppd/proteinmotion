"""AlphaFold confidence for calmodulin, then hydropathy and conservation on ubiquitin.

Part 1 colors the AlphaFold DB model of human calmodulin (AF-P0DP23-F1, v6) by
pLDDT and shows its predicted aligned error. The two lobes are confident on their
own, but their relative position is not: the PAE blocks between them are high.
Part 2 colors the ubiquitin surface (PDB 1UBQ) by Kyte–Doolittle hydropathy, then
by conservation across the Pfam ubiquitin family seed alignment (PF00240).
"""

from pathlib import Path

from proteinmotion import (
    ColorByProperty,
    ColorLegend,
    FadeIn,
    FadeOut,
    Focus,
    Heatmap,
    Protein,
    ProteinScene,
    ResidueValues,
    Text,
    Unwrite,
    Write,
)

DATA = Path(__file__).parent / "data"
MUTED = "#a3b3c7"


class ConfidenceAndConservation(ProteinScene):
    def construct(self):
        aspect = self.width / self.height
        model = Protein.from_file(DATA / "AF-P0DP23-F1-model_v6.cif").cartoon().center()
        model.color_by("plddt")
        c_lobe = model.select(residues=(82, 148))
        pae = Heatmap.pae(
            model,
            DATA / "AF-P0DP23-F1-predicted_aligned_error_v6.json",
            highlight=c_lobe,
            position=(0.63, 0.14),
            size=(0.34, 0.6),
        ).with_panel()
        plddt = ColorLegend(ResidueValues.plddt(model), position=(0.05, 0.82))
        title = Text("AlphaFold confidence", font_size=46, font="semibold", position=(0.05, 0.07))
        subtitle = Text("Calmodulin, AF-P0DP23-F1", font_size=26, color=MUTED, position=(0.051, 0.135))
        self.add(model)
        self.camera.frame(model, margin=0.95, aspect=aspect, screen_position=(0.32, 0.5))
        self.play(Write(title), FadeIn(subtitle), FadeIn(pae), FadeIn(plddt), run_time=1.5)
        self.play(self.camera.animate.orbit(0.8), run_time=4)
        self.play(FadeOut(model), FadeOut(pae), FadeOut(plddt), FadeOut(subtitle), Unwrite(title), run_time=1)

        ubiquitin = Protein.from_file(DATA / "1ubq.cif").surface(color="hydropathy").center()
        hydropathy = ResidueValues.hydropathy(ubiquitin)
        conservation = ResidueValues.conservation(ubiquitin, DATA / "pf00240-seed.sto")
        title = Text("Hydropathy", font_size=46, font="semibold", position=(0.05, 0.07))
        subtitle = Text("Ubiquitin surface, PDB 1UBQ", font_size=26, color=MUTED, position=(0.051, 0.135))
        legend = ColorLegend(hydropathy, position=(0.05, 0.82))
        self.camera.theta, self.camera.phi = 0.0, 0.0
        self.play(
            FadeIn(ubiquitin),
            Write(title),
            FadeIn(subtitle),
            FadeIn(legend),
            Focus(self.camera, ubiquitin, margin=0.9, aspect=aspect),
            run_time=1.5,
        )
        self.play(self.camera.animate.orbit(0.9), run_time=3)
        swap = Text("Conservation", font_size=46, font="semibold", position=(0.05, 0.07))
        source = Text(
            "Pfam PF00240 seed alignment, 59 sequences", font_size=26, color=MUTED, position=(0.051, 0.135)
        )
        consurf = ColorLegend(conservation, position=(0.05, 0.82))
        self.play(
            ColorByProperty(ubiquitin, conservation, stagger=0.4),
            Unwrite(title),
            FadeOut(subtitle),
            FadeOut(legend),
            Write(swap),
            FadeIn(source),
            FadeIn(consurf),
            run_time=2.5,
        )
        self.play(self.camera.animate.orbit(1.2), run_time=4)
