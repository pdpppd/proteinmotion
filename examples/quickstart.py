"""A first film: rotation, a live region highlight, focus, and ball-and-stick."""

from pathlib import Path

from proteinmotion import FadeIn, FadeOut, Protein, ProteinScene, Representation, Rotate

DATA = Path(__file__).parent / "data" / "1ubq.cif"


class Quickstart(ProteinScene):
    def construct(self):
        protein = Protein.from_file(DATA).cartoon().center()
        self.add(protein)
        self.camera.frame(protein, aspect=self.width / self.height)

        self.play(FadeIn(protein), run_time=0.8)
        self.play(Rotate(protein, 1.2), run_time=2)

        helix = protein.select(chain="A", residues=(23, 34))
        highlight = helix.highlight(style="box", color="#f2ba67")
        self.play(FadeIn(highlight), run_time=0.6)
        self.focus(helix, margin=1.7, run_time=1.5)
        self.play(Rotate(protein, 0.5), run_time=1.5)
        self.play(FadeOut(highlight), run_time=0.6)
        self.focus(protein, run_time=1.2)

        self.play(Representation(protein, "ball_and_stick"), run_time=1)
        self.wait(0.8)


if __name__ == "__main__":
    Quickstart(width=1920, height=1080, fps=60).render("first-film.mp4")
