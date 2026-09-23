"""Cadmium sites and a bound sulfate in troponin C (PDB 1NCX, chain A).

The two Cd²⁺ ions occupy EF-hands III and IV in this crystal structure. Residues with an
atom within 3 Å of each ion are shown; the sulfate sits against Arg47. The switch to
ball-and-stick and back shows how the atom palette blends between representations.

Run from a source checkout:
    proteinmotion render examples/troponin_sites.py TroponinSites --fps 60 -o troponin-sites.mp4
"""

from pathlib import Path

from proteinmotion import (
    FadeIn,
    HideAtoms,
    Protein,
    ProteinScene,
    Representation,
    ShowSideChains,
    Text,
    Unwrite,
    Write,
)

DATA = Path(__file__).parent / "data"
GOLD, TEAL, MUTED = "#f2ba67", "#50e0d0", "#a3b3c7"


class TroponinSites(ProteinScene):
    def construct(self):
        aspect = self.width / self.height
        p = Protein.from_file(DATA / "1ncx.cif").cartoon().center()
        self.add(p)
        self.camera.frame(p, margin=0.95, aspect=aspect)
        self.camera.theta, self.camera.phi = -0.3, 0.2
        self.camera.depth_cue = 0.3

        title = Text("Troponin C metal sites", font_size=56, font="semibold", position=(0.06, 0.07))
        subtitle = Text(
            "PDB 1NCX · two Cd²⁺ ions and a sulfate, shown by default",
            font_size=26,
            color=MUTED,
            position=(0.062, 0.14),
        )
        self.play(Write(title, stroke_width=1.6), FadeIn(subtitle), run_time=1.6)
        self.play(self.camera.animate.orbit(0.6), run_time=2.5)

        # EF-hands III and IV: whole residues within 3 Å of each ion.
        ions = p.select(ions=True)
        sites = p.select(within=3.0, of=ions)
        self.focus(sites | ions, margin=1.15, run_time=2)
        self.play(ShowSideChains(sites, residue_delay=0.12), run_time=2.2)
        site_three = p.select(ions=True, residues=163).callout(
            "Cd²⁺", subtitle="EF-hand III · Asp106–Glu117", position=(0.70, 0.24), font_size=36, color=GOLD
        )
        site_four = p.select(ions=True, residues=164).callout(
            "Cd²⁺", subtitle="EF-hand IV · Asp142–Glu153", position=(0.70, 0.72), font_size=36, color=GOLD
        )
        self.play(Write(site_three), Write(site_four), run_time=1.6)
        self.play(self.camera.animate.orbit(0.8, 0.08), run_time=4)
        self.play(Unwrite(site_three), Unwrite(site_four), run_time=0.8)

        # The sulfate and the arginine that holds it.
        sulfate = p.select(resname="SO4")
        arginine = p.select(within=3.0, of=sulfate)
        note = sulfate.callout(
            "SO₄²⁻", subtitle="Held by Arg47", position=(0.72, 0.30), font_size=38, color=TEAL
        )
        self.focus(sulfate | arginine, margin=1.4, run_time=2)
        self.play(ShowSideChains(arginine), Write(note), run_time=1.5)
        self.play(self.camera.animate.orbit(0.6), run_time=2.5)
        self.play(Unwrite(note), run_time=0.6)

        # Every atom, then back: shown atoms keep their state and blend their colors.
        self.focus(p, margin=0.95, run_time=1.8)
        self.play(Representation(p, "ball_and_stick"), run_time=1.6)
        self.play(self.camera.animate.orbit(0.5), run_time=2)
        self.play(Representation(p, "cartoon"), run_time=1.6)
        self.play(HideAtoms(sulfate), self.camera.animate.orbit(0.4), run_time=1.6)
        self.play(Unwrite(title), run_time=0.8)
        self.wait(0.4)
