"""Rendered website examples for Studio and readable scene authoring.

Render: proteinmotion render examples/studio_examples.py CartoonThreading --fps 60 -o cartoon.mp4
Refresh website clips: python scripts/render_docs_examples.py --only studio-threading studio-atoms studio-cutaway readable-timing

Inputs: deposited PDB 2DRI (E. coli ribose-binding protein, chain A, RIP A272)
and 1UBQ (ubiquitin, chain A), from https://www.rcsb.org/structure/2DRI and
https://www.rcsb.org/structure/1UBQ. Threading is an illustrative entrance, not
protein folding or molecular dynamics. The SES is a voxel approximation.
Download the structure files into data/ alongside this script.
"""

from pathlib import Path

from proteinmotion import (
    Conceal,
    FadeIn,
    MolecularStyle,
    Protein,
    ProteinScene,
    Representation,
    Reveal,
    StudioLook,
    Text,
    Thread,
    Write,
)

DATA = Path(__file__).parent / "data"
INPUTS = [DATA / "2dri.cif", DATA / "1ubq.cif"]


# docs:start studio-threading
class CartoonThreading(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="clean")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.cartoon(color="#e8cda5", atom_scale=0.22, bond_radius=0.1)
        self.camera.frame(protein, margin=1.12, screen_position=(0.62, 0.53))
        self.add(Text("Cartoon threading", position=(0.05, 0.09), font_size=48))
        self.add(Text("2DRI · Eased entrance", position=(0.05, 0.16), font_size=26))
        self.play(Thread(protein, easing="smooth", swirl=0.25, glow=0, seed=4), run_time=6)
        self.play(self.camera.animate.orbit(0.3), run_time=2)
        self.wait(1)


# docs:end studio-threading


# docs:start studio-atoms
class AtomThreading(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="studio", material="glossy", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.ball_and_stick(atom_scale=0.2, bond_radius=0.1)
        self.camera.frame(protein, margin=1.12, screen_position=(0.62, 0.53))
        self.add(Text("Atom threading", position=(0.05, 0.09), font_size=48))
        self.add(Text("2DRI · Ball and stick", position=(0.05, 0.16), font_size=26))
        self.play(Thread(protein, easing="smooth", swirl=0.25, glow=0, seed=4), run_time=6)
        self.play(self.camera.animate.orbit(0.3), run_time=2)
        self.wait(1)


# docs:end studio-atoms


# docs:start studio-cutaway
class BuriedLigand(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="dramatic", material="medium", effects="subtle")

    def construct(self):
        protein = Protein.from_file(DATA / "2dri.cif", chains="A").center()
        protein.cartoon(color="#8fe3c0", atom_scale=0.3, bond_radius=0.12)
        ligand = protein.select(chain="A", resname="RIP", residues=272)
        ligand.filter(element="C").set_color("#ffc46b")
        self.camera.theta, self.camera.phi = 1.5, 0.25
        self.camera.frame(protein, margin=1.1, screen_position=(0.63, 0.53))
        self.add(protein)
        self.add(Text("Buried ribose", position=(0.05, 0.09), font_size=48))
        caption = Text("2DRI · Cartoon", position=(0.05, 0.16), font_size=26)
        self.add(caption)
        self.wait(1)
        self.play(
            Representation(protein, "surface", kind="ses", grid_spacing=0.6),
            caption.animate.set_text("Solvent-excluded surface"),
            run_time=1.5,
        )
        self.play(
            Reveal(self.camera, ligand, window=1.6, shape="tunnel", rings=5, padding=2, surface_keep=0.15),
            caption.animate.set_text("RIP A272 · Depth rings every 5 Å"),
            run_time=2,
        )
        self.play(self.camera.animate.orbit(0.25), run_time=2)
        self.play(Conceal(self.camera), Representation(protein, "cartoon"), run_time=1.5)
        self.play(caption.animate.set_text("2DRI · Cartoon"), run_time=0.5)
        self.wait(1)


# docs:end studio-cutaway


# docs:start readable-timing
class ReadableTiming(ProteinScene):
    renderer = "studio"
    look = StudioLook(lighting="soft", material="medium", effects="film")

    def construct(self):
        style = MolecularStyle("cartoon", {"color": "#8fe3c0"})
        protein = Protein.from_file(DATA / "1ubq.cif", chains="A").set_style(style).center()
        helix = protein.select(residue_range=(23, 34))
        self.camera.frame(protein, margin=1.12, screen_position=(0.64, 0.54))
        caption = Text("Independent timing", position=(0.05, 0.09), font_size=48)
        self.play(FadeIn(protein).during(2), Write(caption).during(1, delay=0.5))
        self.play(
            helix.animate.set_color("#ffc46b").show_atoms().during(2),
            (~helix).animate.set_opacity(0.25).during(1, delay=0.5),
            self.camera.animate.orbit(0.6).during(3),
        )
        self.play(
            (~helix).animate.set_opacity(1),
            caption.animate.set_text("One scene · Shared styles"),
            run_time=1.5,
        )
        self.wait(1)


# docs:end readable-timing


EXAMPLES = [
    (
        "studio-threading",
        CartoonThreading,
        "Cartoon threading",
        "An eased entrance using the actual cartoon geometry of 2DRI; an illustration, not a folding simulation.",
        7,
    ),
    (
        "studio-atoms",
        AtomThreading,
        "Ball-and-stick threading",
        "The same entrance with atoms and bonds, glossy materials, and subtle film effects.",
        7,
    ),
    (
        "studio-cutaway",
        BuriedLigand,
        "A buried-ligand cutaway",
        "A solvent-excluded surface opens a tunnel onto RIP A272 in 2DRI, with depth rings every 5 Å, then returns to cartoon.",
        5,
    ),
    (
        "readable-timing",
        ReadableTiming,
        "Independent animation timing",
        "A reusable style, a readable helix selection, and separate durations for styling, text, and camera motion. Studio film effects add grain, bloom, and halation.",
        4,
    ),
]
