"""Buried ribose: a four-stage integration film for the Studio renderer.

Render from this checkout:
    .venv/bin/proteinmotion render examples/studio_buried_ligand.py BuriedRibose \
        --renderer studio --fps 60 --width 1920 --height 1080 --msaa 4 \
        -o output/studio-buried-ligand/buried-ribose.mp4

Structure: E. coli ribose-binding protein, PDB 2DRI, chain A; RIP A272.
Source: https://www.rcsb.org/structure/2DRI (1.60 A X-ray structure).
Binding-site comparison: https://pmc.ncbi.nlm.nih.gov/articles/PMC2630998/,
Figure 5. No coordinates are moved: threading is a visual entrance, not folding.
The surface is a voxel approximation of the solvent-excluded surface.
Waters are omitted. Eleven direct polar contacts are selected using
standard protein donor/acceptor identities and an N/O--O cutoff of 3.2 A.
These visualize the putative hydrogen-bond network. No hydrogens are deposited,
so D-H-A angles, protonation, and water-mediated bonds are not evaluated.
"""

from pathlib import Path

import numpy as np

from proteinmotion import (
    Conceal,
    Distance,
    FadeIn,
    FadeOut,
    Focus,
    Protein,
    ProteinScene,
    Region,
    Representation,
    Reveal,
    SetOpacity,
    ShowSideChains,
    Text,
    Thread,
    Write,
)
from proteinmotion.interactions import ACCEPTORS, DONORS

DATA = Path(__file__).resolve().parent / "data" / "2dri.cif"
INK, MINT, CREAM, MUTED, GOLD = "#0B1210", "#8FE3C0", "#F3EEE3", "#9EB3A8", "#FFC46B"


def ribose_contacts(protein):
    """Chemically plausible direct contacts; return actual atom indices/distances."""
    atoms, xyz = protein.topology.atoms, protein.positions
    ligand = protein.select(chain="A", resname="RIP", residues=272)
    contacts = []
    for i in ligand.atom_indices:
        a = atoms[i]
        if a.element != "O":
            continue
        for j, b in enumerate(atoms):
            if protein.topology.residue_categories[b.residue_index] != "polymer":
                continue
            # O5 is the ribose ring ether: acceptor only. O1--O4 are hydroxyls.
            donor = b.name in DONORS.get(b.resname, set()) or (b.name == "N" and b.resname != "PRO")
            acceptor = b.name in ACCEPTORS.get(b.resname, set()) or b.name in {"O", "OXT"}
            if not (donor or (acceptor and a.name != "O5")):
                continue
            distance = float(np.linalg.norm(xyz[i] - xyz[j]))
            if 1.5 < distance <= 3.2:
                contacts.append((int(i), j, distance))
    return contacts


class BuriedRibose(ProteinScene):
    renderer = "studio"

    def __init__(self, **kwargs):
        kwargs.setdefault("background", INK)
        super().__init__(**kwargs)

    def construct(self):
        aspect = self.width / self.height
        p = Protein.from_file(DATA, chains="A").cartoon(atom_scale=0.25, bond_radius=0.105).center()
        for code, color in {"H": MINT, "E": CREAM, "C": "#75A89B"}.items():
            region = p.select(polymer=True, secondary=code, required=False)
            if region is not None:
                region.set_color(color)
        ligand = p.select(chain="A", resname="RIP", residues=272)
        ligand.filter(element="C").set_color(GOLD)
        contacts = ribose_contacts(p)
        residues = sorted({p.topology.atoms[j].resid for _, j, _ in contacts})
        if len(contacts) != 11 or residues != [13, 89, 90, 141, 190, 215, 235]:
            raise ValueError(f"Unexpected 2DRI contact network: {len(contacts)} pairs, residues {residues}")
        pocket = p.select(chain="A", residues=residues)
        for element, color in {"N": "#638cff", "O": "#f06478"}.items():
            pocket.filter(element=element).set_color(color)
        site = pocket | ligand
        rest = ~site
        self.protein, self.ligand, self.contacts = p, ligand, contacts
        self.camera.theta, self.camera.phi = 0.55, 0.24
        self.camera.frame(p, margin=1.22, screen_position=(0.645, 0.5))
        self.camera.depth_cue = 0.32

        title = Text("Buried ribose", font_size=56, font="semibold", color=CREAM, position=(0.055, 0.055))
        subtitle = Text(
            "Ribose-binding protein\nE. coli · PDB 2DRI",
            font_size=25,
            color=MUTED,
            position=(0.058, 0.125),
            line_spacing=1.5,
        )
        footer = Text(
            "STUDIO RENDERER     ·     1.60 Å crystal structure",
            font_size=19,
            color=MUTED,
            position=(0.058, 0.945),
        )

        def caption(number, heading, detail):
            return (
                Text(
                    f"{number}   {heading}",
                    font_size=35,
                    font="semibold",
                    color=CREAM,
                    position=(0.058, 0.79),
                ),
                Text(detail, font_size=23, color=MUTED, position=(0.060, 0.85), line_spacing=1.5),
            )

        one = caption("01", "Trace the fold", "A glowing backbone\nthreads the protein into view")
        self.play(FadeIn(title), FadeIn(subtitle), FadeIn(footer), *[FadeIn(t) for t in one], run_time=0.8)
        self.play(
            Thread(
                p,
                self.camera,
                mode="wire",
                radius=0.27,
                head=2.0,
                glow=19,
                glow_color=MINT,
                glow_brightness=1.6,
                settle=0.23,
                seed=4,
            ),
            self.camera.animate.orbit(0.30, 0.02),
            run_time=6.2,
        )
        self.play(self.camera.animate.orbit(0.22, -0.04), run_time=1.8)

        two = caption("02", "Close the surface", "The bound ribose is buried\nbetween the two lobes")
        self.play(*[FadeOut(t) for t in one], run_time=0.4)
        self.play(Representation(p, "surface", grid_spacing=0.5), *[FadeIn(t) for t in two], run_time=2.0)
        self.play(self.camera.animate.orbit(0.28, 0.02), run_time=2.0)

        three = caption("03", "Open a cutaway", "Remove the front surface\nto reveal the buried sugar")
        self.play(
            *[FadeOut(t) for t in two],
            *[FadeIn(t) for t in three],
            Focus(self.camera, site, screen_position=(0.645, 0.5), margin=2.25, aspect=aspect),
            run_time=1.6,
        )
        self.play(
            Reveal(self.camera, ligand, window=1.55, shape="cone", softness=0.16, band=2.0), run_time=2.5
        )
        ribose_label = ligand.callout(
            "β-D-ribose", subtitle="RIP · A272", position=(0.065, 0.34), font_size=34, color=GOLD
        )
        self.play(
            Write(ribose_label),
            Focus(self.camera, site, screen_position=(0.645, 0.5), margin=1.85, aspect=aspect),
            run_time=1.8,
        )
        self.play(self.camera.animate.orbit(0.16, 0.03), run_time=2.2)

        four = caption("04", "Hydrogen bonds", "11 direct polar contacts\n7 binding-site residues")
        self.play(
            *[FadeOut(t) for t in three], FadeOut(ribose_label), Representation(p, "cartoon"), run_time=1.8
        )
        self.play(
            Conceal(self.camera),
            SetOpacity(rest, 0.14),
            ShowSideChains(pocket, residue_delay=0.10),
            Focus(self.camera, site, screen_position=(0.645, 0.5), margin=1.15, aspect=aspect),
            *[FadeIn(t) for t in four],
            run_time=1.8,
        )
        network = [
            Distance(
                Region(p, [i]),
                Region(p, [j]),
                mode="3d",
                color=GOLD,
                radius=0.065,
                dash_count=5,
                dash_ratio=0.57,
                show_distance=False,
            )
            for i, j, _ in contacts
        ]
        self.play(*[Write(line) for line in network], run_time=1.8)
        names = Text(
            "Asn13 · Asp89 · Arg90\nArg141 · Asn190\nAsp215 · Gln235",
            font_size=25,
            color=MUTED,
            position=(0.055, 0.36),
            line_spacing=1.6,
        )
        method = Text(
            "Gold dashes: N/O···O ≤ 3.2 Å\nH atoms absent; angles not tested",
            font_size=21,
            color=MUTED,
            position=(0.055, 0.57),
            line_spacing=1.5,
        )
        final_label = ligand.callout(
            "β-D-ribose", subtitle="RIP · A272", position=(0.82, 0.26), font_size=32, color=GOLD
        )
        self.play(FadeIn(names), FadeIn(method), FadeIn(final_label), run_time=0.8)
        self.play(self.camera.animate.orbit(0.25, -0.08), run_time=3.5)
        self.wait(1.5)


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "output" / "studio-buried-ligand"
    destination.mkdir(parents=True, exist_ok=True)
    BuriedRibose(width=1920, height=1080, fps=60).render(destination / "buried-ribose.mp4", renderer="studio")
