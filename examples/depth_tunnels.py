"""Tunnels that show how deep a selection lies in GroEL–GroES (PDB 1AON).

Reveal(shape="tunnel") drills a cylinder along the current view that then stays fixed
to the molecule, with a ring every 5 Å from the outer surface. Depths are measured along
the tunnel centerline, straight in from outside the barrel, to the first open space.
The film compares the shallow ADP pocket of subunit A with residues on the ring's inner
wall, shows the bore in a ghosted side view, flies down it, and carves the surface.
Atoms are hidden or ghosted for visibility only; none move.

    proteinmotion render examples/depth_tunnels.py DepthTunnels --fps 60 -o tunnels.mp4
"""

from pathlib import Path

import numpy as np

from proteinmotion import (
    Conceal,
    FadeIn,
    FadeOut,
    Focus,
    HideSideChains,
    Protein,
    ProteinScene,
    Region,
    Representation,
    Reveal,
    SetOpacity,
    ShowSideChains,
    Text,
    Unwrite,
    Write,
)

DATA = Path(__file__).parent / "data"
GOLD, TEAL, MUTED = "#f2ba67", "#50e0d0", "#a3b3c7"


def upright(p):
    """Rotate the complex so its long axis is vertical; every orbit is then a side view."""
    xyz = p.positions[[r.ca for r in p.topology.residues if r.ca >= 0]]
    axis = np.linalg.svd(xyz - xyz.mean(0), full_matrices=False)[2][0]
    axis = axis if axis[1] >= 0 else -axis
    turn = np.cross(axis, [0, 1, 0])
    if np.linalg.norm(turn) > 1e-6:
        p.rotate(float(np.arccos(np.clip(axis[1], -1, 1))), turn / np.linalg.norm(turn))
    return p.center()


def outside_view(p, region):
    """(theta, phi) looking at ``region`` from outside the barrel, along the radial direction."""
    m = p.model_matrix
    axis_point = (p.positions @ m[:3, :3].T + m[:3, 3]).mean(0)
    target = region.world_positions
    radial = (target.min(0) + target.max(0)) / 2 - axis_point
    radial[1] = 0  # The barrel axis is vertical after upright().
    radial /= np.linalg.norm(radial)
    return float(np.arctan2(radial[0], radial[2])), 0.0


def facing(camera, angles):
    """Nearest equivalent camera turn to (theta, phi)."""
    theta, phi = angles
    return (theta - camera.theta + np.pi) % (2 * np.pi) - np.pi, phi - camera.phi


class DepthTunnels(ProteinScene):
    """Tunnels fixed to the molecule show how deep a selection lies, with a ring every 5 Å."""

    def construct(self):
        aspect = self.width / self.height
        p = Protein.from_file(DATA / "1aon.cif").surface(kind="ses", resolution=1.2).cartoon(color="chain")
        upright(p)
        ligand = p.select(chain="A", resname=["ADP", "MG"])
        pocket = p.select(within=4.0, of=ligand)
        site = pocket | ligand
        carbons = [i for i in ligand.atom_indices if p.topology.atoms[i].element == "C"]
        Region(p, np.array(carbons)).set_color(GOLD)
        # Residues on the inner wall of the ring, facing the central chamber.
        wall = p.select(chain="A", residues=[62, 63, 64, 524])
        self.add(p)
        self.camera.frame(p, margin=0.95, aspect=aspect)
        self.camera.theta, self.camera.phi = 0.9, 0.12
        self.camera.depth_cue = 0.25

        title = Text("How deep is it?", font_size=56, font="semibold", position=(0.06, 0.07))
        subtitle = Text(
            "GroEL–GroES · PDB 1AON · tunnels with a ring every 5 Å",
            font_size=26,
            color=MUTED,
            position=(0.062, 0.14),
        )

        def caption(text):
            return Text(text, font_size=28, color="#edf3fc", position=(0.06, 0.88))

        self.play(
            Write(title, stroke_width=1.6), FadeIn(subtitle), self.camera.animate.orbit(0.35), run_time=2.5
        )
        self.play(self.camera.animate.orbit(0.4), run_time=1.5)

        # 1. The ADP pocket of subunit A, seen from outside the ring.
        note = caption("Seen from outside the ring, the ADP pocket of subunit A is hidden")
        self.play(
            self.camera.animate.orbit(*facing(self.camera, outside_view(p, site))).depth_cue(0.7),
            FadeIn(note),
            run_time=2.5,
        )
        self.play(Focus(self.camera, site, margin=1.6, aspect=aspect), run_time=2)
        self.play(FadeOut(note), run_time=0.4)
        note = caption("Reveal drills a tunnel into the molecule; each ring is 5 Å deeper")
        self.play(Reveal(self.camera, site, window=0.85, shape="tunnel"), FadeIn(note), run_time=2.2)
        depth = self.camera.cutaway_geometry()[3]
        label = ligand.callout(
            "ADP · Mg²⁺",
            subtitle=f"{depth:.0f} Å beneath the outer surface",
            position=(0.68, 0.24),
            font_size=38,
            color=GOLD,
        )
        self.play(ShowSideChains(pocket, residue_delay=0.06), Write(label), run_time=2)
        # Small moves near the tunnel axis: the rings slide past each other, like looking into a well.
        self.play(self.camera.animate.orbit(0.22, 0.12), run_time=2)
        self.play(self.camera.animate.orbit(-0.4, -0.2), run_time=2.5)
        self.play(FadeOut(note), Unwrite(label), Conceal(self.camera), HideSideChains(pocket), run_time=1.5)

        # 2. Residues on the ring's inner wall: the same view from outside, twice as deep.
        note = caption("Residues lining the central chamber lie much deeper")
        self.play(
            self.camera.animate.orbit(*facing(self.camera, outside_view(p, wall))).depth_cue(0.7),
            FadeIn(note),
            run_time=2,
        )
        self.play(Focus(self.camera, wall, margin=2.2, aspect=aspect), run_time=2)
        self.play(
            Reveal(self.camera, wall, window=1.25, shape="tunnel"),
            ShowSideChains(wall, residue_delay=0.1),
            run_time=2.2,
        )
        depth = self.camera.cutaway_geometry()[3]
        label = wall.callout(
            "Leu62–Asp64 · Leu524",
            subtitle=f"{depth:.0f} Å beneath the outer surface",
            position=(0.68, 0.24),
            font_size=38,
            color=TEAL,
        )
        self.play(Write(label), FadeOut(note), run_time=1.2)
        self.play(self.camera.animate.orbit(0.2, 0.12), run_time=2)
        self.play(self.camera.animate.orbit(-0.4, -0.22), run_time=2.5)
        self.play(Unwrite(label), run_time=0.6)

        # 3. X-ray side view: ghost everything else to show the whole bore inside the complex.
        note = caption("Side view with the rest of the complex ghosted: the bore runs 5 Å per ring")
        rest = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), (wall | ligand).atom_indices))
        glow = wall.highlight(style="sphere", color=TEAL, opacity=0.35, padding=3.0)
        self.play(
            SetOpacity(rest, 0.18),
            Focus(self.camera, p, margin=0.75, aspect=aspect),
            FadeIn(glow),
            FadeIn(note),
            run_time=3,
        )
        self.play(self.camera.animate.orbit(1.35, 0.0).depth_cue(0.3), run_time=3)
        self.play(self.camera.animate.orbit(0.35, 0.1), run_time=4)
        self.play(FadeOut(note), FadeOut(glow), SetOpacity(rest, 1.0), run_time=1.5)

        # 4. Fly down the tunnel.
        note = caption("Flying down the tunnel, past one ring every 5 Å")
        self.play(
            self.camera.animate.orbit(*facing(self.camera, outside_view(p, wall))).depth_cue(0.75),
            FadeIn(note),
            run_time=2.5,
        )
        self.play(Focus(self.camera, wall, margin=6.0, aspect=aspect), run_time=2)
        self.play(self.camera.animate.zoom(4.5), run_time=5)
        self.play(FadeOut(note), run_time=0.4)

        # 5. The same tunnel through the molecular surface.
        note = caption("The same tunnel through the molecular surface")
        self.play(Representation(p, "surface"), self.camera.animate.zoom(0.35), FadeIn(note), run_time=2.5)
        self.play(self.camera.animate.orbit(0.7, 0.28), run_time=5)
        self.play(FadeOut(note), Conceal(self.camera), run_time=2)
        self.play(Unwrite(title), FadeOut(subtitle), self.camera.animate.orbit(0.3), run_time=1.5)
        self.wait(0.4)
