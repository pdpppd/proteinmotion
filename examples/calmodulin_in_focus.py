"""Calmodulin in focus: a continuous EEVEE camera study of PDB 1CLL, chain A.

Coordinates stay fixed. Camera movement and lens focus are cinematic choices.
Hydrogen-bond candidates use inferred backbone amide H; the displayed lines
join donor N and acceptor O. The surface colors use deposited C-alpha B factors.

Requires the current ProteinMotion checkout and Blender 4.5+.
    python examples/calmodulin_in_focus.py --output calmodulin-in-focus.mp4
    proteinmotion still examples/calmodulin_in_focus.py CalmodulinInFocus \
        --renderer eevee --time 24 -o still.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

from proteinmotion import (
    ColorByProperty,
    Colorize,
    ColorLegend,
    ColorScale,
    EEVEEOptions,
    FadeIn,
    FadeOut,
    Focus,
    FocusPull,
    HydrogenBonds,
    Protein,
    ProteinScene,
    Region,
    Representation,
    ResidueValues,
    SetOpacity,
    Text,
    Write,
    smooth,
)

HERE = Path(__file__).resolve().parent
BLUE, GOLD, TEAL = "#759dbb", "#efb766", "#64d6cb"
INK, WHITE = "#94a6ba", "#edf1f6"


def outside(region):
    protein = region.protein
    return Region(protein, np.setdiff1d(np.arange(len(protein.topology.atoms)), region.atom_indices))


class CalmodulinInFocus(ProteinScene):
    def caption(self, title, detail, color=WHITE):
        return (
            Text(title, position=(0.06, 0.07), font_size=48, font="semibold", color=color),
            Text(detail, position=(0.06, 0.135), font_size=25, color=INK),
        )

    def replace_caption(self, title, detail, color=WHITE):
        previous = self._caption
        self._caption = self.caption(title, detail, color)
        # Clear the old caption before the new one appears, while the camera moves.
        return [
            *(FadeOut(obj, rate_func=lambda a: smooth(np.clip(a / 0.12, 0, 1))) for obj in previous),
            *(
                FadeIn(obj, rate_func=lambda a: smooth(np.clip((a - 0.16) / 0.14, 0, 1)))
                for obj in self._caption
            ),
        ]

    def fit(self, target, margin):
        return Focus(self.camera, target, margin=margin, aspect=self.width / self.height)

    def construct(self):
        p = Protein.from_file(HERE / "data/1cll.cif", chains="A")
        p.surface(kind="ses", resolution=0.5).ball_and_stick(atom_scale=0.32, bond_radius=0.10)
        p.cartoon(color=BLUE)
        # A stable pose puts the long molecular axis on a diagonal in the frame.
        ca = p.positions[[r.ca for r in p.topology.residues if r.ca >= 0]]
        _, _, axes = np.linalg.svd(ca - ca.mean(0), full_matrices=False)
        p.orientation = np.stack((axes[1], axes[0], axes[2]))
        if np.linalg.det(p.orientation) < 0:
            p.orientation[2] *= -1
        p.center().rotate(-0.52, axis=(0, 0, 1))
        helix = p.select(chain="A", residues=(5, 19))
        far_helix = p.select(chain="A", residues=(118, 128))
        backbone = p.select(chain="A", residues=(5, 19), atoms=["N", "CA", "C", "O"])
        helix.set_color(GOLD)
        self.camera.frame(p, margin=1.22, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi = 0.12, 0.06
        self.camera.set_focus(helix, fstop=5.6)
        self.protein = p
        self._caption = self.caption("Calmodulin", "1CLL · a study in depth and focus")
        identity = Text(
            "ProteinMotion / EEVEE", position=(0.94, 0.94), align="right", font_size=22, color=INK
        )

        # 00–08: a wide reveal and an unhurried orbit.
        self.play(FadeIn(p), Write(self._caption[0]), FadeIn(self._caption[1]), FadeIn(identity), run_time=2)
        self.play(self.camera.animate.orbit(theta=0.30, phi=-0.04).zoom(1.08), run_time=6)

        # 08–17: isolate one helix while keeping a translucent molecular context.
        self.play(
            *self.replace_caption("An alpha helix", "Chain A · residues 5–19", GOLD),
            self.fit(helix, 1.65),
            FocusPull(self.camera, helix, fstop=3.5),
            SetOpacity(outside(helix), 0.10),
            run_time=3,
        )
        note = helix.callout("Residues 5–19", position=(0.71, 0.60), font_size=30, color=GOLD)
        self.play(Write(note), self.camera.animate.orbit(theta=0.08), run_time=1.5)
        self.play(self.camera.animate.orbit(theta=0.28, phi=0.06), run_time=4.5)

        # 17–27: follow the backbone into an atomic view and its H-bond network.
        self.play(
            *self.replace_caption("Inside the helix", "Backbone N···O hydrogen-bond candidates"),
            FadeOut(note),
            Representation(p, "ball_and_stick"),
            Colorize(p.select(residues=(5, 19), atoms=["CA", "C"]), "#b4c8da"),
            Colorize(p.select(residues=(5, 19), atoms="N"), "#6997ef"),
            Colorize(p.select(residues=(5, 19), atoms="O"), "#ee7089"),
            SetOpacity(outside(backbone), 0.035),
            FocusPull(self.camera, backbone, fstop=12),
            self.fit(backbone, 1.05),
            run_time=3,
        )
        hb = HydrogenBonds(
            p,
            donors=p.select(chain="A", residues=(9, 19), atoms="N"),
            acceptors=p.select(chain="A", residues=(5, 15), atoms="O"),
            hydrogens="backbone",
        )
        network = hb.highlight(
            mode="3d",
            endpoints="donor_acceptor",
            show_distances=False,
            color=GOLD,
            radius=0.06,
            dash_count=5,
        )
        self.hbond_pairs = [
            {
                "donor": p.topology.atoms[q.a].resid,
                "acceptor": p.topology.atoms[q.b].resid,
                "distance_angstrom": float(q.distance),
                "angle_degrees": float(q.angle),
            }
            for q in hb.pairs
        ]
        self.play(Write(network), self.camera.animate.orbit(theta=0.10), run_time=2)
        self.play(
            self.camera.animate.orbit(theta=0.42, phi=-0.09),
            FocusPull(self.camera, p.select(chain="A", residues=(11, 14), atoms="CA"), fstop=10),
            run_time=5,
        )

        # 27–35: return through the same structure and establish two focus targets.
        self.play(
            *self.replace_caption("Moving the focus", "Gold: 5–19   /   Teal: 118–128"),
            FadeOut(network),
            Representation(p, "cartoon"),
            SetOpacity(p, 1),
            Colorize(helix, GOLD),
            Colorize(far_helix, TEAL),
            self.fit(p, 1.12),
            FocusPull(self.camera, helix, fstop=3.2),
            run_time=4,
        )
        self.play(
            SetOpacity(outside(helix | far_helix), 0.24),
            self.camera.animate.orbit(theta=0.30, phi=0.03).zoom(1.08),
            run_time=4,
        )

        # 35–45: hold framing nearly still so the focus transfer is visible.
        self.play(
            *self.replace_caption("Focus · residues 5–19", "The lens follows the selected Cα atoms", GOLD),
            FocusPull(self.camera, p.select(residues=(5, 19), atoms="CA"), fstop=1.4),
            self.camera.animate.orbit(theta=0.10),
            run_time=5,
        )
        self.play(
            *self.replace_caption(
                "Focus · residues 118–128", "A continuous focus pull to the second helix", TEAL
            ),
            FocusPull(self.camera, p.select(residues=(118, 128), atoms="CA"), fstop=1.4),
            self.camera.animate.orbit(theta=0.10),
            run_time=5,
        )

        # 45–55: surface geometry, measured residue colors, and a borderless legend.
        values = ResidueValues.b_factors(p)
        scale = ColorScale(0, 50, colors=("#468da7", "#74c9b7", "#f3ce83"))
        legend = ColorLegend(scale, title="Cα B factor", unit="Å²", position=(0.06, 0.82), size=(0.24, 0.11))
        self.play(
            *self.replace_caption("The molecular surface", "Color follows the deposited B factors"),
            SetOpacity(p, 1),
            Representation(p, "surface"),
            ColorByProperty(p, values, scale=scale),
            self.fit(p, 1.15),
            FocusPull(self.camera, far_helix, fstop=5.6),
            FadeIn(legend),
            run_time=3,
        )
        self.play(self.camera.animate.orbit(theta=0.56, phi=0.08), run_time=7)

        # 55–68: return to the backbone and finish with another gentle focus pull.
        self.play(
            *self.replace_caption("Back to the backbone", "One structure · continuous motion"),
            Representation(p, "ribbon"),
            Colorize(p, BLUE),
            FadeOut(legend),
            self.fit(p, 1.16),
            FocusPull(self.camera, helix, fstop=3.5),
            run_time=4,
        )
        self.play(
            *self.replace_caption("Calmodulin in focus", "Made with ProteinMotion and Blender EEVEE"),
            Colorize(helix, GOLD),
            Representation(p, "cartoon"),
            self.camera.animate.orbit(theta=0.42, phi=-0.04).zoom(1.08),
            FocusPull(self.camera, helix, fstop=2),
            run_time=6,
        )
        self.play(
            FadeOut(p),
            *(FadeOut(t) for t in self._caption),
            FadeOut(identity),
            self.camera.animate.orbit(theta=0.06),
            run_time=3,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "calmodulin-in-focus.mp4")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--supersampling", type=float, default=1.25)
    args = parser.parse_args()
    scene = CalmodulinInFocus(width=args.width, height=args.height, fps=args.fps, background="#080f19")
    report = scene.render(
        args.output,
        renderer="eevee",
        bitrate="22M",
        eevee=EEVEEOptions(samples=args.samples, supersampling=args.supersampling, max_blur=28),
    )
    report.update(
        duration=scene.duration, width=args.width, height=args.height, hbond_pairs=scene.hbond_pairs
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
