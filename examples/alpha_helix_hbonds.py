"""A controlled alpha-helix hydrogen-bond test, not a deposited structure or MD.

Backbone-only ideal geometry: phi=-57°, psi=-47°, omega=180°; explicit amide H.
The detector is given no sequence-offset rule. Its output must recover i -> i+4.
"""

import numpy as np

from proteinmotion import (
    Colorize,
    Distance,
    FadeIn,
    FadeOut,
    Focus,
    HydrogenBonds,
    Protein,
    ProteinScene,
    Region,
    SetOpacity,
    Text,
    Write,
)
from proteinmotion.structure import Atom, Residue, Topology


def unit(v):
    return v / np.linalg.norm(v)


def place(a, b, c, length, angle, torsion):
    u = unit(c - b)
    normal = unit(np.cross(b - a, u))
    side = np.cross(normal, u)
    theta, tau = np.deg2rad([angle, torsion])
    return c + length * (-np.cos(theta) * u + np.sin(theta) * (np.cos(tau) * side + np.sin(tau) * normal))


def ideal_helix(n=16, phi=-57, psi=-47):
    if not isinstance(n, int) or n < 6:
        raise ValueError("Use at least six residues for this diagnostic helix")
    # Rounded standard peptide lengths (Å) and angles (degrees). Oxygen and H
    # complete trigonal peptide planes; side chains and terminal caps are omitted.
    bb = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.458, 0.0, 0.0]),
        np.array([1.458 + 1.525 * np.cos(np.deg2rad(68.8)), 1.525 * np.sin(np.deg2rad(68.8)), 0.0]),
    ]
    for j in range(n):
        bb.append(place(*bb[-3:], 1.329, 116.2, psi))
        bb.append(place(*bb[-3:], 1.458, 121.7, 180))
        bb.append(place(*bb[-3:], 1.525, 111.2, phi))
    atoms = []
    xyz = []
    res = []
    bonds = []
    for j in range(n):
        nn, ca, c = bb[j * 3 : j * 3 + 3]
        nn1 = bb[j * 3 + 3]
        o = c - unit(unit(ca - c) + unit(nn1 - c)) * 1.231
        names = ["N", "CA", "C", "O"]
        positions = [nn, ca, c, o]
        if j:
            h = nn - unit(unit(bb[j * 3 - 1] - nn) + unit(ca - nn)) * 1.01
            names += ["H"]
            positions += [h]
        start = len(atoms)
        atoms.extend(Atom("A", j + 1, "", "ALA", name, "C" if name == "CA" else name, j) for name in names)
        xyz.extend(positions)
        res.append(Residue("A", j + 1, "", "ALA", start + 1, start + 3, "H"))
        bonds.extend([(start, start + 1), (start + 1, start + 2), (start + 2, start + 3)])
        if j:
            bonds.extend([(res[j - 1].ca + 1, start), (start, start + 4)])
    xyz = np.array(xyz)
    ca = xyz[[r.ca for r in res]]
    _, _, basis = np.linalg.svd(ca - ca.mean(0))
    y = basis[0]
    if np.dot(y, ca[-1] - ca[0]) < 0:
        y = -y
    x = unit((ca[0] - ca.mean(0)) - np.dot(ca[0] - ca.mean(0), y) * y)
    z = np.cross(x, y)
    xyz = (xyz - ca.mean(0)) @ np.array([x, y, z]).T
    return Protein(
        Topology(tuple(atoms), tuple(res), np.array(bonds, np.uint32), (np.arange(n, dtype=np.uint32),)), xyz
    )


def analyze_helix(p):
    hb = HydrogenBonds(p, hydrogens="explicit", max_distance=3.5, min_angle=150)
    detected = {(p.topology.atoms[r.b].resid, p.topology.atoms[r.a].resid) for r in hb.pairs}
    expected = {(i, i + 4) for i in range(1, len(p.topology.residues) - 3)}
    if detected != expected:
        raise AssertionError(
            f"Unexpected helix network: missing={expected - detected}, extra={detected - expected}"
        )
    return hb


class AlphaHelixHBonds(ProteinScene):
    def construct(self):
        p = ideal_helix().ball_and_stick(atom_scale=0.25, bond_radius=0.09)
        hb = analyze_helix(p)
        self.add(p)
        self.camera.frame(p, margin=1.17, aspect=self.width / self.height)
        self.camera.theta, self.camera.phi, self.camera.depth_cue = 0.45, 0.08, 0.12
        title = Text("Hydrogen bonds in an α helix", font_size=57, font="semibold", position=(0.065, 0.065))
        subtitle = Text(
            "Carbonyl O(i) ··· H–N(i + 4)", font_size=30, color="#afbed0", position=(0.065, 0.145)
        )
        footer = Text(
            "Idealized backbone · 16 residues · explicit amide H · φ = −57° / ψ = −47°",
            font_size=21,
            color="#74869e",
        ).to_corner("DL", buff=0.065)
        legend = Text(
            "Oxygen   red\nNitrogen   blue\nHydrogen   white\nCarbon   gray",
            font_size=25,
            color="#afbed0",
            position=(0.76, 0.36),
            line_spacing=1.7,
        )
        self.play(Write(title), FadeIn(subtitle), FadeIn(footer), FadeIn(legend), run_time=1.3)
        self.wait(0.5)
        network = hb.highlight(
            mode="3d",
            endpoints="hydrogen_acceptor",
            show_distances=False,
            color="#f2ba67",
            radius=0.055,
            dash_count=5,
            dash_ratio=0.65,
        )
        count = Text(
            "12 expected\n12 detected", font_size=42, font="semibold", position=(0.065, 0.35), color="#f2ba67"
        )
        criteria = Text(
            "N···O ≤ 3.5 Å\nN–H···O ≥ 150°\n\nEvery pair is i → i + 4.",
            font_size=25,
            color="#afbed0",
            position=(0.065, 0.54),
            line_spacing=1.55,
        )
        self.play(Write(network), Write(count), FadeIn(criteria), run_time=2)
        self.play(self.camera.animate.orbit(0.9), run_time=4)
        self.wait(0.6)
        record = next(r for r in hb.pairs if p.topology.atoms[r.b].resid == 5)
        turn = p.select(residues=(5, 9))
        other = Region(p, np.setdiff1d(np.arange(len(p.topology.atoms)), turn.atom_indices))
        oxygen = Region(p, [record.b])
        nitrogen = Region(p, [record.a])
        hydrogen = Region(p, [record.hydrogen])
        bond = Distance(
            hydrogen,
            oxygen,
            mode="3d",
            show_distance=False,
            color="#f2ba67",
            radius=0.07,
            dash_count=5,
            dash_ratio=0.65,
        )
        self.play(
            FadeOut(network),
            FadeOut(count),
            FadeOut(criteria),
            FadeOut(legend),
            SetOpacity(other, 0.06),
            Focus(self.camera, turn, margin=1.65, aspect=self.width / self.height),
            run_time=1.6,
        )
        self.play(Colorize(p.select(residues=[5, 9], atoms="CA"), "#50e0d0"), Write(bond), run_time=0.8)
        o_label = oxygen.callout(
            "O · residue 5",
            subtitle="Carbonyl acceptor · i",
            position=(0.72, 0.65),
            color="#f06478",
            font_size=32,
        )
        nh_label = (nitrogen | hydrogen).callout(
            "N–H · residue 9",
            subtitle="Amide donor · i + 4",
            position=(0.72, 0.32),
            color="#638cff",
            font_size=32,
        )
        values = Text(
            f"H···O   {bond.distance:.2f} Å\nN···O   {record.distance:.2f} Å\nN–H···O   {record.angle:.1f}°",
            font_size=30,
            color="#f2ba67",
            position=(0.065, 0.41),
            line_spacing=1.8,
        )
        explanation = Text(
            "Solid N–H: covalent bond\nGold H···O: hydrogen bond",
            font_size=24,
            color="#afbed0",
            position=(0.065, 0.69),
            line_spacing=1.5,
        )
        self.play(Write(o_label), Write(nh_label), Write(values), FadeIn(explanation), run_time=1.8)
        self.play(self.camera.animate.orbit(0.25), run_time=3)
        self.wait(1.2)
        self.play(
            FadeOut(o_label),
            FadeOut(nh_label),
            FadeOut(values),
            FadeOut(explanation),
            FadeOut(bond),
            SetOpacity(other, 1),
            Colorize(p.select(residues=[5, 9], atoms="CA"), None),
            Focus(self.camera, p, margin=1.17, aspect=self.width / self.height),
            run_time=1.6,
        )
        self.play(FadeIn(network), FadeIn(count), FadeIn(criteria), FadeIn(legend), run_time=0.8)
        self.play(self.camera.animate.orbit(0.7), run_time=3)
        self.wait(0.8)


if __name__ == "__main__":
    AlphaHelixHBonds(width=1920, height=1080, fps=60).render("alpha-helix-hbonds.mp4")
