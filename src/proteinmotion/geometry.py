"""Compact static GPU metadata and deformation-stable backbone frames."""

import colorsys

import gemmi
import numpy as np

from .math3d import color, normalize

ELEMENT_COLORS = {
    "C": "#9db6cb",
    "N": "#638cff",
    "O": "#f06478",
    "S": "#f4d267",
    "P": "#ffad61",
    "H": "#e5eaf0",
    "Fe": "#d58b54",
    "Zn": "#afb7d9",
}
SS_COLORS = {"H": "#56d8c0", "E": "#f2ba67", "C": "#91a9ce"}


def residue_colors(protein):
    topo, scheme = protein.topology, protein.color_scheme
    chains = list(dict.fromkeys(r.chain for r in topo.residues))
    result = []
    for i, r in enumerate(topo.residues):
        if scheme == "secondary":
            c = color(SS_COLORS[r.secondary])
        elif scheme == "rainbow":
            c = colorsys.hsv_to_rgb(0.70 * (1 - i / max(1, len(topo.residues) - 1)), 0.58, 0.95)
        elif scheme == "chain":
            c = colorsys.hsv_to_rgb((chains.index(r.chain) * 0.618 + 0.42) % 1, 0.5, 0.93)
        else:
            c = color(scheme)
        result.append(c)
    return np.array(result, np.float32)


def atom_metadata(protein):
    if protein._metadata_override is not None:
        return protein._metadata_override
    data = np.zeros((len(protein.topology.atoms), 4), np.float32)
    for i, atom in enumerate(protein.topology.atoms):
        data[i, :3] = color(ELEMENT_COLORS.get(atom.element, "#c987cb"))
        data[i, 3] = max(float(gemmi.Element(atom.element).vdw_r), 1.0)
    return data


def segments(protein):
    """80-byte segment records: four CA indices + widths/thicknesses + endpoint colors."""
    topo = protein.topology
    palette = residue_colors(protein)
    rows = []
    for chain in topo.chains:
        cas = [topo.residues[i].ca for i in chain]
        width, thick = [], []
        for j, ri in enumerate(chain):
            ss = topo.residues[ri].secondary
            w, h = {"H": (1.15, 0.20), "E": (1.18, 0.16), "C": (0.24, 0.24)}[ss]
            if ss == "E" and (j == len(chain) - 1 or topo.residues[chain[j + 1]].secondary != "E"):
                w = 2.05
            width.append(w * protein._cartoon_scale[ri])
            thick.append(h * protein._cartoon_scale[ri])
        for i in range(len(chain) - 1):
            row = np.zeros(20, np.float32)
            row[:4].view(np.uint32)[:] = [
                cas[max(i - 1, 0)],
                cas[i],
                cas[i + 1],
                cas[min(i + 2, len(cas) - 1)],
            ]
            row[4:8] = [width[i], width[i + 1], thick[i], thick[i + 1]]
            row[8:11], row[12:15] = palette[chain[i]], palette[chain[i + 1]]
            # End cap flags. Surface endpoints collapse to close open ribbon ends.
            row[16:20] = [i == 0, i == len(chain) - 2, 0, 0]
            rows.append(row)
    return np.asarray(rows, np.float32).reshape(-1, 20)


def state_data(topology, xyz, reference_normals=None):
    """Compute frame guides once per keyframe, never once per rendered tween."""
    data = np.zeros((len(xyz), 8), np.float32)
    data[:, :3] = xyz
    for chain in topology.chains:
        ca = np.array([topology.residues[i].ca for i in chain])
        pts = xyz[ca]
        tangents = normalize(np.gradient(pts, axis=0))
        guides = []
        last = None
        for j, ri in enumerate(chain):
            tangent = tangents[j]
            oxygen = topology.residues[ri].oxygen
            guide = xyz[oxygen] - pts[j] if oxygen >= 0 else np.array([0.0, 1.0, 0.0])
            guide -= np.dot(guide, tangent) * tangent
            if np.linalg.norm(guide) < 1e-6:
                guide = np.cross(tangent, [1.0, 0.0, 0.0])
                if np.linalg.norm(guide) < 1e-6:
                    guide = np.cross(tangent, [0.0, 0.0, 1.0])
            guide = normalize(guide)
            if last is not None:
                transported = last - np.dot(last, tangent) * tangent
                transported = normalize(transported)
                if np.dot(guide, transported) < 0:
                    guide = -guide
                guide = normalize(0.65 * guide + 0.35 * transported)
            guides.append(guide)
            last = guide
        guides = np.array(guides)
        for _ in range(2):
            padded = np.pad(guides, ((1, 1), (0, 0)), mode="edge")
            guides = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
            guides -= np.sum(guides * tangents, axis=1)[:, None] * tangents
            guides = normalize(guides)
        if reference_normals is not None:
            # Choose a consistent sign for the whole continuous chain across frames.
            if np.sum(guides * reference_normals[ca]) < 0:
                guides *= -1
        data[ca, 4:7] = guides
    return data


def sweep_grid(steps=10, sides=12):
    vertices = np.array(
        [(i / steps, j / sides) for i in range(steps + 1) for j in range(sides + 1)], np.float32
    )
    indices = []
    for i in range(steps):
        for j in range(sides):
            a, b = i * (sides + 1) + j, (i + 1) * (sides + 1) + j
            indices.extend([a, b, a + 1, a + 1, b, b + 1])
    return vertices, np.asarray(indices, np.uint32)
