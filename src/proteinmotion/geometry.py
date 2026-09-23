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
    "Ca": "#8fd6a0",
    "Mg": "#a3e08a",
    "Mn": "#c49ae8",
    "Na": "#b59cf0",
    "K": "#9c8cf0",
    "Cl": "#8be0c9",
    "Cd": "#f0c987",
}
# Ligand carbons stand apart from the cartoon palette; other ligand atoms use element colors.
LIGAND_CARBON = "#a6d96a"
# Single-atom ions are shown as larger spheres than the atoms of a ball-and-stick model.
ION_SCALE = 1.3
SS_COLORS = {"H": "#56d8c0", "E": "#f2ba67", "C": "#91a9ce"}
BASE_COLORS = {
    "A": "#72cfb3",
    "C": "#72a7ed",
    "G": "#edbf68",
    "T": "#ed8193",
    "U": "#b399e7",
    "I": "#cfaa78",
    "N": "#91a9b8",
}


def residue_colors(protein):
    topo, scheme = protein.topology, protein.color_scheme
    chains = list(dict.fromkeys(r.chain for r in topo.residues))
    result = []
    for i, r in enumerate(topo.residues):
        if scheme in ("secondary", "base", "element"):
            c = color(BASE_COLORS.get(r.base, BASE_COLORS["N"]) if r.is_nucleic else SS_COLORS[r.secondary])
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
    topology = protein.topology
    categories = topology.residue_categories
    data = np.zeros((len(topology.atoms), 4), np.float32)
    for i, atom in enumerate(topology.atoms):
        data[i, :3] = color(ELEMENT_COLORS.get(atom.element, "#c987cb"))
        data[i, 3] = max(float(gemmi.Element(atom.element).vdw_r), 1.0)
        if categories[atom.residue_index] == "ion":
            data[i, 3] *= ION_SCALE
    if protein.color_scheme != "element":
        colors = residue_colors(protein)[[a.residue_index for a in topology.atoms]]
        if protein.color_scheme in ("secondary", "base"):
            # Structure palettes have no meaning for ligands and ions: color them by element.
            detail = topology.untraced_atoms
            colors[detail] = data[detail, :3]
            carbon = detail & np.array([a.element == "C" for a in topology.atoms], bool)
            colors[carbon] = color(LIGAND_CARBON)
        data[:, :3] = colors
    return data


def detail_colors(protein):
    """Atom colors over a cartoon: element colors, with carbons in the structure palette."""
    meta = atom_metadata(protein)
    if protein._metadata_override is not None or protein.color_scheme == "element":
        return meta[:, :3].copy()
    atoms = protein.topology.atoms
    result = np.array([color(ELEMENT_COLORS.get(a.element, "#c987cb")) for a in atoms], np.float32)
    carbon = np.array([a.element == "C" for a in atoms], bool)
    result[carbon] = meta[carbon, :3]
    return result


def atom_colors(protein):
    """Current sphere/bond base colors, blending detail colors into the ball-and-stick palette."""
    ball = float(protein.representation[2])
    return (1 - ball) * detail_colors(protein) + ball * atom_metadata(protein)[:, :3]


def packed_metadata(protein):
    """GPU atom records: ball-and-stick and detail RGBA8 colors (as raw bits) and radius."""

    def pack(rgb):
        c = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint32)
        return (c[:, 0] | c[:, 1] << 8 | c[:, 2] << 16 | np.uint32(255) << 24).view(np.float32)

    meta = atom_metadata(protein)
    data = np.zeros_like(meta)
    data[:, 0], data[:, 1], data[:, 3] = pack(meta[:, :3]), pack(detail_colors(protein)), meta[:, 3]
    return data


def segments(protein):
    """80-byte segment records: four CA indices + widths/thicknesses + endpoint colors."""
    topo = protein.topology
    palette = residue_colors(protein)
    rows = []
    for chain in topo.chains:
        cas = [topo.residues[i].trace_atom for i in chain]
        width, thick = [], []
        for j, ri in enumerate(chain):
            ss = topo.residues[ri].secondary
            w, h = {"H": (1.15, 0.20), "E": (1.18, 0.16), "C": (0.24, 0.24)}[ss]
            if topo.residues[ri].is_nucleic:
                w = h = protein.backbone_radius
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
        ca = np.array([topology.residues[i].trace_atom for i in chain])
        pts = xyz[ca]
        tangents = normalize(np.gradient(pts, axis=0))
        guides = []
        last = None
        for j, ri in enumerate(chain):
            tangent = tangents[j]
            oxygen = topology.residues[ri].guide_atom
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
