"""Compact static GPU metadata and deformation-stable backbone frames.

The native renderer (renderer.py with shaders/molecule.wgsl), the Studio renderer
built on it, and the EEVEE exporter (_eevee_geometry.py) draw a molecule from a few
packed float32 arrays made here:

- ``state_data``: one 8-float row per atom for each coordinate keyframe. It holds the
  position and, on trace atoms, the cartoon's width axis and how firm its twist is.
- ``packed_metadata``: one 4-float row per atom with two RGBA8 colors and a radius.
- ``segments`` and ``studio_cartoon_segments``: one 20-float (80-byte) row per step
  between consecutive residues of a chain. The ribbon vertex shader sweeps each row
  into a short tube along a Catmull–Rom curve through the trace atoms.
- ``sweep_grid``: the (along, around) vertex grid that each segment instance sweeps.

Width axes are computed once per keyframe on the CPU; the shader blends the two
keyframes, so the frames must move continuously when the atoms do.

The color schemes live here as well (``residue_colors``, ``atom_metadata``,
``detail_colors``). Plots, threads, surfaces and nucleotide bases call them, so a
residue has the same color in every view.
"""

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
# Elements missing from ELEMENT_COLORS draw in this mauve.
_OTHER_ELEMENT_COLOR = "#c987cb"
# Ligand carbons stand apart from the cartoon palette; other ligand atoms use element colors.
LIGAND_CARBON = "#a6d96a"
# Single-atom ions are shown as larger spheres than the atoms of a ball-and-stick model.
ION_SCALE = 1.3
# Helix (H), strand (E) and coil (C), the three-state DSSP codes.
SS_COLORS = {"H": "#56d8c0", "E": "#f2ba67", "C": "#91a9ce"}
# One-letter nucleotide bases; I is inosine and N any other or unknown base.
BASE_COLORS = {
    "A": "#72cfb3",
    "C": "#72a7ed",
    "G": "#edbf68",
    "T": "#ed8193",
    "U": "#b399e7",
    "I": "#cfaa78",
    "N": "#91a9b8",
}


# Coil, helix and strand cartoon cross-sections (width, thickness) in Å, in the
# order of Protein._secondary weights. Both are semi-axes: the shader sweeps an
# ellipse of half-width ``width`` and half-thickness ``thickness``.
CARTOON_PROFILE = np.array([[0.24, 0.24], [1.15, 0.20], [1.18, 0.16]])
# The same for Studio's classic cartoons, which sweep superellipses (studio_cartoon.wgsl).
STUDIO_PROFILE = np.array([[0.18, 0.18], [1.25, 0.16], [1.20, 0.12]])
# Half-width in Å at the base of a native sheet arrowhead, on the last residue of a strand.
SHEET_ARROW = 2.05


def secondary_weights(protein):
    """Coil/helix/strand weights per residue, following the topology if they are stale."""
    weights = getattr(protein, "_secondary", None)
    if weights is None or len(weights) != len(protein.topology.residues):
        from .secondary import weights as from_codes

        weights = from_codes("".join(r.secondary for r in protein.topology.residues))
    return np.asarray(weights, np.float32)


class SchemeBlend(str):
    """A color scheme part of the way from one named scheme to another.

    Its text includes the fraction, so every color cache keyed on the scheme
    name refreshes as the blend advances.
    """

    def __new__(cls, start, end, fraction):
        blend = super().__new__(cls, f"{start}>{end}@{fraction:.4f}")
        blend.start, blend.end, blend.fraction = str(start), str(end), float(fraction)
        return blend


def residue_colors(protein):
    """RGB in [0, 1] per residue, shape (residues, 3), for the protein's color scheme.

    Cartoons, ribbons, plots and threading wires color residues with these values.
    """
    scheme = protein.color_scheme
    if isinstance(scheme, SchemeBlend):
        t = scheme.fraction
        start = _scheme_colors(protein, scheme.start)
        return ((1 - t) * start + t * _scheme_colors(protein, scheme.end)).astype(np.float32)
    return _scheme_colors(protein, scheme)


def _scheme_colors(protein, scheme):
    """Per-residue RGB for one named scheme, or for a single color given as the scheme."""
    topo = protein.topology
    if scheme in ("secondary", "base", "element"):
        # Residue colors follow the structure; "element" colors only atoms by element
        # (see atom_metadata), so its cartoon keeps the secondary-structure palette.
        palette = np.array([color(SS_COLORS[k]) for k in "CHE"], np.float32)
        # Weights blend the palette, so a residue changing structure changes color smoothly.
        result = secondary_weights(protein) @ palette
        for i, residue in enumerate(topo.residues):
            if residue.is_nucleic:
                result[i] = color(BASE_COLORS.get(residue.base, BASE_COLORS["N"]))
        return result.astype(np.float32)
    if scheme == "hydropathy":
        from .properties import ColorScale, ResidueValues

        return ColorScale.hydropathy().map(ResidueValues.hydropathy(protein).values)
    chains = list(dict.fromkeys(r.chain for r in topo.residues))
    result = []
    for i, residue in enumerate(topo.residues):
        if scheme == "rainbow":
            # Hue runs from blue (0.70) at the first residue to red (0) at the last.
            rgb = colorsys.hsv_to_rgb(0.70 * (1 - i / max(1, len(topo.residues) - 1)), 0.58, 0.95)
        elif scheme == "chain":
            # Steps of the golden-ratio conjugate (0.618) keep neighbouring chain hues far apart.
            rgb = colorsys.hsv_to_rgb((chains.index(residue.chain) * 0.618 + 0.42) % 1, 0.5, 0.93)
        else:
            rgb = color(scheme)
        result.append(rgb)
    return np.array(result, np.float32).reshape(-1, 3)


def _element_data(topology):
    """Element colors and display radii per atom, computed once per topology.

    Columns 0:3 are RGB and column 3 the sphere radius in Å: the Van der Waals radius,
    at least 1 Å, enlarged by ION_SCALE for single-atom ions. The array is read-only.
    """
    # Cached on the topology, which copies of a protein share.
    cached = topology.__dict__.get("_element_data")
    if cached is None:
        categories = topology.residue_categories
        cached = np.zeros((len(topology.atoms), 4), np.float32)
        for i, atom in enumerate(topology.atoms):
            cached[i, :3] = color(ELEMENT_COLORS.get(atom.element, _OTHER_ELEMENT_COLOR))
            cached[i, 3] = max(float(gemmi.Element(atom.element).vdw_r), 1.0)
            if categories[atom.residue_index] == "ion":
                cached[i, 3] *= ION_SCALE
        cached.flags.writeable = False
        topology.__dict__["_element_data"] = cached
    return cached


def atom_metadata(protein):
    """Ball-and-stick color and radius per atom: (atoms, 4) float32, RGB then radius in Å.

    ``protein._metadata_override`` replaces the whole array when set; Thread wires use
    it to give each wire point its own color and radius.
    """
    if protein._metadata_override is not None:
        return protein._metadata_override
    topology = protein.topology
    data = _element_data(topology).copy()
    scheme = protein.color_scheme
    if isinstance(scheme, SchemeBlend):
        t = scheme.fraction
        data[:, :3] = (1 - t) * _scheme_atom_colors(protein, scheme.start, data) + t * _scheme_atom_colors(
            protein, scheme.end, data
        )
    elif scheme != "element":
        data[:, :3] = _scheme_atom_colors(protein, scheme, data)
    return data


def _scheme_atom_colors(protein, scheme, elements):
    """Each atom takes its residue's scheme color; ``elements`` is the _element_data array."""
    topology = protein.topology
    colors = _scheme_colors(protein, scheme)[[a.residue_index for a in topology.atoms]]
    if scheme in ("secondary", "base"):
        # Structure palettes have no meaning for ligands and ions: color them by element.
        detail = topology.untraced_atoms
        colors[detail] = elements[detail, :3]
        carbon = detail & np.array([a.element == "C" for a in topology.atoms], bool)
        colors[carbon] = color(LIGAND_CARBON)
    return colors


def detail_colors(protein):
    """Atom colors over a cartoon: element colors, with carbons in the structure palette."""
    meta = atom_metadata(protein)
    if protein._metadata_override is not None or protein.color_scheme == "element":
        return meta[:, :3].copy()
    topology = protein.topology
    result = _element_data(topology)[:, :3].copy()
    carbon = topology.__dict__.get("_carbon_atoms")
    if carbon is None:
        carbon = np.array([a.element == "C" for a in topology.atoms], bool)
        topology.__dict__["_carbon_atoms"] = carbon
    result[carbon] = meta[carbon, :3]
    return result


def atom_colors(protein):
    """Current sphere/bond base colors, blending detail colors into the ball-and-stick palette."""
    ball = float(protein.representation[2])
    return (1 - ball) * detail_colors(protein) + ball * atom_metadata(protein)[:, :3]


def packed_metadata(protein):
    """GPU atom records: ball-and-stick and detail RGBA8 colors (as raw bits) and radius.

    One row of four float32 per atom, read by the shader as ``atoms[i]``:

    - column 0: ball-and-stick color, RGBA8 bits (shader: ``unpack4x8unorm(bitcast<u32>(x))``)
    - column 1: detail color for atoms drawn over a cartoon, packed the same way
    - column 2: unused, zero
    - column 3: sphere radius in Å, before ``atom_scale``
    """

    def pack(rgb):
        # Red in the lowest byte and opaque alpha in the highest, as unpack4x8unorm expects.
        # The float32 view only carries the bits; many of these values are NaN patterns,
        # so these columns must never go through float arithmetic.
        c = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint32)
        return (c[:, 0] | c[:, 1] << 8 | c[:, 2] << 16 | np.uint32(255) << 24).view(np.float32)

    meta = atom_metadata(protein)
    data = np.zeros_like(meta)
    data[:, 0], data[:, 1], data[:, 3] = pack(meta[:, :3]), pack(detail_colors(protein)), meta[:, 3]
    return data


def segments(protein):
    """80-byte segment records: four CA indices + widths/thicknesses + endpoint colors.

    Cross-sections blend the coil, helix and strand profiles by the residue's
    secondary-structure weights; the last residue of a strand widens into an arrow.
    """
    # One row of 20 float32 per pair of consecutive residues i, i+1 in a chain, chain
    # after chain. The columns match ``struct Segment`` in molecule.wgsl:
    #   0:4    atoms    trace-atom indices of residues i-1, i, i+1, i+2, stored as uint32 bits.
    #                   These are the Catmull–Rom control points, repeated at chain ends.
    #   4:8    shape    half-width at i, half-width at i+1, half-thickness at i, at i+1 (Å)
    #   8:11   color_a  RGB of residue i; column 11 is 0 here (Studio stores a code there)
    #   12:15  color_b  RGB of residue i+1; column 15 likewise
    #   16:20  caps     first segment of the chain, last segment, arrow weight, unused
    topo = protein.topology
    palette = residue_colors(protein)
    secondary = secondary_weights(protein)
    rows = []
    for chain in topo.chains:
        cas = [topo.residues[i].trace_atom for i in chain]
        weights = secondary[chain]
        width, thick = (weights @ CARTOON_PROFILE).T
        strand = weights[:, 2]
        # 1 on the last residue of a strand (the next one is not strand, or the chain ends).
        arrow = strand * (1 - np.r_[strand[1:], 0.0])
        width = width + arrow * (SHEET_ARROW - width)
        # Nucleotides draw as a round backbone tube.
        nucleic = np.array([topo.residues[ri].is_nucleic for ri in chain])
        width[nucleic] = thick[nucleic] = protein.backbone_radius
        # Per-residue cartoon scale factors, such as from color_by(thickness=...).
        scale = protein._cartoon_scale[chain]
        width, thick = width * scale, thick * scale
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
            # End cap flags (surface endpoints collapse to close open ribbon ends), and the
            # arrow weight of a segment leaving a strand, whose arrowhead keeps the strand's axis.
            row[16:20] = [i == 0, i == len(chain) - 2, arrow[i], 0]
            rows.append(row)
    return np.asarray(rows, np.float32).reshape(-1, 20)


def studio_cartoon_segments(protein):
    """Classic cartoon profile in the existing 80-byte segment layout.

    Color alpha slots carry secondary-structure codes (coil=0, helix=1,
    strand=2, nucleotide=-1); caps.z marks the final span of a sheet arrow.
    Native/legacy metadata and nucleotide dimensions are left unchanged.
    """
    rows = segments(protein)
    topo = protein.topology
    secondary = secondary_weights(protein)
    # Codes are continuous (helix weight + 2 × strand weight) so shapes can blend.
    codes = secondary[:, 1] + 2 * secondary[:, 2]
    dims = secondary @ STUDIO_PROFILE
    # Rows come in the same chain-by-chain order as segments() wrote them.
    offset = 0
    for chain in topo.chains:
        for j in range(len(chain) - 1):
            row = rows[offset]
            offset += 1
            # Endpoint 0 is residue j (columns 4, 6 and 11), endpoint 1 is residue j+1 (5, 7, 15).
            for endpoint, ri in enumerate(chain[j : j + 2]):
                if topo.residues[ri].is_nucleic:
                    # -1 marks a nucleotide, which the Studio shader draws as the native tube.
                    row[11 + 4 * endpoint] = -1
                    continue
                row[11 + 4 * endpoint] = codes[ri]
                scale = protein._cartoon_scale[ri]
                # Replaces the native widths, so the native SHEET_ARROW widening is dropped
                # here; the Studio shader shapes its own arrowhead from caps.z.
                row[4 + endpoint], row[6 + endpoint] = dims[ri] * scale
            before, after = chain[j], chain[j + 1]
            if not topo.residues[before].is_nucleic:
                # The arrow spans the step that leaves a strand. A strand that runs to the
                # end of the chain has no such step, so its last step carries the arrow.
                strand_a, strand_b = secondary[before, 2], secondary[after, 2]
                row[18] = min(1.0, strand_a * (1 - strand_b) + strand_b * (j == len(chain) - 2))
    return rows


# Typical turn of a cartoon's width axis from one residue to the next about the Cα–Cα
# chord, for coil, helix and strand: right-handed in α-helices, slight elsewhere.
CARTOON_TWIST = np.radians([15.0, 49.0, 15.0])


def _turn(v, axis, angle):
    """Rotate the rows of v about unit rows of axis by angle (radians), by Rodrigues' formula."""
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return v * c + np.cross(axis, v) * s + axis * np.sum(axis * v, 1)[:, None] * (1 - c)


def _smoothstep(lo, hi, x):
    """Hermite smoothstep from 0 at ``lo`` to 1 at ``hi``, as in WGSL."""
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _line_directions(tensors, normals):
    """Principal direction of each symmetric tensor within the plane perpendicular to its normal."""
    # An orthonormal basis (u, v) of each plane, from whichever world axis is far from the normal.
    reference = np.where(np.abs(normals[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    u = normalize(np.cross(normals, reference))
    v = np.cross(normals, u)
    # The tensor restricted to the plane is the 2 × 2 matrix [[a, b], [b, d]]. Its major
    # eigenvector makes the angle θ with u where tan 2θ = 2b / (a - d).
    a = np.einsum("ni,nij,nj->n", u, tensors, u)
    b = np.einsum("ni,nij,nj->n", u, tensors, v)
    d = np.einsum("ni,nij,nj->n", v, tensors, v)
    angle = 0.5 * np.arctan2(2 * b, a - d)
    return np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v


def protein_guides(points, secondary=None, carbonyls=None):
    """Cartoon width axes along a Cα trace, and how firmly each step's twist is determined.

    Axes depend only on atoms a few residues away, so they move continuously with
    them: crossing consecutive Cα bisectors gives the local helix axis, and curvature
    binormals orient strands and loops. Cα→O vectors (``carbonyls``, zero where
    missing) only decide where the trace is straight. Lines are averaged as sign-free
    tensors, then each axis is oriented like its predecessor turned by the typical
    twist (CARTOON_TWIST). The firmness of step i → i+1 is 1 where that twist is clear
    and falls to 0 where it is ambiguous (about 90° from typical). Renderers round the
    ribbon there, so a change of twist during an animation cannot flip a flat ribbon.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    tangents = normalize(np.gradient(points, axis=0))
    # Curvature binormals: perpendicular to the plane of two consecutive chords. The end
    # residues have only one chord, so they copy their neighbours.
    binormal = np.zeros_like(points)
    binormal[1:-1] = np.cross(points[1:-1] - points[:-2], points[2:] - points[1:-1])
    binormal[0], binormal[-1] = binormal[1], binormal[-2]
    binormal = normalize(binormal)
    # Bisectors point from each Cα toward the midpoint of its neighbours; in a helix,
    # toward the helix axis. The cross product of two consecutive bisectors then lies
    # along the axis. axis[i] belongs to step i → i+1 and before[i] to step i-1 → i.
    bisector = np.zeros_like(points)
    bisector[1:-1] = normalize(points[:-2] + points[2:] - 2 * points[1:-1])
    axis = np.zeros_like(points)
    axis[1:-2] = np.cross(bisector[1:-2], bisector[2:-1])
    before = np.r_[np.zeros((1, 3)), axis[:-1]]
    chords = normalize(points[1:] - points[:-1])
    # Carbonyl directions, made perpendicular to the chord to the next residue.
    peptide = np.zeros_like(points) if carbonyls is None else np.asarray(carbonyls, np.float64)
    next_chord = np.r_[chords, chords[-1:]]
    peptide = normalize(peptide - np.sum(peptide * next_chord, 1)[:, None] * next_chord)
    # Each direction enters as the outer product x xᵀ, which is the same for x and -x.
    # Missing terms are zero vectors and add nothing. The small carbonyl weight matters
    # only where the trace is straight and the other terms vanish.
    terms = ((1, axis), (1, before), (0.5, binormal), (0.05, peptide))
    lines = sum(w * x[:, :, None] * x[:, None, :] for w, x in terms)
    # Add half of each neighbour's tensor, then take the principal line across the trace.
    window = lines.copy()
    window[1:] += 0.5 * lines[:-1]
    window[:-1] += 0.5 * lines[1:]
    guides = _line_directions(window, tangents)
    # Expected twist of each step, from the mean secondary structure of its two residues.
    weights = np.asarray(secondary, np.float64) if secondary is not None else np.tile([1.0, 0, 0], (n, 1))
    step = 0.5 * (weights[:-1] + weights[1:]) @ CARTOON_TWIST

    def orient(g):
        # Flip an axis when it points against its predecessor turned by the expected
        # twist. The running product carries each flip on down the chain.
        agree = np.sum(g[1:] * _turn(g[:-1], chords, step), 1)
        return g * np.cumprod(np.r_[1.0, np.where(agree < 0, -1.0, 1.0)])[:, None], np.abs(agree)

    guides, _ = orient(guides)
    # Pull each axis toward its neighbours' predictions of it, ignoring ambiguous neighbours.
    ahead, behind = _turn(guides[:-1], chords, step), _turn(guides[1:], chords, -step)
    trust = _smoothstep(0.15, 0.45, np.abs(np.sum(ahead * guides[1:], 1)))[:, None]
    smoothed = guides.copy()
    smoothed[1:] += 0.5 * trust * ahead
    smoothed[:-1] += 0.5 * trust * behind
    smoothed -= np.sum(smoothed * tangents, 1)[:, None] * tangents
    guides, agree = orient(normalize(smoothed))
    # The last residue has no step after it, so its firmness is 1.
    return guides, np.r_[_smoothstep(0.1, 0.4, agree), 1.0]


def _transported_guides(topology, xyz, chain, pts):
    """Width axes for a nucleic-acid chain or a two-residue protein chain, (residues, 3).

    Each residue's guide atom (C1′ for nucleotides, the carbonyl O for amino acids)
    sets the direction, blended with the previous axis carried along the chain. ``pts``
    are the chain's trace-atom positions.
    """
    tangents = normalize(np.gradient(pts, axis=0))
    guides = []
    last = None
    for j, ri in enumerate(chain):
        tangent = tangents[j]
        oxygen = topology.residues[ri].guide_atom
        guide = xyz[oxygen] - pts[j] if oxygen >= 0 else np.array([0.0, 1.0, 0.0])
        guide -= np.dot(guide, tangent) * tangent
        # A guide atom on the trace line gives no direction: use any perpendicular.
        if np.linalg.norm(guide) < 1e-6:
            guide = np.cross(tangent, [1.0, 0.0, 0.0])
            if np.linalg.norm(guide) < 1e-6:
                guide = np.cross(tangent, [0.0, 0.0, 1.0])
        guide = normalize(guide)
        if last is not None:
            # Parallel transport: the previous axis, made perpendicular to this tangent.
            transported = last - np.dot(last, tangent) * tangent
            transported = normalize(transported)
            if np.dot(guide, transported) < 0:
                guide = -guide
            guide = normalize(0.65 * guide + 0.35 * transported)
        guides.append(guide)
        last = guide
    guides = np.array(guides)
    # Two passes of a [1/4, 1/2, 1/4] filter, each projected back across the trace.
    for _ in range(2):
        padded = np.pad(guides, ((1, 1), (0, 0)), mode="edge")
        guides = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        guides -= np.sum(guides * tangents, axis=1)[:, None] * tangents
        guides = normalize(guides)
    return guides


def state_data(topology, xyz, reference_normals=None, secondary=None):
    """Compute frame guides once per keyframe, never once per rendered tween.

    Rows hold the position and, on trace atoms, the cartoon width axis plus the
    firmness of the twist to the next residue (see protein_guides). ``secondary``
    gives coil/helix/strand weights per residue, which set the expected twist.

    The result is (atoms, 8) float32, read by the shader as ``struct AtomState``:
    columns 0:3 are the position in Å, column 3 is padding, columns 4:7 the unit
    width axis and column 7 the firmness in [0, 1]. Columns 4:8 are zero on atoms
    that are not trace atoms. ``reference_normals`` (atoms, 3) are the width axes of
    another keyframe; each chain's axes are flipped as a whole to agree with them.
    """
    data = np.zeros((len(xyz), 8), np.float32)
    data[:, :3] = xyz
    for chain in topology.chains:
        ca = np.array([topology.residues[i].trace_atom for i in chain])
        pts = xyz[ca]
        if len(chain) >= 3 and not topology.residues[chain[0]].is_nucleic:
            oxygen = np.array([topology.residues[i].oxygen for i in chain])
            carbonyls = np.where(oxygen[:, None] >= 0, xyz[np.maximum(oxygen, 0)] - pts, 0.0)
            weights = None if secondary is None else secondary[chain]
            guides, firmness = protein_guides(pts, weights, carbonyls)
            if reference_normals is not None and np.sum(guides * reference_normals[ca]) < 0:
                guides *= -1
            data[ca, 4:7], data[ca, 7] = guides, firmness
            continue
        guides = _transported_guides(topology, xyz, chain, pts)
        if reference_normals is not None:
            # Choose a consistent sign for the whole continuous chain across frames.
            if np.sum(guides * reference_normals[ca]) < 0:
                guides *= -1
        data[ca, 4:7], data[ca, 7] = guides, 1.0
    return data


def sweep_grid(steps=10, sides=12):
    """Vertices and triangle indices that a segment instance sweeps into a tube.

    Each vertex is (t, angle) in [0, 1]²: t runs along the segment and angle around
    it, in turns. The angle 1 repeats 0 so the seam can close. ``steps`` × ``sides``
    quads give two triangles each; the native renderer uses 10 × 12, Studio's classic
    cartoons 20 × 24.
    """
    vertices = np.array(
        [(i / steps, j / sides) for i in range(steps + 1) for j in range(sides + 1)], np.float32
    )
    indices = []
    for i in range(steps):
        for j in range(sides):
            a, b = i * (sides + 1) + j, (i + 1) * (sides + 1) + j
            indices.extend([a, b, a + 1, a + 1, b, b + 1])
    return vertices, np.asarray(indices, np.uint32)
