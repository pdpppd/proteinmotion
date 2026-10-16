"""A live Ramachandran plot over favored and allowed regions from high-resolution structures.

The regions enclose 98% (favored) and 99.8% (allowed) of residues from one chain
per 30% sequence-identity cluster of X-ray entries at 1.4 Å resolution or better,
with separate references for general residues, glycine, trans proline and
pre-proline. ``scripts/build_ramachandran_reference.py`` rebuilds them;
``data/ramachandran.json`` lists the entries and filters.

How the reference was built (see that script): entries also need R-free ≤ 0.20, a
single protein entity and 80–700 residues, and a residue counts only when its
backbone and flanking peptide atoms have full occupancy, no alternate locations and
B factors under 30 Å². Each class is histogrammed on a 2° grid, smoothed with a
periodic Gaussian (σ = 5°), and cut at the densities that enclose 98% and 99.8% of
its residues; isolated patches under 12 cells are dropped. ``data/ramachandran.npz``
stores one (180, 180) uint8 grid per class.

``RamachandranPlot`` is an annotation the scene redraws every frame: it measures φ and
ψ from the protein's current coordinates and draws the paths that ``SetTorsions``
leaves in ``Protein._torsion_motion``. ``regions`` is also read by torsions.py, which
turns φ and ψ the way that stays in these regions.
"""

from functools import lru_cache
from importlib.resources import files

import numpy as np

from .geometry import atom_metadata, residue_colors
from .math3d import color as parse_color
from .plots import _Plot, _selection
from .styling import current_tints
from .torsions import dihedrals, torsion_atoms, wrap

BACKGROUNDS = ("general", "glycine", "proline", "preproline")


@lru_cache(maxsize=None)
def regions(background="general"):
    """Levels on a 2° grid indexed [ψ, φ] from −180°: 2 favored, 1 allowed, 0 outside."""
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {', '.join(BACKGROUNDS)}")
    with files("proteinmotion").joinpath("data/ramachandran.npz").open("rb") as handle:
        data = np.load(handle)
        levels = np.array(data[background], dtype=np.uint8)
    # The cached grid is shared by every caller, so it is read-only.
    levels.flags.writeable = False
    return levels


def classify(phi, psi, background="general"):
    """'favored', 'allowed' or 'outlier' for each (φ, ψ) in degrees.

    A residue whose φ or ψ is NaN, such as a chain end, is 'undefined'.
    """
    levels = regions(background)
    step = 360 / levels.shape[0]
    phi, psi = np.asarray(phi, dtype=float), np.asarray(psi, dtype=float)
    # A residue without φ or ψ has no class, and the plot draws no point for it. NaN has
    # no grid cell, so look such residues up at −180° and replace the result.
    defined = np.isfinite(phi) & np.isfinite(psi)
    phi, psi = np.where(defined, phi, -180.0), np.where(defined, psi, -180.0)
    # Cell index of each angle; +180° falls past the last cell, so clip it back in.
    rows = np.clip(((psi + 180) // step).astype(int), 0, levels.shape[0] - 1)
    cols = np.clip(((phi + 180) // step).astype(int), 0, levels.shape[1] - 1)
    names = np.array(["outlier", "allowed", "favored", "undefined"])
    return names[np.where(defined, levels[rows, cols], 3)]


def _runs(mask):
    """(row, first column, length) of each horizontal run of True cells."""
    # Pad with False on both sides so every run has a 0→1 step at its start and a 1→0
    # step after its end.
    padded = np.pad(mask.astype(np.int8), ((0, 0), (1, 1)))
    rows, starts = np.nonzero(np.diff(padded, axis=1) == 1)
    _, stops = np.nonzero(np.diff(padded, axis=1) == -1)
    return np.column_stack((rows, starts, stops - starts)).astype(float)


@lru_cache(maxsize=None)
def _contours(background, level):
    """Smooth region boundaries in grid units (x = φ cell, y = ψ cell from the top).

    ``level`` 0.5 traces the edge of the allowed region and 1.5 the edge of the favored one.
    """
    from skimage.measure import find_contours

    # Flip rows so ψ = +180° is at the top, as on screen.
    grid = regions(background)[::-1].astype(float)
    size = grid.shape[0]
    # Tile periodically so regions crossing ±180° close smoothly, then clip to the square.
    tiled = np.tile(grid, (3, 3))
    result = []
    for contour in find_contours(tiled, level):
        points = contour[:, ::-1] - size + 0.5  # (x, y), cell centers at +0.5
        inside = (points >= 0).all(1) & (points <= size).all(1)
        # Keep the stretches of the contour that lie in the central tile.
        for piece in np.split(np.arange(len(points)), np.flatnonzero(np.diff(inside.astype(int))) + 1):
            if inside[piece[0]] and len(piece) > 1:
                result.append(points[piece])
    return tuple(result)


class RamachandranPlot(_Plot):
    """A live φ/ψ plot: one point per residue over favored and allowed regions.

    Points follow the protein through torsion animations, morphs and trajectories
    and use the residue colors drawn in 3D. Glycine is drawn as a triangle and
    proline as a square. ``highlight`` enlarges, rings and labels a Region's
    residues. While SetTorsions runs, each moving residue shows its path from a
    hollow start marker. ``background`` chooses the reference regions: general,
    glycine, proline, preproline or None.
    """

    def __init__(
        self,
        protein,
        *,
        region=None,
        highlight=None,
        labels=True,
        color=None,
        title="Ramachandran plot",
        position=(0.64, 0.1),
        size=(0.32, 0.56),
        dot_size=5.5,
        background="general",
        region_labels=True,
        paths=True,
        highlight_color="#f5d477",
        favored_color="#2b4566",
        allowed_color="#1a2b42",
    ):
        super().__init__(position, size, title)
        # Validate the highlight and background now rather than at the first frame.
        _selection(protein, highlight)
        if background is not None:
            regions(background)
        ids = _selection(protein, region) if region is not None else range(len(protein.topology.residues))
        # Plot only residues that have both φ and ψ (table columns 0 and 1).
        table = torsion_atoms(protein.topology)
        self.ids = np.array(
            [i for i in sorted(ids) if table[i, 0, 0] >= 0 and table[i, 1, 0] >= 0], dtype=int
        )
        if not len(self.ids):
            raise ValueError("No selected residue has both φ and ψ; select amino acids inside a chain")
        if not np.isfinite(dot_size) or dot_size <= 0:
            raise ValueError("dot_size must be finite and positive")
        self.protein, self.highlight, self.labels = protein, highlight, bool(labels)
        self.color = None if color is None else parse_color(color)
        self.dot_size, self.background = float(dot_size), background
        self.region_labels, self.paths = bool(region_labels), bool(paths)
        self.highlight_color = parse_color(highlight_color)
        self.favored_color, self.allowed_color = parse_color(favored_color), parse_color(allowed_color)
        self.quads = table[self.ids][:, :2]
        # Marker shape per residue: 0 circle, 1 square (proline), 2 triangle (glycine).
        names = [protein.topology.residues[i].name for i in self.ids]
        self.marker = np.array([2 if n == "GLY" else 1 if n == "PRO" else 0 for n in names])

    @property
    def angles(self):
        """Current (φ, ψ) in degrees for each plotted residue."""
        return dihedrals(self.protein.positions, self.quads)

    def _colors(self):
        """(residues, 3) RGB for each plotted residue, matching its color in the 3D view."""
        if self.color is not None:
            return np.broadcast_to(self.color, (len(self.ids), 3))
        protein = self.protein
        trace = [protein.topology.residues[i].trace_atom for i in self.ids]
        tint = current_tints(protein)[trace]
        base = (
            atom_metadata(protein)[trace, :3]
            if protein.color_scheme == "element"
            else residue_colors(protein)[self.ids]
        )
        # A tint row is (r, g, b, strength) with the color already scaled by its strength
        # (all zeros means no tint), so this blends the base color toward the tint.
        return base * (1 - tint[:, 3:]) + tint[:, :3]

    def _marker(self, center, radius, kind, tint):
        """Draw one residue's marker; ``kind`` is 0 circle, 1 square, 2 triangle."""
        if kind == 1:
            r = radius * 0.9
            self._polygon(center + np.array([[-r, -r], [r, -r], [r, r], [-r, r]]), tint)
        elif kind == 2:
            # An equilateral triangle pointing up (screen y grows downward).
            r = radius * 1.3
            self._polygon(center + r * np.array([[0, -1.0], [0.87, 0.5], [-0.87, 0.5]]), tint)
        else:
            self._dot(center, radius, tint)

    def _ring(self, center, radius, tint, width, opacity=1.0):
        """A circle outline in screen pixels."""
        angles = np.linspace(0, 2 * np.pi, 33)
        self._line(center + radius * np.column_stack((np.cos(angles), np.sin(angles))), tint, width, opacity)

    def layout(self, camera, width, height):
        self._begin(width, height)
        x, y = self._origin
        w, h = self._extent
        s = self._pixel_scale
        self._text("title", self.title, [x, y], size=28, tint="#edf3fc")
        # A square plot area, leaving room on the left and below for tick labels and the
        # axis symbols. Lengths here are pixels at 1080p, times s.
        extent = max(1.0, min(w - 88 * s, h - 104 * s))
        left, top = x + 72 * s, y + 50 * s

        def project(phi, psi):
            """Screen pixels for (φ, ψ) in degrees: φ left to right, ψ bottom to top."""
            phi, psi = np.atleast_1d(phi), np.atleast_1d(psi)
            return np.column_stack((left + (phi + 180) / 360 * extent, top + (180 - psi) / 360 * extent))

        # Draw back to front: regions, frame and labels, motion paths, then the residues.
        if self.background is not None:
            self._draw_regions(left, top, extent)
        self._draw_frame(project, left, top, extent)
        angles = self.angles
        colors = self._colors()
        selected = _selection(self.protein, self.highlight)
        motion = self.protein._torsion_motion if self.paths else None
        if motion is not None:
            self._draw_paths(project, motion, colors)
        self._draw_residues(project, angles, colors, selected, left, extent)
        return self._finish()

    def _draw_regions(self, left, top, extent):
        """Fill the allowed and favored regions and outline their edges."""
        grid = regions(self.background)[::-1]
        cell = extent / grid.shape[0]
        # Each horizontal run of cells becomes one rectangle; favored is drawn over allowed.
        for mask, tint in ((grid >= 1, self.allowed_color), (grid >= 2, self.favored_color)):
            runs = _runs(mask)
            boxes = np.column_stack(
                (
                    left + runs[:, 1] * cell,
                    top + runs[:, 0] * cell,
                    runs[:, 2] * cell,
                    np.full(len(runs), cell),
                )
            )
            # Slight overlap hides seams between neighbouring strips.
            self._rects(boxes + [0, -0.2, 0, 0.4], tint)
        for level, tint, line in ((0.5, "#3d5878", 1.0), (1.5, "#5b7ca6", 1.3)):
            for contour in _contours(self.background, level):
                self._line(np.array([left, top]) + contour * cell, tint, width=line)

    def _draw_frame(self, project, left, top, extent):
        """The border, the φ = 0 and ψ = 0 lines, ticks every 90°, axis symbols and region names."""
        s = self._pixel_scale
        corners = np.array(
            [
                [left, top],
                [left + extent, top],
                [left + extent, top + extent],
                [left, top + extent],
                [left, top],
            ]
        )
        self._line(corners, "#7d8fa8", width=1.2)
        # Faint lines through φ = 0 and ψ = 0.
        middle = project([0, -180, 0, 180], [-180, 0, 180, 0])
        self._line(
            np.array([[middle[0], middle[2]], [middle[1], middle[3]]]), "#7d8fa8", width=1, opacity=0.35
        )
        for k, value in enumerate((-180, -90, 0, 90, 180)):
            px, py = project(value, value)[0]
            self._line([[px, top + extent], [px, top + extent + 5 * s]], width=1)
            self._line([[left - 5 * s, py], [left, py]], width=1)
            label = f"{value}".replace("-", "−")
            self._text(f"x{k}", label, [px, top + extent + 9 * s], size=19, align="center", tint="#a9b6c9")
            self._text(f"y{k}", label, [left - 9 * s, py - 11 * s], size=19, align="right", tint="#a9b6c9")
        self._text("phi", "φ", [left + extent / 2, top + extent + 32 * s], size=26, align="center")
        self._text("psi", "ψ", [left - 62 * s, top + extent / 2 - 15 * s], size=26, align="center")
        if self.region_labels and self.background in ("general", None):
            # Labels sit just outside each region so they never cover residues.
            for key, text, phi, psi in (("aR", "αR", -18, -40), ("b", "β", -28, 158), ("aL", "αL", 100, 62)):
                px, py = project(phi, psi)[0]
                self._text(f"region-{key}", text, [px, py - 12 * s], size=20, align="center", tint="#9db1cc")

    def _draw_paths(self, project, motion, colors):
        """Each moving residue's (φ, ψ) path during SetTorsions, from a hollow start ring."""
        s = self._pixel_scale
        residues, start, delta, clip = motion
        # Paths fade in and out with the clip: zero at its start and end, full halfway.
        strength = float(np.sin(np.pi * np.clip(clip, 0, 1)))
        rows = {int(ri): k for k, ri in enumerate(self.ids)}
        if strength > 0.02:
            t = np.linspace(0, 1, 65)[:, None]
            for ri, a, change in zip(residues, start, delta):
                k = rows.get(int(ri))
                if k is None or not np.isfinite(a).all() or not np.any(change):
                    continue
                # SetTorsions moves φ and ψ together along a straight line in torsion space.
                # Break the line where it wraps across ±180° so it does not streak across the plot.
                track = wrap(a + t * change)
                jumps = np.flatnonzero(np.abs(np.diff(track, axis=0)).max(1) > 180)
                for piece in np.split(np.arange(len(track)), jumps + 1):
                    if len(piece) > 1:
                        self._line(project(*track[piece].T), colors[k], width=1.6, opacity=0.55 * strength)
                self._ring(project(*a)[0], 4.5 * s, colors[k], 1.3, 0.75 * strength)

    def _draw_residues(self, project, angles, colors, selected, left, extent):
        """One marker per residue; highlighted residues get a ring and a label."""
        s = self._pixel_scale
        # Highlighted residues are drawn last, on top of the rest.
        order = sorted(range(len(self.ids)), key=lambda k: self.ids[k] in selected)
        for k in order:
            phi, psi = angles[k]
            if not np.isfinite([phi, psi]).all():
                continue
            point = project(phi, psi)[0]
            radius = self.dot_size * s
            chosen = self.ids[k] in selected
            if chosen:
                self._ring(point, 2.3 * radius, self.highlight_color, 2.0)
                radius *= 1.3
            self._marker(point, radius, self.marker[k], colors[k])
            if self.labels and chosen:
                r = self.protein.topology.residues[self.ids[k]]
                # Labels open toward the plot's interior so they stay inside the frame.
                right = point[0] < left + 0.72 * extent
                self._text(
                    f"label{self.ids[k]}",
                    f"{r.name.title()} {r.resid}",
                    point + [15 * s if right else -15 * s, -36 * s],
                    size=21,
                    tint=self.highlight_color,
                    align="left" if right else "right",
                )
