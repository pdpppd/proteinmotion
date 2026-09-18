"""Live distance rulers with depth-tested cylinders or screen overlay leaders."""

import numpy as np

from .annotations import Annotation, AnnotationLayout, Leader, Placement, Text
from .math3d import color as parse_color
from .protein import Protein
from .regions import Region
from .structure import Atom, Residue, Topology


def endpoint(region, anchor):
    if not isinstance(region, Region):
        raise TypeError("Distance endpoints must be Regions selected from proteins")
    if anchor not in ("backbone", "ca", "centroid"):
        raise ValueError("anchor must be backbone, ca, or centroid")
    if anchor != "centroid" and len(region.atom_indices) > 1 and len(region.residue_indices) == 1:
        residue = region.protein.topology.residues[region.residue_indices[0]]
        index = residue.trace_atom if anchor == "backbone" else residue.ca
        if index >= 0:
            return Region(region.protein, [index])
    return region


def project(point, camera, width, height):
    clip = camera.matrix(width / height) @ np.r_[point, 1]
    if clip[3] <= 1e-8:
        return None
    ndc = clip[:3] / clip[3]
    if not (0 <= ndc[2] <= 1) or np.any(np.abs(ndc[:2]) > 1):
        return None
    return np.array([(ndc[0] + 1) * width / 2, (1 - ndc[1]) * height / 2])


def line_intervals(style, count, ratio, progress=1, gap=0):
    intervals = [(0.0, 1.0)] if style == "solid" else [(i / count, (i + ratio) / count) for i in range(count)]
    result = []
    for a, b in intervals:
        b = min(b, progress)
        for lo, hi in ((0, 0.5 - gap / 2), (0.5 + gap / 2, 1)) if gap else ((0, 1),):
            left, right = max(a, lo), min(b, hi)
            if right - left > 1e-8:
                result.append((left, right))
    return result


class _Ruler(Protein):
    def __init__(self, count, radius, color):
        atoms = tuple(Atom("ruler", i + 1, "", "RUL", "CA", "C", i) for i in range(2 * count))
        residues = tuple(Residue("ruler", i + 1, "", "RUL", i, -1) for i in range(2 * count))
        bonds = np.arange(2 * count, dtype=np.uint32).reshape(-1, 2)
        super().__init__(Topology(atoms, residues, bonds, ()), np.zeros((2 * count, 3)))
        self.ball_and_stick(bond_radius=radius)
        self._draw_atoms = False
        self.paint(color)

    def paint(self, color):
        data = np.zeros((len(self.topology.atoms), 4), np.float32)
        data[:, :3] = color
        if self._metadata_override is None or not np.array_equal(data, self._metadata_override):
            data.flags.writeable = False
            self._metadata_override = data

    def update(self, a, b, intervals, opacity):
        count = len(self.topology.bonds)
        points = np.zeros((count * 2, 3), np.float32)
        appearance = self._appearance.copy()
        appearance[:, 12:14] = 0
        for i, (lo, hi) in enumerate(intervals[:count]):
            points[2 * i : 2 * i + 2] = a + np.array([lo, hi])[:, None] * (b - a)
            appearance[2 * i : 2 * i + 2, 12:14] = 1
        if not np.array_equal(points, self._a):
            self.set_positions(points)
        if not np.array_equal(appearance, self._appearance):
            appearance.flags.writeable = False
            self._appearance = appearance
        self.opacity = opacity if np.linalg.norm(b - a) > 1e-8 else 0


class Distance(Annotation):
    """A live ruler. Same-protein model distance ignores display transforms by default."""

    def __init__(
        self,
        start,
        end,
        *,
        mode="3d",
        anchor="backbone",
        space="model",
        color="#f2ba67",
        label_color=None,
        font_size=27,
        precision=1,
        unit="Å",
        prefix="",
        show_distance=True,
        style="dashed",
        line_width=1.7,
        radius=0.09,
        dash_count=12,
        dash_ratio=0.6,
        follow_opacity=True,
        opacity=1,
    ):
        super().__init__()
        if mode not in ("2d", "3d") or space not in ("model", "world"):
            raise ValueError("Use mode=2d|3d and space=model|world")
        if style not in ("solid", "dashed"):
            raise ValueError("style must be solid or dashed")
        if not isinstance(dash_count, int) or not 1 <= dash_count <= 512:
            raise ValueError("dash_count must be between 1 and 512")
        if not 0 < dash_ratio <= 1 or not np.isfinite(dash_ratio):
            raise ValueError("dash_ratio must be in (0, 1]")
        if not isinstance(precision, int) or not 0 <= precision <= 6:
            raise ValueError("precision must be between 0 and 6")
        if any(not np.isfinite(v) or v <= 0 for v in (line_width, radius, font_size)):
            raise ValueError("Line width, radius and font size must be finite and positive")
        self.start, self.end = endpoint(start, anchor), endpoint(end, anchor)
        if space == "model" and start.protein is not end.protein:
            raise ValueError("Use space=world for endpoints on different protein objects")
        self.mode, self.space, self.style = mode, space, style
        self.color = parse_color(color)
        self.show_distance, self.follow_opacity = show_distance, follow_opacity
        if unit not in ("Å", "nm", ""):
            raise ValueError("unit must be Å, nm, or an empty suffix")
        self.precision, self.unit, self.prefix = precision, unit, prefix
        self.line_width, self.radius, self.dash_count, self.dash_ratio = (
            line_width,
            radius,
            dash_count,
            dash_ratio,
        )
        self.title = Text(
            self._caption(),
            font_size=font_size,
            font="semibold",
            color=self.color if label_color is None else label_color,
        )
        self._ruler = _Ruler(dash_count + 2, radius, self.color) if mode == "3d" else None
        self.active = True
        self.set_opacity(opacity)

    @property
    def distance(self):
        if self.space == "model":
            return float(np.linalg.norm(self.start.positions.mean(0) - self.end.positions.mean(0)))
        return float(np.linalg.norm(self.start.world_positions.mean(0) - self.end.world_positions.mean(0)))

    def _caption(self):
        return f"{self.prefix}{self.distance / (10 if self.unit == 'nm' else 1):.{self.precision}f}" + (
            f" {self.unit}" if self.unit else ""
        )

    @property
    def glyph_count(self):
        return self.title.glyph_count if self.show_distance else 0

    @property
    def text_progress(self):
        return float(np.clip((self._write - 0.15) / 0.85, 0, 1))

    def _evaluate(self, camera, width, height):
        if not self.active:
            return None
        a, b = self.start.world_positions.mean(0), self.end.world_positions.mean(0)
        pa, pb = project(a, camera, width, height), project(b, camera, width, height)
        if pa is None or pb is None:
            return None
        if self.show_distance:
            self.title.set_text(self._caption())
        size = (
            np.array([self.title.geometry.width, self.title.geometry.height]) * height / 1080
            if self.show_distance
            else np.zeros(2)
        )
        center = (pa + pb) / 2 if self.mode == "2d" else project((a + b) / 2, camera, width, height)
        if center is None:
            return None
        direction = pb - pa
        length = np.linalg.norm(direction)
        # Leave a readable gap around the centered caption, in both line modes.
        unit = direction / max(length, 1e-8)
        extents = np.where(
            np.abs(unit) > 1e-8, (size / 2 + 7 * height / 1080) / np.maximum(np.abs(unit), 1e-8), np.inf
        )
        gap = min(0.95, 2 * float(extents.min()) / max(length, 1)) if self.show_distance else 0
        intervals = line_intervals(
            self.style, self.dash_count, self.dash_ratio, min(1, self._write / 0.35), gap
        )
        visibility = (
            min(float(r.protein.atom_opacities[r.atom_indices].mean()) for r in (self.start, self.end))
            if self.follow_opacity
            else 1
        )
        return a, b, pa, pb, center - size / 2, intervals, visibility

    def _geometry_objects(self, camera, width, height, *, parent_opacity=1):
        if self._ruler is None:
            return []
        result = self._evaluate(camera, width, height)
        if result is None:
            self._ruler.opacity = 0
            return []
        a, b, _, _, _, intervals, visibility = result
        self._ruler.paint(self.color)
        self._ruler.update(a, b, intervals, self.opacity * visibility * parent_opacity)
        return [self._ruler]

    def layout(self, camera, width, height):
        result = self._evaluate(camera, width, height)
        if result is None:
            return AnnotationLayout([], [])
        _, _, a, b, origin, intervals, visibility = result
        leaders = []
        if self.mode == "2d" and intervals:
            pairs = a + np.asarray(intervals)[:, :, None] * (b - a)
            leaders = [Leader(pairs, self.color, self.line_width * height / 1080, visibility)]
        text = [Placement(self.title, origin, opacity=visibility)] if self.show_distance else []
        return AnnotationLayout(text, leaders)
