"""Shaped font outlines, cached tessellation, and arc-length stroke geometry.

HarfBuzz lays out glyphs; FontTools supplies quadratic/cubic curves. No bitmap
text is rasterized per frame. Coordinates are design pixels at a 1080-pixel height.

``layout_text`` is called by the text annotations (annotations.py). Its
``TextGeometry`` goes to overlay.py, which uploads the filled triangles as a vertex
buffer and the outline segments as a storage buffer for shaders/text.wgsl. The Write
animation first traces each glyph's outline along its length, then fades in the fill.
Glyph outlines are flattened into polygons, triangulated with earcut, and cached per
font, glyph and size. Design pixels have y pointing down, as on screen.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import mapbox_earcut
import numpy as np
import uharfbuzz as hb
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont


@dataclass(frozen=True, eq=False)
class TextGeometry:
    """Laid-out text, in design pixels with the ink's top-left corner near (0, 0).

    ``fill`` is a (3k, 3) float32 triangle list of x, y and the glyph's index.
    ``strokes`` is (m, 8) float32, one outline segment per row: start x, y, end x, y,
    the segment's start and end as fractions of its glyph's outline length, the glyph's
    index, and an unused pad (text.wgsl reads each row as two vec4f). ``width`` covers
    both the ink and the advance width; ``height`` is the ink's height. ``glyph_count``
    counts glyphs with an outline, so spaces do not count, and ``clusters`` gives each
    one's character offset in the source text.
    """

    fill: np.ndarray
    strokes: np.ndarray
    width: float
    height: float
    glyph_count: int
    clusters: tuple[int, ...]


def font_path(font):
    """File path for "regular" or "semibold" (the bundled Source Sans 3) or a user font file."""
    if font in ("regular", "semibold"):
        name = "Regular" if font == "regular" else "Semibold"
        return str(files("proteinmotion").joinpath(f"fonts/SourceSans3-{name}.otf"))
    path = Path(font).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Font not found: {path}")
    return str(path)


@lru_cache(maxsize=16)
def _font(path):
    """The FontTools font (outlines), a HarfBuzz font (shaping) and units per em."""
    font = TTFont(path, fontNumber=0)
    face = hb.Face(Path(path).read_bytes())
    shaped = hb.Font(face)
    # Positions come back in font units; layout_text scales them by size / upem.
    shaped.scale = (face.upem, face.upem)
    hb.ot_font_set_funcs(shaped)
    return font, shaped, face.upem


class _FlattenPen(BasePen):
    """A FontTools pen that flattens a glyph's outline into closed polygons.

    Curves are split until they are within ``tolerance`` design pixels of a straight
    line. ``scale`` converts font units to design pixels. Finished polygons collect in
    ``contours``, as (n, 2) arrays.
    """

    def __init__(self, glyphs, scale, tolerance=0.075):
        super().__init__(glyphs)
        self.scale, self.tolerance = scale, tolerance
        self.contours, self.current = [], []

    def _point(self, p):
        # Fonts have y up; design pixels have y down.
        return np.array([p[0], -p[1]], float) * self.scale

    def _moveTo(self, p):
        if self.current:
            self._closePath()
        self.current = [self._point(p)]

    def _lineTo(self, p):
        self.current.append(self._point(p))

    def _cubic(self, a, b, c, d, depth=0):
        """Append the Bézier curve with control points a, b, c, d, as a polyline."""
        # Flatness: the farthest inner control point from the chord a-d. Where a and d
        # coincide, use the distance from a.
        chord = d - a
        length = np.linalg.norm(chord)
        if length < 1e-10:
            error = max(np.linalg.norm(b - a), np.linalg.norm(c - a))
        else:
            error = max(abs(chord[0] * (p - a)[1] - chord[1] * (p - a)[0]) / length for p in (b, c))
        # Fourteen halvings bound the recursion for degenerate curves.
        if error <= self.tolerance or depth >= 14:
            self.current.append(d)
            return
        # De Casteljau split at t = 1/2 into two cubics, each flattened in turn.
        ab, bc, cd = (a + b) / 2, (b + c) / 2, (c + d) / 2
        abc, bcd = (ab + bc) / 2, (bc + cd) / 2
        middle = (abc + bcd) / 2
        self._cubic(a, ab, abc, middle, depth + 1)
        self._cubic(middle, bcd, cd, d, depth + 1)

    def _curveToOne(self, p1, p2, p3):
        self._cubic(self.current[-1], self._point(p1), self._point(p2), self._point(p3))

    def _qCurveToOne(self, p1, p2):
        # A quadratic curve (TrueType) is the cubic with these control points.
        a, b, c = self.current[-1], self._point(p1), self._point(p2)
        self._cubic(a, a + 2 * (b - a) / 3, c + 2 * (b - c) / 3, c)

    def _closePath(self):
        if len(self.current) >= 3:
            points = np.array(self.current)
            # Drop repeated points, including a last point equal to the first.
            keep = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) > 1e-8
            points = points[keep]
            if len(points) >= 3:
                self.contours.append(points)
        self.current = []

    def _endPath(self):
        self._closePath()


def _area(points):
    """Signed area of a closed polygon (shoelace formula)."""
    other = np.roll(points, -1, axis=0)
    return float(np.sum(points[:, 0] * other[:, 1] - other[:, 0] * points[:, 1]) / 2)


def _inside(point, polygon):
    """Whether point lies inside polygon, by the even-odd rule along a ray toward +x."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    # Edges that straddle the point's y; x is where each crosses that horizontal line.
    crossing = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    denominator = np.where(crossing, b[:, 1] - a[:, 1], 1)
    x = a[:, 0] + (point[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / denominator
    return bool(np.count_nonzero(crossing & (point[0] < x)) % 2)


def _triangulate(contours):
    """Group outer rings and holes; disconnected accent components stay separate."""
    if not contours:
        return np.empty((0, 2), np.float32)
    # Each ring's parent is the smallest larger ring that contains its first point.
    areas = np.array([abs(_area(p)) for p in contours])
    parents = []
    for i, ring in enumerate(contours):
        enclosing = [j for j, outer in enumerate(contours) if areas[j] > areas[i] and _inside(ring[0], outer)]
        parents.append(min(enclosing, key=lambda j: areas[j]) if enclosing else -1)
    # Nesting depth: even depths are filled outlines, odd depths are holes, and the
    # filled island inside a hole (as in "®") is at depth 2.
    depths = []
    for i in range(len(contours)):
        depth, parent = 0, parents[i]
        while parent >= 0:
            depth += 1
            parent = parents[parent]
        depths.append(depth)
    # Earcut each filled outline together with its holes, its direct children.
    triangles = []
    for i, ring in enumerate(contours):
        if depths[i] % 2:
            continue
        rings = [ring] + [p for j, p in enumerate(contours) if parents[j] == i]
        vertices = np.ascontiguousarray(np.concatenate(rings), dtype=np.float32)
        ends = np.cumsum([len(p) for p in rings], dtype=np.uint32)
        indices = mapbox_earcut.triangulate_float32(vertices, ends)
        triangles.append(vertices[indices])
    return np.concatenate(triangles) if triangles else np.empty((0, 2), np.float32)


@lru_cache(maxsize=2048)
def _glyph(path, glyph_id, size):
    """Triangles (3k, 2) and outline segments (m, 6) of one glyph, relative to its pen origin.

    Segment rows are start x, y, end x, y, and the start and end of the segment as
    fractions of the glyph's total outline length, so Write traces the glyph's contours
    one after another at a constant speed.
    """
    font, _, upem = _font(path)
    glyphs = font.getGlyphSet()
    pen = _FlattenPen(glyphs, size / upem)
    glyphs[font.getGlyphOrder()[glyph_id]].draw(pen)
    # Drop zero-area contours, which have nothing to fill.
    contours = [p for p in pen.contours if abs(_area(p)) > 1e-8]
    triangles = _triangulate(contours)
    segments = []
    for points in contours:
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            length = float(np.linalg.norm(b - a))
            if length > 1e-8:
                segments.append((a, b, length))
    total = sum(s[2] for s in segments)
    strokes = []
    distance = 0.0
    for a, b, length in segments:
        strokes.append([*a, *b, distance / total, (distance + length) / total])
        distance += length
    return triangles, np.asarray(strokes, np.float32).reshape(-1, 6)


@lru_cache(maxsize=256)
def layout_text(text, size=36.0, font="regular", line_spacing=1.3, ligatures=True):
    """Shape ``text`` (lines split at newlines) at ``size`` design pixels; return TextGeometry.

    ``line_spacing`` is the baseline-to-baseline distance in multiples of ``size``.
    Results are cached, so their arrays are read-only.
    """
    path = font_path(font)
    _, shaped, upem = _font(path)
    fills, outlines, clusters, advances = [], [], [], []
    glyph_count, source_offset = 0, 0
    for line_number, line in enumerate(text.split("\n")):
        buf = hb.Buffer()
        buf.add_str(line)
        buf.guess_segment_properties()
        hb.shape(shaped, buf, {"liga": ligatures, "clig": ligatures})
        # Pen position of the line's first glyph. After shaping, info.codepoint is a glyph
        # id, and glyph 0 is .notdef, the font's "missing glyph" box.
        x, y = 0.0, line_number * size * line_spacing
        for info, position in zip(buf.glyph_infos or [], buf.glyph_positions or []):
            if info.codepoint == 0:
                raise ValueError(
                    f"Font does not contain a glyph for text near character {source_offset + info.cluster}"
                )
            triangles, segments = _glyph(path, info.codepoint, size)
            # HarfBuzz offsets are in font units with y up.
            offset = np.array([x + position.x_offset * size / upem, y - position.y_offset * size / upem])
            # Glyphs without an outline, such as spaces, only advance the pen.
            if len(segments):
                fill = np.column_stack((triangles + offset, np.full(len(triangles), glyph_count)))
                stroke = np.zeros((len(segments), 8), np.float32)
                stroke[:, :6] = segments
                stroke[:, :2] += offset
                stroke[:, 2:4] += offset
                stroke[:, 6] = glyph_count
                fills.append(fill)
                outlines.append(stroke)
                clusters.append(source_offset + info.cluster)
                glyph_count += 1
            x += position.x_advance * size / upem
            y -= position.y_advance * size / upem
        advances.append(abs(x))
        # The next line's characters start after this line and its newline.
        source_offset += len(line) + 1
    fill = np.concatenate(fills).astype(np.float32) if fills else np.empty((0, 3), np.float32)
    strokes = np.concatenate(outlines) if outlines else np.empty((0, 8), np.float32)
    if len(strokes):
        # Shift the ink's top edge to y = 0. Horizontally, keep the pen origin at x = 0
        # unless ink reaches left of it.
        points = np.concatenate((strokes[:, :2], strokes[:, 2:4]))
        lo, hi = points.min(0), points.max(0)
        lo[0] = min(lo[0], 0)
        fill[:, :2] -= lo
        strokes[:, :2] -= lo
        strokes[:, 2:4] -= lo
        width, height = max(hi[0], max(advances)) - lo[0], hi[1] - lo[1]
    else:
        width, height = max(advances, default=0), 0.0
    fill.flags.writeable = strokes.flags.writeable = False
    return TextGeometry(fill, strokes, float(width), float(height), glyph_count, tuple(clusters))
