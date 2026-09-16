"""Shaped font outlines, cached tessellation, and arc-length stroke geometry.

HarfBuzz lays out glyphs; FontTools supplies quadratic/cubic curves. No bitmap
text is rasterized per frame. Coordinates are design pixels at a 1080-pixel height.
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
    fill: np.ndarray
    strokes: np.ndarray
    width: float
    height: float
    glyph_count: int
    clusters: tuple[int, ...]


def font_path(font):
    if font in ("regular", "semibold"):
        name = "Regular" if font == "regular" else "Semibold"
        return str(files("proteinmotion").joinpath(f"fonts/SourceSans3-{name}.otf"))
    path = Path(font).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Font not found: {path}")
    return str(path)


@lru_cache(maxsize=16)
def _font(path):
    font = TTFont(path, fontNumber=0)
    face = hb.Face(Path(path).read_bytes())
    shaped = hb.Font(face)
    shaped.scale = (face.upem, face.upem)
    hb.ot_font_set_funcs(shaped)
    return font, shaped, face.upem


class _FlattenPen(BasePen):
    def __init__(self, glyphs, scale, tolerance=0.075):
        super().__init__(glyphs)
        self.scale, self.tolerance = scale, tolerance
        self.contours, self.current = [], []

    def _point(self, p):
        return np.array([p[0], -p[1]], float) * self.scale

    def _moveTo(self, p):
        if self.current:
            self._closePath()
        self.current = [self._point(p)]

    def _lineTo(self, p):
        self.current.append(self._point(p))

    def _cubic(self, a, b, c, d, depth=0):
        chord = d - a
        length = np.linalg.norm(chord)
        if length < 1e-10:
            error = max(np.linalg.norm(b - a), np.linalg.norm(c - a))
        else:
            error = max(abs(chord[0] * (p - a)[1] - chord[1] * (p - a)[0]) / length for p in (b, c))
        if error <= self.tolerance or depth >= 14:
            self.current.append(d)
            return
        ab, bc, cd = (a + b) / 2, (b + c) / 2, (c + d) / 2
        abc, bcd = (ab + bc) / 2, (bc + cd) / 2
        middle = (abc + bcd) / 2
        self._cubic(a, ab, abc, middle, depth + 1)
        self._cubic(middle, bcd, cd, d, depth + 1)

    def _curveToOne(self, p1, p2, p3):
        self._cubic(self.current[-1], self._point(p1), self._point(p2), self._point(p3))

    def _qCurveToOne(self, p1, p2):
        a, b, c = self.current[-1], self._point(p1), self._point(p2)
        self._cubic(a, a + 2 * (b - a) / 3, c + 2 * (b - c) / 3, c)

    def _closePath(self):
        if len(self.current) >= 3:
            points = np.array(self.current)
            keep = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) > 1e-8
            points = points[keep]
            if len(points) >= 3:
                self.contours.append(points)
        self.current = []

    def _endPath(self):
        self._closePath()


def _area(points):
    other = np.roll(points, -1, axis=0)
    return float(np.sum(points[:, 0] * other[:, 1] - other[:, 0] * points[:, 1]) / 2)


def _inside(point, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    crossing = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    denominator = np.where(crossing, b[:, 1] - a[:, 1], 1)
    x = a[:, 0] + (point[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / denominator
    return bool(np.count_nonzero(crossing & (point[0] < x)) % 2)


def _triangulate(contours):
    """Group outer rings and holes; disconnected accent components stay separate."""
    if not contours:
        return np.empty((0, 2), np.float32)
    areas = np.array([abs(_area(p)) for p in contours])
    parents = []
    for i, ring in enumerate(contours):
        enclosing = [j for j, outer in enumerate(contours) if areas[j] > areas[i] and _inside(ring[0], outer)]
        parents.append(min(enclosing, key=lambda j: areas[j]) if enclosing else -1)
    depths = []
    for i in range(len(contours)):
        depth, parent = 0, parents[i]
        while parent >= 0:
            depth += 1
            parent = parents[parent]
        depths.append(depth)
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
    font, _, upem = _font(path)
    glyphs = font.getGlyphSet()
    pen = _FlattenPen(glyphs, size / upem)
    glyphs[font.getGlyphOrder()[glyph_id]].draw(pen)
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
    path = font_path(font)
    _, shaped, upem = _font(path)
    fills, outlines, clusters, advances = [], [], [], []
    glyph_count, source_offset = 0, 0
    for line_number, line in enumerate(text.split("\n")):
        buf = hb.Buffer()
        buf.add_str(line)
        buf.guess_segment_properties()
        hb.shape(shaped, buf, {"liga": ligatures, "clig": ligatures})
        x, y = 0.0, line_number * size * line_spacing
        for info, position in zip(buf.glyph_infos or [], buf.glyph_positions or []):
            if info.codepoint == 0:
                raise ValueError(
                    f"Font does not contain a glyph for text near character {source_offset + info.cluster}"
                )
            triangles, segments = _glyph(path, info.codepoint, size)
            offset = np.array([x + position.x_offset * size / upem, y - position.y_offset * size / upem])
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
        source_offset += len(line) + 1
    fill = np.concatenate(fills).astype(np.float32) if fills else np.empty((0, 3), np.float32)
    strokes = np.concatenate(outlines) if outlines else np.empty((0, 8), np.float32)
    if len(strokes):
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
