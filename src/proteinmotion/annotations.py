"""Vector text, screen-facing residue labels, and callouts anchored to live regions."""

from dataclasses import dataclass

import gemmi
import numpy as np

from .animation import Animation
from .math3d import color as parse_color
from .rates import linear, smooth
from .regions import Region
from .text_geometry import layout_text


def _point2(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError(f"{name} needs two finite values")
    return value.copy()


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def project_region(region, camera, width, height):
    """Project a region's world-space centroid; reject targets outside the view frustum."""
    world = region.world_positions.mean(0)
    clip = camera.matrix(width / height) @ np.r_[world, 1.0]
    if clip[3] <= 1e-8:
        return None
    xyz = clip[:3] / clip[3]
    if not (0 <= xyz[2] <= 1) or np.any(np.abs(xyz[:2]) > 1):
        return None
    return np.array([(xyz[0] + 1) * width / 2, (1 - xyz[1]) * height / 2])


@dataclass
class Placement:
    text: object
    origin: np.ndarray
    glyph_offset: int = 0
    opacity: float = 1.0


@dataclass
class Leader:
    points: np.ndarray
    color: np.ndarray
    width: float
    opacity: float


@dataclass
class AnnotationLayout:
    text: list[Placement]
    leaders: list[Leader]
    bounds: np.ndarray | None = None
    anchor: np.ndarray | None = None


class Annotation:
    """A seekable screen overlay. Positions are normalized viewport coordinates."""

    _screen_fixed = True

    def __init__(self):
        self.opacity = 1.0
        self.position = np.zeros(2)
        self._write = 1.0
        self._lag_ratio = 0.0
        self._stroke_width = 1.5
        self._reverse = False

    def set_opacity(self, opacity):
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("Opacity must be in [0, 1]")
        self.opacity = float(opacity)
        return self

    def move_to(self, position):
        if not self._screen_fixed:
            raise ValueError("Residue labels follow atoms; use set_offset() on a ResidueLabel")
        self.position = _point2(position, "position")
        return self

    def shift(self, offset):
        if not self._screen_fixed:
            raise ValueError("Residue labels follow atoms; use set_offset() on a ResidueLabel")
        self.position += _point2(offset, "offset")
        return self

    def snapshot(self):
        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in vars(self).items()
        }

    def restore(self, state):
        for name, value in state.items():
            setattr(self, name, value.copy() if isinstance(value, np.ndarray) else value)

    @property
    def animate(self):
        return AnnotationAnimate(self)

    @property
    def text_progress(self):
        return self._write


class Text(Annotation):
    """HarfBuzz-shaped vector text, sized in pixels at a 1080-pixel render height.

    ``position=(x, y)`` uses the viewport's top-left as (0, 0), bottom-right as
    (1, 1). Text remains screen-facing through camera and protein motion.
    """

    def __init__(
        self,
        text,
        *,
        font_size=36,
        font="regular",
        color="#edf3fc",
        position=(0.06, 0.08),
        align="left",
        line_spacing=1.3,
        ligatures=True,
        opacity=1.0,
    ):
        super().__init__()
        if not isinstance(text, str):
            raise TypeError("Text content must be a string")
        if align not in ("left", "center", "right"):
            raise ValueError("align must be left, center, or right")
        self.text, self.font = text, font
        self.font_size = _positive(font_size, "font_size")
        self.color = parse_color(color)
        self.align, self._vertical_anchor = align, 0
        self.position = _point2(position, "position")
        self.set_opacity(opacity)
        self.geometry = layout_text(
            text, self.font_size, font, _positive(line_spacing, "line_spacing"), bool(ligatures)
        )

    @property
    def glyph_count(self):
        return self.geometry.glyph_count

    def to_corner(self, corner="UL", *, buff=0.06):
        if corner not in ("UL", "UR", "DL", "DR") or not np.isfinite(buff) or not 0 <= buff < 0.5:
            raise ValueError("Use UL, UR, DL, or DR with buff in [0, 0.5)")
        self.align = "left" if corner[1] == "L" else "right"
        self._vertical_anchor = 0 if corner[0] == "U" else 1
        return self.move_to((buff if corner[1] == "L" else 1 - buff, buff if corner[0] == "U" else 1 - buff))

    def layout(self, camera, width, height):
        scale = height / 1080
        size = np.array([self.geometry.width, self.geometry.height]) * scale
        origin = self.position * [width, height]
        origin -= size * [{"left": 0, "center": 0.5, "right": 1}[self.align], self._vertical_anchor]
        return AnnotationLayout([Placement(self, origin)], [], np.r_[origin, origin + size])


def _partial_path(points, progress):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    remaining = lengths.sum() * np.clip(progress, 0, 1)
    output = [points[0]]
    for a, b, length in zip(points, points[1:], lengths):
        if remaining <= 0:
            break
        output.append(a + (b - a) * min(1, remaining / max(length, 1e-8)))
        remaining -= length
    return np.asarray(output)


class Callout(Annotation):
    """Screen-fixed text with a leader tracking a live 3D region centroid."""

    def __init__(
        self,
        region,
        text,
        *,
        subtitle=None,
        position=(0.73, 0.25),
        font_size=32,
        font="semibold",
        color="#edf3fc",
        subtitle_color="#a3b3c7",
        line_color=None,
        line_width=1.5,
        tip="dot",
        clamp=True,
        follow_opacity=True,
        opacity=1.0,
    ):
        super().__init__()
        if not isinstance(region, Region):
            raise TypeError("Callout needs a Region")
        if tip not in ("dot", "arrow", "none"):
            raise ValueError("tip must be dot, arrow, or none")
        self.region = region
        self.title = Text(text, font_size=font_size, font=font, color=color)
        self.subtitle = (
            None if subtitle is None else Text(subtitle, font_size=font_size * 0.68, color=subtitle_color)
        )
        self.line_color = parse_color(line_color if line_color is not None else color)
        self.line_width = _positive(line_width, "line_width")
        self.tip, self.clamp, self.follow_opacity = tip, bool(clamp), bool(follow_opacity)
        self.position = _point2(position, "position")
        self.set_opacity(opacity)

    @property
    def glyph_count(self):
        return self.title.glyph_count + (self.subtitle.glyph_count if self.subtitle else 0)

    @property
    def text_progress(self):
        # Draw the leader first, then overlap its finish with the first letters.
        return float(np.clip((self._write - 0.15) / 0.85, 0, 1))

    def _origin(self, anchor, width, height, offset=None):
        return self.position * [width, height]

    def layout(self, camera, width, height, *, offset=None, progress=None):
        anchor = project_region(self.region, camera, width, height)
        if anchor is None:
            return AnnotationLayout([], [])
        scale = height / 1080
        first = self.title.geometry
        sub = self.subtitle.geometry if self.subtitle else None
        gap = self.title.font_size * 0.42 if sub else 0
        size = (
            np.array(
                [max(first.width, sub.width if sub else 0), first.height + gap + (sub.height if sub else 0)]
            )
            * scale
        )
        origin = self._origin(anchor, width, height, offset)
        if self.clamp:
            margin = 14 * scale
            origin = np.maximum(
                margin, np.minimum(origin, np.maximum(margin, np.array([width, height]) - size - margin))
            )
        visibility = (
            float(self.region.protein.atom_opacities[self.region.atom_indices].mean())
            if self.follow_opacity
            else 1.0
        )
        placements = [Placement(self.title, origin, 0, visibility)]
        if self.subtitle:
            placements.append(
                Placement(
                    self.subtitle, origin + [0, (first.height + gap) * scale], first.glyph_count, visibility
                )
            )
        side = -1 if anchor[0] < origin[0] + size[0] / 2 else 1
        attach = np.array(
            [origin[0] + (size[0] if side > 0 else 0) + side * 12 * scale, origin[1] + size[1] / 2]
        )
        elbow = attach + [side * 26 * scale, 0]
        progress = self._write if progress is None else progress
        leader_progress = float(np.clip(progress / 0.3, 0, 1))
        path = _partial_path(np.array([attach, elbow, anchor]), leader_progress)
        leaders = (
            [Leader(path, self.line_color, self.line_width * scale, visibility)] if len(path) > 1 else []
        )
        if leader_progress >= 1 and self.tip != "none":
            if self.tip == "dot":
                angles = np.linspace(0, 2 * np.pi, 25)
                points = anchor + 3.5 * scale * np.column_stack((np.cos(angles), np.sin(angles)))
            else:
                direction = anchor - elbow
                direction /= max(np.linalg.norm(direction), 1e-8)
                perpendicular = np.array([-direction[1], direction[0]])
                points = np.array(
                    [
                        anchor - 9 * scale * direction + 4 * scale * perpendicular,
                        anchor,
                        anchor - 9 * scale * direction - 4 * scale * perpendicular,
                    ]
                )
            leaders.append(Leader(points, self.line_color, self.line_width * scale, visibility))
        return AnnotationLayout(placements, leaders, np.r_[origin, origin + size], anchor)


def residue_name(region, format="three_letter", include_chain=True):
    if len(region.residue_indices) != 1:
        raise ValueError("An automatic amino-acid label needs exactly one residue")
    residue = region.protein.topology.residues[region.residue_indices[0]]
    if format == "three_letter":
        name = residue.name.title()
    elif format == "one_letter":
        name = gemmi.find_tabulated_residue(residue.name).one_letter_code.upper() or "X"
    else:
        raise ValueError("format must be three_letter or one_letter")
    prefix = f"{residue.chain} · " if include_chain else ""
    return f"{prefix}{name} {residue.resid}{residue.icode}"


class ResidueLabel(Callout):
    """A label following one residue, with an offset in 1080p design pixels."""

    _screen_fixed = False

    def __init__(
        self,
        region,
        text=None,
        *,
        offset=(24, -36),
        format="three_letter",
        include_chain=True,
        font_size=26,
        **kwargs,
    ):
        if not isinstance(region, Region):
            raise TypeError("ResidueLabel needs a Region")
        self.offset = _point2(offset, "offset")
        if len(region.residue_indices) != 1:
            raise ValueError("ResidueLabel needs exactly one residue")
        residue = region.protein.topology.residues[region.residue_indices[0]]
        if residue.ca >= 0:
            region = Region(region.protein, [residue.ca])
        super().__init__(
            region,
            residue_name(region, format, include_chain) if text is None else text,
            font_size=font_size,
            **kwargs,
        )

    def _origin(self, anchor, width, height, offset=None):
        return anchor + (self.offset if offset is None else offset) * height / 1080

    def set_offset(self, offset):
        self.offset = _point2(offset, "offset")
        return self


class ResidueLabels(Annotation):
    """A small set of amino-acid labels with deterministic overlap avoidance."""

    _screen_fixed = False

    def __init__(self, region, *, offsets=None, avoid_overlap=True, **kwargs):
        super().__init__()
        if not isinstance(region, Region):
            raise TypeError("ResidueLabels needs a Region")
        self.labels, self.fixed = [], []
        for index in region.residue_indices:
            residue = region.protein.topology.residues[index]
            if residue.ca < 0:
                continue
            options = dict(kwargs)
            explicit = (offsets or {}).get(
                (residue.chain, residue.resid, residue.icode), (offsets or {}).get(residue.resid)
            )
            if explicit is not None:
                options["offset"] = explicit
            self.labels.append(ResidueLabel(Region(region.protein, [residue.ca]), **options))
            self.fixed.append(explicit is not None)
        if not self.labels:
            raise ValueError("Selection has no amino-acid Cα atoms to label")
        self.labels, self.fixed = tuple(self.labels), tuple(self.fixed)
        self.avoid_overlap = bool(avoid_overlap)

    @property
    def glyph_count(self):
        return sum(label.glyph_count for label in self.labels)

    @property
    def text_progress(self):
        return float(np.clip((self._write - 0.15) / 0.85, 0, 1))

    def layout(self, camera, width, height):
        placements, leaders, boxes = [], [], []
        glyph_offset = 0
        scale = height / 1080
        for label, fixed in zip(self.labels, self.fixed):
            layout = label.layout(camera, width, height, progress=self._write)
            if layout.bounds is not None and self.avoid_overlap and not fixed:
                width_local, height_local = (layout.bounds[2:] - layout.bounds[:2]) / scale
                candidates = [
                    label.offset,
                    (24, -height_local - 24),
                    (-width_local - 24, -height_local - 24),
                    (24, 28),
                    (-width_local - 24, 28),
                    (56, -height_local / 2),
                    (-width_local - 56, -height_local / 2),
                    (24, -height_local - 85),
                    (24, 85),
                    (-width_local - 24, -height_local - 85),
                ]
                best, best_score = layout, float("inf")
                for candidate in candidates:
                    trial = label.layout(
                        camera, width, height, offset=np.asarray(candidate), progress=self._write
                    )
                    box = trial.bounds
                    score = sum(
                        float(
                            np.prod(
                                np.maximum(
                                    0,
                                    np.minimum(box[2:] + 6 * scale, b[2:])
                                    - np.maximum(box[:2] - 6 * scale, b[:2]),
                                )
                            )
                        )
                        for b in boxes
                    )
                    if score < best_score:
                        best, best_score = trial, score
                    if score == 0:
                        break
                layout = best
            if layout.bounds is not None:
                boxes.append(layout.bounds)
            for placement in layout.text:
                placement.glyph_offset += glyph_offset
                placement.opacity *= label.opacity
            for leader in layout.leaders:
                leader.opacity *= label.opacity
            placements.extend(layout.text)
            leaders.extend(layout.leaders)
            glyph_offset += label.glyph_count
        return AnnotationLayout(placements, leaders)


class Write(Animation):
    """Draw glyph contours, then fill them, with Manim-style lagged timing.

    Timing is adapted from Manim's MIT-licensed Write/DrawBorderThenFill. See
    licenses/Manim-LICENSE.txt and THIRD_PARTY.md. Rendering is native Metal.
    """

    channels = frozenset({"write"})

    def __init__(self, target, *, lag_ratio=None, stroke_width=1.5, reverse=False, rate_func=linear):
        if not isinstance(target, Annotation):
            raise TypeError("Write needs Text, Callout, or ResidueLabels")
        super().__init__(target, rate_func=rate_func)
        count = target.glyph_count
        self.lag_ratio = min(4.0 / max(1, count), 0.2) if lag_ratio is None else float(lag_ratio)
        if not np.isfinite(self.lag_ratio) or self.lag_ratio < 0:
            raise ValueError("lag_ratio must be finite and nonnegative")
        self.stroke_width = _positive(stroke_width, "stroke_width")
        self.reverse = bool(reverse)

    def apply(self, alpha):
        self.target._write = float(alpha)
        self.target._lag_ratio = self.lag_ratio
        self.target._stroke_width = self.stroke_width
        self.target._reverse = self.reverse


class Unwrite(Write):
    """Erase filled text back through its outlines, last glyph first by default."""

    def apply(self, alpha):
        super().apply(1 - alpha)


class AnnotationAnimate(Animation):
    """Animate normalized screen position and/or opacity without changing glyph meshes."""

    def __init__(self, target):
        super().__init__(target, rate_func=smooth)
        self.operations, self.channels = [], frozenset()

    def move_to(self, position):
        if isinstance(self.target, (ResidueLabel, ResidueLabels)):
            raise ValueError("Residue labels follow atoms; configure their pixel offsets instead")
        self.operations.append(("position", _point2(position, "position")))
        self.channels |= {"screen_position"}
        return self

    def shift(self, offset):
        current = next(
            (end for name, end in self.operations[::-1] if name == "position"), self.target.position
        )
        return self.move_to(current + _point2(offset, "offset"))

    def set_offset(self, offset):
        if not isinstance(self.target, ResidueLabel):
            raise TypeError("set_offset() needs a single ResidueLabel")
        self.operations.append(("offset", _point2(offset, "offset")))
        self.channels |= {"label_offset"}
        return self

    def set_opacity(self, value):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Opacity must be in [0, 1]")
        self.operations.append(("opacity", float(value)))
        self.channels |= {"opacity"}
        return self

    def bind(self):
        super().bind()
        self.start = self.target.snapshot()

    def apply(self, alpha):
        for name, end in self.operations:
            setattr(self.target, name, (1 - alpha) * self.start[name] + alpha * end)
