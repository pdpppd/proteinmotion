"""Vector text, screen-facing residue labels, and callouts anchored to live regions.

Annotation is the base class of every 2D overlay: the classes here, the plots in
``plots.py``, and the labels of distances, torsions and interactions. An annotation is a
scene object like a Protein. Scene.seek() restores its authoring state and replays the
animations up to the requested time (Write, Unwrite and ``annotation.animate``, defined
below). The renderer's OverlayRenderer (``overlay.py``) then calls
``layout(camera, width, height)`` on each visible annotation and draws the returned
AnnotationLayout over the finished 3D image.

Coordinates:

- ``width`` and ``height`` are the render target in pixels, supersampling included.
  Layouts are in pixels with (0, 0) at the top-left and y pointing down.
- ``position`` is normalized: (0, 0) is the viewport's top-left, (1, 1) its bottom-right.
- Font sizes, offsets, gaps and line widths are 1080p design pixels, multiplied by
  ``height / 1080`` so that overlays keep their proportions at any output size.
- Labels anchored in 3D project a Region's world-space centroid through the camera with
  project_region().

Write progress is stored on the annotation as ``_write`` in [0, 1]. ``text_progress``
is the part of it that drives the glyphs, which ``shaders/text.wgsl`` staggers glyph by
glyph into outlines that draw and then fill.
"""

from dataclasses import dataclass

import gemmi
import numpy as np

from .animation import Animation
from .math3d import color as parse_color
from .rates import linear, smooth
from .regions import Region
from .text_geometry import layout_text


def _point2(value, name):
    """Check a 2-vector (a normalized position or a pixel offset) and return a float copy."""
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
    # Homogeneous clip coordinates. w <= 0 means the point is at or behind the camera.
    clip = camera.matrix(width / height) @ np.r_[world, 1.0]
    if clip[3] <= 1e-8:
        return None
    # Normalized device coordinates: x and y in [-1, 1] and WebGPU depth in [0, 1]
    # (near to far plane) inside the view.
    xyz = clip[:3] / clip[3]
    if not (0 <= xyz[2] <= 1) or np.any(np.abs(xyz[:2]) > 1):
        return None
    # NDC y points up while pixel rows count down, hence the flip.
    return np.array([(xyz[0] + 1) * width / 2, (1 - xyz[1]) * height / 2])


@dataclass
class Placement:
    """One Text drawn with its top-left corner at ``origin`` (pixels).

    ``glyph_offset`` is the rank of the text's first glyph within the whole annotation,
    so Write staggers glyphs across all of an annotation's texts. ``opacity`` multiplies
    the annotation's own opacity.
    """

    text: object
    origin: np.ndarray
    glyph_offset: int = 0
    opacity: float = 1.0


@dataclass
class Leader:
    """A line with round ends: a polyline (N, 2) or N separate segments (N, 2, 2), in pixels.

    ``color`` is RGB in [0, 1] and ``width`` is in pixels.
    """

    points: np.ndarray
    color: np.ndarray
    width: float
    opacity: float


@dataclass
class AnnotationLayout:
    """Everything an annotation draws in one frame, in pixels of the render target.

    ``bounds`` is the text box (x0, y0, x1, y1), used to avoid overlaps. ``anchor`` is
    the projected 3D point a callout points at. ``triangles`` is an (M, 6) array of
    filled triangle vertices with columns x, y, r, g, b, a, used by plots.
    """

    text: list[Placement]
    leaders: list[Leader]
    bounds: np.ndarray | None = None
    anchor: np.ndarray | None = None
    triangles: np.ndarray | None = None


class Annotation:
    """A seekable screen overlay. Positions are normalized viewport coordinates."""

    # False for labels that follow atoms; their screen position comes from the projection.
    _screen_fixed = True

    def __init__(self):
        self.opacity = 1.0
        self.position = np.zeros(2)
        # Write state, set by Write.apply() and read by the overlay each frame. A new
        # annotation is fully written, so it shows without a Write.
        self._write = 1.0
        self._lag_ratio = 0.0
        self._stroke_width = 1.5
        self._reverse = False

    def set_opacity(self, opacity):
        """Set opacity in [0, 1]."""
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("Opacity must be in [0, 1]")
        self.opacity = float(opacity)
        return self

    def move_to(self, position):
        """Place the annotation at ``position`` = (x, y), normalized: (0, 0) top-left, (1, 1) bottom-right."""
        if not self._screen_fixed:
            raise ValueError("Residue labels follow atoms; use set_offset() on a ResidueLabel")
        self.position = _point2(position, "position")
        return self

    def shift(self, offset):
        """Move the annotation by ``offset`` = (dx, dy) in normalized viewport units."""
        if not self._screen_fixed:
            raise ValueError("Residue labels follow atoms; use set_offset() on a ResidueLabel")
        self.position += _point2(offset, "offset")
        return self

    def snapshot(self):
        """Capture every attribute, so the scene can rewind to this state when it seeks."""
        # Arrays are copied so that in-place edits, such as shift() on position, cannot
        # change a saved state.
        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in vars(self).items()
        }

    def restore(self, state):
        """Return to a state captured by snapshot()."""
        # Copy again so that later in-place edits cannot change the saved state.
        for name, value in state.items():
            setattr(self, name, value.copy() if isinstance(value, np.ndarray) else value)

    @property
    def animate(self):
        """A builder for animated changes; pass it to scene.play()."""
        return AnnotationAnimate(self)

    @property
    def text_progress(self):
        """Write progress of the glyphs in [0, 1]. Callouts delay it so their leader draws first."""
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
        font_size=None,
        font=None,
        color=None,
        position=(0.06, 0.08),
        align="left",
        line_spacing=None,
        ligatures=True,
        opacity=1.0,
        style=None,
        max_width=None,
    ):
        from .styles import TextStyle

        style = TextStyle() if style is None else style
        if not isinstance(style, TextStyle):
            raise TypeError("style must be a TextStyle")
        font_size = style.font_size if font_size is None else font_size
        font = style.font if font is None else font
        color = style.color if color is None else color
        line_spacing = style.line_spacing if line_spacing is None else line_spacing
        self.max_width = None if max_width is None else _positive(max_width, "max_width")
        super().__init__()
        if not isinstance(text, str):
            raise TypeError("Text content must be a string")
        if align not in ("left", "center", "right"):
            raise ValueError("align must be left, center, or right")
        self.text, self.font = text, font
        self.font_size = _positive(font_size, "font_size")
        self.color = parse_color(color)
        # _vertical_anchor is the fraction of the text height above ``position``: 0 puts the
        # top edge there, 1 the bottom edge (set by to_corner for the lower corners).
        self.align, self._vertical_anchor = align, 0
        self.position = _point2(position, "position")
        self.set_opacity(opacity)
        self.line_spacing = _positive(line_spacing, "line_spacing")
        self.ligatures = bool(ligatures)
        self.geometry = self._layout_text(text)

    def _layout_text(self, text):
        """Shaped glyph geometry for ``text``, wrapped greedily at word breaks to ``max_width``."""

        # layout_text is memoized, so measuring a trial line again is cheap.
        def measure(value):
            return layout_text(value, self.font_size, self.font, self.line_spacing, self.ligatures)

        if self.max_width is None:
            return measure(text)
        # Fill each line with as many words as fit; explicit newlines start new paragraphs.
        lines = []
        for paragraph in text.split("\n"):
            current = ""
            for word in paragraph.split():
                trial = f"{current} {word}" if current else word
                if current and measure(trial).width > self.max_width:
                    lines.append(current)
                    current = word
                else:
                    current = trial
                # Break oversized words so max_width remains a real constraint.
                while len(current) > 1 and measure(current).width > self.max_width:
                    end = 1
                    while end < len(current) and measure(current[: end + 1]).width <= self.max_width:
                        end += 1
                    lines.append(current[:end])
                    current = current[end:]
            lines.append(current)
        return measure("\n".join(lines))

    def set_text(self, text):
        """Replace the string. The glyphs are shaped again only when the string changes."""
        if not isinstance(text, str):
            raise TypeError("Text content must be a string")
        if text != self.text:
            self.text = text
            self.geometry = self._layout_text(text)
        return self

    @property
    def glyph_count(self):
        return self.geometry.glyph_count

    def to_corner(self, corner="UL", *, buff=0.06):
        """Align the text to a viewport corner: UL, UR, DL or DR.

        ``buff`` is the distance from both edges as a fraction of the viewport width and height.
        """
        if corner not in ("UL", "UR", "DL", "DR") or not np.isfinite(buff) or not 0 <= buff < 0.5:
            raise ValueError("Use UL, UR, DL, or DR with buff in [0, 0.5)")
        self.align = "left" if corner[1] == "L" else "right"
        self._vertical_anchor = 0 if corner[0] == "U" else 1
        return self.move_to((buff if corner[1] == "L" else 1 - buff, buff if corner[0] == "U" else 1 - buff))

    def layout(self, camera, width, height):
        """Place the text for a ``width`` × ``height`` pixel frame. Text ignores the camera."""
        scale = height / 1080
        size = np.array([self.geometry.width, self.geometry.height]) * scale
        # ``position`` marks the left, center or right edge (by align) and the top or bottom
        # edge (by _vertical_anchor); shift back to the text's top-left corner.
        origin = self.position * [width, height]
        origin -= size * [{"left": 0, "center": 0.5, "right": 1}[self.align], self._vertical_anchor]
        return AnnotationLayout([Placement(self, origin)], [], np.r_[origin, origin + size])


def _partial_path(points, progress):
    """The leading part of a polyline (N, 2) that covers ``progress`` (0 to 1) of its length."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    remaining = lengths.sum() * np.clip(progress, 0, 1)
    output = [points[0]]
    for a, b, length in zip(points, points[1:], lengths):
        if remaining <= 0:
            break
        # Whole segments while length remains, then part of the segment where it runs out.
        # The 1e-8 floor guards against zero-length segments (repeated points).
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
        # The leader takes the first 30% of the Write (see layout); text starts at 15%.
        return float(np.clip((self._write - 0.15) / 0.85, 0, 1))

    def _origin(self, anchor, width, height, offset=None):
        """Top-left pixel of the text block. A callout stays at its screen ``position``."""
        return self.position * [width, height]

    def layout(self, camera, width, height, *, offset=None, progress=None):
        """Place the text and leader for a ``width`` × ``height`` pixel frame.

        ResidueLabels passes ``offset`` (design pixels) to try other label positions and
        ``progress`` to share its own Write progress. An off-screen target hides the callout.
        """
        anchor = project_region(self.region, camera, width, height)
        if anchor is None:
            return AnnotationLayout([], [])
        scale = height / 1080
        # Size of the text block in pixels: the title, then the subtitle below a gap of
        # 0.42 title font sizes.
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
        # Keep the block 14 design pixels inside the frame. If it is too large to fit, the
        # top-left margin wins.
        if self.clamp:
            margin = 14 * scale
            origin = np.maximum(
                margin, np.minimum(origin, np.maximum(margin, np.array([width, height]) - size - margin))
            )
        # The label fades with the atoms it points at, for example when they are ghosted.
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
        # The leader leaves the side of the block that faces the anchor, 12 design pixels out
        # at mid-height, runs 26 design pixels horizontally to an elbow, then goes straight
        # to the anchor.
        side = -1 if anchor[0] < origin[0] + size[0] / 2 else 1
        attach = np.array(
            [origin[0] + (size[0] if side > 0 else 0) + side * 12 * scale, origin[1] + size[1] / 2]
        )
        elbow = attach + [side * 26 * scale, 0]
        # The leader grows over the first 30% of the Write.
        progress = self._write if progress is None else progress
        leader_progress = float(np.clip(progress / 0.3, 0, 1))
        path = _partial_path(np.array([attach, elbow, anchor]), leader_progress)
        leaders = (
            [Leader(path, self.line_color, self.line_width * scale, visibility)] if len(path) > 1 else []
        )
        # The tip appears once the leader has reached the anchor.
        if leader_progress >= 1 and self.tip != "none":
            if self.tip == "dot":
                # A closed ring of radius 3.5 design pixels around the anchor.
                angles = np.linspace(0, 2 * np.pi, 25)
                points = anchor + 3.5 * scale * np.column_stack((np.cos(angles), np.sin(angles)))
            else:
                # An open arrowhead: two strokes, 9 design pixels back along the leader and
                # 4 to either side, meeting at the anchor.
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
    """Default label for a one-residue Region, such as ``A · Lys 48`` or ``A · K 48``.

    Nucleotides keep their file name (``DA``, ``U``) or use their one-letter base.
    """
    if len(region.residue_indices) != 1:
        raise ValueError("An automatic residue label needs exactly one residue")
    residue = region.protein.topology.residues[region.residue_indices[0]]
    if format == "three_letter":
        name = residue.name if residue.is_nucleic else residue.name.title()
    elif format == "one_letter":
        # Gemmi gives lowercase codes for modified residues (MSE is "m"), hence upper().
        name = (
            residue.base
            if residue.is_nucleic
            else gemmi.find_tabulated_residue(residue.name).one_letter_code.upper() or "X"
        )
    else:
        raise ValueError("format must be three_letter or one_letter")
    prefix = f"{residue.chain} · " if include_chain else ""
    return f"{prefix}{name} {residue.resid}{residue.icode}"


def _overlap_area(box, boxes, pad):
    """Total area shared by ``box``, grown by ``pad`` on every side, and each of ``boxes``.

    Boxes are (x0, y0, x1, y1) in pixels, so the area is in square pixels.
    """
    return sum(
        float(np.prod(np.maximum(0, np.minimum(box[2:] + pad, b[2:]) - np.maximum(box[:2] - pad, b[:2]))))
        for b in boxes
    )


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
        # Point at the backbone trace atom (Cα or C4′) when there is one, rather than at the
        # residue's centroid. Residues without one, such as ligands, keep the given region.
        if residue.trace_atom >= 0:
            region = Region(region.protein, [residue.trace_atom])
        super().__init__(
            region,
            residue_name(region, format, include_chain) if text is None else text,
            font_size=font_size,
            **kwargs,
        )

    def _origin(self, anchor, width, height, offset=None):
        """Top-left pixel of the text block: the projected atom plus the design-pixel offset."""
        return anchor + (self.offset if offset is None else offset) * height / 1080

    def set_offset(self, offset):
        """Set the offset (dx, dy) in 1080p design pixels from the projected atom to the label's top-left."""
        self.offset = _point2(offset, "offset")
        return self


class ResidueLabels(Annotation):
    """A small set of residue labels with deterministic overlap avoidance."""

    _screen_fixed = False

    def __init__(self, region, *, offsets=None, avoid_overlap=True, **kwargs):
        super().__init__()
        if not isinstance(region, Region):
            raise TypeError("ResidueLabels needs a Region")
        # ``fixed`` marks labels with an explicit offset, which overlap avoidance leaves alone.
        self.labels, self.fixed = [], []
        for index in region.residue_indices:
            residue = region.protein.topology.residues[index]
            options = dict(kwargs)
            # Offsets may be keyed by (chain, resid, icode) or by resid alone; the full key wins.
            explicit = (offsets or {}).get(
                (residue.chain, residue.resid, residue.icode), (offsets or {}).get(residue.resid)
            )
            if explicit is not None:
                options["offset"] = explicit
            # Anchor on the trace atom, or on this residue's atoms within the selection.
            ids = (
                [residue.trace_atom]
                if residue.trace_atom >= 0
                else [
                    i for i in region.atom_indices if region.protein.topology.atoms[i].residue_index == index
                ]
            )
            self.labels.append(ResidueLabel(Region(region.protein, ids), **options))
            self.fixed.append(explicit is not None)
        if not self.labels:
            raise ValueError("Selection has no residues to label")
        self.labels, self.fixed = tuple(self.labels), tuple(self.fixed)
        self.avoid_overlap = bool(avoid_overlap)

    @property
    def glyph_count(self):
        return sum(label.glyph_count for label in self.labels)

    @property
    def text_progress(self):
        # Same timing as Callout: leaders first, text from 15% of the Write.
        return float(np.clip((self._write - 0.15) / 0.85, 0, 1))

    def _avoid_overlap(self, label, layout, boxes, camera, width, height):
        """Lay ``label`` out at the first candidate offset that clears ``boxes``, else the least overlapping.

        ``layout`` is the label at its own offset and ``boxes`` are the pixel boxes of the
        labels placed before it.
        """
        scale = height / 1080
        # Offsets run from the projected atom to the label's top-left corner, in design
        # pixels, so they depend on the label's size. In order: the label's own offset,
        # above-right, above-left, below-right, below-left, level with the atom on either
        # side, then farther above or below.
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
            trial = label.layout(camera, width, height, offset=np.asarray(candidate), progress=self._write)
            # Grow the trial box by 6 design pixels so that labels keep a small gap.
            score = _overlap_area(trial.bounds, boxes, 6 * scale)
            if score < best_score:
                best, best_score = trial, score
            if score == 0:
                break
        return best

    def layout(self, camera, width, height):
        placements, leaders, boxes = [], [], []
        glyph_offset = 0
        # Labels are placed one at a time, each avoiding those placed before it. Nothing
        # carries over from earlier frames, so seeking to a time gives the same placement.
        for label, fixed in zip(self.labels, self.fixed):
            layout = label.layout(camera, width, height, progress=self._write)
            if layout.bounds is not None and self.avoid_overlap and not fixed:
                layout = self._avoid_overlap(label, layout, boxes, camera, width, height)
            if layout.bounds is not None:
                boxes.append(layout.bounds)
            # Merge into one layout: glyph ranks continue across labels so Write runs through
            # them in order, and each label's own opacity applies on top of the group's.
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

    def __init__(
        self,
        target,
        *,
        lag_ratio=None,
        stroke_width=1.5,
        reverse=False,
        rate_func=linear,
        easing=None,
        delay_seconds=None,
    ):
        if not isinstance(target, Annotation):
            raise TypeError("Write needs Text, Callout, or ResidueLabels")
        super().__init__(target, rate_func=rate_func, easing=easing)
        if delay_seconds is not None and (
            lag_ratio is not None or not np.isfinite(delay_seconds) or delay_seconds < 0
        ):
            raise ValueError("Use a nonnegative delay_seconds or lag_ratio, not both")
        self.delay_seconds = delay_seconds
        # lag_ratio is the delay between consecutive glyph starts, as a fraction of one
        # glyph's own duration (see glyph_progress in shaders/text.wgsl). The default is
        # Manim's Write default.
        count = target.glyph_count
        self.lag_ratio = min(4.0 / max(1, count), 0.2) if lag_ratio is None else float(lag_ratio)
        if not np.isfinite(self.lag_ratio) or self.lag_ratio < 0:
            raise ValueError("lag_ratio must be finite and nonnegative")
        self.stroke_width = _positive(stroke_width, "stroke_width")
        self.reverse = bool(reverse)

    def bind(self):
        super().bind()
        # Turn a fixed delay in seconds into lag_ratio. With n glyphs and run time T, each
        # glyph lasts span = T - (n - 1) * delay, so lag_ratio = delay / span. run_time is
        # known only now, when scene.play() binds the animation.
        if self.delay_seconds is not None:
            duration = getattr(self, "run_time", 1.0)
            span = duration - max(0, self.target.glyph_count - 1) * self.delay_seconds
            if span <= 0:
                raise ValueError("Glyph delays exceed run_time; reduce delay_seconds or increase run_time")
            self.lag_ratio = self.delay_seconds / span

    def apply(self, alpha):
        # Store the state on the target; the overlay passes it to the text shader each frame.
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
        # Queued (attribute, end value) pairs. The channels grow with them so the scene can
        # reject two concurrent animations that write the same attribute.
        self.operations, self.channels = [], frozenset()

    def move_to(self, position):
        if isinstance(self.target, (ResidueLabel, ResidueLabels)):
            raise ValueError("Residue labels follow atoms; configure their pixel offsets instead")
        self.operations.append(("position", _point2(position, "position")))
        self.channels |= {"screen_position"}
        return self

    def shift(self, offset):
        # Shift from the last queued move in this builder, or from where the target is now.
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

    def set_text(self, text):
        """Replace caption text at this animation's midpoint, reproducibly when seeking."""
        if not isinstance(self.target, Text) or not isinstance(text, str):
            raise TypeError("set_text needs a Text target and a string")
        self.operations.append(("text", text))
        self.channels |= {"text"}
        return self

    def bind(self):
        super().bind()
        self.start = self.target.snapshot()

    def apply(self, alpha):
        # Interpolate linearly from the state at the start of the clip. Text cannot be
        # interpolated, so it switches at the midpoint.
        for name, end in self.operations:
            if name == "text":
                self.target.set_text(self.start["text"] if alpha < 0.5 else end)
            else:
                setattr(self.target, name, (1 - alpha) * self.start[name] + alpha * end)


class TextGroup(Annotation):
    """Stack Text objects vertically and animate their layout, writing, or opacity together.

    position is normalized screen space; gap is in 1080p design pixels.
    Child positions are ignored; child fonts, colors and alignments are retained.
    """

    def __init__(self, *texts, position=(0.06, 0.08), gap=16):
        super().__init__()
        if not texts or any(not isinstance(t, Text) for t in texts):
            raise TypeError("TextGroup needs one or more Text objects")
        if not np.isfinite(gap) or gap < 0:
            raise ValueError("gap must be finite and nonnegative design pixels")
        self.texts, self.position, self.gap = tuple(texts), _point2(position, "position"), float(gap)

    @property
    def glyph_count(self):
        return sum(t.glyph_count for t in self.texts)

    def snapshot(self):
        # The child Texts are drawn through the group rather than added to the scene, so the
        # group saves and restores their state with its own. Edits to a child between play()
        # calls then survive seeking.
        state = super().snapshot()
        state["children"] = tuple(t.snapshot() for t in self.texts)
        return state

    def restore(self, state):
        super().restore({k: v for k, v in state.items() if k != "children"})
        for text, saved in zip(self.texts, state["children"]):
            text.restore(saved)

    def layout(self, camera, width, height):
        # Stack the texts downward from ``position``. ``cursor`` is the next top edge in
        # pixels; each text aligns horizontally on the group's x by its own ``align``.
        origin = self.position * [width, height]
        scale, cursor, offset = height / 1080, float(origin[1]), 0
        placements, boxes = [], []
        for text in self.texts:
            size = np.array([text.geometry.width, text.geometry.height]) * scale
            x = origin[0] - size[0] * {"left": 0, "center": 0.5, "right": 1}[text.align]
            point = np.array([x, cursor])
            placements.append(Placement(text, point, glyph_offset=offset, opacity=text.opacity))
            boxes.append(np.r_[point, point + size])
            cursor += size[1] + self.gap * scale
            offset += text.glyph_count
        # The group's bounds enclose every child's box.
        boxes = np.asarray(boxes)
        return AnnotationLayout(placements, [], np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)])
