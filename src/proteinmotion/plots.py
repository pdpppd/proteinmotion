"""Native vector plots and residue tracks synchronized with seekable scenes.

Every plot here is an :class:`~proteinmotion.annotations.Annotation`: ColorLegend,
TimeSeriesPlot, SequenceTrack, ContactMap and Heatmap (RamachandranPlot in
``ramachandran.py`` builds on the same ``_Plot`` base). Scene.seek() restores each plot's
authoring state and calls ``_set_time`` where a plot follows the clock. The renderer's
OverlayRenderer (``overlay.py``) then calls ``layout(camera, width, height)`` once per
frame and draws the returned AnnotationLayout over the finished 3D image.

Coordinates:

- ``position`` and ``size`` are fractions of the viewport, with (0, 0) at the top-left
  and (1, 1) at the bottom-right. ``_begin`` turns them into ``_origin`` and ``_extent``
  in pixels of the render target. ``width`` and ``height`` include supersampling.
- Every other length (margins, tick marks, line widths, font sizes) is written in
  1080p design pixels and multiplied by ``s = height / 1080``, so a plot keeps the same
  proportions at 720p, 1080p, 4K or any supersampling factor. Pixel y grows downward.

layout() returns text placements, leader lines (``Leader``: a polyline of shape (N, 2)
or separate segments of shape (N, 2, 2), in pixels) and ``triangles``: an (M, 6) float32
array of filled triangle vertices with columns x, y, r, g, b, a. The overlay draws all
triangles first, then all lines, then all text. Within each kind, later calls paint over
earlier ones, so the order of drawing calls inside a layout method decides what is on top.
"""

import gemmi
import numpy as np
from scipy.spatial.distance import cdist

from .annotations import Annotation, AnnotationLayout, Leader, Placement, Text, _point2
from .geometry import atom_metadata, residue_colors
from .math3d import color as parse_color
from .properties import ColorScale, ResidueValues
from .regions import Region
from .styling import current_tints


def _ticks(lo, hi, count=4):
    """Place ticks at readable 1, 2, 2.5, 5 or 10 multiples within the data range."""
    # The step that would give exactly ``count`` ticks, then the nearest of 1, 2, 2.5, 5
    # and 10 times its power of ten, compared on a log scale.
    raw = (hi - lo) / max(1, count - 1)
    power = 10.0 ** np.floor(np.log10(raw))
    steps = np.array([1, 2, 2.5, 5, 10]) * power
    step = steps[np.argmin(abs(np.log(steps / raw)))]
    # The small tolerance keeps a tick that falls on a limit despite rounding error.
    first, last = np.ceil(lo / step - 1e-10), np.floor(hi / step + 1e-10)
    values = np.arange(first, last + 1) * step
    # Snap near-zero ticks to exactly 0 so the label reads "0", not "-1e-17".
    values[abs(values) < step * 1e-10] = 0
    return values


class _Plot(Annotation):
    """Vector drawing helpers. The scene supplies the background unless a panel is set."""

    def __init__(self, position, size, title):
        super().__init__()
        self.position, self.size = _point2(position, "position"), _point2(size, "size")
        if np.any(self.size <= 0):
            raise ValueError("Plot size must be positive")
        self.title = str(title)
        # Text objects by label key ("title", "x0", ...), created as layout() first needs them.
        self._texts = {}
        self.panel = None

    def with_panel(self, color="#0b1220", opacity=1.0, padding=18):
        """Draw a rounded backing panel so molecules passing behind the plot stay out of view.

        ``padding`` is in 1080p design pixels around the plot's position and size.
        """
        if not np.isfinite(opacity) or not 0 <= opacity <= 1 or not np.isfinite(padding) or padding < 0:
            raise ValueError("Panel opacity must be in [0, 1] and padding nonnegative")
        self.panel = (parse_color(color), float(opacity), float(padding))
        return self

    def _backing(self):
        """Fill a rounded rectangle (14 design-pixel corners) behind the plot's padded box."""
        rgb, opacity, padding = self.panel
        pad, radius = padding * self._pixel_scale, 14 * self._pixel_scale
        lo, hi = self._origin - pad, self._origin + self._extent + pad
        # Corner arc centers with each arc's start angle in units of π. With y pointing down,
        # the arcs run bottom-right, bottom-left, top-left, top-right: one loop around the box.
        corners = ((hi[0] - radius, hi[1] - radius, 0), (lo[0] + radius, hi[1] - radius, 0.5))
        corners += ((lo[0] + radius, lo[1] + radius, 1.0), (hi[0] - radius, lo[1] + radius, 1.5))
        # Seven points per quarter circle.
        outline = [
            (cx + radius * np.cos(np.pi * (start + t / 2)), cy + radius * np.sin(np.pi * (start + t / 2)))
            for cx, cy, start in corners
            for t in np.linspace(0, 1, 7)
        ]
        points = np.array(outline)
        # The outline is convex, so a fan of triangles from its centroid fills it.
        center = points.mean(0)
        triangles = np.stack((np.broadcast_to(center, points.shape), points, np.roll(points, -1, 0)), 1)
        rgba = np.broadcast_to(np.r_[rgb, opacity], (triangles.size // 2, 4))
        self._fills.append(np.column_stack((triangles.reshape(-1, 2), rgba)))

    def snapshot(self):
        # Plot data, cached text and protein references are shared; only authoring
        # state is restored. Large arrays are never copied once per video frame.
        return dict(
            position=self.position.copy(),
            opacity=self.opacity,
            _write=self._write,
            _lag_ratio=self._lag_ratio,
            _stroke_width=self._stroke_width,
            _reverse=self._reverse,
            panel=getattr(self, "panel", None),
        )

    @property
    def glyph_count(self):
        return sum(t.glyph_count for t in self._texts.values()) or 1

    def _begin(self, width, height):
        """Start a frame's layout: clear the drawing lists and convert position and size to pixels."""
        self._placements, self._leaders, self._fills = [], [], []
        # Running glyph count, so Write reveals the plot's labels in the order they are placed.
        self._offset = 0
        # Converts 1080p design pixels to render pixels.
        self._pixel_scale = height / 1080
        self._origin = self.position * [width, height]
        self._extent = self.size * [width, height]
        # The backing panel is the first fill, so everything else paints over it.
        if getattr(self, "panel", None) is not None:
            self._backing()

    def _text(self, key, value, xy, *, size=24, tint="#cdd6e3", align="left"):
        """Place a label whose top edge is at pixel ``xy``; ``size`` is in design pixels.

        ``align`` says whether ``xy[0]`` is the label's left edge, center or right edge.
        """
        # One cached Text per key: HarfBuzz shaping and tessellation rerun only when the
        # string changes, not on every frame.
        if key not in self._texts or self._texts[key].font_size != size:
            self._texts[key] = Text(str(value), font_size=size, color=tint)
        text = self._texts[key]
        text.color = parse_color(tint)
        text.set_text(str(value))
        xy = np.asarray(xy, dtype=float).copy()
        xy[0] -= text.geometry.width * self._pixel_scale * {"left": 0, "center": 0.5, "right": 1}[align]
        self._placements.append(Placement(text, xy, glyph_offset=self._offset))
        self._offset += text.glyph_count

    def _line(self, points, tint="#a9b6c9", width=1.5, opacity=1):
        """Add a polyline (N, 2) or a set of separate segments (N, 2, 2) in pixels; width in design pixels."""
        self._leaders.append(
            Leader(np.asarray(points, dtype=float), parse_color(tint), width * self._pixel_scale, opacity)
        )

    def _rects(self, boxes, colors, alpha=1):
        """Fill pixel boxes given as rows (x, y, width, height); ``colors`` is one RGB or one per box."""
        boxes = np.asarray(boxes).reshape(-1, 4)
        # Two triangles per box, as unit-square corners scaled by the box size.
        corners = np.array([[0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 1]])
        vertices = boxes[:, None, :2] + boxes[:, None, 2:] * corners
        rgba = np.column_stack((np.broadcast_to(colors, (len(boxes), 3)), np.full(len(boxes), alpha)))
        self._fills.append(np.concatenate((vertices, np.repeat(rgba[:, None], 6, 1)), -1).reshape(-1, 6))

    def _polygon(self, points, tint):
        """Fill a convex polygon given by its pixel vertices in order."""
        points = np.asarray(points, dtype=float)
        # A triangle fan from the first vertex, which fills convex shapes correctly.
        triangles = np.stack(
            (np.broadcast_to(points[0], (len(points) - 2, 2)), points[1:-1], points[2:]), axis=1
        ).reshape(-1, 2)
        rgba = np.broadcast_to(np.r_[parse_color(tint), 1], (len(triangles), 4))
        self._fills.append(np.column_stack((triangles, rgba)))

    def _dot(self, point, radius, tint):
        """Fill a disc of ``radius`` pixels, drawn as a 32-sided polygon."""
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        self._polygon(point + radius * np.column_stack((np.cos(angles), np.sin(angles))), tint)

    def _finish(self):
        """Pack this frame's text, lines and fills into the AnnotationLayout the overlay draws."""
        return AnnotationLayout(
            self._placements,
            self._leaders,
            triangles=(np.concatenate(self._fills).astype(np.float32) if self._fills else np.empty((0, 6))),
        )


class ColorLegend(_Plot):
    """A horizontal color scale with numeric limits and an optional measurement unit.

    Pass ResidueValues instead of a ColorScale to use their preset scale, name and unit.
    """

    def __init__(self, scale, *, title=None, unit=None, position=(0.06, 0.82), size=(0.3, 0.12)):
        if isinstance(scale, ResidueValues):
            title = scale.name if title is None else title
            unit = scale.unit if unit is None else unit
            scale = scale.scale
        if not isinstance(scale, ColorScale):
            raise TypeError("scale must be a ColorScale or ResidueValues")
        super().__init__(position, size, "" if title is None else title)
        self.color_scale, self.unit = scale, "" if unit is None else str(unit)

    def layout(self, camera, width, height):
        self._begin(width, height)
        x, y = self._origin
        w, h = self._extent
        s = self._pixel_scale
        self._text("title", self.title + (f" ({self.unit})" if self.unit else ""), [x, y], size=26)
        # An 8 design-pixel bar of 128 color steps across the full width. Each step is
        # 0.1 px wider than its spacing so neighbours overlap and no seams show.
        bar_top = y + h * 0.42
        values = np.linspace(self.color_scale.vmin, self.color_scale.vmax, 128)
        boxes = np.column_stack(
            (
                x + np.arange(128) * w / 128,
                np.full(128, bar_top),
                np.full(128, w / 128 + 0.1),
                np.full(128, 8 * s),
            )
        )
        self._rects(boxes, self.color_scale.map(values))
        # Label the limits and either the band boundaries of a banded scale or the midpoint.
        lo, hi = self.color_scale.vmin, self.color_scale.vmax
        stops = [
            lo,
            *(self.color_scale.boundaries if self.color_scale.boundaries is not None else [(lo + hi) / 2]),
            hi,
        ]
        for i, value in enumerate(stops):
            fraction = (value - lo) / (hi - lo)
            self._line([[x + fraction * w, bar_top + 12 * s], [x + fraction * w, bar_top + 17 * s]], width=1)
            # Typographic minus signs. The end labels align inward so they stay over the bar.
            self._text(
                f"tick{i}",
                f"{value:.3g}".replace("-", "−"),
                [x + fraction * w, bar_top + 23 * s],
                size=22,
                align="left" if i == 0 else "right" if i == len(stops) - 1 else "center",
            )
        return self._finish()


class TimeSeriesPlot(_Plot):
    """A trace with a moving cursor. By default the x coordinate is scene seconds.

    Pass ``protein`` to map its current trajectory state to ``times`` (one sample
    per state), including reverse playback and eased interpolation. ``live_value``
    can supply a pure function of current coordinates for the cursor's y value.
    The trace retains the supplied samples; a moving distance can differ between
    samples because coordinates, rather than distances, are interpolated.
    """

    def __init__(
        self,
        times,
        values,
        *,
        protein=None,
        trajectory=None,
        title="",
        xlabel="Time (s)",
        ylabel="Value",
        position=(0.64, 0.56),
        size=(0.32, 0.34),
        color="#f5d477",
        ylim=None,
        reveal=False,
        live_value=None,
        grid=False,
        tips=True,
        show_value=True,
    ):
        super().__init__(position, size, title)
        self.times = np.array(times, dtype=float, copy=True)
        self.values = np.array(values, dtype=float, copy=True)
        if self.times.ndim != 1 or len(self.times) < 2 or self.values.shape != self.times.shape:
            raise ValueError("times and values must be matching 1D arrays with at least two samples")
        if (
            not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
            or np.isinf(self.values).any()
        ):
            raise ValueError("times must strictly increase; values must be finite or NaN")
        if not np.isfinite(self.values).any():
            raise ValueError("The plot needs at least one finite value")
        # A protein's own trajectory is the default source of the plot's states.
        trajectory = (
            trajectory if trajectory is not None else (None if protein is None else protein.trajectory)
        )
        if trajectory is not None and protein is None:
            raise ValueError("Pass protein with trajectory so the plot can follow playback")
        # Same atom count and, when the trajectory carries a topology, the same atom identities.
        if trajectory is not None and (
            trajectory.n_atoms != len(protein.topology.atoms)
            or trajectory.topology is not None
            and trajectory.topology.keys != protein.topology.keys
        ):
            raise ValueError("Plot trajectory must match the protein topology")
        if protein is not None and len(self.times) != len(trajectory):
            raise ValueError("A trajectory plot needs one sample for each state in protein.trajectory")
        if live_value is not None and not callable(live_value):
            raise TypeError("live_value must be a callable")
        # snapshot() shares these arrays instead of copying them, so they stay read-only.
        self.times.flags.writeable = self.values.flags.writeable = False
        self.protein, self.live_value, self.reveal = protein, live_value, bool(reveal)
        self.grid, self.tips, self.show_value = bool(grid), bool(tips), bool(show_value)
        self._trajectory = trajectory
        self.xlabel, self.ylabel, self.color = str(xlabel), str(ylabel), parse_color(color)
        if ylim is None:
            # 12% headroom above and below the data; at least 0.1 so a flat trace has a range.
            lo, hi = np.nanmin(self.values), np.nanmax(self.values)
            margin = max(float(hi - lo) * 0.12, 0.1)
            ylim = (lo - margin, hi + margin)
        self.ylim = np.asarray(ylim, dtype=float)
        if self.ylim.shape != (2,) or not np.isfinite(self.ylim).all() or self.ylim[1] <= self.ylim[0]:
            raise ValueError("ylim must be finite and increasing")
        self.cursor = float(self.times[0])

    @classmethod
    def distance(cls, first, second, *, times=None, trajectory=None, anchor="backbone", **kwargs):
        """Trace distance using the same anchors as Distance between two Regions of the same protein, in Å.

        Coordinates are measured before scene transforms. When times is omitted,
        the x axis contains state indices. Supply physical times for an MD trace.
        """
        if (
            not isinstance(first, Region)
            or not isinstance(second, Region)
            or first.protein is not second.protein
        ):
            raise ValueError("Distance traces need two regions of the same protein")
        from .distances import endpoint

        first, second = endpoint(first, anchor), endpoint(second, anchor)
        protein = first.protein
        trajectory = trajectory if trajectory is not None else protein.trajectory
        # One sample per stored state: the distance between the two anchor centroids.
        values = []
        for i in range(len(trajectory)):
            xyz = trajectory.frame(i)
            values.append(np.linalg.norm(xyz[first.atom_indices].mean(0) - xyz[second.atom_indices].mean(0)))
        if times is None:
            physical = trajectory.times
            times = np.arange(len(values)) if physical is None else physical
            kwargs.setdefault(
                "xlabel", "State index" if physical is None else f"Time ({trajectory.time_unit})"
            )
        kwargs.setdefault("ylabel", "Distance (Å)")

        # The cursor measures the current coordinates, which are interpolated between states.
        def current():
            return float(np.linalg.norm(first.positions.mean(0) - second.positions.mean(0)))

        return cls(times, values, protein=protein, trajectory=trajectory, live_value=current, **kwargs)

    def _set_time(self, time):
        """Move the cursor for scene time ``time`` (seconds); Scene.seek() calls this on every seek.

        A plot that follows a protein ignores the clock and reads the protein's fractional
        trajectory state, which the replayed animations have already set for this time.
        """
        # PlayTrajectory records the trajectory it is playing. Refuse one these samples do not
        # belong to.
        if (
            self.protein is not None
            and self.protein._trajectory_source is not None
            and self.protein._trajectory_source is not self._trajectory
        ):
            raise ValueError(
                "Plot samples belong to a different trajectory; build the plot for the trajectory being played"
            )
        # State k sits at times[k]; a fractional state falls linearly between two samples.
        self.cursor = (
            float(time)
            if self.protein is None
            else float(np.interp(self.protein.trajectory_frame, np.arange(len(self.times)), self.times))
        )

    @property
    def current_value(self):
        """Current measurement, or linear interpolation of the supplied samples."""
        value = (
            self.live_value()
            if self.live_value is not None
            else np.interp(self.cursor, self.times, self.values)
        )
        return float(value)

    def _to_pixels(self, area, t, v):
        """Pixel points (N, 2) for sample times ``t`` and values ``v`` inside the axes box.

        ``area`` is (left, right, top, bottom) in pixels. The x axis spans the sample
        times; values outside ylim are clamped to the top or bottom edge.
        """
        left, right, top, bottom = area
        lo, hi = self.ylim
        return np.column_stack(
            (
                left + (np.asarray(t) - self.times[0]) / np.ptp(self.times) * (right - left),
                bottom - np.clip((np.asarray(v) - lo) / (hi - lo), 0, 1) * (bottom - top),
            )
        )

    def _draw_y_ticks(self, area, yticks, ylabels):
        """Tick marks and labels on the y axis, with optional horizontal grid lines."""
        left, right = area[:2]
        s = self._pixel_scale
        for i, (v, label) in enumerate(zip(yticks, ylabels)):
            yy = self._to_pixels(area, [self.times[0]], [v])[0, 1]
            if self.grid:
                self._line([[left, yy], [right, yy]], width=1, opacity=0.16)
            self._line([[left - 4 * s, yy], [left + 4 * s, yy]])
            self._text(f"y{i}", label, [left - 13 * s, yy - 13 * s], size=22, align="right")

    def _draw_axes(self, area):
        """The L-shaped axis lines, with arrowheads at their open ends when ``tips`` is set."""
        left, right, top, bottom = area
        s = self._pixel_scale
        self._line([[left, top], [left, bottom], [right, bottom]])
        if self.tips:
            self._polygon(
                [[left, top - 7 * s], [left - 4 * s, top + 3 * s], [left + 4 * s, top + 3 * s]], "#a9b6c9"
            )
            self._polygon(
                [[right + 7 * s, bottom], [right - 3 * s, bottom - 4 * s], [right - 3 * s, bottom + 4 * s]],
                "#a9b6c9",
            )

    def _draw_x_ticks(self, area, w):
        """Tick marks and labels on the x axis, then the axis label. ``w`` is the plot width in pixels."""
        left, right, top, bottom = area
        s = self._pixel_scale
        lo = self.ylim[0]
        # Ask _ticks for two to six x ticks, about one per 120 design pixels of width.
        for i, t in enumerate(_ticks(self.times[0], self.times[-1], max(2, min(6, int(w / s / 120))))):
            xx = self._to_pixels(area, [t], [lo])[0, 0]
            self._line([[xx, bottom - 4 * s], [xx, bottom + 4 * s]])
            self._text(f"x{i}", f"{t:g}", [xx, bottom + 12 * s], size=22, align="center")
        self._text("xlabel", self.xlabel, [(left + right) / 2, bottom + 42 * s], size=23, align="center")

    def _trace_indices(self):
        """Indices of the samples to draw, in increasing order.

        Long traces are reduced, and with ``reveal`` the trace stops at the cursor.
        """
        # Keep NaN gaps. Bucket min/max reduction retains narrow peaks in long traces.
        indices = np.arange(len(self.times))
        if len(indices) > 4000:
            # Per bucket keep both ends, the lowest and highest finite samples, and the
            # first NaN, so that peaks and gaps both survive the reduction.
            reduced = []
            for bucket in np.array_split(indices, 1000):
                finite = bucket[np.isfinite(self.values[bucket])]
                reduced.extend([bucket[0], bucket[-1]])
                if len(finite):
                    reduced.extend(
                        [finite[np.argmin(self.values[finite])], finite[np.argmax(self.values[finite])]]
                    )
                if np.isnan(self.values[bucket]).any():
                    reduced.append(bucket[np.flatnonzero(np.isnan(self.values[bucket]))[0]])
            indices = np.unique(reduced)
        if self.reveal:
            indices = indices[self.times[indices] <= self.cursor]
        return indices

    def _draw_trace(self, area):
        """The data line, drawn as separate segments so missing values leave gaps."""
        indices = self._trace_indices()
        points = self._to_pixels(area, self.times[indices], self.values[indices])
        if len(points) >= 2:
            pairs = np.stack((points[:-1], points[1:]), 1)
            # missing[k] counts the NaN samples up to k. Two drawn samples with different
            # counts have a gap between them, even if reduction dropped the NaN itself.
            missing = np.cumsum(np.isnan(self.values))
            crossed_gap = missing[indices[1:]] != missing[indices[:-1]]
            # A NaN value projects to a NaN pixel, so isfinite also drops its segments.
            pairs = pairs[np.isfinite(pairs).all((1, 2)) & ~crossed_gap]
            if len(pairs):
                self._line(pairs, self.color, width=2.8)

    def _draw_cursor(self, area):
        """Mark the current value with a dot and a dashed drop line; returns that value."""
        bottom = area[3]
        s = self._pixel_scale
        cursor = np.clip(self.cursor, self.times[0], self.times[-1])
        current = self.current_value
        if np.isfinite(current):
            px, py = self._to_pixels(area, [cursor], [current])[0]
            # A short dashed projection keeps the moving point easy to read.
            for yy in np.arange(py + 10 * s, bottom, 11 * s):
                self._line([[px, yy], [px, min(yy + 5 * s, bottom)]], self.color, width=1.3, opacity=0.5)
            self._dot([px, py], 5 * s, self.color)
        return current

    def layout(self, camera, width, height):
        self._begin(width, height)
        x, y = self._origin
        w, h = self._extent
        s = self._pixel_scale
        self._text("title", self.title, [x, y], size=28, tint="#edf3fc")
        self._text("ylabel", self.ylabel, [x, y + 37 * s], size=23)
        # Ask _ticks for two to five y ticks, about one per 90 design pixels of height.
        lo, hi = self.ylim
        yticks = _ticks(lo, hi, max(2, min(5, int(h / s / 90))))
        ylabels = [f"{v:g}" for v in yticks]
        # The left margin widens with the longest y label, about 13 design pixels per character.
        gutter = max(50, max(map(len, ylabels), default=0) * 13 + 16) * s
        # The axes box leaves room above for the title and y label, below for the x labels.
        left, right, top, bottom = x + gutter, x + w - 16 * s, y + 79 * s, y + h - 62 * s
        area = (left, right, top, bottom)
        self._draw_y_ticks(area, yticks, ylabels)
        self._draw_axes(area)
        self._draw_x_ticks(area, w)
        self._draw_trace(area)
        current = self._draw_cursor(area)
        if self.show_value:
            self._text(
                "value",
                f"{current:.2f}" if np.isfinite(current) else "missing",
                [right, y + 37 * s],
                size=24,
                tint=self.color,
                align="right",
            )
        return self._finish()


def _selection(protein, region):
    """Topology residue indices of ``region`` as a set; empty for None. Checks it belongs to protein."""
    if region is None:
        return set()
    if not isinstance(region, Region) or region.protein is not protein:
        raise ValueError("selection must be a Region of this protein")
    region._validate()
    return set(region.residue_indices)


def _region_aliases(region, selection, display_region, highlight_region):
    """Resolve the legacy names: ``region`` for display_region and ``selection`` for highlight_region."""
    if display_region is not None:
        if region is not None:
            raise ValueError("Use display_region or its legacy alias region, not both")
        region = display_region
    if highlight_region is not None:
        if selection is not None:
            raise ValueError("Use highlight_region or its legacy alias selection, not both")
        selection = highlight_region
    return region, selection


def _polymer_ids(protein, region):
    """Sorted topology indices of the residues to show (``region``, or all) that have a backbone trace atom.

    The trace atom is Cα for amino acids and C4′ (or P) for nucleotides. Residues without
    one, such as ligands, ions and waters, are left out.
    """
    ids = _selection(protein, region) if region is not None else range(len(protein.topology.residues))
    return np.array([i for i in sorted(ids) if protein.topology.residues[i].trace_atom >= 0], dtype=int)


class SequenceTrack(_Plot):
    """A sequence with a thin color strip in topology order.

    Use the same Region for ``selection`` and a 3D highlight to link the views.
    Restrict long sequences with ``region``. Letters appear when space allows.
    """

    def __init__(
        self,
        protein,
        *,
        region=None,
        selection=None,
        display_region=None,
        highlight_region=None,
        values=None,
        scale=None,
        title="Sequence",
        position=(0.06, 0.83),
        size=(0.88, 0.13),
        highlight_color="#f5d477",
    ):
        region, selection = _region_aliases(region, selection, display_region, highlight_region)
        super().__init__(position, size, title)
        self.protein, self.selection = protein, selection
        _selection(protein, selection)
        self.ids = _polymer_ids(protein, region)
        if not len(self.ids):
            raise ValueError("Sequence track needs polymer backbone anchors")
        self.highlight_color = parse_color(highlight_color)
        if values is not None:
            values = values if isinstance(values, ResidueValues) else ResidueValues(protein, values)
            values._validate(protein)
        self.values = values
        self.color_scale = (
            (ColorScale.from_values(values) if scale is None else scale) if values is not None else None
        )

    def _strip_colors(self):
        """RGB per shown residue (N, 3): mapped ``values``, or the colors the residues have in 3D."""
        if self.values is not None:
            return self.color_scale.map(self.values.values)[self.ids]
        trace_atoms = [self.protein.topology.residues[i].trace_atom for i in self.ids]
        tint = current_tints(self.protein)[trace_atoms]
        base = (
            atom_metadata(self.protein)[trace_atoms, :3]
            if self.protein.color_scheme == "element"
            else residue_colors(self.protein)[self.ids]
        )
        # Tints are (r, g, b, weight) with the color already scaled by its weight. This is the
        # blend the 3D renderers apply, so the strip matches the molecule during color animations.
        return base * (1 - tint[:, 3:]) + tint[:, :3]

    def layout(self, camera, width, height):
        self._begin(width, height)
        x, y = self._origin
        w, h = self._extent
        s = self._pixel_scale
        self._text("title", self.title, [x, y], size=26)
        # One cell per residue across the full width. Letters need more than 17 design
        # pixels per cell; without letters the strip moves up into their row.
        left, top, cell = x, y + h * 0.34, w / len(self.ids)
        letters = cell > 17 * s
        strip_y = top + (29 * s if letters else 8 * s)
        # The 5 design-pixel color strip, with a small gap between neighbouring cells.
        boxes = np.array(
            [[left + i * cell, strip_y, max(cell - 1 * s, cell * 0.8), 5 * s] for i in range(len(self.ids))]
        )
        self._rects(boxes, self._strip_colors())
        # Underline highlighted residues below the strip and tint their letters.
        selected = _selection(self.protein, self.selection)
        for j, residue_index in enumerate(self.ids):
            residue = self.protein.topology.residues[residue_index]
            if residue_index in selected:
                bx = left + j * cell
                self._line(
                    [[bx, strip_y + 13 * s], [bx + cell, strip_y + 13 * s]], self.highlight_color, width=2
                )
            if letters:
                letter = (
                    residue.base
                    if residue.is_nucleic
                    else gemmi.find_tabulated_residue(residue.name).one_letter_code or "X"
                )
                self._text(
                    f"letter{j}",
                    letter.upper(),
                    [left + (j + 0.5) * cell, top + 2 * s],
                    size=22,
                    tint=self.highlight_color if residue_index in selected else "#cdd6e3",
                    align="center",
                )
        # Author residue numbers for the first, middle and last residues.
        for j in sorted({0, len(self.ids) // 2, len(self.ids) - 1}):
            residue = self.protein.topology.residues[self.ids[j]]
            self._text(
                f"number{j}",
                f"{residue.chain}:{residue.resid}{residue.icode}",
                [left + (j + 0.5) * cell, strip_y + 24 * s],
                size=21,
                align="left" if j == 0 else "right" if j == len(self.ids) - 1 else "center",
            )
        return self._finish()


class ContactMap(_Plot):
    """Live binary backbone contact map (Cα for proteins, C4′/P for nucleotides). Residue rows use topology order and author labels.

    A contact has distance <= cutoff Å. ``min_separation`` excludes that many
    neighboring topology residues within each chain. The diagonal is excluded.
    ``max_residues`` bounds the quadratic calculation; select a region for large proteins.
    """

    def __init__(
        self,
        protein,
        *,
        region=None,
        selection=None,
        display_region=None,
        highlight_region=None,
        cutoff=8.0,
        min_separation=3,
        position=(0.69, 0.12),
        size=(0.27, 0.40),
        title="Backbone contacts",
        max_residues=512,
        contact_color="#65c8bd",
        highlight_color="#f5d477",
    ):
        region, selection = _region_aliases(region, selection, display_region, highlight_region)
        super().__init__(position, size, title)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        if not isinstance(min_separation, int) or min_separation < 0:
            raise ValueError("min_separation must be a nonnegative integer")
        self.protein, self.selection = protein, selection
        _selection(protein, selection)
        self.ids = _polymer_ids(protein, region)
        if not 1 <= len(self.ids) <= max_residues:
            raise ValueError("Select between 1 and max_residues polymer residues")
        self.cas = np.array([protein.topology.residues[i].trace_atom for i in self.ids])
        self.cutoff, self.min_separation = float(cutoff), min_separation
        self.contact_color, self.highlight_color = parse_color(contact_color), parse_color(highlight_color)

    @property
    def matrix(self):
        """Boolean contact matrix for the protein's current coordinates."""
        xyz = self.protein.positions[self.cas]
        contacts = cdist(xyz, xyz) <= self.cutoff
        # Drop pairs within min_separation topology positions on the same chain. Because
        # min_separation >= 0, this always removes the diagonal.
        chains = np.array([self.protein.topology.residues[i].chain for i in self.ids])
        exclude = (chains[:, None] == chains[None, :]) & (
            abs(self.ids[:, None] - self.ids) <= self.min_separation
        )
        contacts[exclude] = False
        return contacts

    def layout(self, camera, width, height):
        self._begin(width, height)
        x, y = self._origin
        w, h = self._extent
        s = self._pixel_scale
        self._text("title", f"{self.title} · {self.cutoff:g} Å", [x, y], size=28, tint="#edf3fc")
        # The largest square that leaves room for the title and axis labels. It is centered
        # in the plot box, then shifted right by 32 design pixels to make room for the y labels.
        extent = max(1, min(w - 96 * s, h - 98 * s))
        left, top = x + (w - extent + 64 * s) / 2, y + 49 * s
        n = len(self.ids)
        cell = extent / n
        # Axis lines sit 11 design pixels outside the map, and a faint line marks the diagonal.
        axis_x, axis_y = left - 11 * s, top + extent + 11 * s
        self._line([[axis_x, top], [axis_x, axis_y], [left + extent, axis_y]], width=1.2)
        self._line([[left, top], [left + extent, top + extent]], opacity=0.15, width=1)
        # One filled cell per contact; row j is residue ids[j] from the top.
        rows, cols = np.nonzero(self.matrix)
        if len(rows):
            boxes = np.column_stack(
                (left + cols * cell, top + rows * cell, np.full(len(rows), cell), np.full(len(rows), cell))
            )
            self._rects(boxes, self.contact_color)
        selected = _selection(self.protein, self.selection)
        # Mark selected rows and columns at the edges, preserving contact colors.
        for j, residue_index in enumerate(self.ids):
            if residue_index in selected:
                self._line(
                    [[left - 5 * s, top + j * cell], [left - 5 * s, top + (j + 1) * cell]],
                    self.highlight_color,
                    width=3,
                )
                self._line(
                    [[left + j * cell, top + extent + 5 * s], [left + (j + 1) * cell, top + extent + 5 * s]],
                    self.highlight_color,
                    width=3,
                )
        # Label the first, middle and last residues on both axes.
        for j in sorted({0, n // 2, n - 1}):
            residue = self.protein.topology.residues[self.ids[j]]
            label = f"{residue.chain}:{residue.resid}{residue.icode}"
            px, py = left + (j + 0.5) * cell, top + (j + 0.5) * cell
            self._line([[px, axis_y], [px, axis_y + 4 * s]], width=1)
            self._line([[axis_x - 4 * s, py], [axis_x, py]], width=1)
            self._text(f"x{j}", label, [px, axis_y + 11 * s], size=21, align="center")
            self._text(f"y{j}", label, [axis_x - 13 * s, py - 12 * s], size=21, align="right")
        return self._finish()


def _residue_list(protein, value):
    """Topology residue indices for one heatmap axis.

    None gives every residue with a backbone trace atom, a Region gives its residues in
    topology order, and an index sequence is checked and used as given.
    """
    if value is None:
        return np.array([i for i, r in enumerate(protein.topology.residues) if r.trace_atom >= 0], dtype=int)
    if isinstance(value, Region):
        return np.array(sorted(_selection(protein, value)), dtype=int)
    ids = np.asarray(value, dtype=int)
    if ids.ndim != 1 or (len(ids) and (ids.min() < 0 or ids.max() >= len(protein.topology.residues))):
        raise ValueError("rows and columns need topology residue indices or a Region")
    return ids


def _blocks(count, limit):
    """Index groups that average a long axis down to at most ``limit`` cells."""
    return np.array_split(np.arange(count), min(count, limit))


def _cell_center(blocks, k, cell):
    """Pixel offset from the matrix edge to the center of the drawn cell that holds index ``k``.

    ``blocks`` are the index groups from _blocks and ``cell`` is the cell size in pixels.
    """
    cell_index = next(j for j, block in enumerate(blocks) if block[0] <= k <= block[-1])
    return (cell_index + 0.5) * cell


class Heatmap(_Plot):
    """A matrix of colored cells with a color bar; rows and columns can be residues.

    ``values`` is a 2D array, or a function returning one that is called whenever
    the frame changes, so live matrices follow moving coordinates. Matrices with
    more than ``max_cells`` rows or columns are averaged into blocks. With
    ``protein``, ``rows`` and ``columns`` (topology residue indices or Regions)
    label the axes with residue numbers and mark chain boundaries, and
    ``highlight`` marks a Region's rows and columns. Use Heatmap.pae() for
    AlphaFold predicted aligned error and Heatmap.distances() for residue distances.
    """

    def __init__(
        self,
        values,
        *,
        protein=None,
        rows=None,
        columns=None,
        scale=None,
        title="",
        unit="",
        xlabel="",
        ylabel="",
        position=(0.64, 0.1),
        size=(0.32, 0.56),
        highlight=None,
        highlight_color="#f5d477",
        max_cells=160,
        colorbar=True,
    ):
        super().__init__(position, size, title)
        if not isinstance(max_cells, int) or max_cells < 2:
            raise ValueError("max_cells must be an integer of at least 2")
        # A callable is a live matrix, evaluated now to learn its shape and default scale,
        # then again by ``matrix`` when the coordinates change (every frame without a protein).
        self._function = values if callable(values) else None
        initial = np.asarray(values() if callable(values) else values, dtype=float)
        if initial.ndim != 2 or not initial.size or np.isinf(initial).any():
            raise ValueError("Heatmap values must be a nonempty 2D array of finite values or NaN")
        if not np.isfinite(initial).any():
            raise ValueError("Heatmap values need at least one finite value")
        self._static = None if callable(values) else initial
        self.shape = initial.shape
        self.protein = protein
        if protein is not None:
            self.rows = _residue_list(protein, rows)
            self.columns = self.rows if columns is None else _residue_list(protein, columns)
            if (len(self.rows), len(self.columns)) != self.shape:
                raise ValueError(
                    f"Matrix shape {self.shape} does not match {len(self.rows)}×{len(self.columns)} residues"
                )
            _selection(protein, highlight)
        elif rows is not None or columns is not None or highlight is not None:
            raise ValueError("Residue rows, columns and highlights need protein=")
        else:
            self.rows = self.columns = None
        self.highlight = highlight
        self.color_scale = ColorScale.from_values(initial) if scale is None else scale
        if not isinstance(self.color_scale, ColorScale):
            raise TypeError("scale must be a ColorScale")
        self.unit, self.xlabel, self.ylabel = str(unit), str(xlabel), str(ylabel)
        self.highlight_color = parse_color(highlight_color)
        self.max_cells, self.colorbar = max_cells, bool(colorbar)
        # Full-matrix indices averaged into each drawn row and column.
        self._row_blocks, self._col_blocks = (
            _blocks(self.shape[0], max_cells),
            _blocks(self.shape[1], max_cells),
        )
        self._cache_key, self._cache = None, None

    @classmethod
    def pae(cls, protein, source, *, title="Predicted aligned error", **kwargs):
        """AlphaFold predicted aligned error (Å) from a JSON or .npy file, or an array.

        Rows are the residue each prediction is aligned on and columns the residue
        whose position error is shown, as on AlphaFold DB pages. The matrix must
        match the protein's polymer residues; AlphaFold 3 ligand tokens are averaged.
        """
        from .confidence import pae_for

        matrix, ids, maximum = pae_for(protein, source)
        kwargs.setdefault("scale", ColorScale.pae(maximum))
        kwargs.setdefault("unit", "Å")
        kwargs.setdefault("xlabel", "Scored residue")
        kwargs.setdefault("ylabel", "Aligned residue")
        return cls(matrix, protein=protein, rows=ids, columns=ids, title=title, **kwargs)

    @classmethod
    def distances(
        cls,
        protein,
        *,
        region=None,
        anchor="backbone",
        reference=None,
        max_distance=None,
        max_residues=600,
        title=None,
        **kwargs,
    ):
        """Live residue–residue distances in Å (Cα for amino acids, C4′ for nucleotides).

        ``anchor="centroid"`` uses residue centroids. With ``reference`` (a Protein
        with the same residues, or coordinates in this protein's atom order) the
        plot shows the change from that structure: positive where residues moved apart.
        """
        if anchor not in ("backbone", "centroid"):
            raise ValueError("anchor must be backbone or centroid")
        ids = _residue_list(protein, region)
        if not 1 <= len(ids) <= max_residues:
            raise ValueError("Select between 1 and max_residues polymer residues")
        topology = protein.topology
        # The atoms averaged into each residue's point: its trace atom, or all of its atoms.
        if anchor == "backbone":
            groups = [[topology.residues[i].trace_atom] for i in ids]
        else:
            owners = np.array([a.residue_index for a in topology.atoms])
            groups = [np.flatnonzero(owners == i) for i in ids]

        def points(xyz):
            return np.array([xyz[g].mean(0) for g in groups])

        # Distances in the reference structure, subtracted from the live ones.
        base = None
        if reference is not None:
            if hasattr(reference, "topology"):
                # A reference Protein may order its residues differently, so match residues by
                # author chain, number and insertion code.
                lookup = {(r.chain, r.resid, r.icode): i for i, r in enumerate(reference.topology.residues)}
                keys = [
                    (topology.residues[i].chain, topology.residues[i].resid, topology.residues[i].icode)
                    for i in ids
                ]
                missing = [k for k in keys if k not in lookup]
                if missing:
                    raise ValueError(f"Reference lacks residues {missing[:3]}")
                other = reference.topology
                if anchor == "backbone":
                    ref_groups = [[other.residues[lookup[k]].trace_atom] for k in keys]
                else:
                    owners = np.array([a.residue_index for a in other.atoms])
                    ref_groups = [np.flatnonzero(owners == lookup[k]) for k in keys]
                ref_xyz = reference.positions
                ref_points = np.array([ref_xyz[g].mean(0) for g in ref_groups])
                base = cdist(ref_points, ref_points)
            else:
                xyz = np.asarray(reference, dtype=float)
                if xyz.shape != (len(topology.atoms), 3):
                    raise ValueError("Reference coordinates must follow this protein's atom order")
                base = cdist(points(xyz), points(xyz))

        def current():
            matrix = cdist(points(protein.positions), points(protein.positions))
            return matrix if base is None else matrix - base

        # Default color limits from the coordinates at construction: the largest distance
        # rounded up to 10 Å, or the largest change rounded up to an even number of Å (at least 2).
        initial = current()
        if base is None:
            limit = max_distance or float(np.ceil(np.nanmax(initial) / 10) * 10)
            kwargs.setdefault("scale", ColorScale.distance(limit))
            kwargs.setdefault("unit", "Å")
            title = (
                ("Cα distances" if anchor == "backbone" else "Residue distances") if title is None else title
            )
        else:
            limit = max_distance or max(2.0, float(np.ceil(np.nanmax(abs(initial)) / 2) * 2))
            kwargs.setdefault("scale", ColorScale.difference(limit))
            kwargs.setdefault("unit", "Å")
            title = "Change in distance" if title is None else title
        return cls(current, protein=protein, rows=ids, columns=ids, title=title, **kwargs)

    @property
    def matrix(self):
        """The current full-resolution matrix."""
        if self._function is None:
            return self._static
        # A protein's coordinates are a per-atom blend of two coordinate sets. Their keys, the
        # blend fraction and the per-atom timing controls identify the current coordinates,
        # so the live function reruns only when they change. Without a protein it runs on
        # every call.
        protein = self.protein
        key = (
            None if protein is None else (protein._key_a, protein._key_b, protein._mix, id(protein._controls))
        )
        if key is None or key != self._cache_key:
            self._cache = np.asarray(self._function(), dtype=float)
            if self._cache.shape != self.shape:
                raise ValueError("A live heatmap must keep the same matrix shape")
            self._cache_key = key
        return self._cache

    def _binned(self):
        """The matrix averaged over row and column blocks, ignoring NaN; NaN where a block has no data."""
        matrix = self.matrix
        # Every block is a single index, so there is nothing to average.
        if len(self._row_blocks) == self.shape[0] and len(self._col_blocks) == self.shape[1]:
            return matrix
        # Block sums of the finite values divided by block counts of finite values.
        # reduceat sums the slices that start at each block's first index.
        finite = np.isfinite(matrix)
        filled = np.where(finite, matrix, 0)
        starts_r = [b[0] for b in self._row_blocks]
        starts_c = [b[0] for b in self._col_blocks]
        sums = np.add.reduceat(np.add.reduceat(filled, starts_r, 0), starts_c, 1)
        counts = np.add.reduceat(np.add.reduceat(finite.astype(float), starts_r, 0), starts_c, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, sums / counts, np.nan)

    def _label(self, index):
        """Author label, such as ``A:42``, of the residue at topology ``index``."""
        residue = self.protein.topology.residues[index]
        return f"{residue.chain}:{residue.resid}{residue.icode}"

    def _draw_cells(self, values, left, top, cell):
        """One square per binned value, ``cell`` pixels wide, and a frame around the matrix."""
        rows, cols = values.shape
        width_px, height_px = cell * cols, cell * rows
        r_index, c_index = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        # Cells are 0.3 px larger than their spacing so neighbours overlap and no seams show.
        boxes = np.column_stack(
            (
                left + c_index.ravel() * cell,
                top + r_index.ravel() * cell,
                np.full(rows * cols, cell + 0.3),
                np.full(rows * cols, cell + 0.3),
            )
        )
        self._rects(boxes, self.color_scale.map(values.ravel()))
        self._line(
            [
                [left, top],
                [left + width_px, top],
                [left + width_px, top + height_px],
                [left, top + height_px],
                [left, top],
            ],
            "#7d8fa8",
            width=1.0,
        )

    def _draw_axis_titles(self, left, top, width_px, height_px):
        """The y title above the matrix and the x title below it, with arrows for the reading direction."""
        s = self._pixel_scale
        if self.ylabel:
            self._text("ylabel", f"{self.ylabel} ↓", [left, top - 30 * s], size=20, tint="#a9b6c9")
        if self.xlabel:
            self._text(
                "xlabel",
                f"{self.xlabel} →",
                [left + width_px / 2, top + height_px + 36 * s],
                size=20,
                align="center",
                tint="#a9b6c9",
            )

    def _draw_residue_axes(self, left, top, cell, width_px, height_px):
        """Chain boundary lines and residue labels on the bottom (columns) and left (rows) edges."""
        s = self._pixel_scale
        for axis, ids, blocks in (
            ("x", self.columns, self._col_blocks),
            ("y", self.rows, self._row_blocks),
        ):
            # A light line across the matrix wherever the chain changes between neighbours.
            chains = [self.protein.topology.residues[i].chain for i in ids]
            for k in range(1, len(ids)):
                if chains[k] != chains[k - 1]:
                    offset = _cell_center(blocks, k, cell) - cell / 2
                    if axis == "x":
                        self._line(
                            [[left + offset, top], [left + offset, top + height_px]],
                            "#edf3fc",
                            width=1.2,
                            opacity=0.6,
                        )
                    else:
                        self._line(
                            [[left, top + offset], [left + width_px, top + offset]],
                            "#edf3fc",
                            width=1.2,
                            opacity=0.6,
                        )
            # Tick and author label for the first, middle and last residues.
            for j in sorted({0, len(ids) // 2, len(ids) - 1}):
                offset = _cell_center(blocks, j, cell)
                label = self._label(ids[j])
                if axis == "x":
                    self._line(
                        [[left + offset, top + height_px], [left + offset, top + height_px + 4 * s]],
                        width=1,
                    )
                    self._text(
                        f"x{j}",
                        label,
                        [left + offset, top + height_px + 7 * s],
                        size=19,
                        align="center",
                        tint="#a9b6c9",
                    )
                else:
                    self._line([[left - 4 * s, top + offset], [left, top + offset]], width=1)
                    self._text(
                        f"y{j}",
                        label,
                        [left - 8 * s, top + offset - 11 * s],
                        size=19,
                        align="right",
                        tint="#a9b6c9",
                    )

    def _draw_highlight(self, left, top, cell, width_px):
        """Bars along the top (columns) and right (rows) edges for the ``highlight`` residues."""
        s = self._pixel_scale
        selected = _selection(self.protein, self.highlight)
        for axis, ids, blocks in (
            ("x", self.columns, self._col_blocks),
            ("y", self.rows, self._row_blocks),
        ):
            # owner[k] is the drawn cell that holds full-matrix index k.
            owner = np.concatenate([np.full(len(block), j) for j, block in enumerate(blocks)])
            cells = np.unique(owner[[k for k, i in enumerate(ids) if i in selected]]).astype(int)
            # One mark per run of consecutive highlighted cells.
            for run in np.split(cells, np.flatnonzero(np.diff(cells) > 1) + 1) if len(cells) else ():
                start, stop = run[0] * cell, (run[-1] + 1) * cell
                if axis == "x":
                    points = [[left + start, top - 5 * s], [left + stop, top - 5 * s]]
                else:
                    points = [
                        [left + width_px + 5 * s, top + start],
                        [left + width_px + 5 * s, top + stop],
                    ]
                self._line(points, self.highlight_color, width=3)

    def _draw_colorbar(self, left, top, width_px, height_px):
        """A vertical color bar right of the matrix, high values at the top, with tick labels and unit."""
        s = self._pixel_scale
        # The bar starts 22 design pixels right of the matrix and spans its middle 84%.
        bx, by, bh = left + width_px + 22 * s, top + height_px * 0.08, height_px * 0.84
        # 96 color steps, each 0.3 px taller than its spacing so no seams show.
        steps = 96
        levels = np.linspace(self.color_scale.vmax, self.color_scale.vmin, steps)
        boxes = np.column_stack(
            (
                np.full(steps, bx),
                by + np.arange(steps) * bh / steps,
                np.full(steps, 10 * s),
                np.full(steps, bh / steps + 0.3),
            )
        )
        self._rects(boxes, self.color_scale.map(levels))
        # Label the limits and either the band boundaries of a banded scale or the midpoint.
        lo, hi = self.color_scale.vmin, self.color_scale.vmax
        stops = [
            hi,
            *(
                (self.color_scale.boundaries[::-1])
                if self.color_scale.boundaries is not None
                else [(lo + hi) / 2]
            ),
            lo,
        ]
        for k, value in enumerate(stops):
            yy = by + (hi - value) / (hi - lo) * bh
            self._line([[bx + 10 * s, yy], [bx + 14 * s, yy]], width=1)
            self._text(
                f"bar{k}",
                f"{value:.3g}".replace("-", "−"),
                [bx + 17 * s, yy - 11 * s],
                size=19,
                tint="#a9b6c9",
            )
        if self.unit:
            self._text("unit", self.unit, [bx, by - 30 * s], size=20, tint="#a9b6c9")

    def layout(self, camera, width, height):
        self._begin(width, height)
        x, y = self._origin
        w, h = self._extent
        s = self._pixel_scale
        self._text("title", self.title, [x, y], size=28, tint="#edf3fc")
        # Room on the right for the color bar, when there is one.
        bar = 64 * s if self.colorbar else 0
        # Matrix origin: 60 design pixels in for the y labels, 70 down for the title.
        left, top = x + 60 * s, y + 70 * s
        values = self._binned()
        rows, cols = values.shape
        # Square cells, as large as fit beside the color bar and above the x labels;
        # never below half a pixel.
        cell = max(0.5, min((w - 66 * s - bar) / cols, (h - 136 * s) / rows))
        width_px, height_px = cell * cols, cell * rows
        self._draw_cells(values, left, top, cell)
        self._draw_axis_titles(left, top, width_px, height_px)
        if self.protein is not None:
            self._draw_residue_axes(left, top, cell, width_px, height_px)
            self._draw_highlight(left, top, cell, width_px)
        if self.colorbar:
            self._draw_colorbar(left, top, width_px, height_px)
        return self._finish()
