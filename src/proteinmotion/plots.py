"""Native vector plots and residue tracks synchronized with seekable scenes."""

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
    raw = (hi - lo) / max(1, count - 1)
    power = 10.0 ** np.floor(np.log10(raw))
    steps = np.array([1, 2, 2.5, 5, 10]) * power
    step = steps[np.argmin(abs(np.log(steps / raw)))]
    first, last = np.ceil(lo / step - 1e-10), np.floor(hi / step + 1e-10)
    values = np.arange(first, last + 1) * step
    values[abs(values) < step * 1e-10] = 0
    return values


class _Plot(Annotation):
    """Vector drawing helpers. The scene itself supplies the background."""

    def __init__(self, position, size, title):
        super().__init__()
        self.position, self.size = _point2(position, "position"), _point2(size, "size")
        if np.any(self.size <= 0):
            raise ValueError("Plot size must be positive")
        self.title = str(title)
        self._texts = {}

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
        )

    @property
    def glyph_count(self):
        return sum(t.glyph_count for t in self._texts.values()) or 1

    def _begin(self, width, height):
        self._placements, self._leaders, self._fills = [], [], []
        self._offset = 0
        self._pixel_scale = height / 1080
        self._origin = self.position * [width, height]
        self._extent = self.size * [width, height]

    def _text(self, key, value, xy, *, size=24, tint="#cdd6e3", align="left"):
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
        self._leaders.append(
            Leader(np.asarray(points, dtype=float), parse_color(tint), width * self._pixel_scale, opacity)
        )

    def _rects(self, boxes, colors, alpha=1):
        boxes = np.asarray(boxes).reshape(-1, 4)
        corners = np.array([[0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 1]])
        vertices = boxes[:, None, :2] + boxes[:, None, 2:] * corners
        rgba = np.column_stack((np.broadcast_to(colors, (len(boxes), 3)), np.full(len(boxes), alpha)))
        self._fills.append(np.concatenate((vertices, np.repeat(rgba[:, None], 6, 1)), -1).reshape(-1, 6))

    def _polygon(self, points, tint):
        points = np.asarray(points, dtype=float)
        triangles = np.stack(
            (np.broadcast_to(points[0], (len(points) - 2, 2)), points[1:-1], points[2:]), axis=1
        ).reshape(-1, 2)
        rgba = np.broadcast_to(np.r_[parse_color(tint), 1], (len(triangles), 4))
        self._fills.append(np.column_stack((triangles, rgba)))

    def _dot(self, point, radius, tint):
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        self._polygon(point + radius * np.column_stack((np.cos(angles), np.sin(angles))), tint)

    def _finish(self):
        return AnnotationLayout(
            self._placements,
            self._leaders,
            triangles=(np.concatenate(self._fills).astype(np.float32) if self._fills else np.empty((0, 6))),
        )


class ColorLegend(_Plot):
    """A horizontal color scale with numeric limits and an optional measurement unit."""

    def __init__(self, scale, *, title="", unit="", position=(0.06, 0.82), size=(0.3, 0.12)):
        if not isinstance(scale, ColorScale):
            raise TypeError("scale must be a ColorScale")
        super().__init__(position, size, title)
        self.color_scale, self.unit = scale, str(unit)

    def layout(self, camera, width, height):
        self._begin(width, height)
        x, y = self._origin
        w, h = self._extent
        s = self._pixel_scale
        self._text("title", self.title + (f" ({self.unit})" if self.unit else ""), [x, y], size=26)
        xx, yy, ww = x, y + h * 0.42, w
        values = np.linspace(self.color_scale.vmin, self.color_scale.vmax, 128)
        boxes = np.column_stack(
            (
                xx + np.arange(128) * ww / 128,
                np.full(128, yy),
                np.full(128, ww / 128 + 0.1),
                np.full(128, 8 * s),
            )
        )
        self._rects(boxes, self.color_scale.map(values))
        for i, a in enumerate((0, 0.5, 1)):
            value = self.color_scale.vmin + a * (self.color_scale.vmax - self.color_scale.vmin)
            self._line([[xx + a * ww, yy + 12 * s], [xx + a * ww, yy + 17 * s]], width=1)
            self._text(
                f"tick{i}",
                f"{value:.3g}",
                [xx + a * ww, yy + 23 * s],
                size=22,
                align=("left", "center", "right")[i],
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
        if protein is not None and len(self.times) != len(protein.trajectory):
            raise ValueError("A trajectory plot needs one sample for each state in protein.trajectory")
        if live_value is not None and not callable(live_value):
            raise TypeError("live_value must be a callable")
        self.times.flags.writeable = self.values.flags.writeable = False
        self.protein, self.live_value, self.reveal = protein, live_value, bool(reveal)
        self.grid, self.tips = bool(grid), bool(tips)
        self._trajectory = None if protein is None else protein.trajectory
        self.xlabel, self.ylabel, self.color = str(xlabel), str(ylabel), parse_color(color)
        if ylim is None:
            lo, hi = np.nanmin(self.values), np.nanmax(self.values)
            margin = max(float(hi - lo) * 0.12, 0.1)
            ylim = (lo - margin, hi + margin)
        self.ylim = np.asarray(ylim, dtype=float)
        if self.ylim.shape != (2,) or not np.isfinite(self.ylim).all() or self.ylim[1] <= self.ylim[0]:
            raise ValueError("ylim must be finite and increasing")
        self.cursor = float(self.times[0])

    @classmethod
    def distance(cls, first, second, *, times=None, **kwargs):
        """Trace centroid distance between two Regions of the same protein, in Å.

        Coordinates are measured before scene transforms. When times is omitted,
        the x axis contains state indices. Supply physical times for an MD trace.
        """
        if (
            not isinstance(first, Region)
            or not isinstance(second, Region)
            or first.protein is not second.protein
        ):
            raise ValueError("Distance traces need two regions of the same protein")
        p = first.protein
        values = []
        for i in range(len(p.trajectory)):
            xyz = p.trajectory.frame(i)
            values.append(np.linalg.norm(xyz[first.atom_indices].mean(0) - xyz[second.atom_indices].mean(0)))
        if times is None:
            times = np.arange(len(values))
            kwargs.setdefault("xlabel", "State index")
        kwargs.setdefault("ylabel", "Distance (Å)")

        def current():
            return float(np.linalg.norm(first.positions.mean(0) - second.positions.mean(0)))

        return cls(times, values, protein=p, live_value=current, **kwargs)

    def _set_time(self, time):
        if (
            self.protein is not None
            and self.protein._trajectory_source is not None
            and self.protein._trajectory_source is not self._trajectory
        ):
            raise ValueError(
                "Plot samples belong to a different trajectory; build the plot for the trajectory being played"
            )
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

    def layout(self, camera, width, height):
        self._begin(width, height)
        x, y = self._origin
        w, h = self._extent
        s = self._pixel_scale
        self._text("title", self.title, [x, y], size=28, tint="#edf3fc")
        self._text("ylabel", self.ylabel, [x, y + 37 * s], size=23)
        lo, hi = self.ylim
        yticks = _ticks(lo, hi, max(2, min(5, int(h / s / 90))))
        ylabels = [f"{v:g}" for v in yticks]
        gutter = max(50, max(map(len, ylabels), default=0) * 13 + 16) * s
        left, right, top, bottom = x + gutter, x + w - 16 * s, y + 79 * s, y + h - 62 * s

        def project(t, v):
            return np.column_stack(
                (
                    left + (np.asarray(t) - self.times[0]) / np.ptp(self.times) * (right - left),
                    bottom - np.clip((np.asarray(v) - lo) / (hi - lo), 0, 1) * (bottom - top),
                )
            )

        for i, (v, label) in enumerate(zip(yticks, ylabels)):
            yy = project([self.times[0]], [v])[0, 1]
            if self.grid:
                self._line([[left, yy], [right, yy]], width=1, opacity=0.16)
            self._line([[left - 4 * s, yy], [left + 4 * s, yy]])
            self._text(f"y{i}", label, [left - 13 * s, yy - 13 * s], size=22, align="right")
        self._line([[left, top], [left, bottom], [right, bottom]])
        if self.tips:
            self._polygon(
                [[left, top - 7 * s], [left - 4 * s, top + 3 * s], [left + 4 * s, top + 3 * s]], "#a9b6c9"
            )
            self._polygon(
                [[right + 7 * s, bottom], [right - 3 * s, bottom - 4 * s], [right - 3 * s, bottom + 4 * s]],
                "#a9b6c9",
            )
        for i, t in enumerate(_ticks(self.times[0], self.times[-1], max(2, min(6, int(w / s / 120))))):
            xx = project([t], [lo])[0, 0]
            self._line([[xx, bottom - 4 * s], [xx, bottom + 4 * s]])
            self._text(f"x{i}", f"{t:g}", [xx, bottom + 12 * s], size=22, align="center")
        self._text("xlabel", self.xlabel, [(left + right) / 2, bottom + 42 * s], size=23, align="center")
        # Keep NaN gaps. Bucket min/max reduction retains narrow peaks in long traces.
        indices = np.arange(len(self.times))
        if len(indices) > 4000:
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
        points = project(self.times[indices], self.values[indices])
        if len(points) >= 2:
            pairs = np.stack((points[:-1], points[1:]), 1)
            missing = np.cumsum(np.isnan(self.values))
            crossed_gap = missing[indices[1:]] != missing[indices[:-1]]
            pairs = pairs[np.isfinite(pairs).all((1, 2)) & ~crossed_gap]
            if len(pairs):
                self._line(pairs, self.color, width=2.8)
        cursor = np.clip(self.cursor, self.times[0], self.times[-1])
        current = self.current_value
        if np.isfinite(current):
            px, py = project([cursor], [current])[0]
            # A short dashed projection keeps the moving point easy to read.
            for yy in np.arange(py + 10 * s, bottom, 11 * s):
                self._line([[px, yy], [px, min(yy + 5 * s, bottom)]], self.color, width=1.3, opacity=0.5)
            self._dot([px, py], 5 * s, self.color)
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
    if region is None:
        return set()
    if not isinstance(region, Region) or region.protein is not protein:
        raise ValueError("selection must be a Region of this protein")
    region._validate()
    return set(region.residue_indices)


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
        values=None,
        scale=None,
        title="Sequence",
        position=(0.06, 0.83),
        size=(0.88, 0.13),
        highlight_color="#f5d477",
    ):
        super().__init__(position, size, title)
        self.protein, self.selection = protein, selection
        _selection(protein, selection)
        ids = _selection(protein, region) if region is not None else range(len(protein.topology.residues))
        self.ids = np.array([i for i in sorted(ids) if protein.topology.residues[i].ca >= 0], dtype=int)
        if not len(self.ids):
            raise ValueError("Sequence track needs Cα residues")
        self.highlight_color = parse_color(highlight_color)
        if values is not None:
            values = values if isinstance(values, ResidueValues) else ResidueValues(protein, values)
            values._validate(protein)
        self.values = values
        self.color_scale = (
            (ColorScale.from_values(values) if scale is None else scale) if values is not None else None
        )

    def layout(self, camera, width, height):
        self._begin(width, height)
        x, y = self._origin
        w, h = self._extent
        s = self._pixel_scale
        self._text("title", self.title, [x, y], size=26)
        left, top, cell = x, y + h * 0.34, w / len(self.ids)
        letters = cell > 17 * s
        strip_y = top + (29 * s if letters else 8 * s)
        if self.values is not None:
            colors = self.color_scale.map(self.values.values)[self.ids]
        else:
            ca = [self.protein.topology.residues[i].ca for i in self.ids]
            tint = current_tints(self.protein)[ca]
            base = (
                atom_metadata(self.protein)[ca, :3]
                if self.protein.color_scheme == "element"
                else residue_colors(self.protein)[self.ids]
            )
            colors = base * (1 - tint[:, 3:]) + tint[:, :3]
        boxes = np.array(
            [[left + i * cell, strip_y, max(cell - 1 * s, cell * 0.8), 5 * s] for i in range(len(self.ids))]
        )
        self._rects(boxes, colors)
        selected = _selection(self.protein, self.selection)
        for j, ri in enumerate(self.ids):
            r = self.protein.topology.residues[ri]
            if ri in selected:
                bx = left + j * cell
                self._line(
                    [[bx, strip_y + 13 * s], [bx + cell, strip_y + 13 * s]], self.highlight_color, width=2
                )
            if letters:
                letter = gemmi.find_tabulated_residue(r.name).one_letter_code or "X"
                self._text(
                    f"letter{j}",
                    letter.upper(),
                    [left + (j + 0.5) * cell, top + 2 * s],
                    size=22,
                    tint=self.highlight_color if ri in selected else "#cdd6e3",
                    align="center",
                )
        for j in sorted({0, len(self.ids) // 2, len(self.ids) - 1}):
            r = self.protein.topology.residues[self.ids[j]]
            self._text(
                f"number{j}",
                f"{r.chain}:{r.resid}{r.icode}",
                [left + (j + 0.5) * cell, strip_y + 24 * s],
                size=21,
                align="left" if j == 0 else "right" if j == len(self.ids) - 1 else "center",
            )
        return self._finish()


class ContactMap(_Plot):
    """Live binary Cα contact map. Residue rows use topology order and author labels.

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
        cutoff=8.0,
        min_separation=3,
        position=(0.69, 0.12),
        size=(0.27, 0.40),
        title="Cα contacts",
        max_residues=512,
        contact_color="#65c8bd",
        highlight_color="#f5d477",
    ):
        super().__init__(position, size, title)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        if not isinstance(min_separation, int) or min_separation < 0:
            raise ValueError("min_separation must be a nonnegative integer")
        self.protein, self.selection = protein, selection
        _selection(protein, selection)
        ids = _selection(protein, region) if region is not None else range(len(protein.topology.residues))
        self.ids = np.array([i for i in sorted(ids) if protein.topology.residues[i].ca >= 0], dtype=int)
        if not 1 <= len(self.ids) <= max_residues:
            raise ValueError("Select between 1 and max_residues Cα residues")
        self.cas = np.array([protein.topology.residues[i].ca for i in self.ids])
        self.cutoff, self.min_separation = float(cutoff), min_separation
        self.contact_color, self.highlight_color = parse_color(contact_color), parse_color(highlight_color)

    @property
    def matrix(self):
        """Boolean contact matrix for the protein's current coordinates."""
        xyz = self.protein.positions[self.cas]
        contacts = cdist(xyz, xyz) <= self.cutoff
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
        extent = max(1, min(w - 96 * s, h - 98 * s))
        left, top = x + (w - extent + 64 * s) / 2, y + 49 * s
        n = len(self.ids)
        cell = extent / n
        axis_x, axis_y = left - 11 * s, top + extent + 11 * s
        self._line([[axis_x, top], [axis_x, axis_y], [left + extent, axis_y]], width=1.2)
        self._line([[left, top], [left + extent, top + extent]], opacity=0.15, width=1)
        rows, cols = np.nonzero(self.matrix)
        if len(rows):
            boxes = np.column_stack(
                (left + cols * cell, top + rows * cell, np.full(len(rows), cell), np.full(len(rows), cell))
            )
            self._rects(boxes, self.contact_color)
        selected = _selection(self.protein, self.selection)
        # Mark selected rows and columns at the edges, preserving contact colors.
        for j, ri in enumerate(self.ids):
            if ri in selected:
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
        for j in sorted({0, n // 2, n - 1}):
            r = self.protein.topology.residues[self.ids[j]]
            label = f"{r.chain}:{r.resid}{r.icode}"
            px, py = left + (j + 0.5) * cell, top + (j + 0.5) * cell
            self._line([[px, axis_y], [px, axis_y + 4 * s]], width=1)
            self._line([[axis_x - 4 * s, py], [axis_x, py]], width=1)
            self._text(f"x{j}", label, [px, axis_y + 11 * s], size=21, align="center")
            self._text(f"y{j}", label, [axis_x - 13 * s, py - 12 * s], size=21, align="right")
        return self._finish()
